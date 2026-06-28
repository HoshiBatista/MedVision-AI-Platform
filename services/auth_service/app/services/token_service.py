"""DB-backed refresh-token lifecycle: issue, rotate (with reuse detection), revoke.

Refresh tokens are opaque random strings; only their SHA-256 hash is persisted.
Access tokens stay short-lived and stateless — these tokens are what make a
session long-lived and revocable.
"""

from datetime import UTC, datetime, timedelta

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.security import create_refresh_token, create_reset_token, hash_token
from app.models.password_reset_token import PasswordResetToken
from app.models.refresh_token import RefreshToken


class RefreshError(Exception):
    """Raised when a refresh token is missing, expired, or reused."""

    def __init__(self, reason: str, *, result: str) -> None:
        super().__init__(reason)
        self.result = result  # metric label: invalid | reuse_detected


class ResetError(Exception):
    """Raised when a password-reset token is missing, used, or expired."""


def _expiry() -> datetime:
    return datetime.now(UTC) + timedelta(days=settings.refresh_token_expire_days)


async def issue_refresh_token(db: AsyncSession, user_id: int) -> str:
    """Create a new refresh token for a user; return the raw (uncommitted-safe) value."""
    raw = create_refresh_token()
    db.add(RefreshToken(user_id=user_id, token_hash=hash_token(raw), expires_at=_expiry()))
    await db.commit()
    return raw


async def revoke_all_for_user(db: AsyncSession, user_id: int) -> int:
    """Revoke every active refresh token for a user; return how many were revoked."""
    result = await db.execute(
        update(RefreshToken)
        .where(RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None))
        .values(revoked_at=datetime.now(UTC))
    )
    await db.commit()
    return result.rowcount or 0


async def rotate_refresh_token(db: AsyncSession, raw: str) -> tuple[int, str]:
    """Validate + rotate a refresh token.

    Returns ``(user_id, new_raw_token)``. Revokes the presented token and issues a
    replacement. Raises :class:`RefreshError` if the token is unknown/expired, or
    revokes the user's whole active set on reuse of an already-revoked token.
    """
    now = datetime.now(UTC)
    result = await db.execute(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(raw))
    )
    row = result.scalar_one_or_none()

    if row is None:
        raise RefreshError("unknown refresh token", result="invalid")

    if row.revoked_at is not None:
        # A revoked token presented again ⇒ likely theft of a rotated token.
        await revoke_all_for_user(db, row.user_id)
        raise RefreshError("refresh token reuse detected", result="reuse_detected")

    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < now:
        raise RefreshError("refresh token expired", result="invalid")

    # Rotate: revoke the presented token, mint a replacement in one transaction.
    row.revoked_at = now
    new_raw = create_refresh_token()
    db.add(RefreshToken(user_id=row.user_id, token_hash=hash_token(new_raw), expires_at=_expiry()))
    await db.commit()
    return row.user_id, new_raw


# ── Password-reset tokens ─────────────────────────────────────────────────────


def _reset_expiry() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=settings.password_reset_expire_minutes)


async def create_password_reset(db: AsyncSession, user_id: int) -> str:
    """Issue a single-use reset token, invalidating any prior unused one. Returns raw."""
    await db.execute(
        update(PasswordResetToken)
        .where(PasswordResetToken.user_id == user_id, PasswordResetToken.used_at.is_(None))
        .values(used_at=datetime.now(UTC))
    )
    raw = create_reset_token()
    db.add(
        PasswordResetToken(
            user_id=user_id, token_hash=hash_token(raw), expires_at=_reset_expiry()
        )
    )
    await db.commit()
    return raw


async def consume_password_reset(db: AsyncSession, raw: str) -> int:
    """Validate + burn a reset token. Returns the owning user_id; raises on failure."""
    now = datetime.now(UTC)
    result = await db.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == hash_token(raw))
    )
    row = result.scalar_one_or_none()

    if row is None or row.used_at is not None:
        raise ResetError("unknown or already-used reset token")

    expires_at = row.expires_at
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if expires_at < now:
        raise ResetError("reset token expired")

    row.used_at = now
    await db.commit()
    return row.user_id
