"""Unit tests for password hashing and JWT helpers."""

import pytest
from app.core import security
from app.core.config import settings
from jose import ExpiredSignatureError, JWTError, jwt


def test_password_hash_roundtrip():
    hashed = security.hash_password("s3cret-pw")
    assert hashed != "s3cret-pw"
    assert security.verify_password("s3cret-pw", hashed)
    assert not security.verify_password("wrong-pw", hashed)


def test_hash_is_salted_and_unique():
    # bcrypt salts each hash, so the same input yields different digests.
    assert security.hash_password("same") != security.hash_password("same")


def test_create_and_decode_token():
    token = security.create_access_token(user_id=42, role="admin")
    payload = security.decode_access_token(token)
    assert payload["sub"] == "42"
    assert payload["role"] == "admin"
    assert "exp" in payload


def test_decode_rejects_tampered_secret():
    token = security.create_access_token(1, "user")
    with pytest.raises(JWTError):
        jwt.decode(token, "the-wrong-secret", algorithms=[settings.jwt_algorithm])


def test_decode_rejects_expired_token(monkeypatch):
    monkeypatch.setattr(settings, "access_token_expire_minutes", -1)
    token = security.create_access_token(1, "user")
    with pytest.raises(ExpiredSignatureError):
        security.decode_access_token(token)
