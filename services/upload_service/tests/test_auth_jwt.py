"""JWT Bearer auth on the upload endpoints (tokens issued by auth_service)."""

from app.core.config import settings
from jose import jwt

LIST = "/api/v1/upload"


def _token(sub: str = "1", **extra) -> str:
    payload = {"sub": sub, **extra}
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def test_valid_jwt_authorizes(anon_client):
    res = anon_client.get(LIST, headers={"Authorization": f"Bearer {_token('1')}"})
    assert res.status_code == 200, res.text
    assert res.json()["total"] == 0


def test_missing_token_rejected(anon_client):
    assert anon_client.get(LIST).status_code == 401


def test_malformed_token_rejected(anon_client):
    res = anon_client.get(LIST, headers={"Authorization": "Bearer not-a-jwt"})
    assert res.status_code == 401


def test_token_signed_with_wrong_secret_rejected(anon_client):
    bad = jwt.encode({"sub": "1"}, "wrong-secret", algorithm=settings.jwt_algorithm)
    res = anon_client.get(LIST, headers={"Authorization": f"Bearer {bad}"})
    assert res.status_code == 401
