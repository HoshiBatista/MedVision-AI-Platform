"""API tests for the auth endpoints (register / login / refresh / logout)."""


def _login_tokens(client, username="admin", password="admin") -> dict:
    res = client.post("/api/v1/auth/login", data={"username": username, "password": password})
    res.raise_for_status()
    return res.json()


def test_register_success(client):
    res = client.post(
        "/api/v1/auth/register",
        json={"email": "new@example.com", "password": "password123", "full_name": "New User"},
    )
    assert res.status_code == 201
    body = res.json()
    assert body["email"] == "new@example.com"
    assert body["role"] == "user"
    assert body["is_active"] is True
    assert "hashed_password" not in body  # never leak the hash


def test_register_duplicate_email_conflicts(client):
    payload = {"email": "dup@example.com", "password": "password123"}
    assert client.post("/api/v1/auth/register", json=payload).status_code == 201
    res = client.post("/api/v1/auth/register", json=payload)
    assert res.status_code == 409


def test_register_short_password_rejected(client):
    res = client.post(
        "/api/v1/auth/register",
        json={"email": "short@example.com", "password": "tiny"},
    )
    assert res.status_code == 422


def test_login_success_returns_token(client):
    res = client.post("/api/v1/auth/login", data={"username": "admin", "password": "admin"})
    assert res.status_code == 200
    body = res.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]
    assert body["expires_in"] == 30 * 60


def test_login_wrong_password_unauthorized(client):
    res = client.post("/api/v1/auth/login", data={"username": "admin", "password": "nope"})
    assert res.status_code == 401


def test_login_unknown_user_unauthorized(client):
    res = client.post("/api/v1/auth/login", data={"username": "ghost", "password": "whatever"})
    assert res.status_code == 401


def test_login_disabled_account_forbidden(client, admin_token, bearer):
    # register a user, then have the admin disable them
    client.post(
        "/api/v1/auth/register",
        json={"email": "disabled@example.com", "password": "password123"},
    )
    users = client.get("/api/v1/admin/users", headers=bearer(admin_token)).json()
    target = next(u for u in users if u["email"] == "disabled@example.com")
    client.patch(
        f"/api/v1/admin/users/{target['id']}",
        json={"is_active": False},
        headers=bearer(admin_token),
    )
    res = client.post(
        "/api/v1/auth/login",
        data={"username": "disabled@example.com", "password": "password123"},
    )
    assert res.status_code == 403


def test_logout_requires_auth(client):
    assert client.post("/api/v1/auth/logout").status_code == 401


def test_logout_with_token(client, admin_token, bearer):
    res = client.post("/api/v1/auth/logout", headers=bearer(admin_token))
    assert res.status_code == 204


# ── Refresh-token flow ────────────────────────────────────────────────────────

def test_login_returns_refresh_token(client):
    body = _login_tokens(client)
    assert body["access_token"]
    assert body["refresh_token"]
    assert body["access_token"] != body["refresh_token"]


def test_refresh_rotates_and_returns_new_tokens(client):
    first = _login_tokens(client)
    res = client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert res.status_code == 200, res.text
    rotated = res.json()
    # A fresh access token and a *new* (rotated) refresh token are returned.
    assert rotated["access_token"]
    assert rotated["refresh_token"] != first["refresh_token"]


def test_refresh_with_unknown_token_rejected(client):
    res = client.post("/api/v1/auth/refresh", json={"refresh_token": "not-a-real-token"})
    assert res.status_code == 401


def test_refresh_reuse_of_rotated_token_rejected(client):
    first = _login_tokens(client)
    # Rotate once — the original token is now revoked.
    client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    # Presenting the already-rotated token again must fail (reuse detection).
    reuse = client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    assert reuse.status_code == 401


def test_reuse_detection_revokes_whole_family(client):
    first = _login_tokens(client)
    second = client.post(
        "/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]}
    ).json()
    # Reuse the revoked original → should revoke the active (second) token too.
    client.post("/api/v1/auth/refresh", json={"refresh_token": first["refresh_token"]})
    after = client.post("/api/v1/auth/refresh", json={"refresh_token": second["refresh_token"]})
    assert after.status_code == 401


def test_logout_revokes_refresh_tokens(client, bearer):
    tokens = _login_tokens(client)
    logout = client.post("/api/v1/auth/logout", headers=bearer(tokens["access_token"]))
    assert logout.status_code == 204
    # After logout the refresh token can no longer be exchanged.
    res = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert res.status_code == 401


# ── Password reset ────────────────────────────────────────────────────────────

def _register(client, email="reset@example.com", password="password123"):
    client.post("/api/v1/auth/register", json={"email": email, "password": password})
    return email, password


def test_forgot_password_unknown_email_is_generic(client):
    res = client.post("/api/v1/auth/forgot-password", json={"email": "ghost@example.com"})
    assert res.status_code == 200
    body = res.json()
    assert body["message"]
    assert body["reset_token"] is None  # no token minted for a non-existent account


def test_forgot_password_returns_token_in_dev(client):
    email, _ = _register(client)
    res = client.post("/api/v1/auth/forgot-password", json={"email": email})
    assert res.status_code == 200
    # ENVIRONMENT=test (non-production) → token echoed for local completion.
    assert res.json()["reset_token"]


def test_reset_password_changes_password(client):
    email, old = _register(client)
    token = client.post(
        "/api/v1/auth/forgot-password", json={"email": email}
    ).json()["reset_token"]

    res = client.post(
        "/api/v1/auth/reset-password", json={"token": token, "new_password": "brand-new-pw1"}
    )
    assert res.status_code == 200

    # Old password rejected, new one works.
    assert client.post(
        "/api/v1/auth/login", data={"username": email, "password": old}
    ).status_code == 401
    assert client.post(
        "/api/v1/auth/login", data={"username": email, "password": "brand-new-pw1"}
    ).status_code == 200


def test_reset_token_is_single_use(client):
    email, _ = _register(client)
    token = client.post(
        "/api/v1/auth/forgot-password", json={"email": email}
    ).json()["reset_token"]

    first = client.post(
        "/api/v1/auth/reset-password", json={"token": token, "new_password": "first-new-pw1"}
    )
    assert first.status_code == 200
    second = client.post(
        "/api/v1/auth/reset-password", json={"token": token, "new_password": "second-new-pw1"}
    )
    assert second.status_code == 400


def test_reset_password_unknown_token_rejected(client):
    res = client.post(
        "/api/v1/auth/reset-password", json={"token": "nope", "new_password": "whatever12"}
    )
    assert res.status_code == 400


def test_reset_password_short_password_rejected(client):
    email, _ = _register(client)
    token = client.post(
        "/api/v1/auth/forgot-password", json={"email": email}
    ).json()["reset_token"]
    res = client.post(
        "/api/v1/auth/reset-password", json={"token": token, "new_password": "short"}
    )
    assert res.status_code == 422  # schema validation (min length)


def test_requesting_new_reset_invalidates_prior(client):
    email, _ = _register(client)
    first = client.post(
        "/api/v1/auth/forgot-password", json={"email": email}
    ).json()["reset_token"]
    # Requesting again invalidates the first token.
    client.post("/api/v1/auth/forgot-password", json={"email": email})
    res = client.post(
        "/api/v1/auth/reset-password", json={"token": first, "new_password": "another-pw12"}
    )
    assert res.status_code == 400


def test_reset_password_revokes_refresh_tokens(client):
    email, _ = _register(client)
    tokens = _login_tokens(client, email, "password123")
    reset_token = client.post(
        "/api/v1/auth/forgot-password", json={"email": email}
    ).json()["reset_token"]

    client.post(
        "/api/v1/auth/reset-password", json={"token": reset_token, "new_password": "post-reset-pw1"}
    )
    # The pre-reset refresh token must no longer work.
    res = client.post("/api/v1/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert res.status_code == 401
