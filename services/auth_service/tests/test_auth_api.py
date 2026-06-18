"""API tests for the auth endpoints (register / login / logout)."""


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
