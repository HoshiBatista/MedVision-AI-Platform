"""API tests for the admin user-management endpoints (RBAC enforced)."""


def test_admin_endpoints_require_admin_role(client, user_token, bearer):
    # a regular user must not reach admin routes
    assert client.get("/api/v1/admin/users", headers=bearer(user_token)).status_code == 403


def test_admin_endpoints_require_auth(client):
    assert client.get("/api/v1/admin/users").status_code == 401


def test_list_users(client, admin_token, user_token, bearer):
    res = client.get("/api/v1/admin/users", headers=bearer(admin_token))
    assert res.status_code == 200
    emails = {u["email"] for u in res.json()}
    assert {"admin", "user@example.com"} <= emails


def test_get_user_found_and_missing(client, admin_token, bearer):
    users = client.get("/api/v1/admin/users", headers=bearer(admin_token)).json()
    admin_id = next(u["id"] for u in users if u["email"] == "admin")

    assert client.get(f"/api/v1/admin/users/{admin_id}", headers=bearer(admin_token)).status_code == 200
    assert client.get("/api/v1/admin/users/99999", headers=bearer(admin_token)).status_code == 404


def test_update_user_role_and_validation(client, admin_token, user_token, bearer):
    users = client.get("/api/v1/admin/users", headers=bearer(admin_token)).json()
    uid = next(u["id"] for u in users if u["email"] == "user@example.com")

    ok = client.patch(
        f"/api/v1/admin/users/{uid}",
        json={"role": "radiologist"},
        headers=bearer(admin_token),
    )
    assert ok.status_code == 200
    assert ok.json()["role"] == "radiologist"

    bad = client.patch(
        f"/api/v1/admin/users/{uid}",
        json={"role": "superhero"},
        headers=bearer(admin_token),
    )
    assert bad.status_code == 422


def test_deactivate_user(client, admin_token, user_token, bearer):
    users = client.get("/api/v1/admin/users", headers=bearer(admin_token)).json()
    uid = next(u["id"] for u in users if u["email"] == "user@example.com")

    res = client.delete(f"/api/v1/admin/users/{uid}", headers=bearer(admin_token))
    assert res.status_code == 204

    fetched = client.get(f"/api/v1/admin/users/{uid}", headers=bearer(admin_token)).json()
    assert fetched["is_active"] is False


def test_admin_cannot_deactivate_self(client, admin_token, bearer):
    users = client.get("/api/v1/admin/users", headers=bearer(admin_token)).json()
    admin_id = next(u["id"] for u in users if u["email"] == "admin")

    res = client.delete(f"/api/v1/admin/users/{admin_id}", headers=bearer(admin_token))
    assert res.status_code == 400
