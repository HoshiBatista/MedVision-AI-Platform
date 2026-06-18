"""API tests for the self-service /users endpoints."""


def test_me_requires_auth(client):
    assert client.get("/api/v1/users/me").status_code == 401


def test_me_rejects_invalid_token(client, bearer):
    res = client.get("/api/v1/users/me", headers=bearer("not-a-real-token"))
    assert res.status_code == 401


def test_me_returns_current_user(client, admin_token, bearer):
    res = client.get("/api/v1/users/me", headers=bearer(admin_token))
    assert res.status_code == 200
    body = res.json()
    assert body["email"] == "admin"
    assert body["role"] == "admin"


def test_update_me_changes_full_name(client, user_token, bearer):
    res = client.patch(
        "/api/v1/users/me",
        json={"full_name": "Renamed User"},
        headers=bearer(user_token),
    )
    assert res.status_code == 200
    assert res.json()["full_name"] == "Renamed User"

    # change persists
    me = client.get("/api/v1/users/me", headers=bearer(user_token)).json()
    assert me["full_name"] == "Renamed User"
