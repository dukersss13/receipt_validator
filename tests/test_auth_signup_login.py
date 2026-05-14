from typing import Any

from werkzeug.security import generate_password_hash

import backend_app as webapp_module
from src.data.database import DataBase


def test_signup_then_login_roundtrip(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_signup_login_roundtrip", reset_db=True
    )
    monkeypatch.setattr(webapp_module, "database", db)

    client = webapp_module.app.test_client()

    signup = client.post(
        "/api/auth/signup",
        json={"email": "newuser@example.com", "password": "StrongPass123!"},
    )
    assert signup.status_code == 200
    signup_payload = signup.get_json()
    assert signup_payload["token"]
    assert signup_payload["user"]["email"] == "newuser@example.com"

    login = client.post(
        "/api/auth/login",
        json={"email": "newuser@example.com", "password": "StrongPass123!"},
    )
    assert login.status_code == 200
    login_payload = login.get_json()
    assert login_payload["token"]
    assert login_payload["user"]["email"] == "newuser@example.com"


def test_signup_on_google_account_sets_password_for_login(monkeypatch: Any) -> None:
    db = DataBase(engine_name="tests/data/db/test_signup_google_upgrade", reset_db=True)
    monkeypatch.setattr(webapp_module, "database", db)

    # Simulate user who first registered through Google token auth.
    db.create_or_link_google_user(
        email="googleonly@example.com",
        provider_id="google-sub-abc",
        password_hash_fallback=generate_password_hash("google-fallback"),
    )

    client = webapp_module.app.test_client()
    signup = client.post(
        "/api/auth/signup",
        json={"email": "googleonly@example.com", "password": "MyNewPass123!"},
    )
    assert signup.status_code == 200
    signup_payload = signup.get_json()
    assert signup_payload["token"]
    assert signup_payload["user"]["provider"] == "google"

    login = client.post(
        "/api/auth/login",
        json={"email": "googleonly@example.com", "password": "MyNewPass123!"},
    )
    assert login.status_code == 200
    assert login.get_json()["user"]["provider"] == "google"


def test_signup_duplicate_email_account_still_conflicts(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_signup_duplicate_conflict", reset_db=True
    )
    monkeypatch.setattr(webapp_module, "database", db)

    client = webapp_module.app.test_client()
    first = client.post(
        "/api/auth/signup",
        json={"email": "dup@example.com", "password": "StrongPass123!"},
    )
    assert first.status_code == 200

    second = client.post(
        "/api/auth/signup",
        json={"email": "dup@example.com", "password": "AnotherPass123!"},
    )
    assert second.status_code == 409
    payload = second.get_json()
    assert "already exists" in payload["error"].lower()
    assert payload["errorClass"] == "email_conflict"


def test_signup_password_requires_uppercase(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_signup_password_requires_uppercase",
        reset_db=True,
    )
    monkeypatch.setattr(webapp_module, "database", db)

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/auth/signup",
        json={"email": "loweronly@example.com", "password": "lowerpass123!"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["errorClass"] == "validation_error"
    assert "capital letter" in payload["error"]


def test_signup_password_requires_number(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_signup_password_requires_number",
        reset_db=True,
    )
    monkeypatch.setattr(webapp_module, "database", db)

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/auth/signup",
        json={"email": "nonumber@example.com", "password": "NoDigitsPass!"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["errorClass"] == "validation_error"
    assert "1 number" in payload["error"]


def test_signup_password_requires_special_character(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_signup_password_requires_special",
        reset_db=True,
    )
    monkeypatch.setattr(webapp_module, "database", db)

    client = webapp_module.app.test_client()
    response = client.post(
        "/api/auth/signup",
        json={"email": "nospecial@example.com", "password": "NoSpecial123"},
    )

    assert response.status_code == 400
    payload = response.get_json()
    assert payload["errorClass"] == "validation_error"
    assert "special character" in payload["error"]


def test_auth_login_rate_limit_returns_429(monkeypatch: Any) -> None:
    db = DataBase(
        engine_name="tests/data/db/test_auth_login_rate_limit",
        reset_db=True,
    )
    monkeypatch.setattr(webapp_module, "database", db)
    webapp_module._AUTH_RATE_LIMIT_EVENTS.clear()
    monkeypatch.setenv("ARVEE_AUTH_RATE_LIMIT_ENABLED", "true")
    monkeypatch.setenv("ARVEE_AUTH_RATE_LIMIT_MAX_REQUESTS", "2")
    monkeypatch.setenv("ARVEE_AUTH_RATE_LIMIT_WINDOW_SECONDS", "60")

    client = webapp_module.app.test_client()

    first = client.post(
        "/api/auth/login",
        json={"email": "none@example.com", "password": "bad"},
    )
    second = client.post(
        "/api/auth/login",
        json={"email": "none@example.com", "password": "bad"},
    )
    third = client.post(
        "/api/auth/login",
        json={"email": "none@example.com", "password": "bad"},
    )

    assert first.status_code in {400, 401}
    assert second.status_code in {400, 401}
    assert third.status_code == 429
    payload = third.get_json()
    assert payload["errorClass"] == "rate_limited"
    assert third.headers.get("Retry-After")
