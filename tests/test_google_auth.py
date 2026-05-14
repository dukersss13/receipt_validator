from typing import Any

import backend_app as webapp_module


def test_google_config_disabled_when_client_id_missing(monkeypatch: Any) -> None:
    monkeypatch.setattr(webapp_module, "_google_oauth_client_id", lambda: "")
    monkeypatch.setattr(webapp_module, "_google_oauth_ios_client_id", lambda: "")
    monkeypatch.setattr(webapp_module, "_google_oauth_redirect_scheme", lambda: "arvee")

    client = webapp_module.app.test_client()
    response = client.get("/api/auth/google/config")

    assert response.status_code == 200
    assert response.get_json() == {
        "enabled": False,
        "clientId": "",
        "iosClientId": "",
        "redirectScheme": "arvee",
    }


def test_google_config_enabled_when_client_id_present(monkeypatch: Any) -> None:
    monkeypatch.setattr(
        webapp_module,
        "_google_oauth_client_id",
        lambda: "demo-client.apps.googleusercontent.com",
    )
    monkeypatch.setattr(webapp_module, "_google_oauth_ios_client_id", lambda: "")
    monkeypatch.setattr(webapp_module, "_google_oauth_redirect_scheme", lambda: "arvee")

    client = webapp_module.app.test_client()
    response = client.get("/api/auth/google/config")

    assert response.status_code == 200
    assert response.get_json() == {
        "enabled": True,
        "clientId": "demo-client.apps.googleusercontent.com",
        "iosClientId": "",
        "redirectScheme": "arvee",
    }


def test_google_config_enabled_with_ios_client_id(monkeypatch: Any) -> None:
    monkeypatch.setattr(webapp_module, "_google_oauth_client_id", lambda: "")
    monkeypatch.setattr(
        webapp_module,
        "_google_oauth_ios_client_id",
        lambda: "ios-client.apps.googleusercontent.com",
    )
    monkeypatch.setattr(webapp_module, "_google_oauth_redirect_scheme", lambda: "arvee")

    client = webapp_module.app.test_client()
    response = client.get("/api/auth/google/config")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["enabled"] is True
    assert payload["iosClientId"] == "ios-client.apps.googleusercontent.com"


def test_google_token_login_success(monkeypatch: Any) -> None:
    class StubUser:
        email = "person@example.com"

    class StubDB:
        def create_or_link_google_user(
            self,
            email: str,
            provider_id: str,
            password_hash_fallback: str,
        ) -> StubUser:
            assert email == "person@example.com"
            assert provider_id == "google-sub-123"
            assert password_hash_fallback
            return StubUser()

    monkeypatch.setattr(
        webapp_module,
        "_verify_google_id_token",
        lambda token: {
            "email": "person@example.com",
            "providerId": "google-sub-123",
            "name": "Person",
            "picture": "https://example.com/p.png",
        },
    )
    monkeypatch.setattr(webapp_module, "database", StubDB())

    client = webapp_module.app.test_client()
    response = client.post("/api/auth/google/token", json={"idToken": "dummy-token"})

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["token"]
    assert payload["user"] == {
        "email": "person@example.com",
        "provider": "google",
        "name": "Person",
        "picture": "https://example.com/p.png",
    }


def test_google_token_login_rejects_invalid_token(monkeypatch: Any) -> None:
    def _raise_invalid(_: str) -> dict[str, Any]:
        raise ValueError("Invalid Google identity token.")

    monkeypatch.setattr(webapp_module, "_verify_google_id_token", _raise_invalid)

    client = webapp_module.app.test_client()
    response = client.post("/api/auth/google/token", json={"idToken": "invalid"})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "Invalid Google identity token.",
        "errorClass": "oauth_verify_failure",
    }


def test_google_token_login_requires_id_token() -> None:
    client = webapp_module.app.test_client()
    response = client.post("/api/auth/google/token", json={})

    assert response.status_code == 400
    assert response.get_json() == {
        "error": "idToken is required.",
        "errorClass": "validation_error",
    }
