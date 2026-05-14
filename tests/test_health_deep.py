from typing import Any

import backend_app as webapp_module


class _HealthyDB:
    def get_user_auth(self, email: str) -> None:
        assert email == "healthcheck@example.com"
        return None


class _FailingDB:
    def get_user_auth(self, email: str) -> None:
        assert email == "healthcheck@example.com"
        raise RuntimeError("db unavailable")


def test_health_deep_reports_dependency_status(monkeypatch: Any) -> None:
    monkeypatch.setattr(webapp_module, "database", _HealthyDB())
    monkeypatch.setattr(
        webapp_module,
        "_google_oauth_client_id",
        lambda: "demo-client.apps.googleusercontent.com",
    )
    monkeypatch.setattr(webapp_module, "_google_oauth_redirect_scheme", lambda: "arvee")

    client = webapp_module.app.test_client()
    response = client.get("/api/health/deep")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ok"
    assert payload["checks"]["database"]["ok"] is True
    assert payload["checks"]["googleOAuth"]["enabled"] is True


def test_health_deep_returns_503_when_database_fails(monkeypatch: Any) -> None:
    monkeypatch.setattr(webapp_module, "database", _FailingDB())
    monkeypatch.setattr(webapp_module, "_google_oauth_client_id", lambda: "")
    monkeypatch.setattr(webapp_module, "_google_oauth_redirect_scheme", lambda: "arvee")

    client = webapp_module.app.test_client()
    response = client.get("/api/health/deep")

    assert response.status_code == 503
    payload = response.get_json()
    assert payload["status"] == "degraded"
    assert payload["checks"]["database"]["ok"] is False
    assert "db unavailable" in str(payload["checks"]["database"]["error"])
