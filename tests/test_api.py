"""Tests for the admin API endpoints."""

from fastapi.testclient import TestClient

from blogboard.api.app import app

client = TestClient(app)


def test_health_endpoint():
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "llm_configured" in body
    assert "storage_backend" in body


def test_stats_endpoint():
    res = client.get("/api/stats")
    assert res.status_code == 200
    body = res.json()
    assert isinstance(body["total_articles"], int)
    assert isinstance(body["by_domain"], dict)


def test_articles_endpoint_filters_by_domain():
    res = client.get("/api/articles", params={"domain": "ml"})
    assert res.status_code == 200
    body = res.json()
    assert all(a.get("category") == "ml" for a in body["articles"])


def test_generate_endpoint_dry_run_no_auth_needed():
    """With no ADMIN_TOKEN configured, the call must succeed."""
    res = client.post("/api/generate", params={"dry_run": True, "domain": "ml"})
    assert res.status_code == 200
    body = res.json()
    assert body["domain"] == "ml"
    assert body["md_path"]


def test_generate_endpoint_rejects_bad_token(monkeypatch):
    monkeypatch.setattr("blogboard.api.app.app_settings.admin_token", "s3cret")
    res = client.post(
        "/api/generate",
        params={"dry_run": True},
        headers={"X-Admin-Token": "wrong"},
    )
    assert res.status_code == 401
