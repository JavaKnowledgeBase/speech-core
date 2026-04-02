from __future__ import annotations

from copy import deepcopy

import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app, validate_runtime_safety
from app.observability import anonymize_identifier


@pytest.fixture
def restore_settings():
    snapshot = deepcopy(settings.__dict__)
    try:
        yield
    finally:
        settings.__dict__.update(snapshot)


class TestProductionSafety:
    def test_health_is_available_without_auth(self, restore_settings):
        settings.app_env = "production"
        settings.service_api_key = "secret"
        settings.supabase_url = "https://example.supabase.co"
        settings.supabase_key = "supabase-key"

        with TestClient(app) as client:
            response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["auth_required"] is True

    def test_filter_requires_api_key_in_production(self, restore_settings):
        settings.app_env = "production"
        settings.service_api_key = "secret"
        settings.supabase_url = "https://example.supabase.co"
        settings.supabase_key = "supabase-key"

        with TestClient(app) as client:
            response = client.post(
                "/filter",
                json={"audience": "child", "text": "Let us try.", "context": "general"},
            )

        assert response.status_code == 401
        assert response.json()["detail"] == "Unauthorized"

    def test_filter_accepts_api_key_in_production(self, restore_settings):
        settings.app_env = "production"
        settings.service_api_key = "secret"
        settings.supabase_url = "https://example.supabase.co"
        settings.supabase_key = "supabase-key"

        with TestClient(app) as client:
            response = client.post(
                "/filter",
                headers={"x-service-api-key": "secret"},
                json={"audience": "child", "text": "Great job.", "context": "success", "owner_id": "child-1"},
            )

        assert response.status_code == 200
        assert response.json()["confidence"] == 0.92

    def test_responses_set_safer_headers(self, restore_settings):
        with TestClient(app) as client:
            response = client.get("/health")

        assert response.headers["Cache-Control"] == "no-store"
        assert response.headers["X-Content-Type-Options"] == "nosniff"
        assert response.headers["Referrer-Policy"] == "no-referrer"
        assert response.headers["X-Request-Id"]

    def test_validate_runtime_safety_rejects_unsafe_production(self, restore_settings):
        settings.app_env = "production"
        settings.service_api_key = ""
        settings.supabase_url = ""
        settings.supabase_key = ""

        with pytest.raises(RuntimeError, match="Unsafe production configuration"):
            validate_runtime_safety()


class TestObservabilityHelpers:
    def test_anonymize_identifier_is_stable_and_not_raw(self, restore_settings):
        settings.log_identifier_salt = "test-salt"

        hashed = anonymize_identifier("child-1")

        assert hashed == anonymize_identifier("child-1")
        assert hashed != "child-1"
        assert len(hashed) == 12
