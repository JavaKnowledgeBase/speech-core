from __future__ import annotations

from fastapi.testclient import TestClient

from app.client import FilterServiceClient
from app.main import app


client = TestClient(app)


class TestProfileResolutionEndpoints:
    def test_filter_auto_loads_child_profile_from_owner_id(self):
        response = client.post(
            "/filter",
            json={
                "audience": "child",
                "text": "Fantastic work today.",
                "context": "success",
                "owner_id": "child-1",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["confidence"] == 0.92
        assert payload["filtered_text"].lower().startswith("quiet try")

    def test_filter_preview_loads_parent_profile_from_child_id(self):
        response = client.post(
            "/filter/preview",
            json={
                "audience": "parent",
                "text": "Please help the child.",
                "context": "escalation",
                "child_id": "child-1",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["confidence"] == 0.92
        assert payload["filtered_text"].lower().startswith("one calm prompt")

    def test_filter_batch_resolves_profiles_for_each_item(self):
        response = client.post(
            "/filter/batch",
            json={
                "items": [
                    {
                        "audience": "child",
                        "text": "Nice work.",
                        "context": "success",
                        "owner_id": "child-1",
                    },
                    {
                        "audience": "parent",
                        "text": "Please support the child.",
                        "context": "guidance",
                        "owner_id": "caregiver-1",
                    },
                ]
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert [item["confidence"] for item in payload["results"]] == [0.92, 0.92]

    def test_filter_returns_404_for_unknown_explicit_profile_id(self):
        response = client.post(
            "/filter",
            json={
                "audience": "child",
                "text": "Let us try.",
                "context": "general",
                "profile_id": "missing-profile",
            },
        )

        assert response.status_code == 404
        assert response.json()["detail"] == "Profile 'missing-profile' not found."


class TestFilterServiceClientPayload:
    def test_payload_includes_owner_id(self):
        payload = FilterServiceClient._build_payload(
            audience="child",
            text="Let us try.",
            context="retry",
            owner_id="child-1",
            retries_used=1,
        )

        assert payload["owner_id"] == "child-1"
