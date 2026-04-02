"""
FilterServiceClient — drop-in adapter for speech-intelligence.

The speech-intelligence platform can import and use this client instead of
calling OutputFilterExpert directly. The client either calls this filter
service over HTTP (when FILTER_SERVICE_URL is set) or falls back to running
the local heuristic pipeline in-process.

Usage in speech-intelligence agentic.py:

    from filter_client import FilterServiceClient
    client = FilterServiceClient()

    # Replace direct OutputFilterExpert calls:
    filtered, trace = client.filter(audience="child", text=raw_text, owner_id=child_id)

Config:
    FILTER_SERVICE_URL   — base URL of the running filter service
                           e.g. http://localhost:8001
                           If not set, runs the pipeline in-process.
    FILTER_SERVICE_TIMEOUT — HTTP timeout in seconds (default 3)
"""
from __future__ import annotations

import os
from typing import Any


class FilterServiceClient:
    """
    Adapter that speech-intelligence uses to call the output filter layer.

    Two modes:
      - HTTP mode  (FILTER_SERVICE_URL set): calls the running FastAPI service
      - Local mode (no URL):                 imports and runs the pipeline in-process

    Both modes return the same (filtered_text, style_tags) tuple so the caller
    doesn't need to care which mode is active.
    """

    def __init__(self) -> None:
        self._base_url = os.getenv("FILTER_SERVICE_URL", "").rstrip("/")
        self._timeout = int(os.getenv("FILTER_SERVICE_TIMEOUT", "3"))
        self._local_pipeline = None  # lazily loaded

    # ── Public API ────────────────────────────────────────────────────────────

    def filter(
        self,
        audience: str,
        text: str,
        context: str = "general",
        owner_id: str | None = None,
        engagement_score: float = 0.75,
        retries_used: int = 0,
        frustration_flag: bool = False,
        last_action: str = "none",
    ) -> tuple[str, list[dict]]:
        """
        Filter a message and return (filtered_text, trace_list).

        Args:
            audience:         "child" or "parent"
            text:             Raw text to filter
            context:          One of: session_start, success, retry, escalation,
                              reengagement, guidance, general
            owner_id:         child_id or caregiver_id — used to look up stored profile
            engagement_score: Current child engagement (0-1)
            retries_used:     How many retries have occurred this turn
            frustration_flag: Whether the child appears frustrated
            last_action:      Last therapy action: advance, retry, escalate, none

        Returns:
            (filtered_text, trace) where trace is a list of dicts with
            filter_name, applied, reason fields.
        """
        if self._base_url:
            return self._call_http(
                audience=audience,
                text=text,
                context=context,
                owner_id=owner_id,
                engagement_score=engagement_score,
                retries_used=retries_used,
                frustration_flag=frustration_flag,
                last_action=last_action,
            )
        return self._call_local(
            audience=audience,
            text=text,
            context=context,
            owner_id=owner_id,
            engagement_score=engagement_score,
            retries_used=retries_used,
            frustration_flag=frustration_flag,
            last_action=last_action,
        )

    def filter_child(
        self,
        text: str,
        context: str = "general",
        owner_id: str | None = None,
        engagement_score: float = 0.75,
        retries_used: int = 0,
        frustration_flag: bool = False,
    ) -> tuple[str, list[dict]]:
        """Convenience wrapper for child-facing messages."""
        return self.filter(
            audience="child",
            text=text,
            context=context,
            owner_id=owner_id,
            engagement_score=engagement_score,
            retries_used=retries_used,
            frustration_flag=frustration_flag,
        )

    def filter_parent(
        self,
        text: str,
        context: str = "guidance",
        owner_id: str | None = None,
        retries_used: int = 0,
        frustration_flag: bool = False,
    ) -> tuple[str, list[dict]]:
        """Convenience wrapper for parent-facing messages."""
        return self.filter(
            audience="parent",
            text=text,
            context=context,
            owner_id=owner_id,
            retries_used=retries_used,
            frustration_flag=frustration_flag,
        )

    # ── HTTP mode ─────────────────────────────────────────────────────────────

    def _call_http(self, **kwargs: Any) -> tuple[str, list[dict]]:
        try:
            import httpx
        except ImportError:
            # httpx not installed — fall through to local
            return self._call_local(**kwargs)

        payload = self._build_payload(**kwargs)
        try:
            with httpx.Client(timeout=self._timeout) as client:
                resp = client.post(f"{self._base_url}/filter", json=payload)
                resp.raise_for_status()
                data = resp.json()
                trace = [
                    {"filter_name": s["filter_name"], "applied": s["applied"], "reason": s.get("reason", "")}
                    for s in data.get("filter_trace", [])
                ]
                return data["filtered_text"], trace
        except Exception:  # noqa: BLE001
            # Network failure — fall back to local pipeline transparently
            return self._call_local(**kwargs)

    # ── Local mode ────────────────────────────────────────────────────────────

    def _call_local(self, **kwargs: Any) -> tuple[str, list[dict]]:
        from app.models import ChildState, FilterRequest
        from app.providers import get_filter_provider

        payload = self._build_payload(**kwargs)
        child_state_data = payload.get("child_state")
        child_state = ChildState(**child_state_data) if child_state_data else None

        req = FilterRequest(
            audience=payload["audience"],
            text=payload["text"],
            context=payload["context"],
            child_state=child_state,
        )

        # Try to attach profile from store if owner_id was provided
        owner_id = kwargs.get("owner_id")
        if owner_id:
            from app.data import profile_store
            profile = profile_store.get_by_owner(owner_id)
            if profile:
                req = req.model_copy(update={"profile": profile})

        result = get_filter_provider().run(req)
        trace = [
            {"filter_name": s.filter_name, "applied": s.applied, "reason": s.reason}
            for s in result.filter_trace
        ]
        return result.filtered_text, trace

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_payload(**kwargs: Any) -> dict:
        engagement_score = kwargs.get("engagement_score", 0.75)
        retries_used = kwargs.get("retries_used", 0)
        frustration_flag = kwargs.get("frustration_flag", False)
        last_action = kwargs.get("last_action", "none")

        child_state = {
            "engagement_score": engagement_score,
            "retries_used": retries_used,
            "frustration_flag": frustration_flag,
            "last_action": last_action,
        }

        return {
            "audience": kwargs["audience"],
            "text": kwargs["text"],
            "context": kwargs.get("context", "general"),
            "child_state": child_state,
            "owner_id": kwargs.get("owner_id"),
        }


# Module-level singleton — import and use directly
filter_client = FilterServiceClient()

