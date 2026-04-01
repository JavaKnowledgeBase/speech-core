"""
Integration tests for the full OutputFilterPipeline.

Tests verify end-to-end behaviour across all five filter combinations,
correct trace construction, confidence scoring, and profile-aware routing.
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.models import (
    BatchFilterRequest,
    ChildState,
    CommunicationProfile,
    FilterRequest,
    FilterResponse,
    OutputPolicy,
)
from app.pipeline import OutputFilterPipeline


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def pl() -> OutputFilterPipeline:
    return OutputFilterPipeline()


@pytest.fixture
def child_profile() -> CommunicationProfile:
    return CommunicationProfile(
        profile_id="comm-child-1",
        audience="child",
        owner_id="child-1",
        preferred_tone="gentle and warm",
        preferred_pacing="slow and short",
        sensory_notes=["low stimulation"],
        banned_styles=["loud", "fast", "overexcited"],
        preferred_phrases=["quiet try", "small step"],
        policy=OutputPolicy(
            policy_id="policy-child-1",
            calmness_level=5,
            verbosity_limit=72,
            encouragement_level=3,
        ),
    )


@pytest.fixture
def parent_profile() -> CommunicationProfile:
    return CommunicationProfile(
        profile_id="comm-parent-1",
        audience="parent",
        owner_id="caregiver-1",
        preferred_tone="calm and practical",
        preferred_pacing="clear and brief",
        sensory_notes=["avoid overload"],
        banned_styles=["alarmist", "verbose"],
        preferred_phrases=["one calm prompt", "brief model"],
        policy=OutputPolicy(
            policy_id="policy-parent-1",
            calmness_level=5,
            verbosity_limit=132,
            encouragement_level=2,
        ),
    )


# ── Response shape ────────────────────────────────────────────────────────────

class TestResponseShape:
    def test_returns_filter_response(self, pl):
        req = FilterRequest(audience="child", text="Let us try.", context="general")
        result = pl.run(req)
        assert isinstance(result, FilterResponse)

    def test_original_text_preserved(self, pl):
        text = "Let us try the next sound."
        req = FilterRequest(audience="child", text=text, context="general")
        result = pl.run(req)
        assert result.original_text == text

    def test_filtered_text_not_empty(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        assert result.filtered_text.strip()

    def test_trace_has_five_steps(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        assert len(result.filter_trace) == 5

    def test_trace_filter_names_in_order(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        names = [step.filter_name for step in result.filter_trace]
        assert names == [
            "calming_filter",
            "encouraging_filter",
            "frustration_filter",
            "reengagement_filter",
            "parent_guidance_filter",
        ]

    def test_style_tags_is_list(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        assert isinstance(result.style_tags, list)

    def test_confidence_without_profile(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        assert result.confidence == pytest.approx(0.88)

    def test_confidence_with_profile(self, pl, child_profile):
        req = FilterRequest(audience="child", text="Good work.", context="general", profile=child_profile)
        result = pl.run(req)
        assert result.confidence == pytest.approx(0.92)

    def test_filtered_text_ends_with_period(self, pl):
        req = FilterRequest(audience="child", text="Good work", context="general")
        result = pl.run(req)
        assert result.filtered_text.endswith(".")

    def test_audience_preserved_in_response(self, pl):
        req = FilterRequest(audience="parent", text="Please support the child.", context="guidance")
        result = pl.run(req)
        assert result.audience == "parent"


# ── Calming always active ─────────────────────────────────────────────────────

class TestCalmingAlwaysActive:
    def test_calming_step_always_applied(self, pl):
        req = FilterRequest(audience="child", text="Let's go right now!", context="general")
        result = pl.run(req)
        calming = next(s for s in result.filter_trace if s.filter_name == "calming_filter")
        assert calming.applied is True

    def test_exclamation_removed_in_all_contexts(self, pl):
        for ctx in ["general", "success", "retry", "escalation", "reengagement"]:
            req = FilterRequest(audience="child", text="Great!", context=ctx)  # type: ignore[arg-type]
            result = pl.run(req)
            assert "!" not in result.filtered_text

    def test_calm_tag_always_present(self, pl):
        req = FilterRequest(audience="child", text="Good work.", context="general")
        result = pl.run(req)
        assert "calm" in result.style_tags


# ── Context routing ───────────────────────────────────────────────────────────

class TestContextRouting:
    def test_session_start_only_calming_active(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let us practice 'ba' now.",
            context="session_start",
            child_state=ChildState(engagement_score=0.8, retries_used=0),
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["calming_filter"] is True
        assert trace["encouraging_filter"] is False
        assert trace["frustration_filter"] is False
        assert trace["reengagement_filter"] is False
        assert trace["parent_guidance_filter"] is False

    def test_success_activates_encouraging(self, pl):
        req = FilterRequest(
            audience="child",
            text="Fantastic work today.",
            context="success",
            child_state=ChildState(engagement_score=0.9, retries_used=0),
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["encouraging_filter"] is True

    def test_retry_activates_frustration(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let us try again.",
            context="retry",
            child_state=ChildState(engagement_score=0.6, retries_used=1),
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["frustration_filter"] is True

    def test_low_engagement_activates_reengagement(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let us continue.",
            context="general",
            child_state=ChildState(engagement_score=0.3),
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["reengagement_filter"] is True

    def test_parent_audience_activates_parent_guidance(self, pl):
        req = FilterRequest(
            audience="parent",
            text="Session escalated.",
            context="escalation",
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["parent_guidance_filter"] is True

    def test_child_audience_skips_parent_guidance(self, pl):
        req = FilterRequest(
            audience="child",
            text="We can pause here.",
            context="escalation",
            child_state=ChildState(engagement_score=0.3, retries_used=2),
        )
        result = pl.run(req)
        trace = {s.filter_name: s.applied for s in result.filter_trace}
        assert trace["parent_guidance_filter"] is False


# ── Real therapy message tests ────────────────────────────────────────────────

class TestRealTherapyMessages:
    """Tests using messages taken directly from the speech-intelligence agentic system."""

    def test_session_start_message(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let's practice ba with a short, playful repetition round.",
            context="session_start",
        )
        result = pl.run(req)
        assert "let us" in result.filtered_text.lower()
        assert "!" not in result.filtered_text

    def test_success_advance_message(self, pl):
        req = FilterRequest(
            audience="child",
            text="Nice work. We can move to the next sound now.",
            context="success",
            child_state=ChildState(engagement_score=0.9, retries_used=0),
        )
        result = pl.run(req)
        assert result.filtered_text
        assert "!" not in result.filtered_text

    def test_retry_coaching_message(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let us try that one again with one quiet extra cue.",
            context="retry",
            child_state=ChildState(engagement_score=0.6, retries_used=1),
        )
        result = pl.run(req)
        assert "try again" not in result.filtered_text.lower()

    def test_escalation_child_message(self, pl):
        req = FilterRequest(
            audience="child",
            text="We can pause here. A grown-up will help with the next try.",
            context="escalation",
            child_state=ChildState(engagement_score=0.3, retries_used=2),
        )
        result = pl.run(req)
        assert result.filtered_text
        assert len(result.filtered_text) < 120

    def test_escalation_parent_message_de_alarmed(self, pl):
        req = FilterRequest(
            audience="parent",
            text="Please help Liam with target 'ba'. Use a calm short prompt and model the sound once.",
            context="escalation",
            child_state=ChildState(engagement_score=0.4, retries_used=2),
        )
        result = pl.run(req)
        assert result.filtered_text
        # Agentic messages should already be reasonable — filter should not break them
        assert "ba" in result.filtered_text

    def test_environment_alert_to_parent(self, pl):
        req = FilterRequest(
            audience="parent",
            text="Please adjust the room before starting. Clear bright toys from the immediate view.",
            context="guidance",
        )
        result = pl.run(req)
        assert "please calmly" in result.filtered_text.lower() or "please" in result.filtered_text.lower()
        assert "immediately" not in result.filtered_text.lower()

    def test_session_start_with_profile(self, pl, child_profile):
        req = FilterRequest(
            audience="child",
            text="Let's practice 'ma' with a short repetition round.",
            context="session_start",
            profile=child_profile,
        )
        result = pl.run(req)
        assert result.confidence == pytest.approx(0.92)
        # Verbosity limit respected
        assert len(result.filtered_text) <= child_profile.policy.verbosity_limit + 5

    def test_escalation_with_parent_profile(self, pl, parent_profile):
        req = FilterRequest(
            audience="parent",
            text="Session escalated. Child has failed. Critical intervention required.",
            context="escalation",
            child_state=ChildState(engagement_score=0.3, retries_used=2, frustration_flag=True),
            profile=parent_profile,
        )
        result = pl.run(req)
        assert "escalated" not in result.filtered_text.lower()
        assert "failed" not in result.filtered_text.lower()
        assert "critical" not in result.filtered_text.lower()
        # Profile phrase should be prepended
        assert result.filtered_text.lower().startswith("one calm prompt")


# ── Batch ─────────────────────────────────────────────────────────────────────

class TestBatch:
    def test_batch_returns_same_count(self, pl):
        items = [
            FilterRequest(audience="child", text="Good work.", context="success"),
            FilterRequest(audience="parent", text="Please help.", context="guidance"),
            FilterRequest(audience="child", text="Let us try.", context="retry",
                         child_state=ChildState(retries_used=1)),
        ]
        results = pl.run_batch(items)
        assert len(results) == 3

    def test_batch_each_item_independent(self, pl):
        items = [
            FilterRequest(audience="child", text="Fantastic!", context="success"),
            FilterRequest(audience="child", text="Let us try.", context="session_start"),
        ]
        results = pl.run_batch(items)
        # First should have encouraging filter active
        trace0 = {s.filter_name: s.applied for s in results[0].filter_trace}
        assert trace0["encouraging_filter"] is True
        # Second should not
        trace1 = {s.filter_name: s.applied for s in results[1].filter_trace}
        assert trace1["encouraging_filter"] is False

class TestOutputKindPolicyMatrix:
    def test_general_child_output_returns_child_policy(self, pl):
        req = FilterRequest(audience="child", text="Say ba.", context="general")
        result = pl.run(req)
        assert result.output_kind == "child_output"
        assert result.policy.output_kind == "child_output"
        assert "encouragement to speak" in result.policy.must_preserve
        assert "text-heavy wording" in result.policy.should_block

    def test_retry_prompt_returns_retry_policy(self, pl):
        req = FilterRequest(
            audience="child",
            text="Try again.",
            context="retry",
            child_state=ChildState(retries_used=1),
        )
        result = pl.run(req)
        assert result.output_kind == "retry_prompt"
        assert result.policy.output_kind == "retry_prompt"
        assert "frustration_filter" in result.policy.preferred_filters
        assert "pressure language" in result.policy.should_block or "try harder phrasing" in result.policy.should_block

    def test_environment_guidance_returns_environment_policy(self, pl):
        req = FilterRequest(audience="parent", text="Clear bright toys.", context="guidance")
        result = pl.run(req)
        assert result.output_kind == "environment_adjustment_request"
        assert result.policy.output_kind == "environment_adjustment_request"
        assert "one actionable adjustment" in result.policy.must_preserve
        assert "judgmental phrasing" in result.policy.should_block

class TestEnvironmentAwareFiltering:
    def test_child_environment_high_distraction_reduces_length(self, pl):
        req = FilterRequest(
            audience="child",
            text="Let us keep trying the next sound with a calm cue and one more prompt.",
            context="general",
            environment={"distraction_level": 0.8, "noise_level": 0.7},
        )
        result = pl.run(req)
        assert "environment-aware" in result.style_tags
        assert len(result.filtered_text) <= 75

    def test_reengagement_uses_request_limit(self, pl):
        req = FilterRequest(
            audience="child",
            text="Pay attention and let us keep going with the session.",
            context="reengagement",
            environment={"screen_on": True},
            limits={"reengagement_max_chars": 40},
        )
        result = pl.run(req)
        assert len(result.filtered_text) <= 43

    def test_parent_environment_guidance_uses_room_state(self, pl):
        req = FilterRequest(
            audience="parent",
            text="Please adjust the room before starting.",
            context="guidance",
            environment={"screen_on": True, "bright_toys_visible": True, "noise_level": 0.7},
        )
        result = pl.run(req)
        assert "room-state-aware" in result.style_tags
        assert "change" in result.filtered_text.lower() or "screen" in result.filtered_text.lower()

    def test_parent_limit_override_applies(self, pl):
        req = FilterRequest(
            audience="parent",
            text="Please support the child with a calm short model and then reduce room distractions before another try.",
            context="guidance",
            limits={"parent_max_chars": 55},
        )
        result = pl.run(req)
        assert len(result.filtered_text) <= 58
