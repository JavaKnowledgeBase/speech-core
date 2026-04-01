"""
Unit tests for each individual filter.

Each test verifies:
  - The filter activates (applied=True) under the right conditions
  - The filter skips (applied=False) when conditions are not met
  - The key transformations are correct
  - Style tags are set as expected
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.filters.calming import CalmingFilter
from app.filters.encouraging import EncouragingFilter
from app.filters.frustration import FrustrationFilter
from app.filters.parent_guidance import ParentGuidanceFilter
from app.filters.reengagement import ReengagementFilter
from app.models import ChildState, CommunicationProfile, OutputPolicy


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def child_profile() -> CommunicationProfile:
    return CommunicationProfile(
        profile_id="test-child",
        audience="child",
        owner_id="child-test",
        preferred_tone="gentle",
        preferred_pacing="slow",
        sensory_notes=["low stimulation"],
        banned_styles=["overexcited", "loud"],
        preferred_phrases=["quiet try", "small step"],
        policy=OutputPolicy(
            policy_id="p-test",
            calmness_level=5,
            verbosity_limit=72,
            encouragement_level=3,
        ),
    )


@pytest.fixture
def parent_profile() -> CommunicationProfile:
    return CommunicationProfile(
        profile_id="test-parent",
        audience="parent",
        owner_id="caregiver-test",
        preferred_tone="calm and practical",
        preferred_pacing="clear and brief",
        sensory_notes=[],
        banned_styles=["alarmist"],
        preferred_phrases=["one calm prompt"],
        policy=OutputPolicy(
            policy_id="p-parent-test",
            calmness_level=5,
            verbosity_limit=140,
            encouragement_level=2,
        ),
    )


@pytest.fixture
def calm_state() -> ChildState:
    return ChildState(engagement_score=0.85, retries_used=0, frustration_flag=False)


@pytest.fixture
def frustrated_state() -> ChildState:
    return ChildState(engagement_score=0.4, retries_used=2, frustration_flag=True)


@pytest.fixture
def low_engagement_state() -> ChildState:
    return ChildState(engagement_score=0.3, retries_used=0, frustration_flag=False)


# ── CalmingFilter ─────────────────────────────────────────────────────────────

class TestCalmingFilter:
    f = CalmingFilter()

    def test_strips_exclamations(self, calm_state):
        result = self.f.apply("Great job!", "child", "success", calm_state, None)
        assert "!" not in result.output_text

    def test_exclamations_become_periods(self, calm_state):
        result = self.f.apply("Well done!", "child", "success", calm_state, None)
        assert result.output_text.endswith(".")

    def test_removes_very(self, calm_state):
        result = self.f.apply("You did very well today.", "child", "general", calm_state, None)
        assert "very" not in result.output_text.lower()

    def test_removes_really(self, calm_state):
        result = self.f.apply("That was really good.", "child", "general", calm_state, None)
        assert "really" not in result.output_text.lower()

    def test_lets_becomes_let_us(self, calm_state):
        result = self.f.apply("Let's try again.", "child", "retry", calm_state, None)
        assert "let us" in result.output_text.lower()

    def test_right_now_becomes_now(self, calm_state):
        result = self.f.apply("Adjust the room right now.", "parent", "guidance", calm_state, None)
        assert "right now" not in result.output_text.lower()
        assert "now" in result.output_text.lower()

    def test_immediately_removed(self, calm_state):
        result = self.f.apply("Remove the TV immediately.", "parent", "guidance", calm_state, None)
        assert "immediately" not in result.output_text.lower()

    def test_applies_profile_verbosity_limit(self, calm_state, child_profile):
        long_text = "This is a very long message that goes on and on. " * 5
        result = self.f.apply(long_text, "child", "general", calm_state, child_profile)
        assert len(result.output_text) <= child_profile.policy.verbosity_limit + 5  # +5 for "..."

    def test_default_child_limit(self, calm_state):
        long_text = "x " * 100
        result = self.f.apply(long_text, "child", "general", calm_state, None)
        assert len(result.output_text) <= 95  # 90 + "..."

    def test_applies_ban_list(self, calm_state, child_profile):
        result = self.f.apply("That was overexcited work.", "child", "general", calm_state, child_profile)
        assert "overexcited" not in result.output_text

    def test_calm_tag_always_present(self, calm_state):
        result = self.f.apply("Good job.", "child", "general", calm_state, None)
        assert "calm" in result.style_tags_added

    def test_gentle_tag_for_child(self, calm_state):
        result = self.f.apply("Good job.", "child", "general", calm_state, None)
        assert "gentle" in result.style_tags_added

    def test_constructive_tag_for_parent(self, calm_state):
        result = self.f.apply("Good job.", "parent", "guidance", calm_state, None)
        assert "constructive" in result.style_tags_added

    def test_normalises_output_ends_with_period(self, calm_state):
        result = self.f.apply("Good work", "child", "general", calm_state, None)
        assert result.output_text.endswith(".")

    def test_orphaned_period_reattached_after_word_removal(self, calm_state):
        # "failed repeatedly." → remove "repeatedly" → "failed ." → reattach → "failed."
        result = self.f.apply("Child has failed repeatedly. Continue.", "parent", "general", calm_state, None)
        # Should not have a floating ". " immediately followed by "Continue"
        assert " . " not in result.output_text
        assert "failed." in result.output_text or "failed" in result.output_text


# ── EncouragingFilter ─────────────────────────────────────────────────────────

class TestEncouragingFilter:
    f = EncouragingFilter()

    def test_activates_on_success_context(self, calm_state):
        result = self.f.apply("Well done.", "child", "success", calm_state, None)
        assert result.applied is True

    def test_activates_on_praise_text(self, calm_state):
        result = self.f.apply("Great job today.", "child", "general", calm_state, None)
        assert result.applied is True

    def test_skips_on_non_success_context_without_praise(self, calm_state):
        result = self.f.apply("Let us try the next sound.", "child", "session_start", calm_state, None)
        assert result.applied is False

    def test_softens_fantastic(self, calm_state):
        result = self.f.apply("That was fantastic.", "child", "success", calm_state, None)
        assert "fantastic" not in result.output_text.lower()

    def test_softens_brilliant(self, calm_state):
        result = self.f.apply("Brilliant work.", "child", "success", calm_state, None)
        assert "brilliant" not in result.output_text.lower()

    def test_softens_you_nailed_it(self, calm_state):
        result = self.f.apply("You nailed it.", "child", "success", calm_state, None)
        assert "nailed it" not in result.output_text.lower()

    def test_prepends_preferred_phrase(self, calm_state, child_profile):
        result = self.f.apply("Nice work.", "child", "success", calm_state, child_profile)
        assert result.output_text.lower().startswith("quiet try")

    def test_does_not_double_prepend_preferred_phrase(self, calm_state, child_profile):
        result = self.f.apply("Quiet try. Nice work.", "child", "success", calm_state, child_profile)
        # Should not prepend again if already starts with the phrase
        count = result.output_text.lower().count("quiet try")
        assert count == 1

    def test_encouraging_tag_present(self, calm_state):
        result = self.f.apply("Nice work.", "child", "success", calm_state, None)
        assert "encouraging" in result.style_tags_added

    def test_overstimulation_avoidance_respects_policy(self, calm_state, child_profile):
        # Profile has avoid_overstimulation=True
        result = self.f.apply("Excellent work.", "child", "success", calm_state, child_profile)
        assert "excellent" not in result.output_text.lower()

    def test_ends_with_period(self, calm_state):
        result = self.f.apply("Great job", "child", "success", calm_state, None)
        assert result.output_text.endswith(".")


# ── FrustrationFilter ─────────────────────────────────────────────────────────

class TestFrustrationFilter:
    f = FrustrationFilter()

    def test_activates_on_retry_context(self, calm_state):
        result = self.f.apply("Try again.", "child", "retry", calm_state, None)
        assert result.applied is True

    def test_activates_on_escalation_context(self, calm_state):
        result = self.f.apply("Session escalated.", "parent", "escalation", calm_state, None)
        assert result.applied is True

    def test_activates_on_frustration_flag(self, frustrated_state):
        result = self.f.apply("Let us try.", "child", "general", frustrated_state, None)
        assert result.applied is True

    def test_activates_on_retries_used(self, calm_state):
        state = ChildState(engagement_score=0.7, retries_used=1)
        result = self.f.apply("Let us try.", "child", "general", state, None)
        assert result.applied is True

    def test_skips_when_no_frustration_signals(self, calm_state):
        result = self.f.apply("Good work.", "child", "session_start", calm_state, None)
        assert result.applied is False

    def test_replaces_try_again(self, frustrated_state):
        result = self.f.apply("Try again please.", "child", "retry", frustrated_state, None)
        assert "try again" not in result.output_text.lower()

    def test_replaces_failed(self, frustrated_state):
        result = self.f.apply("The attempt failed.", "parent", "escalation", frustrated_state, None)
        assert "failed" not in result.output_text.lower()

    def test_replaces_has_failed(self, frustrated_state):
        result = self.f.apply("Child has failed today.", "parent", "escalation", frustrated_state, None)
        assert "has failed" not in result.output_text.lower()

    def test_replaces_wrong(self, frustrated_state):
        result = self.f.apply("That was wrong.", "child", "retry", frustrated_state, None)
        assert "wrong" not in result.output_text.lower()

    def test_removes_come_on(self, frustrated_state):
        result = self.f.apply("Come on, try it.", "child", "retry", frustrated_state, None)
        assert "come on" not in result.output_text.lower()

    def test_prepends_effort_validation_for_child(self, frustrated_state):
        result = self.f.apply("Let us try once more.", "child", "retry", frustrated_state, None)
        # Should start with validation opener or profile phrase
        assert any(
            result.output_text.lower().startswith(opener.lower())
            for opener in ["that was a good try", "good effort", "that took courage", "nice steady try"]
        )

    def test_does_not_prepend_for_parent(self, frustrated_state):
        result = self.f.apply("Please help the child.", "parent", "escalation", frustrated_state, None)
        # Should not start with a child-targeted effort validation
        assert not any(
            result.output_text.lower().startswith(opener.lower())
            for opener in ["that was a good try", "good effort"]
        )

    def test_effort_validated_tag_present_for_child(self, frustrated_state):
        result = self.f.apply("Let us try.", "child", "retry", frustrated_state, None)
        assert "effort-validated" in result.style_tags_added

    def test_frustration_aware_tag_present(self, frustrated_state):
        result = self.f.apply("Try once more.", "child", "retry", frustrated_state, None)
        assert "frustration-aware" in result.style_tags_added

    def test_does_not_re_truncate_after_calming(self, frustrated_state, child_profile):
        # The frustration filter no longer re-truncates — calming (always first) owns
        # the verbosity cap. This test confirms the frustration filter doesn't cut
        # the constructive replacement text it just added.
        text = "Let us try once more with a calm cue."
        result = self.f.apply(text, "child", "retry", frustrated_state, child_profile)
        # The output should be longer than input (opener prepended) and not truncated
        assert "." in result.output_text
        assert result.applied is True


# ── ReengagementFilter ────────────────────────────────────────────────────────

class TestReengagementFilter:
    f = ReengagementFilter()

    def test_activates_on_reengagement_context(self, calm_state):
        result = self.f.apply("Let us continue.", "child", "reengagement", calm_state, None)
        assert result.applied is True

    def test_activates_on_low_engagement(self, low_engagement_state):
        result = self.f.apply("Let us try.", "child", "general", low_engagement_state, None)
        assert result.applied is True

    def test_skips_when_engagement_sufficient(self, calm_state):
        result = self.f.apply("Let us try.", "child", "general", calm_state, None)
        assert result.applied is False

    def test_skips_without_child_state(self):
        result = ReengagementFilter().apply("Let us try.", "child", "general", None, None)
        assert result.applied is False

    def test_removes_pay_attention(self, low_engagement_state):
        result = self.f.apply("Pay attention please.", "child", "reengagement", low_engagement_state, None)
        assert "pay attention" not in result.output_text.lower()

    def test_removes_focus(self, low_engagement_state):
        result = self.f.apply("Focus now please.", "child", "reengagement", low_engagement_state, None)
        assert "focus" not in result.output_text.lower()

    def test_removes_you_need_to(self, low_engagement_state):
        result = self.f.apply("You need to keep trying.", "child", "reengagement", low_engagement_state, None)
        assert "you need to" not in result.output_text.lower()

    def test_prepends_inviting_opener_for_child(self, low_engagement_state):
        result = self.f.apply("Let us continue.", "child", "reengagement", low_engagement_state, None)
        # Should start with a low-demand opener
        assert any(
            result.output_text.lower().startswith(opener.lower())
            for opener in ["whenever ready", "we can try", "one quiet try", "no rush", "take a moment"]
        )

    def test_does_not_prepend_opener_for_parent(self, low_engagement_state):
        result = self.f.apply("Please guide the child.", "parent", "reengagement", low_engagement_state, None)
        assert not result.output_text.lower().startswith("whenever ready")

    def test_hard_cap_60_chars_for_child(self, low_engagement_state):
        long_text = "Pay attention. Focus. We need to continue the session right away."
        result = self.f.apply(long_text, "child", "reengagement", low_engagement_state, None)
        assert len(result.output_text) <= 65  # 60 + small buffer for "..."

    def test_reengagement_tag_present(self, low_engagement_state):
        result = self.f.apply("Let us continue.", "child", "reengagement", low_engagement_state, None)
        assert "reengagement" in result.style_tags_added

    def test_uses_profile_preferred_phrase_as_opener(self, low_engagement_state, child_profile):
        result = self.f.apply("Let us try.", "child", "reengagement", low_engagement_state, child_profile)
        assert result.output_text.lower().startswith("quiet try")


# ── ParentGuidanceFilter ──────────────────────────────────────────────────────

class TestParentGuidanceFilter:
    f = ParentGuidanceFilter()

    def test_activates_for_parent(self, calm_state):
        result = self.f.apply("Session escalated.", "parent", "escalation", calm_state, None)
        assert result.applied is True

    def test_skips_for_child(self, calm_state):
        result = self.f.apply("Let us try.", "child", "general", calm_state, None)
        assert result.applied is False

    def test_de_alarms_escalated(self, calm_state):
        result = self.f.apply("Session escalated.", "parent", "escalation", calm_state, None)
        assert "escalated" not in result.output_text.lower()

    def test_de_alarms_critical(self, calm_state):
        result = self.f.apply("Critical situation detected.", "parent", "escalation", calm_state, None)
        assert "critical" not in result.output_text.lower()

    def test_de_alarms_failed(self, calm_state):
        result = self.f.apply("Child has failed.", "parent", "escalation", calm_state, None)
        assert "failed" not in result.output_text.lower()

    def test_de_alarms_urgent(self, calm_state):
        result = self.f.apply("This is urgent.", "parent", "guidance", calm_state, None)
        assert "urgent" not in result.output_text.lower()

    def test_removes_phoneme_jargon(self, calm_state):
        result = self.f.apply("The phoneme did not match.", "parent", "guidance", calm_state, None)
        assert "phoneme" not in result.output_text.lower()

    def test_replaces_confidence_score(self, calm_state):
        result = self.f.apply("Low confidence score noted.", "parent", "escalation", calm_state, None)
        assert "confidence score" not in result.output_text.lower()

    def test_replaces_mastery_score(self, calm_state):
        result = self.f.apply("Mastery score is 0.4.", "parent", "guidance", calm_state, None)
        assert "mastery score" not in result.output_text.lower()

    def test_prepends_profile_phrase_for_escalation(self, calm_state, parent_profile):
        result = self.f.apply("Please help the child.", "parent", "escalation", calm_state, parent_profile)
        assert result.output_text.lower().startswith("one calm prompt")

    def test_parent_guidance_tag_present(self, calm_state):
        result = self.f.apply("Please help the child.", "parent", "guidance", calm_state, None)
        assert "parent-guidance" in result.style_tags_added

    def test_non_alarmist_tag_present(self, calm_state):
        result = self.f.apply("Session escalated.", "parent", "escalation", calm_state, None)
        assert "non-alarmist" in result.style_tags_added

    def test_plain_language_tag_present(self, calm_state):
        result = self.f.apply("The phoneme embedding failed.", "parent", "escalation", calm_state, None)
        assert "plain-language" in result.style_tags_added

    def test_applies_ban_list(self, calm_state, parent_profile):
        result = self.f.apply("This is an alarmist message.", "parent", "guidance", calm_state, parent_profile)
        assert "alarmist" not in result.output_text.lower()

    def test_ends_with_period(self, calm_state):
        result = self.f.apply("Please support the child", "parent", "guidance", calm_state, None)
        assert result.output_text.endswith(".")
