from __future__ import annotations

import re

from app.filters.base import BaseFilter
from app.models import (
    CommunicationProfile,
    ChildState,
    EnvironmentContext,
    FilterAudience,
    FilterContext,
    FilterLimits,
    FilterOutputKind,
    FilterStepResult,
)


class ParentGuidanceFilter(BaseFilter):
    """Shapes messages directed at the parent or caregiver."""

    name = "parent_guidance_filter"

    _ALARM_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bescalated\b", "needs a moment of support"),
        (r"\bfailed\b", "needs more practice"),
        (r"\blow confidence score\b", "how well the attempt went"),
        (r"\blow confidence\b", "needs a bit more support"),
        (r"\bcritical\b", ""),
        (r"\bsevere\b", "notable"),
        (r"\bimmediate attention\b", "a calm moment of support"),
        (r"\burgent\b", "worth addressing"),
        (r"\bcannot proceed\b", "is pausing for now"),
        (r"\bsession terminated\b", "session is pausing"),
        (r"\bintervention required\b", "your gentle help would be useful here"),
        (r"\bplease intervene\b", "a calm prompt from you would help"),
        (r"\bchild is struggling\b", "your child needs a short calm moment"),
        (r"\byour child has failed\b", "your child needs a short calm moment"),
    ]

    _JARGON_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bphoneme\b", "sound"),
        (r"\bphonemic\b", "sound-based"),
        (r"\bpronunciation score\b", "how clearly the sound was made"),
        (r"\bconfidence score\b", "how well the attempt went"),
        (r"\bcosine similarity\b", "how close the attempt was"),
        (r"\bembedding\b", ""),
        (r"\bvector\b", ""),
        (r"\bescalation\b", "support request"),
        (r"\bmastery score\b", "progress level"),
    ]

    _ENVIRONMENT_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bclear\b", "gently clear"),
        (r"\bremove\b", "move"),
        (r"\bturn off\b", "lower"),
        (r"\bfix\b", "adjust"),
    ]

    def apply(
        self,
        text: str,
        audience: FilterAudience,
        context: FilterContext,
        child_state: ChildState | None,
        profile: CommunicationProfile | None,
        output_kind: FilterOutputKind = "parent_output",
        environment: EnvironmentContext | None = None,
        limits: FilterLimits | None = None,
    ) -> FilterStepResult:
        original = text
        style_tags: list[str] = []

        if audience != "parent":
            return FilterStepResult(
                filter_name=self.name,
                applied=False,
                input_text=original,
                output_text=original,
                reason="Not parent-facing - parent guidance filter skipped.",
            )

        for pattern, replacement in self._ALARM_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        style_tags.append("non-alarmist")

        for pattern, replacement in self._JARGON_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        style_tags.append("plain-language")

        if output_kind == "environment_adjustment_request":
            for pattern, replacement in self._ENVIRONMENT_REPLACEMENTS:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            style_tags.append("environment-aware")

        if environment and output_kind == "environment_adjustment_request":
            hints: list[str] = []
            if environment.screen_on:
                hints.append("lower screen distractions")
            if environment.bright_toys_visible:
                hints.append("move bright toys out of view")
            if environment.noise_level >= 0.6:
                hints.append("reduce background noise")
            if hints and not any(hint in text.lower() for hint in hints):
                text = f"{text} Focus on one change: {hints[0]}."
            style_tags.append("room-state-aware")

        if profile and profile.banned_styles:
            text = self._apply_ban_list(text, profile.banned_styles)

        if (output_kind in ("caregiver_alert", "environment_adjustment_request") or context in ("escalation", "guidance")) and profile and profile.preferred_phrases:
            opener = profile.preferred_phrases[0].rstrip(".").capitalize() + "."
            if not text.lower().startswith(opener.lower().rstrip(".")):
                text = f"{opener} {text}"
            style_tags.append("profile-phrase")

        if limits and output_kind == "environment_adjustment_request" and limits.environment_guidance_max_chars is not None:
            limit = limits.environment_guidance_max_chars
        elif limits and limits.parent_max_chars is not None:
            limit = limits.parent_max_chars
        else:
            limit = profile.policy.verbosity_limit if profile else 140

        if environment and environment.parent_stress_level >= 0.6:
            limit = min(limit, 110)
            style_tags.append("parent-state-aware")

        text = self._truncate(text, limit)
        text = self._clean(text)

        style_tags.append("parent-guidance")
        style_tags.append("supportive")

        return FilterStepResult(
            filter_name=self.name,
            applied=True,
            input_text=original,
            output_text=text,
            style_tags_added=list(dict.fromkeys(style_tags)),
            reason="Applied parent-guidance tone: de-alarmed, plain language, and actionable wording.",
        )
