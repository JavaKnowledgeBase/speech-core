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


class CalmingFilter(BaseFilter):
    """Always-on first-pass filter."""

    name = "calming_filter"

    _INTENSITY_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bPlease\b", "Please calmly"),
        (r"\bLet's\b", "Let us"),
        (r"\blets\b", "let us"),
        (r"\bright now\b", "now"),
        (r"\bimmediately\b", "now"),
        (r"\bhurry\b", ""),
        (r"\bquickly\b", ""),
        (r"\bfast\b", ""),
        (r"\bcome on\b", ""),
        (r"\bdon't worry\b", "that is okay"),
        (r"\bno worries\b", "that is okay"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?failed\s+(?:that|it|this)\b", "that was close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?failed\b", "that was close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?got\s+(?:that|it|this)\s+wrong\b", "that was close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?did\s+(?:that|it|this)\s+wrong\b", "that was close"),
        (r"\byou\s+(?:just\s+)?can't\s+(?:do this|do it|get it)\b", "we can keep trying"),
    ]

    def apply(
        self,
        text: str,
        audience: FilterAudience,
        context: FilterContext,
        child_state: ChildState | None,
        profile: CommunicationProfile | None,
        output_kind: FilterOutputKind = "child_output",
        environment: EnvironmentContext | None = None,
        limits: FilterLimits | None = None,
    ) -> FilterStepResult:
        original = text
        style_tags: list[str] = []

        if profile is None or profile.policy.avoid_exclamations:
            text = self._strip_exclamations(text)
            style_tags.append("no-exclamations")

        for pattern, replacement in self._INTENSITY_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        text = self._reduce_intensity_words(text)
        style_tags.append("low-intensity")

        if profile is None or profile.policy.avoid_chatter:
            text = self._remove_chatty_fillers(text)
            style_tags.append("non-chatty")

        if profile is not None and profile.banned_styles:
            text = self._apply_ban_list(text, profile.banned_styles)

        if limits and output_kind == "retry_prompt" and limits.retry_max_chars is not None:
            limit = limits.retry_max_chars
        elif limits and output_kind == "environment_adjustment_request" and limits.environment_guidance_max_chars is not None:
            limit = limits.environment_guidance_max_chars
        elif limits and audience == "child" and limits.child_max_chars is not None:
            limit = limits.child_max_chars
        elif limits and audience == "parent" and limits.parent_max_chars is not None:
            limit = limits.parent_max_chars
        elif profile is not None:
            limit = profile.policy.verbosity_limit
        elif audience == "child":
            limit = 90
        elif output_kind == "environment_adjustment_request":
            limit = 120
        else:
            limit = 140

        if environment and audience == "child" and (environment.distraction_level >= 0.6 or environment.noise_level >= 0.6):
            limit = min(limit, 72)
            style_tags.append("environment-aware")

        if environment and environment.parent_stress_level >= 0.6 and audience == "parent":
            limit = min(limit, 110)
            style_tags.append("parent-state-aware")

        text = self._truncate(text, limit)
        text = self._clean(text)
        style_tags.append("calm")
        if audience == "child":
            style_tags.append("gentle")
        else:
            style_tags.append("constructive")

        return FilterStepResult(
            filter_name=self.name,
            applied=True,
            input_text=original,
            output_text=text,
            style_tags_added=list(dict.fromkeys(style_tags)),
            reason="Reduced arousal: stripped exclamations, intensity words, and filler phrases.",
        )
