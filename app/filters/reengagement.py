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


class ReengagementFilter(BaseFilter):
    """Reshapes output to gently recapture a drifting child's attention."""

    name = "reengagement_filter"

    _DIRECTIVE_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bpay attention\b", "we can continue when ready"),
        (r"\blisten\b", "when ready"),
        (r"\bfocus\b", ""),
        (r"\bstay with me\b", ""),
        (r"\blook at me\b", ""),
        (r"\bcome back\b", ""),
        (r"\bstop\b", ""),
        (r"\bwait\b", ""),
        (r"\bdo this\b", "try this"),
        (r"\byou need to\b", ""),
        (r"\byou must\b", ""),
        (r"\byou have to\b", ""),
    ]

    _CHILD_OPENERS: list[str] = [
        "Whenever ready.",
        "We can try when you feel like it.",
        "One quiet try.",
        "No rush.",
        "Take a moment.",
    ]

    def _is_active(self, context: FilterContext, child_state: ChildState | None) -> bool:
        if context == "reengagement":
            return True
        if child_state is not None and child_state.engagement_score < 0.55:
            return True
        return False

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

        if audience != "child" or not self._is_active(context, child_state):
            return FilterStepResult(
                filter_name=self.name,
                applied=False,
                input_text=original,
                output_text=original,
                reason="Engagement is sufficient - reengagement filter skipped.",
            )

        for pattern, replacement in self._DIRECTIVE_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        style_tags.append("low-demand")

        opener = self._CHILD_OPENERS[0]
        if profile and profile.preferred_phrases:
            opener = profile.preferred_phrases[0].rstrip(".").capitalize() + "."
        elif environment and (environment.screen_on or environment.bright_toys_visible):
            opener = "One quiet try."
            style_tags.append("environment-aware")
        if not text.lower().startswith(opener.lower().rstrip(".")):
            text = f"{opener} {text}"

        if limits and limits.reengagement_max_chars is not None:
            limit = limits.reengagement_max_chars
        else:
            limit = min(profile.policy.verbosity_limit if profile else 90, 60)

        if environment and (environment.distraction_level >= 0.7 or environment.noise_level >= 0.7):
            limit = min(limit, 50)
            style_tags.append("extra-brief")

        text = self._truncate(text, limit)
        text = self._clean(text)
        style_tags.append("brief")
        style_tags.append("reengagement")

        return FilterStepResult(
            filter_name=self.name,
            applied=True,
            input_text=original,
            output_text=text,
            style_tags_added=list(dict.fromkeys(style_tags)),
            reason="Child engagement is low - output shortened and directive language removed.",
        )
