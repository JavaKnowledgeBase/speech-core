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


class FrustrationFilter(BaseFilter):
    """Replaces frustrating, blaming, or pressuring language with constructive alternatives."""

    name = "frustration_filter"

    _PRESSURE_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?failed\s+(?:that|it|this)\b", "that was close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?got\s+(?:that|it|this)\s+wrong\b", "that was really close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?did\s+(?:that|it|this)\s+wrong\b", "that was really close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?got\s+it\s+wrong\b", "that was really close"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?failed\b", "that was a good attempt"),
        (r"\byou\s+(?:really\s+|truly\s+|just\s+)?messed\s+(?:that|it|this)\s+up\b", "that was close"),
        (r"\byou\s+(?:just\s+)?can't\s+(?:do this|do it|get it)\b", "we can keep practising together"),
        (r"\byou\s+(?:just\s+)?won't\s+(?:do this|do it|get it)\b", "we can keep practising together"),
        (r"\btry again\b", "one more quiet try when ready"),
        (r"\btry once more\b", "one more calm try"),
        (r"\bone more time\b", "one more calm try"),
        (r"\bagain\b", "once more"),
        (r"\bwhy can't you\b", "we can keep working on"),
        (r"\byou should\b", ""),
        (r"\byou need to\b", ""),
        (r"\bcome on\b", ""),
        (r"\bhurry up\b", ""),
        (r"\bthat(?:'s| was) wrong\b", "that was close"),
        (r"\bthat(?:'s| was) incorrect\b", "that was close"),
        (r"\bincorrect\b", "almost there"),
        (r"\bnot quite right\b", "getting closer"),
        (r"\bnot quite\b", "getting closer"),
        (r"\bwrong\b", "almost there"),
        (r"\bmistake\b", "attempt"),
        (r"\berror\b", "attempt"),
        (r"\bbad\b", ""),
        (r"\bnope\b", ""),
        (r"\bno\b(?!\s*,)", ""),
        (r"\btoo hard\b", "worth practising"),
        (r"\btoo difficult\b", "worth practising"),
        (r"\bcan't do it\b", "can keep practising"),
        (r"\bcouldn't do it\b", "is still practising"),
        (r"\bgave up\b", "took a break"),
        (r"\bquit\b", "paused"),
        (r"\bgiving up\b", "taking a moment"),
        (r"\bhas failed\b", "is making progress"),
        (r"\bhave failed\b", "are making progress"),
        (r"\bfailed\b", "gave a great try"),
        (r"\bpoor performance\b", "great effort"),
        (r"\bunderperforming\b", "still developing"),
        (r"\blow performance\b", "early stage"),
        (r"\bnot progressing\b", "building towards the goal"),
        (r"\bregressed\b", "had a quieter session"),
        (r"\bregression\b", "a quieter session"),
    ]

    _VALIDATION_OPENERS: list[str] = [
        "That was a good try.",
        "Good effort.",
        "That took courage.",
        "Nice steady try.",
        "That was brave.",
    ]

    def _is_active(self, context: FilterContext, output_kind: FilterOutputKind, child_state: ChildState | None) -> bool:
        if context in ("retry", "escalation"):
            return True
        if output_kind in ("retry_prompt", "escalation_message", "caregiver_alert"):
            return True
        if child_state is None:
            return False
        return child_state.frustration_flag or child_state.retries_used >= 1

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

        if not self._is_active(context, output_kind, child_state):
            return FilterStepResult(
                filter_name=self.name,
                applied=False,
                input_text=original,
                output_text=original,
                reason="No frustration signals - frustration filter skipped.",
            )

        for pattern, replacement in self._PRESSURE_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        style_tags.append("pressure-free")
        style_tags.append("constructive")

        if audience == "child":
            opener = self._VALIDATION_OPENERS[0]
            if profile and profile.preferred_phrases:
                opener = profile.preferred_phrases[0].rstrip(".").capitalize() + "."
            if not text.lower().startswith(opener.lower().rstrip(".")):
                text = f"{opener} {text}"
            style_tags.append("effort-validated")

        text = self._clean(text)
        style_tags.append("frustration-aware")

        return FilterStepResult(
            filter_name=self.name,
            applied=True,
            input_text=original,
            output_text=text,
            style_tags_added=list(dict.fromkeys(style_tags)),
            reason="Replaced pressure/blame language with constructive encouragement.",
        )
