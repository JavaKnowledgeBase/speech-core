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


class EncouragingFilter(BaseFilter):
    """Shapes success and positive-feedback messages without overstimulation."""

    name = "encouraging_filter"

    _PRAISE_REPLACEMENTS: list[tuple[str, str]] = [
        (r"\bperfect\b", "good"),
        (r"\bbrilliant\b", "good"),
        (r"\bincredible\b", "good"),
        (r"\bfantastic\b", "good"),
        (r"\bunbelievable\b", "good"),
        (r"\boutstanding\b", "nice"),
        (r"\bexcellent\b", "nice"),
        (r"\bgreat job\b", "nice work"),
        (r"\bwonderful\b", "good"),
        (r"\bsuperb\b", "good"),
        (r"\byou did it\b", "that worked"),
        (r"\byou nailed it\b", "that worked"),
        (r"\bkudos\b", ""),
        (r"\bbravo\b", ""),
        (r"\byay\b", ""),
        (r"\bwoo\b", ""),
        (r"\bwoohoo\b", ""),
    ]

    def _is_active(self, context: FilterContext, output_kind: FilterOutputKind, text: str) -> bool:
        if context == "success" or output_kind == "praise_reinforcement":
            return True
        praise_pattern = r"\b(nice work|good job|well done|great|excellent|fantastic|brilliant|perfect|you did)\b"
        return bool(re.search(praise_pattern, text, re.IGNORECASE))

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

        if audience != "child" or not self._is_active(context, output_kind, text):
            return FilterStepResult(
                filter_name=self.name,
                applied=False,
                input_text=original,
                output_text=original,
                reason="Not a child praise/reinforcement message - filter skipped.",
            )

        avoid_overstimulation = profile.policy.avoid_overstimulation if profile else True

        if avoid_overstimulation:
            for pattern, replacement in self._PRAISE_REPLACEMENTS:
                text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
            style_tags.append("low-stimulation-praise")

        if environment and (environment.noise_level >= 0.6 or environment.distraction_level >= 0.6):
            text = self._truncate(text, 60)
            style_tags.append("environment-aware")

        if profile and profile.preferred_phrases:
            preferred = profile.preferred_phrases[0].rstrip(".").capitalize()
            if not text.lower().startswith(preferred.lower()):
                text = f"{preferred}. {text}"
            style_tags.append("profile-phrase")

        text = self._clean(text)
        style_tags.append("encouraging")

        return FilterStepResult(
            filter_name=self.name,
            applied=True,
            input_text=original,
            output_text=text,
            style_tags_added=list(dict.fromkeys(style_tags)),
            reason="Softened praise amplitude to prevent overstimulation.",
        )
