from __future__ import annotations

from typing import Optional

from app.filters.base import BaseFilter
from app.filters.calming import CalmingFilter
from app.filters.encouraging import EncouragingFilter
from app.filters.frustration import FrustrationFilter
from app.filters.parent_guidance import ParentGuidanceFilter
from app.filters.reengagement import ReengagementFilter
from app.models import (
    FilterAudience,
    FilterRequest,
    FilterResponse,
    FilterStepResult,
    ToneMatchSuggestion,
    infer_output_kind,
)
from app.policy_matrix import get_output_policy
from app.vectors.models import PhraseContext


class OutputFilterPipeline:
    """Chains specialist filters and records an explicit output architecture trace."""

    def __init__(self) -> None:
        self._filters: list[BaseFilter] = [
            CalmingFilter(),
            EncouragingFilter(),
            FrustrationFilter(),
            ReengagementFilter(),
            ParentGuidanceFilter(),
        ]

    _CONTEXT_MAP: dict[tuple[str, FilterAudience], PhraseContext] = {
        ("praise_reinforcement", "child"): "celebration",
        ("retry_prompt", "child"): "retry_prompt",
        ("escalation_message", "child"): "effort_validation",
        ("reengagement", "child"): "reengagement",
        ("parent_output", "parent"): "parent_guidance",
        ("caregiver_alert", "parent"): "parent_escalation",
        ("environment_adjustment_request", "parent"): "parent_guidance",
    }

    def _tone_suggestion(self, request: FilterRequest, output_kind: str) -> Optional[ToneMatchSuggestion]:
        if not request.child_id:
            return None

        phrase_ctx = "reengagement" if request.context == "reengagement" else self._CONTEXT_MAP.get((output_kind, request.audience))
        if phrase_ctx is None:
            return None

        from app.vectors.matcher import best_match  # noqa: PLC0415

        result = best_match(request.child_id, phrase_ctx)
        if result is None:
            return None

        return ToneMatchSuggestion(
            phrase_id=result.phrase_id,
            text=result.text,
            context=result.context,
            cosine_similarity=result.cosine_similarity,
            tone_tags=result.tone_tags,
            matched_by=result.matched_by,
        )

    def run(self, request: FilterRequest) -> FilterResponse:
        text = request.text
        trace: list[FilterStepResult] = []
        all_tags: list[str] = []
        output_kind = infer_output_kind(request.audience, request.context, request.output_kind)
        policy = get_output_policy(output_kind)

        for f in self._filters:
            result = f.apply(
                text=text,
                audience=request.audience,
                context=request.context,
                output_kind=output_kind,
                child_state=request.child_state,
                profile=request.profile,
                environment=request.environment,
                limits=request.limits,
            )
            trace.append(result)
            if result.applied:
                text = result.output_text
                all_tags.extend(result.style_tags_added)

        confidence = 0.92 if request.profile is not None else 0.88

        return FilterResponse(
            audience=request.audience,
            output_kind=output_kind,
            original_text=request.text,
            filtered_text=text,
            style_tags=list(dict.fromkeys(all_tags)),
            filter_trace=trace,
            architecture="rules_only",
            policy=policy,
            confidence=confidence,
            tone_suggestion=self._tone_suggestion(request, output_kind),
        )

    def run_batch(self, requests: list[FilterRequest]) -> list[FilterResponse]:
        return [self.run(r) for r in requests]


pipeline = OutputFilterPipeline()
