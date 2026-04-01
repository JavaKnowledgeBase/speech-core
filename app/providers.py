from __future__ import annotations

from app.config import settings
from app.models import (
    CommunicationProfile,
    ChildState,
    EnvironmentContext,
    FilterAudience,
    FilterContext,
    FilterLimits,
    FilterRequest,
    FilterResponse,
    FilterStepResult,
)
from app.pipeline import pipeline


class HeuristicFilterProvider:
    """Default provider - runs the full heuristic rule-based pipeline."""

    name = "heuristic"

    def run(self, request: FilterRequest) -> FilterResponse:
        return pipeline.run(request)


class OpenAIFilterProvider:
    """Live provider - heuristics first, then OpenAI refinement."""

    name = "openai"
    _model = "gpt-4o-mini"

    _SYSTEM_PROMPT = (
        "You are an empathy and tone filter for a child speech-therapy platform called TalkBuddy AI. "
        "Your job is to take a pre-cleaned message and polish it so that it is:\n"
        "- Calm and non-alarming\n"
        "- Constructive and kind\n"
        "- Concise (do not add words or padding)\n"
        "- Free of urgency, pressure, or clinical jargon\n"
        "- Appropriate for the specified audience and output kind\n\n"
        "For child messages: use simple words, short sentences, one idea at a time.\n"
        "For parent messages: be practical, supportive, and plain-language.\n"
        "Return ONLY the polished message text - no preamble, no explanation, no quotes."
    )

    def run(self, request: FilterRequest) -> FilterResponse:
        heuristic_result = pipeline.run(request)

        if not settings.configured(settings.openai_api_key):
            return heuristic_result

        try:
            import openai
            client = openai.OpenAI(api_key=settings.openai_api_key)

            user_prompt = self._build_user_prompt(
                text=heuristic_result.filtered_text,
                audience=request.audience,
                context=request.context,
                output_kind=heuristic_result.output_kind,
                child_state=request.child_state,
                environment=request.environment,
                limits=request.limits,
                profile=request.profile,
            )

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
            )

            openai_text = response.choices[0].message.content.strip()
            if not openai_text.endswith("."):
                openai_text += "."

            openai_step = FilterStepResult(
                filter_name="openai_refinement",
                applied=True,
                input_text=heuristic_result.filtered_text,
                output_text=openai_text,
                style_tags_added=["openai-refined"],
                reason=f"OpenAI {self._model} tone refinement applied.",
            )

            all_tags = list(dict.fromkeys(heuristic_result.style_tags + ["openai-refined"]))

            return FilterResponse(
                audience=request.audience,
                output_kind=heuristic_result.output_kind,
                original_text=request.text,
                filtered_text=openai_text,
                style_tags=all_tags,
                filter_trace=heuristic_result.filter_trace + [openai_step],
                architecture="hybrid_rules_model",
                policy=heuristic_result.policy,
                confidence=0.96,
                tone_suggestion=heuristic_result.tone_suggestion,
            )

        except Exception as exc:  # noqa: BLE001
            fallback_step = FilterStepResult(
                filter_name="openai_refinement",
                applied=False,
                input_text=heuristic_result.filtered_text,
                output_text=heuristic_result.filtered_text,
                reason=f"OpenAI call failed ({type(exc).__name__}); heuristic result returned.",
            )
            return FilterResponse(
                audience=request.audience,
                output_kind=heuristic_result.output_kind,
                original_text=request.text,
                filtered_text=heuristic_result.filtered_text,
                style_tags=heuristic_result.style_tags,
                filter_trace=heuristic_result.filter_trace + [fallback_step],
                architecture=heuristic_result.architecture,
                policy=heuristic_result.policy,
                confidence=heuristic_result.confidence,
                tone_suggestion=heuristic_result.tone_suggestion,
            )

    @staticmethod
    def _build_user_prompt(
        text: str,
        audience: FilterAudience,
        context: FilterContext,
        output_kind: str,
        child_state: ChildState | None,
        environment: EnvironmentContext | None,
        limits: FilterLimits | None,
        profile: CommunicationProfile | None,
    ) -> str:
        parts = [
            f"Audience: {audience}",
            f"Therapy context: {context}",
            f"Output kind: {output_kind}",
        ]
        if child_state:
            parts.append(f"Child engagement score: {child_state.engagement_score:.2f}")
            parts.append(f"Retries used: {child_state.retries_used}")
            if child_state.frustration_flag:
                parts.append("Child is showing frustration - be especially gentle.")
        if environment:
            parts.append(f"Distraction level: {environment.distraction_level:.2f}")
            parts.append(f"Noise level: {environment.noise_level:.2f}")
            parts.append(f"Parent stress level: {environment.parent_stress_level:.2f}")
            if environment.screen_on:
                parts.append("A screen is on in the room.")
            if environment.bright_toys_visible:
                parts.append("Bright toys are visible.")
            if environment.notes:
                parts.append(f"Environment notes: {', '.join(environment.notes)}")
        if limits:
            configured_limits = {k: v for k, v in limits.model_dump().items() if v is not None}
            if configured_limits:
                parts.append(f"Request limits: {configured_limits}")
        if profile:
            parts.append(f"Preferred tone: {profile.preferred_tone}")
            parts.append(f"Preferred pacing: {profile.preferred_pacing}")
            if profile.sensory_notes:
                parts.append(f"Sensory notes: {', '.join(profile.sensory_notes)}")
            if profile.banned_styles:
                parts.append(f"Avoid these styles: {', '.join(profile.banned_styles)}")
            if profile.preferred_phrases:
                parts.append(f"Preferred phrases to use: {', '.join(profile.preferred_phrases[:2])}")
            parts.append(f"Max length: {profile.policy.verbosity_limit} characters")
        parts.append(f"\nPre-cleaned message to polish:\n{text}")
        return "\n".join(parts)


def get_filter_provider() -> HeuristicFilterProvider | OpenAIFilterProvider:
    if settings.use_live_provider_calls and settings.configured(settings.openai_api_key):
        return OpenAIFilterProvider()
    return HeuristicFilterProvider()
