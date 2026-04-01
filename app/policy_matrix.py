from __future__ import annotations

from app.models import FilterOutputKind, OutputKindPolicy


_POLICY_MATRIX: dict[FilterOutputKind, OutputKindPolicy] = {
    "child_output": OutputKindPolicy(
        output_kind="child_output",
        preferred_filters=["calming_filter"],
        must_preserve=["encouragement to speak", "short prompts", "low-pressure pacing"],
        should_reduce=["verbosity", "chatter", "emotional intensity"],
        should_block=["text-heavy wording", "pressure language", "overstimulating praise"],
        notes="Default child-facing output should stay brief, calm, and speaking-first.",
    ),
    "parent_output": OutputKindPolicy(
        output_kind="parent_output",
        preferred_filters=["calming_filter", "parent_guidance_filter"],
        must_preserve=["clarity", "actionability", "calm support"],
        should_reduce=["jargon", "alarm", "extra explanation"],
        should_block=["blame", "critical tone", "noisy escalation wording"],
        notes="General parent-facing output should be plain-language and practical.",
    ),
    "caregiver_alert": OutputKindPolicy(
        output_kind="caregiver_alert",
        preferred_filters=["calming_filter", "frustration_filter", "parent_guidance_filter"],
        must_preserve=["clear next step", "calm support request", "parent confidence"],
        should_reduce=["alarm", "shame", "clinical language"],
        should_block=["panic wording", "harsh correction", "failure framing"],
        notes="Caregiver alerts should ask for help without escalating parent stress.",
    ),
    "retry_prompt": OutputKindPolicy(
        output_kind="retry_prompt",
        preferred_filters=["calming_filter", "frustration_filter"],
        must_preserve=["encouragement to speak", "one small next try", "effort validation"],
        should_reduce=["pressure", "word count", "repetitive chatter"],
        should_block=["blame", "try harder phrasing", "emotionally intense correction"],
        notes="Retry prompts should feel safe, short, and doable.",
    ),
    "praise_reinforcement": OutputKindPolicy(
        output_kind="praise_reinforcement",
        preferred_filters=["calming_filter", "encouraging_filter"],
        must_preserve=["warm reinforcement", "speaking confidence", "brief celebration"],
        should_reduce=["amplitude", "repetition", "overexcitement"],
        should_block=["loud praise", "chatty celebration", "long detours"],
        notes="Praise should reinforce progress without pushing arousal up.",
    ),
    "escalation_message": OutputKindPolicy(
        output_kind="escalation_message",
        preferred_filters=["calming_filter", "frustration_filter"],
        must_preserve=["felt safety", "clear transition", "calm reassurance"],
        should_reduce=["intensity", "uncertainty", "excess detail"],
        should_block=["abrupt handoff wording", "fear-inducing language", "blame"],
        notes="Child-facing escalation should sound safe and steady, not alarming.",
    ),
    "environment_adjustment_request": OutputKindPolicy(
        output_kind="environment_adjustment_request",
        preferred_filters=["calming_filter", "parent_guidance_filter"],
        must_preserve=["one actionable adjustment", "calm tone", "environment clarity"],
        should_reduce=["criticism", "urgency", "multi-step overload"],
        should_block=["judgmental phrasing", "noisy instructions", "harsh commands"],
        notes="Environment guidance should stay actionable and non-critical.",
    ),
}


def get_output_policy(output_kind: FilterOutputKind) -> OutputKindPolicy:
    return _POLICY_MATRIX[output_kind].model_copy(deep=True)


def list_output_policies() -> list[OutputKindPolicy]:
    return [policy.model_copy(deep=True) for policy in _POLICY_MATRIX.values()]
