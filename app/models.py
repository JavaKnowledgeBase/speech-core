from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


FilterAudience = Literal["child", "parent"]

FilterOutputKind = Literal[
    "child_output",
    "parent_output",
    "caregiver_alert",
    "retry_prompt",
    "praise_reinforcement",
    "escalation_message",
    "environment_adjustment_request",
]

FilterArchitecture = Literal[
    "rules_only",
    "hybrid_rules_model",
    "model_only",
    "third_party_api_assisted",
]

FilterContext = Literal[
    "session_start",
    "success",
    "retry",
    "escalation",
    "reengagement",
    "guidance",
    "general",
]

FilterTone = Literal["calm", "encouraging", "reengagement", "parent_guidance", "frustration_aware", "neutral"]


class OutputPolicy(BaseModel):
    policy_id: str
    calmness_level: int = Field(ge=1, le=5, default=5)
    verbosity_limit: int = Field(ge=20, le=240, default=100)
    encouragement_level: int = Field(ge=1, le=5, default=3)
    avoid_overstimulation: bool = True
    avoid_exclamations: bool = True
    avoid_chatter: bool = True


class CommunicationProfile(BaseModel):
    profile_id: str
    audience: FilterAudience
    owner_id: str
    preferred_tone: str
    preferred_pacing: str
    sensory_notes: list[str] = Field(default_factory=list)
    banned_styles: list[str] = Field(default_factory=list)
    preferred_phrases: list[str] = Field(default_factory=list)
    policy: OutputPolicy


class ChildState(BaseModel):
    """Snapshot of the child's current in-session state, used by context-aware filters."""
    engagement_score: float = Field(ge=0.0, le=1.0, default=0.75)
    retries_used: int = Field(ge=0, default=0)
    frustration_flag: bool = False
    last_action: Literal["advance", "retry", "escalate", "none"] = "none"


class EnvironmentContext(BaseModel):
    distraction_level: float = Field(ge=0.0, le=1.0, default=0.0)
    noise_level: float = Field(ge=0.0, le=1.0, default=0.0)
    parent_stress_level: float = Field(ge=0.0, le=1.0, default=0.0)
    screen_on: bool = False
    bright_toys_visible: bool = False
    notes: list[str] = Field(default_factory=list)


class FilterLimits(BaseModel):
    child_max_chars: int | None = Field(default=None, ge=20, le=240)
    parent_max_chars: int | None = Field(default=None, ge=20, le=240)
    retry_max_chars: int | None = Field(default=None, ge=20, le=180)
    reengagement_max_chars: int | None = Field(default=None, ge=20, le=120)
    environment_guidance_max_chars: int | None = Field(default=None, ge=20, le=200)


class FilterStepResult(BaseModel):
    """Result produced by a single filter in the pipeline."""
    filter_name: str
    applied: bool
    input_text: str
    output_text: str
    style_tags_added: list[str] = Field(default_factory=list)
    reason: str = ""


class OutputKindPolicy(BaseModel):
    output_kind: FilterOutputKind
    preferred_filters: list[str] = Field(default_factory=list)
    must_preserve: list[str] = Field(default_factory=list)
    should_reduce: list[str] = Field(default_factory=list)
    should_block: list[str] = Field(default_factory=list)
    notes: str = ""


class FilterRequest(BaseModel):
    audience: FilterAudience
    text: str
    context: FilterContext = "general"
    output_kind: FilterOutputKind | None = None
    child_state: ChildState | None = None
    environment: EnvironmentContext | None = None
    limits: FilterLimits | None = None
    profile: CommunicationProfile | None = None
    child_id: str | None = None


class ToneMatchSuggestion(BaseModel):
    phrase_id: str
    text: str
    context: str
    cosine_similarity: float
    tone_tags: list[str]
    matched_by: str


class FilterResponse(BaseModel):
    audience: FilterAudience
    output_kind: FilterOutputKind
    original_text: str
    filtered_text: str
    style_tags: list[str]
    filter_trace: list[FilterStepResult]
    architecture: FilterArchitecture
    policy: OutputKindPolicy
    confidence: float = Field(ge=0.0, le=1.0)
    tone_suggestion: ToneMatchSuggestion | None = None


class BatchFilterRequest(BaseModel):
    items: list[FilterRequest]


class BatchFilterResponse(BaseModel):
    results: list[FilterResponse]


class ProfileUpsertRequest(BaseModel):
    profile: CommunicationProfile


class ProfileUpsertResponse(BaseModel):
    profile_id: str
    stored: bool


FilterPreviewRequest = FilterRequest
FilterPreviewResponse = FilterResponse


def infer_output_kind(
    audience: FilterAudience,
    context: FilterContext,
    output_kind: FilterOutputKind | None = None,
) -> FilterOutputKind:
    if output_kind is not None:
        return output_kind
    if context == "retry":
        return "retry_prompt"
    if context == "success":
        return "praise_reinforcement"
    if context == "escalation":
        return "escalation_message" if audience == "child" else "caregiver_alert"
    if context == "guidance":
        return "environment_adjustment_request" if audience == "parent" else "child_output"
    if audience == "parent":
        return "parent_output"
    return "child_output"
