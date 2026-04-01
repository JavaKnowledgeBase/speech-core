from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


TargetType = Literal["letter", "number", "word"]
ReferenceModality = Literal["audio", "noise", "lip", "emotion"]


class TargetProfileRecord(BaseModel):
    target_id: str
    target_type: TargetType
    display_text: str
    phoneme_group: str = ""
    difficulty_level: int = Field(default=1, ge=1, le=10)
    active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ReferenceVectorRecord(BaseModel):
    reference_id: str
    target_id: str
    modality: ReferenceModality
    embedding: list[float] = Field(default_factory=list)
    source_label: str = ""
    quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    age_band: str = ""
    notes: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ChildAttemptVectorRecord(BaseModel):
    attempt_id: str
    child_id: str
    target_id: str
    session_id: str
    audio_embedding: list[float] = Field(default_factory=list)
    lip_embedding: list[float] = Field(default_factory=list)
    emotion_embedding: list[float] = Field(default_factory=list)
    noise_embedding: list[float] = Field(default_factory=list)
    top_match_reference_id: str | None = None
    cosine_similarity: float | None = Field(default=None, ge=0.0, le=1.0)
    success_flag: bool | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OutputFilterProfileRecord(BaseModel):
    profile_id: str
    child_id: str | None = None
    caregiver_id: str | None = None
    preferred_tone_embedding: list[float] = Field(default_factory=list)
    safe_expression_embedding: list[float] = Field(default_factory=list)
    best_reengagement_style: list[float] = Field(default_factory=list)
    parent_guidance_style: list[float] = Field(default_factory=list)
    overstimulation_flags: list[str] = Field(default_factory=list)
    verbosity_limit: int = Field(default=100, ge=20, le=240)
    calming_style_vector: list[float] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class EnvironmentStandardProfileRecord(BaseModel):
    environment_profile_id: str
    child_id: str
    baseline_room_embedding: list[float] = Field(default_factory=list)
    baseline_visual_clutter_score: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_noise_score: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_lighting_score: float = Field(default=0.0, ge=0.0, le=1.0)
    baseline_distraction_notes: list[str] = Field(default_factory=list)
    recommended_adjustments: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
