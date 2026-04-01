from __future__ import annotations

from pydantic import BaseModel, Field

from app.vector_entities import ChildAttemptVectorRecord, ReferenceModality


class ReferenceMatchResult(BaseModel):
    reference_id: str
    target_id: str
    modality: ReferenceModality
    cosine_similarity: float = Field(ge=-1.0, le=1.0)
    source_label: str = ""
    quality_score: float = Field(ge=0.0, le=1.0)


class TargetBlendResult(BaseModel):
    target_id: str
    blended_similarity: float = Field(ge=-1.0, le=1.0)
    matched_modalities: list[ReferenceModality] = Field(default_factory=list)
    top_reference_id: str | None = None


class AttemptIngestRequest(BaseModel):
    attempt: ChildAttemptVectorRecord
    top_k: int = Field(default=3, ge=1, le=20)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)


class AttemptIngestResponse(BaseModel):
    attempt: ChildAttemptVectorRecord
    top_reference_matches: list[ReferenceMatchResult] = Field(default_factory=list)
    blended_target_matches: list[TargetBlendResult] = Field(default_factory=list)
