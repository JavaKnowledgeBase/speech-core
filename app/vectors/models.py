from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

# Context types the phrase library is indexed by
PhraseContext = Literal[
    "effort_validation",   # opener after a failed attempt — child facing
    "close_attempt",       # replacement for "you failed / wrong / not quite"
    "retry_prompt",        # replacement for "try again"
    "reengagement",        # replacement for "pay attention / focus"
    "celebration",         # softened praise on success
    "parent_guidance",     # parent-facing support instruction
    "parent_escalation",   # parent-facing escalation de-alarm
]

EmbeddingSource = Literal["mock", "openai"]


class ToneEmbedding(BaseModel):
    """A floating-point vector representing the tonal quality of a phrase."""
    vector: list[float]
    source: EmbeddingSource = "mock"
    dimensions: int = 8  # mock uses 8-dim; OpenAI text-embedding-3-small uses 1536


class CandidatePhrase(BaseModel):
    """
    A single candidate output phrase with its tone embedding.

    The system picks the candidate whose embedding is closest to
    the child's learned preferred_tone_embedding.
    """
    phrase_id: str
    text: str
    context: PhraseContext
    tone_tags: list[str] = Field(default_factory=list)
    embedding: ToneEmbedding


class ToneMatchResult(BaseModel):
    """What the matcher found for a given child + context lookup."""
    phrase_id: str
    text: str
    context: PhraseContext
    cosine_similarity: float
    tone_tags: list[str]
    matched_by: EmbeddingSource  # "mock" or "openai"


class ChildToneProfile(BaseModel):
    """
    Per-child learned tone preferences, stored as embeddings.

    These vectors are updated over time as the system observes
    which tones the child responds to best.

    Notes reference fields:
      preferred_tone_embedding  — tone style the child responds to most
      safe_expression_embedding — which expressions feel safe/calm to the child
      calming_style_vector      — what calming delivery looks like for this child
      best_reengagement_style   — which reengagement approach works best
    """
    profile_id: str
    child_id: str

    # Core tone preference vectors
    preferred_tone_embedding: list[float] = Field(default_factory=list)
    safe_expression_embedding: list[float] = Field(default_factory=list)
    calming_style_vector: list[float] = Field(default_factory=list)
    reengagement_style_vector: list[float] = Field(default_factory=list)

    # Learned from session outcomes
    successful_phrase_ids: list[str] = Field(default_factory=list)   # phrases that preceded success
    unsuccessful_phrase_ids: list[str] = Field(default_factory=list) # phrases that preceded failure
    overstimulation_flags: list[str] = Field(default_factory=list)   # tone tags to avoid

    # Metadata
    total_sessions: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    embedding_source: EmbeddingSource = "mock"


class ToneOutcome(BaseModel):
    """
    Feedback signal after a therapy turn — used to update the child's tone profile.
    Call tone_store.record_outcome() after each session turn.
    """
    child_id: str
    phrase_id: str          # phrase that was delivered
    context: PhraseContext
    success: bool           # did the child succeed on the next attempt?
    engagement_score: float = Field(ge=0.0, le=1.0)
