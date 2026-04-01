"""
ToneProfileStore — per-child tone profile storage and update logic.

Responsibilities:
  1. Store ChildToneProfile records (one per child_id)
  2. Provide default profiles when a child is new
  3. Update preferred_tone_embedding after each ToneOutcome using weighted_centroid
  4. Track successful and unsuccessful phrase IDs for profile evolution

Storage:
  Currently in-memory (dict). The Supabase wiring stub is in _persist() —
  when SUPABASE_URL is set, call upsert on the child_tone_profiles table.

Default embedding:
  New children get a default 8-dim profile oriented toward calm, warm, brief
  language — matching the lowest-stimulation phrases in the library:
    [warmth=0.75, energy=0.10, direct=0.35, validate=0.70,
     urgency=0.00, playful=0.15, brevity=0.85, safety=0.90]

  This default ensures the matcher immediately returns sensible results
  even before any session outcomes have been recorded.

Update logic (after each outcome):
  - On success: add phrase embedding to successful pool with weight=1.5,
    then recompute preferred_tone_embedding via weighted_centroid
  - On failure: add phrase_id to unsuccessful_phrase_ids (not reweighted)
  - overstimulation_flags updated from tone_tags of any phrase with
    engagement_score < 0.3
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from app.vectors.models import ChildToneProfile, ToneOutcome
from app.vectors.phrase_library import get_phrase_by_id
from app.vectors.similarity import weighted_centroid

# Default tone profile for a new child — calm, warm, low-energy
_DEFAULT_PREFERRED: list[float] = [0.75, 0.10, 0.35, 0.70, 0.00, 0.15, 0.85, 0.90]
_DEFAULT_SAFE:      list[float] = [0.80, 0.05, 0.20, 0.65, 0.00, 0.10, 0.90, 0.95]
_DEFAULT_CALMING:   list[float] = [0.75, 0.05, 0.20, 0.55, 0.00, 0.10, 0.85, 0.95]
_DEFAULT_REENGAG:   list[float] = [0.65, 0.05, 0.15, 0.40, 0.00, 0.10, 1.00, 0.95]


class ToneProfileStore:
    """
    In-memory store for ChildToneProfile records.

    Usage:
        store = ToneProfileStore()
        profile = store.get_or_create("child-123")
        store.record_outcome(outcome)
    """

    def __init__(self) -> None:
        self._profiles: dict[str, ChildToneProfile] = {}
        # Seed with default profile for demo child IDs
        for child_id in ("child-1", "child-2"):
            self._profiles[child_id] = self._make_default(child_id)

    # ── Public API ──────────────────────────────────────────────────────────

    def get_or_create(self, child_id: str) -> ChildToneProfile:
        """Return existing profile or create a default one."""
        if child_id not in self._profiles:
            self._profiles[child_id] = self._make_default(child_id)
        return self._profiles[child_id]

    def get(self, child_id: str) -> Optional[ChildToneProfile]:
        """Return profile if it exists, otherwise None."""
        return self._profiles.get(child_id)

    def upsert(self, profile: ChildToneProfile) -> None:
        """Store or replace a profile."""
        self._profiles[profile.child_id] = profile
        self._persist(profile)

    def record_outcome(self, outcome: ToneOutcome) -> ChildToneProfile:
        """
        Update the child's tone profile based on a session outcome.

        On success:
          - Add phrase embedding to successful pool (weight=1.5 for outcome,
            weight=0.5 for unsuccessful phrases to pull average gently)
          - Recompute preferred_tone_embedding via weighted_centroid
          - Add phrase_id to successful_phrase_ids

        On failure / low engagement:
          - Add phrase_id to unsuccessful_phrase_ids
          - If engagement_score < 0.3, add phrase tone_tags to overstimulation_flags
        """
        profile = self.get_or_create(outcome.child_id)
        phrase = get_phrase_by_id(outcome.phrase_id)

        if phrase is None:
            # Unknown phrase ID — nothing to learn from
            return profile

        if outcome.success or outcome.engagement_score >= 0.5:
            # Record success
            if outcome.phrase_id not in profile.successful_phrase_ids:
                profile.successful_phrase_ids.append(outcome.phrase_id)

            # Recompute preferred_tone_embedding
            # Gather all successful phrase embeddings + current preferred vector
            vectors: list[list[float]] = [profile.preferred_tone_embedding]
            weights: list[float] = [1.0]  # current preference anchor

            for pid in profile.successful_phrase_ids:
                p = get_phrase_by_id(pid)
                if p:
                    w = 1.5 if pid == outcome.phrase_id else 1.0
                    vectors.append(p.embedding.vector)
                    weights.append(w)

            profile.preferred_tone_embedding = weighted_centroid(vectors, weights)

        else:
            # Record failure
            if outcome.phrase_id not in profile.unsuccessful_phrase_ids:
                profile.unsuccessful_phrase_ids.append(outcome.phrase_id)

            # Flag overstimulating tone tags
            if outcome.engagement_score < 0.3:
                for tag in phrase.tone_tags:
                    if tag not in profile.overstimulation_flags:
                        profile.overstimulation_flags.append(tag)

        profile.total_sessions += 1
        profile.last_updated = datetime.utcnow()

        self.upsert(profile)
        return profile

    def all_profiles(self) -> list[ChildToneProfile]:
        """Return all stored profiles."""
        return list(self._profiles.values())

    # ── Internal ────────────────────────────────────────────────────────────

    @staticmethod
    def _make_default(child_id: str) -> ChildToneProfile:
        return ChildToneProfile(
            profile_id=f"tone-{child_id}",
            child_id=child_id,
            preferred_tone_embedding=list(_DEFAULT_PREFERRED),
            safe_expression_embedding=list(_DEFAULT_SAFE),
            calming_style_vector=list(_DEFAULT_CALMING),
            reengagement_style_vector=list(_DEFAULT_REENGAG),
            successful_phrase_ids=[],
            unsuccessful_phrase_ids=[],
            overstimulation_flags=[],
            total_sessions=0,
            last_updated=datetime.utcnow(),
            embedding_source="mock",
        )

    @staticmethod
    def _persist(profile: ChildToneProfile) -> None:
        """
        Supabase persistence stub.

        When SUPABASE_URL is configured, upsert the profile into the
        child_tone_profiles table. Currently a no-op — in-memory only.

        TODO: wire supabase-py client here when Supabase is configured.
        """
        pass


# Module-level singleton
tone_store = ToneProfileStore()
