"""
ToneProfileStore - per-child tone profile storage and update logic.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from app.data import output_filter_profile_repository
from app.vector_entities import OutputFilterProfileRecord
from app.vectors.models import ChildToneProfile, ToneOutcome
from app.vectors.phrase_library import get_phrase_by_id
from app.vectors.similarity import weighted_centroid

_DEFAULT_PREFERRED: list[float] = [0.75, 0.10, 0.35, 0.70, 0.00, 0.15, 0.85, 0.90]
_DEFAULT_SAFE:      list[float] = [0.80, 0.05, 0.20, 0.65, 0.00, 0.10, 0.90, 0.95]
_DEFAULT_CALMING:   list[float] = [0.75, 0.05, 0.20, 0.55, 0.00, 0.10, 0.85, 0.95]
_DEFAULT_REENGAG:   list[float] = [0.65, 0.05, 0.15, 0.40, 0.00, 0.10, 1.00, 0.95]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ToneProfileStore:
    def __init__(self) -> None:
        self._profiles: dict[str, ChildToneProfile] = {}
        for child_id in ("child-1", "child-2"):
            self._profiles[child_id] = self._load_or_make_default(child_id)

    def get_or_create(self, child_id: str) -> ChildToneProfile:
        if child_id not in self._profiles:
            self._profiles[child_id] = self._load_or_make_default(child_id)
        return self._profiles[child_id]

    def get(self, child_id: str) -> Optional[ChildToneProfile]:
        return self._profiles.get(child_id)

    def upsert(self, profile: ChildToneProfile) -> None:
        self._profiles[profile.child_id] = profile
        self._persist(profile)

    def record_outcome(self, outcome: ToneOutcome) -> ChildToneProfile:
        profile = self.get_or_create(outcome.child_id)
        phrase = get_phrase_by_id(outcome.phrase_id)

        if phrase is None:
            return profile

        if outcome.success or outcome.engagement_score >= 0.5:
            if outcome.phrase_id not in profile.successful_phrase_ids:
                profile.successful_phrase_ids.append(outcome.phrase_id)

            vectors: list[list[float]] = [profile.preferred_tone_embedding]
            weights: list[float] = [1.0]

            for pid in profile.successful_phrase_ids:
                p = get_phrase_by_id(pid)
                if p:
                    w = 1.5 if pid == outcome.phrase_id else 1.0
                    vectors.append(p.embedding.vector)
                    weights.append(w)

            profile.preferred_tone_embedding = weighted_centroid(vectors, weights)
        else:
            if outcome.phrase_id not in profile.unsuccessful_phrase_ids:
                profile.unsuccessful_phrase_ids.append(outcome.phrase_id)

            if outcome.engagement_score < 0.3:
                for tag in phrase.tone_tags:
                    if tag not in profile.overstimulation_flags:
                        profile.overstimulation_flags.append(tag)

        profile.total_sessions += 1
        profile.last_updated = _utc_now()

        self.upsert(profile)
        return profile

    def all_profiles(self) -> list[ChildToneProfile]:
        return list(self._profiles.values())

    def _load_or_make_default(self, child_id: str) -> ChildToneProfile:
        stored = output_filter_profile_repository.get_by_child(child_id)
        if stored is None:
            return self._make_default(child_id)
        return ChildToneProfile(
            profile_id=stored.profile_id,
            child_id=child_id,
            preferred_tone_embedding=list(stored.preferred_tone_embedding or _DEFAULT_PREFERRED),
            safe_expression_embedding=list(stored.safe_expression_embedding or _DEFAULT_SAFE),
            calming_style_vector=list(stored.calming_style_vector or _DEFAULT_CALMING),
            reengagement_style_vector=list(stored.best_reengagement_style or _DEFAULT_REENGAG),
            successful_phrase_ids=[],
            unsuccessful_phrase_ids=[],
            overstimulation_flags=list(stored.overstimulation_flags),
            total_sessions=0,
            last_updated=stored.updated_at,
            embedding_source="mock",
        )

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
            last_updated=_utc_now(),
            embedding_source="mock",
        )

    @staticmethod
    def _persist(profile: ChildToneProfile) -> None:
        output_filter_profile_repository.upsert(
            OutputFilterProfileRecord(
                profile_id=profile.profile_id,
                child_id=profile.child_id,
                preferred_tone_embedding=list(profile.preferred_tone_embedding),
                safe_expression_embedding=list(profile.safe_expression_embedding),
                best_reengagement_style=list(profile.reengagement_style_vector),
                parent_guidance_style=[],
                overstimulation_flags=list(profile.overstimulation_flags),
                verbosity_limit=100,
                calming_style_vector=list(profile.calming_style_vector),
                updated_at=profile.last_updated,
            )
        )


tone_store = ToneProfileStore()

