from __future__ import annotations

from collections import defaultdict

from app.data import child_attempt_repository, reference_vector_repository
from app.vector_entities import ChildAttemptVectorRecord, ReferenceModality, ReferenceVectorRecord
from app.vector_retrieval_models import AttemptIngestResponse, ReferenceMatchResult, TargetBlendResult
from app.vectors.similarity import cosine_similarity


_MODALITY_TO_FIELD: dict[ReferenceModality, str] = {
    "audio": "audio_embedding",
    "noise": "noise_embedding",
    "lip": "lip_embedding",
    "emotion": "emotion_embedding",
}


def _get_attempt_vector(attempt: ChildAttemptVectorRecord, modality: ReferenceModality) -> list[float]:
    return list(getattr(attempt, _MODALITY_TO_FIELD[modality], []) or [])


def modality_matches(
    attempt: ChildAttemptVectorRecord,
    modality: ReferenceModality,
    k: int = 3,
    min_similarity: float = 0.0,
) -> list[ReferenceMatchResult]:
    query = _get_attempt_vector(attempt, modality)
    if not query:
        return []

    candidates = [
        item for item in reference_vector_repository.list_all()
        if item.modality == modality and item.embedding
    ]

    scored: list[ReferenceMatchResult] = []
    for item in candidates:
        sim = cosine_similarity(query, item.embedding)
        if sim >= min_similarity:
            scored.append(
                ReferenceMatchResult(
                    reference_id=item.reference_id,
                    target_id=item.target_id,
                    modality=item.modality,
                    cosine_similarity=sim,
                    source_label=item.source_label,
                    quality_score=item.quality_score,
                )
            )

    scored.sort(key=lambda item: item.cosine_similarity, reverse=True)
    return scored[:k]


def blended_target_matches(
    attempt: ChildAttemptVectorRecord,
    k: int = 3,
    min_similarity: float = 0.0,
) -> list[TargetBlendResult]:
    by_target: dict[str, dict[ReferenceModality, ReferenceMatchResult]] = defaultdict(dict)

    for modality in _MODALITY_TO_FIELD:
        for match in modality_matches(attempt, modality, k=20, min_similarity=min_similarity):
            current = by_target[match.target_id].get(match.modality)
            if current is None or match.cosine_similarity > current.cosine_similarity:
                by_target[match.target_id][match.modality] = match

    blended: list[TargetBlendResult] = []
    for target_id, matches in by_target.items():
        if not matches:
            continue
        ordered = sorted(matches.values(), key=lambda item: item.cosine_similarity, reverse=True)
        average = round(sum(item.cosine_similarity for item in ordered) / len(ordered), 6)
        blended.append(
            TargetBlendResult(
                target_id=target_id,
                blended_similarity=average,
                matched_modalities=[item.modality for item in ordered],
                top_reference_id=ordered[0].reference_id,
            )
        )

    blended.sort(key=lambda item: item.blended_similarity, reverse=True)
    return blended[:k]


def ingest_attempt(
    attempt: ChildAttemptVectorRecord,
    k: int = 3,
    min_similarity: float = 0.0,
) -> AttemptIngestResponse:
    top_reference_matches: list[ReferenceMatchResult] = []
    for modality in _MODALITY_TO_FIELD:
        top_reference_matches.extend(modality_matches(attempt, modality, k=k, min_similarity=min_similarity))
    top_reference_matches.sort(key=lambda item: item.cosine_similarity, reverse=True)
    top_reference_matches = top_reference_matches[:k]

    blended_matches = blended_target_matches(attempt, k=k, min_similarity=min_similarity)

    updated_attempt = attempt.model_copy(deep=True)
    if blended_matches:
        updated_attempt.top_match_reference_id = blended_matches[0].top_reference_id
        updated_attempt.cosine_similarity = blended_matches[0].blended_similarity

    child_attempt_repository.upsert(updated_attempt)

    return AttemptIngestResponse(
        attempt=updated_attempt,
        top_reference_matches=top_reference_matches,
        blended_target_matches=blended_matches,
    )
