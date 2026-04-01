"""
ToneMatcher — selects the best candidate phrase for a given child and context.

Algorithm:
  1. Load the child's ChildToneProfile (or use defaults for new children)
  2. Choose the query vector based on context:
       - reengagement → reengagement_style_vector
       - effort_validation, close_attempt, retry_prompt → preferred_tone_embedding
       - celebration → preferred_tone_embedding (slightly elevated energy tolerated)
       - parent_* → preferred_tone_embedding (parent profiles use separate child_ids)
  3. Fetch all CandidatePhrase records for the requested PhraseContext
  4. Filter out any phrases in unsuccessful_phrase_ids (learned avoidance)
  5. Penalise phrases whose tone_tags overlap with overstimulation_flags
  6. Run cosine similarity against the query vector
  7. Return top-k results as ToneMatchResult records

The matcher is stateless — it reads from phrase_library and tone_store but
never writes. Call tone_store.record_outcome() separately after each turn.
"""
from __future__ import annotations

from app.vectors.models import ChildToneProfile, PhraseContext, ToneMatchResult
from app.vectors.phrase_library import get_candidates
from app.vectors.similarity import cosine_similarity
from app.vectors.tone_store import tone_store

_OVERSTIM_PENALTY = 0.20   # subtract this from score if tag overlap detected


def _select_query_vector(
    profile: ChildToneProfile,
    context: PhraseContext,
) -> list[float]:
    """Pick which profile vector to query against for this context."""
    if context == "reengagement":
        return profile.reengagement_style_vector or profile.preferred_tone_embedding
    return profile.preferred_tone_embedding


def match(
    child_id: str,
    context: PhraseContext,
    k: int = 1,
    min_similarity: float = 0.0,
) -> list[ToneMatchResult]:
    """
    Return the top-k tone-matched phrases for a child in a given context.

    Args:
        child_id:       The child's ID (used to load their ToneProfile)
        context:        Which phrase category to search (e.g. "retry_prompt")
        k:              Number of results to return
        min_similarity: Minimum cosine similarity (0.0 = no threshold)

    Returns:
        List of ToneMatchResult sorted highest similarity first.
        Returns an empty list if no candidates exist for the context.
    """
    profile = tone_store.get_or_create(child_id)
    query_vec = _select_query_vector(profile, context)

    candidates = get_candidates(context)
    if not candidates:
        return []

    scored: list[tuple[str, str, PhraseContext, list[str], float, str]] = []

    for phrase in candidates:
        # Skip phrases learned to be ineffective for this child
        if phrase.phrase_id in profile.unsuccessful_phrase_ids:
            continue

        sim = cosine_similarity(query_vec, phrase.embedding.vector)

        # Apply overstimulation penalty if any tags match flagged tags
        if profile.overstimulation_flags:
            overlap = set(phrase.tone_tags) & set(profile.overstimulation_flags)
            if overlap:
                sim = max(0.0, sim - _OVERSTIM_PENALTY)

        if sim >= min_similarity:
            scored.append((
                phrase.phrase_id,
                phrase.text,
                phrase.context,
                phrase.tone_tags,
                sim,
                phrase.embedding.source,
            ))

    scored.sort(key=lambda x: x[4], reverse=True)
    top = scored[:k]

    return [
        ToneMatchResult(
            phrase_id=pid,
            text=text,
            context=ctx,
            cosine_similarity=round(sim, 6),
            tone_tags=tags,
            matched_by=source,
        )
        for pid, text, ctx, tags, sim, source in top
    ]


def best_match(
    child_id: str,
    context: PhraseContext,
) -> ToneMatchResult | None:
    """
    Convenience wrapper — returns the single best match or None.

    Used by the filter pipeline when it needs one replacement phrase.
    """
    results = match(child_id, context, k=1)
    return results[0] if results else None
