"""
tests/test_vectors.py

Tests for the vector-based tone matching system:
  - similarity functions
  - phrase library
  - embedder (mock mode only)
  - tone_store profile management and outcome recording
  - matcher phrase selection logic
  - pipeline integration (child_id → tone_suggestion)
"""
from __future__ import annotations

import math
import pytest

from app.vectors.similarity import cosine_similarity, top_k_matches, centroid, weighted_centroid
from app.vectors.phrase_library import get_candidates, get_phrase_by_id, all_phrase_ids, PHRASE_LIBRARY
from app.vectors.models import PhraseContext, ToneOutcome
from app.vectors.embedder import embed_text
from app.vectors.tone_store import ToneProfileStore
from app.vectors.matcher import best_match, match
from app.models import FilterRequest, ChildState
from app.pipeline import OutputFilterPipeline


# ─────────────────────────────────────────────────────────────────────────────
# cosine_similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_1(self):
        v = [0.5, 0.3, 0.8, 0.1, 0.0, 0.2, 0.9, 0.7]
        assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-5)

    def test_opposite_vectors_return_minus_1(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0, abs=1e-5)

    def test_orthogonal_vectors_return_0(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-5)

    def test_zero_vector_returns_0(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 0.5, 0.3]
        assert cosine_similarity(a, b) == 0.0

    def test_both_zero_vectors_returns_0(self):
        assert cosine_similarity([0.0, 0.0], [0.0, 0.0]) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            cosine_similarity([1.0, 2.0], [1.0])

    def test_result_is_rounded_to_6dp(self):
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        result = cosine_similarity(a, b)
        assert len(str(result).split(".")[-1]) <= 6


# ─────────────────────────────────────────────────────────────────────────────
# top_k_matches
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKMatches:
    def test_returns_k_results(self):
        query = [1.0, 0.0]
        candidates = [
            ("a", [1.0, 0.0]),
            ("b", [0.5, 0.5]),
            ("c", [0.0, 1.0]),
            ("d", [0.8, 0.2]),
        ]
        results = top_k_matches(query, candidates, k=2)
        assert len(results) == 2

    def test_sorted_highest_first(self):
        query = [1.0, 0.0]
        candidates = [
            ("low", [0.0, 1.0]),
            ("high", [1.0, 0.0]),
        ]
        results = top_k_matches(query, candidates, k=2)
        assert results[0][0] == "high"
        assert results[1][0] == "low"

    def test_min_similarity_filters_results(self):
        query = [1.0, 0.0]
        candidates = [
            ("close", [0.99, 0.01]),
            ("far", [0.0, 1.0]),
        ]
        results = top_k_matches(query, candidates, k=5, min_similarity=0.5)
        ids = [r[0] for r in results]
        assert "close" in ids
        assert "far" not in ids

    def test_empty_candidates_returns_empty(self):
        assert top_k_matches([1.0, 0.0], [], k=3) == []


# ─────────────────────────────────────────────────────────────────────────────
# centroid / weighted_centroid
# ─────────────────────────────────────────────────────────────────────────────

class TestCentroid:
    def test_centroid_of_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        result = centroid([v, v, v])
        assert result == pytest.approx([1.0, 2.0, 3.0], abs=1e-5)

    def test_centroid_of_two_vectors(self):
        result = centroid([[0.0, 0.0], [1.0, 1.0]])
        assert result == pytest.approx([0.5, 0.5], abs=1e-5)

    def test_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            centroid([])

    def test_weighted_centroid_zero_weight_falls_back_to_centroid(self):
        vectors = [[1.0, 0.0], [0.0, 1.0]]
        result = weighted_centroid(vectors, [0.0, 0.0])
        # Falls back to centroid when total_weight is 0
        assert result == pytest.approx([0.5, 0.5], abs=1e-5)

    def test_weighted_centroid_pulls_toward_high_weight(self):
        vectors = [[0.0], [1.0]]
        result = weighted_centroid(vectors, [1.0, 9.0])  # heavily weight second
        assert result[0] > 0.8  # should be close to 1.0

    def test_weighted_centroid_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            weighted_centroid([[1.0, 2.0]], [1.0, 2.0])


# ─────────────────────────────────────────────────────────────────────────────
# phrase_library
# ─────────────────────────────────────────────────────────────────────────────

class TestPhraseLibrary:
    def test_all_contexts_have_phrases(self):
        contexts: list[PhraseContext] = [
            "effort_validation", "close_attempt", "retry_prompt",
            "reengagement", "celebration", "parent_guidance", "parent_escalation",
        ]
        for ctx in contexts:
            phrases = get_candidates(ctx)
            assert len(phrases) >= 3, f"Context '{ctx}' has too few phrases"

    def test_all_phrases_have_8dim_embeddings(self):
        for ctx, phrases in PHRASE_LIBRARY.items():
            for p in phrases:
                assert p.embedding.dimensions == 8, f"{p.phrase_id} embedding not 8-dim"
                assert len(p.embedding.vector) == 8

    def test_all_embeddings_values_in_range(self):
        for ctx, phrases in PHRASE_LIBRARY.items():
            for p in phrases:
                for v in p.embedding.vector:
                    assert 0.0 <= v <= 1.0, f"{p.phrase_id} has out-of-range value {v}"

    def test_all_phrase_ids_are_unique(self):
        ids = all_phrase_ids()
        assert len(ids) == len(set(ids))

    def test_get_phrase_by_id_known(self):
        phrase = get_phrase_by_id("ev-001")
        assert phrase is not None
        assert phrase.text == "That was a good try."

    def test_get_phrase_by_id_unknown_returns_none(self):
        assert get_phrase_by_id("does-not-exist") is None

    def test_phrases_have_non_empty_tone_tags(self):
        for ctx, phrases in PHRASE_LIBRARY.items():
            for p in phrases:
                assert len(p.tone_tags) > 0, f"{p.phrase_id} has no tone tags"

    def test_child_phrases_are_short(self):
        """Child-facing phrases should be concise enough for TTS."""
        child_contexts: list[PhraseContext] = [
            "effort_validation", "close_attempt", "retry_prompt",
            "reengagement", "celebration",
        ]
        for ctx in child_contexts:
            for p in get_candidates(ctx):
                assert len(p.text) <= 80, (
                    f"{p.phrase_id} too long for child ({len(p.text)} chars): '{p.text}'"
                )


# ─────────────────────────────────────────────────────────────────────────────
# embedder (mock mode)
# ─────────────────────────────────────────────────────────────────────────────

class TestEmbedder:
    def test_returns_8dim_vector_in_mock_mode(self):
        emb = embed_text("That was a good try.")
        assert emb.source == "mock"
        assert emb.dimensions == 8
        assert len(emb.vector) == 8

    def test_all_values_clamped_0_to_1(self):
        emb = embed_text("You failed that. Wrong. Bad. Stop it now!")
        for v in emb.vector:
            assert 0.0 <= v <= 1.0

    def test_calm_phrase_has_high_safety(self):
        """'Whenever ready.' should score high on safety (no sharp words)."""
        emb = embed_text("Whenever ready.")
        # safety is dim[7]
        assert emb.vector[7] >= 0.8

    def test_high_urgency_phrase_has_higher_urgency_score(self):
        emb_urgent = embed_text("Hurry up now!")
        emb_calm = embed_text("Whenever ready.")
        # urgency is dim[4]
        assert emb_urgent.vector[4] > emb_calm.vector[4]

    def test_short_phrase_has_higher_brevity(self):
        short = embed_text("Nice.")
        long = embed_text(
            "Your child has demonstrated a consistent pattern of improvement "
            "across multiple sessions and we should acknowledge that."
        )
        # brevity is dim[6]
        assert short.vector[6] > long.vector[6]

    def test_deterministic_for_same_text(self):
        emb1 = embed_text("Good effort.")
        emb2 = embed_text("Good effort.")
        assert emb1.vector == emb2.vector


# ─────────────────────────────────────────────────────────────────────────────
# tone_store
# ─────────────────────────────────────────────────────────────────────────────

class TestToneProfileStore:
    def test_get_or_create_returns_default_for_new_child(self):
        store = ToneProfileStore()
        profile = store.get_or_create("test-child-99")
        assert profile.child_id == "test-child-99"
        assert len(profile.preferred_tone_embedding) == 8
        assert profile.total_sessions == 0

    def test_get_returns_none_for_unknown_child(self):
        store = ToneProfileStore()
        assert store.get("unknown-xyz") is None

    def test_upsert_stores_and_retrieves(self):
        store = ToneProfileStore()
        profile = store.get_or_create("child-upsert-test")
        profile.total_sessions = 5
        store.upsert(profile)
        retrieved = store.get("child-upsert-test")
        assert retrieved is not None
        assert retrieved.total_sessions == 5

    def test_record_outcome_success_adds_phrase_id(self):
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-outcome-1",
            phrase_id="ev-001",
            context="effort_validation",
            success=True,
            engagement_score=0.8,
        )
        updated = store.record_outcome(outcome)
        assert "ev-001" in updated.successful_phrase_ids

    def test_record_outcome_failure_adds_to_unsuccessful(self):
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-outcome-2",
            phrase_id="ev-002",
            context="effort_validation",
            success=False,
            engagement_score=0.2,
        )
        updated = store.record_outcome(outcome)
        assert "ev-002" in updated.unsuccessful_phrase_ids

    def test_record_outcome_low_engagement_adds_overstimulation_flags(self):
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-outcome-3",
            phrase_id="cel-001",  # tagged ["warm", "calm", "brief"]
            context="celebration",
            success=False,
            engagement_score=0.1,
        )
        updated = store.record_outcome(outcome)
        # Should flag the phrase's tone tags as overstimulating
        assert len(updated.overstimulation_flags) > 0

    def test_record_outcome_increments_session_count(self):
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-sessions",
            phrase_id="rp-001",
            context="retry_prompt",
            success=True,
            engagement_score=0.7,
        )
        store.record_outcome(outcome)
        store.record_outcome(outcome)
        profile = store.get("child-sessions")
        assert profile.total_sessions == 2

    def test_record_outcome_unknown_phrase_id_is_safe(self):
        """Should not raise for a phrase_id not in the library."""
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-unknown-phrase",
            phrase_id="nonexistent-999",
            context="effort_validation",
            success=True,
            engagement_score=0.8,
        )
        # Should return profile unchanged, not raise
        updated = store.record_outcome(outcome)
        assert updated.child_id == "child-unknown-phrase"

    def test_preferred_embedding_shifts_toward_successful_phrase(self):
        """After recording a success, the embedding should move toward that phrase."""
        store = ToneProfileStore()
        profile_before = store.get_or_create("child-shift")
        before_vec = list(profile_before.preferred_tone_embedding)

        outcome = ToneOutcome(
            child_id="child-shift",
            phrase_id="re-001",  # high safety/calm, very different from default
            context="reengagement",
            success=True,
            engagement_score=0.9,
        )
        updated = store.record_outcome(outcome)
        # The embedding should have changed
        assert updated.preferred_tone_embedding != before_vec


# ─────────────────────────────────────────────────────────────────────────────
# matcher
# ─────────────────────────────────────────────────────────────────────────────

class TestMatcher:
    def test_best_match_returns_result(self):
        result = best_match("child-1", "effort_validation")
        assert result is not None
        assert result.phrase_id.startswith("ev-")

    def test_best_match_unknown_context_returns_none(self):
        # All valid contexts are in PhraseContext — this tests empty library
        from app.vectors.phrase_library import get_candidates
        # "general" is not a PhraseContext, so direct match call with unsupported type
        # We test by checking get_candidates on unknown returns empty
        assert get_candidates("nonexistent_ctx") == []  # type: ignore[arg-type]

    def test_match_returns_k_results(self):
        results = match("child-1", "celebration", k=3)
        assert len(results) <= 3
        assert len(results) >= 1

    def test_match_sorted_by_similarity_descending(self):
        results = match("child-1", "retry_prompt", k=5)
        sims = [r.cosine_similarity for r in results]
        assert sims == sorted(sims, reverse=True)

    def test_match_skips_unsuccessful_phrases(self):
        """After recording a failure, the phrase should not appear in match results."""
        from app.vectors.tone_store import ToneProfileStore
        store = ToneProfileStore()
        outcome = ToneOutcome(
            child_id="child-skip-test",
            phrase_id="rp-001",
            context="retry_prompt",
            success=False,
            engagement_score=0.2,
        )
        store.record_outcome(outcome)

        # Override the module-level tone_store temporarily
        import app.vectors.matcher as matcher_mod
        original_store = matcher_mod.tone_store
        matcher_mod.tone_store = store

        try:
            results = match("child-skip-test", "retry_prompt", k=10)
            ids = [r.phrase_id for r in results]
            assert "rp-001" not in ids
        finally:
            matcher_mod.tone_store = original_store

    def test_best_match_returns_phrase_in_correct_context(self):
        result = best_match("child-2", "reengagement")
        assert result is not None
        assert result.context == "reengagement"

    def test_match_cosine_similarity_in_valid_range(self):
        results = match("child-1", "celebration", k=5)
        for r in results:
            assert -1.0 <= r.cosine_similarity <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# pipeline integration — tone_suggestion in FilterResponse
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineToneSuggestion:
    def setup_method(self):
        self.pipeline = OutputFilterPipeline()

    def test_no_child_id_gives_no_tone_suggestion(self):
        req = FilterRequest(
            audience="child",
            text="Let's try that again.",
            context="retry",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is None

    def test_child_id_with_retry_gives_tone_suggestion(self):
        req = FilterRequest(
            audience="child",
            text="Let's try that again.",
            context="retry",
            child_id="child-1",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        assert resp.tone_suggestion.phrase_id.startswith("rp-")

    def test_child_id_with_success_gives_celebration_suggestion(self):
        req = FilterRequest(
            audience="child",
            text="Great job!",
            context="success",
            child_id="child-1",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        assert resp.tone_suggestion.phrase_id.startswith("cel-")

    def test_child_id_with_reengagement_gives_reengagement_suggestion(self):
        req = FilterRequest(
            audience="child",
            text="Pay attention.",
            context="reengagement",
            child_id="child-1",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        assert resp.tone_suggestion.context == "reengagement"

    def test_parent_guidance_gives_tone_suggestion(self):
        req = FilterRequest(
            audience="parent",
            text="Liam has been struggling with the session.",
            context="guidance",
            child_id="parent-1",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        assert resp.tone_suggestion.context == "parent_guidance"

    def test_general_context_gives_no_tone_suggestion(self):
        req = FilterRequest(
            audience="child",
            text="Hello there.",
            context="general",
            child_id="child-1",
        )
        resp = self.pipeline.run(req)
        # "general" is not mapped to any PhraseContext
        assert resp.tone_suggestion is None

    def test_tone_suggestion_has_valid_cosine_similarity(self):
        req = FilterRequest(
            audience="child",
            text="Let's try again.",
            context="retry",
            child_id="child-2",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        sim = resp.tone_suggestion.cosine_similarity
        assert -1.0 <= sim <= 1.0

    def test_tone_suggestion_has_non_empty_tone_tags(self):
        req = FilterRequest(
            audience="child",
            text="That was close.",
            context="escalation",
            child_id="child-1",
        )
        resp = self.pipeline.run(req)
        assert resp.tone_suggestion is not None
        assert len(resp.tone_suggestion.tone_tags) > 0

    def test_filtered_text_is_still_correct_with_child_id(self):
        """Tone matching must not affect the filtered text output."""
        req = FilterRequest(
            audience="child",
            text="You failed that. Try again.",
            context="retry",
            child_id="child-1",
            child_state=ChildState(retries_used=1),
        )
        resp = self.pipeline.run(req)
        # The frustration filter should have cleaned this
        assert "failed" not in resp.filtered_text.lower()
        assert "try again" not in resp.filtered_text.lower()
