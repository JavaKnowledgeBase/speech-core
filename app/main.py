from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.data import (
    child_attempt_repository,
    environment_standard_repository,
    profile_store,
    reference_vector_repository,
    target_profile_repository,
)
from app.models import (
    BatchFilterRequest,
    BatchFilterResponse,
    CommunicationProfile,
    FilterContext,
    FilterPreviewRequest,
    FilterPreviewResponse,
    FilterRequest,
    FilterResponse,
    OutputKindPolicy,
    ProfileUpsertRequest,
    ProfileUpsertResponse,
)
from app.policy_matrix import list_output_policies
from app.providers import get_filter_provider
from app.vector_entities import (
    ChildAttemptVectorRecord,
    EnvironmentStandardProfileRecord,
    ReferenceVectorRecord,
    TargetProfileRecord,
)
from app.vector_retrieval import blended_target_matches, ingest_attempt, modality_matches
from app.vector_retrieval_models import AttemptIngestRequest, AttemptIngestResponse, ReferenceMatchResult, TargetBlendResult
from app.vectors.embedder import embed_text
from app.vectors.matcher import best_match, match
from app.vectors.models import PhraseContext, ToneOutcome
from app.vectors.phrase_library import PHRASE_LIBRARY
from app.vectors.tone_store import tone_store

app = FastAPI(
    title="TalkBuddy Output Filter Service",
    description=(
        "Standalone empathy and output-filter layer for TalkBuddy AI. "
        "All child-facing and parent-facing messages should pass through this service "
        "before delivery. Applies calming, encouraging, re-engagement, parent-guidance, "
        "and frustration-aware filters in sequence. "
        "Set USE_LIVE_PROVIDER_CALLS=true and OPENAI_API_KEY to enable OpenAI refinement."
    ),
    version="1.3.0",
)


@app.get("/health", tags=["System"])
def health() -> dict:
    return {
        "status": "ok",
        "env": settings.app_env,
        "provider": get_filter_provider().name,
        "live_providers": settings.use_live_provider_calls,
        "openai_configured": settings.configured(settings.openai_api_key),
        "supabase_enabled": settings.supabase_enabled,
    }


@app.post("/filter", response_model=FilterResponse, tags=["Filter"])
def filter_output(request: FilterRequest) -> FilterResponse:
    request = _resolve_profile(request)
    return get_filter_provider().run(request)


@app.post("/filter/preview", response_model=FilterPreviewResponse, tags=["Filter"])
def filter_preview(request: FilterPreviewRequest) -> FilterPreviewResponse:
    request = _resolve_profile(request)
    return get_filter_provider().run(request)


@app.post("/filter/batch", response_model=BatchFilterResponse, tags=["Filter"])
def filter_batch(request: BatchFilterRequest) -> BatchFilterResponse:
    provider = get_filter_provider()
    results = [provider.run(_resolve_profile(item)) for item in request.items]
    return BatchFilterResponse(results=results)


@app.post("/filter/child", response_model=FilterResponse, tags=["Filter"])
def filter_child(text: str, context: FilterContext = "general", owner_id: str | None = None) -> FilterResponse:
    profile = profile_store.get_by_owner(owner_id) if owner_id else None
    return get_filter_provider().run(FilterRequest(audience="child", text=text, context=context, profile=profile))


@app.post("/filter/parent", response_model=FilterResponse, tags=["Filter"])
def filter_parent(text: str, context: FilterContext = "guidance", owner_id: str | None = None) -> FilterResponse:
    profile = profile_store.get_by_owner(owner_id) if owner_id else None
    return get_filter_provider().run(FilterRequest(audience="parent", text=text, context=context, profile=profile))


@app.post("/filter/caregiver-alert", response_model=FilterResponse, tags=["Filter"])
def filter_caregiver_alert(text: str, owner_id: str | None = None) -> FilterResponse:
    profile = profile_store.get_by_owner(owner_id) if owner_id else None
    return get_filter_provider().run(
        FilterRequest(audience="parent", text=text, context="escalation", output_kind="caregiver_alert", profile=profile)
    )


@app.post("/filter/environment-guidance", response_model=FilterResponse, tags=["Filter"])
def filter_environment_guidance(text: str, owner_id: str | None = None) -> FilterResponse:
    profile = profile_store.get_by_owner(owner_id) if owner_id else None
    return get_filter_provider().run(
        FilterRequest(
            audience="parent",
            text=text,
            context="guidance",
            output_kind="environment_adjustment_request",
            profile=profile,
        )
    )


@app.post("/profiles", response_model=ProfileUpsertResponse, tags=["Profiles"])
def upsert_profile(request: ProfileUpsertRequest) -> ProfileUpsertResponse:
    profile_store.upsert(request.profile)
    return ProfileUpsertResponse(profile_id=request.profile.profile_id, stored=True)


@app.get("/profiles", response_model=list[CommunicationProfile], tags=["Profiles"])
def list_profiles(audience: str | None = None) -> list[CommunicationProfile]:
    if audience:
        return profile_store.list_by_audience(audience)
    return profile_store.list_all()


@app.get("/profiles/{profile_id}", response_model=CommunicationProfile, tags=["Profiles"])
def get_profile(profile_id: str) -> CommunicationProfile:
    profile = profile_store.get(profile_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return profile


@app.get("/profiles/owner/{owner_id}", response_model=CommunicationProfile, tags=["Profiles"])
def get_profile_by_owner(owner_id: str) -> CommunicationProfile:
    profile = profile_store.get_by_owner(owner_id)
    if profile is None:
        raise HTTPException(status_code=404, detail=f"No profile found for owner '{owner_id}'.")
    return profile


@app.delete("/profiles/{profile_id}", tags=["Profiles"])
def delete_profile(profile_id: str) -> dict:
    deleted = profile_store.delete(profile_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Profile '{profile_id}' not found.")
    return {"deleted": True, "profile_id": profile_id}


@app.get("/targets", response_model=list[TargetProfileRecord], tags=["Vector DB"])
def list_targets() -> list[TargetProfileRecord]:
    return target_profile_repository.list_all()


@app.post("/targets", response_model=TargetProfileRecord, tags=["Vector DB"])
def upsert_target(target: TargetProfileRecord) -> TargetProfileRecord:
    target_profile_repository.upsert(target)
    return target


@app.get("/reference-vectors", response_model=list[ReferenceVectorRecord], tags=["Vector DB"])
def list_reference_vectors(target_id: str | None = None) -> list[ReferenceVectorRecord]:
    if target_id:
        return reference_vector_repository.list_by_target(target_id)
    return reference_vector_repository.list_all()


@app.post("/reference-vectors", response_model=ReferenceVectorRecord, tags=["Vector DB"])
def upsert_reference_vector(item: ReferenceVectorRecord) -> ReferenceVectorRecord:
    reference_vector_repository.upsert(item)
    return item


@app.get("/environment-standards", response_model=list[EnvironmentStandardProfileRecord], tags=["Vector DB"])
def list_environment_standards() -> list[EnvironmentStandardProfileRecord]:
    return environment_standard_repository.list_all()


@app.get("/environment-standards/{child_id}", response_model=EnvironmentStandardProfileRecord, tags=["Vector DB"])
def get_environment_standard(child_id: str) -> EnvironmentStandardProfileRecord:
    item = environment_standard_repository.get_by_child(child_id)
    if item is None:
        raise HTTPException(status_code=404, detail=f"No environment standard found for child '{child_id}'.")
    return item


@app.post("/environment-standards", response_model=EnvironmentStandardProfileRecord, tags=["Vector DB"])
def upsert_environment_standard(item: EnvironmentStandardProfileRecord) -> EnvironmentStandardProfileRecord:
    environment_standard_repository.upsert(item)
    return item


@app.get("/attempts", response_model=list[ChildAttemptVectorRecord], tags=["Vector DB"])
def list_child_attempts(child_id: str | None = None) -> list[ChildAttemptVectorRecord]:
    if child_id:
        return child_attempt_repository.list_by_child(child_id)
    return child_attempt_repository.list_all()


@app.post("/attempts", response_model=ChildAttemptVectorRecord, tags=["Vector DB"])
def upsert_child_attempt(item: ChildAttemptVectorRecord) -> ChildAttemptVectorRecord:
    child_attempt_repository.upsert(item)
    return item


@app.get("/filters/catalogue", tags=["System"])
def filter_catalogue() -> list[dict]:
    return [
        {"name": "calming_filter", "order": 1, "always_active": True, "audience": "child + parent", "trigger": "All output", "what_it_does": "Reduces urgency, exclamations, chatter, and raw intensity before any specialist filter runs."},
        {"name": "encouraging_filter", "order": 2, "always_active": False, "audience": "child", "trigger": "context=success or output_kind=praise_reinforcement", "what_it_does": "Softens praise amplitude and keeps reinforcement short, warm, and low-stimulation."},
        {"name": "frustration_filter", "order": 3, "always_active": False, "audience": "child + parent", "trigger": "retry, escalation, caregiver alerts, or frustration signals", "what_it_does": "Removes blame and pressure language from retry prompts and escalation-oriented messages."},
        {"name": "reengagement_filter", "order": 4, "always_active": False, "audience": "child", "trigger": "context=reengagement or low engagement score", "what_it_does": "Shortens drifting-child prompts and replaces directive language with low-demand invitations."},
        {"name": "parent_guidance_filter", "order": 5, "always_active": False, "audience": "parent", "trigger": "all parent-facing output, with extra shaping for caregiver alerts and environment guidance", "what_it_does": "De-alarms, de-jargons, and keeps parent-facing actions calm and practical."},
        {"name": "architecture", "order": 6, "always_active": True, "audience": "child + parent", "trigger": "every response", "what_it_does": "Resolves an explicit output_kind, attaches its policy matrix, and reports whether the active strategy is rules_only or hybrid_rules_model."},
    ]


@app.get("/filters/policies", response_model=list[OutputKindPolicy], tags=["System"])
def filter_policies() -> list[OutputKindPolicy]:
    return list_output_policies()


@app.get("/providers/status", tags=["System"])
def provider_status() -> dict:
    provider = get_filter_provider()
    return {
        "active_provider": provider.name,
        "openai_configured": settings.configured(settings.openai_api_key),
        "live_mode_enabled": settings.use_live_provider_calls,
        "supabase_enabled": settings.supabase_enabled,
        "strategy": "hybrid_rules_model" if provider.name == "openai" else "rules_only",
        "note": (
            "Set USE_LIVE_PROVIDER_CALLS=true and OPENAI_API_KEY to enable OpenAI refinement."
            if provider.name == "heuristic"
            else "OpenAI two-pass refinement is active."
        ),
    }


@app.get("/tones/library", tags=["Tone Matching"])
def tone_library() -> dict:
    return {
        ctx: [{"phrase_id": p.phrase_id, "text": p.text, "tone_tags": p.tone_tags, "embedding_source": p.embedding.source} for p in phrases]
        for ctx, phrases in PHRASE_LIBRARY.items()
    }


@app.get("/tones/library/{context}", tags=["Tone Matching"])
def tone_library_context(context: PhraseContext) -> list[dict]:
    phrases = PHRASE_LIBRARY.get(context, [])
    return [{"phrase_id": p.phrase_id, "text": p.text, "tone_tags": p.tone_tags, "embedding": p.embedding.vector} for p in phrases]


@app.get("/tones/match/{child_id}/{context}", tags=["Tone Matching"])
def tone_match(child_id: str, context: PhraseContext, k: int = 3) -> list[dict]:
    results = match(child_id, context, k=k)
    return [{"phrase_id": r.phrase_id, "text": r.text, "cosine_similarity": r.cosine_similarity, "tone_tags": r.tone_tags, "matched_by": r.matched_by} for r in results]


@app.get("/tones/best/{child_id}/{context}", tags=["Tone Matching"])
def tone_best_match(child_id: str, context: PhraseContext) -> dict:
    result = best_match(child_id, context)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No phrases available for context '{context}'.")
    return {"phrase_id": result.phrase_id, "text": result.text, "cosine_similarity": result.cosine_similarity, "tone_tags": result.tone_tags, "matched_by": result.matched_by}


@app.get("/tones/profiles", tags=["Tone Matching"])
def list_tone_profiles() -> list[dict]:
    return [{"profile_id": p.profile_id, "child_id": p.child_id, "total_sessions": p.total_sessions, "successful_phrase_ids": p.successful_phrase_ids, "overstimulation_flags": p.overstimulation_flags, "embedding_source": p.embedding_source, "last_updated": p.last_updated.isoformat()} for p in tone_store.all_profiles()]


@app.get("/tones/profiles/{child_id}", tags=["Tone Matching"])
def get_tone_profile(child_id: str) -> dict:
    profile = tone_store.get_or_create(child_id)
    return {
        "profile_id": profile.profile_id,
        "child_id": profile.child_id,
        "preferred_tone_embedding": profile.preferred_tone_embedding,
        "safe_expression_embedding": profile.safe_expression_embedding,
        "calming_style_vector": profile.calming_style_vector,
        "reengagement_style_vector": profile.reengagement_style_vector,
        "successful_phrase_ids": profile.successful_phrase_ids,
        "unsuccessful_phrase_ids": profile.unsuccessful_phrase_ids,
        "overstimulation_flags": profile.overstimulation_flags,
        "total_sessions": profile.total_sessions,
        "embedding_source": profile.embedding_source,
        "last_updated": profile.last_updated.isoformat(),
    }


@app.post("/tones/outcome", tags=["Tone Matching"])
def record_tone_outcome(outcome: ToneOutcome) -> dict:
    updated = tone_store.record_outcome(outcome)
    return {
        "updated": True,
        "child_id": updated.child_id,
        "total_sessions": updated.total_sessions,
        "preferred_tone_embedding": updated.preferred_tone_embedding,
        "successful_phrase_ids": updated.successful_phrase_ids,
        "overstimulation_flags": updated.overstimulation_flags,
    }


@app.post("/tones/embed", tags=["Tone Matching"])
def embed_phrase(text: str) -> dict:
    embedding = embed_text(text)
    return {"text": text, "vector": embedding.vector, "source": embedding.source, "dimensions": embedding.dimensions}


def _resolve_profile(request: FilterRequest) -> FilterRequest:
    if request.profile is not None:
        return request
    return request

@app.post("/attempts/ingest", response_model=AttemptIngestResponse, tags=["Vector DB"])
def ingest_child_attempt(request: AttemptIngestRequest) -> AttemptIngestResponse:
    return ingest_attempt(request.attempt, k=request.top_k, min_similarity=request.min_similarity)


@app.get("/retrieval/modality-match", response_model=list[ReferenceMatchResult], tags=["Vector DB"])
def retrieval_modality_match(
    modality: str,
    child_id: str,
    target_id: str,
    session_id: str,
    embedding: list[float],
    k: int = 3,
    min_similarity: float = 0.0,
) -> list[ReferenceMatchResult]:
    attempt = ChildAttemptVectorRecord(
        attempt_id=f"preview-{child_id}-{target_id}-{modality}",
        child_id=child_id,
        target_id=target_id,
        session_id=session_id,
        **{f"{modality}_embedding": embedding},
    )
    return modality_matches(attempt, modality, k=k, min_similarity=min_similarity)  # type: ignore[arg-type]


@app.post("/retrieval/blended-match", response_model=list[TargetBlendResult], tags=["Vector DB"])
def retrieval_blended_match(
    attempt: ChildAttemptVectorRecord,
    k: int = 3,
    min_similarity: float = 0.0,
) -> list[TargetBlendResult]:
    return blended_target_matches(attempt, k=k, min_similarity=min_similarity)

