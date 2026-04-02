# Speech Filters Handoff Note

## Snapshot

This repo is in a strong implementation state for its scoped subsystem.

What is already in place:

- FastAPI service with filter, profile, vector, retrieval, and tone endpoints
- Multi-stage rule pipeline for child and parent output shaping
- Optional OpenAI refinement provider layered after heuristics
- In-memory plus Supabase-backed repository pattern
- Seed data, CSV/JSON import paths, and Supabase schema scaffold
- Retrieval, attempt ingest, and tone-matching flows
- Passing automated test suite

Estimated completion for this repo's current scope: about 80-85%.

## Remaining Work

### 1. Replace scaffold data with real production-grade data

Current status:

- The repo still relies on generated starter vectors and seed profiles.
- The README explicitly says the current 80 reference vectors are scaffold/generated starter data and are not production data.

Why it matters:

- Retrieval quality and tone matching cannot be trusted for real therapy use until the embeddings and targets come from validated sources.

Remaining tasks:

- Collect real target profiles, reference vectors, and environment baselines.
- Define the source-of-truth workflow for updating those datasets.
- Add data validation rules for malformed embeddings, missing targets, duplicate reference IDs, and modality coverage gaps.
- Add tests that use representative real-world examples instead of only scaffold fixtures.

### 2. Finish real persistence and environment integration

Current status:

- Repositories support in-memory fallback and optional Supabase access.
- This is good for development, but not yet a complete production persistence story.

Why it matters:

- Runtime behavior, data consistency, and operational debugging will depend on predictable persistence and error handling.

Remaining tasks:

- Confirm the Supabase schema matches all current Pydantic models and endpoint payloads.
- Add migration/versioning ownership for schema changes.
- Add explicit handling for Supabase failures, timeouts, and partial-write scenarios.
- Decide whether fallback-to-memory is acceptable in production or should be disabled outside development.
- Add integration tests that run against a real or containerized database.

### 3. Wire real profile resolution into the filter flow

Current status:

- The `_resolve_profile()` helper currently returns the request unchanged when no profile is supplied.
- Some endpoints resolve by `owner_id`, but the general request path does not enrich the request automatically.

Why it matters:

- Profile-aware filtering is part of the intended behavior, but the main request path does not fully enforce it yet.

Remaining tasks:

- Decide how a filter request should discover a profile when `profile` is absent.
- Add request fields or lookup rules for `owner_id`, `child_id`, or caregiver identity.
- Make profile resolution consistent across single, preview, and batch endpoints.
- Add tests for automatic profile loading and failure cases.

### 4. Harden the OpenAI provider path

Current status:

- There is an optional live OpenAI refinement step after heuristic filtering.
- The provider uses an older chat-completions style integration and a fixed model string.

Why it matters:

- This path is useful, but it should be modernized and made easier to operate safely before being treated as a production dependency.

Remaining tasks:

- Confirm the intended production OpenAI API path and model choice.
- Add clear retry, timeout, and rate-limit handling.
- Add tests that mock live provider failures and verify safe fallback behavior.
- Decide whether the service should expose provider latency/usage metadata.
- Document rollout rules for when live refinement should be enabled.

### 5. Strengthen API and deployment readiness

Current status:

- The service is runnable and well covered by unit/integration-style tests.
- Deployment and service-operation concerns are still lightly documented.

Why it matters:

- A strong codebase still needs basic production guardrails before handoff into a larger system.

Remaining tasks:

- Add startup instructions for local dev, test, and deployment environments.
- Add pinned runtime/deployment guidance for FastAPI serving, env vars, and health monitoring.
- Add structured logging around filter decisions, provider selection, and retrieval outcomes.
- Add request/response examples for consumers of this service.
- Add authentication/authorization expectations if this will be exposed beyond internal traffic.

### 6. Clean up small implementation seams

Current status:

- The repo is functional, but there are a few signs of first-pass implementation.

Remaining tasks:

- Replace deprecated `datetime.utcnow()` usage with timezone-aware UTC timestamps.
- Clean up encoding artifacts in some test comments.
- Review naming consistency between `speech-core`, `speech-filters`, and `TalkBuddy Output Filter Service`.
- Decide whether tone/profile/vector endpoints belong in this repo long term or should be split by responsibility.

## Suggested Priority Order

1. Real data and validation
2. Real persistence and Supabase integration hardening
3. Profile-resolution completion
4. OpenAI provider modernization
5. Deployment and API operations readiness
6. Cleanup and architecture boundary decisions

## Recommended Next Milestone

The best next milestone is:

`Production-readiness pass for the filter subsystem`

That milestone should include:

- real dataset ingestion
- stable Supabase-backed persistence
- automatic profile resolution
- documented deployment and configuration
- live-provider hardening

## Handoff Summary

This repo should not be treated as a blank or early prototype. It is a mostly built subsystem with strong test coverage and clear architecture. The biggest unfinished areas are not the core filtering logic itself; they are the productionization layers around data quality, persistence, profile-aware integration, and operational readiness.
