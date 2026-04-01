from __future__ import annotations

from app.models import CommunicationProfile, OutputPolicy


class ProfileStore:
    """
    In-memory profile store seeded with the same child and caregiver profiles
    that are used in the speech-intelligence platform.

    In production, replace _load() with Supabase reads.
    Profiles are keyed by profile_id.
    """

    def __init__(self) -> None:
        self._profiles: dict[str, CommunicationProfile] = {}
        self._load()

    # ── Seed ──────────────────────────────────────────────────────────────────

    def _load(self) -> None:
        profiles: list[CommunicationProfile] = [
            # ── Child profiles ────────────────────────────────────────────────
            CommunicationProfile(
                profile_id="comm-child-1",
                audience="child",
                owner_id="child-1",
                preferred_tone="gentle and warm",
                preferred_pacing="slow and short",
                sensory_notes=["low stimulation", "one cue at a time"],
                banned_styles=["loud", "fast", "overexcited"],
                preferred_phrases=["quiet try", "small step", "good calm work"],
                policy=OutputPolicy(
                    policy_id="policy-child-1",
                    calmness_level=5,
                    verbosity_limit=72,
                    encouragement_level=3,
                    avoid_overstimulation=True,
                    avoid_exclamations=True,
                    avoid_chatter=True,
                ),
            ),
            CommunicationProfile(
                profile_id="comm-child-2",
                audience="child",
                owner_id="child-2",
                preferred_tone="soft and encouraging",
                preferred_pacing="short and rhythmic",
                sensory_notes=["brief prompts", "steady pacing"],
                banned_styles=["chatty", "intense"],
                preferred_phrases=["try together", "quiet sound", "good steady try"],
                policy=OutputPolicy(
                    policy_id="policy-child-2",
                    calmness_level=5,
                    verbosity_limit=78,
                    encouragement_level=4,
                    avoid_overstimulation=True,
                    avoid_exclamations=True,
                    avoid_chatter=True,
                ),
            ),
            # ── Caregiver / parent profiles ───────────────────────────────────
            CommunicationProfile(
                profile_id="comm-parent-1",
                audience="parent",
                owner_id="caregiver-1",
                preferred_tone="calm and practical",
                preferred_pacing="clear and brief",
                sensory_notes=["avoid overload"],
                banned_styles=["alarmist", "verbose"],
                preferred_phrases=["one calm prompt", "brief model", "quiet support"],
                policy=OutputPolicy(
                    policy_id="policy-parent-1",
                    calmness_level=5,
                    verbosity_limit=132,
                    encouragement_level=2,
                    avoid_overstimulation=True,
                    avoid_exclamations=True,
                    avoid_chatter=True,
                ),
            ),
            CommunicationProfile(
                profile_id="comm-parent-2",
                audience="parent",
                owner_id="caregiver-2",
                preferred_tone="steady and supportive",
                preferred_pacing="short and actionable",
                sensory_notes=["minimize interruptions"],
                banned_styles=["chatter", "pressure"],
                preferred_phrases=["simple cue", "single model", "calm repetition"],
                policy=OutputPolicy(
                    policy_id="policy-parent-2",
                    calmness_level=5,
                    verbosity_limit=136,
                    encouragement_level=2,
                    avoid_overstimulation=True,
                    avoid_exclamations=True,
                    avoid_chatter=True,
                ),
            ),
        ]
        for p in profiles:
            self._profiles[p.profile_id] = p

    # ── CRUD ──────────────────────────────────────────────────────────────────

    def get(self, profile_id: str) -> CommunicationProfile | None:
        return self._profiles.get(profile_id)

    def get_by_owner(self, owner_id: str) -> CommunicationProfile | None:
        for p in self._profiles.values():
            if p.owner_id == owner_id:
                return p
        return None

    def upsert(self, profile: CommunicationProfile) -> None:
        self._profiles[profile.profile_id] = profile

    def delete(self, profile_id: str) -> bool:
        if profile_id in self._profiles:
            del self._profiles[profile_id]
            return True
        return False

    def list_all(self) -> list[CommunicationProfile]:
        return list(self._profiles.values())

    def list_by_audience(self, audience: str) -> list[CommunicationProfile]:
        return [p for p in self._profiles.values() if p.audience == audience]


# Module-level singleton
profile_store = ProfileStore()
