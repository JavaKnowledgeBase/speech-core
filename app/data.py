from __future__ import annotations

import json
from pathlib import Path

from app.models import CommunicationProfile, OutputPolicy
from app.repositories import (
    ChildAttemptRepository,
    CommunicationProfileRepository,
    EnvironmentStandardRepository,
    OutputFilterProfileRepository,
    ReferenceVectorRepository,
    TargetProfileRepository,
)
from app.vector_entities import (
    EnvironmentStandardProfileRecord,
    OutputFilterProfileRecord,
    ReferenceVectorRecord,
    TargetProfileRecord,
)

_SEED_DIR = Path(__file__).resolve().parent.parent / "seed_data"


def _load_json(name: str) -> list[dict]:
    path = _SEED_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class SeedData:
    @staticmethod
    def communication_profiles() -> list[CommunicationProfile]:
        return [
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

    @staticmethod
    def target_profiles() -> list[TargetProfileRecord]:
        return [TargetProfileRecord.model_validate(item) for item in _load_json("target_profiles.json")]

    @staticmethod
    def reference_vectors() -> list[ReferenceVectorRecord]:
        return [ReferenceVectorRecord.model_validate(item) for item in _load_json("reference_vectors.json")]

    @staticmethod
    def output_filter_profiles() -> list[OutputFilterProfileRecord]:
        return [
            OutputFilterProfileRecord(
                profile_id="ofp-child-1",
                child_id="child-1",
                caregiver_id="caregiver-1",
                preferred_tone_embedding=[0.75, 0.10, 0.35, 0.70, 0.00, 0.15, 0.85, 0.90],
                safe_expression_embedding=[0.80, 0.05, 0.20, 0.65, 0.00, 0.10, 0.90, 0.95],
                best_reengagement_style=[0.65, 0.05, 0.15, 0.40, 0.00, 0.10, 1.00, 0.95],
                parent_guidance_style=[0.78, 0.15, 0.35, 0.68, 0.00, 0.05, 0.65, 0.85],
                overstimulation_flags=[],
                verbosity_limit=72,
                calming_style_vector=[0.75, 0.05, 0.20, 0.55, 0.00, 0.10, 0.85, 0.95],
            )
        ]

    @staticmethod
    def environment_profiles() -> list[EnvironmentStandardProfileRecord]:
        return [
            EnvironmentStandardProfileRecord(
                environment_profile_id="env-child-1",
                child_id="child-1",
                baseline_room_embedding=[0.20, 0.18, 0.75, 0.12, 0.80, 0.35, 0.22, 0.10],
                baseline_visual_clutter_score=0.2,
                baseline_noise_score=0.15,
                baseline_lighting_score=0.8,
                baseline_distraction_notes=["screen off", "minimal bright toys"],
                recommended_adjustments=["move bright toys out of view", "keep TV off during practice"],
            )
        ]


profile_store = CommunicationProfileRepository()
profile_store.seed(SeedData.communication_profiles())

target_profile_repository = TargetProfileRepository()
target_profile_repository.seed(SeedData.target_profiles())

reference_vector_repository = ReferenceVectorRepository()
reference_vector_repository.seed(SeedData.reference_vectors())

child_attempt_repository = ChildAttemptRepository()

output_filter_profile_repository = OutputFilterProfileRepository()
for record in SeedData.output_filter_profiles():
    output_filter_profile_repository.upsert(record)

environment_standard_repository = EnvironmentStandardRepository()
for record in SeedData.environment_profiles():
    environment_standard_repository.upsert(record)
