from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    use_live_provider_calls: bool = field(
        default_factory=lambda: os.getenv("USE_LIVE_PROVIDER_CALLS", "false").lower() == "true"
    )
    default_child_verbosity_limit: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHILD_VERBOSITY_LIMIT", "90"))
    )
    default_parent_verbosity_limit: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_PARENT_VERBOSITY_LIMIT", "140"))
    )
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    supabase_profiles_table: str = field(default_factory=lambda: os.getenv("SUPABASE_PROFILES_TABLE", "communication_profiles"))
    supabase_target_profiles_table: str = field(default_factory=lambda: os.getenv("SUPABASE_TARGET_PROFILES_TABLE", "target_profiles"))
    supabase_reference_vectors_table: str = field(default_factory=lambda: os.getenv("SUPABASE_REFERENCE_VECTORS_TABLE", "reference_vectors"))
    supabase_child_attempts_table: str = field(default_factory=lambda: os.getenv("SUPABASE_CHILD_ATTEMPTS_TABLE", "child_attempt_vectors"))
    supabase_output_filter_profiles_table: str = field(default_factory=lambda: os.getenv("SUPABASE_OUTPUT_FILTER_PROFILES_TABLE", "output_filter_profiles"))
    supabase_environment_profiles_table: str = field(default_factory=lambda: os.getenv("SUPABASE_ENVIRONMENT_PROFILES_TABLE", "environment_standard_profiles"))

    @staticmethod
    def configured(value: str) -> bool:
        return bool(value and value.strip())

    @property
    def supabase_enabled(self) -> bool:
        return self.configured(self.supabase_url) and self.configured(self.supabase_key)


settings = Settings()
