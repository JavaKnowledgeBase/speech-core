from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env_bool(name: str, default: bool) -> bool:
    return os.getenv(name, str(default).lower()).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    app_env: str = field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    use_live_provider_calls: bool = field(default_factory=lambda: _env_bool("USE_LIVE_PROVIDER_CALLS", False))
    default_child_verbosity_limit: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_CHILD_VERBOSITY_LIMIT", "90"))
    )
    default_parent_verbosity_limit: int = field(
        default_factory=lambda: int(os.getenv("DEFAULT_PARENT_VERBOSITY_LIMIT", "140"))
    )
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", ""))
    supabase_profiles_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_PROFILES_TABLE", "communication_profiles")
    )
    supabase_target_profiles_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_TARGET_PROFILES_TABLE", "target_profiles")
    )
    supabase_reference_vectors_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_REFERENCE_VECTORS_TABLE", "reference_vectors")
    )
    supabase_child_attempts_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_CHILD_ATTEMPTS_TABLE", "child_attempt_vectors")
    )
    supabase_output_filter_profiles_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_OUTPUT_FILTER_PROFILES_TABLE", "output_filter_profiles")
    )
    supabase_environment_profiles_table: str = field(
        default_factory=lambda: os.getenv("SUPABASE_ENVIRONMENT_PROFILES_TABLE", "environment_standard_profiles")
    )
    service_api_key: str = field(default_factory=lambda: os.getenv("SERVICE_API_KEY", ""))
    require_auth_in_production: bool = field(default_factory=lambda: _env_bool("REQUIRE_AUTH_IN_PRODUCTION", True))
    allow_openapi_in_production: bool = field(default_factory=lambda: _env_bool("ALLOW_OPENAPI_IN_PRODUCTION", False))
    enable_audit_logging: bool = field(default_factory=lambda: _env_bool("ENABLE_AUDIT_LOGGING", True))
    log_identifier_salt: str = field(default_factory=lambda: os.getenv("LOG_IDENTIFIER_SALT", "speech-core"))

    @staticmethod
    def configured(value: str) -> bool:
        return bool(value and value.strip())

    @property
    def supabase_enabled(self) -> bool:
        return self.configured(self.supabase_url) and self.configured(self.supabase_key)

    @property
    def is_production(self) -> bool:
        return self.app_env.strip().lower() == "production"

    @property
    def auth_required(self) -> bool:
        return self.is_production and self.require_auth_in_production

    @property
    def openapi_enabled(self) -> bool:
        return (not self.is_production) or self.allow_openapi_in_production

    def production_readiness_issues(self) -> list[str]:
        issues: list[str] = []
        if not self.is_production:
            return issues
        if self.require_auth_in_production and not self.configured(self.service_api_key):
            issues.append("SERVICE_API_KEY must be configured when REQUIRE_AUTH_IN_PRODUCTION=true.")
        if not self.supabase_enabled:
            issues.append("SUPABASE_URL and SUPABASE_KEY must be configured for production persistence.")
        return issues


settings = Settings()
