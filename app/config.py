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

    @staticmethod
    def configured(value: str) -> bool:
        return bool(value and value.strip())


settings = Settings()
