from __future__ import annotations

import re
from abc import ABC, abstractmethod

from app.models import (
    CommunicationProfile,
    ChildState,
    EnvironmentContext,
    FilterAudience,
    FilterContext,
    FilterLimits,
    FilterOutputKind,
    FilterStepResult,
)


class BaseFilter(ABC):
    """Abstract base for all output filters in the pipeline."""

    name: str = "base_filter"

    @abstractmethod
    def apply(
        self,
        text: str,
        audience: FilterAudience,
        context: FilterContext,
        child_state: ChildState | None,
        profile: CommunicationProfile | None,
        output_kind: FilterOutputKind = "child_output",
        environment: EnvironmentContext | None = None,
        limits: FilterLimits | None = None,
    ) -> FilterStepResult:
        ...

    @staticmethod
    def _normalize(text: str) -> str:
        return " ".join(text.strip().split())

    @staticmethod
    def _strip_exclamations(text: str) -> str:
        return text.replace("!", ".")

    @staticmethod
    def _reduce_intensity_words(text: str) -> str:
        pattern = r"\b(very|really|so much|super|absolutely|totally|truly|extremely|incredibly|repeatedly|constantly|always|never)\b"
        return re.sub(pattern, "", text, flags=re.IGNORECASE)

    @staticmethod
    def _remove_chatty_fillers(text: str) -> str:
        fillers = [
            r"\b(you know|i mean|like|basically|literally|right\?|okay\?|alright\?|ok\?)\b",
            r"\b(let me tell you|guess what|oh wow|oh my|oh no|oh yes)\b",
        ]
        for pattern in fillers:
            text = re.sub(pattern, "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _enforce_sentence_end(text: str) -> str:
        stripped = text.rstrip()
        if stripped.endswith(("...", ".", "!", "?")):
            return stripped
        return stripped.rstrip(",;: ") + "."

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[: max(limit - 3, 10)].rstrip()
        last_space = cut.rfind(" ")
        if last_space > limit // 2:
            cut = cut[:last_space]
        return cut.rstrip(".!?,;: ") + "..."

    @staticmethod
    def _apply_ban_list(text: str, banned: list[str]) -> str:
        for phrase in banned:
            text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def _capitalise_sentences(text: str) -> str:
        import re as _re

        def _cap(m: "re.Match") -> str:
            return m.group(1) + m.group(2).upper()

        return _re.sub(r"((?:^|[.!?]\s+))([a-z])", _cap, text)

    @staticmethod
    def _repair_punctuation(text: str) -> str:
        text = re.sub(r"\(\s*[.,]?\s*\)", "", text)
        text = re.sub(r"(\w)\s+\.(\s|$)", r"\1.\2", text)
        text = re.sub(r"(^|\.\s+)\s*\.\s+", r"\1", text)
        text = re.sub(r"\.(\s*\.)+", ".", text)
        text = re.sub(r" {2,}", " ", text)
        text = re.sub(r"^\s*[,;:.]+\s*", "", text)
        text = re.sub(r"(\.\s+),\s*", r"\1", text)
        return text

    @classmethod
    def _clean(cls, text: str) -> str:
        text = cls._repair_punctuation(text)
        text = cls._normalise_orphaned_words(text)
        text = cls._capitalise_sentences(text)
        return cls._normalize(cls._enforce_sentence_end(text))

    @staticmethod
    def _normalise_orphaned_words(text: str) -> str:
        return re.sub(r"\.\s+([a-z]{1,8})\.", lambda m: "." if len(m.group(1)) <= 6 else m.group(0), text)
