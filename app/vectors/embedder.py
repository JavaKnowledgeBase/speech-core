"""
Embedder — converts text to tone vectors.

Two modes:
  - mock   : returns a deterministic 8-dim vector derived from simple heuristics.
             Used in development and all unit tests. Zero external calls.
  - openai : calls text-embedding-3-small (1536 dims). Requires OPENAI_API_KEY.
             Activated when USE_LIVE_PROVIDER_CALLS=true in env.

The mock embedder is intentionally lightweight — it scores text along the same
8 tonal dimensions used by the phrase library so that cosine similarity produces
meaningful results without an API key.

Heuristic scoring (mock mode):
  [0] warmth      — presence of warm words: good, kind, well, proud, lovely
  [1] energy      — presence of energy markers: great, wonderful, wow, yes!
  [2] directness  — sentence length heuristic: short = direct, long = indirect
  [3] validation  — presence of validation words: try, effort, attempt, brave
  [4] urgency     — presence of urgency words: now, quick, hurry, must
  [5] playfulness — presence of play words: fun, let's, together, play
  [6] brevity     — inverse of word count (more words → less brief)
  [7] safety      — absence of sharp words: wrong, fail, bad, no, can't
"""
from __future__ import annotations

import math
import re

from app.config import settings
from app.vectors.models import ToneEmbedding

# ---------------------------------------------------------------------------
# Heuristic word sets for 8-dim mock embedding
# ---------------------------------------------------------------------------
_WARM_WORDS      = {"good", "well", "kind", "proud", "lovely", "nice", "warm", "brave", "gentle"}
_ENERGY_WORDS    = {"great", "wonderful", "wow", "yes", "amazing", "brilliant", "excellent", "super"}
_VALIDATION_WDS  = {"try", "effort", "attempt", "tried", "trying", "brave", "courage", "kept"}
_URGENCY_WORDS   = {"now", "quick", "hurry", "must", "immediately", "asap", "fast", "rush"}
_PLAYFUL_WORDS   = {"fun", "together", "play", "lets", "let's", "game", "explore", "discover"}
_SHARP_WORDS     = {"wrong", "fail", "failed", "bad", "no", "can't", "cannot", "mistake", "error",
                    "incorrect", "not", "stop", "never", "quit"}


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, round(v, 6)))


def _mock_embed(text: str) -> list[float]:
    """
    Produce a deterministic 8-dim tonal embedding using simple heuristics.
    Designed so that calm, warm, brief phrases score close to the default
    child tone profile initialised by ToneProfileStore.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    total = max(len(words), 1)
    word_set = set(words)

    warm      = _clamp(len(word_set & _WARM_WORDS) / 3.0)
    energy    = _clamp(len(word_set & _ENERGY_WORDS) / 2.0)
    direct    = _clamp(1.0 - (total - 1) / 20.0)     # <5 words=direct, 20+ words=indirect
    validate  = _clamp(len(word_set & _VALIDATION_WDS) / 2.0)
    urgency   = _clamp(len(word_set & _URGENCY_WORDS) / 2.0)
    playful   = _clamp(len(word_set & _PLAYFUL_WORDS) / 2.0)
    brevity   = _clamp(1.0 - (total - 1) / 15.0)     # short=brief
    safety    = _clamp(1.0 - len(word_set & _SHARP_WORDS) / 3.0)

    return [warm, energy, direct, validate, urgency, playful, brevity, safety]


# ---------------------------------------------------------------------------
# OpenAI embedder (lazy import — only used when live mode is active)
# ---------------------------------------------------------------------------

def _openai_embed(text: str) -> list[float]:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "openai package not installed. Run: pip install openai"
        ) from exc

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY not set but USE_LIVE_PROVIDER_CALLS=true")

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float",
    )
    return response.data[0].embedding


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> ToneEmbedding:
    """
    Embed a phrase using the configured provider.

    Returns a ToneEmbedding with either:
      - 8-dim mock vector (source="mock"), or
      - 1536-dim OpenAI vector (source="openai")
    """
    if settings.use_live_provider_calls:
        vector = _openai_embed(text)
        return ToneEmbedding(vector=vector, source="openai", dimensions=len(vector))
    else:
        vector = _mock_embed(text)
        return ToneEmbedding(vector=vector, source="mock", dimensions=8)


def embed_texts(texts: list[str]) -> list[ToneEmbedding]:
    """Embed a batch of texts. Uses a single OpenAI call in live mode."""
    if not texts:
        return []

    if settings.use_live_provider_calls:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError("openai package not installed") from exc

        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        client = OpenAI(api_key=settings.openai_api_key)
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
            encoding_format="float",
        )
        return [
            ToneEmbedding(vector=item.embedding, source="openai", dimensions=len(item.embedding))
            for item in sorted(response.data, key=lambda x: x.index)
        ]
    else:
        return [embed_text(t) for t in texts]
