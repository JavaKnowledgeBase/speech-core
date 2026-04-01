"""
Candidate phrase library for tone-matched output selection.

Each phrase is indexed by PhraseContext and carries a pre-computed 8-dim mock
embedding that encodes its tonal profile along these dimensions:

    [0] warmth        — 0.0 = neutral, 1.0 = very warm
    [1] energy        — 0.0 = very calm/quiet, 1.0 = high energy
    [2] directness    — 0.0 = soft/indirect, 1.0 = firm/direct
    [3] validation    — 0.0 = low effort acknowledgement, 1.0 = strong validation
    [4] urgency       — 0.0 = no time pressure, 1.0 = urgent
    [5] playfulness   — 0.0 = serious/clinical, 1.0 = playful/light
    [6] brevity       — 0.0 = longer phrasing, 1.0 = very short/simple
    [7] safety        — 0.0 = challenging/stimulating, 1.0 = safe/soothing

When OpenAI embeddings are active, these 8-dim vectors are replaced by 1536-dim
text-embedding-3-small vectors. The cosine similarity logic is identical.

Design principles (from speech therapy and AAC research):
  - Language must be honest — never false praise, always grounded in the attempt
  - Tone must be calm and low-energy for children who overstimulate easily
  - Effort is always acknowledged before outcome
  - Parent phrases de-alarm, de-jargon, and orient toward next steps
  - Phrases are short enough to be used as TTS output without pacing problems
"""
from __future__ import annotations

from app.vectors.models import CandidatePhrase, PhraseContext, ToneEmbedding

# ---------------------------------------------------------------------------
# Phrase definitions
# Each tuple: (phrase_id, text, context, tone_tags, embedding_vector_8d)
# ---------------------------------------------------------------------------

_RAW: list[tuple[str, str, PhraseContext, list[str], list[float]]] = [

    # ── effort_validation ────────────────────────────────────────────────────
    # Used after a failed attempt — shown to the child BEFORE offering a retry.
    # Key design: honest, warm, low-energy. Never "great job!" on a fail.
    (
        "ev-001", "That was a good try.",
        "effort_validation",
        ["warm", "calm", "validating"],
        [0.80, 0.20, 0.40, 0.85, 0.00, 0.20, 0.90, 0.80],
    ),
    (
        "ev-002", "Good effort.",
        "effort_validation",
        ["warm", "very-brief", "calm"],
        [0.70, 0.15, 0.50, 0.70, 0.00, 0.10, 1.00, 0.85],
    ),
    (
        "ev-003", "That took courage.",
        "effort_validation",
        ["warm", "brave", "calm"],
        [0.85, 0.20, 0.35, 0.90, 0.00, 0.10, 0.85, 0.80],
    ),
    (
        "ev-004", "Nice steady try.",
        "effort_validation",
        ["calm", "steady", "validating"],
        [0.75, 0.10, 0.30, 0.80, 0.00, 0.15, 0.90, 0.90],
    ),
    (
        "ev-005", "That was brave.",
        "effort_validation",
        ["warm", "brave", "brief"],
        [0.80, 0.25, 0.40, 0.85, 0.00, 0.20, 0.95, 0.75],
    ),
    (
        "ev-006", "You kept going — that matters.",
        "effort_validation",
        ["warm", "persistence", "calm"],
        [0.85, 0.20, 0.30, 0.90, 0.00, 0.10, 0.70, 0.80],
    ),
    (
        "ev-007", "Every try builds something.",
        "effort_validation",
        ["calm", "growth", "gentle"],
        [0.75, 0.15, 0.25, 0.85, 0.00, 0.15, 0.80, 0.85],
    ),

    # ── close_attempt ────────────────────────────────────────────────────────
    # Replaces "wrong", "failed", "incorrect" — must be TRUE and constructive.
    # Never "almost!" (false enthusiasm). "Close" or "getting there" is honest.
    (
        "ca-001", "That was close.",
        "close_attempt",
        ["calm", "honest", "brief"],
        [0.65, 0.15, 0.40, 0.60, 0.00, 0.10, 0.95, 0.85],
    ),
    (
        "ca-002", "That was really close.",
        "close_attempt",
        ["warm", "honest", "brief"],
        [0.70, 0.20, 0.35, 0.70, 0.00, 0.15, 0.90, 0.80],
    ),
    (
        "ca-003", "Getting closer.",
        "close_attempt",
        ["calm", "progress", "brief"],
        [0.65, 0.20, 0.45, 0.65, 0.00, 0.20, 0.95, 0.80],
    ),
    (
        "ca-004", "Almost there.",
        "close_attempt",
        ["warm", "progress", "gentle"],
        [0.70, 0.25, 0.40, 0.70, 0.00, 0.20, 0.90, 0.80],
    ),
    (
        "ca-005", "That was a strong attempt.",
        "close_attempt",
        ["warm", "validating", "calm"],
        [0.78, 0.20, 0.45, 0.80, 0.00, 0.10, 0.80, 0.80],
    ),
    (
        "ca-006", "You're making progress.",
        "close_attempt",
        ["warm", "growth", "gentle"],
        [0.75, 0.20, 0.30, 0.75, 0.00, 0.10, 0.85, 0.80],
    ),
    (
        "ca-007", "That one was tricky.",
        "close_attempt",
        ["calm", "normalising", "gentle"],
        [0.70, 0.15, 0.25, 0.65, 0.00, 0.20, 0.85, 0.85],
    ),

    # ── retry_prompt ─────────────────────────────────────────────────────────
    # Replaces "try again", "one more time", "try harder".
    # Must remove time pressure. Child chooses when ready.
    (
        "rp-001", "One more quiet try when ready.",
        "retry_prompt",
        ["calm", "no-pressure", "child-led"],
        [0.75, 0.10, 0.30, 0.60, 0.00, 0.15, 0.80, 0.90],
    ),
    (
        "rp-002", "One more calm try.",
        "retry_prompt",
        ["calm", "brief", "no-pressure"],
        [0.65, 0.10, 0.40, 0.55, 0.00, 0.10, 0.90, 0.90],
    ),
    (
        "rp-003", "Whenever you're ready.",
        "retry_prompt",
        ["calm", "child-led", "very-brief"],
        [0.70, 0.05, 0.20, 0.50, 0.00, 0.15, 0.95, 0.95],
    ),
    (
        "rp-004", "Take your time.",
        "retry_prompt",
        ["soothing", "no-pressure", "brief"],
        [0.75, 0.05, 0.15, 0.50, 0.00, 0.10, 0.90, 0.95],
    ),
    (
        "rp-005", "We can try that sound again together.",
        "retry_prompt",
        ["warm", "collaborative", "no-pressure"],
        [0.85, 0.15, 0.25, 0.65, 0.00, 0.20, 0.70, 0.85],
    ),
    (
        "rp-006", "Ready when you are.",
        "retry_prompt",
        ["calm", "child-led", "brief"],
        [0.70, 0.10, 0.25, 0.50, 0.00, 0.10, 0.90, 0.90],
    ),

    # ── reengagement ─────────────────────────────────────────────────────────
    # Replaces "pay attention", "focus", "listen up".
    # Low energy, no pressure. Used when engagement drops.
    (
        "re-001", "Whenever ready.",
        "reengagement",
        ["calm", "no-pressure", "very-brief"],
        [0.65, 0.05, 0.15, 0.40, 0.00, 0.10, 1.00, 0.95],
    ),
    (
        "re-002", "No rush.",
        "reengagement",
        ["soothing", "no-pressure", "very-brief"],
        [0.70, 0.05, 0.10, 0.35, 0.00, 0.10, 1.00, 0.95],
    ),
    (
        "re-003", "We can take a moment.",
        "reengagement",
        ["calm", "soothing", "child-led"],
        [0.75, 0.05, 0.15, 0.40, 0.00, 0.10, 0.90, 0.95],
    ),
    (
        "re-004", "Just here when you want to continue.",
        "reengagement",
        ["calm", "no-pressure", "child-led"],
        [0.70, 0.05, 0.15, 0.35, 0.00, 0.10, 0.80, 0.95],
    ),
    (
        "re-005", "Let's take a breath first.",
        "reengagement",
        ["calm", "soothing", "regulating"],
        [0.75, 0.05, 0.20, 0.40, 0.00, 0.15, 0.85, 0.95],
    ),
    (
        "re-006", "All good. We'll go again when it feels right.",
        "reengagement",
        ["warm", "no-pressure", "soothing"],
        [0.80, 0.10, 0.15, 0.45, 0.00, 0.10, 0.70, 0.90],
    ),

    # ── celebration ──────────────────────────────────────────────────────────
    # On genuine success — softened, not overwhelming.
    # Avoids "brilliant!", "amazing!", "fantastic!" — too high energy for
    # children with sensory sensitivity. "Nice." > "WONDERFUL!!".
    (
        "cel-001", "Nice work.",
        "celebration",
        ["warm", "calm", "brief"],
        [0.75, 0.30, 0.50, 0.70, 0.00, 0.20, 0.95, 0.75],
    ),
    (
        "cel-002", "That worked.",
        "celebration",
        ["calm", "honest", "very-brief"],
        [0.60, 0.25, 0.55, 0.60, 0.00, 0.10, 1.00, 0.80],
    ),
    (
        "cel-003", "Good job.",
        "celebration",
        ["warm", "brief", "calm"],
        [0.70, 0.30, 0.55, 0.65, 0.00, 0.20, 0.95, 0.75],
    ),
    (
        "cel-004", "You got it.",
        "celebration",
        ["warm", "direct", "brief"],
        [0.70, 0.35, 0.65, 0.65, 0.00, 0.25, 0.95, 0.70],
    ),
    (
        "cel-005", "That's exactly it.",
        "celebration",
        ["warm", "precise", "calm"],
        [0.75, 0.30, 0.60, 0.70, 0.00, 0.15, 0.90, 0.75],
    ),
    (
        "cel-006", "Well done.",
        "celebration",
        ["warm", "calm", "brief"],
        [0.75, 0.25, 0.50, 0.70, 0.00, 0.15, 0.95, 0.80],
    ),
    (
        "cel-007", "You did it.",
        "celebration",
        ["warm", "direct", "brief"],
        [0.75, 0.35, 0.60, 0.70, 0.00, 0.20, 0.95, 0.70],
    ),

    # ── parent_guidance ───────────────────────────────────────────────────────
    # Clinician/system → parent. De-alarmed, practical, next-step focused.
    # Research basis: parents under stress can't process jargon. Keep it plain.
    (
        "pg-001", "Liam is still building this skill — that's completely normal at this stage.",
        "parent_guidance",
        ["reassuring", "normalising", "practical"],
        [0.80, 0.20, 0.30, 0.75, 0.00, 0.00, 0.50, 0.80],
    ),
    (
        "pg-002", "Today's session was quieter — that happens and doesn't signal a setback.",
        "parent_guidance",
        ["reassuring", "calm", "normalising"],
        [0.80, 0.15, 0.25, 0.70, 0.00, 0.00, 0.55, 0.85],
    ),
    (
        "pg-003", "The best support right now is following their lead — no pressure.",
        "parent_guidance",
        ["practical", "calm", "child-led"],
        [0.75, 0.15, 0.35, 0.65, 0.00, 0.00, 0.60, 0.80],
    ),
    (
        "pg-004", "Consistent short sessions matter more than longer ones right now.",
        "parent_guidance",
        ["practical", "actionable", "calm"],
        [0.70, 0.20, 0.50, 0.60, 0.00, 0.00, 0.60, 0.75],
    ),
    (
        "pg-005", "Your presence during practice is the most important variable.",
        "parent_guidance",
        ["warm", "reassuring", "practical"],
        [0.85, 0.20, 0.40, 0.80, 0.00, 0.00, 0.60, 0.75],
    ),
    (
        "pg-006", "Celebrate the attempt, not just the result — it builds confidence.",
        "parent_guidance",
        ["practical", "validating", "warm"],
        [0.80, 0.25, 0.40, 0.75, 0.00, 0.00, 0.55, 0.75],
    ),

    # ── parent_escalation ────────────────────────────────────────────────────
    # When the system needs to flag a harder moment to the parent —
    # without alarm language. Replaces "escalated", "critical", "regressed".
    (
        "pe-001", "Today was a harder session — this is a normal part of the process.",
        "parent_escalation",
        ["de-alarming", "normalising", "calm"],
        [0.75, 0.15, 0.30, 0.65, 0.00, 0.00, 0.55, 0.85],
    ),
    (
        "pe-002", "Your child needs a moment of extra support right now — you're already providing it.",
        "parent_escalation",
        ["warm", "de-alarming", "reassuring"],
        [0.85, 0.15, 0.25, 0.80, 0.00, 0.00, 0.50, 0.85],
    ),
    (
        "pe-003", "Progress isn't always linear — quieter sessions are part of the journey.",
        "parent_escalation",
        ["de-alarming", "normalising", "calm"],
        [0.75, 0.15, 0.25, 0.70, 0.00, 0.00, 0.55, 0.85],
    ),
    (
        "pe-004", "We're adjusting the approach based on today — no action needed from you.",
        "parent_escalation",
        ["reassuring", "practical", "calm"],
        [0.75, 0.20, 0.40, 0.65, 0.00, 0.00, 0.55, 0.80],
    ),
    (
        "pe-005", "This pattern is something we track carefully — you'll see a note from the team.",
        "parent_escalation",
        ["practical", "transparent", "calm"],
        [0.70, 0.20, 0.45, 0.60, 0.00, 0.00, 0.60, 0.75],
    ),
]


def _build_library() -> dict[PhraseContext, list[CandidatePhrase]]:
    library: dict[PhraseContext, list[CandidatePhrase]] = {}
    for phrase_id, text, context, tags, vec in _RAW:
        phrase = CandidatePhrase(
            phrase_id=phrase_id,
            text=text,
            context=context,
            tone_tags=tags,
            embedding=ToneEmbedding(vector=vec, source="mock", dimensions=8),
        )
        library.setdefault(context, []).append(phrase)
    return library


# Module-level singleton — built once at import time
PHRASE_LIBRARY: dict[PhraseContext, list[CandidatePhrase]] = _build_library()


def get_candidates(context: PhraseContext) -> list[CandidatePhrase]:
    """Return all candidate phrases for a given context."""
    return PHRASE_LIBRARY.get(context, [])


def get_phrase_by_id(phrase_id: str) -> CandidatePhrase | None:
    """Look up a phrase by its ID across all contexts."""
    for phrases in PHRASE_LIBRARY.values():
        for phrase in phrases:
            if phrase.phrase_id == phrase_id:
                return phrase
    return None


def all_phrase_ids() -> list[str]:
    """Return all phrase IDs across the entire library."""
    return [p.phrase_id for phrases in PHRASE_LIBRARY.values() for p in phrases]
