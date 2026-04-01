from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.data import SeedData

_MODALITY_BIAS = {
    "audio": [0.04, 0.00, 0.02, 0.03, 0.00, 0.01, 0.00, 0.01],
    "noise": [0.00, 0.01, 0.01, 0.00, 0.02, 0.00, 0.00, 0.00],
    "lip": [0.01, 0.00, 0.04, 0.01, 0.00, 0.00, 0.01, 0.00],
    "emotion": [0.03, 0.01, 0.00, 0.04, 0.00, 0.02, 0.00, 0.03],
}


def _base_vector(index: int) -> list[float]:
    seed = index + 1
    raw = [
        0.58 + ((seed * 3) % 9) * 0.03,
        0.08 + ((seed * 5) % 5) * 0.02,
        0.22 + ((seed * 7) % 7) * 0.03,
        0.42 + ((seed * 11) % 6) * 0.04,
        0.01 + ((seed * 13) % 3) * 0.01,
        0.08 + ((seed * 17) % 5) * 0.02,
        0.50 + ((seed * 19) % 6) * 0.05,
        0.60 + ((seed * 23) % 6) * 0.05,
    ]
    return [round(min(0.98, max(0.0, value)), 2) for value in raw]


def _modality_vector(base: list[float], modality: str) -> list[float]:
    bias = _MODALITY_BIAS[modality]
    return [round(min(0.99, max(0.0, v + b)), 2) for v, b in zip(base, bias)]


def main() -> int:
    targets = SeedData.target_profiles()
    output_path = Path(__file__).resolve().parent.parent / "seed_data" / "reference_vectors.json"

    rows: list[dict] = []
    for index, target in enumerate(targets):
        base = _base_vector(index)
        for modality in ("audio", "noise", "lip", "emotion"):
            rows.append(
                {
                    "reference_id": f"ref-{target.display_text}-{modality}-1",
                    "target_id": target.target_id,
                    "modality": modality,
                    "embedding": _modality_vector(base, modality),
                    "source_label": f"seed-{modality}",
                    "quality_score": round(0.82 + ((index + len(modality)) % 4) * 0.04, 2),
                    "age_band": "early-childhood",
                    "notes": [f"starter {modality} reference", f"target {target.display_text}"],
                }
            )

    output_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} reference vectors to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
