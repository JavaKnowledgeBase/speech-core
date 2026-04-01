from __future__ import annotations

import csv
import json
from pathlib import Path

from app.vector_entities import ReferenceVectorRecord, TargetProfileRecord


def _load_json_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _parse_embedding(value: str | list[float]) -> list[float]:
    if isinstance(value, list):
        return [float(item) for item in value]

    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        return [float(item) for item in parsed]
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def _parse_notes(value: str | list[str]) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    text = value.strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        return [str(item) for item in parsed]
    return [item.strip() for item in text.split("|") if item.strip()]


def load_target_profiles(path: Path) -> list[TargetProfileRecord]:
    if path.suffix.lower() == ".json":
        return [TargetProfileRecord.model_validate(item) for item in _load_json_records(path)]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            return [TargetProfileRecord.model_validate(row) for row in reader]
    raise ValueError(f"Unsupported target file format: {path.suffix}")


def load_reference_vectors(path: Path) -> list[ReferenceVectorRecord]:
    if path.suffix.lower() == ".json":
        rows = _load_json_records(path)
        return [ReferenceVectorRecord.model_validate(item) for item in rows]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            rows: list[ReferenceVectorRecord] = []
            for row in reader:
                payload = dict(row)
                payload["embedding"] = _parse_embedding(payload.get("embedding", ""))
                payload["notes"] = _parse_notes(payload.get("notes", ""))
                rows.append(ReferenceVectorRecord.model_validate(payload))
            return rows
    raise ValueError(f"Unsupported reference vector file format: {path.suffix}")


def export_reference_vectors_csv(rows: list[ReferenceVectorRecord], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "reference_id",
                "target_id",
                "modality",
                "embedding",
                "source_label",
                "quality_score",
                "age_band",
                "notes",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "reference_id": row.reference_id,
                    "target_id": row.target_id,
                    "modality": row.modality,
                    "embedding": json.dumps(row.embedding),
                    "source_label": row.source_label,
                    "quality_score": row.quality_score,
                    "age_band": row.age_band,
                    "notes": "|".join(row.notes),
                }
            )
