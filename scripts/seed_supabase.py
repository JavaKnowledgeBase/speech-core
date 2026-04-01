from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.data import (
    child_attempt_repository,
    environment_standard_repository,
    output_filter_profile_repository,
    profile_store,
    reference_vector_repository,
    target_profile_repository,
)
from app.importers import export_reference_vectors_csv, load_reference_vectors, load_target_profiles
from app.vector_entities import ReferenceVectorRecord, TargetProfileRecord


@dataclass
class SeedSummary:
    name: str
    count: int


def _seed_profiles() -> SeedSummary:
    items = profile_store.list_all()
    for item in items:
        profile_store.upsert(item)
    return SeedSummary("communication_profiles", len(items))


def _seed_targets(items: list[TargetProfileRecord] | None = None) -> SeedSummary:
    records = items or target_profile_repository.list_all()
    for item in records:
        target_profile_repository.upsert(item)
    return SeedSummary("target_profiles", len(records))


def _seed_reference_vectors(items: list[ReferenceVectorRecord] | None = None) -> SeedSummary:
    records = items or reference_vector_repository.list_all()
    for item in records:
        reference_vector_repository.upsert(item)
    return SeedSummary("reference_vectors", len(records))


def _seed_output_filter_profiles() -> SeedSummary:
    items = output_filter_profile_repository.list_all()
    for item in items:
        output_filter_profile_repository.upsert(item)
    return SeedSummary("output_filter_profiles", len(items))


def _seed_environment_profiles() -> SeedSummary:
    items = environment_standard_repository.list_all()
    for item in items:
        environment_standard_repository.upsert(item)
    return SeedSummary("environment_standard_profiles", len(items))


def _seed_child_attempts() -> SeedSummary:
    items = child_attempt_repository.list_all()
    for item in items:
        child_attempt_repository.upsert(item)
    return SeedSummary("child_attempt_vectors", len(items))


def seed_all(
    targets: list[TargetProfileRecord] | None = None,
    reference_vectors: list[ReferenceVectorRecord] | None = None,
) -> list[SeedSummary]:
    return [
        _seed_profiles(),
        _seed_targets(targets),
        _seed_reference_vectors(reference_vectors),
        _seed_output_filter_profiles(),
        _seed_environment_profiles(),
        _seed_child_attempts(),
    ]


def _write_plan_csv(path: Path, summaries: list[SeedSummary]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["dataset", "count"])
        for summary in summaries:
            writer.writerow([summary.name, summary.count])


def main() -> int:
    parser = argparse.ArgumentParser(description="Seed speech-core starter data into Supabase.")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be seeded without writing to Supabase.")
    parser.add_argument("--targets-file", type=Path, help="Optional JSON or CSV file for target profiles.")
    parser.add_argument("--references-file", type=Path, help="Optional JSON or CSV file for reference vectors.")
    parser.add_argument("--plan-csv", type=Path, help="Optional CSV output path for the seed plan.")
    parser.add_argument("--export-references-csv", type=Path, help="Optional CSV export path for the currently loaded reference vectors.")
    args = parser.parse_args()

    target_items = load_target_profiles(args.targets_file) if args.targets_file else target_profile_repository.list_all()
    reference_items = load_reference_vectors(args.references_file) if args.references_file else reference_vector_repository.list_all()

    summaries = [
        SeedSummary("communication_profiles", len(profile_store.list_all())),
        SeedSummary("target_profiles", len(target_items)),
        SeedSummary("reference_vectors", len(reference_items)),
        SeedSummary("output_filter_profiles", len(output_filter_profile_repository.list_all())),
        SeedSummary("environment_standard_profiles", len(environment_standard_repository.list_all())),
        SeedSummary("child_attempt_vectors", len(child_attempt_repository.list_all())),
    ]

    print("Seed plan:")
    for summary in summaries:
        print(f"- {summary.name}: {summary.count}")

    if args.plan_csv:
        _write_plan_csv(args.plan_csv, summaries)
        print(f"Plan CSV written: {args.plan_csv}")

    if args.export_references_csv:
        export_reference_vectors_csv(reference_items, args.export_references_csv)
        print(f"Reference CSV exported: {args.export_references_csv}")

    if args.dry_run:
        print("Dry run only. No data written.")
        return 0

    if not settings.supabase_enabled:
        print("Supabase is not configured. Set SUPABASE_URL and SUPABASE_KEY first.")
        return 1

    results = seed_all(targets=target_items, reference_vectors=reference_items)
    print("Seed complete:")
    for result in results:
        print(f"- {result.name}: {result.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
