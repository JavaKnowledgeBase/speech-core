# speech-core

Speech-first filtering and vector-retrieval service for TalkBuddy-style therapy workflows.

## Current state

This repo now includes:
- output filtering for child and parent responses
- output-kind policy matrix and environment-aware filtering
- Supabase/pgvector schema scaffold
- repository layer with safe in-memory fallback
- starter target/reference seed data
- retrieval and attempt-ingestion flow for stored reference vectors
- CSV and JSON import support for target profiles and reference vectors
- production-safety controls for authenticated internal access, audit logging, and safer HTTP headers
- Docker packaging for local and deployment-style runs

## Medical and privacy boundary

This service should be treated as clinical-support infrastructure for a child speech workflow, not as diagnostic or autonomous treatment software.

Production expectations:
- keep this service behind authenticated internal traffic
- avoid logging raw child or caregiver text when audit logs are enabled
- persist production data in Supabase or another approved store instead of memory fallback
- obtain the required parental consent and privacy notices before collecting child data
- review deployment with legal/compliance owners for HIPAA/COPPA scope and BAAs where applicable

This codebase does not by itself make an organization HIPAA-, COPPA-, or FDA-compliant. It adds technical guardrails that support those programs.

## Key files

- `Dockerfile`: production-oriented container image for the API
- `docker-compose.yml`: local/deployment-style container run config
- `sql/supabase_schema.sql`: first-pass Supabase + pgvector schema
- `seed_data/target_profiles.json`: starter month-one targets
- `seed_data/reference_vectors.json`: generated starter references for all 20 targets x 4 modalities
- `seed_data/reference_vectors.csv`: CSV export of the full starter reference set
- `seed_data/reference_vectors.sample.csv`: tiny hand-editable sample for real data contributors
- `scripts/generate_reference_seed.py`: regenerates the scaffold reference dataset
- `scripts/seed_supabase.py`: dry-run, import, export, and Supabase seed script
- `app/observability.py`: structured audit logging with identifier hashing
- `docs/medical-readiness.md`: deployment and standards-alignment checklist
- `docs/speech-filters-postman-collection.json`: Postman collection for local testing against port `18100`

## Docker

Build the image:

```powershell
docker build -t speech-filters-core .
```

Run the container directly:

```powershell
docker run --rm -p 18100:8000 `
  -e APP_ENV=production `
  -e SERVICE_API_KEY=replace-me `
  -e SUPABASE_URL=https://your-project.supabase.co `
  -e SUPABASE_KEY=replace-me `
  speech-filters-core
```

Run with Compose:

```powershell
docker compose up --build
```

Important container notes:
- Compose publishes the service on host port `18100` to avoid conflicts with apps already using `8000`, `8080`, `5173`, `5180`, `5432`, `6379`, `8765`, and `18080`
- this repo uses the unique image name `speech-filters-core:latest` to avoid collisions with other local `speech-core` images
- the container defaults to `APP_ENV=production`
- non-health endpoints require `SERVICE_API_KEY` in production
- `SUPABASE_URL` and `SUPABASE_KEY` are required for production startup

## Reference vector CSV format

Columns:
- `reference_id`: unique row id
- `target_id`: must match a target profile id like `target-a`
- `modality`: one of `audio`, `noise`, `lip`, `emotion`
- `embedding`: JSON array string or comma-separated vector values
- `source_label`: source batch or collection name
- `quality_score`: `0.0` to `1.0`
- `age_band`: free text such as `early-childhood`
- `notes`: pipe-separated notes such as `quiet room|front camera`

Example row:

```csv
ref-sample-a-audio-1,target-a,audio,"[0.71, 0.08, 0.24, 0.65, 0.02, 0.13, 0.55, 0.86]",clinic-batch-1,0.95,early-childhood,clear recording|quiet room
```

## Common commands

Generate the scaffold starter references:

```powershell
python scripts\generate_reference_seed.py
```

Preview what would be seeded:

```powershell
python scripts\seed_supabase.py --dry-run
```

Export the currently loaded reference set to CSV:

```powershell
python scripts\seed_supabase.py --dry-run --export-references-csv seed_data\reference_vectors.csv
```

Seed custom files:

```powershell
python scripts\seed_supabase.py --targets-file your_targets.json --references-file your_vectors.csv
```

Seed Supabase after setting `SUPABASE_URL` and `SUPABASE_KEY`:

```powershell
python scripts\seed_supabase.py
```

## Notes

The current 80 reference vectors are scaffold/generated starter data meant to exercise the schema, retrieval flow, and import pipeline. They are not a substitute for clinically collected or production embedding data.
