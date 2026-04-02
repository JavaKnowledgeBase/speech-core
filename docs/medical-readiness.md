# Medical Readiness

## Scope boundary

This service is intended to support child speech-therapy workflows by filtering output, retrieving reference vectors, and shaping caregiver guidance. It should be positioned as support software, not diagnosis software, not autonomous treatment planning, and not an emergency-response system.

## Technical controls now in repo

- production auth gate via `SERVICE_API_KEY`
- production fail-fast checks for missing auth or missing persistent storage configuration
- structured audit logging without raw child/caregiver text
- identifier hashing for child/owner references in logs
- safer HTTP response headers and `Cache-Control: no-store`
- timezone-aware UTC timestamps for stored records

## Deployment checklist

- set `APP_ENV=production`
- set `SERVICE_API_KEY` and keep it in secret storage
- set `SUPABASE_URL` and `SUPABASE_KEY`
- keep `ALLOW_OPENAPI_IN_PRODUCTION=false` unless docs exposure is explicitly approved
- terminate TLS upstream and restrict service ingress to approved internal callers
- define retention rules for logs, vectors, session data, and caregiver guidance
- review business associate agreement requirements if a covered entity is involved
- implement parental consent and privacy notice flow in the parent-facing product layer

## Still required outside code

- legal/privacy review for HIPAA and state-law applicability
- COPPA notice and verifiable parental consent process
- role-based access control for end-user applications that call this service
- incident response, backup, recovery, and monitoring procedures
- clinical validation of production datasets and prompts
- FDA/regulatory review if product claims move toward diagnosis, treatment recommendation, or device-software scope

## Source anchors

Primary official references used for this standards pass:
- HHS HIPAA Security Rule guidance
- FTC COPPA compliance guidance and verifiable parental consent resources
- FDA digital health / device-software-function guidance
- FDA clinical decision support software guidance
