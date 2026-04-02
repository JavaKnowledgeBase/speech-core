from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

from app.config import settings


audit_logger = logging.getLogger("speech_core.audit")
if not audit_logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    audit_logger.addHandler(handler)
    audit_logger.setLevel(logging.INFO)
    audit_logger.propagate = False


def anonymize_identifier(value: str | None) -> str | None:
    if not value:
        return None
    salted = f"{settings.log_identifier_salt}:{value}".encode("utf-8")
    return hashlib.sha256(salted).hexdigest()[:12]


def audit_event(event: str, **fields: Any) -> None:
    if not settings.enable_audit_logging:
        return

    payload = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **fields,
    }
    audit_logger.info(json.dumps(payload, sort_keys=True, default=str))
