from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

from app.config import settings
from app.models import CommunicationProfile
from app.observability import audit_event
from app.vector_entities import (
    ChildAttemptVectorRecord,
    EnvironmentStandardProfileRecord,
    OutputFilterProfileRecord,
    ReferenceVectorRecord,
    TargetProfileRecord,
)

T = TypeVar("T")


class SupabaseTableRepository(Generic[T]):
    def __init__(self, table_name: str, factory: Callable[[dict], T], id_field: str) -> None:
        self._table_name = table_name
        self._factory = factory
        self._id_field = id_field
        self._client = None

    def enabled(self) -> bool:
        return settings.supabase_enabled

    def _get_client(self):
        if not self.enabled():
            return None
        if self._client is None:
            try:
                from supabase import create_client
            except ImportError:
                return None
            try:
                self._client = create_client(settings.supabase_url, settings.supabase_key)
            except Exception as exc:  # noqa: BLE001
                audit_event(
                    "repository_remote_unavailable",
                    table=self._table_name,
                    operation="create_client",
                    error_type=type(exc).__name__,
                )
                return None
        return self._client

    def upsert(self, payload: dict) -> None:
        client = self._get_client()
        if client is None:
            return
        try:
            client.table(self._table_name).upsert(payload).execute()
        except Exception as exc:  # noqa: BLE001
            audit_event(
                "repository_remote_failed",
                table=self._table_name,
                operation="upsert",
                error_type=type(exc).__name__,
            )

    def select_all(self) -> list[T]:
        client = self._get_client()
        if client is None:
            return []
        try:
            rows = client.table(self._table_name).select("*").execute().data or []
        except Exception as exc:  # noqa: BLE001
            audit_event(
                "repository_remote_failed",
                table=self._table_name,
                operation="select_all",
                error_type=type(exc).__name__,
            )
            return []
        return [self._factory(row) for row in rows]

    def select_one(self, field: str, value: str) -> T | None:
        client = self._get_client()
        if client is None:
            return None
        try:
            rows = client.table(self._table_name).select("*").eq(field, value).limit(1).execute().data or []
        except Exception as exc:  # noqa: BLE001
            audit_event(
                "repository_remote_failed",
                table=self._table_name,
                operation="select_one",
                field=field,
                error_type=type(exc).__name__,
            )
            return None
        return self._factory(rows[0]) if rows else None

    def delete(self, value: str) -> bool:
        client = self._get_client()
        if client is None:
            return False
        try:
            client.table(self._table_name).delete().eq(self._id_field, value).execute()
        except Exception as exc:  # noqa: BLE001
            audit_event(
                "repository_remote_failed",
                table=self._table_name,
                operation="delete",
                error_type=type(exc).__name__,
            )
            return False
        return True


class CommunicationProfileRepository:
    def __init__(self) -> None:
        self._memory: dict[str, CommunicationProfile] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_profiles_table,
            factory=lambda row: CommunicationProfile.model_validate(row),
            id_field="profile_id",
        )

    def seed(self, profiles: list[CommunicationProfile]) -> None:
        for profile in profiles:
            self._memory[profile.profile_id] = profile

    def get(self, profile_id: str) -> CommunicationProfile | None:
        return self._remote.select_one("profile_id", profile_id) or self._memory.get(profile_id)

    def get_by_owner(self, owner_id: str) -> CommunicationProfile | None:
        remote = self._remote.select_one("owner_id", owner_id)
        if remote is not None:
            return remote
        for profile in self._memory.values():
            if profile.owner_id == owner_id:
                return profile
        return None

    def upsert(self, profile: CommunicationProfile) -> None:
        self._memory[profile.profile_id] = profile
        self._remote.upsert(profile.model_dump(mode="json"))

    def delete(self, profile_id: str) -> bool:
        deleted = profile_id in self._memory
        self._memory.pop(profile_id, None)
        remote_deleted = self._remote.delete(profile_id)
        return deleted or remote_deleted

    def list_all(self) -> list[CommunicationProfile]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())

    def list_by_audience(self, audience: str) -> list[CommunicationProfile]:
        return [profile for profile in self.list_all() if profile.audience == audience]


class TargetProfileRepository:
    def __init__(self) -> None:
        self._memory: dict[str, TargetProfileRecord] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_target_profiles_table,
            factory=lambda row: TargetProfileRecord.model_validate(row),
            id_field="target_id",
        )

    def seed(self, items: list[TargetProfileRecord]) -> None:
        for item in items:
            self._memory[item.target_id] = item

    def upsert(self, item: TargetProfileRecord) -> None:
        self._memory[item.target_id] = item
        self._remote.upsert(item.model_dump(mode="json"))

    def get(self, target_id: str) -> TargetProfileRecord | None:
        return self._remote.select_one("target_id", target_id) or self._memory.get(target_id)

    def list_all(self) -> list[TargetProfileRecord]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())


class ReferenceVectorRepository:
    def __init__(self) -> None:
        self._memory: dict[str, ReferenceVectorRecord] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_reference_vectors_table,
            factory=lambda row: ReferenceVectorRecord.model_validate(row),
            id_field="reference_id",
        )

    def seed(self, items: list[ReferenceVectorRecord]) -> None:
        for item in items:
            self._memory[item.reference_id] = item

    def upsert(self, item: ReferenceVectorRecord) -> None:
        self._memory[item.reference_id] = item
        self._remote.upsert(item.model_dump(mode="json"))

    def get(self, reference_id: str) -> ReferenceVectorRecord | None:
        return self._remote.select_one("reference_id", reference_id) or self._memory.get(reference_id)

    def list_all(self) -> list[ReferenceVectorRecord]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())

    def list_by_target(self, target_id: str) -> list[ReferenceVectorRecord]:
        return [item for item in self.list_all() if item.target_id == target_id]


class ChildAttemptRepository:
    def __init__(self) -> None:
        self._memory: dict[str, ChildAttemptVectorRecord] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_child_attempts_table,
            factory=lambda row: ChildAttemptVectorRecord.model_validate(row),
            id_field="attempt_id",
        )

    def seed(self, items: list[ChildAttemptVectorRecord]) -> None:
        for item in items:
            self._memory[item.attempt_id] = item

    def upsert(self, item: ChildAttemptVectorRecord) -> None:
        self._memory[item.attempt_id] = item
        self._remote.upsert(item.model_dump(mode="json"))

    def get(self, attempt_id: str) -> ChildAttemptVectorRecord | None:
        return self._remote.select_one("attempt_id", attempt_id) or self._memory.get(attempt_id)

    def list_all(self) -> list[ChildAttemptVectorRecord]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())

    def list_by_child(self, child_id: str) -> list[ChildAttemptVectorRecord]:
        return [item for item in self.list_all() if item.child_id == child_id]


class OutputFilterProfileRepository:
    def __init__(self) -> None:
        self._memory: dict[str, OutputFilterProfileRecord] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_output_filter_profiles_table,
            factory=lambda row: OutputFilterProfileRecord.model_validate(row),
            id_field="profile_id",
        )

    def seed(self, items: list[OutputFilterProfileRecord]) -> None:
        for item in items:
            self._memory[item.profile_id] = item

    def upsert(self, item: OutputFilterProfileRecord) -> None:
        self._memory[item.profile_id] = item
        self._remote.upsert(item.model_dump(mode="json"))

    def get(self, profile_id: str) -> OutputFilterProfileRecord | None:
        return self._remote.select_one("profile_id", profile_id) or self._memory.get(profile_id)

    def get_by_child(self, child_id: str) -> OutputFilterProfileRecord | None:
        return self._remote.select_one("child_id", child_id) or next((item for item in self._memory.values() if item.child_id == child_id), None)

    def list_all(self) -> list[OutputFilterProfileRecord]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())


class EnvironmentStandardRepository:
    def __init__(self) -> None:
        self._memory: dict[str, EnvironmentStandardProfileRecord] = {}
        self._remote = SupabaseTableRepository(
            settings.supabase_environment_profiles_table,
            factory=lambda row: EnvironmentStandardProfileRecord.model_validate(row),
            id_field="environment_profile_id",
        )

    def seed(self, items: list[EnvironmentStandardProfileRecord]) -> None:
        for item in items:
            self._memory[item.environment_profile_id] = item

    def upsert(self, item: EnvironmentStandardProfileRecord) -> None:
        self._memory[item.environment_profile_id] = item
        self._remote.upsert(item.model_dump(mode="json"))

    def get(self, environment_profile_id: str) -> EnvironmentStandardProfileRecord | None:
        return self._remote.select_one("environment_profile_id", environment_profile_id) or self._memory.get(environment_profile_id)

    def get_by_child(self, child_id: str) -> EnvironmentStandardProfileRecord | None:
        return self._remote.select_one("child_id", child_id) or next((item for item in self._memory.values() if item.child_id == child_id), None)

    def list_all(self) -> list[EnvironmentStandardProfileRecord]:
        remote_rows = self._remote.select_all()
        if remote_rows:
            return remote_rows
        return list(self._memory.values())
