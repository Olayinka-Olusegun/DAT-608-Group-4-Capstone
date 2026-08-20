"""A file-backed Avro schema registry.

The Confluent registry is a network service holding one immutable version chain
per subject and refusing writes that break compatibility. Standing up that
service is not possible inside this assessment environment, so the same contract
is implemented locally: subjects are versioned, a schema is registered only once,
and a producer that changes a schema in a backward-incompatible way is rejected
before it can publish. Swapping in the hosted registry later means replacing this
class, not the producers, because they only ever ask for a schema id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..config import Settings, get_settings
from ..logging_utils import get_logger

LOGGER = get_logger(__name__)
SCHEMA_DIR = Path(__file__).resolve().parents[1] / "schemas"


class IncompatibleSchemaError(RuntimeError):
    """Raised when a new schema version would break existing consumers."""


@dataclass(frozen=True)
class RegisteredSchema:
    subject: str
    version: int
    schema_id: str
    schema: dict


def load_schema_file(name: str) -> dict:
    return json.loads((SCHEMA_DIR / f"{name}.avsc").read_text(encoding="utf-8"))


def _fingerprint(schema: dict) -> str:
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _field_map(schema: dict) -> dict[str, dict]:
    return {field["name"]: field for field in schema.get("fields", [])}


def check_backward_compatible(previous: dict, candidate: dict) -> None:
    """A consumer on the previous schema must still be able to read new records.

    That holds when no field is removed and every added field carries a default.
    """
    old_fields, new_fields = _field_map(previous), _field_map(candidate)
    removed = set(old_fields) - set(new_fields)
    if removed:
        raise IncompatibleSchemaError(f"fields removed: {sorted(removed)}")
    for name in set(new_fields) - set(old_fields):
        if "default" not in new_fields[name]:
            raise IncompatibleSchemaError(f"added field {name!r} has no default")
    for name in set(new_fields) & set(old_fields):
        if json.dumps(old_fields[name]["type"], sort_keys=True) != json.dumps(
            new_fields[name]["type"], sort_keys=True
        ):
            raise IncompatibleSchemaError(f"type of field {name!r} changed")


class SchemaRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._path = self._settings.schema_registry_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._state: dict[str, list[dict]] = (
            json.loads(self._path.read_text(encoding="utf-8")) if self._path.exists() else {}
        )

    def _persist(self) -> None:
        self._path.write_text(json.dumps(self._state, indent=2), encoding="utf-8")

    def register(self, subject: str, schema: dict) -> RegisteredSchema:
        versions = self._state.setdefault(subject, [])
        schema_id = _fingerprint(schema)
        for entry in versions:
            if entry["schema_id"] == schema_id:
                return RegisteredSchema(subject, entry["version"], schema_id, schema)
        if versions:
            check_backward_compatible(versions[-1]["schema"], schema)
        entry = {
            "version": len(versions) + 1,
            "schema_id": schema_id,
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "schema": schema,
        }
        versions.append(entry)
        self._persist()
        LOGGER.info("registered subject=%s version=%d id=%s", subject, entry["version"], schema_id)
        return RegisteredSchema(subject, entry["version"], schema_id, schema)

    def latest(self, subject: str) -> RegisteredSchema:
        versions = self._state.get(subject)
        if not versions:
            raise KeyError(f"subject {subject!r} is not registered")
        entry = versions[-1]
        return RegisteredSchema(subject, entry["version"], entry["schema_id"], entry["schema"])

    def by_id(self, subject: str, schema_id: str) -> RegisteredSchema:
        for entry in self._state.get(subject, []):
            if entry["schema_id"] == schema_id:
                return RegisteredSchema(subject, entry["version"], schema_id, entry["schema"])
        raise KeyError(f"schema id {schema_id!r} not found under {subject!r}")

    def subjects(self) -> list[str]:
        return sorted(self._state)


def bootstrap(settings: Settings | None = None) -> dict[str, RegisteredSchema]:
    """Register the three record contracts the nine producers share."""
    registry = SchemaRegistry(settings)
    return {
        kind: registry.register(f"nga.{kind}.v1-value", load_schema_file(kind))
        for kind in ("incident", "document", "chatter")
    }
