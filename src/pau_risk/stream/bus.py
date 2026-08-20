"""Avro transport over Kafka, with a file sink that preserves the same semantics.

Every connector publishes through :class:`EventBus`. When a broker is reachable
the payloads go to Kafka encoded in the Confluent wire format: a magic byte, the
schema identifier, then the Avro binary body. When no broker is reachable the
identical encoded payloads are appended to Avro object container files, one per
topic and ingestion date, and :meth:`EventBus.replay` reads them back in order.

Keeping one interface over both paths means the ingestion, validation and
serialisation logic is exercised in full even without a running cluster, and
moving to a real cluster is a configuration change rather than a rewrite.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterator, Sequence

import fastavro

from ..config import Settings, get_settings
from ..logging_utils import get_logger
from .registry import RegisteredSchema, SchemaRegistry, load_schema_file

LOGGER = get_logger(__name__)
MAGIC_BYTE = b"\x00"


def _coerce_for_avro(record: dict[str, Any], schema: dict) -> dict[str, Any]:
    """Align a plain dictionary with the declared Avro types."""
    typed: dict[str, Any] = {}
    for field_schema in schema["fields"]:
        name = field_schema["name"]
        value = record.get(name, field_schema.get("default"))
        declared = field_schema["type"]
        is_date = (
            isinstance(declared, dict) and declared.get("logicalType") == "date"
        ) or (
            isinstance(declared, list)
            and any(
                isinstance(option, dict) and option.get("logicalType") == "date"
                for option in declared
            )
        )
        if is_date and isinstance(value, str) and value:
            value = date.fromisoformat(value[:10])
        elif is_date and isinstance(value, datetime):
            value = value.date()
        if isinstance(value, datetime):
            value = value.isoformat(timespec="seconds")
        typed[name] = value
    return typed


@dataclass
class Published:
    topic: str
    count: int
    transport: str
    path: Path | None = None


@dataclass
class EventBus:
    settings: Settings = field(default_factory=get_settings)
    _registry: SchemaRegistry = field(init=False)
    _producer: Any = field(init=False, default=None)
    _transport: str = field(init=False, default="file")

    def __post_init__(self) -> None:
        self._registry = SchemaRegistry(self.settings)
        for kind in ("incident", "document", "chatter"):
            self._registry.register(f"nga.{kind}.v1-value", load_schema_file(kind))
        self._producer = self._connect()
        self._transport = "kafka" if self._producer is not None else "file"

    # ------------------------------------------------------------- transport
    def _connect(self) -> Any:
        servers = self.settings.kafka_bootstrap
        try:
            from kafka import KafkaProducer  # type: ignore[import-not-found]

            producer = KafkaProducer(
                bootstrap_servers=servers.split(","),
                api_version_auto_timeout_ms=3000,
                request_timeout_ms=5000,
                max_block_ms=5000,
                acks="all",
            )
            LOGGER.info("kafka transport active on %s", servers)
            return producer
        except Exception as exc:  # noqa: BLE001 - absence of a broker is expected offline
            LOGGER.info(
                "kafka unavailable (%s), using Avro file sink at %s",
                exc.__class__.__name__,
                self.settings.paths.streams,
            )
            return None

    @property
    def transport(self) -> str:
        return self._transport

    def subject_for(self, kind: str) -> RegisteredSchema:
        return self._registry.latest(f"nga.{kind}.v1-value")

    # --------------------------------------------------------------- publish
    def encode(self, kind: str, record: dict[str, Any]) -> bytes:
        registered = self.subject_for(kind)
        typed = _coerce_for_avro(record, registered.schema)
        buffer = io.BytesIO()
        fastavro.schemaless_writer(buffer, registered.schema, typed)
        return MAGIC_BYTE + registered.schema_id.encode("ascii") + buffer.getvalue()

    def decode(self, kind: str, payload: bytes) -> dict[str, Any]:
        if payload[:1] != MAGIC_BYTE:
            raise ValueError("payload is not in the expected wire format")
        schema_id = payload[1:17].decode("ascii")
        registered = self._registry.by_id(f"nga.{kind}.v1-value", schema_id)
        return fastavro.schemaless_reader(io.BytesIO(payload[17:]), registered.schema)

    def publish(
        self, kind: str, records: Sequence[dict[str, Any]], producer_name: str
    ) -> Published:
        if not records:
            return Published(topic=self.settings.topics[kind], count=0, transport=self._transport)

        topic = self.settings.topics[kind]
        registered = self.subject_for(kind)
        typed = [_coerce_for_avro(record, registered.schema) for record in records]
        # Validation happens once, before anything leaves the process, so a
        # malformed connector fails at its own boundary rather than downstream.
        for row in typed:
            if not fastavro.validate(row, registered.schema, raise_errors=False):
                fastavro.validate(row, registered.schema, raise_errors=True)

        if self._producer is not None:
            for row in typed:
                buffer = io.BytesIO()
                fastavro.schemaless_writer(buffer, registered.schema, row)
                self._producer.send(
                    topic,
                    key=producer_name.encode("utf-8"),
                    value=MAGIC_BYTE + registered.schema_id.encode("ascii") + buffer.getvalue(),
                )
            self._producer.flush()
            LOGGER.info("published %d %s records to %s", len(typed), kind, topic)
            return Published(topic=topic, count=len(typed), transport="kafka")

        path = self._sink_path(topic, producer_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a+b" if path.exists() else "wb"
        with path.open(mode) as handle:
            writer = fastavro.writer if mode == "wb" else fastavro.writer
            writer(handle, registered.schema, typed, codec="deflate")
        LOGGER.info("wrote %d %s records to %s", len(typed), kind, path.name)
        return Published(topic=topic, count=len(typed), transport="file", path=path)

    def _sink_path(self, topic: str, producer_name: str) -> Path:
        stamp = date.today().isoformat()
        return self.settings.paths.streams / topic / f"{producer_name}-{stamp}.avro"

    # ---------------------------------------------------------------- replay
    def replay(self, kind: str) -> Iterator[dict[str, Any]]:
        """Read every record on a topic, from Kafka or from the file sink."""
        topic = self.settings.topics[kind]
        if self._producer is not None:
            yield from self._replay_kafka(topic, kind)
            return
        directory = self.settings.paths.streams / topic
        if not directory.exists():
            return
        for path in sorted(directory.glob("*.avro")):
            with path.open("rb") as handle:
                for record in fastavro.reader(handle):
                    yield dict(record)

    def _replay_kafka(self, topic: str, kind: str) -> Iterator[dict[str, Any]]:
        from kafka import KafkaConsumer  # type: ignore[import-not-found]

        consumer = KafkaConsumer(
            topic,
            bootstrap_servers=self.settings.kafka_bootstrap.split(","),
            auto_offset_reset="earliest",
            enable_auto_commit=False,
            consumer_timeout_ms=8000,
            group_id="pau-risk-feature-pipeline",
        )
        for message in consumer:
            yield self.decode(kind, message.value)
        consumer.close()

    def close(self) -> None:
        if self._producer is not None:
            self._producer.close(timeout=5)


def describe_topics(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return json.dumps(settings.topics, indent=2)
