"""Central configuration.

Settings come from three places, in increasing order of precedence: the defaults
declared here, ``config/settings.yaml``, and the process environment. Secrets are
only ever read from the environment so that the YAML file stays safe to commit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS_PATH = PROJECT_ROOT / "config" / "settings.yaml"


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader so the package has no hard dependency on python-dotenv."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class Paths:
    root: Path
    data_root: Path
    raw: Path
    reference: Path
    processed: Path
    streams: Path
    artifacts: Path

    def ensure(self) -> None:
        for attribute in ("data_root", "raw", "reference", "processed", "streams", "artifacts"):
            getattr(self, attribute).mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Settings:
    raw: dict[str, Any] = field(repr=False)
    paths: Paths

    # -- convenience accessors -------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        return dict(self.raw.get(name, {}))

    @property
    def horizon_days(self) -> int:
        return int(self.raw["project"]["horizon_days"])

    @property
    def database_url(self) -> str | None:
        return os.environ.get("DATABASE_URL") or None

    @property
    def sqlite_path(self) -> Path:
        return self.paths.root / self.raw["storage"]["sqlite_path"]

    @property
    def kafka_bootstrap(self) -> str:
        return os.environ.get(
            "KAFKA_BOOTSTRAP_SERVERS", self.raw["stream"]["bootstrap_servers"]
        )

    @property
    def schema_registry_path(self) -> Path:
        return self.paths.root / self.raw["stream"]["schema_registry_path"]

    @property
    def topics(self) -> dict[str, str]:
        return dict(self.raw["stream"]["topics"])

    @property
    def mlflow_uri(self) -> str:
        # The file store is in maintenance mode in current MLflow, so the local
        # default is a SQLite tracking database. Point MLFLOW_TRACKING_URI at a
        # tracking server to share runs across a team.
        return os.environ.get(
            "MLFLOW_TRACKING_URI", f"sqlite:///{self.paths.artifacts / 'mlflow.db'}"
        )

    def env(self, key: str) -> str | None:
        value = os.environ.get(key, "").strip()
        return value or None

    def as_date(self, section: str, key: str) -> date:
        return date.fromisoformat(str(self.raw[section][key]))


@lru_cache(maxsize=1)
def get_settings(settings_path: str | os.PathLike[str] | None = None) -> Settings:
    _load_dotenv(PROJECT_ROOT / ".env")
    path = Path(settings_path) if settings_path else DEFAULT_SETTINGS_PATH
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    path_cfg = raw["paths"]
    paths = Paths(
        root=PROJECT_ROOT,
        data_root=PROJECT_ROOT / path_cfg["data_root"],
        raw=PROJECT_ROOT / path_cfg["raw"],
        reference=PROJECT_ROOT / path_cfg["reference"],
        processed=PROJECT_ROOT / path_cfg["processed"],
        streams=PROJECT_ROOT / path_cfg["streams"],
        artifacts=PROJECT_ROOT / path_cfg["artifacts"],
    )
    paths.ensure()
    return Settings(raw=raw, paths=paths)
