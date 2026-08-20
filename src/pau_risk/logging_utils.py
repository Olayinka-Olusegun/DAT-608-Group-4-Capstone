"""Uniform logging for every entry point in the package."""

from __future__ import annotations

import logging
import os


def configure(level: str | None = None) -> None:
    resolved = (level or os.environ.get("LOG_LEVEL", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, resolved, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("py4j", "urllib3", "botocore", "matplotlib", "httpx", "pyspark"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # A refused broker connection is the expected path when Kafka is not running
    # locally, and the bus reports the fallback itself, so the driver's own retry
    # noise is suppressed rather than presented as a failure.
    logging.getLogger("kafka").setLevel(logging.CRITICAL)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
