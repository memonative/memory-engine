"""Logging + per-request observability for Memonative.

Provides:
  - setup_logging(): one-line app-wide log config with a request_id slot
  - request_id contextvar so every log line during a request carries it
  - time_stage(): a context manager that emits stage=... duration_ms=...
"""
import logging
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar


_request_id: ContextVar[str] = ContextVar("request_id", default="-")


class _RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True


def setup_logging(level: str = "INFO") -> None:
    """Install one stdout handler with a request_id-aware format.

    Idempotent: calling more than once replaces handlers rather than
    stacking them, which matters under uvicorn --reload.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(request_id)s] %(name)s: %(message)s"
    ))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)


def set_request_id(value: str | None = None) -> str:
    """Set (and return) the current request id for this async context."""
    rid = value or uuid.uuid4().hex[:12]
    _request_id.set(rid)
    return rid


def get_request_id() -> str:
    return _request_id.get()


@contextmanager
def time_stage(logger: logging.Logger, stage: str):
    """Time a block and emit `stage=<name> duration_ms=<n>`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        logger.info("stage=%s duration_ms=%.1f", stage, elapsed_ms)
