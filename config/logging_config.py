"""Structured JSON logging configuration."""

import logging
import logging.handlers
import json
from pathlib import Path
from typing import Any

from config.settings import LOG_LEVEL


class _JsonFormatter(logging.Formatter):
    """Emit log records as single-line JSON; redacts sensitive fields."""

    _REDACTED = "***REDACTED***"
    _SENSITIVE_KEYS = frozenset(
        {"password", "passwd", "secret", "token", "credential", "ftp_pass"}
    )

    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Merge any extra fields, redacting sensitive keys
        for key, value in record.__dict__.items():
            if key in (
                "msg",
                "args",
                "exc_info",
                "exc_text",
                "stack_info",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "name",
                "lineno",
                "taskName",
            ):
                continue
            if key.lower() in self._SENSITIVE_KEYS:
                payload[key] = self._REDACTED
            elif not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str)


def configure_logging(log_dir: Path | None = None) -> None:
    """Set up root logger with JSON formatting and optional file rotation.

    Args:
        log_dir: If provided, also write rotating logs to this directory.
    """
    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    # Remove any existing handlers (e.g., from pytest capture)
    root.handlers.clear()

    formatter = _JsonFormatter()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Rotating file handler
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            log_dir / "ppe_detection.log",
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
