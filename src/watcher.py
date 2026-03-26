"""FTP directory polling watcher (Phase 2).

Polls a watch directory at a configurable interval and yields new image
paths for processing.  Tracks already-seen files by path so each image
is processed exactly once per run.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from config.settings import ALLOWED_EXTENSIONS, FTP_WATCH_DIR, POLL_INTERVAL_SEC

logger = logging.getLogger(__name__)


class DirectoryWatcher:
    """Poll a directory for new image files.

    Args:
        watch_dir: Directory to monitor.  Defaults to :data:`~config.settings.FTP_WATCH_DIR`.
        poll_interval: Seconds between directory scans.  Defaults to
            :data:`~config.settings.POLL_INTERVAL_SEC`.
    """

    def __init__(
        self,
        watch_dir: Path | None = None,
        poll_interval: float | None = None,
    ) -> None:
        self._watch_dir = (watch_dir or FTP_WATCH_DIR).resolve()
        self._poll_interval = (
            poll_interval if poll_interval is not None else POLL_INTERVAL_SEC
        )
        self._seen: set[Path] = set()
        logger.info(
            "DirectoryWatcher initialised",
            extra={"watch_dir": str(self._watch_dir)},
        )

    def _scan(self) -> list[Path]:
        """Return new image files found in the watch directory since last scan.

        Returns:
            Sorted list of :class:`~pathlib.Path` objects for new images.
        """
        if not self._watch_dir.is_dir():
            logger.warning(
                "Watch directory does not exist",
                extra={"watch_dir": str(self._watch_dir)},
            )
            return []

        found: list[Path] = []
        for entry in sorted(self._watch_dir.iterdir()):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            resolved = entry.resolve()
            if resolved not in self._seen:
                self._seen.add(resolved)
                found.append(resolved)

        if found:
            logger.info("New images detected", extra={"count": len(found)})
        return found

    def poll_once(self) -> list[Path]:
        """Perform a single directory scan and return new image paths.

        Returns:
            List of new image :class:`~pathlib.Path` objects (may be empty).
        """
        return self._scan()

    def watch(self):
        """Blocking generator that yields batches of new image paths.

        Each iteration sleeps for :attr:`_poll_interval` seconds before
        scanning.  Yields an empty list during quiet periods so callers
        can emit heartbeat logs or perform other housekeeping.

        Yields:
            List of :class:`~pathlib.Path` objects for newly discovered images.
        """
        logger.info("Watcher started", extra={"interval_sec": self._poll_interval})
        while True:
            new_images = self._scan()
            yield new_images
            time.sleep(self._poll_interval)
