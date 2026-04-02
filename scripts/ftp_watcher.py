"""FTP server-side watcher script.

Runs ON the FTP server. Polls a local watch directory for new images,
POSTs each to the remote PPE detection API (/detect/full), saves the
annotated image to {top_level_dir}_ppestatus/, and logs the compliance
JSON metadata.

Directory convention:
    Input:  /IHS-LAG-1197A/2026/3/15/IHS-LAG-1197A__00__213131313113.jpg
    Output: /IHS-LAG-1197A_ppestatus/2026/3/15/IHS-LAG-1197A__00__213131313113.jpg

API response format (from /detect/full):
    {
        "Status": "Safe" | "Unsafe: No Helmet" | ...,
        "Alarm": true | false,
        "PPE-missing": ["Hard Hat", "Safety Vest"],
        "persons_detected": int,
        "safe": int,
        "unsafe": int,
        "image_base64": "<base64 JPEG>"
    }

Usage:
    python ftp_watcher.py                          # continuous polling
    python ftp_watcher.py --once                   # single scan, then exit
    python ftp_watcher.py --watch-dir /path/to/IHS-LAG-1197A
    python ftp_watcher.py --api-url http://1.2.3.4:5000/detect/full

Requirements (install on the FTP server):
    pip install requests
"""

from __future__ import annotations

import argparse
import base64
import logging
import os
import signal
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Configuration (env vars with CLI overrides)
# ---------------------------------------------------------------------------
WATCH_DIR = os.environ.get("PPE_WATCH_DIR", "/IHS-LAG-1197A")
API_URL = os.environ.get("PPE_API_URL", "http://localhost:5000/detect/full")
POLL_INTERVAL = float(os.environ.get("PPE_POLL_INTERVAL", "5"))
REQUEST_TIMEOUT = int(os.environ.get("PPE_REQUEST_TIMEOUT", "60"))
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("ftp_watcher")


# ---------------------------------------------------------------------------
# Output path logic
# ---------------------------------------------------------------------------

def build_output_path(image_path: Path, watch_dir: Path) -> Path:
    """Build the _ppestatus output path for a given image.

    The top-level directory (watch_dir itself) gets ``_ppestatus`` appended.
    Everything below it (year/month/day subdirs and filename) is preserved.

    Examples:
        watch_dir = /IHS-LAG-1197A
        image     = /IHS-LAG-1197A/2026/3/15/img.jpg
        output    = /IHS-LAG-1197A_ppestatus/2026/3/15/img.jpg
    """
    rel = image_path.relative_to(watch_dir)
    output_root = watch_dir.parent / (watch_dir.name + "_ppestatus")
    return output_root / rel


# ---------------------------------------------------------------------------
# Core watcher
# ---------------------------------------------------------------------------

class FTPWatcher:
    """Polls a local directory, sends images to the PPE API, saves results."""

    def __init__(
        self,
        watch_dir: str | Path,
        api_url: str,
        poll_interval: float = 5.0,
        request_timeout: int = 60,
    ) -> None:
        self._watch_dir = Path(watch_dir).resolve()
        self._api_url = api_url
        self._poll_interval = poll_interval
        self._request_timeout = request_timeout
        self._seen: set[Path] = set()
        self._running = True

        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum: int, frame) -> None:
        logger.info("Received signal %d, shutting down…", signum)
        self._running = False

    def _init_seen_from_existing_outputs(self) -> None:
        """Mark images that already have a _ppestatus output as seen.

        This makes restarts idempotent — we don't reprocess images whose
        annotated output already exists.
        """
        count = 0
        for img_path in self._iter_images():
            out_path = build_output_path(img_path, self._watch_dir)
            if out_path.exists():
                self._seen.add(img_path)
                count += 1
        if count:
            logger.info("Skipping %d already-processed images", count)

    def _iter_images(self) -> list[Path]:
        """Recursively find all image files under the watch directory."""
        if not self._watch_dir.is_dir():
            logger.warning("Watch directory does not exist: %s", self._watch_dir)
            return []

        images = []
        for entry in sorted(self._watch_dir.rglob("*")):
            if not entry.is_file():
                continue
            if entry.suffix.lower() not in ALLOWED_EXTENSIONS:
                continue
            # Skip anything inside a _ppestatus directory
            if "_ppestatus" in entry.parts:
                continue
            images.append(entry)
        return images

    def _send_to_api(self, image_path: Path) -> dict | None:
        """POST an image to /detect/full and return the parsed JSON response.

        Returns a dict with keys: Status, Alarm, PPE-missing, image_base64, etc.
        Returns None on failure.
        """
        try:
            with open(image_path, "rb") as f:
                resp = requests.post(
                    self._api_url,
                    files={"image": (image_path.name, f, "image/jpeg")},
                    timeout=self._request_timeout,
                )

            if resp.status_code != 200:
                logger.error(
                    "API returned %d for %s: %s",
                    resp.status_code,
                    image_path.name,
                    resp.text[:200],
                )
                return None

            data = resp.json()
            if "image_base64" not in data:
                logger.error("API response missing image_base64 for %s", image_path.name)
                return None

            return data

        except requests.RequestException as exc:
            logger.error("API request failed for %s: %s", image_path.name, exc)
            return None

    def _save_output(self, image_b64: str, output_path: Path) -> bool:
        """Decode base64 image and save to the output path."""
        try:
            image_bytes = base64.b64decode(image_b64)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(image_bytes)
            return True
        except (OSError, ValueError) as exc:
            logger.error("Failed to save %s: %s", output_path, exc)
            return False

    def scan_once(self) -> int:
        """Run a single scan: find new images, process, save. Returns count processed."""
        images = self._iter_images()
        new_images = [img for img in images if img not in self._seen]

        if not new_images:
            return 0

        logger.info("Found %d new image(s)", len(new_images))
        processed = 0

        for img_path in new_images:
            if not self._running:
                break

            out_path = build_output_path(img_path, self._watch_dir)
            rel = img_path.relative_to(self._watch_dir)

            # Double-check idempotency
            if out_path.exists():
                self._seen.add(img_path)
                continue

            logger.info("Processing: %s", rel)
            result = self._send_to_api(img_path)

            if result is None:
                logger.warning("Skipping %s (API failure), will retry next cycle", rel)
                continue

            # Log the compliance metadata
            status = result.get("Status", "Unknown")
            alarm = result.get("Alarm", False)
            ppe_missing = result.get("PPE-missing", [])
            persons = result.get("persons_detected", 0)
            logger.info(
                "Result: %s | Status=%s | Alarm=%s | PPE-missing=%s | Persons=%d",
                rel, status, alarm, ppe_missing, persons,
            )

            if self._save_output(result["image_base64"], out_path):
                self._seen.add(img_path)
                processed += 1
                logger.info("Saved: %s → %s", rel, out_path)
            # If save fails, don't add to _seen — will retry next cycle

        return processed

    def watch(self) -> None:
        """Blocking polling loop. Runs until SIGTERM/SIGINT."""
        logger.info(
            "Starting watcher — dir=%s api=%s interval=%ss",
            self._watch_dir,
            self._api_url,
            self._poll_interval,
        )

        self._init_seen_from_existing_outputs()

        while self._running:
            try:
                count = self.scan_once()
                if count:
                    logger.info("Cycle complete: %d image(s) processed", count)
            except Exception:
                logger.exception("Unexpected error during scan cycle")

            # Sleep in small increments so shutdown is responsive
            elapsed = 0.0
            while elapsed < self._poll_interval and self._running:
                time.sleep(min(1.0, self._poll_interval - elapsed))
                elapsed += 1.0

        logger.info("Watcher stopped")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="FTP server-side watcher for PPE detection pipeline."
    )
    parser.add_argument(
        "--watch-dir",
        default=WATCH_DIR,
        help=f"Local directory to monitor (default: {WATCH_DIR})",
    )
    parser.add_argument(
        "--api-url",
        default=API_URL,
        help=f"PPE detection API URL (default: {API_URL})",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=POLL_INTERVAL,
        help=f"Seconds between scans (default: {POLL_INTERVAL})",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=REQUEST_TIMEOUT,
        help=f"HTTP request timeout in seconds (default: {REQUEST_TIMEOUT})",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single scan and exit.",
    )
    args = parser.parse_args()

    watcher = FTPWatcher(
        watch_dir=args.watch_dir,
        api_url=args.api_url,
        poll_interval=args.poll_interval,
        request_timeout=args.timeout,
    )

    if args.once:
        watcher._init_seen_from_existing_outputs()
        count = watcher.scan_once()
        print(f"Processed {count} image(s)")
    else:
        watcher.watch()


if __name__ == "__main__":
    main()
