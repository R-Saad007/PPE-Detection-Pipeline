"""Centralised configuration — all values read from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
FLASK_ENV: str = os.environ.get("FLASK_ENV", "development")
FLASK_PORT: int = int(os.environ.get("FLASK_PORT", "5000"))

# ---------------------------------------------------------------------------
# Directory paths
# ---------------------------------------------------------------------------
UPLOAD_DIR: Path = Path(os.environ.get("UPLOAD_DIR", "uploads"))
OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", "outputs"))
DEAD_LETTER_DIR: Path = Path(os.environ.get("DEAD_LETTER_DIR", "failed"))

# ---------------------------------------------------------------------------
# Phase 2 — FTP watcher
# ---------------------------------------------------------------------------
FTP_WATCH_DIR: Path = Path(os.environ.get("FTP_WATCH_DIR", "/ftp/uploads"))
POLL_INTERVAL_SEC: float = float(os.environ.get("POLL_INTERVAL_SEC", "2"))

# ---------------------------------------------------------------------------
# HuggingFace model
# ---------------------------------------------------------------------------
HF_MODEL_REPO: str = os.environ.get(
    "HF_MODEL_REPO", "yihong1120/Construction-Hazard-Detection"
)
HF_MODEL_FILE: str = os.environ.get(
    "HF_MODEL_FILE", "models/yolo11/pt/yolo11m.pt"
)

# ---------------------------------------------------------------------------
# Image validation
# ---------------------------------------------------------------------------
MAX_IMAGE_SIZE_MB: float = float(os.environ.get("MAX_IMAGE_SIZE_MB", "20"))
MAX_IMAGE_SIZE_BYTES: int = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

# First bytes of common image formats (magic-byte validation)
IMAGE_MAGIC_BYTES: dict[bytes, str] = {
    b"\xff\xd8\xff": "jpeg",
    b"\x89PNG\r\n\x1a\n": "png",
    b"BM": "bmp",
    b"RIFF": "webp",
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
PERSON_CONF_THRESHOLD: float = float(
    os.environ.get("PERSON_CONF_THRESHOLD", "0.25")
)
# Minimum person bounding-box area as a fraction of total image pixels.
# Filters spurious detections on blurry/low-res regions (pixel noise).
# At 0.1% a real person must occupy ≥ ~8 300 px in a 4K frame.
MIN_PERSON_AREA_FRAC: float = float(
    os.environ.get("MIN_PERSON_AREA_FRAC", "0.001")
)
# Minimum height-to-width ratio for a valid person box.
# A person wider than tall (h/w < 0.75) is physically implausible upright
# or crouching — indicates a vehicle, equipment, or other horizontal object.
MIN_PERSON_ASPECT_RATIO: float = float(
    os.environ.get("MIN_PERSON_ASPECT_RATIO", "0.75")
)

# ---------------------------------------------------------------------------
# Site Region of Interest (ROI)
# ---------------------------------------------------------------------------
# Optional rectangular zone defining the inside of the site, as normalised
# fractions of image dimensions: "x1,y1,x2,y2" each in [0.0, 1.0].
# Persons whose centre falls outside this zone are treated as pedestrians /
# civilians and excluded from compliance checks entirely.
# Leave empty (default) to check the full frame.
# Example: SITE_ROI=0.05,0.0,0.85,1.0  (exclude far-left and far-right strips)
SITE_ROI: str = os.environ.get("SITE_ROI", "").strip()
PPE_CONF_THRESHOLD: float = float(os.environ.get("PPE_CONF_THRESHOLD", "0.35"))
VEST_IOU_THRESHOLD: float = float(os.environ.get("VEST_IOU_THRESHOLD", "0.3"))
HELMET_IOU_THRESHOLD: float = float(
    os.environ.get("HELMET_IOU_THRESHOLD", "0.2")
)
NMS_IOU_THRESHOLD: float = 0.45

# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------
DRAW_PPE_BOXES: bool = os.environ.get("DRAW_PPE_BOXES", "true").lower() == "true"
