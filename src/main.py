"""Flask application — PPE detection HTTP service.

Routes
------
GET  /health   Liveness probe; confirms the model is loaded.
POST /detect   Accept a multipart image upload; return annotated JPEG.
GET  /         Minimal HTML upload form for manual browser testing.
"""

from __future__ import annotations

import base64
import gc
import io
import logging
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from config.logging_config import configure_logging
from config.settings import DEAD_LETTER_DIR, DRAW_PPE_BOXES, FLASK_PORT, UPLOAD_DIR
from src.annotator import annotate
from src.compliance import assess_compliance
from src.detector import (
    LABEL_HARDHAT,
    LABEL_NO_HARDHAT,
    LABEL_NO_SAFETY_VEST,
    LABEL_SAFETY_VEST,
    PPEDetector,
)
from src.utils.image_io import load_image, save_image, validate_image_path

configure_logging(log_dir=Path("logs"))
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model singleton — loaded once at startup
# ---------------------------------------------------------------------------
_detector: PPEDetector | None = None


def _get_detector() -> PPEDetector:
    """Return the shared detector, initialising it on the first call."""
    global _detector
    if _detector is None:
        _detector = PPEDetector.get_instance()
    return _detector


# Pre-load at import time so the first request is not slow.
try:
    _get_detector()
    _model_loaded = True
except Exception:
    logger.exception("Failed to load model at startup")
    _model_loaded = False


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe.

    Returns:
        JSON ``{"status": "ok", "model_loaded": <bool>}``.
    """
    return jsonify({"status": "ok", "model_loaded": _model_loaded})


@app.get("/")
def index():
    """Minimal HTML upload form for manual browser testing."""
    html = """<!doctype html>
<html><head><title>PPE Detection</title></head>
<body>
<h2>PPE Detection</h2>
<form method="post" action="/detect" enctype="multipart/form-data">
  <input type="file" name="image" accept="image/*" required>
  <button type="submit">Detect</button>
</form>
</body></html>"""
    return html, 200, {"Content-Type": "text/html"}


@app.post("/detect")
def detect():
    """Accept a multipart/form-data image upload and return an annotated JPEG.

    Query params:
        json=1  Return a JSON summary instead of the annotated image.

    Returns:
        Annotated ``image/jpeg`` response, or JSON metadata when ``?json=1``.
        HTTP 400 on bad input; HTTP 500 on inference error.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save upload to a temp location inside UPLOAD_DIR
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / Path(file.filename).name

    try:
        file.save(str(upload_path))
    except OSError as exc:
        logger.error("Failed to save upload", extra={"error": str(exc)})
        return jsonify({"error": "Could not save uploaded file"}), 500

    try:
        validated = validate_image_path(upload_path, watch_dir=UPLOAD_DIR)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.warning("Invalid upload", extra={"reason": str(exc)})
        _move_to_dead_letter(upload_path)
        return jsonify({"error": str(exc)}), 400

    img: np.ndarray | None = None
    try:
        img = load_image(validated)
        detector = _get_detector()
        detections = detector.detect(img)
        results = assess_compliance(detections)

        # PPE boxes for overlay (Flask / debug mode)
        ppe_dets = [
            d for d in detections
            if d.label in (
                LABEL_HARDHAT, LABEL_SAFETY_VEST,
                LABEL_NO_HARDHAT, LABEL_NO_SAFETY_VEST,
            )
        ]

        annotated = annotate(img, results, ppe_detections=ppe_dets, draw_ppe_boxes=DRAW_PPE_BOXES)

        safe_count = sum(1 for r in results if r.is_safe)
        unsafe_count = len(results) - safe_count
        status = (
            "Safe"
            if results and all(r.is_safe for r in results)
            else ("Unsafe" if results else "No person detected")
        )

        if request.args.get("json") == "1":
            return jsonify(
                {
                    "status": status,
                    "persons_detected": len(results),
                    "safe": safe_count,
                    "unsafe": unsafe_count,
                }
            )

        # Encode annotated image and stream back
        success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            return jsonify({"error": "Failed to encode annotated image"}), 500

        return send_file(
            io.BytesIO(buf.tobytes()),
            mimetype="image/jpeg",
            download_name="annotated.jpg",
        )

    except Exception:
        logger.exception("Inference error", extra={"file": file.filename})
        _move_to_dead_letter(upload_path)
        return jsonify({"error": "Internal inference error"}), 500

    finally:
        if img is not None:
            del img
            gc.collect()


@app.post("/detect/full")
def detect_full():
    """Accept a multipart image and return JSON with annotated image + compliance metadata.

    Returns JSON:
        {
            "Status": "Safe" | "Unsafe: No Helmet" | "Unsafe: No Vest" | "Unsafe: No Helmet & Vest" | "No person detected",
            "Alarm": true | false,
            "PPE-missing": ["Hard Hat", "Safety Vest"],  // empty if Safe or no person
            "persons_detected": int,
            "safe": int,
            "unsafe": int,
            "image_base64": "<base64 JPEG>"
        }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_path = UPLOAD_DIR / Path(file.filename).name

    try:
        file.save(str(upload_path))
    except OSError as exc:
        logger.error("Failed to save upload", extra={"error": str(exc)})
        return jsonify({"error": "Could not save uploaded file"}), 500

    try:
        validated = validate_image_path(upload_path, watch_dir=UPLOAD_DIR)
    except (ValueError, FileNotFoundError, OSError) as exc:
        logger.warning("Invalid upload", extra={"reason": str(exc)})
        _move_to_dead_letter(upload_path)
        return jsonify({"error": str(exc)}), 400

    img = None
    try:
        img = load_image(validated)
        detector = _get_detector()
        detections = detector.detect(img)
        results = assess_compliance(detections)

        ppe_dets = [
            d for d in detections
            if d.label in (
                LABEL_HARDHAT, LABEL_SAFETY_VEST,
                LABEL_NO_HARDHAT, LABEL_NO_SAFETY_VEST,
            )
        ]

        annotated = annotate(img, results, ppe_detections=ppe_dets, draw_ppe_boxes=DRAW_PPE_BOXES)

        # Encode annotated image to base64
        success, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not success:
            return jsonify({"error": "Failed to encode annotated image"}), 500
        image_b64 = base64.b64encode(buf.tobytes()).decode("ascii")

        # Build compliance metadata
        safe_count = sum(1 for r in results if r.is_safe)
        unsafe_count = len(results) - safe_count

        if not results:
            status = "No person detected"
            alarm = False
            ppe_missing: list[str] = []
        elif unsafe_count == 0:
            status = "Safe"
            alarm = False
            ppe_missing = []
        else:
            # Aggregate: worst-case status across all persons
            missing_helmet = any(not r.has_helmet for r in results if not r.is_safe)
            missing_vest = any(not r.has_vest for r in results if not r.is_safe)
            ppe_missing = []
            if missing_helmet:
                ppe_missing.append("Hard Hat")
            if missing_vest:
                ppe_missing.append("Safety Vest")

            if missing_helmet and missing_vest:
                status = "Unsafe: No Helmet & Vest"
            elif missing_helmet:
                status = "Unsafe: No Helmet"
            else:
                status = "Unsafe: No Vest"
            alarm = True

        return jsonify({
            "Status": status,
            "Alarm": alarm,
            "PPE-missing": ppe_missing,
            "persons_detected": len(results),
            "safe": safe_count,
            "unsafe": unsafe_count,
            "image_base64": image_b64,
        })

    except Exception:
        logger.exception("Inference error", extra={"file": file.filename})
        _move_to_dead_letter(upload_path)
        return jsonify({"error": "Internal inference error"}), 500

    finally:
        if img is not None:
            del img
            gc.collect()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _move_to_dead_letter(path: Path) -> None:
    """Move a failed image to the dead-letter directory.

    Args:
        path: Source image path.
    """
    import shutil

    try:
        DEAD_LETTER_DIR.mkdir(parents=True, exist_ok=True)
        dest = DEAD_LETTER_DIR / path.name
        shutil.move(str(path), dest)
        logger.warning("Moved to dead-letter", extra={"file": path.name})
    except Exception:
        logger.exception("Could not move to dead-letter", extra={"file": path.name})


# ---------------------------------------------------------------------------
# Dev entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=FLASK_PORT, debug=(FLASK_PORT == 5000))
