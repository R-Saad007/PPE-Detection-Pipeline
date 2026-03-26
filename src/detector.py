"""Dual-model inference wrapper.

Two models are used together:
- YOLOv8s (COCO) → ``Person`` detections.
- yihong1120/Construction-Hazard-Detection (YOLO11m) → PPE detections
  (``Hardhat``, ``NO-Hardhat``, ``Safety Vest``, ``NO-Safety Vest``).

Both are loaded once at startup via :meth:`PPEDetector.get_instance` and
reused for every subsequent call to :meth:`PPEDetector.detect`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar, NamedTuple

import numpy as np

from config.settings import (
    HF_MODEL_FILE,
    HF_MODEL_REPO,
    MIN_PERSON_AREA_FRAC,
    MIN_PERSON_ASPECT_RATIO,
    NMS_IOU_THRESHOLD,
    PERSON_CONF_THRESHOLD,
    PPE_CONF_THRESHOLD,
    SITE_ROI,
)
from src.utils.bbox import Box

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------
LABEL_PERSON: str = "Person"
LABEL_HARDHAT: str = "Hardhat"
LABEL_SAFETY_VEST: str = "Safety Vest"
LABEL_NO_HARDHAT: str = "NO-Hardhat"
LABEL_NO_SAFETY_VEST: str = "NO-Safety Vest"

# COCO class id for person
_COCO_PERSON_ID: int = 0


class Detection(NamedTuple):
    """A single object detection result.

    Attributes:
        label: Class label string (e.g. ``"Person"``, ``"Hardhat"``).
        confidence: Detection confidence in (0, 1].
        x1: Left pixel coordinate.
        y1: Top pixel coordinate.
        x2: Right pixel coordinate.
        y2: Bottom pixel coordinate.
    """

    label: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> Box:
        """Return the bounding box as ``(x1, y1, x2, y2)``."""
        return (self.x1, self.y1, self.x2, self.y2)


class PPEDetector:
    """Dual-model singleton: COCO person detector + PPE classifier.

    Load once at Flask startup via :meth:`get_instance`; call
    :meth:`detect` for each image.
    """

    _instance: ClassVar[PPEDetector | None] = None

    def __init__(self) -> None:
        self._person_model = self._load_person_model()
        self._ppe_model = self._load_ppe_model()
        logger.info(
            "PPEDetector initialised (dual-model: COCO person + yihong1120 PPE)"
        )

    # ------------------------------------------------------------------
    # Singleton accessor
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls) -> PPEDetector:
        """Return the shared :class:`PPEDetector`, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_person_model():
        """Load YOLOv8n (COCO) for person detection."""
        try:
            from ultralytics import YOLO  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("ultralytics not installed") from exc

        # Prefer YOLOv8s (small) for better recall; fall back to nano.
        models_dir = Path(__file__).parent.parent / "models"
        for candidate in ("yolov8s.pt", "yolov8n.pt"):
            local = models_dir / candidate
            if local.exists():
                model = YOLO(str(local))
                logger.info("Person model loaded: %s", local.name)
                return model
        # Neither found locally — download small model.
        model = YOLO("yolov8s.pt")
        logger.info("Person model downloaded: yolov8s.pt")
        return model

    @staticmethod
    def _load_ppe_model():
        """Download (once) and load the PPE model from HuggingFace."""
        try:
            from ultralytics import YOLO  # type: ignore[import]
            from huggingface_hub import hf_hub_download  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics / huggingface_hub not installed"
            ) from exc

        weights = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename=HF_MODEL_FILE,
        )
        model = YOLO(weights)
        logger.info(
            "PPE model loaded from HuggingFace: %s — classes: %s",
            HF_MODEL_REPO,
            list(model.names.values()),
        )
        return model

    @staticmethod
    def _boxes_to_detections(
        results,
        model,
        label_filter: set[str] | None = None,
        conf_threshold: float = 0.0,
        label_override: str | None = None,
    ) -> list[Detection]:
        """Convert ultralytics result boxes to :class:`Detection` objects.

        Args:
            results: ``results[0]`` from ``model.predict()``.
            model: The YOLO model (for ``model.names``).
            label_filter: Only keep detections whose label is in this set.
                ``None`` means keep all.
            conf_threshold: Minimum confidence.
            label_override: Force all detections to this label string
                (used to normalise COCO ``"person"`` → ``"Person"``).
        """
        detections: list[Detection] = []
        for box in results.boxes:
            raw_label = model.names[int(box.cls)]
            label = label_override if label_override else raw_label
            conf = float(box.conf)
            if conf < conf_threshold:
                continue
            if label_filter is not None and raw_label not in label_filter:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            detections.append(
                Detection(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)
            )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, img_bgr: np.ndarray) -> list[Detection]:
        """Run both models and return merged detections.

        Args:
            img_bgr: Input image as a BGR ``uint8`` NumPy array.

        Returns:
            Combined list of :class:`Detection` objects.
        """
        # --- Person detection (COCO model) ---
        img_pixels = img_bgr.shape[0] * img_bgr.shape[1]
        min_person_area = img_pixels * MIN_PERSON_AREA_FRAC

        person_results = self._person_model.predict(
            img_bgr,
            conf=PERSON_CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
            classes=[_COCO_PERSON_ID],
        )
        roi = _parse_roi(SITE_ROI, img_bgr.shape[1], img_bgr.shape[0])
        persons = [
            d for d in self._boxes_to_detections(
                person_results[0],
                self._person_model,
                label_filter={"person"},
                conf_threshold=PERSON_CONF_THRESHOLD,
                label_override=LABEL_PERSON,
            )
            if (d.x2 - d.x1) * (d.y2 - d.y1) >= min_person_area
            and (d.x2 - d.x1) > 0
            and (d.y2 - d.y1) / (d.x2 - d.x1) >= MIN_PERSON_ASPECT_RATIO
            and (roi is None or _center_in_roi(d, roi))
        ]

        # --- PPE detection (yihong1120 model) ---
        ppe_results = self._ppe_model.predict(
            img_bgr,
            conf=PPE_CONF_THRESHOLD,
            iou=NMS_IOU_THRESHOLD,
            verbose=False,
        )
        ppe_dets = self._boxes_to_detections(
            ppe_results[0],
            self._ppe_model,
            label_filter={
                LABEL_HARDHAT, LABEL_NO_HARDHAT,
                LABEL_SAFETY_VEST, LABEL_NO_SAFETY_VEST,
            },
            conf_threshold=PPE_CONF_THRESHOLD,
        )

        detections = persons + ppe_dets
        logger.debug(
            "Inference complete",
            extra={"persons": len(persons), "ppe": len(ppe_dets)},
        )
        return detections


# ---------------------------------------------------------------------------
# ROI helpers (module-level so they can be unit-tested independently)
# ---------------------------------------------------------------------------

def _parse_roi(
    roi_str: str,
    img_w: int,
    img_h: int,
) -> tuple[float, float, float, float] | None:
    """Parse a ``"x1,y1,x2,y2"`` normalised ROI string into pixel coordinates.

    Args:
        roi_str: Comma-separated normalised fractions, e.g. ``"0.05,0,0.85,1"``.
            Empty string disables ROI filtering.
        img_w: Image width in pixels.
        img_h: Image height in pixels.

    Returns:
        ``(x1, y1, x2, y2)`` in pixel coordinates, or ``None`` if *roi_str*
        is empty or malformed.
    """
    if not roi_str:
        return None
    try:
        parts = [float(v) for v in roi_str.split(",")]
        if len(parts) != 4:
            logger.warning("SITE_ROI must be 'x1,y1,x2,y2' — ROI disabled")
            return None
        x1, y1, x2, y2 = parts
        return (x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h)
    except ValueError:
        logger.warning("SITE_ROI parse error — ROI disabled")
        return None


def _center_in_roi(
    det: Detection,
    roi: tuple[float, float, float, float],
) -> bool:
    """Return True if the centre of *det* falls within *roi*.

    Args:
        det: A person detection.
        roi: ``(x1, y1, x2, y2)`` pixel-coordinate ROI rectangle.
    """
    cx = (det.x1 + det.x2) / 2.0
    cy = (det.y1 + det.y2) / 2.0
    return roi[0] <= cx <= roi[2] and roi[1] <= cy <= roi[3]
