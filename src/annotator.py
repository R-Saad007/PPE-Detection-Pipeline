"""Draw compliance bounding boxes and labels on images.

In batch mode (``DRAW_PPE_BOXES=false``) only the person box and compliance
label are drawn.  In debug/Flask mode (``DRAW_PPE_BOXES=true``) individual
PPE bounding boxes are also drawn in yellow.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from config.settings import DRAW_PPE_BOXES
from src.compliance import ComplianceResult
from src.detector import Detection

logger = logging.getLogger(__name__)

# Colours in BGR
_COLOR_SAFE = (0, 200, 0)
_COLOR_UNSAFE = (0, 0, 220)
_COLOR_TEXT_BG_SAFE = (0, 180, 0)
_COLOR_TEXT_BG_UNSAFE = (0, 0, 200)
_COLOR_WHITE = (255, 255, 255)
_COLOR_PPE_BOX = (0, 200, 255)  # yellow in BGR

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_BOX_THICKNESS = 2
_LABEL_PAD = 4


def annotate(
    img: np.ndarray,
    results: list[ComplianceResult],
    ppe_detections: list[Detection] | None = None,
    draw_ppe_boxes: bool | None = None,
) -> np.ndarray:
    """Draw compliance annotations on a copy of *img*.

    For each :class:`~src.compliance.ComplianceResult`:

    - Draws a solid green/red rectangle around the person.
    - Places a filled label badge (``"Safe"`` / ``"Unsafe - PPE Hazard"``).
      The badge is drawn **above** the box when space permits; when the box
      sits at or near the top of the image the badge is drawn **inside** the
      box so it is always fully visible.

    When ``draw_ppe_boxes`` is ``True`` (or the ``DRAW_PPE_BOXES`` env var is
    set), individual PPE bounding boxes are drawn in yellow.

    Args:
        img: Source BGR ``uint8`` image (not modified in place).
        results: Compliance results from :func:`~src.compliance.assess_compliance`.
        ppe_detections: Optional list of individual PPE detections to overlay.
            Only used when *draw_ppe_boxes* is ``True``.
        draw_ppe_boxes: Override the ``DRAW_PPE_BOXES`` env-var setting.
            ``None`` uses the config value.

    Returns:
        Annotated BGR ``uint8`` image (new array).
    """
    out = img.copy()
    img_h, img_w = out.shape[:2]
    should_draw_ppe = draw_ppe_boxes if draw_ppe_boxes is not None else DRAW_PPE_BOXES

    # Optional individual PPE boxes (yellow) — debug / Flask mode only
    if should_draw_ppe and ppe_detections:
        for det in ppe_detections:
            x1 = max(0, int(det.x1))
            y1 = max(0, int(det.y1))
            x2 = min(img_w - 1, int(det.x2))
            y2 = min(img_h - 1, int(det.y2))
            cv2.rectangle(out, (x1, y1), (x2, y2), _COLOR_PPE_BOX, 1)

    for result in results:
        x1 = max(0, int(result.person.x1))
        y1 = max(0, int(result.person.y1))
        x2 = min(img_w - 1, int(result.person.x2))
        y2 = min(img_h - 1, int(result.person.y2))

        color = _COLOR_SAFE if result.is_safe else _COLOR_UNSAFE
        bg_color = _COLOR_TEXT_BG_SAFE if result.is_safe else _COLOR_TEXT_BG_UNSAFE

        # Person bounding box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, _BOX_THICKNESS)

        # Adaptive font scale proportional to box height — readable at any res
        box_h = max(1, y2 - y1)
        font_scale = max(0.4, min(1.2, box_h / 400.0))
        thickness = max(1, int(font_scale * 2))

        label = result.label
        (text_w, text_h), baseline = cv2.getTextSize(
            label, _FONT, font_scale, thickness
        )
        badge_h = text_h + baseline + 2 * _LABEL_PAD
        badge_w = text_w + 2 * _LABEL_PAD

        # --- Badge horizontal placement (never exceeds image width) ---
        badge_x1 = x1
        badge_x2 = min(img_w, x1 + badge_w)

        # --- Badge vertical placement ---
        # Prefer above the box; fall back to inside-top when no room.
        if y1 >= badge_h:
            # Enough space above the box
            badge_y1 = y1 - badge_h
            badge_y2 = y1
        else:
            # Draw inside the box, just below the top edge
            badge_y1 = y1
            badge_y2 = min(img_h, y1 + badge_h)

        cv2.rectangle(out, (badge_x1, badge_y1), (badge_x2, badge_y2), bg_color, -1)

        text_x = badge_x1 + _LABEL_PAD
        text_y = badge_y2 - baseline - _LABEL_PAD
        cv2.putText(
            out,
            label,
            (text_x, text_y),
            _FONT,
            font_scale,
            _COLOR_WHITE,
            thickness,
            cv2.LINE_AA,
        )

    logger.debug("Annotated image", extra={"persons": len(results)})
    return out
