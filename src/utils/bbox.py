"""Bounding-box utility functions: IoU, containment, NMS."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Box = tuple[float, float, float, float]  # (x1, y1, x2, y2)


def iou(box_a: Box, box_b: Box) -> float:
    """Compute Intersection-over-Union between two bounding boxes.

    Args:
        box_a: (x1, y1, x2, y2) in pixel coordinates.
        box_b: (x1, y1, x2, y2) in pixel coordinates.

    Returns:
        IoU value in [0, 1].
    """
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0.0:
        return 0.0
    return inter_area / union_area


def center_contained(inner: Box, outer: Box) -> bool:
    """Return True if the center point of *inner* lies within *outer*.

    Args:
        inner: Box whose center is tested.
        outer: Box acting as the containing region.

    Returns:
        True when the center of *inner* is strictly inside *outer*.
    """
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def nms(
    boxes: list[Box],
    scores: list[float],
    iou_threshold: float = 0.45,
) -> list[int]:
    """Apply greedy non-maximum suppression and return surviving indices.

    Args:
        boxes: List of (x1, y1, x2, y2) boxes.
        scores: Confidence score per box.
        iou_threshold: Overlap threshold above which weaker boxes are suppressed.

    Returns:
        Sorted list of indices of kept boxes (highest score first).
    """
    if not boxes:
        return []

    arr = np.array(boxes, dtype=np.float32)
    sc = np.array(scores, dtype=np.float32)
    order = sc.argsort()[::-1].tolist()

    kept: list[int] = []
    while order:
        idx = order.pop(0)
        kept.append(idx)
        order = [
            j for j in order if iou(boxes[idx], boxes[j]) <= iou_threshold
        ]
    return kept
