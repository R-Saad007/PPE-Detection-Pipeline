"""PPE compliance association logic.

The PPE model detects ``Hardhat``, ``NO-Hardhat``, ``Safety Vest``,
and ``NO-Safety Vest``.  Person boxes come from the COCO YOLOv8s model.

Matching uses **nearest-neighbour greedy assignment** for both hardhats
and vests.  A ``NO-Hardhat`` or ``NO-Safety Vest`` detection near the
person overrides any positive match and forces an Unsafe result.

A person is **Safe** only when they have both a hardhat AND a vest.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from config.settings import PERSON_CONF_THRESHOLD
from src.detector import (
    LABEL_HARDHAT,
    LABEL_NO_HARDHAT,
    LABEL_NO_SAFETY_VEST,
    LABEL_PERSON,
    LABEL_SAFETY_VEST,
    Detection,
)
from src.utils.bbox import Box

logger = logging.getLogger(__name__)

# Fraction of person-box height above y1 used as the head-reference point
# and as the upward extension of the detection zone.
_HEAD_FRAC: float = 0.15


@dataclass
class ComplianceResult:
    """Compliance assessment for one person bounding box.

    Attributes:
        person: The :class:`~src.detector.Detection` for the person.
        is_safe: True when the person has both a hardhat and a vest.
        has_vest: True when a Safety Vest was matched to this person.
        has_helmet: True when a Hardhat was matched to this person.
    """

    person: Detection
    is_safe: bool
    has_vest: bool
    has_helmet: bool

    @property
    def label(self) -> str:
        """Human-readable compliance label."""
        return "Safe" if self.is_safe else "Unsafe - PPE Hazard"


def assess_compliance(detections: list[Detection]) -> list[ComplianceResult]:
    """Evaluate PPE compliance for every person in a detection list.

    A person is Safe only when both a Hardhat AND a Safety Vest are
    matched, with no negative override signals present.

    Args:
        detections: Raw detections from :class:`~src.detector.PPEDetector`.

    Returns:
        One :class:`ComplianceResult` per qualifying person detection.
    """
    persons = [
        d for d in detections
        if d.label == LABEL_PERSON and d.confidence >= PERSON_CONF_THRESHOLD
    ]
    hardhats = [d for d in detections if d.label == LABEL_HARDHAT]
    no_hardhats = [d for d in detections if d.label == LABEL_NO_HARDHAT]
    vests = [d for d in detections if d.label == LABEL_SAFETY_VEST]
    no_vests = [d for d in detections if d.label == LABEL_NO_SAFETY_VEST]

    if not persons:
        return []

    # --- Nearest-neighbour greedy assignment ---
    helmet_assigned: dict[int, bool] = _assign_ppe_to_head(persons, hardhats)
    vest_assigned: dict[int, bool] = _assign_ppe_to_torso(persons, vests)

    # --- Negative-signal overrides per person ---
    results: list[ComplianceResult] = []
    for idx, person in enumerate(persons):
        extended = _extend_box_upward(person.box)
        no_hat_signal = any(_center_in_box(d.box, extended) for d in no_hardhats)
        no_vest_signal = any(_center_in_box(d.box, person.box) for d in no_vests)

        has_helmet = helmet_assigned.get(idx, False) and not no_hat_signal
        has_vest = vest_assigned.get(idx, False) and not no_vest_signal

        logger.debug(
            "Person assessed",
            extra={
                "conf": round(person.confidence, 3),
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "no_hat_signal": no_hat_signal,
                "no_vest_signal": no_vest_signal,
            },
        )
        results.append(
            ComplianceResult(
                person=person,
                is_safe=has_helmet and has_vest,
                has_vest=has_vest,
                has_helmet=has_helmet,
            )
        )
    return results


# ---------------------------------------------------------------------------
# Matching helpers
# ---------------------------------------------------------------------------

def _assign_ppe_to_head(
    persons: list[Detection],
    hardhats: list[Detection],
) -> dict[int, bool]:
    """Nearest-neighbour greedy matching of hardhats to persons.

    A hat is a candidate if its center falls within the person's extended
    zone (person box + 15% upward margin).  Assigns closest pairs first.

    Args:
        persons: List of person detections.
        hardhats: List of Hardhat detections.

    Returns:
        Mapping of person index -> True if a hat was assigned.
    """
    result: dict[int, bool] = {i: False for i in range(len(persons))}
    if not hardhats:
        return result

    pairings: list[tuple[float, int, int]] = []
    for p_idx, person in enumerate(persons):
        zone = _extend_box_upward(person.box)
        p_head_x = (person.x1 + person.x2) / 2.0
        p_head_y = person.y1 + (person.y2 - person.y1) * _HEAD_FRAC

        for h_idx, hat in enumerate(hardhats):
            if not _center_in_box(hat.box, zone):
                continue
            h_cx = (hat.x1 + hat.x2) / 2.0
            h_cy = (hat.y1 + hat.y2) / 2.0
            dist = math.hypot(h_cx - p_head_x, h_cy - p_head_y)
            pairings.append((dist, p_idx, h_idx))

    pairings.sort(key=lambda t: t[0])
    assigned_persons: set[int] = set()
    assigned_hats: set[int] = set()

    for _, p_idx, h_idx in pairings:
        if p_idx not in assigned_persons and h_idx not in assigned_hats:
            result[p_idx] = True
            assigned_persons.add(p_idx)
            assigned_hats.add(h_idx)

    return result


def _assign_ppe_to_torso(
    persons: list[Detection],
    vests: list[Detection],
) -> dict[int, bool]:
    """Nearest-neighbour greedy matching of safety vests to persons.

    A vest is a candidate if its center falls within the person's bounding
    box (no upward extension — vests sit on the torso, not above the head).
    Assigns closest pairs first (vest center -> person torso center).

    Args:
        persons: List of person detections.
        vests: List of Safety Vest detections.

    Returns:
        Mapping of person index -> True if a vest was assigned.
    """
    result: dict[int, bool] = {i: False for i in range(len(persons))}
    if not vests:
        return result

    pairings: list[tuple[float, int, int]] = []
    for p_idx, person in enumerate(persons):
        # Torso reference: center of person box
        p_cx = (person.x1 + person.x2) / 2.0
        p_cy = (person.y1 + person.y2) / 2.0

        for v_idx, vest in enumerate(vests):
            if not _center_in_box(vest.box, person.box):
                continue
            v_cx = (vest.x1 + vest.x2) / 2.0
            v_cy = (vest.y1 + vest.y2) / 2.0
            dist = math.hypot(v_cx - p_cx, v_cy - p_cy)
            pairings.append((dist, p_idx, v_idx))

    pairings.sort(key=lambda t: t[0])
    assigned_persons: set[int] = set()
    assigned_vests: set[int] = set()

    for _, p_idx, v_idx in pairings:
        if p_idx not in assigned_persons and v_idx not in assigned_vests:
            result[p_idx] = True
            assigned_persons.add(p_idx)
            assigned_vests.add(v_idx)

    return result


def _extend_box_upward(box: Box) -> Box:
    """Return *box* with its top edge extended upward by ``_HEAD_FRAC``."""
    x1, y1, x2, y2 = box
    margin = (y2 - y1) * _HEAD_FRAC
    return (x1, max(0.0, y1 - margin), x2, y2)


def _center_in_box(inner: Box, outer: Box) -> bool:
    """Return True if the centre of *inner* lies within *outer*."""
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]
