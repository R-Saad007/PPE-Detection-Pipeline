"""Unit tests for src/compliance.py."""

from __future__ import annotations

import pytest

from src.compliance import ComplianceResult, assess_compliance
from src.detector import (
    LABEL_HARDHAT,
    LABEL_NO_HARDHAT,
    LABEL_NO_SAFETY_VEST,
    LABEL_PERSON,
    LABEL_SAFETY_VEST,
    Detection,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_det(
    label: str,
    x1: float, y1: float, x2: float, y2: float,
    conf: float = 0.9,
) -> Detection:
    return Detection(label=label, confidence=conf, x1=x1, y1=y1, x2=x2, y2=y2)


def _make_full_ppe_person(x1=100, y1=50, x2=300, y2=500, conf=0.9):
    """Return (person, hat, vest) tuple — person with both PPE items."""
    person = make_det(LABEL_PERSON, x1, y1, x2, y2, conf=conf)
    hat = make_det(LABEL_HARDHAT, x1 + 10, y1 + 5, x2 - 10, y1 + 180)
    vest = make_det(LABEL_SAFETY_VEST, x1 + 10, y1 + 100, x2 - 10, y2 - 150)
    return person, hat, vest


# ---------------------------------------------------------------------------
# Safe = hardhat + vest
# ---------------------------------------------------------------------------

class TestSinglePersonSafe:
    def test_safe_when_both_helmet_and_vest(self, person_box, helmet_overlapping, vest_overlapping):
        """Both helmet and vest inside person box → Safe."""
        results = assess_compliance([person_box, helmet_overlapping, vest_overlapping])
        assert len(results) == 1
        assert results[0].is_safe is True
        assert results[0].has_helmet is True
        assert results[0].has_vest is True

    def test_label_is_safe(self, person_box, helmet_overlapping, vest_overlapping):
        results = assess_compliance([person_box, helmet_overlapping, vest_overlapping])
        assert results[0].label == "Safe"


class TestSinglePersonUnsafe:
    def test_unsafe_nothing(self, person_box):
        """No PPE at all → Unsafe."""
        results = assess_compliance([person_box])
        assert results[0].is_safe is False
        assert results[0].has_helmet is False
        assert results[0].has_vest is False

    def test_label_unsafe(self, person_box):
        results = assess_compliance([person_box])
        assert results[0].label == "Unsafe - PPE Hazard"

    def test_vest_alone_is_unsafe(self, person_box, vest_overlapping):
        """Safety Vest alone does not satisfy compliance — hardhat also required."""
        results = assess_compliance([person_box, vest_overlapping])
        assert results[0].is_safe is False
        assert results[0].has_vest is True
        assert results[0].has_helmet is False

    def test_hardhat_alone_is_unsafe(self, person_box, helmet_overlapping):
        """Hardhat alone does not satisfy compliance — vest also required."""
        results = assess_compliance([person_box, helmet_overlapping])
        assert results[0].is_safe is False
        assert results[0].has_helmet is True
        assert results[0].has_vest is False

    def test_hardhat_center_outside_person_not_matched(self, person_box):
        """Hardhat whose center is far outside person box must not match."""
        far_hat = Detection(LABEL_HARDHAT, 0.9, 500.0, 250.0, 600.0, 350.0)
        results = assess_compliance([person_box, far_hat])
        assert results[0].has_helmet is False

    def test_vest_center_outside_person_not_matched(self, person_box):
        """Vest whose center is far outside person box must not match."""
        far_vest = Detection(LABEL_SAFETY_VEST, 0.9, 500.0, 400.0, 700.0, 600.0)
        results = assess_compliance([person_box, far_vest])
        assert results[0].has_vest is False


class TestNegativeSignals:
    def test_no_hardhat_overrides_hardhat_match(
        self, person_box, helmet_overlapping, vest_overlapping,
    ):
        """A NO-Hardhat centered inside the person box makes person Unsafe."""
        no_hat = Detection(LABEL_NO_HARDHAT, 0.75, 110.0, 55.0, 290.0, 230.0)
        results = assess_compliance([person_box, helmet_overlapping, vest_overlapping, no_hat])
        assert results[0].is_safe is False
        assert results[0].has_helmet is False

    def test_no_vest_overrides_vest_match(
        self, person_box, helmet_overlapping, vest_overlapping,
    ):
        """A NO-Safety Vest centered inside the person box makes person Unsafe."""
        no_vest = Detection(LABEL_NO_SAFETY_VEST, 0.75, 110.0, 150.0, 290.0, 350.0)
        results = assess_compliance([person_box, helmet_overlapping, vest_overlapping, no_vest])
        assert results[0].is_safe is False
        assert results[0].has_vest is False

    def test_distant_no_hardhat_does_not_affect(
        self, person_box, helmet_overlapping, vest_overlapping,
    ):
        """NO-Hardhat whose center is outside person box must be ignored."""
        no_hat_far = Detection(LABEL_NO_HARDHAT, 0.75, 500.0, 500.0, 600.0, 600.0)
        results = assess_compliance([person_box, helmet_overlapping, vest_overlapping, no_hat_far])
        assert results[0].is_safe is True

    def test_head_above_person_box_still_matches(self):
        """Helmet slightly above person box top is still matched (margin)."""
        person = make_det(LABEL_PERSON, 100, 200, 300, 600, conf=0.9)
        hat = make_det(LABEL_HARDHAT, 150, 140, 250, 200)
        vest = make_det(LABEL_SAFETY_VEST, 150, 300, 250, 500)
        results = assess_compliance([person, hat, vest])
        assert results[0].has_helmet is True


class TestNoPersons:
    def test_empty_detections(self):
        assert assess_compliance([]) == []

    def test_ppe_only_no_persons(self, helmet_overlapping):
        assert assess_compliance([helmet_overlapping]) == []

    def test_person_below_confidence_threshold(self):
        low_conf = Detection(LABEL_PERSON, 0.1, 0.0, 0.0, 100.0, 200.0)
        assert assess_compliance([low_conf]) == []


# ---------------------------------------------------------------------------
# Multi-person scenes
# ---------------------------------------------------------------------------

class TestMultiPerson:
    def test_two_persons_both_safe(self):
        """Two well-separated persons each with hat + vest → both Safe."""
        p_a, hat_a, vest_a = _make_full_ppe_person(0, 0, 100, 300, conf=0.9)
        p_b, hat_b, vest_b = _make_full_ppe_person(400, 0, 500, 300, conf=0.85)

        results = assess_compliance([p_a, p_b, hat_a, hat_b, vest_a, vest_b])
        assert len(results) == 2
        assert all(r.is_safe for r in results)

    def test_no_double_counting_hats(self):
        """A single hardhat must not be assigned to two persons."""
        person_a = make_det(LABEL_PERSON, 0, 0, 200, 300, conf=0.95)
        person_b = make_det(LABEL_PERSON, 50, 0, 300, 300, conf=0.80)
        hat = make_det(LABEL_HARDHAT, 70, 10, 130, 80)

        results = assess_compliance([person_a, person_b, hat])
        assert sum(r.has_helmet for r in results) == 1

    def test_no_double_counting_vests(self):
        """A single vest must not be assigned to two persons."""
        person_a = make_det(LABEL_PERSON, 0, 0, 200, 300, conf=0.95)
        person_b = make_det(LABEL_PERSON, 50, 0, 300, 300, conf=0.80)
        vest = make_det(LABEL_SAFETY_VEST, 70, 100, 180, 250)

        results = assess_compliance([person_a, person_b, vest])
        assert sum(r.has_vest for r in results) == 1

    def test_mixed_compliance(self):
        """Person A has full PPE → Safe; Person B has nothing → Unsafe."""
        p_a, hat_a, vest_a = _make_full_ppe_person(0, 0, 100, 300, conf=0.9)
        person_b = make_det(LABEL_PERSON, 400, 0, 500, 300, conf=0.85)

        results = assess_compliance([p_a, person_b, hat_a, vest_a])
        by_person = {r.person: r for r in results}
        assert by_person[p_a].is_safe is True
        assert by_person[person_b].is_safe is False

    def test_nearest_neighbour_assigns_correctly_in_overlapping_zones(self):
        """When two person zones overlap, each hat/vest goes to nearest person."""
        person_a = make_det(LABEL_PERSON, 0, 0, 300, 400, conf=0.9)
        person_b = make_det(LABEL_PERSON, 100, 0, 400, 400, conf=0.85)

        # Hats — each closest to their person's head
        hat_a = make_det(LABEL_HARDHAT, 110, 20, 150, 60)
        hat_b = make_det(LABEL_HARDHAT, 260, 20, 300, 60)

        # Vests — each closest to their person's torso
        vest_a = make_det(LABEL_SAFETY_VEST, 50, 150, 180, 300)
        vest_b = make_det(LABEL_SAFETY_VEST, 200, 150, 350, 300)

        results = assess_compliance([
            person_a, person_b, hat_a, hat_b, vest_a, vest_b,
        ])
        assert len(results) == 2
        assert all(r.is_safe for r in results), (
            "Both persons have hat + vest — both must be Safe."
        )
