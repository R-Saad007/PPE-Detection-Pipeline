"""Unit tests for src/detector.py."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.detector import (
    LABEL_HARDHAT,
    LABEL_NO_HARDHAT,
    LABEL_PERSON,
    LABEL_SAFETY_VEST,
    Detection,
    PPEDetector,
)
from src.utils.bbox import iou


# ---------------------------------------------------------------------------
# Detection NamedTuple
# ---------------------------------------------------------------------------

class TestDetectionNamedTuple:
    def test_box_property(self):
        d = Detection(LABEL_PERSON, 0.9, 10.0, 20.0, 110.0, 220.0)
        assert d.box == (10.0, 20.0, 110.0, 220.0)

    def test_label_field(self):
        d = Detection(LABEL_HARDHAT, 0.75, 0.0, 0.0, 50.0, 50.0)
        assert d.label == LABEL_HARDHAT
        assert d.confidence == pytest.approx(0.75)

    def test_all_label_constants_are_strings(self):
        for label in (LABEL_PERSON, LABEL_HARDHAT, LABEL_NO_HARDHAT):
            assert isinstance(label, str)
            assert len(label) > 0


# ---------------------------------------------------------------------------
# PPEDetector — singleton behaviour
# ---------------------------------------------------------------------------

class TestPPEDetectorSingleton:
    def test_singleton_returns_same_instance(self):
        PPEDetector._instance = None
        mock_det = PPEDetector.__new__(PPEDetector)
        mock_det._person_model = MagicMock()
        mock_det._ppe_model = MagicMock()
        PPEDetector._instance = mock_det
        assert PPEDetector.get_instance() is mock_det
        PPEDetector._instance = None

    def test_detect_merges_persons_and_ppe(self):
        """detect() should return persons from COCO model + PPE from keremberke."""
        det = PPEDetector.__new__(PPEDetector)

        # Mock COCO person model
        person_box = MagicMock()
        person_box.cls = MagicMock(); person_box.cls.__int__ = lambda s: 0
        person_box.conf = MagicMock(); person_box.conf.__float__ = lambda s: 0.8
        person_box.xyxy = [[0.0, 0.0, 100.0, 200.0]]

        person_result = MagicMock()
        person_result.boxes = [person_box]
        person_model = MagicMock()
        person_model.names = {0: "person"}
        person_model.predict.return_value = [person_result]

        # Mock PPE model
        hat_box = MagicMock()
        hat_box.cls = MagicMock(); hat_box.cls.__int__ = lambda s: 0
        hat_box.conf = MagicMock(); hat_box.conf.__float__ = lambda s: 0.75
        hat_box.xyxy = [[5.0, 5.0, 90.0, 80.0]]

        ppe_result = MagicMock()
        ppe_result.boxes = [hat_box]
        ppe_model = MagicMock()
        ppe_model.names = {0: "Hardhat"}
        ppe_model.predict.return_value = [ppe_result]

        det._person_model = person_model
        det._ppe_model = ppe_model

        img = np.zeros((300, 300, 3), dtype=np.uint8)
        results = det.detect(img)

        labels = {r.label for r in results}
        assert LABEL_PERSON in labels
        assert LABEL_HARDHAT in labels


# ---------------------------------------------------------------------------
# IoU helper
# ---------------------------------------------------------------------------

class TestIoUHelper:
    def test_perfect_overlap(self):
        box = (0.0, 0.0, 100.0, 100.0)
        assert iou(box, box) == pytest.approx(1.0)

    def test_no_overlap(self):
        assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == pytest.approx(0.0)

    def test_partial_overlap(self):
        a = (0.0, 0.0, 10.0, 10.0)
        b = (5.0, 5.0, 15.0, 15.0)
        assert 0.0 < iou(a, b) < 1.0
