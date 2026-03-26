"""Unit tests for src/annotator.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.annotator import _COLOR_SAFE, _COLOR_UNSAFE, annotate
from src.compliance import ComplianceResult
from src.detector import LABEL_PERSON, Detection


def make_result(is_safe: bool, x1=100, y1=50, x2=300, y2=400) -> ComplianceResult:
    person = Detection(
        label=LABEL_PERSON,
        confidence=0.9,
        x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
    )
    return ComplianceResult(
        person=person,
        is_safe=is_safe,
        has_vest=is_safe,
        has_helmet=is_safe,
    )


class TestAnnotate:
    def test_returns_new_array(self, blank_image):
        result = make_result(True)
        out = annotate(blank_image, [result])
        assert out is not blank_image

    def test_output_shape_unchanged(self, blank_image):
        result = make_result(True)
        out = annotate(blank_image, [result])
        assert out.shape == blank_image.shape

    def test_no_results_leaves_image_unchanged(self, blank_image):
        out = annotate(blank_image, [])
        np.testing.assert_array_equal(out, blank_image)

    def test_safe_draws_green_pixels(self, blank_image):
        result = make_result(True, x1=100, y1=50, x2=300, y2=400)
        out = annotate(blank_image, [result])
        b, g, r = out[200, 100]
        assert g > b and g > r, "Expected dominant green on safe box edge"

    def test_unsafe_draws_red_pixels(self, blank_image):
        result = make_result(False, x1=100, y1=50, x2=300, y2=400)
        out = annotate(blank_image, [result])
        b, g, r = out[200, 100]
        assert r > g and r > b, "Expected dominant red on unsafe box edge"

    def test_multiple_results(self, blank_image):
        results = [
            make_result(True, x1=10, y1=10, x2=100, y2=200),
            make_result(False, x1=300, y1=10, x2=400, y2=200),
        ]
        out = annotate(blank_image, results)
        assert out.shape == blank_image.shape

    def test_box_clipped_to_image_boundary(self):
        img = np.full((200, 200, 3), 200, dtype=np.uint8)
        result = make_result(True, x1=-10, y1=-10, x2=250, y2=250)
        out = annotate(img, [result])
        assert out.shape == img.shape

    def test_draw_ppe_boxes_false_skips_overlay(self, blank_image):
        """When draw_ppe_boxes=False, passing ppe_detections has no effect."""
        from src.detector import LABEL_HARDHAT
        ppe_det = Detection(LABEL_HARDHAT, 0.8, 110.0, 55.0, 290.0, 230.0)
        result = make_result(True, x1=100, y1=50, x2=300, y2=400)
        out_without = annotate(blank_image, [result], ppe_detections=[ppe_det], draw_ppe_boxes=False)
        out_without_ppe = annotate(blank_image, [result], draw_ppe_boxes=False)
        np.testing.assert_array_equal(out_without, out_without_ppe)
