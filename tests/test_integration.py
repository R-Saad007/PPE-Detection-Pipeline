"""Integration tests: full pipeline and Flask routes.

No real YOLO model required — the detector is mocked.
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from src.annotator import annotate
from src.compliance import assess_compliance
from src.detector import (
    LABEL_HARDHAT,
    LABEL_NO_HARDHAT,
    LABEL_NO_SAFETY_VEST,
    LABEL_PERSON,
    LABEL_SAFETY_VEST,
    Detection,
)
from src.utils.image_io import save_image


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def safe_detections() -> list[Detection]:
    """One person wearing a hardhat."""
    return [
        Detection(LABEL_PERSON, 0.9, 100, 50, 300, 500),
        Detection(LABEL_HARDHAT, 0.80, 110, 55, 290, 230),
        Detection(LABEL_SAFETY_VEST, 0.85, 110, 150, 290, 350),
    ]


@pytest.fixture()
def sample_image() -> np.ndarray:
    return np.full((640, 640, 3), 200, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_safe_person_annotated_green(self, sample_image, safe_detections):
        results = assess_compliance(safe_detections)
        assert len(results) == 1
        assert results[0].is_safe is True

        annotated = annotate(sample_image, results)
        assert annotated.shape == sample_image.shape
        b, g, r = annotated[200, 100]
        assert g > r and g > b

    def test_output_written_to_disk(self, tmp_path, sample_image, safe_detections):
        results = assess_compliance(safe_detections)
        annotated = annotate(sample_image, results)
        out_path = tmp_path / "result.jpg"
        save_image(annotated, out_path)
        assert out_path.exists()
        assert out_path.stat().st_size > 0

    def test_no_persons_returns_unannotated_copy(self, sample_image):
        results = assess_compliance([])
        annotated = annotate(sample_image, results)
        np.testing.assert_array_equal(annotated, sample_image)

    def test_no_hardhat_signal_makes_unsafe(self, sample_image):
        detections = [
            Detection(LABEL_PERSON, 0.9, 100, 50, 300, 500),
            Detection(LABEL_HARDHAT, 0.80, 110, 55, 290, 230),
            Detection(LABEL_NO_HARDHAT, 0.75, 110, 55, 290, 230),
        ]
        results = assess_compliance(detections)
        assert results[0].is_safe is False


# ---------------------------------------------------------------------------
# Flask route tests
# ---------------------------------------------------------------------------

def _make_jpeg_bytes() -> bytes:
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


@pytest.fixture()
def flask_client(tmp_path):
    import src.main as main_module
    with (
        patch.object(main_module, "UPLOAD_DIR", tmp_path / "uploads"),
        patch.object(main_module, "DEAD_LETTER_DIR", tmp_path / "failed"),
        patch.object(main_module, "_model_loaded", True),
    ):
        app = main_module.app
        app.config["TESTING"] = True
        with app.test_client() as client:
            yield client


class TestHealthRoute:
    def test_returns_200(self, flask_client):
        assert flask_client.get("/health").status_code == 200

    def test_json_keys(self, flask_client):
        data = flask_client.get("/health").get_json()
        assert data["status"] == "ok"
        assert "model_loaded" in data


class TestIndexRoute:
    def test_returns_html(self, flask_client):
        resp = flask_client.get("/")
        assert resp.status_code == 200
        assert b"<form" in resp.data


class TestDetectRoute:
    def test_no_image_field_returns_400(self, flask_client):
        assert flask_client.post("/detect").status_code == 400

    def test_valid_jpeg_safe_person(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            Detection(LABEL_PERSON, 0.9, 10, 10, 80, 90),
            Detection(LABEL_HARDHAT, 0.80, 12, 12, 78, 50),
        ]
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        assert resp.content_type == "image/jpeg"

    def test_json_param_returns_json(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect?json=1",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "status" in data
        assert data["status"] == "No person detected"


# ---------------------------------------------------------------------------
# /detect/full route tests
# ---------------------------------------------------------------------------

class TestDetectFullRoute:
    def test_no_image_returns_400(self, flask_client):
        assert flask_client.post("/detect/full").status_code == 400

    def test_no_person_response(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["Status"] == "No person detected"
        assert data["Alarm"] is False
        assert data["PPE-missing"] == []
        assert data["persons_detected"] == 0
        assert "image_base64" in data

    def test_safe_person_response(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            Detection(LABEL_PERSON, 0.9, 10, 10, 80, 90),
            Detection(LABEL_HARDHAT, 0.80, 12, 12, 78, 50),
            Detection(LABEL_SAFETY_VEST, 0.85, 12, 40, 78, 80),
        ]
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["Status"] == "Safe"
        assert data["Alarm"] is False
        assert data["PPE-missing"] == []
        assert data["persons_detected"] == 1
        assert data["safe"] == 1

    def test_unsafe_no_helmet_response(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            Detection(LABEL_PERSON, 0.9, 10, 10, 80, 90),
            Detection(LABEL_SAFETY_VEST, 0.85, 12, 40, 78, 80),
            # No hardhat
        ]
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["Status"] == "Unsafe: No Helmet"
        assert data["Alarm"] is True
        assert data["PPE-missing"] == ["Hard Hat"]

    def test_unsafe_no_vest_response(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            Detection(LABEL_PERSON, 0.9, 10, 10, 80, 90),
            Detection(LABEL_HARDHAT, 0.80, 12, 12, 78, 50),
            # No vest
        ]
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["Status"] == "Unsafe: No Vest"
        assert data["Alarm"] is True
        assert data["PPE-missing"] == ["Safety Vest"]

    def test_unsafe_both_missing_response(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = [
            Detection(LABEL_PERSON, 0.9, 10, 10, 80, 90),
            # No hardhat, no vest
        ]
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        data = resp.get_json()
        assert data["Status"] == "Unsafe: No Helmet & Vest"
        assert data["Alarm"] is True
        assert "Hard Hat" in data["PPE-missing"]
        assert "Safety Vest" in data["PPE-missing"]

    def test_image_base64_is_valid_jpeg(self, flask_client):
        jpeg = _make_jpeg_bytes()
        mock_det = MagicMock()
        mock_det.detect.return_value = []
        import src.main as main_module
        with patch.object(main_module, "_get_detector", return_value=mock_det):
            resp = flask_client.post(
                "/detect/full",
                data={"image": (io.BytesIO(jpeg), "test.jpg")},
                content_type="multipart/form-data",
            )
        import base64
        data = resp.get_json()
        decoded = base64.b64decode(data["image_base64"])
        assert decoded[:3] == b"\xff\xd8\xff"  # JPEG magic bytes
