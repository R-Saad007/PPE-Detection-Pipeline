"""Tests for the FTP server-side watcher script."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

# Import from scripts — adjust path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ftp_watcher import FTPWatcher, build_output_path


def _fake_api_response(status="Safe", alarm=False, ppe_missing=None):
    """Build a mock /detect/full JSON response."""
    fake_jpeg = b"\xff\xd8\xff fake annotated jpeg"
    return {
        "Status": status,
        "Alarm": alarm,
        "PPE-missing": ppe_missing or [],
        "persons_detected": 1,
        "safe": 1 if not alarm else 0,
        "unsafe": 0 if not alarm else 1,
        "image_base64": base64.b64encode(fake_jpeg).decode("ascii"),
    }


# ---------------------------------------------------------------------------
# build_output_path
# ---------------------------------------------------------------------------

class TestBuildOutputPath:
    """Verify _ppestatus directory naming convention."""

    def test_standard_path(self, tmp_path: Path) -> None:
        watch = tmp_path / "IHS-LAG-1197A"
        img = watch / "2026" / "3" / "15" / "IHS-LAG-1197A__00__123.jpg"
        result = build_output_path(img, watch)
        expected = tmp_path / "IHS-LAG-1197A_ppestatus" / "2026" / "3" / "15" / "IHS-LAG-1197A__00__123.jpg"
        assert result == expected

    def test_single_level(self, tmp_path: Path) -> None:
        watch = tmp_path / "factory"
        img = watch / "image1.jpg"
        result = build_output_path(img, watch)
        expected = tmp_path / "factory_ppestatus" / "image1.jpg"
        assert result == expected

    def test_preserves_filename(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        img = watch / "2026" / "1" / "photo.png"
        result = build_output_path(img, watch)
        assert result.name == "photo.png"

    def test_preserves_subdirectory_structure(self, tmp_path: Path) -> None:
        watch = tmp_path / "cam"
        img = watch / "a" / "b" / "c" / "d.jpg"
        result = build_output_path(img, watch)
        expected = tmp_path / "cam_ppestatus" / "a" / "b" / "c" / "d.jpg"
        assert result == expected


# ---------------------------------------------------------------------------
# FTPWatcher._iter_images
# ---------------------------------------------------------------------------

class TestIterImages:
    """Verify image discovery and filtering."""

    def _make_watcher(self, watch_dir: Path) -> FTPWatcher:
        return FTPWatcher(watch_dir=watch_dir, api_url="http://test:5000/detect/full")

    def test_finds_jpg_files(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        (watch / "2026" / "3").mkdir(parents=True)
        (watch / "2026" / "3" / "img.jpg").write_bytes(b"\xff\xd8\xff")
        watcher = self._make_watcher(watch)
        images = watcher._iter_images()
        assert len(images) == 1
        assert images[0].name == "img.jpg"

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        (watch / "readme.txt").write_text("hello")
        (watch / "data.csv").write_text("1,2,3")
        watcher = self._make_watcher(watch)
        assert watcher._iter_images() == []

    def test_skips_ppestatus_directories(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        (watch / "2026").mkdir(parents=True)
        (watch / "2026" / "img.jpg").write_bytes(b"\xff\xd8\xff")
        ppestatus = watch / "_ppestatus" / "2026"
        ppestatus.mkdir(parents=True)
        (ppestatus / "img.jpg").write_bytes(b"\xff\xd8\xff")
        watcher = self._make_watcher(watch)
        images = watcher._iter_images()
        assert len(images) == 1

    def test_nonexistent_dir_returns_empty(self, tmp_path: Path) -> None:
        watcher = self._make_watcher(tmp_path / "nonexistent")
        assert watcher._iter_images() == []

    def test_multiple_extensions(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            (watch / f"img{ext}").write_bytes(b"x")
        watcher = self._make_watcher(watch)
        assert len(watcher._iter_images()) == 5


# ---------------------------------------------------------------------------
# FTPWatcher._init_seen_from_existing_outputs
# ---------------------------------------------------------------------------

class TestIdempotency:
    """Verify that already-processed images are skipped on restart."""

    def test_skips_images_with_existing_output(self, tmp_path: Path) -> None:
        watch = tmp_path / "IHS"
        (watch / "2026").mkdir(parents=True)
        img = watch / "2026" / "img.jpg"
        img.write_bytes(b"\xff\xd8\xff")

        out = tmp_path / "IHS_ppestatus" / "2026" / "img.jpg"
        out.parent.mkdir(parents=True)
        out.write_bytes(b"annotated")

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")
        watcher._init_seen_from_existing_outputs()
        assert img.resolve() in watcher._seen or img in watcher._seen

    def test_does_not_skip_unprocessed_images(self, tmp_path: Path) -> None:
        watch = tmp_path / "IHS"
        (watch / "2026").mkdir(parents=True)
        (watch / "2026" / "img.jpg").write_bytes(b"\xff\xd8\xff")

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")
        watcher._init_seen_from_existing_outputs()
        assert len(watcher._seen) == 0


# ---------------------------------------------------------------------------
# FTPWatcher.scan_once (integration with mocked API)
# ---------------------------------------------------------------------------

class TestScanOnce:
    """Verify the full scan → POST → save cycle."""

    def test_processes_new_image_and_saves_decoded(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        (watch / "2026").mkdir(parents=True)
        img = watch / "2026" / "test.jpg"
        img.write_bytes(b"\xff\xd8\xff fake jpeg data")

        api_data = _fake_api_response("Safe", False)

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_data

        with patch("scripts.ftp_watcher.requests.post", return_value=mock_resp):
            count = watcher.scan_once()

        assert count == 1
        out_path = tmp_path / "site_ppestatus" / "2026" / "test.jpg"
        assert out_path.exists()
        # Verify it was base64-decoded correctly
        assert out_path.read_bytes() == base64.b64decode(api_data["image_base64"])

    def test_skips_on_api_error(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        (watch / "test.jpg").write_bytes(b"\xff\xd8\xff")

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")

        mock_resp = MagicMock()
        mock_resp.status_code = 500
        mock_resp.text = "Internal Server Error"

        with patch("scripts.ftp_watcher.requests.post", return_value=mock_resp):
            count = watcher.scan_once()

        assert count == 0
        assert len(watcher._seen) == 0

    def test_no_reprocess_after_seen(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        (watch / "test.jpg").write_bytes(b"\xff\xd8\xff")

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = _fake_api_response()

        with patch("scripts.ftp_watcher.requests.post", return_value=mock_resp) as mock_post:
            watcher.scan_once()
            watcher.scan_once()  # second scan — should not POST again

        assert mock_post.call_count == 1

    def test_retries_on_network_error(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        (watch / "test.jpg").write_bytes(b"\xff\xd8\xff")

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")

        with patch(
            "scripts.ftp_watcher.requests.post",
            side_effect=requests.ConnectionError("refused"),
        ):
            count = watcher.scan_once()

        assert count == 0
        assert len(watcher._seen) == 0  # will retry next cycle

    def test_unsafe_response_logged_and_saved(self, tmp_path: Path) -> None:
        watch = tmp_path / "site"
        watch.mkdir()
        (watch / "test.jpg").write_bytes(b"\xff\xd8\xff")

        api_data = _fake_api_response(
            status="Unsafe: No Helmet",
            alarm=True,
            ppe_missing=["Hard Hat"],
        )

        watcher = FTPWatcher(watch_dir=watch, api_url="http://test:5000/detect/full")

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = api_data

        with patch("scripts.ftp_watcher.requests.post", return_value=mock_resp):
            count = watcher.scan_once()

        assert count == 1
        out_path = tmp_path / "site_ppestatus" / "test.jpg"
        assert out_path.exists()
