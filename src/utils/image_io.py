"""Safe image loading and saving with validation."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from config.settings import (
    ALLOWED_EXTENSIONS,
    IMAGE_MAGIC_BYTES,
    MAX_IMAGE_SIZE_BYTES,
    UPLOAD_DIR,
)

logger = logging.getLogger(__name__)


def _check_magic_bytes(path: Path) -> bool:
    """Return True if the file's leading bytes match a known image format."""
    try:
        with path.open("rb") as fh:
            header = fh.read(12)
    except OSError:
        return False

    for magic in IMAGE_MAGIC_BYTES:
        if header.startswith(magic):
            return True
    return False


def validate_image_path(raw_path: str | Path, watch_dir: Path | None = None) -> Path:
    """Resolve, sanitize, and validate an image path from an untrusted source.

    Args:
        raw_path: Path string received from the upload handler or watcher.
        watch_dir: Trusted root directory; defaults to ``UPLOAD_DIR``.

    Returns:
        Resolved :class:`~pathlib.Path` that is safe to open.

    Raises:
        ValueError: If the path escapes *watch_dir*, has a disallowed
            extension, or fails magic-byte validation.
        FileNotFoundError: If the file does not exist.
        OSError: If the file exceeds the maximum allowed size.
    """
    base = (watch_dir or UPLOAD_DIR).resolve()
    resolved = Path(raw_path).resolve()

    # Path-traversal guard
    try:
        resolved.relative_to(base)
    except ValueError:
        raise ValueError(f"Path escapes trusted directory: {resolved}") from None

    if not resolved.exists():
        raise FileNotFoundError(f"Image not found: {resolved}")

    # Extension check
    if resolved.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Disallowed file extension: {resolved.suffix!r}")

    # Size check
    size = resolved.stat().st_size
    if size > MAX_IMAGE_SIZE_BYTES:
        raise OSError(
            f"File too large ({size / 1_048_576:.1f} MB > "
            f"{MAX_IMAGE_SIZE_BYTES / 1_048_576:.1f} MB): {resolved.name}"
        )

    # Magic-bytes check
    if not _check_magic_bytes(resolved):
        raise ValueError(
            f"File does not have valid image magic bytes: {resolved.name}"
        )

    return resolved


def load_image(path: Path) -> np.ndarray:
    """Load an image as a BGR NumPy array.

    Args:
        path: Validated path to an image file.

    Returns:
        BGR ``uint8`` array of shape ``(H, W, 3)``.

    Raises:
        ValueError: If OpenCV cannot decode the file.
    """
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"OpenCV failed to decode image: {path.name}")
    return img


def save_image(img: np.ndarray, output_path: Path) -> None:
    """Write a BGR image to disk, creating parent directories as needed.

    Args:
        img: BGR ``uint8`` array.
        output_path: Destination file path.

    Raises:
        OSError: If the write fails.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(str(output_path), img)
    if not success:
        raise OSError(f"Failed to write image: {output_path}")
    logger.debug("Saved annotated image", extra={"output": output_path.name})
