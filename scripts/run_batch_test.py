"""Batch test script — Phase 1 validation.

Walks a directory of camera images (``2025/``), runs YOLOv8m PPE inference
on every JPEG, and writes annotated copies to a mirrored output directory.

Usage
-----
    python scripts/run_batch_test.py --input 2025/ --output outputs/
    python scripts/run_batch_test.py --input 2025/ --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# Ensure project root is on the path when called directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.logging_config import configure_logging
from config.settings import DEAD_LETTER_DIR
from src.annotator import annotate
from src.compliance import assess_compliance
from src.detector import PPEDetector
from src.utils.image_io import load_image, save_image

configure_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------

class _Tally:
    processed: int = 0
    safe: int = 0
    unsafe: int = 0
    no_person: int = 0
    failed: int = 0


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _validate_jpeg(path: Path) -> bool:
    """Return True if *path* starts with JPEG magic bytes (FFD8FF)."""
    try:
        with path.open("rb") as fh:
            return fh.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def _build_output_path(
    image_path: Path,
    input_root: Path,
    output_root: Path,
) -> Path:
    """Mirror the source directory structure under *output_root*.

    Appends ``_annotated`` before the file extension to avoid collisions.

    Args:
        image_path: Absolute path to the source image.
        input_root: Root of the input directory tree.
        output_root: Root of the output directory tree.

    Returns:
        Destination :class:`~pathlib.Path` for the annotated image.
    """
    rel = image_path.relative_to(input_root)
    return output_root / rel.parent / (rel.stem + "_annotated" + rel.suffix)


def _summarise(
    rel_path: Path,
    results: list,
) -> tuple[str, str]:
    """Build the one-line stdout summary and status string for an image.

    Args:
        rel_path: Path relative to the input root (for display).
        results: Compliance results list from ``assess_compliance()``.

    Returns:
        Tuple of (status_string, summary_line).
    """
    if not results:
        return "No person detected", f"[OK]  {rel_path} → No person detected"

    safe_count = sum(1 for r in results if r.is_safe)
    unsafe_count = len(results) - safe_count
    total = len(results)

    if unsafe_count == 0:
        status = "Safe"
        detail = f"{total} person{'s' if total != 1 else ''}"
        line = f"[OK]  {rel_path} → Safe ({detail})"
    else:
        status = "Unsafe - PPE Hazard"
        line = (
            f"[OK]  {rel_path} → Unsafe - PPE Hazard "
            f"({unsafe_count} of {total} person{'s' if total != 1 else ''} non-compliant)"
        )
    return status, line


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def run_batch(
    input_dir: Path,
    output_dir: Path,
    dry_run: bool = False,
) -> _Tally:
    """Walk *input_dir*, run inference, write annotated outputs.

    Args:
        input_dir: Root directory containing JPEG images (recursed).
        output_dir: Root directory for annotated output images.
        dry_run: If ``True``, list images without running inference.

    Returns:
        :class:`_Tally` with final counts.
    """
    tally = _Tally()

    # Collect all JPEG files (case-insensitive)
    all_jpgs = sorted(
        p for p in input_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg")
    )

    if not all_jpgs:
        logger.warning("No JPEG files found under %s", input_dir)
        return tally

    if dry_run:
        print(f"Dry run — {len(all_jpgs)} image(s) found under {input_dir}:")
        for p in all_jpgs:
            print(f"  {p.relative_to(input_dir)}")
        return tally

    # Load model once
    logger.info("Loading YOLOv8m model…")
    detector = PPEDetector()

    dead_letter = DEAD_LETTER_DIR
    dead_letter.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    for img_path in all_jpgs:
        rel = img_path.relative_to(input_dir)

        # Magic-byte check
        if not _validate_jpeg(img_path):
            logger.warning("[SKIP] Not a valid JPEG: %s", rel)
            continue

        try:
            img = load_image(img_path)
            detections = detector.detect(img)
            results = assess_compliance(detections)

            # Batch output: person box + compliance label only (no PPE boxes)
            annotated = annotate(img, results, draw_ppe_boxes=False)

            out_path = _build_output_path(img_path, input_dir, output_dir)
            save_image(annotated, out_path)

            status, summary = _summarise(rel, results)
            print(summary)
            tally.processed += 1

            if not results:
                tally.no_person += 1
            elif all(r.is_safe for r in results):
                tally.safe += 1
            else:
                tally.unsafe += 1

            del img

        except Exception:
            logger.exception("[FAIL] Error processing %s", rel)
            try:
                shutil.copy2(str(img_path), dead_letter / img_path.name)
                logger.warning("Copied failed image to dead-letter: %s", img_path.name)
            except Exception:
                logger.exception("Could not copy to dead-letter: %s", img_path.name)
            tally.failed += 1

    return tally


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run YOLOv8m PPE batch inference on a directory of JPEG images."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("2025"),
        help="Root directory of source images (default: 2025/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs"),
        help="Root directory for annotated outputs (default: outputs/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List images that would be processed without running inference.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for the batch test script."""
    args = _parse_args(argv)

    input_dir = args.input.resolve()
    output_dir = args.output.resolve()

    if not input_dir.is_dir():
        print(f"ERROR: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    if args.dry_run:
        print("Mode:   dry-run")
    print()

    tally = run_batch(input_dir, output_dir, dry_run=args.dry_run)

    if not args.dry_run:
        print()
        print(
            f"Processed: {tally.processed} | "
            f"Safe: {tally.safe} | "
            f"Unsafe: {tally.unsafe} | "
            f"No person: {tally.no_person} | "
            f"Failed: {tally.failed}"
        )


if __name__ == "__main__":
    main()
