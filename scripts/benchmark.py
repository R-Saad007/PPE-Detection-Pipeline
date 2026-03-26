"""Latency and throughput benchmark for the YOLOv8m inference pipeline.

Usage:
    python scripts/benchmark.py --images tests/fixtures/ --iterations 100
"""

from __future__ import annotations

import argparse
import logging
import statistics
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark PPE inference latency."
    )
    parser.add_argument(
        "--images",
        required=True,
        type=Path,
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--iterations",
        default=100,
        type=int,
        help="Total number of inference calls (default: 100).",
    )
    parser.add_argument(
        "--warmup",
        default=5,
        type=int,
        help="Number of warm-up iterations excluded from timing (default: 5).",
    )
    return parser.parse_args()


def run_benchmark(image_dir: Path, iterations: int, warmup: int) -> None:
    """Load all images from *image_dir*, run the detector, and report timing.

    Args:
        image_dir: Directory with test images.
        iterations: Total number of detect() calls (warm-up included).
        warmup: First *warmup* iterations excluded from reported statistics.

    Raises:
        SystemExit: If no images are found or the detector cannot be initialised.
    """
    # Import here so benchmark errors are clear
    try:
        from src.detector import PPEDetector
        from src.utils.image_io import load_image
    except ImportError as exc:
        logger.error("Cannot import pipeline: %s", exc)
        sys.exit(1)

    images_paths = sorted(image_dir.glob("*.jpg")) + sorted(image_dir.glob("*.png"))
    if not images_paths:
        logger.error("No .jpg/.png images found in %s", image_dir)
        sys.exit(1)

    logger.info("Loading detector…")
    try:
        detector = PPEDetector()
    except Exception as exc:
        logger.error("Detector init failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Benchmarking %d iterations (%d warm-up) over %d image(s)…",
        iterations,
        warmup,
        len(images_paths),
    )

    latencies: list[float] = []
    for i in range(iterations):
        img_path = images_paths[i % len(images_paths)]
        try:
            img = load_image(img_path)
        except Exception as exc:
            logger.warning("Skipping %s: %s", img_path.name, exc)
            continue

        t0 = time.perf_counter()
        detector.detect(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        if i >= warmup:
            latencies.append(elapsed_ms)

    if not latencies:
        logger.error("No valid timing data collected.")
        sys.exit(1)

    logger.info("--- Benchmark Results ---")
    logger.info("Samples      : %d", len(latencies))
    logger.info("Mean (ms)    : %.2f", statistics.mean(latencies))
    logger.info("Median (ms)  : %.2f", statistics.median(latencies))
    logger.info("Stdev (ms)   : %.2f", statistics.stdev(latencies) if len(latencies) > 1 else 0)
    logger.info("Min (ms)     : %.2f", min(latencies))
    logger.info("Max (ms)     : %.2f", max(latencies))
    logger.info(
        "Throughput   : %.1f img/s",
        1000.0 / statistics.mean(latencies),
    )


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_benchmark(args.images, args.iterations, args.warmup)


if __name__ == "__main__":
    main()
