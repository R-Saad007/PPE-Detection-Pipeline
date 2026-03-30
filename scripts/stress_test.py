"""Stress test: measure RAM and CPU utilization across pipeline states.

Measures three states:
1. Idle — models loaded, no inference running
2. Single image processing
3. Batch image processing (multiple images sequentially)

Reports RSS (MB) and CPU usage (%) for devops sizing decisions.

Usage:
    python scripts/stress_test.py --images scripts/2025/ --batch-size 10
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import psutil


def get_process_metrics(proc: psutil.Process) -> dict:
    """Snapshot current RSS (MB) and CPU (%) for the process."""
    mem = proc.memory_info()
    # cpu_percent since last call (interval=None for non-blocking)
    cpu = proc.cpu_percent(interval=None)
    return {
        "rss_mb": mem.rss / (1024 * 1024),
        "cpu_pct": cpu,
    }


def measure_cpu_over(proc: psutil.Process, duration: float, samples: int = 20) -> list[float]:
    """Sample CPU % over a duration, return list of readings."""
    interval = duration / samples
    readings = []
    for _ in range(samples):
        readings.append(proc.cpu_percent(interval=interval))
    return readings


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test: RAM & CPU profiling")
    parser.add_argument("--images", type=Path, default=Path("scripts/2025/"),
                        help="Directory containing test images")
    parser.add_argument("--batch-size", type=int, default=10,
                        help="Number of images for batch test (default: 10)")
    args = parser.parse_args()

    proc = psutil.Process(os.getpid())
    cpu_count = psutil.cpu_count(logical=True)

    # --- Baseline before model load ---
    proc.cpu_percent(interval=None)  # prime the counter
    baseline = get_process_metrics(proc)
    print(f"=== BASELINE (before model load) ===")
    print(f"  RSS: {baseline['rss_mb']:.1f} MB")
    print()

    # --- Load models ---
    print("Loading models...")
    from src.detector import PPEDetector
    from src.utils.image_io import load_image

    t0 = time.perf_counter()
    detector = PPEDetector()
    load_time = time.perf_counter() - t0
    print(f"  Model load time: {load_time:.1f}s")
    print()

    # ===================================================================
    # STATE 1: IDLE (models loaded, no inference)
    # ===================================================================
    print("=== STATE 1: IDLE (models loaded, no inference) ===")
    time.sleep(1)  # let things settle
    proc.cpu_percent(interval=None)  # reset counter

    # Sample CPU over 5 seconds while idle
    idle_cpu_readings = measure_cpu_over(proc, duration=5.0, samples=10)
    idle_metrics = get_process_metrics(proc)

    idle_cpu_avg = sum(idle_cpu_readings) / len(idle_cpu_readings)
    # Convert CPU % (which is per-core %) to approximate millicores
    # psutil reports CPU% where 100% = 1 full core
    idle_millicores = (idle_cpu_avg / 100.0) * 1000

    print(f"  RSS:          {idle_metrics['rss_mb']:.1f} MB")
    print(f"  CPU avg:      {idle_cpu_avg:.1f}% ({idle_millicores:.0f} millicores)")
    print(f"  CPU readings: {[f'{r:.1f}%' for r in idle_cpu_readings]}")
    print()

    # --- Collect images ---
    all_imgs = sorted(
        p for p in args.images.rglob("*")
        if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    if not all_imgs:
        print(f"ERROR: No images found in {args.images}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(all_imgs)} images in {args.images}")
    print()

    # ===================================================================
    # STATE 2: SINGLE IMAGE PROCESSING
    # ===================================================================
    print("=== STATE 2: SINGLE IMAGE PROCESSING ===")
    test_img_path = all_imgs[0]
    print(f"  Image: {test_img_path.name}")

    # Warm up (first inference is slower)
    img = load_image(test_img_path)
    detector.detect(img)

    # Measure single image processing (run 3 times, take averages)
    single_rss_readings = []
    single_cpu_readings = []
    single_latencies = []

    for i in range(3):
        img = load_image(test_img_path)
        proc.cpu_percent(interval=None)  # reset

        t0 = time.perf_counter()
        detector.detect(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        single_latencies.append(elapsed_ms)

        # Get CPU for the inference period
        cpu_after = proc.cpu_percent(interval=None)
        mem = proc.memory_info()
        single_rss_readings.append(mem.rss / (1024 * 1024))
        single_cpu_readings.append(cpu_after)
        del img

    # Also measure CPU during a single inference with proper interval sampling
    img = load_image(test_img_path)
    proc.cpu_percent(interval=None)  # reset
    t0 = time.perf_counter()
    detector.detect(img)
    elapsed = time.perf_counter() - t0
    single_sustained_cpu = proc.cpu_percent(interval=None)
    del img

    avg_latency = sum(single_latencies) / len(single_latencies)
    peak_rss = max(single_rss_readings)
    avg_rss = sum(single_rss_readings) / len(single_rss_readings)
    single_millicores = (single_sustained_cpu / 100.0) * 1000

    print(f"  Avg latency:    {avg_latency:.0f} ms")
    print(f"  Avg RSS:        {avg_rss:.1f} MB")
    print(f"  Peak RSS:       {peak_rss:.1f} MB")
    print(f"  CPU (sustained):{single_sustained_cpu:.1f}% ({single_millicores:.0f} millicores)")
    print()

    # ===================================================================
    # STATE 3: BATCH IMAGE PROCESSING
    # ===================================================================
    batch_size = min(args.batch_size, len(all_imgs))
    print(f"=== STATE 3: BATCH IMAGE PROCESSING ({batch_size} images) ===")

    batch_imgs = all_imgs[:batch_size]
    batch_rss_readings = []
    batch_cpu_readings = []
    batch_latencies = []

    proc.cpu_percent(interval=None)  # reset
    batch_start = time.perf_counter()

    for img_path in batch_imgs:
        img = load_image(img_path)
        proc.cpu_percent(interval=None)  # reset per-image

        t0 = time.perf_counter()
        detector.detect(img)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        batch_latencies.append(elapsed_ms)

        cpu_reading = proc.cpu_percent(interval=None)
        mem = proc.memory_info()
        batch_rss_readings.append(mem.rss / (1024 * 1024))
        batch_cpu_readings.append(cpu_reading)
        del img

    batch_wall = time.perf_counter() - batch_start

    # Final sustained CPU measurement over a full batch run
    proc.cpu_percent(interval=None)  # reset
    for img_path in batch_imgs[:5]:  # re-run 5 for sustained measurement
        img = load_image(img_path)
        detector.detect(img)
        del img
    batch_sustained_cpu = proc.cpu_percent(interval=None)

    batch_avg_latency = sum(batch_latencies) / len(batch_latencies)
    batch_peak_rss = max(batch_rss_readings)
    batch_avg_rss = sum(batch_rss_readings) / len(batch_rss_readings)
    batch_min_rss = min(batch_rss_readings)
    batch_throughput = batch_size / batch_wall
    batch_avg_cpu = sum(batch_cpu_readings) / len(batch_cpu_readings) if batch_cpu_readings else 0
    batch_peak_cpu = max(batch_cpu_readings) if batch_cpu_readings else 0
    batch_sustained_millicores = (batch_sustained_cpu / 100.0) * 1000
    batch_avg_millicores = (batch_avg_cpu / 100.0) * 1000
    batch_peak_millicores = (batch_peak_cpu / 100.0) * 1000

    print(f"  Wall time:       {batch_wall:.1f}s")
    print(f"  Throughput:      {batch_throughput:.2f} img/sec")
    print(f"  Avg latency:     {batch_avg_latency:.0f} ms/image")
    print(f"  Min RSS:         {batch_min_rss:.1f} MB")
    print(f"  Avg RSS:         {batch_avg_rss:.1f} MB")
    print(f"  Peak RSS:        {batch_peak_rss:.1f} MB")
    print(f"  CPU sustained:   {batch_sustained_cpu:.1f}% ({batch_sustained_millicores:.0f} millicores)")
    print(f"  CPU avg:         {batch_avg_cpu:.1f}% ({batch_avg_millicores:.0f} millicores)")
    print(f"  CPU peak:        {batch_peak_cpu:.1f}% ({batch_peak_millicores:.0f} millicores)")
    print()

    # ===================================================================
    # SUMMARY TABLE
    # ===================================================================
    print("=" * 70)
    print("SUMMARY FOR DEVOPS SIZING")
    print("=" * 70)
    print(f"  CPU cores (logical): {cpu_count}")
    print(f"  System RAM total:    {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print()
    print(f"  {'State':<30} {'RSS (MB)':<15} {'CPU (millicores)':<20}")
    print(f"  {'-'*30} {'-'*15} {'-'*20}")
    print(f"  {'Idle (models loaded)':<30} {idle_metrics['rss_mb']:<15.0f} {idle_millicores:<20.0f}")
    print(f"  {'Single image processing':<30} {peak_rss:<15.0f} {single_millicores:<20.0f}")
    print(f"  {f'Batch ({batch_size} images)':<30} {batch_peak_rss:<15.0f} {batch_sustained_millicores:<20.0f}")
    print()
    print("NOTE: CPU millicores = (cpu_percent / 100) * 1000.")
    print("      1000 millicores = 1 full CPU core.")
    print(f"      psutil cpu% is relative to 1 core (max = {cpu_count * 100}% on this machine).")


if __name__ == "__main__":
    main()
