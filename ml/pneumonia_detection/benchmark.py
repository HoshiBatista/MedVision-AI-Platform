"""
Pneumonia Detection — Inference benchmark: latency & throughput.

Usage:
    python benchmark.py --weights best.pt
    python benchmark.py --weights best.pt --runs 200 --batch 4
    python benchmark.py --weights best.onnx --onnx
"""

import argparse
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import numpy as np

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "ml"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True)
    p.add_argument("--imgsz",   type=int, default=640)
    p.add_argument("--runs",    type=int, default=100)
    p.add_argument("--warmup",  type=int, default=10)
    p.add_argument("--batch",   type=int, default=1)
    p.add_argument("--onnx",    action="store_true")
    return p.parse_args()


def benchmark_pytorch(weights: str, imgsz: int, batch: int, runs: int, warmup: int) -> None:
    from ultralytics import YOLO
    model  = YOLO(weights)
    dummy  = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    source = [dummy] * batch

    print(f"Warming up ({warmup} runs)...")
    for _ in range(warmup):
        model.predict(source, device="mps", verbose=False)

    print(f"Benchmarking ({runs} runs, batch={batch})...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(source, device="mps", verbose=False)
        latencies.append((time.perf_counter() - t0) * 1000)

    _print_stats(latencies, batch)


def benchmark_onnx(weights: str, imgsz: int, batch: int, runs: int, warmup: int) -> None:
    import onnxruntime as ort
    sess     = ort.InferenceSession(weights, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    dummy    = np.random.rand(batch, 3, imgsz, imgsz).astype(np.float32)

    print(f"Warming up ONNX ({warmup} runs)...")
    for _ in range(warmup):
        sess.run(None, {inp_name: dummy})

    print(f"Benchmarking ONNX ({runs} runs, batch={batch})...")
    latencies = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, {inp_name: dummy})
        latencies.append((time.perf_counter() - t0) * 1000)

    _print_stats(latencies, batch)


def _print_stats(latencies: list[float], batch: int) -> None:
    fps = 1000.0 / mean(latencies) * batch
    print("\n─── Pneumonia Detection Benchmark ───────────────────────")
    print(f"  Runs       : {len(latencies)}")
    print(f"  Batch size : {batch}")
    print(f"  Mean       : {mean(latencies):.2f} ms")
    print(f"  Median     : {median(latencies):.2f} ms")
    print(f"  Std        : {stdev(latencies):.2f} ms")
    print(f"  P95        : {sorted(latencies)[int(len(latencies)*0.95)]:.2f} ms")
    print(f"  P99        : {sorted(latencies)[int(len(latencies)*0.99)]:.2f} ms")
    print(f"  Min        : {min(latencies):.2f} ms")
    print(f"  Max        : {max(latencies):.2f} ms")
    print(f"  Throughput : {fps:.1f} img/s")
    print("─────────────────────────────────��───────────────────────")


if __name__ == "__main__":
    args = parse_args()
    if args.onnx:
        benchmark_onnx(args.weights, args.imgsz, args.batch, args.runs, args.warmup)
    else:
        benchmark_pytorch(args.weights, args.imgsz, args.batch, args.runs, args.warmup)
