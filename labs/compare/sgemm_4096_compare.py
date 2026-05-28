#!/usr/bin/env python3
"""Same-run AeroNum core hipBLAS SGEMM vs PyTorch ROCm matmul comparison."""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare AeroNum hipBLAS SGEMM with PyTorch ROCm matmul.")
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def median_tflops(n: int, median_ms: float) -> float:
    flops = 2.0 * float(n) ** 3
    return (flops / (median_ms / 1000.0)) / 1e12


def summarize(values: list[float]) -> dict[str, float]:
    return {
        "mean_ms": float(statistics.mean(values)),
        "median_ms": float(statistics.median(values)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
    }


def run_aeronum(args: argparse.Namespace) -> dict:
    command = [
        "cargo",
        "run",
        "--release",
        "-p",
        "aeronum-core",
        "--example",
        "hip_sgemm_4096",
        "--",
        "--n",
        str(args.n),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
    ]
    proc = subprocess.run(command, text=True, capture_output=True, errors="replace")
    result = {
        "command": command,
        "exit_code": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        return result
    json_lines = [line for line in proc.stdout.splitlines() if line.startswith("{")]
    if not json_lines:
        result["parse_error"] = "no JSON line found in AeroNum output"
        return result
    result["metrics"] = json.loads(json_lines[-1])
    return result


def run_pytorch(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch CUDA/HIP device is not available")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    a = torch.ones((args.n, args.n), dtype=torch.float32, device=device)
    b = torch.ones((args.n, args.n), dtype=torch.float32, device=device)

    for _ in range(args.warmup):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    run_ms: list[float] = []
    for _ in range(args.runs):
        start = time.perf_counter()
        out = torch.matmul(a, b)
        torch.cuda.synchronize(device)
        run_ms.append((time.perf_counter() - start) * 1000.0)

    sample = out.flatten()[:: max(1, out.numel() // 4096)]
    valid = bool(torch.allclose(sample, torch.full_like(sample, float(args.n)), atol=1e-3, rtol=0.0))
    metrics = summarize(run_ms)
    metrics.update(
        {
            "backend": "pytorch_rocm",
            "kernel": "torch.matmul",
            "n": args.n,
            "runs": args.runs,
            "warmup": args.warmup,
            "median_tflops": median_tflops(args.n, metrics["median_ms"]),
            "validation": "sampled_all_ones_expected_n" if valid else "failed",
            "torch": torch.__version__,
            "torch_hip": getattr(torch.version, "hip", None),
            "device": torch.cuda.get_device_name(0),
        }
    )
    return {"metrics": metrics}


def main() -> int:
    args = parse_args()
    aeronum = run_aeronum(args)
    pytorch = run_pytorch(args)
    result = {
        "benchmark": "aeronum_core_hipblas_vs_pytorch_rocm_sgemm_4096",
        "n": args.n,
        "runs": args.runs,
        "warmup": args.warmup,
        "aeronum": aeronum,
        "pytorch": pytorch,
    }
    if "metrics" in aeronum and "metrics" in pytorch:
        aero_ms = aeronum["metrics"]["median_ms"]
        torch_ms = pytorch["metrics"]["median_ms"]
        result["median_ms_ratio_pytorch_over_aeronum"] = torch_ms / aero_ms
        result["aeronum_faster_than_pytorch"] = aero_ms < torch_ms

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if aeronum.get("exit_code") == 0 else int(aeronum.get("exit_code", 1))


if __name__ == "__main__":
    raise SystemExit(main())
