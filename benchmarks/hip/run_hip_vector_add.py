#!/usr/bin/env python3
"""Compile and run the HIP vector-add benchmark for ROCm."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run HIP vector-add benchmark")
    parser.add_argument("--size", type=int, default=1 << 24, help="Element count (float32)")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup runs")
    parser.add_argument("--block-size", type=int, default=256, help="Kernel block size")
    parser.add_argument("--arch", default="gfx1101", help="GPU target arch (default: gfx1101)")
    parser.add_argument("--skip-build", action="store_true", help="Skip compiling benchmark")
    parser.add_argument("--output", default="", help="Optional explicit JSON output path")
    return parser.parse_args()


def find_hipcc() -> str | None:
    env_path = os.environ.get("HIPCC")
    if env_path and Path(env_path).exists():
        return env_path

    resolved = shutil.which("hipcc")
    if resolved:
        return resolved

    candidates = [
        Path(r"C:\Program Files\AMD\ROCm\7.2\bin\hipcc.exe"),
        Path(r"C:\Program Files\AMD\ROCm\7.1\bin\hipcc.exe"),
        Path(r"C:\Program Files\AMD\ROCm\bin\hipcc.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    src = repo_root / "benchmarks" / "hip" / "vector_add.hip.cpp"
    out_dir = repo_root / "benchmarks" / "results" / "hip"
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = out_dir / "vector_add_hip.exe"

    hipcc = find_hipcc()
    if not hipcc:
        print("hipcc not found. Install ROCm HIP SDK or set HIPCC to hipcc.exe", file=sys.stderr)
        return 2

    if not args.skip_build:
        cmd = [
            hipcc,
            str(src),
            "-O3",
            f"--offload-arch={args.arch}",
            "-o",
            str(exe),
        ]
        print("Compiling:", " ".join(cmd))
        compile_proc = subprocess.run(cmd, capture_output=True, text=True)
        if compile_proc.returncode != 0:
            print("hipcc compilation failed", file=sys.stderr)
            print(compile_proc.stdout)
            print(compile_proc.stderr, file=sys.stderr)
            return compile_proc.returncode

    if not exe.exists():
        print(f"Compiled benchmark not found: {exe}", file=sys.stderr)
        return 3

    run_cmd = [
        str(exe),
        "--size",
        str(args.size),
        "--runs",
        str(args.runs),
        "--warmup",
        str(args.warmup),
        "--block-size",
        str(args.block_size),
    ]
    print("Running:", " ".join(run_cmd))
    proc = subprocess.run(run_cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        return proc.returncode

    stdout_lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        print("Benchmark produced no output", file=sys.stderr)
        return 4

    json_line = stdout_lines[-1]
    try:
        payload = json.loads(json_line)
    except json.JSONDecodeError:
        print(proc.stdout)
        print("Failed to parse benchmark JSON payload", file=sys.stderr)
        return 5

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path(args.output) if args.output else out_dir / f"hip_vector_add_{timestamp}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(proc.stdout)
    print(f"Saved result: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
