#!/usr/bin/env python3
"""Run a local llama.cpp CLI GGUF inference benchmark and emit JSON metrics."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run llama.cpp CLI against a local GGUF model.")
    parser.add_argument("--llama-cli", default="/home/rob/llama.cpp/build-gpu/bin/llama-cli")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="AeroNum GGUF inference smoke")
    parser.add_argument("--predict", type=int, default=16)
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--split-mode", default="none")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def parse_rate(text: str, label: str) -> float | None:
    pattern = (
        rf"^llama_perf_context_print:\s*{re.escape(label)}\s*="
        rf"\s*[-+0-9.]+\s*ms\s*/.*?,\s*([-+0-9.]+)\s*tokens per second"
    )
    match = re.search(pattern, text, flags=re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_layers(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    args = parse_args()
    command = [
        args.llama_cli,
        "-m",
        args.model,
        "-p",
        args.prompt,
        "-n",
        str(args.predict),
        "-ngl",
        str(args.gpu_layers),
        "-sm",
        args.split_mode,
        "-mg",
        str(args.main_gpu),
        "--no-conversation",
        "--no-display-prompt",
    ]

    start = time.perf_counter()
    proc = subprocess.run(
        command,
        text=True,
        capture_output=True,
        errors="replace",
        timeout=args.timeout,
    )
    wall_seconds = time.perf_counter() - start
    combined = f"{proc.stdout}\n{proc.stderr}"
    offloaded_layers, total_layers = parse_layers(combined)

    result = {
        "benchmark": "aeronum_llama_cpp_cli_gguf_reference",
        "backend": "llama.cpp CLI",
        "command": command,
        "exit_code": proc.returncode,
        "wall_seconds": wall_seconds,
        "model": str(Path(args.model).resolve()),
        "prompt": args.prompt,
        "predict": args.predict,
        "gpu_layers_requested": args.gpu_layers,
        "main_gpu": args.main_gpu,
        "split_mode": args.split_mode,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "prompt_eval_tokens_per_second": parse_rate(combined, "prompt eval time"),
        "eval_tokens_per_second": parse_rate(combined, "eval time"),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
