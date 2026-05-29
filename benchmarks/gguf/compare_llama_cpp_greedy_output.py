#!/usr/bin/env python3
"""Compare AeroNum greedy GGUF output with llama.cpp greedy CLI output."""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import time
from pathlib import Path


def parse_llama_ids(stdout: str) -> list[int]:
    value = ast.literal_eval(stdout.strip())
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"unexpected llama-tokenize output: {stdout!r}")
    return value


def parse_layers(text: str) -> tuple[int | None, int | None]:
    match = re.search(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare AeroNum greedy GGUF generated text with llama.cpp CLI."
    )
    parser.add_argument("--input", required=True, help="AeroNum autoregressive result JSON")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument(
        "--llama-cli",
        default="/home/rob/llama.cpp/build-gpu/bin/llama-cli",
        help="Path to llama-cli",
    )
    parser.add_argument(
        "--llama-tokenize",
        default="/home/rob/llama.cpp/build-gpu/bin/llama-tokenize",
        help="Path to llama-tokenize",
    )
    parser.add_argument("--gpu-layers", type=int, default=999)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--split-mode", default="none")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--output", required=True, help="Comparison result JSON")
    args = parser.parse_args()

    aeronum = json.loads(Path(args.input).read_text(encoding="utf-8"))
    prompt = str(aeronum["prompt"])
    predict = int(aeronum["generated_token_count"])
    expected_text = str(aeronum["generated_text"])
    expected_ids = [int(value) for value in aeronum["generated_token_ids"]]

    llama_cmd = [
        args.llama_cli,
        "-m",
        args.model,
        "--override-kv",
        "tokenizer.ggml.add_bos_token=bool:false",
        "-p",
        prompt,
        "-n",
        str(predict),
        "-ngl",
        str(args.gpu_layers),
        "-sm",
        args.split_mode,
        "-mg",
        str(args.main_gpu),
        "--no-conversation",
        "--no-display-prompt",
        "--no-warmup",
        "--seed",
        "12345",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1.0",
        "--min-p",
        "0.0",
        "--repeat-penalty",
        "1.0",
        "--no-perf",
    ]
    start = time.perf_counter()
    llama = subprocess.run(
        llama_cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
        timeout=args.timeout,
    )
    wall_seconds = time.perf_counter() - start
    llama_stdout = llama.stdout
    llama_text = llama_stdout.rstrip("\n")

    tokenize_cmd = [
        args.llama_tokenize,
        "--model",
        args.model,
        "--no-bos",
        "--ids",
        "--prompt",
        llama_text,
        "--log-disable",
    ]
    tokenized = subprocess.run(
        tokenize_cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        errors="replace",
    )
    llama_ids = parse_llama_ids(tokenized.stdout) if tokenized.returncode == 0 else []
    combined = f"{llama.stdout}\n{llama.stderr}"
    offloaded_layers, total_layers = parse_layers(combined)
    passed = (
        llama.returncode == 0
        and tokenized.returncode == 0
        and llama_text == expected_text
        and llama_ids == expected_ids
    )

    result = {
        "comparison": "llama_cpp_greedy_generated_output",
        "input": args.input,
        "model": args.model,
        "prompt": prompt,
        "predict": predict,
        "expected_text": expected_text,
        "llama_text": llama_text,
        "expected_token_ids": expected_ids,
        "llama_token_ids": llama_ids,
        "passed": passed,
        "llama_cli_command": llama_cmd,
        "llama_cli_returncode": llama.returncode,
        "llama_cli_wall_seconds": wall_seconds,
        "llama_cli_generated_text_normalized": llama_text,
        "llama_cli_stdout": llama_stdout,
        "llama_cli_stderr": llama.stderr,
        "llama_tokenize_command": tokenize_cmd,
        "llama_tokenize_returncode": tokenized.returncode,
        "llama_tokenize_stdout": tokenized.stdout,
        "llama_tokenize_stderr": tokenized.stderr,
        "offloaded_layers": offloaded_layers,
        "total_layers": total_layers,
        "limitations": [
            "compares deterministic greedy generated text and re-tokenized IDs only",
            "uses llama.cpp CLI stdout rather than an internal detokenization trace",
            "not sampled-output parity",
            "not KV-cache parity",
            "not AeroNum-native GGUF token inference throughput",
        ],
    }
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
