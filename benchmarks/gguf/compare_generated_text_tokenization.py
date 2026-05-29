#!/usr/bin/env python3
"""Verify generated text by tokenizing it with llama.cpp and AeroNum result IDs."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
from pathlib import Path


def parse_llama_ids(stdout: str) -> list[int]:
    value = ast.literal_eval(stdout.strip())
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"unexpected llama-tokenize output: {stdout!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare generated text tokenization against generated token IDs."
    )
    parser.add_argument("--input", required=True, help="AeroNum autoregressive result JSON")
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument(
        "--llama-tokenize",
        default="/home/rob/llama.cpp/build-gpu/bin/llama-tokenize",
        help="Path to llama-tokenize",
    )
    parser.add_argument("--output", required=True, help="Comparison result JSON")
    args = parser.parse_args()

    result = json.loads(Path(args.input).read_text())
    generated_text = result["generated_text"]
    generated_token_ids = [int(value) for value in result["generated_token_ids"]]
    llama_cmd = [
        args.llama_tokenize,
        "--model",
        args.model,
        "--no-bos",
        "--ids",
        "--prompt",
        generated_text,
        "--log-disable",
    ]
    completed = subprocess.run(
        llama_cmd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    llama_ids = parse_llama_ids(completed.stdout) if completed.returncode == 0 else []
    passed = completed.returncode == 0 and llama_ids == generated_token_ids
    output = {
        "comparison": "generated_text_tokenization",
        "input": args.input,
        "model": args.model,
        "llama_tokenize": args.llama_tokenize,
        "llama_tokenize_returncode": completed.returncode,
        "llama_tokenize_stderr": completed.stderr,
        "generated_text": generated_text,
        "generated_token_ids": generated_token_ids,
        "llama_token_ids": llama_ids,
        "passed": passed,
        "limitations": [
            "verifies generated text re-tokenizes to generated token IDs",
            "not a llama.cpp detokenization trace",
            "not sampled decoding",
            "not KV-cache decoding",
            "not AeroNum-native GGUF token inference throughput",
        ],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
