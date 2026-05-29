#!/usr/bin/env python3
"""Compare AeroNum RoPE sample values against an independent Python reference."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def sample_key(sample: dict) -> tuple[int, int]:
    return int(sample["token_position"]), int(sample["value_index"])


def expected_rope_value(
    even_value: float,
    odd_value: float,
    value_index: int,
    token_position: int,
    head_dimension: int,
    rope_freq_base: float,
) -> tuple[float, float]:
    dim = value_index % head_dimension
    if dim % 2 != 0:
        dim -= 1
    angle = token_position / (rope_freq_base ** (dim / head_dimension))
    cos_value = math.cos(angle)
    sin_value = math.sin(angle)
    return (
        even_value * cos_value - odd_value * sin_value,
        even_value * sin_value + odd_value * cos_value,
    )


def compare_projection(
    pre_samples: list[dict],
    rope_samples: list[dict],
    head_dimension: int,
    rope_freq_base: float,
    tolerance: float,
) -> dict:
    pre_by_key = {sample_key(sample): float(sample["value"]) for sample in pre_samples}
    rope_by_key = {sample_key(sample): float(sample["value"]) for sample in rope_samples}
    compared = 0
    max_abs_diff = 0.0
    mismatches = []

    for token_position, value_index in sorted(rope_by_key):
        dim = value_index % head_dimension
        even_index = value_index - 1 if dim % 2 else value_index
        odd_index = even_index + 1
        if (token_position, even_index) not in pre_by_key:
            continue
        if (token_position, odd_index) not in pre_by_key:
            continue
        if (token_position, even_index) not in rope_by_key:
            continue
        if (token_position, odd_index) not in rope_by_key:
            continue

        expected_even, expected_odd = expected_rope_value(
            pre_by_key[(token_position, even_index)],
            pre_by_key[(token_position, odd_index)],
            even_index,
            token_position,
            head_dimension,
            rope_freq_base,
        )
        actual_even = rope_by_key[(token_position, even_index)]
        actual_odd = rope_by_key[(token_position, odd_index)]
        for current_index, actual, expected in (
            (even_index, actual_even, expected_even),
            (odd_index, actual_odd, expected_odd),
        ):
            abs_diff = abs(actual - expected)
            max_abs_diff = max(max_abs_diff, abs_diff)
            compared += 1
            if abs_diff > tolerance:
                mismatches.append(
                    {
                        "token_position": token_position,
                        "value_index": current_index,
                        "actual": actual,
                        "expected": expected,
                        "abs_diff": abs_diff,
                    }
                )

    return {
        "compared_values": compared,
        "max_abs_diff": max_abs_diff,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:10],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare GGUF multi-token attention RoPE samples with a Python reference."
    )
    parser.add_argument("--input", required=True, help="AeroNum attention result JSON")
    parser.add_argument("--output", required=True, help="Comparison result JSON")
    parser.add_argument("--tolerance", type=float, default=1.0e-5)
    args = parser.parse_args()

    result = json.loads(Path(args.input).read_text())
    sample = result["samples"][0]
    head_dimension = int(sample["head_dimension"])
    rope_freq_base = float(sample["rope_freq_base"])

    query = compare_projection(
        sample["query_projection_samples"],
        sample["rope_query_samples"],
        head_dimension,
        rope_freq_base,
        args.tolerance,
    )
    key = compare_projection(
        sample["key_projection_samples"],
        sample["rope_key_samples"],
        head_dimension,
        rope_freq_base,
        args.tolerance,
    )
    passed = (
        query["compared_values"] > 0
        and key["compared_values"] > 0
        and query["mismatch_count"] == 0
        and key["mismatch_count"] == 0
    )
    output = {
        "comparison": "gguf_rope_python_reference",
        "input": args.input,
        "tolerance": args.tolerance,
        "head_dimension": head_dimension,
        "rope_freq_base": rope_freq_base,
        "query": query,
        "key": key,
        "passed": passed,
        "limitations": [
            "sampled Q/K RoPE values only",
            "independent Python mathematical reference, not llama.cpp internal trace",
            "not full attention parity",
            "not generated-token logits",
            "not AeroNum-native GGUF token inference throughput",
        ],
    }
    Path(args.output).write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
