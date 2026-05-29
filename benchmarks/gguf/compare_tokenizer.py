#!/usr/bin/env python3
"""Compare AeroNum GGUF byte-BPE tokenization with llama.cpp token IDs."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path


def run_command(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def parse_llama_ids(stdout: str) -> list[int]:
    value = ast.literal_eval(stdout.strip())
    if not isinstance(value, list) or not all(isinstance(item, int) for item in value):
        raise ValueError(f"unexpected llama-tokenize output: {stdout!r}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare AeroNum GGUF tokenizer output against llama.cpp."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--llama-tokenize",
        default="/home/rob/llama.cpp/build-gpu/bin/llama-tokenize",
    )
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    model_path = str(Path(args.model).resolve())
    llama_tokenize = str(Path(args.llama_tokenize).resolve())

    aeronum_command = [
        "cargo",
        "run",
        "-q",
        "-p",
        "aeronum-core",
        "--example",
        "gguf_tokenizer_compare",
        "--",
        "--model",
        model_path,
    ]
    aeronum = run_command(aeronum_command, repo_root)
    if aeronum.returncode != 0:
        print(aeronum.stderr, file=sys.stderr)
        return aeronum.returncode

    aeronum_result = json.loads(aeronum.stdout)
    comparisons = []
    llama_commands = []
    for check in aeronum_result["checks"]:
        text = check["text"]
        has_special = check.get("has_special", False)
        with_bos_command = [
            llama_tokenize,
            "-m",
            model_path,
            "--ids",
            "--log-disable",
            "-p",
            text,
        ]
        without_bos_command = [
            llama_tokenize,
            "-m",
            model_path,
            "--ids",
            "--log-disable",
            "--no-bos",
            "-p",
            text,
        ]
        no_parse_with_bos_command = [
            llama_tokenize,
            "-m",
            model_path,
            "--ids",
            "--log-disable",
            "--no-parse-special",
            "-p",
            text,
        ]
        no_parse_without_bos_command = [
            llama_tokenize,
            "-m",
            model_path,
            "--ids",
            "--log-disable",
            "--no-bos",
            "--no-parse-special",
            "-p",
            text,
        ]
        with_bos = run_command(with_bos_command, repo_root)
        without_bos = run_command(without_bos_command, repo_root)
        no_parse_with_bos = None if has_special else run_command(no_parse_with_bos_command, repo_root)
        no_parse_without_bos = (
            None if has_special else run_command(no_parse_without_bos_command, repo_root)
        )
        llama_commands.extend([with_bos_command, without_bos_command])
        if not has_special:
            llama_commands.extend([no_parse_with_bos_command, no_parse_without_bos_command])

        if (
            with_bos.returncode != 0
            or without_bos.returncode != 0
            or (no_parse_with_bos is not None and no_parse_with_bos.returncode != 0)
            or (no_parse_without_bos is not None and no_parse_without_bos.returncode != 0)
        ):
            comparisons.append(
                {
                    "label": check["label"],
                    "text": text,
                    "has_special": has_special,
                    "with_bos": {
                        "aeronum_ids": check["with_bos"],
                        "llama_cpp_ids": None,
                        "match": False,
                        "exit_code": with_bos.returncode,
                        "stderr": with_bos.stderr,
                    },
                    "without_bos": {
                        "aeronum_ids": check["without_bos"],
                        "llama_cpp_ids": None,
                        "match": False,
                        "exit_code": without_bos.returncode,
                        "stderr": without_bos.stderr,
                    },
                    "no_parse_with_bos": {
                        "aeronum_ids": check["no_parse_with_bos"],
                        "llama_cpp_ids": None,
                        "match": False,
                        "exit_code": None if no_parse_with_bos is None else no_parse_with_bos.returncode,
                        "stderr": None if no_parse_with_bos is None else no_parse_with_bos.stderr,
                    },
                    "no_parse_without_bos": {
                        "aeronum_ids": check["no_parse_without_bos"],
                        "llama_cpp_ids": None,
                        "match": False,
                        "exit_code": None
                        if no_parse_without_bos is None
                        else no_parse_without_bos.returncode,
                        "stderr": None
                        if no_parse_without_bos is None
                        else no_parse_without_bos.stderr,
                    },
                }
            )
            continue

        llama_with_bos = parse_llama_ids(with_bos.stdout)
        llama_without_bos = parse_llama_ids(without_bos.stdout)
        llama_no_parse_with_bos = (
            None if no_parse_with_bos is None else parse_llama_ids(no_parse_with_bos.stdout)
        )
        llama_no_parse_without_bos = (
            None
            if no_parse_without_bos is None
            else parse_llama_ids(no_parse_without_bos.stdout)
        )
        comparisons.append(
            {
                "label": check["label"],
                "text": text,
                "has_special": has_special,
                "with_bos": {
                    "aeronum_ids": check["with_bos"],
                    "llama_cpp_ids": llama_with_bos,
                    "match": check["with_bos"] == llama_with_bos,
                    "token_count": len(check["with_bos"]),
                },
                "without_bos": {
                    "aeronum_ids": check["without_bos"],
                    "llama_cpp_ids": llama_without_bos,
                    "match": check["without_bos"] == llama_without_bos,
                    "token_count": len(check["without_bos"]),
                },
                "no_parse_with_bos": {
                    "aeronum_ids": check["no_parse_with_bos"],
                    "llama_cpp_ids": llama_no_parse_with_bos,
                    "match": None
                    if has_special
                    else check["no_parse_with_bos"] == llama_no_parse_with_bos,
                    "token_count": len(check["no_parse_with_bos"]),
                    "skipped": has_special,
                },
                "no_parse_without_bos": {
                    "aeronum_ids": check["no_parse_without_bos"],
                    "llama_cpp_ids": llama_no_parse_without_bos,
                    "match": None
                    if has_special
                    else check["no_parse_without_bos"] == llama_no_parse_without_bos,
                    "token_count": len(check["no_parse_without_bos"]),
                    "skipped": has_special,
                },
            }
        )

    all_match = all(
        item["with_bos"]["match"]
        and item["without_bos"]["match"]
        and (
            item["has_special"]
            or (
                item["no_parse_with_bos"]["match"]
                and item["no_parse_without_bos"]["match"]
            )
        )
        for item in comparisons
    )
    comparison_count = sum(2 if item["has_special"] else 4 for item in comparisons)
    output = {
        "benchmark": "aeronum_core_gguf_tokenizer_llama_cpp_compare",
        "model_path": model_path,
        "llama_tokenize": llama_tokenize,
        "aeronum_command": aeronum_command,
        "llama_command_count": len(llama_commands),
        "tokenizer_model": aeronum_result["tokenizer_model"],
        "tokenizer_pre": aeronum_result["tokenizer_pre"],
        "token_count": aeronum_result["token_count"],
        "merge_count": aeronum_result["merge_count"],
        "prompt_count": len(comparisons),
        "comparison_count": comparison_count,
        "special_prompt_count": sum(1 for item in comparisons if item["has_special"]),
        "all_match": all_match,
        "comparisons": comparisons,
        "limitations": [
            "fixed prompt set only",
            "does not verify exhaustive special-token parsing",
            "does not verify literal no-parse-special parity for prompts containing special tokens",
            "does not verify GGUF token inference throughput",
        ],
    }
    print(json.dumps(output, indent=2))
    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())
