#!/usr/bin/env python3
"""Check whether the visible ROCm devices are suitable for a local NCCL DDP run."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight local ROCm/NCCL topology for DDP.")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--device-ids", default="0,1")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def run_text(command: list[str]) -> str:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, errors="replace", timeout=20)
    except Exception as exc:  # noqa: BLE001 - diagnostic tool should capture failures.
        return f"{type(exc).__name__}: {exc}"
    return (proc.stdout + "\n" + proc.stderr).strip()


def classify_device(name: str) -> str:
    lower = name.lower()
    if "graphics" in lower and "radeon rx" not in lower:
        return "integrated"
    if "radeon rx" in lower:
        return "discrete"
    return "unknown"


def main() -> int:
    args = parse_args()
    requested_ids = [int(item) for item in args.device_ids.split(",") if item.strip()]
    cmdline = Path("/proc/cmdline").read_text(encoding="utf-8").strip()
    devices = []

    for idx in range(torch.cuda.device_count() if torch.cuda.is_available() else 0):
        props = torch.cuda.get_device_properties(idx)
        name = torch.cuda.get_device_name(idx)
        devices.append(
            {
                "id": idx,
                "name": name,
                "class": classify_device(name),
                "gcn_arch": f"{props.major}.{props.minor}",
                "total_memory_bytes": props.total_memory,
            }
        )

    selected = [device for device in devices if device["id"] in requested_ids[: args.world_size]]
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("torch.cuda.is_available() is false")
    if len(devices) < args.world_size:
        reasons.append(f"only {len(devices)} ROCm-visible device(s), need {args.world_size}")
    if len(set(requested_ids[: args.world_size])) < args.world_size:
        reasons.append("requested NCCL ranks include duplicate device IDs")
    if len(selected) < args.world_size:
        reasons.append("requested device IDs are not all visible")
    if any(device["class"] != "discrete" for device in selected):
        reasons.append("requested NCCL ranks include a non-discrete/integrated GPU")
    if len({device["gcn_arch"] for device in selected}) > 1:
        reasons.append("requested NCCL ranks span different ROCm GPU architectures")
    if args.world_size > 1 and "iommu=pt" not in cmdline.split():
        reasons.append("kernel command line does not include iommu=pt")

    result = {
        "benchmark": "aeronum_nccl_topology_preflight",
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "world_size": args.world_size,
        "requested_device_ids": requested_ids,
        "visible_devices": devices,
        "selected_devices": selected,
        "kernel_cmdline": cmdline,
        "rocminfo_selected": run_text(["rocminfo"]),
        "rocm_smi_product": run_text(["rocm-smi", "--showdriverversion", "--showproductname", "--showuniqueid"]),
        "compatible_for_requested_nccl": len(reasons) == 0,
        "blocking_reasons": reasons,
    }

    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0 if result["compatible_for_requested_nccl"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
