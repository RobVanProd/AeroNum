import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP


def classify_device(name):
    lower = name.lower()
    if "graphics" in lower and "radeon rx" not in lower:
        return "integrated"
    if "radeon rx" in lower:
        return "discrete"
    return "unknown"


def nccl_topology_check(args):
    devices = []
    if torch.cuda.is_available():
        for idx in range(torch.cuda.device_count()):
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

    selected_ids = args.device_ids[: args.world_size]
    selected = [device for device in devices if device["id"] in selected_ids]
    reasons = []
    if not torch.cuda.is_available():
        reasons.append("torch.cuda.is_available() is false")
    if len(devices) < args.world_size:
        reasons.append(f"only {len(devices)} ROCm-visible device(s), need {args.world_size}")
    if len(set(selected_ids)) < args.world_size:
        reasons.append("requested NCCL ranks include duplicate device IDs")
    if len(selected) < args.world_size:
        reasons.append("requested device IDs are not all visible")
    if args.world_size > 1 and any(device["class"] != "discrete" for device in selected):
        reasons.append("requested NCCL ranks include a non-discrete/integrated GPU")
    if args.world_size > 1 and len({device["gcn_arch"] for device in selected}) > 1:
        reasons.append("requested NCCL ranks span different ROCm GPU architectures")
    kernel_cmdline = Path("/proc/cmdline").read_text(encoding="utf-8").strip()
    if args.world_size > 1 and "iommu=pt" not in kernel_cmdline.split():
        reasons.append("kernel command line does not include iommu=pt")

    return {
        "benchmark": "pytorch_ddp_smoke_topology_preflight",
        "backend": args.backend,
        "world_size": args.world_size,
        "requested_device_ids": args.device_ids,
        "selected_device_ids": selected_ids,
        "visible_devices": devices,
        "selected_devices": selected,
        "kernel_cmdline": kernel_cmdline,
        "compatible_for_requested_nccl": len(reasons) == 0,
        "blocking_reasons": reasons,
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
    }


def setup(rank, world_size, backend, master_port):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(master_port)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class SimpleTransformerEngine(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, dim_feedforward, num_layers):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.encoder(x)
        return self.lm_head(x)

def demo_basic(rank, world_size, args, result_queue):
    device_id = args.device_ids[rank]
    device = torch.device(f"cuda:{device_id}" if args.backend == "nccl" else "cpu")
    setup(rank, world_size, args.backend, args.master_port)

    if device.type == "cuda":
        torch.cuda.set_device(device_id)

    model = SimpleTransformerEngine(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.layers,
    ).to(device)
    ddp_model = DDP(model, device_ids=[device_id] if device.type == "cuda" else None)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(ddp_model.parameters(), lr=args.learning_rate)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    last_loss = None
    for _step in range(args.steps):
        optimizer.zero_grad()

        dummy_inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
        dummy_labels = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

        outputs = ddp_model(dummy_inputs)
        loss = loss_fn(outputs.reshape(-1, args.vocab_size), dummy_labels.reshape(-1))

        loss.backward()
        optimizer.step()

        last_loss = float(loss.detach().cpu().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    metric = torch.tensor([elapsed, last_loss if last_loss is not None else 0.0], device=device)
    dist.all_reduce(metric, op=dist.ReduceOp.SUM)
    metric = metric / world_size

    if rank == 0:
        result_queue.put(
            {
                "benchmark": "pytorch_ddp_smoke",
                "backend": args.backend,
                "world_size": world_size,
                "device_ids": args.device_ids,
                "steps": args.steps,
                "batch_size": args.batch_size,
                "seq_len": args.seq_len,
                "vocab_size": args.vocab_size,
                "d_model": args.d_model,
                "layers": args.layers,
                "nhead": args.nhead,
                "mean_rank_seconds": float(metric[0].detach().cpu().item()),
                "mean_last_loss": float(metric[1].detach().cpu().item()),
                "tokens_per_second": (args.steps * args.batch_size * args.seq_len * world_size)
                / max(float(metric[0].detach().cpu().item()), 1e-9),
                "torch": torch.__version__,
                "torch_hip": getattr(torch.version, "hip", None),
            }
        )

    cleanup()


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PyTorch DDP smoke benchmark.")
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--device-ids", default="0", help="Comma-separated CUDA/HIP device ids")
    parser.add_argument("--master-port", type=int, default=12355)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument(
        "--skip-topology-check",
        action="store_true",
        help="Launch NCCL even if local ROCm topology preflight reports a known blocker.",
    )
    parser.add_argument("--output", default="", help="Optional JSON result output path")
    parsed = parser.parse_args()
    parsed.device_ids = [int(x) for x in parsed.device_ids.split(",") if x.strip()]
    if len(parsed.device_ids) < parsed.world_size:
        raise ValueError("--device-ids must include at least --world-size ids")
    if parsed.backend == "nccl" and not torch.cuda.is_available():
        raise RuntimeError("NCCL requested but torch.cuda.is_available() is false")
    if parsed.backend == "nccl" and torch.cuda.device_count() < parsed.world_size:
        raise RuntimeError("NCCL requested but not enough CUDA/HIP devices are visible")
    return parsed


def main():
    args = parse_args()
    print("============================================================================")
    print("PyTorch Reference: DistributedDataParallel Smoke Benchmark")
    print("============================================================================")
    print(f"Executing backend={args.backend} world_size={args.world_size} device_ids={args.device_ids}")

    if args.backend == "nccl" and not args.skip_topology_check:
        preflight = nccl_topology_check(args)
        if not preflight["compatible_for_requested_nccl"]:
            result = {
                "benchmark": "pytorch_ddp_smoke",
                "status": "blocked_by_topology_preflight",
                "launched_ddp": False,
                "preflight": preflight,
            }
            print(json.dumps(result, sort_keys=True))
            if args.output:
                Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
            return 1

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    mp.spawn(demo_basic, args=(args.world_size, args, result_queue), nprocs=args.world_size, join=True)
    result = result_queue.get(timeout=30)
    print(json.dumps(result, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main() or 0)
