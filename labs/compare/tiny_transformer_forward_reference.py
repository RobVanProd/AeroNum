#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch reference for AeroNum tiny transformer forward.")
    parser.add_argument("--repeats", type=int, default=1000)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def tiny_transformer_forward(device: torch.device):
    seq_len = 4
    dim = 8
    vocab = 16
    tokens = torch.tensor([3, 8, 13, 2], dtype=torch.long, device=device)
    targets = torch.tensor([8, 13, 2, 7], dtype=torch.long, device=device)

    embeddings = torch.tensor(
        [((i % 17) - 8.0) / 11.0 for i in range(vocab * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(vocab, dim)
    positions = torch.tensor(
        [((i % 13) - 6.0) / 17.0 for i in range(seq_len * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(seq_len, dim)
    wq = torch.tensor(
        [((i % 19) - 9.0) / 23.0 for i in range(dim * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, dim)
    wk = torch.tensor(
        [((i * 3 % 23) - 11.0) / 29.0 for i in range(dim * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, dim)
    wv = torch.tensor(
        [((i * 5 % 29) - 14.0) / 31.0 for i in range(dim * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, dim)
    wo = torch.tensor(
        [((i * 7 % 31) - 15.0) / 37.0 for i in range(dim * vocab)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, vocab)

    inputs = embeddings[tokens] + positions
    q = inputs @ wq
    k = inputs @ wk
    v = inputs @ wv
    scores = q @ k.T / (dim**0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    context = weights @ v
    logits = context @ wo
    loss = torch.nn.functional.cross_entropy(logits, targets)
    checksum = (logits * torch.arange(1, vocab + 1, dtype=torch.float32, device=device)).sum()
    return loss, checksum


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but torch.cuda.is_available() is false")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    loss = None
    checksum = None
    for _ in range(args.repeats):
        loss, checksum = tiny_transformer_forward(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    tokens_processed = args.repeats * 4
    result = {
        "benchmark": "pytorch_tiny_transformer_forward_reference",
        "training_scope": "pytorch_reference_for_causal_self_attention_lm_forward_not_gpt2_training",
        "device": str(device),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "repeats": args.repeats,
        "sequence_len": 4,
        "dim": 8,
        "vocab": 16,
        "tokens_processed": tokens_processed,
        "loss": float(loss.detach().cpu().item()),
        "checksum": float(checksum.detach().cpu().item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": tokens_processed / max(elapsed, 1e-9),
    }
    print(json.dumps(result, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
