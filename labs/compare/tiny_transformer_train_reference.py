#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch reference for AeroNum tiny transformer training.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def fixed_attention_context(device: torch.device):
    seq_len = 4
    dim = 8
    vocab = 16
    tokens = torch.tensor([3, 8, 13, 2], dtype=torch.long, device=device)
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

    inputs = embeddings[tokens] + positions
    q = inputs @ wq
    k = inputs @ wk
    v = inputs @ wv
    scores = q @ k.T / (dim**0.5)
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    scores = scores.masked_fill(mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return weights @ v


def train_epoch(context, projection, targets, learning_rate):
    loss = 0.0
    for row in range(targets.numel()):
        logits = context[row] @ projection
        probs = torch.softmax(logits, dim=0)
        loss += float((-torch.log(probs[targets[row]].clamp_min(1e-9))).item())
        grad = probs.clone()
        grad[targets[row]] -= 1.0
        projection -= learning_rate * torch.outer(context[row], grad)

    logits = context @ projection
    checksum = (logits.reshape(-1) * torch.arange(1, 65, dtype=torch.float32, device=context.device)).sum()
    return loss / targets.numel(), checksum


def main():
    args = parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but torch.cuda.is_available() is false")

    dim = 8
    vocab = 16
    learning_rate = 0.2
    targets = torch.tensor([8, 13, 2, 7], dtype=torch.long, device=device)
    context = fixed_attention_context(device)
    projection = torch.tensor(
        [((i * 7 % 31) - 15.0) / 37.0 for i in range(dim * vocab)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, vocab)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    first_loss = None
    last_loss = None
    checksum = None
    for run in range(args.runs):
        for epoch in range(args.epochs):
            loss, checksum = train_epoch(context, projection, targets, learning_rate)
            if run == 0 and epoch == 0:
                first_loss = loss
            last_loss = loss
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    total_tokens = args.epochs * args.runs * int(targets.numel())
    result = {
        "benchmark": "pytorch_tiny_transformer_train_reference",
        "training_scope": "pytorch_reference_for_causal_self_attention_lm_output_projection_training_not_gpt2",
        "device": str(device),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "epochs": args.epochs,
        "runs": args.runs,
        "sequence_len": int(targets.numel()),
        "dim": dim,
        "vocab": vocab,
        "total_tokens": total_tokens,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decreased": last_loss < first_loss,
        "checksum": float(checksum.detach().cpu().item()),
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
    }
    print(json.dumps(result, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
