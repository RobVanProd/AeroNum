#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch reference for AeroNum tiny LM training.")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main():
    args = parse_args()
    vocab = 16
    dim = 8
    learning_rate = 0.05
    sequence = torch.tensor([(i * 5 + 3) % vocab for i in range(64)], dtype=torch.long)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    embeddings = torch.tensor(
        [((i % 7) - 3.0) / 7.0 for i in range(vocab * dim)],
        dtype=torch.float32,
        device=device,
    ).reshape(vocab, dim)
    projection = torch.tensor(
        [((i % 11) - 5.0) / 20.0 for i in range(dim * vocab)],
        dtype=torch.float32,
        device=device,
    ).reshape(dim, vocab)
    sequence = sequence.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    first_loss = None
    last_loss = None

    for run in range(args.runs):
        for epoch in range(args.epochs):
            epoch_loss = 0.0
            for pos in range(sequence.numel()):
                token = int(sequence[pos].item())
                prev = int(sequence[pos - 1].item()) if pos > 0 else int(sequence[-1].item())
                target = int(sequence[(pos + 1) % sequence.numel()].item())
                hidden = 0.75 * embeddings[token] + 0.25 * embeddings[prev]
                logits = hidden @ projection
                probs = torch.softmax(logits, dim=0)
                epoch_loss += float((-torch.log(probs[target].clamp_min(1e-9))).item())

                grad = probs.clone()
                grad[target] -= 1.0
                projection -= learning_rate * torch.outer(hidden, grad)

            mean_loss = epoch_loss / sequence.numel()
            if run == 0 and epoch == 0:
                first_loss = mean_loss
            last_loss = mean_loss

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    total_tokens = args.epochs * sequence.numel() * args.runs
    result = {
        "benchmark": "pytorch_tiny_lm_train_reference",
        "device": str(device),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "epochs": args.epochs,
        "runs": args.runs,
        "vocab": vocab,
        "dim": dim,
        "sequence_len": int(sequence.numel()),
        "total_tokens": int(total_tokens),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_decreased": last_loss < first_loss,
        "elapsed_seconds": elapsed,
        "tokens_per_second": total_tokens / max(elapsed, 1e-9),
        "training_scope": "pytorch_reference_for_explicit_gradient_tiny_language_model_not_gpt2",
    }
    print(json.dumps(result, sort_keys=True))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
