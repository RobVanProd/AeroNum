import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Config, GPT2LMHeadModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run a PyTorch/Hugging Face GPT-2 training reference.")
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--embedding-dim", type=int, default=384)
    parser.add_argument("--positions", type=int, default=512)
    parser.add_argument("--vocab-size", type=int, default=50257)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.0003)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output", default="", help="Optional JSON result output path")
    return parser.parse_args()


def run_hf_baseline(args):
    print("============================================================================")
    print("PyTorch/Hugging Face GPT-2 Training Reference")
    print("============================================================================")
    print(
        "Initializing GPT-2 layout "
        f"({args.layers} layers, {args.heads} heads, {args.embedding_dim} dim)..."
    )

    configuration = GPT2Config(
        vocab_size=args.vocab_size,
        n_positions=args.positions,
        n_ctx=args.positions,
        n_embd=args.embedding_dim,
        n_layer=args.layers,
        n_head=args.heads,
    )

    model = GPT2LMHeadModel(configuration)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA/HIP device requested but torch.cuda.is_available() is false")

    print(f"Hardware Context Check: [{device.type.upper()} Layout]")
    model.to(device)

    X = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)
    y = torch.randint(0, args.vocab_size, (args.batch_size, args.seq_len), device=device)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    print("Deploying Execution Constraints...")
    start_time = time.time()

    last_loss = None
    for _epoch in range(args.epochs):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x, labels=batch_y)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            last_loss = float(loss.detach().cpu().item())

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    end_time = time.time()

    total_tokens = args.epochs * args.batch_size * args.seq_len
    total_time = end_time - start_time
    tps = total_tokens / total_time

    result = {
        "benchmark": "pytorch_hf_gpt2_training_reference",
        "device": str(device),
        "torch": torch.__version__,
        "torch_hip": getattr(torch.version, "hip", None),
        "cuda_available": torch.cuda.is_available(),
        "layers": args.layers,
        "heads": args.heads,
        "embedding_dim": args.embedding_dim,
        "positions": args.positions,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "epochs": args.epochs,
        "total_tokens": total_tokens,
        "total_time_seconds": total_time,
        "tokens_per_second": tps,
        "last_loss": last_loss,
    }

    print(f"PyTorch Epoch Execution Block: {total_time:.2f} seconds")
    print(f"PyTorch Base Throughput limits: {tps:.2f} Tokens / Sec")
    print(json.dumps(result, sort_keys=True))

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")

    return result

if __name__ == "__main__":
    run_hf_baseline(parse_args())
