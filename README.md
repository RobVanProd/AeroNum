# AeroNum

AeroNum is a numerical computing and deep-learning experiment library for the
[Aero programming language](https://github.com/RobVanProd/Aero). The repository
contains Aero array APIs, AeroNN examples, ROCm/HIP runtime experiments,
benchmark scripts, and historical benchmark artifacts.

## Runtime ROCm Status

Current repo contents include:

- `aeronum-core` GPU runtime metadata and HIP buffer plumbing.
- A vendored Aero compiler at `aero-compiler/aero`. Current tracked binary:
  `Aero compiler version 1.0.0`, SHA-256
  `2bf72d1965f0d515428a570044da134cd382e91c9550ddd015ddc4a8b95a1b3e`.
- `aeronn::LlamaModel` device-offload paths for `rocm`, `gpu`, and `cuda`
  target strings.
- A HIP vector-add benchmark runner at
  `benchmarks/hip/run_hip_vector_add.py`.
- A HIP/hipBLAS SGEMM benchmark runner at
  `benchmarks/hip/run_hip_sgemm.py`.
- Distributed-training and NCCL/MPI files under `labs/`, currently represented
  as Aero source/blueprints rather than a verified multi-GPU benchmark run.

## Verified Results

Verification artifacts are tracked under
[`claim-verification/`](claim-verification/). The latest local verification was
run on 2026-05-28 after reboot on this hardware:

- CPU: AMD Ryzen 9 9950X 16-Core Processor
- GPU 0: Radeon RX 7900 XTX (`gfx1100`, PCI device `1002:744c`)
- GPU 1: AMD Radeon Graphics (`gfx1036`, PCI device `1002:13c0`)
- PyTorch: `2.5.1+rocm6.2`
- HIP: `6.2.41133-dd7f95766`

Verified current results:

- HIP vector add passed on the Radeon RX 7900 XTX with 16,777,216 float32
  elements, 20 measured runs, median 0.259509 ms, 64.649967 GFLOP/s, and
  775.799606 GB/s
  ([result JSON](claim-verification/results/aeronum_hip_vector_add_7900xtx_20260528T191500Z/hip_vector_add_result.json)).
- HIP/hipBLAS SGEMM passed on the Radeon RX 7900 XTX for 4096x4096 float32
  matrices with 10 measured runs, median 4.953900 ms, and 27.743587 TFLOP/s.
  This is a ROCm library reference benchmark, not an AeroNum-language matmul
  speedup claim
  ([result JSON](claim-verification/results/aeronum_hip_sgemm_4096_7900xtx_20260528T222200Z/claim_result.json)).
- `labs/compare/transformer_compare.py` now reports a PyTorch/Hugging Face
  GPT-2 training reference without AeroNum speedup claims. The 6-layer,
  6-head, 384-dim run completed on the Radeon RX 7900 XTX with 24,576 total
  tokens in 0.7213051319 s, or 34,071.572366 tokens/s
  ([result JSON](claim-verification/results/aeronum_pytorch_gpt2_reference_7900xtx_20260528T224500Z/claim_result.json)).
- `labs/compare/aeronn_gpu_compare.py` measured a PyTorch reference 4096x4096
  matmul on the same machine: CPU 0.1620 s, GPU 0.0067 s, relative speedup
  24.28x. This is a PyTorch CPU-vs-GPU reference only, not an AeroNum matmul
  result
  ([raw log](claim-verification/results/aeronum_pytorch_matmul_reference_7900xtx_20260528T191500Z/aeronn_gpu_compare.stdout.log)).
- `labs/compare/distributed_compare.py` now runs a real PyTorch DDP smoke
  benchmark instead of a simulated message. NCCL world size 1 passed on GPU 0
  with 3 steps, mean rank time 0.3393106461 s, and 565.853156 tokens/s
  ([result JSON](claim-verification/results/aeronum_nccl_ddp_single_gpu_7900xtx_20260528T224500Z/claim_result.json)).
- After updating the vendored compiler from Aero `0.1.0` to `1.0.0`, the
  repo-local command `./aero-compiler/aero run` executed matrix/arithmetic Aero
  examples that previously hit the old compiler's binary-expression failure:
  `examples/aero/minimal_aeronum.aero` reported exit code 60,
  `labs/benchmarks/aero/real_matrix_operations.aero` reported exit code 234,
  and `tests/aero/test_matrix_operations.aero` reported exit code 42
  ([result JSON](claim-verification/results/aeronum_aero_compiler_v1_matrix_examples_7900xtx_20260528T225200Z/claim_result.json)).
- `benchmarks/run_benchmarks.sh` completed on the rebased commit, but redirects
  command output to `/dev/null` and does not emit fresh raw timings
  ([raw log](claim-verification/results/aeronum_runner_b727dfb_7900xtx_20260528T192000Z/run_benchmarks.stdout.log)).

Blocked or omitted claims:

- GPT-2 training vs PyTorch is omitted because no current AeroNum-vs-PyTorch
  GPT-2 training result was produced. The current
  `labs/compare/transformer_compare.py` result is a PyTorch/Hugging Face
  reference only.
- GPU 4096x4096 AeroNum-language matmul speedup is omitted.
  `benches/core_ops.aero` contains simulated timings, and
  `benches/matmul.aero` still does not run with the Aero 1.0.0 compiler because
  the source uses language/library constructs this compiler does not parse.
  Smaller matrix/arithmetic Aero examples now execute successfully with the
  updated vendored compiler
  ([failed attempt](claim-verification/results/aeronum_matmul_b727dfb_7900xtx_20260528T192000Z/)).
- NCCL/MPI multi-GPU scaling is omitted. A real NCCL/DDP single-GPU smoke test
  passed, but the local two-device attempt using the Radeon RX 7900 XTX plus
  integrated AMD Radeon Graphics failed with RCCL `hipIpcGetMemHandle failed:
  invalid argument`; with `NCCL_P2P_DISABLE=1`, initialization progressed
  further but rank 1 failed with `invalid device function`
  ([failed attempt](claim-verification/results/aeronum_nccl_ddp_two_device_p2p_disabled_attempt_7900xtx_20260528T225400Z/)).
- GGUF/inference benchmark claims are omitted because no current raw GGUF
  inference benchmark result was found or rerun.

Historical benchmark CSVs remain in the repo, but README claims above only use
fresh local reruns and captured artifacts.

## Quick Start

```bash
git clone https://github.com/RobVanProd/AeroNum.git
cd AeroNum
```

Run the verified HIP vector-add benchmark on a ROCm-visible 7900 XTX:

```bash
python3 benchmarks/hip/run_hip_vector_add.py --arch gfx1100 --size 16777216 --runs 20 --warmup 5
```

Run the verified HIP/hipBLAS SGEMM reference benchmark:

```bash
python3 benchmarks/hip/run_hip_sgemm.py --arch gfx1100 --n 4096 --runs 10 --warmup 3
```

Run the generic benchmark harness:

```bash
./benchmarks/run_benchmarks.sh
```

## Repository Layout

```
AeroNum/
├── src/                      # Core Aero library files
├── core/                     # Rust core support
├── aeronum-python/           # Python packaging/bindings scaffold
├── examples/                 # Aero examples
├── benches/                  # Aero benchmark sources
├── benchmarks/               # Benchmark scripts and outputs
├── labs/                     # Experimental AI/GPU/distributed code
└── claim-verification/       # Verification manifests and raw rerun artifacts
```

## License

MIT © RobVanProd and contributors.
