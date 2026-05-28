# AeroNum

AeroNum is a numerical computing and deep-learning experiment library for the
[Aero programming language](https://github.com/RobVanProd/Aero). The repository
contains Aero array APIs, AeroNN examples, ROCm/HIP runtime experiments,
benchmark scripts, and historical benchmark artifacts.

## Runtime ROCm Status

Current repo contents include:

- `aeronum-core` GPU runtime metadata and HIP buffer plumbing. The HIP runtime
  loader now supports Linux `libamdhip64.so` in addition to Windows
  `amdhip64.dll`.
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
- `aeronum-core` HIP runtime tests passed on Linux/ROCm. The new
  `runtime_can_roundtrip_device_copy_when_available` test created a HIP runtime
  on device 0 and round-tripped a float32 host buffer through device memory;
  the filtered GPU test set passed 3/3 tests, and the filtered core matmul test
  set passed 5/5 tests. The full `aeronum-core` suite passed 31/31 tests
  ([result JSON](claim-verification/results/aeronum_core_linux_hip_runtime_7900xtx_20260528T231000Z/claim_result.json)).
- `aeronum-core` now includes a minimal hipBLAS SGEMM bridge and release example
  benchmark. The repo-owned command
  `cargo run --release -p aeronum-core --example hip_sgemm_4096 -- --n 4096 --runs 10 --warmup 3`
  passed on the Radeon RX 7900 XTX with median 4.950619 ms and 27.761973 TFLOP/s.
  This verifies an AeroNum core HIP/hipBLAS 4096x4096 matmul path, not a
  speedup versus another framework
  ([result JSON](claim-verification/results/aeronum_core_hipblas_sgemm_4096_7900xtx_20260528T232000Z/claim_result.json)).
- `labs/compare/sgemm_4096_compare.py` now runs a same-run 4096x4096 float32
  all-ones comparison between AeroNum core hipBLAS SGEMM and PyTorch ROCm
  `torch.matmul`. On the Radeon RX 7900 XTX with 10 measured runs, AeroNum
  median was 4.955240 ms and PyTorch median was 4.981353 ms, a 1.005270x
  median-time ratio. This is a near-parity same-run measurement, not a broader
  speedup claim
  ([result JSON](claim-verification/results/aeronum_sgemm_vs_pytorch_4096_7900xtx_20260528T231630Z/claim_result.json)).
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
- `core/examples/tiny_lm_train.rs` now runs an AeroNum-owned explicit-gradient
  tiny language-model training loop, with a matching PyTorch ROCm reference in
  `labs/compare/tiny_lm_train_reference.py`. Both commands trained 25,600
  tokens and reduced loss from about 2.75995 to 0.516646. The release AeroNum
  example reported 790,368.788238 tokens/s; the PyTorch ROCm reference reported
  8,339.328453 tokens/s. This is a tiny training analogue, not GPT-2 and not an
  AeroNum-vs-PyTorch GPT-2 speedup
  ([result JSON](claim-verification/results/aeronum_tiny_lm_train_7900xtx_20260528T231229Z/claim_result.json)).
- `labs/compare/aeronn_gpu_compare.py` measured a PyTorch reference 4096x4096
  matmul on the same machine: CPU 0.1620 s, GPU 0.0067 s, relative speedup
  24.28x. This is a PyTorch CPU-vs-GPU reference only, not an AeroNum matmul
  result
  ([raw log](claim-verification/results/aeronum_pytorch_matmul_reference_7900xtx_20260528T191500Z/aeronn_gpu_compare.stdout.log)).
- `labs/compare/distributed_compare.py` now runs a real PyTorch DDP smoke
  benchmark instead of a simulated message. NCCL world size 1 passed on GPU 0
  with 3 steps, mean rank time 0.3393106461 s, and 565.853156 tokens/s
  ([result JSON](claim-verification/results/aeronum_nccl_ddp_single_gpu_7900xtx_20260528T224500Z/claim_result.json)).
- `labs/compare/nccl_topology_preflight.py` now gates local multi-device NCCL
  runs before launching DDP. On this machine, the requested two-device topology
  is blocked because device 1 is integrated AMD Radeon Graphics, selected
  devices span ROCm architectures 11.0 and 10.3, and the kernel command line
  does not include `iommu=pt`. A fresh NCCL world-size-1 DDP smoke still passed
  on device 0 with 562.417411 tokens/s
  ([result JSON](claim-verification/results/aeronum_nccl_preflight_7900xtx_20260528T231927Z/claim_result.json)).
- `aeronum-core` now parses GGUF metadata and tensor directory records before
  constructing a `LlamaModel`. The repo-owned command
  `cargo run -p aeronum-core --example gguf_header_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --device rocm --max-tokens 16 --prompt "AeroNum GGUF directory smoke prompt"`
  passed against the local Mistral GGUF file, SHA-256
  `c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0`, and
  reported GGUF version 3, 363 parsed tensor infos, 45 parsed metadata entries,
  architecture `llama`, and quantization version `2`. This is a
  directory/load smoke result with placeholder generation, not real GGUF token
  inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_directory_smoke_7900xtx_20260528T232438Z/claim_result.json)).
- `benchmarks/gguf/run_llama_cpp_cli.py` ran a real local llama.cpp CLI ROCm
  GGUF inference reference on the same Mistral GGUF file. The llama.cpp build
  reported version 7074 (`22e1ce2f8`) with HIP 6.2.41133-dd7f95766, offloaded
  41/41 layers to ROCm0 Radeon RX 7900 XTX, and measured 125.22 prompt eval
  tokens/s plus 44.58 eval tokens/s for 16 predicted tokens. This is a
  llama.cpp reference benchmark through an AeroNum repo wrapper, not
  AeroNum-native GGUF tensor execution
  ([result JSON](claim-verification/results/aeronum_llama_cpp_cli_gguf_7900xtx_20260528T230730Z/claim_result.json)).
- After updating the vendored compiler from Aero `0.1.0` to `1.0.0`, the
  repo-local command `./aero-compiler/aero run` executed matrix/arithmetic Aero
  examples that previously hit the old compiler's binary-expression failure:
  `examples/aero/minimal_aeronum.aero` reported exit code 60,
  `labs/benchmarks/aero/real_matrix_operations.aero` reported exit code 234,
  and `tests/aero/test_matrix_operations.aero` reported exit code 42
  ([result JSON](claim-verification/results/aeronum_aero_compiler_v1_matrix_examples_7900xtx_20260528T225200Z/claim_result.json)).
- `benches/matmul.aero` is now a compiler-compatible 2x2 integer matmul smoke
  benchmark. It executed with the repo-local Aero 1.0.0 compiler and reported
  checksum exit code 134
  ([result JSON](claim-verification/results/aeronum_matmul_smoke_aero_compiler_v1_7900xtx_20260528T225800Z/claim_result.json)).
- `benchmarks/run_benchmarks.sh` completed on the rebased commit, but redirects
  command output to `/dev/null` and does not emit fresh raw timings
  ([raw log](claim-verification/results/aeronum_runner_b727dfb_7900xtx_20260528T192000Z/run_benchmarks.stdout.log)).

Blocked or omitted claims:

- GPT-2 training vs PyTorch is omitted because no current AeroNum-vs-PyTorch
  GPT-2 training result was produced. The current
  `labs/compare/transformer_compare.py` result is a PyTorch/Hugging Face
  reference only. The current AeroNum-owned training result is a tiny
  explicit-gradient language-model analogue, not GPT-2.
- Broad GPU 4096x4096 speedup claims are omitted. The verified current
  same-run AeroNum core hipBLAS vs PyTorch ROCm measurement is near parity
  with a 1.005270x median-time ratio on an all-ones SGEMM workload.
- NCCL/MPI multi-GPU scaling is omitted. A real NCCL/DDP single-GPU smoke test
  passed, but the local two-device attempts using the Radeon RX 7900 XTX plus
  integrated AMD Radeon Graphics failed. After reboot, the default
  heterogeneous run failed with RCCL `hipIpcGetMemHandle failed: invalid argument`.
  A `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` attempt failed with rank 1
  `invalid device function`. A two-rank single-XTX attempt was rejected by RCCL
  as duplicate GPU usage. The current preflight also blocks the requested
  two-device run because device 1 is integrated, the selected devices span ROCm
  architectures 11.0 and 10.3, and the kernel command line lacks `iommu=pt`.
  No compatible second discrete ROCm GPU was verified on this machine
  ([debug result JSON](claim-verification/results/aeronum_nccl_debug_20260528T225759Z/claim_result.json)).
- AeroNum-native GGUF token-inference throughput claims are omitted. The
  verified current AeroNum core result parses local GGUF metadata and tensor
  directory records and reaches placeholder generation, and the verified
  token-inference result is a llama.cpp reference through an AeroNum repo
  wrapper.

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
