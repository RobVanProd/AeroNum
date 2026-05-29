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
- `core/examples/tiny_transformer_forward.rs` and
  `labs/compare/tiny_transformer_forward_reference.py` ran a deterministic
  causal self-attention language-model forward pass and matching PyTorch ROCm
  reference with the same tensors. AeroNum reported loss `2.79602051`, PyTorch
  reported loss `2.79602027`, the absolute loss difference was `0.0000002406`,
  and the checksum absolute difference was `0.0000016723`. This is a tiny
  transformer-forward analogue, not GPT-2 training and not an
  AeroNum-vs-PyTorch GPT-2 speedup
  ([result JSON](claim-verification/results/aeronum_tiny_transformer_forward_7900xtx_20260528T233024Z/claim_result.json)).
- `core/examples/tiny_transformer_train.rs` and
  `labs/compare/tiny_transformer_train_reference.py` ran a deterministic
  causal self-attention language-model output-projection training analogue and
  matching PyTorch ROCm reference with the same tensors. AeroNum loss decreased
  from `2.79611373` to `0.28436375`; PyTorch loss decreased from `2.79611397`
  to `0.28436361`; the final-loss absolute difference was `0.0000001375`.
  This is a tiny transformer-training analogue, not GPT-2 training and not an
  AeroNum-vs-PyTorch GPT-2 speedup
  ([result JSON](claim-verification/results/aeronum_tiny_transformer_train_7900xtx_20260528T233801Z/claim_result.json)).
- `labs/compare/aeronn_gpu_compare.py` measured a PyTorch reference 4096x4096
  matmul on the same machine: CPU 0.1620 s, GPU 0.0067 s, relative speedup
  24.28x. This is a PyTorch CPU-vs-GPU reference only, not an AeroNum matmul
  result
  ([raw log](claim-verification/results/aeronum_pytorch_matmul_reference_7900xtx_20260528T191500Z/aeronn_gpu_compare.stdout.log)).
- `labs/compare/distributed_compare.py` now runs a real PyTorch DDP smoke
  benchmark instead of a simulated message. NCCL world size 1 passed on GPU 0
  with 3 steps, mean rank time 0.3393106461 s, and 565.853156 tokens/s
  ([result JSON](claim-verification/results/aeronum_nccl_ddp_single_gpu_7900xtx_20260528T224500Z/claim_result.json)).
- `labs/compare/distributed_compare.py` now runs an integrated NCCL topology
  guard before spawning multi-rank DDP. On this machine, the requested
  two-device topology is blocked before launch because device 1 is integrated
  AMD Radeon Graphics, selected devices span ROCm architectures 11.0 and 10.3,
  and the kernel command line does not include `iommu=pt`. The same script
  still passed an NCCL world-size-1 DDP smoke on device 0 with 191.244289
  tokens/s for the one-step smoke
  ([result JSON](claim-verification/results/aeronum_distributed_compare_guard_7900xtx_20260528T233403Z/claim_result.json)).
- `aeronum-core` now parses GGUF metadata, sampled tokenizer string-array
  metadata, tensor directory records, tensor data byte ranges, loads all 81
  F32 tensors into `LlamaModel`, and offloads those model weights through the
  model ROCm path. It also builds an exact-token lookup index from the GGUF
  tokenizer metadata and round-trips known token pieces through that index. The
  repo-owned command
  `cargo run -p aeronum-core --example gguf_header_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --device rocm --max-tokens 16 --prompt "AeroNum GGUF all F32 weights prompt"`
  passed against the local Mistral GGUF file, SHA-256
  `c5743c1bf39db0ae8a5ade5df0374b8e9e492754a199cfdad7ef393c1590f7c0`, and
  reported GGUF version 3, 363 parsed tensor infos, 45 parsed metadata entries,
  alignment `32`, data offset `7884256`, file size `14333910496`, 363 tensors
  with known byte sizes, and `tensor_layout_within_file=true`. It loaded
  414,720 F32 values across 81 F32 tensors, totaling 1,658,880 bytes. The
  `output_norm.weight` tensor remained at weight index 0 with 5,120 values,
  checksum `57094807.65625`, and F32 samples `4.21875`, `4.46875`,
  `4.34375`, and `4.3125`. It copied that 20,480-byte tensor to ROCm device 0
  (`Radeon RX 7900 XTX`) and back with round-trip checksum `57094807.65625`
  and `max_abs_diff=0.0`; `model.to("rocm")` then reported
  `loaded_weight_count=81` and `hip_weight_count=81`. The tokenizer index
  reported 131,072 tokens, 269,443 merges, BOS token id 1, EOS token id 2,
  unknown token id 0, exact-token ids for `<unk>`, `<s>`, `</s>`, `[INST]`,
  and `[/INST]`, and exact-piece encode/decode for
  `["<s>","[INST]","[/INST]","</s>"]` as `[1,3,4,2]`. It also parsed 131,072
  `tokenizer.ggml.token_type` entries and reported token types `[3,3,3,3]`
  for those exact pieces. This is a metadata/directory/layout/tokenizer-array
  and F32 model-weight offload smoke result. The same artifact reported
  tokenizer config `pre=tekken`, padding token id 11, add-BOS true, add-EOS
  false, add-space-prefix false, chat template length 2,002, chat template
  checksum 156,939,758, context length 131,072, embedding length 5,120, 40
  blocks, 32 attention heads, 8 KV heads, rope frequency base 1,000,000,000,
  and RMS epsilon 0.00001. Generation is still placeholder; this is not full
  GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_all_f32_weights_7900xtx_20260529T002801Z/claim_result.json)).
- `aeronum-core` now includes a byte-level BPE tokenizer path built from GGUF
  `tokenizer.ggml.tokens`, `tokenizer.ggml.merges`, and
  `tokenizer.ggml.token_type`, with GPT-style pre-tokenization, default
  special-token parsing, and literal `--no-parse-special` behavior for the
  fixed prompt set. The repo-owned comparison wrapper matched
  `/home/rob/llama.cpp/build-gpu/bin/llama-tokenize` on the same local Mistral
  GGUF file for 14 fixed prompts and 56 comparisons. The prompt set includes
  punctuation, leading spaces, contractions, digits, `Äpfel`, `🚀`,
  `this is 🦙.cpp`, `[INST]`, `<s>[INST]Hello[/INST]`,
  `Hello [INST] world`, and `[AVAILABLE_TOOLS]`. This is a fixed-prompt
  tokenizer parity check, not exhaustive tokenizer parity or GGUF token
  inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_tokenizer_parse_modes_7900xtx_20260529T010000Z/claim_result.json)).
- `aeronum-core` now reads and CPU-decodes the first quantized GGUF block for
  one Q4_K tensor and one Q6_K tensor from the same local Mistral GGUF file.
  The repo-owned command
  `cargo run -p aeronum-core --example gguf_quantized_block_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf`
  validated `token_embd.weight` as Q4_K with block size 256, type size 144,
  tensor byte size 377,487,360, and 256 decoded values; it also validated
  `output.weight` as Q6_K with block size 256, type size 210, tensor byte size
  550,502,400, and 256 decoded values. This is first-block CPU decode only,
  not full q4_K/q6_K tensor execution, GPU matmul, or AeroNum-native GGUF
  token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_quantized_block_decode_7900xtx_20260529T010756Z/claim_result.json)).
- `aeronum-core` now also reads and CPU-decodes selected full rows from those
  quantized tensors. The repo-owned command
  `cargo run -p aeronum-core --example gguf_quantized_block_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --q4-row 22177 --q6-row 100`
  decoded 5,120 values from Q4_K `token_embd.weight` row 22177 and 5,120
  values from Q6_K `output.weight` row 100. This verifies selected-row CPU
  decode only, not full q4_K/q6_K tensor execution, GPU matmul, or
  AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_quantized_row_decode_7900xtx_20260529T011641Z/claim_result.json)).
- `aeronum-core` now computes a CPU dot product between selected decoded
  quantized rows. The same row-smoke command decoded Q4_K
  `token_embd.weight` row 22177 and Q6_K `output.weight` row 100, then
  computed their 5,120-value dot product as `-0.000096131074`. This is a
  selected-row arithmetic smoke only, not full logits, full q4_K/q6_K tensor
  execution, GPU matmul, or AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_quantized_row_dot_7900xtx_20260529T012052Z/claim_result.json)).
- `aeronum-core` now computes CPU prefix logits over decoded quantized rows.
  The repo-owned command
  `cargo run -p aeronum-core --example gguf_quantized_block_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --q4-row 22177 --q6-row 100 --logit-start 0 --logit-rows 256 --top-k 5`
  decoded Q4_K `token_embd.weight` row 22177, decoded the first 256 Q6_K
  `output.weight` rows, computed 256 dot-product logits, and reported the
  highest prefix logit at row 2 with value `0.000449394164`. This is a
  prefix-logits CPU smoke only, not full-vocabulary logits, full q4_K/q6_K
  tensor execution, GPU matmul, or AeroNum-native GGUF token inference
  throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_quantized_prefix_logits_7900xtx_20260529T012458Z/claim_result.json)).
- `aeronum-core` now stream-decodes all Q6_K `output.weight` rows and computes
  full output-vocabulary CPU logits for one selected Q4_K embedding row. The
  repo-owned command
  `cargo run --release -p aeronum-core --example gguf_quantized_block_smoke -- --model /home/rob/models/mistralai_Mistral-Small-3.1-24B-Instruct-2503-Q4_K_M.gguf --q4-row 22177 --q6-row 100 --logit-start 0 --logit-rows 131072 --top-k 5`
  computed 131,072 logits over 5,120-dimensional decoded rows and reported
  top rows 109,526, 31,494, 123,618, 57,996, and 6,731. This is full
  output-vocabulary arithmetic for one selected embedding row, not
  transformer hidden-state logits, generated-token logits, GPU matmul, or
  AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_quantized_full_vocab_logits_7900xtx_20260529T012919Z/claim_result.json)).
- `aeronum-core` now applies the final RMS/output norm before full
  output-vocabulary CPU logits for one selected Q4_K embedding row. The same
  release command decoded `token_embd.weight` row 22177, applied
  `output_norm.weight` with RMS epsilon `0.00001`, stream-decoded all 131,072
  Q6_K `output.weight` rows, and reported top rows 109,526, 123,618, 57,996,
  113,893, and 31,494. This is final-LM-head-style arithmetic for one selected
  embedding row, not transformer hidden-state logits, generated-token logits,
  GPU matmul, or AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_normalized_full_vocab_logits_7900xtx_20260529T013621Z/claim_result.json)).
- `aeronum-core` now verifies a first-layer V-projection CPU path from decoded
  GGUF tensors. The repo-owned release command decoded `token_embd.weight` row
  22177, applied `blk.0.attn_norm.weight`, stream-decoded all 1,024 rows of
  Q6_K `blk.0.attn_v.weight`, and reported normalized top projection rows 94,
  919, 750, 1,007, and 104. This is first-layer V-projection arithmetic only,
  not attention output, FFN execution, transformer hidden-state logits,
  generated-token logits, GPU matmul, or AeroNum-native GGUF token inference
  throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_first_layer_v_projection_7900xtx_20260529T020500Z/claim_result.json)).
- `aeronum-core` now verifies a single-token first-layer attention-output CPU
  subpath. The repo-owned release command decoded `token_embd.weight` row 22177,
  applied `blk.0.attn_norm.weight`, computed all 1,024 rows of
  `blk.0.attn_v.weight`, repeated the V vector by GQA factor 4 from model
  metadata, computed all 5,120 rows of `blk.0.attn_output.weight`, and reported
  top attention-output rows 1,034, 4,481, 2,121, 148, and 4,367. This is a
  single-token attention-output subpath only, not multi-token attention scores,
  RoPE validation, FFN execution, generated-token logits, GPU matmul, or
  AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_single_token_attention_output_7900xtx_20260529T022500Z/claim_result.json)).
- `aeronum-core` now verifies a first-layer multi-token attention CPU subpath
  from decoded GGUF tensors. The repo-owned release command decoded
  `token_embd.weight` rows 1, 22177, and 1044, applied `blk.0.attn_norm.weight`,
  computed Q/K/V projections from `blk.0.attn_q.weight`, `blk.0.attn_k.weight`,
  and `blk.0.attn_v.weight`, applied internal RoPE arithmetic with
  `llama.rope.freq_base` 1,000,000,000, formed 192 causal attention scores, and
  computed the final-token `blk.0.attn_output.weight` projection. This is a
  first-layer CPU attention subpath only; the RoPE arithmetic is not external
  parity, and this is not FFN execution, full transformer execution,
  generated-token logits, GPU matmul, or AeroNum-native GGUF token inference
  throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_multi_token_attention_7900xtx_20260529T034500Z/claim_result.json)).
- `aeronum-core` now extends that multi-token first-layer CPU subpath through
  the final token's FFN and final-head logits. The repo-owned release command
  used the same three token rows, added the final-token attention residual,
  applied `blk.0.ffn_norm.weight`, computed all 32,768 rows of
  `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`, applied `SiLU(gate) * up`,
  computed all 5,120 rows of `blk.0.ffn_down.weight`, applied
  `output_norm.weight`, stream-decoded all 131,072 rows of `output.weight`, and
  reported top logit rows 64,162, 68,637, 117,577, 64,303, and 128,129. This is
  first-layer final-token CPU arithmetic only, not full transformer execution,
  generated-token logits, GPU matmul, or AeroNum-native GGUF token inference
  throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_multi_token_layer_logits_7900xtx_20260529T041500Z/claim_result.json)).
- `aeronum-core` now verifies a single-token first-layer attention-plus-FFN CPU
  subpath. The repo-owned release command ran the single-token attention-output
  subpath, added the residual, applied `blk.0.ffn_norm.weight`, computed all
  32,768 rows of `blk.0.ffn_gate.weight` and `blk.0.ffn_up.weight`, applied
  `SiLU(gate) * up`, computed all 5,120 rows of `blk.0.ffn_down.weight`, and
  reported top FFN-output rows 3,662, 4,865, 2,575, 531, and 4,397. This is a
  single-token first-layer CPU subpath only, not multi-token attention scores,
  RoPE validation, full transformer execution, generated-token logits, GPU
  matmul, or AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_single_token_ffn_output_7900xtx_20260529T024000Z/claim_result.json)).
- `aeronum-core` now verifies full-vocabulary final-head CPU logits from the
  single-token layer-0 hidden state. The repo-owned release command formed the
  layer-0 residual hidden state, applied `output_norm.weight`, stream-decoded all
  131,072 Q6_K `output.weight` rows, and reported top logit rows 89,186,
  122,211, 123,618, 64,162, and 90,406. This is first-layer hidden-state
  final-head arithmetic only, not full transformer execution, generated-token
  logits, multi-token attention scores, RoPE validation, GPU matmul, or
  AeroNum-native GGUF token inference throughput
  ([result JSON](claim-verification/results/aeronum_core_gguf_single_token_layer_logits_7900xtx_20260529T025500Z/claim_result.json)).
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
  reference only. The current AeroNum-owned language-model results are a tiny
  explicit-gradient training analogue, a tiny causal self-attention forward
  analogue, and a tiny causal self-attention output-projection training
  analogue, not GPT-2.
- Broad GPU 4096x4096 speedup claims are omitted. The verified current
  same-run AeroNum core hipBLAS vs PyTorch ROCm measurement is near parity
  with a 1.005270x median-time ratio on an all-ones SGEMM workload.
- NCCL/MPI multi-GPU scaling is omitted. A real NCCL/DDP single-GPU smoke test
  passed, but the local two-device attempts using the Radeon RX 7900 XTX plus
  integrated AMD Radeon Graphics failed. After reboot, the default
  heterogeneous run failed with RCCL `hipIpcGetMemHandle failed: invalid argument`.
  A `NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1` attempt failed with rank 1
  `invalid device function`. A two-rank single-XTX attempt was rejected by RCCL
  as duplicate GPU usage. The current `distributed_compare.py` guard now blocks
  the requested two-device run before DDP launch because device 1 is
  integrated, the selected devices span ROCm architectures 11.0 and 10.3, and
  the kernel command line lacks `iommu=pt`. No compatible second discrete ROCm
  GPU was verified on this machine
  ([guard result JSON](claim-verification/results/aeronum_distributed_compare_guard_7900xtx_20260528T233403Z/claim_result.json)).
- AeroNum-native GGUF token-inference throughput claims are omitted. The
  verified current AeroNum core result parses local GGUF metadata, tokenizer
  string-array samples, the full tokenizer token array, tokenizer merges,
  tokenizer token-type array, exact-token-piece encode/decode for known
  special tokens, a byte-level BPE path with default and literal
  no-parse-special modes matching llama.cpp on 14 fixed prompts, tokenizer
  config, selected Llama hyperparameters, tensor directory
  records, tensor byte ranges, loads all 81 F32 tensors into `LlamaModel`,
  offloads those model weights through ROCm device 0, then reaches placeholder
  generation. First-block decode, selected-row decode, a selected-row CPU dot
  product, 256-row prefix logits, full output-vocabulary CPU arithmetic, final
  RMS/output-norm full output-vocabulary CPU arithmetic for one selected Q4_K
  embedding row against Q6_K `output.weight`, first-layer V-projection CPU
  arithmetic against Q6_K `blk.0.attn_v.weight`, a single-token first-layer
  attention-output CPU subpath against Q4_K `blk.0.attn_output.weight`, and a
  first-layer multi-token attention CPU subpath with Q/K/V projections, internal
  RoPE arithmetic, 192 causal attention scores, and final-token attention-output
  projection, plus a final-token first-layer FFN and final-head logits CPU
  subpath, and a single-token first-layer attention-plus-FFN CPU subpath through
  `blk.0.ffn_gate.weight`, `blk.0.ffn_up.weight`, and `blk.0.ffn_down.weight`,
  plus full-vocabulary final-head CPU logits from that single-token layer-0
  hidden state are verified, but exhaustive tokenizer parity, full q4_K/q6_K
  transformer execution, external RoPE parity, generated-token logits, and
  AeroNum-native token inference throughput are not yet verified. The verified
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
