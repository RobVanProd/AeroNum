# HIP ROCm Benchmarks

These benchmarks compile and run ROCm workloads on a HIP-visible AMD GPU.

## Prerequisites
- ROCm HIP SDK installed (`hipcc` available in PATH)
- AMD GPU supported by ROCm

## Run

```bash
python3 benchmarks/hip/run_hip_vector_add.py --arch gfx1100 --size 16777216 --runs 20 --warmup 5
python3 benchmarks/hip/run_hip_sgemm.py --arch gfx1100 --n 4096 --runs 10 --warmup 3
```

The runners write JSON output to:

- `benchmarks/results/hip/hip_vector_add_<timestamp>.json`
- `benchmarks/results/hip/hip_sgemm_<timestamp>.json`

## Output Metrics
- `mean_ms`
- `median_ms`
- `min_ms`
- `max_ms`
- vector add: `gflops`, `bandwidth_gbps`
- SGEMM: `median_tflops`
