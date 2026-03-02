# HIP Vector-Add Benchmark

This benchmark compiles and runs a real HIP kernel (`vector_add`) on ROCm.

## Prerequisites
- ROCm HIP SDK installed (`hipcc` available in PATH)
- AMD GPU supported by ROCm (for this project: `gfx1101` / RX 7800 XT)

## Run

```powershell
python benchmarks/hip/run_hip_vector_add.py --arch gfx1101 --size 16777216 --runs 20 --warmup 5
```

The runner writes JSON output to:

- `benchmarks/results/hip/hip_vector_add_<timestamp>.json`

## Output Metrics
- `mean_ms`
- `median_ms`
- `min_ms`
- `max_ms`
- `gflops`
- `bandwidth_gbps`
