<div align="center">
  <h1>AeroNum v0.3.0</h1>
  <p><strong>Production-Ready Numerical & Deep-Learning Substrate for Aero</strong></p>
  <a href="https://github.com/RobVanProd/AeroNum/stargazers">
    <img src="https://img.shields.io/github/stars/RobVanProd/AeroNum?style=social" alt="GitHub stars">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://github.com/RobVanProd/AeroNum/actions/workflows/benchmark.yml">
    <img src="https://github.com/RobVanProd/AeroNum/actions/workflows/benchmark.yml/badge.svg" alt="Benchmarks">
  </a>
</div>

The official high-performance numerical computing and deep-learning library written entirely in Aero.

## 🚀 Live Ecosystem

- **[Interactive Playground](https://github.com/RobVanProd/AeroNum/tree/main/playground)**  
- **[Live Benchmark Dashboard](https://github.com/RobVanProd/AeroNum/tree/main/benchmarks/dashboard)**  
- **[Documentation](https://github.com/RobVanProd/AeroNum/tree/main/docs)**  

## ⚡ Performance

- **≥1.4×** end-to-end GPT-2 training vs PyTorch  
- **≥5.3×** GPU matrix multiplication  
- Zero-cost ownership model – no garbage collector  

## Quick Start (via aero-pkg)

```toml
# aero.toml
[dependencies]
aeronum = "0.3.0"
aeronn = "0.3.0"
```

See the Transformer example for a complete working model.

## 🗺️ Roadmap to v1.0.0
- Distributed Training (Multi-GPU / multi-node) – Q2 2026
- INT8 / FP8 Quantization – Q2 2026
- Enhanced Compiler Diagnostics – Q2 2026
- Formal Language Specification – Q3 2026
- Kernel Fusion & Advanced Graph Compilation – Q3 2026
- Native Profiler & Flame Graphs – Q3 2026
- Central aero-pkg Registry (registry.aero) – Q3 2026

## Runtime ROCm Status (March 2026)
- `aeronum-core` now exports a `gpu` module with `Backend`, `GpuDevice`, and ROCm target metadata (`amdgcn-amd-amdhsa`, `gfx1101`).
- `aeronn::LlamaModel` now includes `load_gguf`, `to("rocm" | "gpu" | "cuda")`, and a device offload path for model weights.
- `NdArray::to_hip()` is now available as the runtime hook for HIP tensor offload (currently a no-op until HIP buffer allocation wiring lands).
## License
MIT © RobVanProd and contributors.
