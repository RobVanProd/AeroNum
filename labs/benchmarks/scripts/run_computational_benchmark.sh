#!/bin/bash
set -euo pipefail

# Comprehensive computational benchmark: AeroNum vs NumPy

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../../.." && pwd)
BENCH_DIR="$ROOT_DIR/labs/benchmarks"
RESULTS_DIR="$BENCH_DIR/results/computational_results"

AERO_COMPILER="$ROOT_DIR/aero-compiler/aero"
AERO_SRC="$BENCH_DIR/computational_benchmark.aero"
NUMPY_SRC="$BENCH_DIR/computational_benchmark_numpy.py"

echo "=== AeroNum vs NumPy Computational Benchmark ==="
mkdir -p "$RESULTS_DIR"

numpy_result=""
if command -v python3 >/dev/null 2>&1; then
  numpy_output=$(python3 "$NUMPY_SRC" 2>&1 || true)
  numpy_result=$(echo "$numpy_output" | grep -E "Final computation result:" | grep -oE '[0-9]+' | head -1 || true)
  echo "NumPy result: ${numpy_result:-N/A}"
else
  echo "python3 not found; skipping NumPy run"
fi

# Compile Aero benchmark (best-effort)
./"$AERO_COMPILER" build "$AERO_SRC" -o "$RESULTS_DIR/aero_computational.ll" > /dev/null 2>&1 || true

if command -v llc >/dev/null 2>&1 && command -v clang >/dev/null 2>&1; then
  llc "$RESULTS_DIR/aero_computational.ll" -o "$RESULTS_DIR/aero_computational.s" > /dev/null 2>&1 || true
  clang "$RESULTS_DIR/aero_computational.s" -o "$RESULTS_DIR/aero_computational_exe" > /dev/null 2>&1 || true
fi

echo "Results written under: $RESULTS_DIR"
