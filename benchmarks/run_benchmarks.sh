#!/bin/bash
set -euo pipefail

# Benchmark suite for Aero vs C++ vs Python.
# NOTE: This repo is in flux; this script is best-effort and avoids hard-coded paths.

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/.." && pwd)
RESULTS_DIR="$SCRIPT_DIR/results"

mkdir -p "$RESULTS_DIR"

echo "=== AeroNum Performance Benchmark Suite ==="

# Aero compilation benchmark (if toolchain available)
AERO_COMPILER="$ROOT_DIR/aero-compiler/aero"
AERO_SRC="$ROOT_DIR/examples/aero/working_prototype.aero"

if [ -x "$AERO_COMPILER" ]; then
  echo "Compiling Aero benchmark source: $AERO_SRC"
  "$AERO_COMPILER" build "$AERO_SRC" -o "$RESULTS_DIR/aero_output.ll" > /dev/null 2>&1 || true
else
  echo "Aero compiler not found at $AERO_COMPILER; skipping Aero compile"
fi

# C++ compilation benchmark (if g++ available)
if command -v g++ >/dev/null 2>&1; then
  echo "Compiling C++ benchmark..."
  g++ -O2 "$SCRIPT_DIR/benchmark.cpp" -o "$RESULTS_DIR/cpp_benchmark" > /dev/null 2>&1 || true
else
  echo "g++ not found; skipping C++ compile"
fi

# Python benchmark (if python3 available)
if command -v python3 >/dev/null 2>&1; then
  echo "Running Python benchmark..."
  python3 "$SCRIPT_DIR/benchmark.py" > /dev/null 2>&1 || true
else
  echo "python3 not found; skipping Python run"
fi

echo "Results directory: $RESULTS_DIR"
