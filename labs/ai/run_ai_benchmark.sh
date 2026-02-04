#!/bin/bash
set -euo pipefail

# Historic AI Benchmark: First Aero AI vs Python vs C++

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
RESULTS_DIR="$SCRIPT_DIR/results/ai_benchmark_results"

AERO_COMPILER="$ROOT_DIR/aero-compiler/aero"
AERO_SRC="$SCRIPT_DIR/first_aero_ai.aero"
PY_SRC="$SCRIPT_DIR/first_aero_ai_python.py"
CPP_SRC="$SCRIPT_DIR/first_aero_ai_cpp.cpp"

echo "HISTORIC MILESTONE: First AI Written in Aero Language!"
echo "====================================================="

mkdir -p "$RESULTS_DIR"

# Compile Aero AI
./"$AERO_COMPILER" build "$AERO_SRC" -o "$RESULTS_DIR/aero_ai.ll" > /dev/null 2>&1 || true

# Generate executable (best-effort; toolchain may be absent in CI)
if command -v llc >/dev/null 2>&1 && command -v clang >/dev/null 2>&1; then
  llc "$RESULTS_DIR/aero_ai.ll" -o "$RESULTS_DIR/aero_ai.s" > /dev/null 2>&1 || true
  clang "$RESULTS_DIR/aero_ai.s" -o "$RESULTS_DIR/aero_ai_exe" > /dev/null 2>&1 || true
fi

# Run Python + C++ references if present
python_result=""
if command -v python3 >/dev/null 2>&1; then
  python_output=$(python3 "$PY_SRC" 2>&1 || true)
  python_result=$(echo "$python_output" | grep -E "Final AI prediction result:" | grep -oE '[0-9]+' | head -1 || true)
fi

cpp_result=""
if command -v g++ >/dev/null 2>&1; then
  g++ -std=c++17 -O2 "$CPP_SRC" -o "$RESULTS_DIR/cpp_ai_exe" > /dev/null 2>&1 || true
  cpp_result=$("$RESULTS_DIR/cpp_ai_exe" 2>/dev/null; echo $?)
fi

echo "Results written under: $RESULTS_DIR"
