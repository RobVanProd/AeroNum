#!/bin/bash

# Comprehensive benchmark script for Aero vs C++ vs Python
# Tests both compilation time and execution time

echo "=== AeroNum Performance Benchmark Suite ==="
echo "Comparing Aero vs C++ vs Python"
echo "============================================"

# Create results directory
mkdir -p results

# Function to run multiple iterations and get average
run_multiple_times() {
    local command="$1"
    local iterations=10
    local total=0
    
    for i in $(seq 1 $iterations); do
        local result=$($command 2>&1 | grep -o '[0-9.]*' | head -1)
        if [[ -n "$result" ]]; then
            total=$(echo "$total + $result" | bc -l)
        fi
    done
    
    echo "scale=2; $total / $iterations" | bc -l
}

echo ""
echo "1. COMPILATION TIME BENCHMARKS"
echo "==============================="

# Aero compilation benchmark
echo "Testing Aero compilation time..."
cd /home/ubuntu/AeroNum
aero_compile_times=()
for i in {1..10}; do
    start_time=$(date +%s%N)
    ./aero-compiler/aero build working_prototype.aero -o benchmarks/results/aero_output.ll > /dev/null 2>&1
    end_time=$(date +%s%N)
    compile_time=$(echo "scale=2; ($end_time - $start_time) / 1000" | bc -l)  # Convert to microseconds
    aero_compile_times+=($compile_time)
done

# Calculate average Aero compilation time
aero_avg=0
for time in "${aero_compile_times[@]}"; do
    aero_avg=$(echo "$aero_avg + $time" | bc -l)
done
aero_avg=$(echo "scale=2; $aero_avg / 10" | bc -l)

echo "Aero average compilation time: ${aero_avg} μs"

# C++ compilation benchmark
echo "Testing C++ compilation time..."
cd benchmarks
cpp_compile_times=()
for i in {1..10}; do
    start_time=$(date +%s%N)
    g++ -O2 benchmark.cpp -o results/cpp_benchmark > /dev/null 2>&1
    end_time=$(date +%s%N)
    compile_time=$(echo "scale=2; ($end_time - $start_time) / 1000" | bc -l)  # Convert to microseconds
    cpp_compile_times+=($compile_time)
done

# Calculate average C++ compilation time
cpp_avg=0
for time in "${cpp_compile_times[@]}"; do
    cpp_avg=$(echo "$cpp_avg + $time" | bc -l)
done
cpp_avg=$(echo "scale=2; $cpp_avg / 10" | bc -l)

echo "C++ average compilation time: ${cpp_avg} μs"

# Python doesn't have compilation time (interpreted)
echo "Python compilation time: 0 μs (interpreted language)"

echo ""
echo "2. EXECUTION TIME BENCHMARKS"
echo "============================"

# Aero execution benchmark
echo "Testing Aero execution time..."
cd /home/ubuntu/AeroNum
llc benchmarks/results/aero_output.ll -o benchmarks/results/aero_output.s > /dev/null 2>&1
clang benchmarks/results/aero_output.s -o benchmarks/results/aero_exe > /dev/null 2>&1

aero_exec_times=()
for i in {1..10}; do
    start_time=$(date +%s%N)
    ./benchmarks/results/aero_exe > /dev/null 2>&1
    end_time=$(date +%s%N)
    exec_time=$(echo "scale=2; ($end_time - $start_time) / 1000" | bc -l)  # Convert to microseconds
    aero_exec_times+=($exec_time)
done

# Calculate average Aero execution time
aero_exec_avg=0
for time in "${aero_exec_times[@]}"; do
    aero_exec_avg=$(echo "$aero_exec_avg + $time" | bc -l)
done
aero_exec_avg=$(echo "scale=2; $aero_exec_avg / 10" | bc -l)

echo "Aero average execution time: ${aero_exec_avg} μs"

# C++ execution benchmark
echo "Testing C++ execution time..."
cd /home/ubuntu/AeroNum/benchmarks
cpp_exec_times=()
for i in {1..10}; do
    start_time=$(date +%s%N)
    ./results/cpp_benchmark > /dev/null 2>&1
    end_time=$(date +%s%N)
    exec_time=$(echo "scale=2; ($end_time - $start_time) / 1000" | bc -l)  # Convert to microseconds
    cpp_exec_times+=($exec_time)
done

# Calculate average C++ execution time
cpp_exec_avg=0
for time in "${cpp_exec_times[@]}"; do
    cpp_exec_avg=$(echo "$cpp_exec_avg + $time" | bc -l)
done
cpp_exec_avg=$(echo "scale=2; $cpp_exec_avg / 10" | bc -l)

echo "C++ average execution time: ${cpp_exec_avg} μs"

# Python execution benchmark
echo "Testing Python execution time..."
python_exec_times=()
for i in {1..10}; do
    start_time=$(date +%s%N)
    python3 benchmark.py > /dev/null 2>&1
    end_time=$(date +%s%N)
    exec_time=$(echo "scale=2; ($end_time - $start_time) / 1000" | bc -l)  # Convert to microseconds
    python_exec_times+=($exec_time)
done

# Calculate average Python execution time
python_exec_avg=0
for time in "${python_exec_times[@]}"; do
    python_exec_avg=$(echo "$python_exec_avg + $time" | bc -l)
done
python_exec_avg=$(echo "scale=2; $python_exec_avg / 10" | bc -l)

echo "Python average execution time: ${python_exec_avg} μs"

echo ""
echo "3. RESULTS SUMMARY"
echo "=================="

echo "Compilation Time Comparison:"
echo "  Aero:   ${aero_avg} μs"
echo "  C++:    ${cpp_avg} μs"
echo "  Python: 0 μs (interpreted)"

echo ""
echo "Execution Time Comparison:"
echo "  Aero:   ${aero_exec_avg} μs"
echo "  C++:    ${cpp_exec_avg} μs"
echo "  Python: ${python_exec_avg} μs"

echo ""
echo "Performance Analysis:"

# Calculate speedup ratios
if (( $(echo "$cpp_exec_avg > 0" | bc -l) )); then
    aero_vs_cpp=$(echo "scale=2; $cpp_exec_avg / $aero_exec_avg" | bc -l)
    echo "  Aero vs C++: ${aero_vs_cpp}x faster"
fi

if (( $(echo "$python_exec_avg > 0" | bc -l) )); then
    aero_vs_python=$(echo "scale=2; $python_exec_avg / $aero_exec_avg" | bc -l)
    echo "  Aero vs Python: ${aero_vs_python}x faster"
fi

if (( $(echo "$cpp_avg > 0" | bc -l) )); then
    compile_speedup=$(echo "scale=2; $cpp_avg / $aero_avg" | bc -l)
    echo "  Aero compilation vs C++: ${compile_speedup}x faster"
fi

# Save results to CSV for visualization
echo "Language,Compilation_Time_μs,Execution_Time_μs" > results/benchmark_results.csv
echo "Aero,$aero_avg,$aero_exec_avg" >> results/benchmark_results.csv
echo "C++,$cpp_avg,$cpp_exec_avg" >> results/benchmark_results.csv
echo "Python,0,$python_exec_avg" >> results/benchmark_results.csv

echo ""
echo "Results saved to: benchmarks/results/benchmark_results.csv"
echo "Benchmark complete! 🎉"

