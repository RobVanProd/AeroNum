#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#define HIP_CHECK(cmd)                                                              \
    do {                                                                            \
        hipError_t err = (cmd);                                                     \
        if (err != hipSuccess) {                                                    \
            std::cerr << "HIP error: " << hipGetErrorString(err)                  \
                      << " (" << static_cast<int>(err) << ") at " << __FILE__   \
                      << ":" << __LINE__ << " for " << #cmd << std::endl;       \
            return 1;                                                               \
        }                                                                           \
    } while (0)

__global__ void vector_add(const float* a, const float* b, float* c, std::size_t n) {
    std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

static double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    double sum = std::accumulate(values.begin(), values.end(), 0.0);
    return sum / static_cast<double>(values.size());
}

static double compute_median(std::vector<double> values) {
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return (values[mid - 1] + values[mid]) * 0.5;
    }
    return values[mid];
}

static int parse_int_arg(int argc, char** argv, const std::string& name, int default_value) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            return std::stoi(argv[i + 1]);
        }
    }
    return default_value;
}

static std::size_t parse_size_arg(
    int argc,
    char** argv,
    const std::string& name,
    std::size_t default_value
) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (name == argv[i]) {
            return static_cast<std::size_t>(std::stoull(argv[i + 1]));
        }
    }
    return default_value;
}

int main(int argc, char** argv) {
    const std::size_t n = parse_size_arg(argc, argv, "--size", 1ULL << 24);
    const int runs = parse_int_arg(argc, argv, "--runs", 20);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 5);
    const int block_size = parse_int_arg(argc, argv, "--block-size", 256);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n, 0.0f);

    for (std::size_t i = 0; i < n; ++i) {
        h_a[i] = static_cast<float>(i % 1024) * 0.25f;
        h_b[i] = static_cast<float>((i * 3) % 1024) * 0.5f;
    }

    HIP_CHECK(hipSetDevice(0));

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;

    const std::size_t bytes = n * sizeof(float);

    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));

    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));

    const int grid_size = static_cast<int>((n + static_cast<std::size_t>(block_size) - 1) /
                                           static_cast<std::size_t>(block_size));

    for (int i = 0; i < warmup; ++i) {
        hipLaunchKernelGGL(vector_add,
                           dim3(static_cast<unsigned int>(grid_size)),
                           dim3(static_cast<unsigned int>(block_size)),
                           0,
                           0,
                           d_a,
                           d_b,
                           d_c,
                           n);
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> run_ms;
    run_ms.reserve(static_cast<std::size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        hipLaunchKernelGGL(vector_add,
                           dim3(static_cast<unsigned int>(grid_size)),
                           dim3(static_cast<unsigned int>(block_size)),
                           0,
                           0,
                           d_a,
                           d_b,
                           d_c,
                           n);
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start)
                .count();
        run_ms.push_back(elapsed_ms);
    }

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    bool valid = true;
    const std::size_t checks = std::min<std::size_t>(n, 2048);
    for (std::size_t i = 0; i < checks; ++i) {
        float expected = h_a[i] + h_b[i];
        if (std::fabs(h_c[i] - expected) > 1e-5f) {
            valid = false;
            break;
        }
    }

    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    if (!valid) {
        std::cerr << "Validation failed: vector add output mismatch" << std::endl;
        return 2;
    }

    const double mean_ms = compute_mean(run_ms);
    const double median_ms = compute_median(run_ms);
    const double min_ms = *std::min_element(run_ms.begin(), run_ms.end());
    const double max_ms = *std::max_element(run_ms.begin(), run_ms.end());

    const double seconds = median_ms / 1000.0;
    const double elements_per_second = static_cast<double>(n) / seconds;
    const double gflops = elements_per_second / 1e9;  // one add per element
    const double bytes_per_run = static_cast<double>(3ULL * n * sizeof(float));
    const double bandwidth_gbps = (bytes_per_run / seconds) / 1e9;

    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
         << "{"
         << "\"backend\":\"hip_rocm\"," 
         << "\"kernel\":\"vector_add\"," 
         << "\"size\":" << n << ","
         << "\"runs\":" << runs << ","
         << "\"warmup\":" << warmup << ","
         << "\"block_size\":" << block_size << ","
         << "\"mean_ms\":" << mean_ms << ","
         << "\"median_ms\":" << median_ms << ","
         << "\"min_ms\":" << min_ms << ","
         << "\"max_ms\":" << max_ms << ","
         << "\"gflops\":" << gflops << ","
         << "\"bandwidth_gbps\":" << bandwidth_gbps
         << "}";

    std::cout << "hip_vector_add: size=" << n
              << " runs=" << runs
              << " median_ms=" << median_ms
              << " gflops=" << gflops
              << " bandwidth_gbps=" << bandwidth_gbps
              << std::endl;
    std::cout << json.str() << std::endl;
    return 0;
}
