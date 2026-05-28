#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
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

#define HIPBLAS_CHECK(cmd)                                                          \
    do {                                                                            \
        hipblasStatus_t status = (cmd);                                             \
        if (status != HIPBLAS_STATUS_SUCCESS) {                                     \
            std::cerr << "hipBLAS error: " << static_cast<int>(status)             \
                      << " at " << __FILE__ << ":" << __LINE__                   \
                      << " for " << #cmd << std::endl;                            \
            return 1;                                                               \
        }                                                                           \
    } while (0)

static double compute_mean(const std::vector<double>& values) {
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) /
           static_cast<double>(values.size());
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

int main(int argc, char** argv) {
    const int n = parse_int_arg(argc, argv, "--n", 4096);
    const int runs = parse_int_arg(argc, argv, "--runs", 10);
    const int warmup = parse_int_arg(argc, argv, "--warmup", 3);

    if (n <= 0 || runs <= 0 || warmup < 0) {
        std::cerr << "Invalid arguments: n and runs must be positive; warmup must be >= 0"
                  << std::endl;
        return 2;
    }

    HIP_CHECK(hipSetDevice(0));

    const std::size_t elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
    const std::size_t bytes = elements * sizeof(float);

    std::vector<float> h_a(elements, 1.0f);
    std::vector<float> h_b(elements, 1.0f);
    std::vector<float> h_c(elements, 0.0f);

    float* d_a = nullptr;
    float* d_b = nullptr;
    float* d_c = nullptr;
    HIP_CHECK(hipMalloc(&d_a, bytes));
    HIP_CHECK(hipMalloc(&d_b, bytes));
    HIP_CHECK(hipMalloc(&d_c, bytes));
    HIP_CHECK(hipMemcpy(d_a, h_a.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_b, h_b.data(), bytes, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_c, 0, bytes));

    hipblasHandle_t handle = nullptr;
    HIPBLAS_CHECK(hipblasCreate(&handle));

    const float alpha = 1.0f;
    const float beta = 0.0f;
    const auto run_sgemm = [&]() -> hipblasStatus_t {
        return hipblasSgemm(handle,
                            HIPBLAS_OP_N,
                            HIPBLAS_OP_N,
                            n,
                            n,
                            n,
                            &alpha,
                            d_a,
                            n,
                            d_b,
                            n,
                            &beta,
                            d_c,
                            n);
    };

    for (int i = 0; i < warmup; ++i) {
        HIPBLAS_CHECK(run_sgemm());
    }
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<double> run_ms;
    run_ms.reserve(static_cast<std::size_t>(runs));

    for (int i = 0; i < runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        HIPBLAS_CHECK(run_sgemm());
        HIP_CHECK(hipDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        const double elapsed_ms =
            std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start)
                .count();
        run_ms.push_back(elapsed_ms);
    }

    HIP_CHECK(hipMemcpy(h_c.data(), d_c, bytes, hipMemcpyDeviceToHost));

    bool valid = true;
    const float expected = static_cast<float>(n);
    const std::size_t sample_stride = std::max<std::size_t>(1, elements / 4096);
    for (std::size_t i = 0; i < elements; i += sample_stride) {
        if (std::fabs(h_c[i] - expected) > 1e-3f) {
            valid = false;
            break;
        }
    }

    HIPBLAS_CHECK(hipblasDestroy(handle));
    HIP_CHECK(hipFree(d_a));
    HIP_CHECK(hipFree(d_b));
    HIP_CHECK(hipFree(d_c));

    if (!valid) {
        std::cerr << "Validation failed: sampled SGEMM outputs did not equal " << expected
                  << std::endl;
        return 3;
    }

    const double mean_ms = compute_mean(run_ms);
    const double median_ms = compute_median(run_ms);
    const double min_ms = *std::min_element(run_ms.begin(), run_ms.end());
    const double max_ms = *std::max_element(run_ms.begin(), run_ms.end());
    const double flops = 2.0 * static_cast<double>(n) * static_cast<double>(n) *
                         static_cast<double>(n);
    const double median_tflops = (flops / (median_ms / 1000.0)) / 1e12;

    std::ostringstream json;
    json << std::fixed << std::setprecision(6)
         << "{"
         << "\"backend\":\"hipblas_rocm\","
         << "\"kernel\":\"sgemm\","
         << "\"n\":" << n << ","
         << "\"runs\":" << runs << ","
         << "\"warmup\":" << warmup << ","
         << "\"mean_ms\":" << mean_ms << ","
         << "\"median_ms\":" << median_ms << ","
         << "\"min_ms\":" << min_ms << ","
         << "\"max_ms\":" << max_ms << ","
         << "\"median_tflops\":" << median_tflops << ","
         << "\"validation\":\"sampled_all_ones_expected_n\""
         << "}";

    std::cout << "hip_sgemm: n=" << n << " runs=" << runs << " median_ms=" << median_ms
              << " median_tflops=" << median_tflops << std::endl;
    std::cout << json.str() << std::endl;
    return 0;
}
