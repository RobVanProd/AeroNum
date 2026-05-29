#include <hip/hip_runtime.h>

#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

static void check_hip(hipError_t code, const char *call) {
    if (code != hipSuccess) {
        std::cerr << call << " failed: " << hipGetErrorString(code) << "\n";
        std::exit(2);
    }
}

static std::string arg_value(int argc, char **argv, const std::string &name, const std::string &fallback = "") {
    for (int i = 1; i + 1 < argc; ++i) {
        if (argv[i] == name) {
            return argv[i + 1];
        }
    }
    return fallback;
}

static std::vector<uint8_t> read_u8_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "could not open " << path << "\n";
        std::exit(2);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::vector<float> read_f32_file(const std::string &path) {
    auto bytes = read_u8_file(path);
    if (bytes.size() % sizeof(float) != 0) {
        std::cerr << "f32 file size is not divisible by 4\n";
        std::exit(2);
    }
    std::vector<float> values(bytes.size() / sizeof(float));
    std::memcpy(values.data(), bytes.data(), bytes.size());
    return values;
}

__device__ float f16_to_f32_device(uint16_t bits) {
    uint32_t sign = (static_cast<uint32_t>(bits & 0x8000u)) << 16;
    int32_t exp = static_cast<int32_t>((bits >> 10) & 0x1fu);
    uint32_t frac = static_cast<uint32_t>(bits & 0x03ffu);
    uint32_t out = 0;
    if (exp == 0) {
        if (frac == 0) {
            out = sign;
        } else {
            int32_t exponent = -14;
            while ((frac & 0x0400u) == 0) {
                frac <<= 1;
                exponent -= 1;
            }
            frac &= 0x03ffu;
            uint32_t exp32 = static_cast<uint32_t>(exponent + 127);
            out = sign | (exp32 << 23) | (frac << 13);
        }
    } else if (exp == 0x1f) {
        out = sign | 0x7f800000u | (frac << 13);
    } else {
        uint32_t exp32 = static_cast<uint32_t>(exp - 15 + 127);
        out = sign | (exp32 << 23) | (frac << 13);
    }
    return __uint_as_float(out);
}

__device__ float decode_q6k_value(const uint8_t *block, int t) {
    const uint8_t *ql = block;
    const uint8_t *qh = block + 128;
    const int8_t *scales = reinterpret_cast<const int8_t *>(block + 192);
    uint16_t d_bits = static_cast<uint16_t>(block[208]) | (static_cast<uint16_t>(block[209]) << 8);
    float d = f16_to_f32_device(d_bits);

    int n = t < 128 ? 0 : 128;
    int r = t - n;
    int ql_base = n / 2;
    int qh_base = n / 4;
    int scale_base = n / 16;
    int l = 0;
    int low = 0;
    int high = 0;
    int scale_index = 0;
    if (r < 32) {
        l = r;
        low = ql[ql_base + l] & 0x0f;
        high = (qh[qh_base + l] >> 0) & 3;
        scale_index = scale_base + l / 16;
    } else if (r < 64) {
        l = r - 32;
        low = ql[ql_base + l + 32] & 0x0f;
        high = (qh[qh_base + l] >> 2) & 3;
        scale_index = scale_base + l / 16 + 2;
    } else if (r < 96) {
        l = r - 64;
        low = ql[ql_base + l] >> 4;
        high = (qh[qh_base + l] >> 4) & 3;
        scale_index = scale_base + l / 16 + 4;
    } else {
        l = r - 96;
        low = ql[ql_base + l + 32] >> 4;
        high = (qh[qh_base + l] >> 6) & 3;
        scale_index = scale_base + l / 16 + 6;
    }

    int q = (low | (high << 4)) - 32;
    return d * static_cast<float>(scales[scale_index]) * static_cast<float>(q);
}

__global__ void q6k_decode_dot_kernel(const float *lhs, const uint8_t *rhs, float *partials, int block_count) {
    __shared__ float scratch[256];
    int block_index = blockIdx.x;
    int t = threadIdx.x;
    float value = 0.0f;
    if (block_index < block_count && t < 256) {
        const uint8_t *q6_block = rhs + block_index * 210;
        value = lhs[block_index * 256 + t] * decode_q6k_value(q6_block, t);
    }
    scratch[t] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            scratch[t] += scratch[t + stride];
        }
        __syncthreads();
    }
    if (t == 0) {
        partials[block_index] = scratch[0];
    }
}

int main(int argc, char **argv) {
    std::string lhs_path = arg_value(argc, argv, "--lhs-bin");
    std::string rhs_path = arg_value(argc, argv, "--rhs-q6k-bin");
    int device = std::stoi(arg_value(argc, argv, "--device", "0"));
    double expected = std::stod(arg_value(argc, argv, "--expected-dot", "0"));
    if (lhs_path.empty() || rhs_path.empty()) {
        std::cerr << "usage: q6k_decode_dot_hip --lhs-bin <path> --rhs-q6k-bin <path> --expected-dot <value> [--device <id>]\n";
        return 2;
    }

    auto lhs = read_f32_file(lhs_path);
    auto rhs = read_u8_file(rhs_path);
    if (rhs.size() % 210 != 0 || lhs.size() != (rhs.size() / 210) * 256) {
        std::cerr << "input dimensions do not match Q6_K row layout\n";
        return 2;
    }
    int block_count = static_cast<int>(rhs.size() / 210);
    check_hip(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");

    float *d_lhs = nullptr;
    uint8_t *d_rhs = nullptr;
    float *d_partials = nullptr;
    check_hip(hipMalloc(&d_lhs, lhs.size() * sizeof(float)), "hipMalloc lhs");
    check_hip(hipMalloc(&d_rhs, rhs.size()), "hipMalloc rhs");
    check_hip(hipMalloc(&d_partials, block_count * sizeof(float)), "hipMalloc partials");
    check_hip(hipMemcpy(d_lhs, lhs.data(), lhs.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy lhs");
    check_hip(hipMemcpy(d_rhs, rhs.data(), rhs.size(), hipMemcpyHostToDevice), "hipMemcpy rhs");

    q6k_decode_dot_kernel<<<block_count, 256>>>(d_lhs, d_rhs, d_partials, block_count);
    check_hip(hipGetLastError(), "q6k_decode_dot_kernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<float> partials(block_count);
    check_hip(hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy partials");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize copy");
    check_hip(hipFree(d_lhs), "hipFree lhs");
    check_hip(hipFree(d_rhs), "hipFree rhs");
    check_hip(hipFree(d_partials), "hipFree partials");

    double gpu_dot = 0.0;
    double partial_checksum = 0.0;
    for (size_t i = 0; i < partials.size(); ++i) {
        gpu_dot += static_cast<double>(partials[i]);
        partial_checksum += static_cast<double>(i + 1) * static_cast<double>(partials[i]);
    }
    double abs_diff = std::abs(gpu_dot - expected);

    std::cout << std::fixed << std::setprecision(12)
              << "{"
              << "\"benchmark\":\"q6k_decode_dot_hip\","
              << "\"device_id\":" << device << ","
              << "\"device_name\":\"" << props.name << "\","
              << "\"dimension\":" << lhs.size() << ","
              << "\"q6k_block_count\":" << block_count << ","
              << "\"q6k_row_bytes\":" << rhs.size() << ","
              << "\"expected_cpu_dot\":" << expected << ","
              << "\"gpu_q6k_decode_dot\":" << gpu_dot << ","
              << "\"abs_diff\":" << abs_diff << ","
              << "\"partial_checksum\":" << partial_checksum << ","
              << "\"validation\":\"gpu_q6k_decode_dot_matches_cpu_reference\","
              << "\"limitations\":["
              << "\"GPU decodes one Q6_K row and computes per-block dot partials\","
              << "\"input vector is decoded on CPU before GPU execution\","
              << "\"final sum over GPU partials is performed on host\","
              << "\"not full q4_K/q6_K tensor execution on GPU\","
              << "\"not transformer layer execution on GPU\","
              << "\"not GPU autoregressive decoding\","
              << "\"not optimized AeroNum-native GGUF token inference throughput\""
              << "]"
              << "}\n";
    return abs_diff <= 1e-5 ? 0 : 1;
}
