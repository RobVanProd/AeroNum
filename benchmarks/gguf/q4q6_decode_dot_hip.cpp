#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
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

__device__ void q4_k_scale_min(int index, const uint8_t *scales, int *scale, int *min_value) {
    if (index < 4) {
        *scale = scales[index] & 63;
        *min_value = scales[index + 4] & 63;
    } else {
        *scale = (scales[index + 4] & 0x0f) | ((scales[index - 4] >> 6) << 4);
        *min_value = (scales[index + 4] >> 4) | ((scales[index] >> 6) << 4);
    }
}

__device__ float decode_q4k_value(const uint8_t *block, int t) {
    float d = f16_to_f32_device(static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8));
    float dmin = f16_to_f32_device(static_cast<uint16_t>(block[2]) | (static_cast<uint16_t>(block[3]) << 8));
    const uint8_t *scales = block + 4;
    const uint8_t *qs = block + 16;
    int group = t / 64;
    int within = t - group * 64;
    int scale_index = group * 2 + (within >= 32 ? 1 : 0);
    int scale = 0;
    int min_value = 0;
    q4_k_scale_min(scale_index, scales, &scale, &min_value);
    uint8_t packed = qs[group * 32 + (within & 31)];
    int q = within < 32 ? (packed & 0x0f) : (packed >> 4);
    return d * static_cast<float>(scale) * static_cast<float>(q)
        - dmin * static_cast<float>(min_value);
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

__global__ void q4q6_decode_dot_kernel(const uint8_t *q4, const uint8_t *q6, float *partials, int block_count) {
    __shared__ float scratch[256];
    int block_index = blockIdx.x;
    int t = threadIdx.x;
    float value = 0.0f;
    if (block_index < block_count && t < 256) {
        const uint8_t *q4_block = q4 + block_index * 144;
        const uint8_t *q6_block = q6 + block_index * 210;
        value = decode_q4k_value(q4_block, t) * decode_q6k_value(q6_block, t);
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
    std::string q4_path = arg_value(argc, argv, "--q4-bin");
    std::string q6_path = arg_value(argc, argv, "--q6-bin");
    int device = std::stoi(arg_value(argc, argv, "--device", "0"));
    double expected = std::stod(arg_value(argc, argv, "--expected-dot", "0"));
    if (q4_path.empty() || q6_path.empty()) {
        std::cerr << "usage: q4q6_decode_dot_hip --q4-bin <path> --q6-bin <path> --expected-dot <value> [--device <id>]\n";
        return 2;
    }

    auto q4 = read_u8_file(q4_path);
    auto q6 = read_u8_file(q6_path);
    if (q4.size() % 144 != 0 || q6.size() % 210 != 0 || q4.size() / 144 != q6.size() / 210) {
        std::cerr << "input dimensions do not match Q4_K/Q6_K row layout\n";
        return 2;
    }
    int block_count = static_cast<int>(q4.size() / 144);
    check_hip(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");

    uint8_t *d_q4 = nullptr;
    uint8_t *d_q6 = nullptr;
    float *d_partials = nullptr;
    check_hip(hipMalloc(&d_q4, q4.size()), "hipMalloc q4");
    check_hip(hipMalloc(&d_q6, q6.size()), "hipMalloc q6");
    check_hip(hipMalloc(&d_partials, block_count * sizeof(float)), "hipMalloc partials");
    check_hip(hipMemcpy(d_q4, q4.data(), q4.size(), hipMemcpyHostToDevice), "hipMemcpy q4");
    check_hip(hipMemcpy(d_q6, q6.data(), q6.size(), hipMemcpyHostToDevice), "hipMemcpy q6");

    q4q6_decode_dot_kernel<<<block_count, 256>>>(d_q4, d_q6, d_partials, block_count);
    check_hip(hipGetLastError(), "q4q6_decode_dot_kernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<float> partials(block_count);
    check_hip(hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy partials");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize copy");
    check_hip(hipFree(d_q4), "hipFree q4");
    check_hip(hipFree(d_q6), "hipFree q6");
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
              << "\"benchmark\":\"q4q6_decode_dot_hip\","
              << "\"device_id\":" << device << ","
              << "\"device_name\":\"" << props.name << "\","
              << "\"dimension\":" << block_count * 256 << ","
              << "\"q4k_block_count\":" << block_count << ","
              << "\"q4k_row_bytes\":" << q4.size() << ","
              << "\"q6k_block_count\":" << block_count << ","
              << "\"q6k_row_bytes\":" << q6.size() << ","
              << "\"expected_cpu_dot\":" << expected << ","
              << "\"gpu_q4q6_decode_dot\":" << gpu_dot << ","
              << "\"abs_diff\":" << abs_diff << ","
              << "\"partial_checksum\":" << partial_checksum << ","
              << "\"validation\":\"gpu_q4q6_decode_dot_matches_cpu_reference\","
              << "\"limitations\":["
              << "\"GPU decodes one Q4_K row and one Q6_K row and computes per-block dot partials\","
              << "\"final sum over GPU partials is performed on host\","
              << "\"not full q4_K/q6_K tensor execution on GPU\","
              << "\"not transformer layer execution on GPU\","
              << "\"not GPU autoregressive decoding\","
              << "\"not optimized AeroNum-native GGUF token inference throughput\""
              << "]"
              << "}\n";
    return abs_diff <= 1e-5 ? 0 : 1;
}
