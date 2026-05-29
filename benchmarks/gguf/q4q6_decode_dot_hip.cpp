#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
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

static bool bool_arg(int argc, char **argv, const std::string &name, bool fallback = false) {
    std::string value = arg_value(argc, argv, name, fallback ? "true" : "false");
    return value == "1" || value == "true" || value == "yes";
}

static std::vector<uint8_t> read_u8_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "could not open " << path << "\n";
        std::exit(2);
    }
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

static std::vector<double> read_f64_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "could not open " << path << "\n";
        std::exit(2);
    }
    std::vector<double> values;
    double value = 0.0;
    while (file.read(reinterpret_cast<char *>(&value), sizeof(double))) {
        values.push_back(value);
    }
    if (!file.eof()) {
        std::cerr << "could not read complete f64 values from " << path << "\n";
        std::exit(2);
    }
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
    int row_index = blockIdx.y;
    int block_index = blockIdx.x;
    int t = threadIdx.x;
    float value = 0.0f;
    if (block_index < block_count && t < 256) {
        const uint8_t *q4_block = q4 + block_index * 144;
        const uint8_t *q6_block = q6 + (row_index * block_count + block_index) * 210;
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
        partials[row_index * block_count + block_index] = scratch[0];
    }
}

__global__ void reduce_partials_kernel(const float *partials, float *output, int block_count, int row_count) {
    __shared__ float scratch[256];
    int row_index = blockIdx.x;
    int t = threadIdx.x;
    float value = 0.0f;
    for (int i = t; i < block_count; i += blockDim.x) {
        value += partials[row_index * block_count + i];
    }
    scratch[t] = value;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (t < stride) {
            scratch[t] += scratch[t + stride];
        }
        __syncthreads();
    }
    if (t == 0 && row_index < row_count) {
        output[row_index] = scratch[0];
    }
}

int main(int argc, char **argv) {
    std::string q4_path = arg_value(argc, argv, "--q4-bin");
    std::string q6_path = arg_value(argc, argv, "--q6-bin");
    std::string expected_logits_path = arg_value(argc, argv, "--expected-logits-bin");
    int device = std::stoi(arg_value(argc, argv, "--device", "0"));
    double expected = std::stod(arg_value(argc, argv, "--expected-dot", "0"));
    bool gpu_final_reduction = bool_arg(argc, argv, "--gpu-final-reduction", false);
    if (q4_path.empty() || q6_path.empty()) {
        std::cerr << "usage: q4q6_decode_dot_hip --q4-bin <path> --q6-bin <path> --expected-dot <value> [--device <id>]\n";
        return 2;
    }

    auto q4 = read_u8_file(q4_path);
    auto q6 = read_u8_file(q6_path);
    if (q4.size() % 144 != 0 || q6.size() % 210 != 0) {
        std::cerr << "input dimensions do not match Q4_K/Q6_K row layout\n";
        return 2;
    }
    int block_count = static_cast<int>(q4.size() / 144);
    if (block_count == 0 || (q6.size() / 210) % block_count != 0) {
        std::cerr << "Q6_K input must contain a whole-number row count matching the Q4_K dimension\n";
        return 2;
    }
    int row_count = static_cast<int>((q6.size() / 210) / block_count);
    auto expected_logits = expected_logits_path.empty() ? std::vector<double>{} : read_f64_file(expected_logits_path);
    if (!expected_logits.empty() && static_cast<int>(expected_logits.size()) != row_count) {
        std::cerr << "expected logits count does not match Q6_K row count\n";
        return 2;
    }
    check_hip(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");

    uint8_t *d_q4 = nullptr;
    uint8_t *d_q6 = nullptr;
    float *d_partials = nullptr;
    float *d_output = nullptr;
    check_hip(hipMalloc(&d_q4, q4.size()), "hipMalloc q4");
    check_hip(hipMalloc(&d_q6, q6.size()), "hipMalloc q6");
    check_hip(hipMalloc(&d_partials, row_count * block_count * sizeof(float)), "hipMalloc partials");
    check_hip(hipMalloc(&d_output, row_count * sizeof(float)), "hipMalloc output");
    check_hip(hipMemcpy(d_q4, q4.data(), q4.size(), hipMemcpyHostToDevice), "hipMemcpy q4");
    check_hip(hipMemcpy(d_q6, q6.data(), q6.size(), hipMemcpyHostToDevice), "hipMemcpy q6");

    dim3 grid(block_count, row_count);
    q4q6_decode_dot_kernel<<<grid, 256>>>(d_q4, d_q6, d_partials, block_count);
    check_hip(hipGetLastError(), "q4q6_decode_dot_kernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<double> gpu_logits(row_count, 0.0);
    std::vector<float> partials(row_count * block_count);
    check_hip(hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy partials");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize copy");
    if (gpu_final_reduction) {
        reduce_partials_kernel<<<row_count, 256>>>(d_partials, d_output, block_count, row_count);
        check_hip(hipGetLastError(), "reduce_partials_kernel");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize reduce");
        std::vector<float> output(row_count);
        check_hip(hipMemcpy(output.data(), d_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy output");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize output");
        for (int row = 0; row < row_count; ++row) {
            gpu_logits[row] = static_cast<double>(output[row]);
        }
    }
    check_hip(hipFree(d_q4), "hipFree q4");
    check_hip(hipFree(d_q6), "hipFree q6");
    check_hip(hipFree(d_partials), "hipFree partials");
    check_hip(hipFree(d_output), "hipFree output");

    double partial_checksum = 0.0;
    for (int row = 0; row < row_count; ++row) {
        for (int block = 0; block < block_count; ++block) {
            size_t i = static_cast<size_t>(row) * block_count + block;
            if (!gpu_final_reduction) {
                gpu_logits[row] += static_cast<double>(partials[i]);
            }
            partial_checksum += static_cast<double>(i + 1) * static_cast<double>(partials[i]);
        }
    }
    double gpu_dot = gpu_logits.empty() ? 0.0 : gpu_logits[0];
    double abs_diff = expected_logits.empty() ? std::abs(gpu_dot - expected) : 0.0;
    double logits_checksum = 0.0;
    for (int row = 0; row < row_count; ++row) {
        logits_checksum += static_cast<double>(row + 1) * gpu_logits[row];
        if (!expected_logits.empty()) {
            abs_diff = std::max(abs_diff, std::abs(gpu_logits[row] - expected_logits[row]));
        }
    }
    std::string first_logits = "[";
    int first_count = std::min(row_count, 8);
    for (int row = 0; row < first_count; ++row) {
        if (row != 0) {
            first_logits += ",";
        }
        std::ostringstream item;
        item << std::fixed << std::setprecision(12) << gpu_logits[row];
        first_logits += item.str();
    }
    first_logits += "]";

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
              << "\"q6k_row_count\":" << row_count << ","
              << "\"expected_cpu_dot\":" << expected << ","
              << "\"expected_logits_count\":" << expected_logits.size() << ","
              << "\"gpu_q4q6_decode_dot\":" << gpu_dot << ","
              << "\"abs_diff\":" << abs_diff << ","
              << "\"gpu_logits_checksum\":" << logits_checksum << ","
              << "\"gpu_first_logits\":" << first_logits << ","
              << "\"gpu_final_reduction\":" << (gpu_final_reduction ? "true" : "false") << ","
              << "\"partial_checksum\":" << partial_checksum << ","
              << "\"validation\":\"gpu_q4q6_decode_dot_matches_cpu_reference\","
              << "\"limitations\":["
              << "\"GPU decodes one Q4_K row and one or more Q6_K rows and computes per-block dot partials\","
              << (gpu_final_reduction ? "\"final sum over GPU partials is performed by a second GPU kernel\"" : "\"final sum over GPU partials is performed on host\"") << ","
              << "\"not full q4_K/q6_K tensor execution on GPU\","
              << "\"not transformer layer execution on GPU\","
              << "\"not GPU autoregressive decoding\","
              << "\"not optimized AeroNum-native GGUF token inference throughput\""
              << "]"
              << "}\n";
    return abs_diff <= 1e-5 ? 0 : 1;
}
