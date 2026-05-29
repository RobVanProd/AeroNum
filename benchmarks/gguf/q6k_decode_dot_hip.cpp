#include <hip/hip_runtime.h>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
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

static std::vector<uint8_t> read_u8_range(const std::string &path, std::uint64_t offset, std::uint64_t byte_count) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        std::cerr << "could not open " << path << "\n";
        std::exit(2);
    }
    file.seekg(static_cast<std::streamoff>(offset));
    if (!file) {
        std::cerr << "could not seek " << path << "\n";
        std::exit(2);
    }
    std::vector<uint8_t> bytes(static_cast<size_t>(byte_count));
    file.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size())) {
        std::cerr << "could not read requested byte range from " << path << "\n";
        std::exit(2);
    }
    return bytes;
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

static bool bool_arg(int argc, char **argv, const std::string &name, bool fallback = false) {
    std::string value = arg_value(argc, argv, name, fallback ? "true" : "false");
    return value == "1" || value == "true" || value == "yes";
}

struct TopLogit {
    int row_index;
    double value;
};

static std::vector<TopLogit> top_k_logits(const std::vector<double> &values, int top_k, int row_start) {
    std::vector<TopLogit> top;
    for (size_t i = 0; i < values.size(); ++i) {
        top.push_back(TopLogit{row_start + static_cast<int>(i), values[i]});
    }
    std::sort(top.begin(), top.end(), [](const TopLogit &left, const TopLogit &right) {
        if (left.value == right.value) {
            return left.row_index < right.row_index;
        }
        return left.value > right.value;
    });
    if (static_cast<int>(top.size()) > top_k) {
        top.resize(top_k);
    }
    return top;
}

static std::string top_logits_json(const std::vector<TopLogit> &values) {
    std::ostringstream out;
    out << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ",";
        }
        out << std::fixed << std::setprecision(12)
            << "{\"row_index\":" << values[i].row_index << ",\"value\":" << values[i].value << "}";
    }
    out << "]";
    return out.str();
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
    int row_index = blockIdx.x / block_count;
    int block_index = blockIdx.x - row_index * block_count;
    int t = threadIdx.x;
    float value = 0.0f;
    if (block_index < block_count && t < 256) {
        const uint8_t *q6_block = rhs + (row_index * block_count + block_index) * 210;
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

__global__ void top1_block_kernel(const float *values, float *top_values, int *top_indices, int row_count, int row_start) {
    __shared__ float scratch_values[256];
    __shared__ int scratch_indices[256];
    int global = blockIdx.x * blockDim.x + threadIdx.x;
    float value = -INFINITY;
    int index = row_start + global;
    if (global < row_count) {
        value = values[global];
    }
    scratch_values[threadIdx.x] = value;
    scratch_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float right_value = scratch_values[threadIdx.x + stride];
            int right_index = scratch_indices[threadIdx.x + stride];
            float left_value = scratch_values[threadIdx.x];
            int left_index = scratch_indices[threadIdx.x];
            if (right_value > left_value || (right_value == left_value && right_index < left_index)) {
                scratch_values[threadIdx.x] = right_value;
                scratch_indices[threadIdx.x] = right_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        top_values[blockIdx.x] = scratch_values[0];
        top_indices[blockIdx.x] = scratch_indices[0];
    }
}

__global__ void top1_final_kernel(const float *block_values, const int *block_indices, float *top_value, int *top_index, int block_count) {
    __shared__ float scratch_values[256];
    __shared__ int scratch_indices[256];
    float value = -INFINITY;
    int index = INT_MAX;
    for (int i = threadIdx.x; i < block_count; i += blockDim.x) {
        float candidate_value = block_values[i];
        int candidate_index = block_indices[i];
        if (candidate_value > value || (candidate_value == value && candidate_index < index)) {
            value = candidate_value;
            index = candidate_index;
        }
    }
    scratch_values[threadIdx.x] = value;
    scratch_indices[threadIdx.x] = index;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            float right_value = scratch_values[threadIdx.x + stride];
            int right_index = scratch_indices[threadIdx.x + stride];
            float left_value = scratch_values[threadIdx.x];
            int left_index = scratch_indices[threadIdx.x];
            if (right_value > left_value || (right_value == left_value && right_index < left_index)) {
                scratch_values[threadIdx.x] = right_value;
                scratch_indices[threadIdx.x] = right_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *top_value = scratch_values[0];
        *top_index = scratch_indices[0];
    }
}

int main(int argc, char **argv) {
    std::string lhs_path = arg_value(argc, argv, "--lhs-bin");
    std::string rhs_path = arg_value(argc, argv, "--rhs-q6k-bin");
    std::string rhs_model_path = arg_value(argc, argv, "--rhs-q6k-model");
    std::uint64_t rhs_offset = std::stoull(arg_value(argc, argv, "--rhs-q6k-offset", "0"));
    std::uint64_t rhs_bytes = std::stoull(arg_value(argc, argv, "--rhs-q6k-bytes", "0"));
    std::string expected_logits_path = arg_value(argc, argv, "--expected-logits-bin");
    int row_start = std::stoi(arg_value(argc, argv, "--row-start", "0"));
    int top_k = std::stoi(arg_value(argc, argv, "--top-k", "5"));
    int device = std::stoi(arg_value(argc, argv, "--device", "0"));
    double expected = std::stod(arg_value(argc, argv, "--expected-dot", "0"));
    bool gpu_final_reduction = bool_arg(argc, argv, "--gpu-final-reduction", false);
    bool gpu_top1 = bool_arg(argc, argv, "--gpu-top1", false);
    if (lhs_path.empty() || (rhs_path.empty() && (rhs_model_path.empty() || rhs_bytes == 0))) {
        std::cerr << "usage: q6k_decode_dot_hip --lhs-bin <path> (--rhs-q6k-bin <path> | --rhs-q6k-model <path> --rhs-q6k-offset <bytes> --rhs-q6k-bytes <bytes>) --expected-dot <value> [--device <id>]\n";
        return 2;
    }
    if (gpu_top1 && !gpu_final_reduction) {
        std::cerr << "--gpu-top1 requires --gpu-final-reduction true\n";
        return 2;
    }

    auto lhs = read_f32_file(lhs_path);
    auto rhs = rhs_path.empty() ? read_u8_range(rhs_model_path, rhs_offset, rhs_bytes) : read_u8_file(rhs_path);
    if (rhs.size() % 210 != 0 || lhs.size() % 256 != 0) {
        std::cerr << "input dimensions do not match Q6_K row layout\n";
        return 2;
    }
    int block_count = static_cast<int>(lhs.size() / 256);
    if (block_count == 0 || (rhs.size() / 210) % block_count != 0) {
        std::cerr << "Q6_K input must contain a whole-number row count matching the input dimension\n";
        return 2;
    }
    int row_count = static_cast<int>((rhs.size() / 210) / block_count);
    auto expected_logits = expected_logits_path.empty() ? std::vector<double>{} : read_f64_file(expected_logits_path);
    if (!expected_logits.empty() && static_cast<int>(expected_logits.size()) != row_count) {
        std::cerr << "expected logits count does not match Q6_K row count\n";
        return 2;
    }
    check_hip(hipSetDevice(device), "hipSetDevice");
    hipDeviceProp_t props{};
    check_hip(hipGetDeviceProperties(&props, device), "hipGetDeviceProperties");

    float *d_lhs = nullptr;
    uint8_t *d_rhs = nullptr;
    float *d_partials = nullptr;
    float *d_output = nullptr;
    check_hip(hipMalloc(&d_lhs, lhs.size() * sizeof(float)), "hipMalloc lhs");
    check_hip(hipMalloc(&d_rhs, rhs.size()), "hipMalloc rhs");
    check_hip(hipMalloc(&d_partials, row_count * block_count * sizeof(float)), "hipMalloc partials");
    check_hip(hipMalloc(&d_output, row_count * sizeof(float)), "hipMalloc output");
    check_hip(hipMemcpy(d_lhs, lhs.data(), lhs.size() * sizeof(float), hipMemcpyHostToDevice), "hipMemcpy lhs");
    check_hip(hipMemcpy(d_rhs, rhs.data(), rhs.size(), hipMemcpyHostToDevice), "hipMemcpy rhs");

    int grid_blocks = row_count * block_count;
    q6k_decode_dot_kernel<<<grid_blocks, 256>>>(d_lhs, d_rhs, d_partials, block_count);
    check_hip(hipGetLastError(), "q6k_decode_dot_kernel");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize");

    std::vector<double> gpu_logits(row_count, 0.0);
    std::vector<float> partials(row_count * block_count);
    int gpu_top1_index = -1;
    double gpu_top1_value = 0.0;
    check_hip(hipMemcpy(partials.data(), d_partials, partials.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy partials");
    check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize copy");
    if (gpu_final_reduction) {
        reduce_partials_kernel<<<row_count, 256>>>(d_partials, d_output, block_count, row_count);
        check_hip(hipGetLastError(), "reduce_partials_kernel");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize reduce");
        if (gpu_top1) {
            int top_block_count = (row_count + 255) / 256;
            float *d_top_values = nullptr;
            int *d_top_indices = nullptr;
            float *d_final_top_value = nullptr;
            int *d_final_top_index = nullptr;
            check_hip(hipMalloc(&d_top_values, top_block_count * sizeof(float)), "hipMalloc top values");
            check_hip(hipMalloc(&d_top_indices, top_block_count * sizeof(int)), "hipMalloc top indices");
            check_hip(hipMalloc(&d_final_top_value, sizeof(float)), "hipMalloc final top value");
            check_hip(hipMalloc(&d_final_top_index, sizeof(int)), "hipMalloc final top index");
            top1_block_kernel<<<top_block_count, 256>>>(d_output, d_top_values, d_top_indices, row_count, row_start);
            check_hip(hipGetLastError(), "top1_block_kernel");
            top1_final_kernel<<<1, 256>>>(d_top_values, d_top_indices, d_final_top_value, d_final_top_index, top_block_count);
            check_hip(hipGetLastError(), "top1_final_kernel");
            check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize top1");
            float top_value = 0.0f;
            int top_index = -1;
            check_hip(hipMemcpy(&top_value, d_final_top_value, sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy top value");
            check_hip(hipMemcpy(&top_index, d_final_top_index, sizeof(int), hipMemcpyDeviceToHost), "hipMemcpy top index");
            check_hip(hipFree(d_top_values), "hipFree top values");
            check_hip(hipFree(d_top_indices), "hipFree top indices");
            check_hip(hipFree(d_final_top_value), "hipFree final top value");
            check_hip(hipFree(d_final_top_index), "hipFree final top index");
            gpu_top1_value = static_cast<double>(top_value);
            gpu_top1_index = top_index;
        }
        std::vector<float> output(row_count);
        check_hip(hipMemcpy(output.data(), d_output, output.size() * sizeof(float), hipMemcpyDeviceToHost), "hipMemcpy output");
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize output");
        for (int row = 0; row < row_count; ++row) {
            gpu_logits[row] = static_cast<double>(output[row]);
        }
    }
    check_hip(hipFree(d_lhs), "hipFree lhs");
    check_hip(hipFree(d_rhs), "hipFree rhs");
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
    auto gpu_top_logits = top_k_logits(gpu_logits, top_k, row_start);
    auto expected_top_logits = expected_logits.empty()
        ? std::vector<TopLogit>{}
        : top_k_logits(expected_logits, top_k, row_start);
    bool top_token_matches = !expected_top_logits.empty() && !gpu_top_logits.empty()
        && expected_top_logits.front().row_index == gpu_top_logits.front().row_index;
    bool gpu_top1_token_matches = !expected_top_logits.empty() && gpu_top1
        && expected_top_logits.front().row_index == gpu_top1_index;

    std::cout << std::fixed << std::setprecision(12)
              << "{"
              << "\"benchmark\":\"q6k_decode_dot_hip\","
              << "\"device_id\":" << device << ","
              << "\"device_name\":\"" << props.name << "\","
              << "\"dimension\":" << lhs.size() << ","
              << "\"q6k_block_count\":" << block_count << ","
              << "\"q6k_row_bytes\":" << rhs.size() << ","
              << "\"q6k_row_count\":" << row_count << ","
              << "\"expected_cpu_dot\":" << expected << ","
              << "\"expected_logits_count\":" << expected_logits.size() << ","
              << "\"gpu_q6k_decode_dot\":" << gpu_dot << ","
              << "\"abs_diff\":" << abs_diff << ","
              << "\"gpu_logits_checksum\":" << logits_checksum << ","
              << "\"gpu_first_logits\":" << first_logits << ","
              << "\"expected_top_logits\":" << top_logits_json(expected_top_logits) << ","
              << "\"gpu_top_logits\":" << top_logits_json(gpu_top_logits) << ","
              << "\"top_token_matches\":" << (top_token_matches ? "true" : "false") << ","
              << "\"gpu_top1_enabled\":" << (gpu_top1 ? "true" : "false") << ","
              << "\"gpu_top1_index\":" << gpu_top1_index << ","
              << "\"gpu_top1_value\":" << gpu_top1_value << ","
              << "\"gpu_top1_token_matches\":" << (gpu_top1_token_matches ? "true" : "false") << ","
              << "\"gpu_final_reduction\":" << (gpu_final_reduction ? "true" : "false") << ","
              << "\"partial_checksum\":" << partial_checksum << ","
              << "\"validation\":\"gpu_q6k_decode_dot_matches_cpu_reference\","
              << "\"limitations\":["
              << "\"GPU decodes one or more Q6_K rows and computes per-block dot partials\","
              << "\"input vector is decoded on CPU before GPU execution\","
              << (gpu_final_reduction ? "\"final sum over GPU partials is performed by a second GPU kernel\"" : "\"final sum over GPU partials is performed on host\"") << ","
              << (gpu_top1 ? "\"top-1 token selection is performed by GPU reduction over the computed logits\"" : "\"top-1 token selection is not performed on GPU\"") << ","
              << "\"not full q4_K/q6_K tensor execution on GPU\","
              << "\"not transformer layer execution on GPU\","
              << "\"not GPU autoregressive decoding\","
              << "\"not optimized AeroNum-native GGUF token inference throughput\""
              << "]"
              << "}\n";
    return abs_diff <= 1e-5 ? 0 : 1;
}
