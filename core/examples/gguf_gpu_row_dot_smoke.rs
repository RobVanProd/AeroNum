use aeronum_core::{GgufHeader, GgufQuantizedRowSample, HipBlas, HipRuntime};
use std::time::Instant;

fn parse_arg(name: &str, default: &str) -> String {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            if let Some(value) = args.next() {
                return value;
            }
        }
    }
    default.to_string()
}

fn parse_u64_arg(name: &str, default: u64) -> u64 {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_i32_arg(name: &str, default: i32) -> i32 {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn checksum_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64 + 1.0) * (*value as f64))
        .sum()
}

fn dot_f32(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (*left as f64) * (*right as f64))
        .sum()
}

fn first_values_json(values: &[f32]) -> String {
    let items = values
        .iter()
        .take(8)
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn row_json(row: &GgufQuantizedRowSample) -> String {
    format!(
        concat!(
            "{{",
            "\"name\":\"{}\",",
            "\"tensor_type\":{},",
            "\"tensor_type_name\":\"{}\",",
            "\"row_index\":{},",
            "\"row_count\":{},",
            "\"column_count\":{},",
            "\"absolute_offset\":{},",
            "\"row_nbytes\":{},",
            "\"block_count\":{},",
            "\"block_size\":{},",
            "\"type_size\":{},",
            "\"row_byte_checksum\":{},",
            "\"decoded_count\":{},",
            "\"decoded_checksum\":{:.12},",
            "\"first_values\":{}",
            "}}"
        ),
        json_escape(&row.name),
        row.tensor_type,
        json_escape(&row.tensor_type_name),
        row.row_index,
        row.row_count,
        row.column_count,
        row.absolute_offset,
        row.row_nbytes,
        row.block_count,
        row.block_size,
        row.type_size,
        row.row_byte_checksum,
        row.decoded_values.len(),
        row.decoded_checksum,
        first_values_json(&row.decoded_values)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_gpu_row_dot_smoke --model <path> [--lhs-tensor <name>] [--rhs-tensor <name>]"
        );
        std::process::exit(2);
    }
    let lhs_tensor = parse_arg("--lhs-tensor", "token_embd.weight");
    let rhs_tensor = parse_arg("--rhs-tensor", "output.weight");
    let lhs_row = parse_u64_arg("--lhs-row", 22177);
    let rhs_row = parse_u64_arg("--rhs-row", 100);
    let device_id = parse_i32_arg("--device", 0);

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let lhs = header
        .read_quantized_row_sample(&lhs_tensor, lhs_row)
        .expect("read lhs GGUF row");
    let rhs = header
        .read_quantized_row_sample(&rhs_tensor, rhs_row)
        .expect("read rhs GGUF row");
    if lhs.decoded_values.len() != rhs.decoded_values.len() {
        eprintln!("decoded row lengths differ");
        std::process::exit(3);
    }

    let cpu_dot = dot_f32(&lhs.decoded_values, &rhs.decoded_values);
    let runtime = HipRuntime::new(device_id).expect("create HIP runtime");
    let device_name = runtime.device_name().unwrap_or_default();
    let blas = HipBlas::new(&runtime).expect("create hipBLAS handle");
    let d_lhs = runtime
        .copy_to_device(&lhs.decoded_values)
        .expect("copy lhs to device");
    let d_rhs = runtime
        .copy_to_device(&rhs.decoded_values)
        .expect("copy rhs to device");
    let d_out = runtime
        .copy_to_device(&[0.0f32])
        .expect("copy output to device");
    blas.sgemm(
        1,
        1,
        lhs.decoded_values.len() as i32,
        &d_lhs,
        &d_rhs,
        &d_out,
    )
    .expect("hipBLAS row dot SGEMM");
    runtime.synchronize().expect("synchronize hipBLAS row dot");
    let mut gpu_output = [0.0f32; 1];
    runtime
        .copy_to_host(&d_out, &mut gpu_output)
        .expect("copy GPU dot to host");
    runtime.synchronize().expect("synchronize copy to host");

    let gpu_dot = gpu_output[0] as f64;
    let abs_diff = (gpu_dot - cpu_dot).abs();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_gpu_row_dot_smoke\",",
            "\"model_path\":\"{}\",",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"device_id\":{},",
            "\"device_name\":\"{}\",",
            "\"backend\":\"aeronum_core_hipblas\",",
            "\"kernel\":\"sgemm_1x1_row_dot\",",
            "\"lhs\":{},",
            "\"rhs\":{},",
            "\"dimension\":{},",
            "\"lhs_checksum\":{:.12},",
            "\"rhs_checksum\":{:.12},",
            "\"cpu_dot\":{:.12},",
            "\"gpu_dot\":{:.12},",
            "\"abs_diff\":{:.12},",
            "\"elapsed_ms\":{:.6},",
            "\"validation\":\"gpu_dot_matches_cpu_dot\",",
            "\"limitations\":[",
            "\"decodes GGUF quantized rows on CPU before GPU execution\",",
            "\"runs one hipBLAS SGEMM dot product on decoded row vectors only\",",
            "\"not full q4_K/q6_K tensor execution on GPU\",",
            "\"not transformer layer execution on GPU\",",
            "\"not AeroNum-native GGUF token inference throughput\"",
            "]",
            "}}"
        ),
        json_escape(&model_path),
        header.version,
        header.tensors.len(),
        header.metadata.len(),
        header.file_size,
        runtime.device_id(),
        json_escape(&device_name),
        row_json(&lhs),
        row_json(&rhs),
        lhs.decoded_values.len(),
        checksum_f32(&lhs.decoded_values),
        checksum_f32(&rhs.decoded_values),
        cpu_dot,
        gpu_dot,
        abs_diff,
        elapsed_ms
    );
}
