use aeronum_core::{GgufHeader, GgufQuantizedRowSample};
use std::fs::File;
use std::io::Write;
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

fn write_f32_file(path: &str, values: &[f32]) {
    let mut file = File::create(path).expect("create f32 output file");
    for value in values {
        file.write_all(&value.to_le_bytes())
            .expect("write f32 output file");
    }
}

fn write_u8_file(path: &str, values: &[u8]) {
    let mut file = File::create(path).expect("create byte output file");
    file.write_all(values).expect("write byte output file");
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
            "\"decoded_checksum\":{:.12}",
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
        row.decoded_checksum
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_q6k_gpu_decode_dot_dump --model <path> --lhs-bin <path> --rhs-q6k-bin <path>"
        );
        std::process::exit(2);
    }
    let lhs_bin = parse_arg("--lhs-bin", "");
    let rhs_q6k_bin = parse_arg("--rhs-q6k-bin", "");
    if lhs_bin.is_empty() || rhs_q6k_bin.is_empty() {
        eprintln!("--lhs-bin and --rhs-q6k-bin are required");
        std::process::exit(2);
    }
    let lhs_tensor = parse_arg("--lhs-tensor", "token_embd.weight");
    let rhs_tensor = parse_arg("--rhs-tensor", "output.weight");
    let lhs_row = parse_u64_arg("--lhs-row", 22177);
    let rhs_row = parse_u64_arg("--rhs-row", 100);

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let lhs = header
        .read_quantized_row_sample(&lhs_tensor, lhs_row)
        .expect("read lhs GGUF row");
    let rhs = header
        .read_quantized_row_sample(&rhs_tensor, rhs_row)
        .expect("read rhs GGUF row");
    if rhs.tensor_type != 14 {
        eprintln!("rhs tensor must be Q6_K for this verifier");
        std::process::exit(3);
    }
    if lhs.decoded_values.len() != rhs.decoded_values.len() {
        eprintln!("decoded row lengths differ");
        std::process::exit(3);
    }
    write_f32_file(&lhs_bin, &lhs.decoded_values);
    write_u8_file(&rhs_q6k_bin, &rhs.row_bytes);
    let cpu_dot = dot_f32(&lhs.decoded_values, &rhs.decoded_values);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_q6k_gpu_decode_dot_dump\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"lhs_bin\":\"{}\",",
            "\"rhs_q6k_bin\":\"{}\",",
            "\"lhs\":{},",
            "\"rhs\":{},",
            "\"dimension\":{},",
            "\"lhs_checksum\":{:.12},",
            "\"rhs_decoded_checksum\":{:.12},",
            "\"cpu_dot\":{:.12},",
            "\"validation\":\"dumped_decoded_lhs_and_raw_q6k_rhs\"",
            "}}"
        ),
        json_escape(&model_path),
        elapsed_ms,
        header.version,
        header.tensors.len(),
        header.metadata.len(),
        header.file_size,
        json_escape(&lhs_bin),
        json_escape(&rhs_q6k_bin),
        row_json(&lhs),
        row_json(&rhs),
        lhs.decoded_values.len(),
        checksum_f32(&lhs.decoded_values),
        rhs.decoded_checksum,
        cpu_dot
    );
}
