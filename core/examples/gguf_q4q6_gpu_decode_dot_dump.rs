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

fn dot_f32(left: &[f32], right: &[f32]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(left, right)| (*left as f64) * (*right as f64))
        .sum()
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
            "usage: gguf_q4q6_gpu_decode_dot_dump --model <path> --q4-bin <path> --q6-bin <path>"
        );
        std::process::exit(2);
    }
    let q4_bin = parse_arg("--q4-bin", "");
    let q6_bin = parse_arg("--q6-bin", "");
    if q4_bin.is_empty() || q6_bin.is_empty() {
        eprintln!("--q4-bin and --q6-bin are required");
        std::process::exit(2);
    }
    let q4_tensor = parse_arg("--q4-tensor", "token_embd.weight");
    let q6_tensor = parse_arg("--q6-tensor", "output.weight");
    let q4_row = parse_u64_arg("--q4-row", 22177);
    let q6_row = parse_u64_arg("--q6-row", 100);

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let q4 = header
        .read_quantized_row_sample(&q4_tensor, q4_row)
        .expect("read q4 GGUF row");
    let q6 = header
        .read_quantized_row_sample(&q6_tensor, q6_row)
        .expect("read q6 GGUF row");
    if q4.tensor_type != 12 || q6.tensor_type != 14 {
        eprintln!("q4 tensor must be Q4_K and q6 tensor must be Q6_K");
        std::process::exit(3);
    }
    if q4.decoded_values.len() != q6.decoded_values.len() {
        eprintln!("decoded row lengths differ");
        std::process::exit(3);
    }
    write_u8_file(&q4_bin, &q4.row_bytes);
    write_u8_file(&q6_bin, &q6.row_bytes);
    let cpu_dot = dot_f32(&q4.decoded_values, &q6.decoded_values);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_q4q6_gpu_decode_dot_dump\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"q4_bin\":\"{}\",",
            "\"q6_bin\":\"{}\",",
            "\"q4\":{},",
            "\"q6\":{},",
            "\"dimension\":{},",
            "\"cpu_dot\":{:.12},",
            "\"validation\":\"dumped_raw_q4k_and_q6k_rows\"",
            "}}"
        ),
        json_escape(&model_path),
        elapsed_ms,
        header.version,
        header.tensors.len(),
        header.metadata.len(),
        header.file_size,
        json_escape(&q4_bin),
        json_escape(&q6_bin),
        row_json(&q4),
        row_json(&q6),
        q4.decoded_values.len(),
        cpu_dot
    );
}
