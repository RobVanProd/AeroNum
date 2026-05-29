use aeronum_core::{GgufHeader, GgufQuantizedBlockSample, GgufQuantizedRowSample};
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

fn json_f32_array(values: &[f32]) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn sample_min(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn sample_max(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn sample_json(sample: &GgufQuantizedBlockSample) -> String {
    let first_values = sample
        .decoded_values
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"name\":\"{}\",",
            "\"tensor_type\":{},",
            "\"tensor_type_name\":\"{}\",",
            "\"absolute_offset\":{},",
            "\"tensor_nbytes\":{},",
            "\"block_size\":{},",
            "\"type_size\":{},",
            "\"block_byte_checksum\":{},",
            "\"decoded_count\":{},",
            "\"decoded_checksum\":{:.8},",
            "\"decoded_min\":{:.8},",
            "\"decoded_max\":{:.8},",
            "\"decoded_first_values\":{}",
            "}}"
        ),
        json_escape(&sample.name),
        sample.tensor_type,
        json_escape(&sample.tensor_type_name),
        sample.absolute_offset,
        sample.tensor_nbytes,
        sample.block_size,
        sample.type_size,
        sample.block_byte_checksum,
        sample.decoded_values.len(),
        sample.decoded_checksum,
        sample_min(&sample.decoded_values),
        sample_max(&sample.decoded_values),
        json_f32_array(&first_values)
    )
}

fn row_sample_json(sample: &GgufQuantizedRowSample) -> String {
    let first_values = sample
        .decoded_values
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>();
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
            "\"decoded_checksum\":{:.8},",
            "\"decoded_min\":{:.8},",
            "\"decoded_max\":{:.8},",
            "\"decoded_first_values\":{}",
            "}}"
        ),
        json_escape(&sample.name),
        sample.tensor_type,
        json_escape(&sample.tensor_type_name),
        sample.row_index,
        sample.row_count,
        sample.column_count,
        sample.absolute_offset,
        sample.row_nbytes,
        sample.block_count,
        sample.block_size,
        sample.type_size,
        sample.row_byte_checksum,
        sample.decoded_values.len(),
        sample.decoded_checksum,
        sample_min(&sample.decoded_values),
        sample_max(&sample.decoded_values),
        json_f32_array(&first_values)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_quantized_block_smoke --model <path> [--q4-row <index>] [--q6-row <index>]"
        );
        std::process::exit(2);
    }
    let q4_row = parse_u64_arg("--q4-row", 22177);
    let q6_row = parse_u64_arg("--q6-row", 100);

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let q4_sample = header
        .read_quantized_block_sample("token_embd.weight")
        .expect("read token_embd.weight Q4_K block sample");
    let q6_sample = header
        .read_quantized_block_sample("output.weight")
        .expect("read output.weight Q6_K block sample");
    let q4_row_sample = header
        .read_quantized_row_sample("token_embd.weight", q4_row)
        .expect("read token_embd.weight Q4_K row sample");
    let q6_row_sample = header
        .read_quantized_row_sample("output.weight", q6_row)
        .expect("read output.weight Q6_K row sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_quantized_block_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"samples\":[{},{}],",
            "\"row_samples\":[{},{}],",
            "\"limitations\":[",
            "\"selected-block and selected-row CPU decode only\",",
            "\"not full tensor execution\",",
            "\"not GPU matmul\",",
            "\"not AeroNum-native GGUF token inference throughput\"",
            "]",
            "}}"
        ),
        json_escape(&model_path),
        elapsed_ms,
        header.version,
        header.tensors.len(),
        header.metadata.len(),
        header.file_size,
        sample_json(&q4_sample),
        sample_json(&q6_sample),
        row_sample_json(&q4_row_sample),
        row_sample_json(&q6_row_sample)
    );
}
