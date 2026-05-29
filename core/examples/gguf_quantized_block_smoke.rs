use aeronum_core::{GgufHeader, GgufQuantizedBlockSample};
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

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!("usage: gguf_quantized_block_smoke --model <path>");
        std::process::exit(2);
    }

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let q4_sample = header
        .read_quantized_block_sample("token_embd.weight")
        .expect("read token_embd.weight Q4_K block sample");
    let q6_sample = header
        .read_quantized_block_sample("output.weight")
        .expect("read output.weight Q6_K block sample");
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
            "\"limitations\":[",
            "\"first-block CPU decode only\",",
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
        sample_json(&q6_sample)
    );
}
