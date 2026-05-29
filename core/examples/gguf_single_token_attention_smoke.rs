use aeronum_core::{GgufHeader, GgufQuantizedLogitValue, GgufSingleTokenAttentionOutputSample};
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

fn parse_usize_arg(name: &str, default: usize) -> usize {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_logit_array(values: &[GgufQuantizedLogitValue]) -> String {
    let items = values
        .iter()
        .map(|value| {
            format!(
                "{{\"row_index\":{},\"value\":{:.12}}}",
                value.row_index, value.value
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn attention_sample_json(sample: &GgufSingleTokenAttentionOutputSample) -> String {
    let first_value_projection = sample
        .value_projection
        .logits
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    let first_attention_output = sample
        .attention_output
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_row\":{},",
            "\"norm_tensor\":\"{}\",",
            "\"value_tensor\":\"{}\",",
            "\"value_row_count\":{},",
            "\"value_dimension\":{},",
            "\"rms_epsilon\":{:.8},",
            "\"rms\":{:.12},",
            "\"norm_weight_checksum\":{:.8},",
            "\"normalized_input_checksum\":{:.8},",
            "\"value_projection_checksum\":{:.12},",
            "\"first_value_projection\":{},",
            "\"top_value_projection\":{},",
            "\"output_tensor\":\"{}\",",
            "\"value_repeat_factor\":{},",
            "\"output_row_count\":{},",
            "\"output_dimension\":{},",
            "\"attention_output_count\":{},",
            "\"attention_output_checksum\":{:.12},",
            "\"first_attention_output\":{},",
            "\"top_attention_output\":{}",
            "}}"
        ),
        json_escape(&sample.value_projection.input.name),
        sample.value_projection.input.row_index,
        json_escape(&sample.value_projection.norm_tensor_name),
        json_escape(&sample.value_projection.output_tensor_name),
        sample.value_projection.output_row_count,
        sample.value_projection.dimension,
        sample.value_projection.rms_epsilon,
        sample.value_projection.rms,
        sample.value_projection.norm_weight_checksum,
        sample.value_projection.normalized_input_checksum,
        sample.value_projection.logits_checksum,
        json_logit_array(&first_value_projection),
        json_logit_array(&sample.value_projection.top_logits),
        json_escape(&sample.output_tensor_name),
        sample.value_repeat_factor,
        sample.output_row_count,
        sample.output_dimension,
        sample.attention_output.len(),
        sample.attention_output_checksum,
        json_logit_array(&first_attention_output),
        json_logit_array(&sample.top_attention_output)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_single_token_attention_smoke --model <path> [--input-row <index>] [--layer <index>]"
        );
        std::process::exit(2);
    }
    let input_row = parse_u64_arg("--input-row", 22177);
    let layer = parse_usize_arg("--layer", 0);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let norm_tensor = parse_arg("--norm-tensor", &format!("blk.{layer}.attn_norm.weight"));
    let value_tensor = parse_arg("--value-tensor", &format!("blk.{layer}.attn_v.weight"));
    let output_tensor = parse_arg(
        "--output-tensor",
        &format!("blk.{layer}.attn_output.weight"),
    );

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_single_token_attention_output_sample(
            &input_tensor,
            input_row,
            &norm_tensor,
            &value_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read single-token attention output sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_single_token_attention_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"single-token first-layer CPU attention-output subpath only\",",
            "\"no multi-token attention scores or RoPE validation\",",
            "\"not FFN execution\",",
            "\"not generated-token logits\",",
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
        layer,
        attention_sample_json(&sample)
    );
}
