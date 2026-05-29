use aeronum_core::{
    GgufHeader, GgufLayerExecutionSummary, GgufMultiLayerFinalLogitsSample, GgufQuantizedLogitValue,
};
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

fn parse_usize_arg(name: &str, default: usize) -> usize {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_rows_arg(name: &str, default: &[u64]) -> Vec<u64> {
    let default_value = default
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    parse_arg(name, &default_value)
        .split(',')
        .filter_map(|value| value.trim().parse::<u64>().ok())
        .collect()
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_u64_array(values: &[u64]) -> String {
    let items = values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
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

fn json_layer_array(values: &[GgufLayerExecutionSummary]) -> String {
    let items = values
        .iter()
        .map(|value| {
            format!(
                concat!(
                    "{{",
                    "\"layer_index\":{},",
                    "\"attention_score_count\":{},",
                    "\"attention_score_checksum\":{:.12},",
                    "\"attention_output_checksum\":{:.8},",
                    "\"residual_checksum\":{:.8},",
                    "\"ffn_rms_checksum\":{:.12},",
                    "\"gate_projection_checksum\":{:.8},",
                    "\"up_projection_checksum\":{:.8},",
                    "\"activated_checksum\":{:.8},",
                    "\"ffn_output_checksum\":{:.8},",
                    "\"layer_output_checksum\":{:.8}",
                    "}}"
                ),
                value.layer_index,
                value.attention_score_count,
                value.attention_score_checksum,
                value.attention_output_checksum,
                value.residual_checksum,
                value.ffn_rms_checksum,
                value.gate_projection_checksum,
                value.up_projection_checksum,
                value.activated_checksum,
                value.ffn_output_checksum,
                value.layer_output_checksum
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn sample_json(sample: &GgufMultiLayerFinalLogitsSample) -> String {
    let first_logits = sample.logits.iter().take(8).cloned().collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_rows\":{},",
            "\"layer_start\":{},",
            "\"layer_count\":{},",
            "\"token_count\":{},",
            "\"embedding_dimension\":{},",
            "\"head_count\":{},",
            "\"kv_head_count\":{},",
            "\"head_dimension\":{},",
            "\"value_repeat_factor\":{},",
            "\"rope_freq_base\":{:.6},",
            "\"layer_summaries\":{},",
            "\"final_token_position\":{},",
            "\"final_norm_tensor\":\"{}\",",
            "\"final_rms_epsilon\":{:.8},",
            "\"final_rms\":{:.12},",
            "\"final_norm_weight_checksum\":{:.8},",
            "\"final_normalized_input_checksum\":{:.8},",
            "\"output_tensor\":\"{}\",",
            "\"output_row_count\":{},",
            "\"logit_count\":{},",
            "\"logits_checksum\":{:.12},",
            "\"first_logits\":{},",
            "\"top_logits\":{}",
            "}}"
        ),
        json_escape(&sample.input_tensor_name),
        json_u64_array(&sample.input_rows),
        sample.layer_start,
        sample.layer_count,
        sample.token_count,
        sample.embedding_dimension,
        sample.head_count,
        sample.kv_head_count,
        sample.head_dimension,
        sample.value_repeat_factor,
        sample.rope_freq_base,
        json_layer_array(&sample.layer_summaries),
        sample.final_token_position,
        json_escape(&sample.final_norm_tensor_name),
        sample.final_rms_epsilon,
        sample.final_rms,
        sample.final_norm_weight_checksum,
        sample.final_normalized_input_checksum,
        json_escape(&sample.output_tensor_name),
        sample.output_row_count,
        sample.logits.len(),
        sample.logits_checksum,
        json_logit_array(&first_logits),
        json_logit_array(&sample.top_logits)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_multi_layer_final_logits_smoke --model <path> [--input-rows <a,b,c>] [--layers <count>]"
        );
        std::process::exit(2);
    }
    let input_rows = parse_rows_arg("--input-rows", &[1, 22177, 1044]);
    let layer_start = parse_usize_arg("--layer-start", 0);
    let layer_count = parse_usize_arg("--layers", 2);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let final_norm_tensor = parse_arg("--final-norm-tensor", "output_norm.weight");
    let output_tensor = parse_arg("--output-tensor", "output.weight");

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_multi_layer_final_logits_sample(
            &input_tensor,
            &input_rows,
            layer_start,
            layer_count,
            &final_norm_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read multi-layer final logits sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_multi_layer_final_logits_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"bounded CPU multi-layer final-token transformer subpath only\",",
            "\"RoPE arithmetic is internal AeroNum computation, not external parity\",",
            "\"not full 40-layer transformer execution unless --layers covers all layers\",",
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
        sample_json(&sample)
    );
}
