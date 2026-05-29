use aeronum_core::{
    GgufHeader, GgufLayerExecutionSummary, GgufMultiLayerCachedFinalLogitsParitySample,
    GgufQuantizedLogitValue,
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

fn sample_json(sample: &GgufMultiLayerCachedFinalLogitsParitySample) -> String {
    let first_cached_logits = sample
        .cached_logits
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"cached_input_rows\":{},",
            "\"query_input_row\":{},",
            "\"full_input_rows\":{},",
            "\"layer_start\":{},",
            "\"layer_count\":{},",
            "\"cache_token_count\":{},",
            "\"total_token_count\":{},",
            "\"embedding_dimension\":{},",
            "\"head_count\":{},",
            "\"kv_head_count\":{},",
            "\"head_dimension\":{},",
            "\"value_repeat_factor\":{},",
            "\"rope_freq_base\":{:.6},",
            "\"full_layer_summaries\":{},",
            "\"cached_layer_summaries\":{},",
            "\"full_final_rms\":{:.12},",
            "\"cached_final_rms\":{:.12},",
            "\"full_final_norm_weight_checksum\":{:.8},",
            "\"cached_final_norm_weight_checksum\":{:.8},",
            "\"full_final_normalized_input_checksum\":{:.8},",
            "\"cached_final_normalized_input_checksum\":{:.8},",
            "\"output_tensor\":\"{}\",",
            "\"logit_count\":{},",
            "\"full_logits_checksum\":{:.12},",
            "\"cached_logits_checksum\":{:.12},",
            "\"logits_abs_max_diff\":{:.12},",
            "\"logits_checksum_diff\":{:.12},",
            "\"top_token_matches\":{},",
            "\"full_top_logits\":{},",
            "\"cached_top_logits\":{},",
            "\"first_cached_logits\":{},",
            "\"validation\":\"{}\"",
            "}}"
        ),
        json_escape(&sample.full_sample.input_tensor_name),
        json_u64_array(&sample.cached_input_rows),
        sample.query_input_row,
        json_u64_array(&sample.full_sample.input_rows),
        sample.full_sample.layer_start,
        sample.full_sample.layer_count,
        sample.cache_token_count,
        sample.total_token_count,
        sample.full_sample.embedding_dimension,
        sample.full_sample.head_count,
        sample.full_sample.kv_head_count,
        sample.full_sample.head_dimension,
        sample.full_sample.value_repeat_factor,
        sample.full_sample.rope_freq_base,
        json_layer_array(&sample.full_sample.layer_summaries),
        json_layer_array(&sample.cached_layer_summaries),
        sample.full_sample.final_rms,
        sample.cached_final_rms,
        sample.full_sample.final_norm_weight_checksum,
        sample.cached_final_norm_weight_checksum,
        sample.full_sample.final_normalized_input_checksum,
        sample.cached_final_normalized_input_checksum,
        json_escape(&sample.full_sample.output_tensor_name),
        sample.cached_logits.len(),
        sample.full_sample.logits_checksum,
        sample.cached_logits_checksum,
        sample.logits_abs_max_diff,
        sample.logits_checksum_diff,
        sample.top_token_matches,
        json_logit_array(&sample.full_sample.top_logits),
        json_logit_array(&sample.cached_top_logits),
        json_logit_array(&first_cached_logits),
        if sample.logits_abs_max_diff <= 0.000000001
            && sample.logits_checksum_diff <= 0.000001
            && sample.top_token_matches
        {
            "cached_final_logits_match_full_context"
        } else {
            "cached_final_logits_diff_exceeds_threshold"
        }
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_multi_layer_cached_final_logits_parity_smoke --model <path> [--cached-rows <a,b>] [--query-row <c>] [--layers <count>]"
        );
        std::process::exit(2);
    }
    let cached_rows = parse_rows_arg("--cached-rows", &[1, 3, 22177]);
    let query_row = parse_arg("--query-row", "4").parse().unwrap_or(4);
    let layer_start = parse_usize_arg("--layer-start", 0);
    let layer_count = parse_usize_arg("--layers", 2);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let final_norm_tensor = parse_arg("--final-norm-tensor", "output_norm.weight");
    let output_tensor = parse_arg("--output-tensor", "output.weight");

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_multi_layer_cached_final_logits_parity_sample(
            &input_tensor,
            &cached_rows,
            query_row,
            layer_start,
            layer_count,
            &final_norm_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read multi-layer cached final logits parity sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_multi_layer_cached_final_logits_parity_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"CPU cached final-token transformer parity for one fixed row sequence only\",",
            "\"computes cached prior-token K/V per layer and propagates one query row\",",
            "\"not autoregressive generated-token decoding\",",
            "\"not llama.cpp KV-cache parity\",",
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
