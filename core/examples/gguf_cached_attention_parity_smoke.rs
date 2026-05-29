use aeronum_core::{GgufCachedAttentionParitySample, GgufHeader};
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

fn sample_json(sample: &GgufCachedAttentionParitySample) -> String {
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"cached_input_rows\":{},",
            "\"query_input_row\":{},",
            "\"full_input_rows\":{},",
            "\"norm_tensor\":\"{}\",",
            "\"query_tensor\":\"{}\",",
            "\"key_tensor\":\"{}\",",
            "\"value_tensor\":\"{}\",",
            "\"output_tensor\":\"{}\",",
            "\"cache_token_count\":{},",
            "\"total_token_count\":{},",
            "\"embedding_dimension\":{},",
            "\"head_count\":{},",
            "\"kv_head_count\":{},",
            "\"head_dimension\":{},",
            "\"value_repeat_factor\":{},",
            "\"rope_freq_base\":{:.6},",
            "\"cached_key_checksum\":{:.8},",
            "\"cached_value_checksum\":{:.8},",
            "\"query_projection_checksum\":{:.8},",
            "\"query_key_checksum\":{:.8},",
            "\"query_value_checksum\":{:.8},",
            "\"rope_query_checksum\":{:.8},",
            "\"rope_key_cache_checksum\":{:.8},",
            "\"full_attention_score_count\":{},",
            "\"full_attention_score_checksum\":{:.12},",
            "\"final_attention_score_count\":{},",
            "\"final_attention_score_checksum\":{:.12},",
            "\"full_last_attention_input_checksum\":{:.8},",
            "\"cached_last_attention_input_checksum\":{:.8},",
            "\"full_attention_output_checksum\":{:.12},",
            "\"cached_attention_output_checksum\":{:.12},",
            "\"attention_output_abs_max_diff\":{:.12},",
            "\"attention_output_checksum_diff\":{:.12},",
            "\"validation\":\"{}\"",
            "}}"
        ),
        json_escape(&sample.full_attention.input_tensor_name),
        json_u64_array(&sample.cached_input_rows),
        sample.query_input_row,
        json_u64_array(&sample.full_attention.input_rows),
        json_escape(&sample.full_attention.norm_tensor_name),
        json_escape(&sample.full_attention.query_tensor_name),
        json_escape(&sample.full_attention.key_tensor_name),
        json_escape(&sample.full_attention.value_tensor_name),
        json_escape(&sample.full_attention.output_tensor_name),
        sample.cache_token_count,
        sample.total_token_count,
        sample.full_attention.embedding_dimension,
        sample.full_attention.head_count,
        sample.full_attention.kv_head_count,
        sample.full_attention.head_dimension,
        sample.full_attention.value_repeat_factor,
        sample.full_attention.rope_freq_base,
        sample.cached_key_checksum,
        sample.cached_value_checksum,
        sample.query_projection_checksum,
        sample.query_key_checksum,
        sample.query_value_checksum,
        sample.rope_query_checksum,
        sample.rope_key_cache_checksum,
        sample.full_attention.attention_score_count,
        sample.full_attention.attention_score_checksum,
        sample.final_attention_score_count,
        sample.final_attention_score_checksum,
        sample.full_attention.last_attention_input_checksum,
        sample.cached_last_attention_input_checksum,
        sample.full_attention.attention_output_checksum,
        sample.cached_attention_output_checksum,
        sample.attention_output_abs_max_diff,
        sample.attention_output_checksum_diff,
        if sample.attention_output_abs_max_diff <= 0.000000001
            && sample.attention_output_checksum_diff <= 0.000001
        {
            "cached_attention_matches_full_context_final_token"
        } else {
            "cached_attention_diff_exceeds_threshold"
        }
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_cached_attention_parity_smoke --model <path> [--cached-rows <a,b>] [--query-row <c>] [--layer <index>]"
        );
        std::process::exit(2);
    }
    let cached_rows = parse_rows_arg("--cached-rows", &[1, 3, 22177]);
    let query_row = parse_arg("--query-row", "4").parse().unwrap_or(4);
    let layer = parse_usize_arg("--layer", 0);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let norm_tensor = parse_arg("--norm-tensor", &format!("blk.{layer}.attn_norm.weight"));
    let query_tensor = parse_arg("--query-tensor", &format!("blk.{layer}.attn_q.weight"));
    let key_tensor = parse_arg("--key-tensor", &format!("blk.{layer}.attn_k.weight"));
    let value_tensor = parse_arg("--value-tensor", &format!("blk.{layer}.attn_v.weight"));
    let output_tensor = parse_arg(
        "--output-tensor",
        &format!("blk.{layer}.attn_output.weight"),
    );

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_cached_attention_parity_sample(
            &input_tensor,
            &cached_rows,
            query_row,
            &norm_tensor,
            &query_tensor,
            &key_tensor,
            &value_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read cached attention parity sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_cached_attention_parity_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"CPU one-layer final-token attention KV-cache parity only\",",
            "\"caches K/V projections for prior tokens and computes only the final query attention\",",
            "\"not FFN execution\",",
            "\"not full transformer KV-cache decoding\",",
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
        sample_json(&sample)
    );
}
