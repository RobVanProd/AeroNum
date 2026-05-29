use aeronum_core::{
    GgufAttentionScoreSample, GgufHeader, GgufMultiTokenAttentionSample, GgufQuantizedLogitValue,
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

fn json_score_array(values: &[GgufAttentionScoreSample]) -> String {
    let items = values
        .iter()
        .map(|value| {
            format!(
                concat!(
                    "{{",
                    "\"query_position\":{},",
                    "\"key_position\":{},",
                    "\"head_index\":{},",
                    "\"kv_head_index\":{},",
                    "\"value\":{:.12}",
                    "}}"
                ),
                value.query_position,
                value.key_position,
                value.head_index,
                value.kv_head_index,
                value.value
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn sample_json(sample: &GgufMultiTokenAttentionSample) -> String {
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
            "\"input_rows\":{},",
            "\"norm_tensor\":\"{}\",",
            "\"query_tensor\":\"{}\",",
            "\"key_tensor\":\"{}\",",
            "\"value_tensor\":\"{}\",",
            "\"output_tensor\":\"{}\",",
            "\"token_count\":{},",
            "\"embedding_dimension\":{},",
            "\"head_count\":{},",
            "\"kv_head_count\":{},",
            "\"head_dimension\":{},",
            "\"value_repeat_factor\":{},",
            "\"rope_freq_base\":{:.6},",
            "\"rms_epsilon\":{:.8},",
            "\"normalized_input_checksum\":{:.8},",
            "\"query_projection_checksum\":{:.8},",
            "\"key_projection_checksum\":{:.8},",
            "\"value_projection_checksum\":{:.8},",
            "\"rope_query_checksum\":{:.8},",
            "\"rope_key_checksum\":{:.8},",
            "\"attention_score_count\":{},",
            "\"attention_score_checksum\":{:.12},",
            "\"top_attention_scores\":{},",
            "\"last_attention_input_count\":{},",
            "\"last_attention_input_checksum\":{:.8},",
            "\"attention_output_count\":{},",
            "\"attention_output_checksum\":{:.12},",
            "\"first_attention_output\":{},",
            "\"top_attention_output\":{}",
            "}}"
        ),
        json_escape(&sample.input_tensor_name),
        json_u64_array(&sample.input_rows),
        json_escape(&sample.norm_tensor_name),
        json_escape(&sample.query_tensor_name),
        json_escape(&sample.key_tensor_name),
        json_escape(&sample.value_tensor_name),
        json_escape(&sample.output_tensor_name),
        sample.token_count,
        sample.embedding_dimension,
        sample.head_count,
        sample.kv_head_count,
        sample.head_dimension,
        sample.value_repeat_factor,
        sample.rope_freq_base,
        sample.rms_epsilon,
        sample.normalized_input_checksum,
        sample.query_projection_checksum,
        sample.key_projection_checksum,
        sample.value_projection_checksum,
        sample.rope_query_checksum,
        sample.rope_key_checksum,
        sample.attention_score_count,
        sample.attention_score_checksum,
        json_score_array(&sample.top_attention_scores),
        sample.last_attention_input_count,
        sample.last_attention_input_checksum,
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
            "usage: gguf_multi_token_attention_smoke --model <path> [--input-rows <a,b,c>] [--layer <index>]"
        );
        std::process::exit(2);
    }
    let input_rows = parse_rows_arg("--input-rows", &[1, 22177, 1044]);
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
        .read_multi_token_attention_sample(
            &input_tensor,
            &input_rows,
            &norm_tensor,
            &query_tensor,
            &key_tensor,
            &value_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read multi-token attention sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_multi_token_attention_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"first-layer CPU multi-token attention subpath only\",",
            "\"RoPE arithmetic is internal AeroNum computation, not external parity\",",
            "\"not FFN execution\",",
            "\"not full transformer execution\",",
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
