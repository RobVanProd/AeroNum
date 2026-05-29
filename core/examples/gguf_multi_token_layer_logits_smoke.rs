use aeronum_core::{GgufHeader, GgufMultiTokenLayerLogitsSample, GgufQuantizedLogitValue};
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

fn sample_json(sample: &GgufMultiTokenLayerLogitsSample) -> String {
    let first_logits = sample.logits.iter().take(8).cloned().collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_rows\":{},",
            "\"last_input_row\":{},",
            "\"token_count\":{},",
            "\"head_count\":{},",
            "\"kv_head_count\":{},",
            "\"head_dimension\":{},",
            "\"attention_score_count\":{},",
            "\"attention_score_checksum\":{:.12},",
            "\"attention_output_count\":{},",
            "\"attention_output_checksum\":{:.12},",
            "\"residual_checksum\":{:.8},",
            "\"ffn_norm_tensor\":\"{}\",",
            "\"ffn_rms_epsilon\":{:.8},",
            "\"ffn_rms\":{:.12},",
            "\"ffn_norm_weight_checksum\":{:.8},",
            "\"ffn_normalized_input_checksum\":{:.8},",
            "\"gate_tensor\":\"{}\",",
            "\"gate_projection_count\":{},",
            "\"gate_projection_checksum\":{:.12},",
            "\"up_tensor\":\"{}\",",
            "\"up_projection_count\":{},",
            "\"up_projection_checksum\":{:.12},",
            "\"activated_count\":{},",
            "\"activated_checksum\":{:.8},",
            "\"down_tensor\":\"{}\",",
            "\"ffn_output_count\":{},",
            "\"ffn_output_checksum\":{:.12},",
            "\"top_ffn_output\":{},",
            "\"layer_output_count\":{},",
            "\"layer_output_checksum\":{:.8},",
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
        json_escape(&sample.attention.input_tensor_name),
        json_u64_array(&sample.attention.input_rows),
        sample.last_input_row,
        sample.attention.token_count,
        sample.attention.head_count,
        sample.attention.kv_head_count,
        sample.attention.head_dimension,
        sample.attention.attention_score_count,
        sample.attention.attention_score_checksum,
        sample.attention.attention_output.len(),
        sample.attention.attention_output_checksum,
        sample.residual_checksum,
        json_escape(&sample.ffn_norm_tensor_name),
        sample.ffn_rms_epsilon,
        sample.ffn_rms,
        sample.ffn_norm_weight_checksum,
        sample.ffn_normalized_input_checksum,
        json_escape(&sample.gate_tensor_name),
        sample.gate_projection_count,
        sample.gate_projection_checksum,
        json_escape(&sample.up_tensor_name),
        sample.up_projection_count,
        sample.up_projection_checksum,
        sample.activated_count,
        sample.activated_checksum,
        json_escape(&sample.down_tensor_name),
        sample.ffn_output.len(),
        sample.ffn_output_checksum,
        json_logit_array(&sample.top_ffn_output),
        sample.layer_output_count,
        sample.layer_output_checksum,
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
            "usage: gguf_multi_token_layer_logits_smoke --model <path> [--input-rows <a,b,c>] [--layer <index>]"
        );
        std::process::exit(2);
    }
    let input_rows = parse_rows_arg("--input-rows", &[1, 22177, 1044]);
    let layer = parse_usize_arg("--layer", 0);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let attn_norm_tensor = parse_arg(
        "--attn-norm-tensor",
        &format!("blk.{layer}.attn_norm.weight"),
    );
    let query_tensor = parse_arg("--query-tensor", &format!("blk.{layer}.attn_q.weight"));
    let key_tensor = parse_arg("--key-tensor", &format!("blk.{layer}.attn_k.weight"));
    let value_tensor = parse_arg("--value-tensor", &format!("blk.{layer}.attn_v.weight"));
    let attn_output_tensor = parse_arg(
        "--attn-output-tensor",
        &format!("blk.{layer}.attn_output.weight"),
    );
    let ffn_norm_tensor = parse_arg("--ffn-norm-tensor", &format!("blk.{layer}.ffn_norm.weight"));
    let gate_tensor = parse_arg("--gate-tensor", &format!("blk.{layer}.ffn_gate.weight"));
    let up_tensor = parse_arg("--up-tensor", &format!("blk.{layer}.ffn_up.weight"));
    let down_tensor = parse_arg("--down-tensor", &format!("blk.{layer}.ffn_down.weight"));
    let final_norm_tensor = parse_arg("--final-norm-tensor", "output_norm.weight");
    let output_tensor = parse_arg("--output-tensor", "output.weight");

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_multi_token_layer_logits_sample(
            &input_tensor,
            &input_rows,
            &attn_norm_tensor,
            &query_tensor,
            &key_tensor,
            &value_tensor,
            &attn_output_tensor,
            &ffn_norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            &final_norm_tensor,
            &output_tensor,
            top_k,
        )
        .expect("read multi-token layer logits sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_multi_token_layer_logits_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"first-layer CPU final-token attention-plus-FFN and final-head logits only\",",
            "\"RoPE arithmetic is internal AeroNum computation, not external parity\",",
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
