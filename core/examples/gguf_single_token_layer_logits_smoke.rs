use aeronum_core::{GgufHeader, GgufQuantizedLogitValue, GgufSingleTokenLayerLogitsSample};
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

fn layer_logits_sample_json(sample: &GgufSingleTokenLayerLogitsSample) -> String {
    let first_logits = sample.logits.iter().take(8).cloned().collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_row\":{},",
            "\"attn_output_count\":{},",
            "\"ffn_output_count\":{},",
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
        json_escape(&sample.ffn.attention.value_projection.input.name),
        sample.ffn.attention.value_projection.input.row_index,
        sample.ffn.attention.attention_output.len(),
        sample.ffn.ffn_output.len(),
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
            "usage: gguf_single_token_layer_logits_smoke --model <path> [--input-row <index>] [--layer <index>]"
        );
        std::process::exit(2);
    }
    let input_row = parse_u64_arg("--input-row", 22177);
    let layer = parse_usize_arg("--layer", 0);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let attn_norm_tensor = parse_arg(
        "--attn-norm-tensor",
        &format!("blk.{layer}.attn_norm.weight"),
    );
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
        .read_single_token_layer_logits_sample(
            &input_tensor,
            input_row,
            &attn_norm_tensor,
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
        .expect("read single-token layer logits sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_single_token_layer_logits_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"single-token first-layer hidden-state final-head CPU logits only\",",
            "\"not full transformer execution\",",
            "\"not generated-token logits\",",
            "\"not multi-token attention scores or RoPE validation\",",
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
        layer_logits_sample_json(&sample)
    );
}
