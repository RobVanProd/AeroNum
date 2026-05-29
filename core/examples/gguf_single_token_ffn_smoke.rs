use aeronum_core::{GgufHeader, GgufQuantizedLogitValue, GgufSingleTokenFfnOutputSample};
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

fn ffn_sample_json(sample: &GgufSingleTokenFfnOutputSample) -> String {
    let first_ffn_output = sample
        .ffn_output
        .iter()
        .take(8)
        .cloned()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_row\":{},",
            "\"attn_norm_tensor\":\"{}\",",
            "\"value_tensor\":\"{}\",",
            "\"attn_output_tensor\":\"{}\",",
            "\"value_repeat_factor\":{},",
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
            "\"first_ffn_output\":{},",
            "\"top_ffn_output\":{}",
            "}}"
        ),
        json_escape(&sample.attention.value_projection.input.name),
        sample.attention.value_projection.input.row_index,
        json_escape(&sample.attention.value_projection.norm_tensor_name),
        json_escape(&sample.attention.value_projection.output_tensor_name),
        json_escape(&sample.attention.output_tensor_name),
        sample.attention.value_repeat_factor,
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
        json_logit_array(&first_ffn_output),
        json_logit_array(&sample.top_ffn_output)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_single_token_ffn_smoke --model <path> [--input-row <index>] [--layer <index>]"
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

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let sample = header
        .read_single_token_ffn_output_sample(
            &input_tensor,
            input_row,
            &attn_norm_tensor,
            &value_tensor,
            &attn_output_tensor,
            &ffn_norm_tensor,
            &gate_tensor,
            &up_tensor,
            &down_tensor,
            top_k,
        )
        .expect("read single-token FFN output sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_single_token_ffn_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"layer\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"single-token first-layer CPU attention-plus-FFN subpath only\",",
            "\"not multi-token attention scores or RoPE validation\",",
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
        ffn_sample_json(&sample)
    );
}
