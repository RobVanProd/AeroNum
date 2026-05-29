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

fn parse_bool_arg(name: &str, default: bool) -> bool {
    match parse_arg(name, if default { "true" } else { "false" }).as_str() {
        "1" | "true" | "yes" => true,
        "0" | "false" | "no" => false,
        _ => default,
    }
}

fn parse_usize_arg(name: &str, default: usize) -> usize {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_u32_array(values: &[u32]) -> String {
    let items = values
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_string_array(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_top_token_array(values: &[GgufQuantizedLogitValue], pieces: &[String]) -> String {
    let items = values
        .iter()
        .zip(pieces.iter())
        .map(|(value, piece)| {
            format!(
                concat!(
                    "{{",
                    "\"token_id\":{},",
                    "\"piece\":\"{}\",",
                    "\"value\":{:.12}",
                    "}}"
                ),
                value.row_index,
                json_escape(piece),
                value.value
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
                    "\"layer_output_checksum\":{:.8}",
                    "}}"
                ),
                value.layer_index,
                value.attention_score_count,
                value.attention_score_checksum,
                value.layer_output_checksum
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn sample_json(
    sample: &GgufMultiLayerFinalLogitsSample,
    prompt: &str,
    prompt_token_pieces: &[String],
    top_token_pieces: &[String],
) -> String {
    format!(
        concat!(
            "{{",
            "\"prompt\":\"{}\",",
            "\"prompt_token_ids\":{},",
            "\"prompt_token_pieces\":{},",
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
            "\"top_next_token_logits\":{}",
            "}}"
        ),
        json_escape(prompt),
        json_u32_array(
            &sample
                .input_rows
                .iter()
                .map(|value| *value as u32)
                .collect::<Vec<_>>()
        ),
        json_string_array(prompt_token_pieces),
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
        json_top_token_array(&sample.top_logits, top_token_pieces)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_prompt_next_token_logits_smoke --model <path> [--prompt <text>] [--layers <count>]"
        );
        std::process::exit(2);
    }
    let prompt = parse_arg("--prompt", "<s>[INST]Hello[/INST]");
    let add_bos = parse_bool_arg("--add-bos", false);
    let parse_special = parse_bool_arg("--parse-special", true);
    let layer_start = parse_usize_arg("--layer-start", 0);
    let layer_count = parse_usize_arg("--layers", 40);
    let top_k = parse_usize_arg("--top-k", 5);
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let final_norm_tensor = parse_arg("--final-norm-tensor", "output_norm.weight");
    let output_tensor = parse_arg("--output-tensor", "output.weight");

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let tokenizer = header.tokenizer_index().expect("tokenizer index");
    let prompt_token_ids = tokenizer
        .encode_byte_bpe_with_special(&prompt, add_bos, parse_special)
        .expect("encode prompt");
    let prompt_token_pieces = tokenizer
        .decode_ids(&prompt_token_ids)
        .expect("decode prompt tokens");
    let input_rows = prompt_token_ids
        .iter()
        .map(|token_id| *token_id as u64)
        .collect::<Vec<_>>();
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
        .expect("read prompt next-token logits sample");
    let top_token_ids = sample
        .top_logits
        .iter()
        .map(|value| value.row_index as u32)
        .collect::<Vec<_>>();
    let top_token_pieces = tokenizer
        .decode_ids(&top_token_ids)
        .expect("decode top token ids");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_prompt_next_token_logits_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"add_bos\":{},",
            "\"parse_special\":{},",
            "\"samples\":[{}],",
            "\"limitations\":[",
            "\"CPU prompt next-token logits for one fixed prompt only\",",
            "\"RoPE arithmetic is internal AeroNum computation, not external parity\",",
            "\"not sampled or decoded generated text\",",
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
        add_bos,
        parse_special,
        sample_json(&sample, &prompt, &prompt_token_pieces, &top_token_pieces)
    );
}
