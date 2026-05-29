use aeronum_core::{GgufHeader, GgufQuantizedLogitValue, GgufRetainedKvAutoregressiveDecodeSample};
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

fn json_u64_array(values: &[u64]) -> String {
    let items = values
        .iter()
        .map(u64::to_string)
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_usize_array(values: &[usize]) -> String {
    let items = values
        .iter()
        .map(usize::to_string)
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
                "{{\"token_id\":{},\"piece\":\"{}\",\"value\":{:.12}}}",
                value.row_index,
                json_escape(piece),
                value.value
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn steps_json(sample: &GgufRetainedKvAutoregressiveDecodeSample, pieces: &[Vec<String>]) -> String {
    sample
        .steps
        .iter()
        .zip(pieces.iter())
        .map(|(step, top_pieces)| {
            format!(
                concat!(
                    "{{",
                    "\"step_index\":{},",
                    "\"context_input_rows\":{},",
                    "\"query_input_row\":{},",
                    "\"cache_token_counts_before\":{},",
                    "\"cache_token_counts_after\":{},",
                    "\"full_logits_checksum\":{:.12},",
                    "\"retained_logits_checksum\":{:.12},",
                    "\"logits_abs_max_diff\":{:.12},",
                    "\"logits_checksum_diff\":{:.12},",
                    "\"top_token_matches\":{},",
                    "\"selected_token_id\":{},",
                    "\"retained_top_logits\":{},",
                    "\"validation\":\"{}\"",
                    "}}"
                ),
                step.step_index,
                json_u64_array(&step.context_input_rows),
                step.query_input_row,
                json_usize_array(&step.cache_token_counts_before),
                json_usize_array(&step.cache_token_counts_after),
                step.full_sample.logits_checksum,
                step.retained_logits_checksum,
                step.logits_abs_max_diff,
                step.logits_checksum_diff,
                step.top_token_matches,
                step.selected_token_id,
                json_top_token_array(&step.retained_top_logits, top_pieces),
                if step.logits_abs_max_diff <= 0.000000001
                    && step.logits_checksum_diff <= 0.000001
                    && step.top_token_matches
                {
                    "retained_kv_logits_match_full_context"
                } else {
                    "retained_kv_logits_diff_exceeds_threshold"
                }
            )
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_retained_kv_autoregressive_decode_smoke --model <path> [--prompt <text>] [--max-new-tokens <count>]"
        );
        std::process::exit(2);
    }
    let prompt = parse_arg("--prompt", "<s>[INST]Hello[/INST]");
    let add_bos = parse_bool_arg("--add-bos", false);
    let parse_special = parse_bool_arg("--parse-special", true);
    let layer_start = parse_usize_arg("--layer-start", 0);
    let layer_count = parse_usize_arg("--layers", 40);
    let max_new_tokens = parse_usize_arg("--max-new-tokens", 2);
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
    let prompt_rows = prompt_token_ids
        .iter()
        .map(|token_id| *token_id as u64)
        .collect::<Vec<_>>();
    let sample = header
        .read_multi_layer_retained_kv_greedy_decode_sample(
            &input_tensor,
            &prompt_rows,
            layer_start,
            layer_count,
            &final_norm_tensor,
            &output_tensor,
            top_k,
            max_new_tokens,
        )
        .expect("read retained KV autoregressive decode sample");
    let generated_token_ids = sample
        .generated_token_ids
        .iter()
        .map(|token_id| *token_id as u32)
        .collect::<Vec<_>>();
    let generated_token_pieces = tokenizer
        .decode_ids(&generated_token_ids)
        .expect("decode generated tokens");
    let final_context_token_ids = sample
        .final_input_rows
        .iter()
        .map(|token_id| *token_id as u32)
        .collect::<Vec<_>>();
    let final_context_token_pieces = tokenizer
        .decode_ids(&final_context_token_ids)
        .expect("decode final context tokens");
    let generated_text = tokenizer
        .decode_byte_bpe_text(&generated_token_ids)
        .expect("decode generated token text");
    let step_top_pieces = sample
        .steps
        .iter()
        .map(|step| {
            let top_token_ids = step
                .retained_top_logits
                .iter()
                .map(|value| value.row_index as u32)
                .collect::<Vec<_>>();
            tokenizer
                .decode_ids(&top_token_ids)
                .expect("decode retained top token ids")
        })
        .collect::<Vec<_>>();
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens_per_second = if elapsed_ms > 0.0 {
        generated_token_ids.len() as f64 / (elapsed_ms / 1000.0)
    } else {
        0.0
    };

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_retained_kv_autoregressive_decode_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"prompt\":\"{}\",",
            "\"prompt_token_ids\":{},",
            "\"prompt_token_pieces\":{},",
            "\"add_bos\":{},",
            "\"parse_special\":{},",
            "\"max_new_tokens\":{},",
            "\"decode_mode\":\"greedy\",",
            "\"top_k\":{},",
            "\"layer_start\":{},",
            "\"layer_count\":{},",
            "\"prefill_token_count\":{},",
            "\"generated_token_count\":{},",
            "\"generated_tokens_per_second\":{:.12},",
            "\"generated_token_ids\":{},",
            "\"generated_token_pieces\":{},",
            "\"generated_text\":\"{}\",",
            "\"final_context_token_ids\":{},",
            "\"final_context_token_pieces\":{},",
            "\"max_logits_abs_diff\":{:.12},",
            "\"max_logits_checksum_diff\":{:.12},",
            "\"all_step_top_tokens_match\":{},",
            "\"steps\":[{}],",
            "\"limitations\":[",
            "\"CPU retained-KV greedy autoregressive decode for one fixed prompt only\",",
            "\"prefills prior prompt token K/V once and appends per-layer K/V for each decoded query token\",",
            "\"each retained-KV step is verified against the full-context CPU logits path before token selection\",",
            "\"not sampled decoding\",",
            "\"not llama.cpp internal KV-cache parity\",",
            "\"not GPU GGUF execution\",",
            "\"not optimized AeroNum-native GGUF token inference throughput\"",
            "]",
            "}}"
        ),
        json_escape(&model_path),
        elapsed_ms,
        header.version,
        header.tensors.len(),
        header.metadata.len(),
        header.file_size,
        json_escape(&prompt),
        json_u32_array(&prompt_token_ids),
        json_string_array(&prompt_token_pieces),
        add_bos,
        parse_special,
        max_new_tokens,
        top_k,
        layer_start,
        layer_count,
        prompt_token_ids.len() - 1,
        generated_token_ids.len(),
        generated_tokens_per_second,
        json_u32_array(&generated_token_ids),
        json_string_array(&generated_token_pieces),
        json_escape(&generated_text),
        json_u32_array(&final_context_token_ids),
        json_string_array(&final_context_token_pieces),
        sample.max_logits_abs_diff,
        sample.max_logits_checksum_diff,
        sample.all_step_top_tokens_match,
        steps_json(&sample, &step_top_pieces)
    );
}
