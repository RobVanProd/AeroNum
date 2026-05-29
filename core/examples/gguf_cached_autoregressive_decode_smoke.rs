use aeronum_core::{
    GgufHeader, GgufMultiLayerCachedFinalLogitsParitySample, GgufQuantizedLogitValue,
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

fn json_u64_array(values: &[u64]) -> String {
    let items = values
        .iter()
        .map(u64::to_string)
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

fn step_json(
    step_index: usize,
    context_token_ids: &[u32],
    context_token_pieces: &[String],
    sample: &GgufMultiLayerCachedFinalLogitsParitySample,
    selected_token_id: u32,
    selected_token_piece: &str,
    top_token_pieces: &[String],
    elapsed_ms: f64,
) -> String {
    format!(
        concat!(
            "{{",
            "\"step_index\":{},",
            "\"context_token_ids\":{},",
            "\"context_token_pieces\":{},",
            "\"cached_input_rows\":{},",
            "\"query_input_row\":{},",
            "\"full_input_rows\":{},",
            "\"layer_start\":{},",
            "\"layer_count\":{},",
            "\"logit_count\":{},",
            "\"full_logits_checksum\":{:.12},",
            "\"cached_logits_checksum\":{:.12},",
            "\"logits_abs_max_diff\":{:.12},",
            "\"logits_checksum_diff\":{:.12},",
            "\"top_token_matches\":{},",
            "\"selected_token_id\":{},",
            "\"selected_token_piece\":\"{}\",",
            "\"top_next_token_logits\":{},",
            "\"elapsed_ms\":{:.6},",
            "\"validation\":\"{}\"",
            "}}"
        ),
        step_index,
        json_u32_array(context_token_ids),
        json_string_array(context_token_pieces),
        json_u64_array(&sample.cached_input_rows),
        sample.query_input_row,
        json_u64_array(&sample.full_sample.input_rows),
        sample.full_sample.layer_start,
        sample.full_sample.layer_count,
        sample.cached_logits.len(),
        sample.full_sample.logits_checksum,
        sample.cached_logits_checksum,
        sample.logits_abs_max_diff,
        sample.logits_checksum_diff,
        sample.top_token_matches,
        selected_token_id,
        json_escape(selected_token_piece),
        json_top_token_array(&sample.cached_top_logits, top_token_pieces),
        elapsed_ms,
        if sample.logits_abs_max_diff <= 0.000000001
            && sample.logits_checksum_diff <= 0.000001
            && sample.top_token_matches
        {
            "cached_step_logits_match_full_context"
        } else {
            "cached_step_logits_diff_exceeds_threshold"
        }
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_cached_autoregressive_decode_smoke --model <path> [--prompt <text>] [--max-new-tokens <count>]"
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
    let mut context_token_ids = prompt_token_ids.clone();
    let mut generated_token_ids = Vec::new();
    let mut generated_token_pieces = Vec::new();
    let mut step_json_values = Vec::new();
    let mut max_logits_abs_diff = 0.0f64;
    let mut max_logits_checksum_diff = 0.0f64;
    let mut all_step_top_tokens_match = true;

    for step_index in 0..max_new_tokens {
        if context_token_ids.len() < 2 {
            eprintln!("cached autoregressive decode requires at least two context tokens");
            std::process::exit(2);
        }
        let step_start = Instant::now();
        let context_pieces = tokenizer
            .decode_ids(&context_token_ids)
            .expect("decode context tokens");
        let cached_rows = context_token_ids[..context_token_ids.len() - 1]
            .iter()
            .map(|token_id| *token_id as u64)
            .collect::<Vec<_>>();
        let query_row = *context_token_ids.last().expect("query token") as u64;
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
            .expect("read cached autoregressive logits sample");
        max_logits_abs_diff = max_logits_abs_diff.max(sample.logits_abs_max_diff);
        max_logits_checksum_diff = max_logits_checksum_diff.max(sample.logits_checksum_diff);
        all_step_top_tokens_match &= sample.top_token_matches;
        let selected_value = sample
            .cached_top_logits
            .first()
            .expect("cached top token for greedy decode");
        let selected_token_id = selected_value.row_index as u32;
        let selected_token_piece = tokenizer
            .decode_ids(&[selected_token_id])
            .expect("decode selected token id")[0]
            .clone();
        let top_token_ids = sample
            .cached_top_logits
            .iter()
            .map(|value| value.row_index as u32)
            .collect::<Vec<_>>();
        let top_token_pieces = tokenizer
            .decode_ids(&top_token_ids)
            .expect("decode top token ids");
        let elapsed_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        step_json_values.push(step_json(
            step_index,
            &context_token_ids,
            &context_pieces,
            &sample,
            selected_token_id,
            &selected_token_piece,
            &top_token_pieces,
            elapsed_ms,
        ));
        context_token_ids.push(selected_token_id);
        generated_token_ids.push(selected_token_id);
        generated_token_pieces.push(selected_token_piece);
    }

    let final_context_pieces = tokenizer
        .decode_ids(&context_token_ids)
        .expect("decode final context tokens");
    let generated_text = tokenizer
        .decode_byte_bpe_text(&generated_token_ids)
        .expect("decode generated token text");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    let generated_tokens_per_second = if elapsed_ms > 0.0 {
        generated_token_ids.len() as f64 / (elapsed_ms / 1000.0)
    } else {
        0.0
    };

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_cached_autoregressive_decode_smoke\",",
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
            "\"CPU cached final-token greedy autoregressive decode for one fixed prompt only\",",
            "\"each generated step verifies cached logits against the full-context path before selection\",",
            "\"cache state is recomputed for verification rather than retained as an optimized runtime cache\",",
            "\"not sampled decoding\",",
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
        json_escape(&prompt),
        json_u32_array(&prompt_token_ids),
        json_string_array(&prompt_token_pieces),
        add_bos,
        parse_special,
        max_new_tokens,
        top_k,
        generated_token_ids.len(),
        generated_tokens_per_second,
        json_u32_array(&generated_token_ids),
        json_string_array(&generated_token_pieces),
        json_escape(&generated_text),
        json_u32_array(&context_token_ids),
        json_string_array(&final_context_pieces),
        max_logits_abs_diff,
        max_logits_checksum_diff,
        all_step_top_tokens_match,
        step_json_values.join(",")
    );
}
