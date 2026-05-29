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

fn parse_u64_arg(name: &str, default: u64) -> u64 {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn parse_f64_arg(name: &str, default: f64) -> f64 {
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

fn json_sampling_candidate_array(
    values: &[GgufQuantizedLogitValue],
    pieces: &[String],
    probabilities: &[f64],
) -> String {
    let mut cumulative_probability = 0.0f64;
    let items = values
        .iter()
        .zip(pieces.iter())
        .zip(probabilities.iter())
        .map(|((value, piece), probability)| {
            cumulative_probability += *probability;
            format!(
                concat!(
                    "{{",
                    "\"token_id\":{},",
                    "\"piece\":\"{}\",",
                    "\"value\":{:.12},",
                    "\"probability\":{:.12},",
                    "\"cumulative_probability\":{:.12}",
                    "}}"
                ),
                value.row_index,
                json_escape(piece),
                value.value,
                probability,
                cumulative_probability
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

fn splitmix64_next(seed: &mut u64) -> u64 {
    *seed = seed.wrapping_add(0x9E3779B97F4A7C15);
    let mut value = *seed;
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D049BB133111EB);
    value ^ (value >> 31)
}

fn splitmix64_unit(seed: &mut u64) -> f64 {
    let value = splitmix64_next(seed) >> 11;
    (value as f64) * (1.0 / ((1u64 << 53) as f64))
}

fn top_k_probabilities(values: &[GgufQuantizedLogitValue], temperature: f64) -> Vec<f64> {
    if values.is_empty() {
        return Vec::new();
    }
    let max_value = values
        .iter()
        .map(|value| value.value as f64 / temperature)
        .fold(f64::NEG_INFINITY, f64::max);
    let weights = values
        .iter()
        .map(|value| ((value.value as f64 / temperature) - max_value).exp())
        .collect::<Vec<_>>();
    let weight_sum = weights.iter().sum::<f64>();
    weights
        .iter()
        .map(|weight| {
            if weight_sum > 0.0 {
                weight / weight_sum
            } else {
                1.0 / values.len() as f64
            }
        })
        .collect()
}

fn choose_sampled_index(probabilities: &[f64], draw: f64) -> usize {
    let mut cumulative_probability = 0.0f64;
    for (index, probability) in probabilities.iter().enumerate() {
        cumulative_probability += *probability;
        if draw <= cumulative_probability {
            return index;
        }
    }
    probabilities.len().saturating_sub(1)
}

fn decode_step_json(
    step_index: usize,
    decode_mode: &str,
    temperature: f64,
    seed_before: u64,
    seed_after: u64,
    sample_draw: f64,
    context_ids: &[u32],
    context_pieces: &[String],
    generated_token_id: u32,
    generated_piece: &str,
    greedy_token_id: u32,
    greedy_piece: &str,
    sample: &GgufMultiLayerFinalLogitsSample,
    top_token_pieces: &[String],
    sampling_probabilities: &[f64],
    elapsed_ms: f64,
) -> String {
    format!(
        concat!(
            "{{",
            "\"step_index\":{},",
            "\"decode_mode\":\"{}\",",
            "\"temperature\":{:.6},",
            "\"seed_before\":{},",
            "\"seed_after\":{},",
            "\"sample_draw\":{:.12},",
            "\"context_token_ids\":{},",
            "\"context_token_pieces\":{},",
            "\"context_token_count\":{},",
            "\"layer_start\":{},",
            "\"layer_count\":{},",
            "\"layer_summaries\":{},",
            "\"final_token_position\":{},",
            "\"final_rms\":{:.12},",
            "\"logit_count\":{},",
            "\"logits_checksum\":{:.12},",
            "\"greedy_token_id\":{},",
            "\"greedy_token_piece\":\"{}\",",
            "\"selected_token_id\":{},",
            "\"selected_token_piece\":\"{}\",",
            "\"top_next_token_logits\":{},",
            "\"sampling_candidates\":{},",
            "\"elapsed_ms\":{:.6}",
            "}}"
        ),
        step_index,
        json_escape(decode_mode),
        temperature,
        seed_before,
        seed_after,
        sample_draw,
        json_u32_array(context_ids),
        json_string_array(context_pieces),
        sample.token_count,
        sample.layer_start,
        sample.layer_count,
        json_layer_array(&sample.layer_summaries),
        sample.final_token_position,
        sample.final_rms,
        sample.logits.len(),
        sample.logits_checksum,
        greedy_token_id,
        json_escape(greedy_piece),
        generated_token_id,
        json_escape(generated_piece),
        json_top_token_array(&sample.top_logits, top_token_pieces),
        json_sampling_candidate_array(&sample.top_logits, top_token_pieces, sampling_probabilities),
        elapsed_ms
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_prompt_autoregressive_decode_smoke --model <path> [--prompt <text>] [--max-new-tokens <count>]"
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
    let decode_mode = parse_arg("--decode-mode", "greedy");
    if decode_mode != "greedy" && decode_mode != "sample" {
        eprintln!("--decode-mode must be greedy or sample");
        std::process::exit(2);
    }
    let temperature = parse_f64_arg("--temperature", 1.0);
    if !temperature.is_finite() || temperature <= 0.0 {
        eprintln!("--temperature must be a positive finite number");
        std::process::exit(2);
    }
    let initial_seed = parse_u64_arg("--seed", 12345);
    let mut sampling_seed = initial_seed;
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

    for step_index in 0..max_new_tokens {
        let step_start = Instant::now();
        let context_pieces = tokenizer
            .decode_ids(&context_token_ids)
            .expect("decode context tokens");
        let input_rows = context_token_ids
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
            .expect("read autoregressive next-token logits sample");
        let top_token_ids = sample
            .top_logits
            .iter()
            .map(|value| value.row_index as u32)
            .collect::<Vec<_>>();
        let top_token_pieces = tokenizer
            .decode_ids(&top_token_ids)
            .expect("decode top token ids");
        let sampling_probabilities = top_k_probabilities(&sample.top_logits, temperature);
        let seed_before = sampling_seed;
        let sample_draw = if decode_mode == "sample" {
            splitmix64_unit(&mut sampling_seed)
        } else {
            0.0
        };
        let seed_after = sampling_seed;
        let selected_index = if decode_mode == "sample" {
            choose_sampled_index(&sampling_probabilities, sample_draw)
        } else {
            0
        };
        let generated_token_id = top_token_ids[selected_index];
        let generated_piece = top_token_pieces[selected_index].clone();
        let greedy_token_id = top_token_ids[0];
        let greedy_piece = top_token_pieces[0].clone();
        let elapsed_ms = step_start.elapsed().as_secs_f64() * 1000.0;
        step_json_values.push(decode_step_json(
            step_index,
            &decode_mode,
            temperature,
            seed_before,
            seed_after,
            sample_draw,
            &context_token_ids,
            &context_pieces,
            generated_token_id,
            &generated_piece,
            greedy_token_id,
            &greedy_piece,
            &sample,
            &top_token_pieces,
            &sampling_probabilities,
            elapsed_ms,
        ));
        context_token_ids.push(generated_token_id);
        generated_token_ids.push(generated_token_id);
        generated_token_pieces.push(generated_piece);
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
            "\"benchmark\":\"gguf_prompt_autoregressive_decode_smoke\",",
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
            "\"decode_mode\":\"{}\",",
            "\"top_k\":{},",
            "\"temperature\":{:.6},",
            "\"initial_seed\":{},",
            "\"final_seed\":{},",
            "\"generated_token_count\":{},",
            "\"generated_tokens_per_second\":{:.12},",
            "\"generated_token_ids\":{},",
            "\"generated_token_pieces\":{},",
            "\"generated_piece_sequence\":\"{}\",",
            "\"generated_text\":\"{}\",",
            "\"final_context_token_ids\":{},",
            "\"final_context_token_pieces\":{},",
            "\"steps\":[{}],",
            "\"limitations\":[",
            "\"CPU autoregressive token-piece selection for one fixed prompt only\",",
            "\"sample mode samples only from the reported top-k logits, not the full vocabulary\",",
            "\"recomputes the full context for each generated token without KV cache\",",
            "\"generated_piece_sequence is tokenizer piece concatenation, not full detokenization parity\",",
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
        json_escape(&decode_mode),
        top_k,
        temperature,
        initial_seed,
        sampling_seed,
        generated_token_ids.len(),
        generated_tokens_per_second,
        json_u32_array(&generated_token_ids),
        json_string_array(&generated_token_pieces),
        json_escape(&generated_token_pieces.join("")),
        json_escape(&generated_text),
        json_u32_array(&context_token_ids),
        json_string_array(&final_context_pieces),
        step_json_values.join(",")
    );
}
