use aeronum_core::{GgufGpuQuantizedLogitsSample, GgufHeader, GgufQuantizedLogitValue};
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

fn parse_i32_arg(name: &str, default: i32) -> i32 {
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

fn json_top_logits(values: &[GgufQuantizedLogitValue], pieces: &[String]) -> String {
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

fn sample_json(
    sample: &GgufGpuQuantizedLogitsSample,
    cpu_pieces: &[String],
    gpu_pieces: &[String],
) -> String {
    format!(
        concat!(
            "{{",
            "\"output_tensor_name\":\"{}\",",
            "\"output_row_start\":{},",
            "\"output_row_count\":{},",
            "\"dimension\":{},",
            "\"device_id\":{},",
            "\"device_name\":\"{}\",",
            "\"decoded_matrix_checksum\":{:.12},",
            "\"cpu_logits_checksum\":{:.12},",
            "\"gpu_logits_checksum\":{:.12},",
            "\"logits_abs_max_diff\":{:.12},",
            "\"logits_checksum_diff\":{:.12},",
            "\"top_token_matches\":{},",
            "\"cpu_top_logits\":{},",
            "\"gpu_top_logits\":{}",
            "}}"
        ),
        json_escape(&sample.output_tensor_name),
        sample.output_row_start,
        sample.output_row_count,
        sample.dimension,
        sample.device_id,
        json_escape(&sample.device_name),
        sample.decoded_matrix_checksum,
        sample.cpu_logits_checksum,
        sample.gpu_logits_checksum,
        sample.logits_abs_max_diff,
        sample.logits_checksum_diff,
        sample.top_token_matches,
        json_top_logits(&sample.cpu_top_logits, cpu_pieces),
        json_top_logits(&sample.gpu_top_logits, gpu_pieces)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!("usage: gguf_gpu_final_head_logits_smoke --model <path> [--logit-rows <count>]");
        std::process::exit(2);
    }
    let prompt = parse_arg("--prompt", "<s>[INST]Hello[/INST]");
    let add_bos = parse_bool_arg("--add-bos", false);
    let parse_special = parse_bool_arg("--parse-special", true);
    let layer_start = parse_usize_arg("--layer-start", 0);
    let layer_count = parse_usize_arg("--layers", 40);
    let max_new_tokens = parse_usize_arg("--max-new-tokens", 1);
    let top_k = parse_usize_arg("--top-k", 5);
    let logit_start = parse_u64_arg("--logit-start", 0);
    let logit_rows = parse_u64_arg("--logit-rows", 4096);
    let device_id = parse_i32_arg("--device", 0);
    let retained_runtime = parse_bool_arg("--retained-runtime", false);
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
    let retained = if retained_runtime {
        header.read_multi_layer_retained_kv_runtime_decode_sample(
            &input_tensor,
            &prompt_rows,
            layer_start,
            layer_count,
            &final_norm_tensor,
            &output_tensor,
            top_k,
            max_new_tokens,
        )
    } else {
        header.read_multi_layer_retained_kv_greedy_decode_sample(
            &input_tensor,
            &prompt_rows,
            layer_start,
            layer_count,
            &final_norm_tensor,
            &output_tensor,
            top_k,
            max_new_tokens,
        )
    }
    .expect("read retained KV decode sample");
    let retained_step = retained
        .steps
        .last()
        .expect("retained KV decode must produce at least one step");
    let sample = header
        .read_gpu_quantized_logits_for_values_sample(
            &retained_step.retained_final_normalized_input,
            &output_tensor,
            logit_start,
            logit_rows,
            top_k,
            device_id,
        )
        .expect("read GPU final-head logits sample");
    let cpu_top_ids = sample
        .cpu_top_logits
        .iter()
        .map(|value| value.row_index as u32)
        .collect::<Vec<_>>();
    let gpu_top_ids = sample
        .gpu_top_logits
        .iter()
        .map(|value| value.row_index as u32)
        .collect::<Vec<_>>();
    let cpu_top_pieces = tokenizer
        .decode_ids(&cpu_top_ids)
        .expect("decode CPU top token ids");
    let gpu_top_pieces = tokenizer
        .decode_ids(&gpu_top_ids)
        .expect("decode GPU top token ids");
    let generated_token_ids = retained
        .generated_token_ids
        .iter()
        .map(|token_id| *token_id as u32)
        .collect::<Vec<_>>();
    let generated_token_pieces = tokenizer
        .decode_ids(&generated_token_ids)
        .expect("decode generated token ids");
    let generated_text = tokenizer
        .decode_byte_bpe_text(&generated_token_ids)
        .expect("decode generated token text");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_gpu_final_head_logits_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"prompt\":\"{}\",",
            "\"prompt_token_ids\":{},",
            "\"prompt_token_pieces\":{},",
            "\"generated_token_ids\":{},",
            "\"generated_token_pieces\":{},",
            "\"generated_text\":\"{}\",",
            "\"retained_runtime\":{},",
            "\"retained_full_context_verification\":{},",
            "\"retained_step_index\":{},",
            "\"retained_step_max_logits_abs_diff\":{:.12},",
            "\"retained_step_logits_checksum_diff\":{:.12},",
            "\"retained_step_top_token_matches\":{},",
            "\"gpu_logits\":{},",
            "\"validation\":\"{}\",",
            "\"limitations\":[",
            "\"uses retained CPU transformer state as the final-head input\",",
            "\"retained runtime mode does not run full-context comparison inside the retained decode loop\",",
            "\"decodes GGUF quantized output.weight rows on CPU before GPU execution\",",
            "\"runs hipBLAS SGEMM for a configured output-vocabulary row range only\",",
            "\"not full q4_K/q6_K tensor execution on GPU\",",
            "\"not transformer layer execution on GPU\",",
            "\"not GPU autoregressive decoding\",",
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
        json_u64_array(&retained.generated_token_ids),
        json_string_array(&generated_token_pieces),
        json_escape(&generated_text),
        retained_runtime,
        retained.full_context_verification,
        retained_step.step_index,
        retained_step.logits_abs_max_diff,
        retained_step.logits_checksum_diff,
        retained_step.top_token_matches,
        sample_json(&sample, &cpu_top_pieces, &gpu_top_pieces),
        if sample.top_token_matches {
            "gpu_final_head_logits_match_cpu"
        } else {
            "gpu_final_head_logits_differs_from_cpu"
        }
    );
}
