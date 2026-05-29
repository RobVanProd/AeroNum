use aeronum_core::{GgufMetadataValue, HipRuntime, LlamaModel};
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

fn parse_f32_arg(name: &str, default: f32) -> f32 {
    parse_arg(name, &default.to_string())
        .parse()
        .unwrap_or(default)
}

fn json_escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_string_array(values: &[String]) -> String {
    let items = values
        .iter()
        .map(|value| format!("\"{}\"", json_escape(value)))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_f32_array(values: &[f32]) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_token_id_checks(values: &[(String, Option<u32>)]) -> String {
    let items = values
        .iter()
        .map(|(token, id)| match id {
            Some(id) => format!("{{\"token\":\"{}\",\"id\":{}}}", json_escape(token), id),
            None => format!("{{\"token\":\"{}\",\"id\":null}}", json_escape(token)),
        })
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_u32_array(values: &[u32]) -> String {
    let items = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn json_i32_array(values: &[i32]) -> String {
    let items = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn checksum_f32(values: &[f32]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(idx, value)| (idx as f64 + 1.0) * (*value as f64))
        .sum::<f64>()
}

fn checksum_bytes(value: &str) -> u64 {
    value
        .as_bytes()
        .iter()
        .enumerate()
        .map(|(idx, byte)| (idx as u64 + 1) * (*byte as u64))
        .sum()
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!("usage: gguf_header_smoke --model <path> [--device rocm|cpu|gpu|cuda]");
        std::process::exit(2);
    }

    let device = parse_arg("--device", "rocm");
    let prompt = parse_arg("--prompt", "AeroNum GGUF smoke prompt");
    let max_tokens = parse_usize_arg("--max-tokens", 16);
    let temperature = parse_f32_arg("--temperature", 0.0);

    let start = Instant::now();
    let mut model = LlamaModel::try_load_gguf(&model_path).expect("load GGUF header");
    let header = model.gguf_header.clone().expect("GGUF header present");
    let metadata_keys = header
        .metadata
        .iter()
        .take(12)
        .map(|entry| entry.key.clone())
        .collect::<Vec<_>>();
    let tensor_names = header
        .tensors
        .iter()
        .take(8)
        .map(|tensor| tensor.name.clone())
        .collect::<Vec<_>>();
    let tensor_layout_samples = header
        .tensors
        .iter()
        .take(4)
        .map(|tensor| {
            format!(
                "{{\"name\":\"{}\",\"type\":{},\"dimensions\":[{}],\"relative_offset\":{},\"absolute_offset\":{},\"nbytes\":{}}}",
                json_escape(&tensor.name),
                tensor.tensor_type,
                tensor.dimensions.iter().map(|dim| dim.to_string()).collect::<Vec<_>>().join(","),
                tensor.offset,
                tensor.absolute_offset,
                tensor.nbytes.unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let tensors_with_known_nbytes = header
        .tensors
        .iter()
        .filter(|tensor| tensor.nbytes.is_some())
        .count();
    let max_tensor_end = header
        .tensors
        .iter()
        .filter_map(|tensor| {
            tensor
                .nbytes
                .and_then(|nbytes| tensor.absolute_offset.checked_add(nbytes))
        })
        .max()
        .unwrap_or(0);
    let tensor_layout_within_file = max_tensor_end <= header.file_size;
    let tensor_byte_sample = header
        .read_tensor_prefix("output_norm.weight", 64)
        .expect("read output_norm.weight tensor prefix");
    let tensor_byte_sample_json = format!(
        "{{\"name\":\"{}\",\"type\":{},\"absolute_offset\":{},\"tensor_nbytes\":{},\"bytes_read\":{},\"byte_checksum\":{},\"first_bytes_hex\":{},\"f32_samples\":{}}}",
        json_escape(&tensor_byte_sample.name),
        tensor_byte_sample.tensor_type,
        tensor_byte_sample.absolute_offset,
        tensor_byte_sample.tensor_nbytes,
        tensor_byte_sample.bytes_read,
        tensor_byte_sample.byte_checksum,
        json_string_array(&tensor_byte_sample.first_bytes_hex),
        json_f32_array(&tensor_byte_sample.f32_samples)
    );
    let loaded_weight_index = model
        .load_f32_weight("output_norm.weight")
        .expect("load output_norm.weight F32 tensor into LlamaModel");
    let output_norm = &model.weights[loaded_weight_index];
    let output_norm_values = output_norm.to_vec();
    let output_norm_checksum = checksum_f32(&output_norm_values);
    let output_norm_samples = output_norm_values
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>();
    let output_norm_tensor_json = format!(
        "{{\"name\":\"{}\",\"model_weight_index\":{},\"shape\":[{}],\"len\":{},\"checksum\":{:.8},\"f32_samples\":{}}}",
        json_escape(&model.weight_names[loaded_weight_index]),
        loaded_weight_index,
        output_norm
            .shape()
            .iter()
            .map(|dim| dim.to_string())
            .collect::<Vec<_>>()
            .join(","),
        output_norm.len(),
        output_norm_checksum,
        json_f32_array(&output_norm_samples)
    );
    let rocm_roundtrip_json = match HipRuntime::new(0) {
        Ok(runtime) => {
            let device_name = runtime.device_name().unwrap_or_default();
            match output_norm.to_hip_buffer(&runtime) {
                Ok(buffer) => {
                    let mut roundtrip_values = vec![0.0f32; output_norm_values.len()];
                    match runtime.copy_to_host(&buffer, &mut roundtrip_values) {
                        Ok(()) => {
                            let roundtrip_checksum = checksum_f32(&roundtrip_values);
                            let max_abs_diff = output_norm_values
                                .iter()
                                .zip(roundtrip_values.iter())
                                .map(|(left, right)| (left - right).abs())
                                .fold(0.0f32, f32::max);
                            let roundtrip_samples = roundtrip_values
                                .iter()
                                .take(8)
                                .copied()
                                .collect::<Vec<_>>();
                            format!(
                                "{{\"attempted\":true,\"success\":true,\"device_id\":{},\"device_name\":\"{}\",\"tensor_name\":\"output_norm.weight\",\"elements\":{},\"bytes\":{},\"roundtrip_checksum\":{:.8},\"max_abs_diff\":{:.8},\"f32_samples\":{}}}",
                                runtime.device_id(),
                                json_escape(&device_name),
                                output_norm.len(),
                                buffer.size_bytes(),
                                roundtrip_checksum,
                                max_abs_diff,
                                json_f32_array(&roundtrip_samples)
                            )
                        }
                        Err(err) => format!(
                            "{{\"attempted\":true,\"success\":false,\"device_id\":{},\"device_name\":\"{}\",\"tensor_name\":\"output_norm.weight\",\"elements\":{},\"bytes\":{},\"error\":\"{}\"}}",
                            runtime.device_id(),
                            json_escape(&device_name),
                            output_norm.len(),
                            buffer.size_bytes(),
                            json_escape(&err.to_string())
                        ),
                    }
                }
                Err(err) => format!(
                    "{{\"attempted\":true,\"success\":false,\"device_id\":{},\"device_name\":\"{}\",\"tensor_name\":\"output_norm.weight\",\"elements\":{},\"bytes\":0,\"error\":\"{}\"}}",
                    runtime.device_id(),
                    json_escape(&device_name),
                    output_norm.len(),
                    json_escape(&err.to_string())
                ),
            }
        }
        Err(err) => format!(
            "{{\"attempted\":true,\"success\":false,\"device_id\":0,\"device_name\":\"\",\"tensor_name\":\"output_norm.weight\",\"elements\":{},\"bytes\":0,\"error\":\"{}\"}}",
            output_norm.len(),
            json_escape(&err.to_string())
        ),
    };
    let architecture = header
        .metadata_value("general.architecture")
        .map(|value| value.summary())
        .unwrap_or_default();
    let quantization_version = header
        .metadata_value("general.quantization_version")
        .map(|value| value.summary())
        .unwrap_or_default();
    let tokenizer_model = header
        .metadata_value("tokenizer.ggml.model")
        .map(|value| value.summary())
        .unwrap_or_default();
    let (tokenizer_token_count, tokenizer_token_samples) =
        match header.metadata_value("tokenizer.ggml.tokens") {
            Some(GgufMetadataValue::Array {
                len,
                string_samples,
                ..
            }) => (*len, string_samples.clone()),
            _ => (0, Vec::new()),
        };
    let tokenizer_index = header.tokenizer_index().expect("tokenizer index");
    let tokenizer_merge_count = header
        .string_array_values("tokenizer.ggml.merges")
        .map(|values| values.len())
        .unwrap_or(0);
    let tokenizer_id_checks = ["<unk>", "<s>", "</s>", "[INST]", "[/INST]"]
        .iter()
        .map(|token| {
            (
                (*token).to_string(),
                tokenizer_index.token_to_id.get(*token).copied(),
            )
        })
        .collect::<Vec<_>>();
    let exact_piece_inputs = ["<s>", "[INST]", "[/INST]", "</s>"];
    let exact_piece_ids = tokenizer_index
        .encode_exact_pieces(exact_piece_inputs)
        .expect("encode exact tokenizer pieces");
    let exact_piece_decoded = tokenizer_index
        .decode_ids(&exact_piece_ids)
        .expect("decode exact tokenizer pieces");
    let token_type_values = header
        .i32_array_values("tokenizer.ggml.token_type")
        .expect("tokenizer token_type array");
    let token_type_checks = exact_piece_ids
        .iter()
        .map(|id| {
            usize::try_from(*id)
                .ok()
                .and_then(|idx| token_type_values.get(idx))
                .copied()
                .unwrap_or(i32::MIN)
        })
        .collect::<Vec<_>>();
    let tokenizer_vocab_json = format!(
        "{{\"token_count\":{},\"token_index_count\":{},\"merge_count\":{},\"token_type_count\":{},\"bos_token_id\":{},\"eos_token_id\":{},\"unknown_token_id\":{},\"exact_token_id_checks\":{},\"exact_piece_encode_decode\":{{\"pieces\":{},\"ids\":{},\"decoded\":{},\"token_types\":{}}}}}",
        tokenizer_token_count,
        tokenizer_index.token_count,
        tokenizer_merge_count,
        token_type_values.len(),
        header.u32_value("tokenizer.ggml.bos_token_id").unwrap_or(0),
        header.u32_value("tokenizer.ggml.eos_token_id").unwrap_or(0),
        header.u32_value("tokenizer.ggml.unknown_token_id").unwrap_or(0),
        json_token_id_checks(&tokenizer_id_checks),
        json_string_array(
            &exact_piece_inputs
                .iter()
                .map(|piece| (*piece).to_string())
                .collect::<Vec<_>>()
        ),
        json_u32_array(&exact_piece_ids),
        json_string_array(&exact_piece_decoded),
        json_i32_array(&token_type_checks)
    );
    let chat_template = header.string_value("tokenizer.chat_template").unwrap_or("");
    let tokenizer_config_json = format!(
        "{{\"pre\":\"{}\",\"padding_token_id\":{},\"add_bos_token\":{},\"add_eos_token\":{},\"add_space_prefix\":{},\"chat_template_len\":{},\"chat_template_checksum\":{},\"chat_template_has_inst\":{},\"chat_template_has_available_tools\":{},\"llama_context_length\":{},\"llama_embedding_length\":{},\"llama_block_count\":{},\"llama_feed_forward_length\":{},\"llama_attention_head_count\":{},\"llama_attention_head_count_kv\":{},\"llama_rope_freq_base\":{:.6},\"llama_rms_epsilon\":{:.8}}}",
        json_escape(header.string_value("tokenizer.ggml.pre").unwrap_or("")),
        header.u32_value("tokenizer.ggml.padding_token_id").unwrap_or(0),
        header
            .bool_value("tokenizer.ggml.add_bos_token")
            .unwrap_or(false),
        header
            .bool_value("tokenizer.ggml.add_eos_token")
            .unwrap_or(false),
        header
            .bool_value("tokenizer.ggml.add_space_prefix")
            .unwrap_or(false),
        chat_template.len(),
        checksum_bytes(chat_template),
        chat_template.contains("[INST]"),
        chat_template.contains("AVAILABLE_TOOLS"),
        header.u32_value("llama.context_length").unwrap_or(0),
        header.u32_value("llama.embedding_length").unwrap_or(0),
        header.u32_value("llama.block_count").unwrap_or(0),
        header.u32_value("llama.feed_forward_length").unwrap_or(0),
        header.u32_value("llama.attention.head_count").unwrap_or(0),
        header.u32_value("llama.attention.head_count_kv").unwrap_or(0),
        header.f32_value("llama.rope.freq_base").unwrap_or(0.0),
        header
            .f32_value("llama.attention.layer_norm_rms_epsilon")
            .unwrap_or(0.0)
    );
    model.to(&device);
    let model_rocm_offload_json = format!(
        "{{\"loaded_weight_count\":{},\"hip_weight_count\":{},\"weight_names\":{}}}",
        model.weights.len(),
        model.hip_weights.len(),
        json_string_array(&model.weight_names)
    );
    let output = model.generate(&prompt, max_tokens, temperature);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{{\"benchmark\":\"aeronum_core_gguf_directory_smoke\",\"model_path\":\"{}\",\"gguf_version\":{},\"tensor_count\":{},\"metadata_kv_count\":{},\"parsed_tensor_infos\":{},\"parsed_metadata_entries\":{},\"alignment\":{},\"data_offset\":{},\"file_size\":{},\"tensors_with_known_nbytes\":{},\"max_tensor_end\":{},\"tensor_layout_within_file\":{},\"tensor_byte_sample\":{},\"loaded_f32_tensor\":{},\"rocm_tensor_roundtrip\":{},\"model_rocm_offload\":{},\"tokenizer_vocab\":{},\"tokenizer_config\":{},\"architecture\":\"{}\",\"quantization_version\":\"{}\",\"tokenizer_model\":\"{}\",\"tokenizer_token_count\":{},\"sample_tokenizer_tokens\":{},\"sample_metadata_keys\":{},\"sample_tensor_names\":{},\"sample_tensor_layouts\":[{}],\"device\":\"{}\",\"max_tokens\":{},\"elapsed_ms\":{:.6},\"output_kind\":\"placeholder\",\"output\":\"{}\"}}",
        json_escape(&model_path),
        header.version,
        header.tensor_count,
        header.metadata_kv_count,
        header.tensors.len(),
        header.metadata.len(),
        header.alignment,
        header.data_offset,
        header.file_size,
        tensors_with_known_nbytes,
        max_tensor_end,
        tensor_layout_within_file,
        tensor_byte_sample_json,
        output_norm_tensor_json,
        rocm_roundtrip_json,
        model_rocm_offload_json,
        tokenizer_vocab_json,
        tokenizer_config_json,
        json_escape(&architecture),
        json_escape(&quantization_version),
        json_escape(&tokenizer_model),
        tokenizer_token_count,
        json_string_array(&tokenizer_token_samples),
        json_string_array(&metadata_keys),
        json_string_array(&tensor_names),
        tensor_layout_samples,
        json_escape(&device),
        max_tokens,
        elapsed_ms,
        json_escape(&output)
    );
}
