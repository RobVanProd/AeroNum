use aeronum_core::{GgufMetadataValue, LlamaModel};
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
    model.to(&device);
    let output = model.generate(&prompt, max_tokens, temperature);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{{\"benchmark\":\"aeronum_core_gguf_directory_smoke\",\"model_path\":\"{}\",\"gguf_version\":{},\"tensor_count\":{},\"metadata_kv_count\":{},\"parsed_tensor_infos\":{},\"parsed_metadata_entries\":{},\"alignment\":{},\"data_offset\":{},\"file_size\":{},\"tensors_with_known_nbytes\":{},\"max_tensor_end\":{},\"tensor_layout_within_file\":{},\"architecture\":\"{}\",\"quantization_version\":\"{}\",\"tokenizer_model\":\"{}\",\"tokenizer_token_count\":{},\"sample_tokenizer_tokens\":{},\"sample_metadata_keys\":{},\"sample_tensor_names\":{},\"sample_tensor_layouts\":[{}],\"device\":\"{}\",\"max_tokens\":{},\"elapsed_ms\":{:.6},\"output_kind\":\"placeholder\",\"output\":\"{}\"}}",
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
