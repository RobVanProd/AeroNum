use aeronum_core::LlamaModel;
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
    model.to(&device);
    let output = model.generate(&prompt, max_tokens, temperature);
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        "{{\"benchmark\":\"aeronum_core_gguf_header_smoke\",\"model_path\":\"{}\",\"gguf_version\":{},\"tensor_count\":{},\"metadata_kv_count\":{},\"device\":\"{}\",\"max_tokens\":{},\"elapsed_ms\":{:.6},\"output_kind\":\"placeholder\",\"output\":\"{}\"}}",
        json_escape(&model_path),
        header.version,
        header.tensor_count,
        header.metadata_kv_count,
        json_escape(&device),
        max_tokens,
        elapsed_ms,
        json_escape(&output)
    );
}
