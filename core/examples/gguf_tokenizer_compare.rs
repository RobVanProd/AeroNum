use aeronum_core::GgufHeader;

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

fn json_escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for ch in value.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            ch if ch.is_control() => out.push_str(&format!("\\u{:04x}", ch as u32)),
            ch => out.push(ch),
        }
    }
    out
}

fn json_u32_array(values: &[u32]) -> String {
    let items = values
        .iter()
        .map(|value| value.to_string())
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
}

fn prompt_cases() -> Vec<(&'static str, &'static str)> {
    vec![
        ("hello_world", "Hello world"),
        ("hello_world_punctuation", "Hello, world!"),
        ("leading_space", " Hello world"),
        ("lowercase_sentence", "this is a test"),
        ("artifact_prompt", "AeroNum GGUF tokenizer smoke"),
        ("digits_and_periods", "3.3 3..3"),
        ("contraction", "I've been told"),
        ("umlaut", "\u{00c4}pfel"),
        ("emoji_rocket", "\u{1f680}"),
        ("llama_cpp_text", "this is \u{1f999}.cpp"),
        ("inst_special", "[INST]"),
        ("wrapped_inst_special", "<s>[INST]Hello[/INST]"),
        ("mixed_special_text", "Hello [INST] world"),
        ("available_tools_special", "[AVAILABLE_TOOLS]"),
    ]
}

fn contains_known_special(text: &str) -> bool {
    [
        "<s>",
        "</s>",
        "[INST]",
        "[/INST]",
        "[AVAILABLE_TOOLS]",
        "[/AVAILABLE_TOOLS]",
    ]
    .iter()
    .any(|token| text.contains(token))
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!("usage: gguf_tokenizer_compare --model <path>");
        std::process::exit(2);
    }

    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let tokenizer = header.tokenizer_index().expect("tokenizer index");
    let checks = prompt_cases()
        .into_iter()
        .map(|(label, text)| {
            let has_special = contains_known_special(text);
            let with_bos = tokenizer
                .encode_byte_bpe_with_special(text, true, true)
                .expect("encode prompt with BOS");
            let without_bos = tokenizer
                .encode_byte_bpe_with_special(text, false, true)
                .expect("encode prompt without BOS");
            let no_parse_with_bos = if has_special {
                tokenizer.encode_byte_bpe_literal(text, true)
            } else {
                tokenizer.encode_byte_bpe_with_special(text, true, false)
            }
            .expect("encode literal prompt with BOS");
            let no_parse_without_bos = if has_special {
                tokenizer.encode_byte_bpe_literal(text, false)
            } else {
                tokenizer.encode_byte_bpe_with_special(text, false, false)
            }
            .expect("encode literal prompt without BOS");
            format!(
                "{{\"label\":\"{}\",\"text\":\"{}\",\"has_special\":{},\"with_bos\":{},\"without_bos\":{},\"no_parse_with_bos\":{},\"no_parse_without_bos\":{}}}",
                json_escape(label),
                json_escape(text),
                has_special,
                json_u32_array(&with_bos),
                json_u32_array(&without_bos),
                json_u32_array(&no_parse_with_bos),
                json_u32_array(&no_parse_without_bos)
            )
        })
        .collect::<Vec<_>>()
        .join(",");

    println!(
        "{{\"benchmark\":\"aeronum_core_gguf_tokenizer_compare\",\"model_path\":\"{}\",\"tokenizer_model\":\"{}\",\"tokenizer_pre\":\"{}\",\"token_count\":{},\"merge_count\":{},\"prompt_count\":{},\"checks\":[{}]}}",
        json_escape(&model_path),
        json_escape(header.string_value("tokenizer.ggml.model").unwrap_or("")),
        json_escape(header.string_value("tokenizer.ggml.pre").unwrap_or("")),
        tokenizer.token_count,
        tokenizer.merge_ranks.len(),
        prompt_cases().len(),
        checks
    );
}
