use aeronum_core::{
    GgufHeader, GgufQuantizedBlockSample, GgufQuantizedLogitValue,
    GgufQuantizedNormalizedLogitsSample, GgufQuantizedPrefixLogitsSample,
    GgufQuantizedRowDotSample, GgufQuantizedRowSample,
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

fn json_f32_array(values: &[f32]) -> String {
    let items = values
        .iter()
        .map(|value| format!("{value:.8}"))
        .collect::<Vec<_>>()
        .join(",");
    format!("[{items}]")
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

fn sample_min(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::INFINITY, f32::min)
}

fn sample_max(values: &[f32]) -> f32 {
    values.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

fn sample_json(sample: &GgufQuantizedBlockSample) -> String {
    let first_values = sample
        .decoded_values
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"name\":\"{}\",",
            "\"tensor_type\":{},",
            "\"tensor_type_name\":\"{}\",",
            "\"absolute_offset\":{},",
            "\"tensor_nbytes\":{},",
            "\"block_size\":{},",
            "\"type_size\":{},",
            "\"block_byte_checksum\":{},",
            "\"decoded_count\":{},",
            "\"decoded_checksum\":{:.8},",
            "\"decoded_min\":{:.8},",
            "\"decoded_max\":{:.8},",
            "\"decoded_first_values\":{}",
            "}}"
        ),
        json_escape(&sample.name),
        sample.tensor_type,
        json_escape(&sample.tensor_type_name),
        sample.absolute_offset,
        sample.tensor_nbytes,
        sample.block_size,
        sample.type_size,
        sample.block_byte_checksum,
        sample.decoded_values.len(),
        sample.decoded_checksum,
        sample_min(&sample.decoded_values),
        sample_max(&sample.decoded_values),
        json_f32_array(&first_values)
    )
}

fn row_sample_json(sample: &GgufQuantizedRowSample) -> String {
    let first_values = sample
        .decoded_values
        .iter()
        .take(8)
        .copied()
        .collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"name\":\"{}\",",
            "\"tensor_type\":{},",
            "\"tensor_type_name\":\"{}\",",
            "\"row_index\":{},",
            "\"row_count\":{},",
            "\"column_count\":{},",
            "\"absolute_offset\":{},",
            "\"row_nbytes\":{},",
            "\"block_count\":{},",
            "\"block_size\":{},",
            "\"type_size\":{},",
            "\"row_byte_checksum\":{},",
            "\"decoded_count\":{},",
            "\"decoded_checksum\":{:.8},",
            "\"decoded_min\":{:.8},",
            "\"decoded_max\":{:.8},",
            "\"decoded_first_values\":{}",
            "}}"
        ),
        json_escape(&sample.name),
        sample.tensor_type,
        json_escape(&sample.tensor_type_name),
        sample.row_index,
        sample.row_count,
        sample.column_count,
        sample.absolute_offset,
        sample.row_nbytes,
        sample.block_count,
        sample.block_size,
        sample.type_size,
        sample.row_byte_checksum,
        sample.decoded_values.len(),
        sample.decoded_checksum,
        sample_min(&sample.decoded_values),
        sample_max(&sample.decoded_values),
        json_f32_array(&first_values)
    )
}

fn row_dot_sample_json(sample: &GgufQuantizedRowDotSample) -> String {
    format!(
        concat!(
            "{{",
            "\"lhs_tensor\":\"{}\",",
            "\"lhs_row\":{},",
            "\"rhs_tensor\":\"{}\",",
            "\"rhs_row\":{},",
            "\"dimension\":{},",
            "\"dot_product\":{:.12},",
            "\"abs_sum\":{:.12},",
            "\"lhs_decoded_checksum\":{:.8},",
            "\"rhs_decoded_checksum\":{:.8}",
            "}}"
        ),
        json_escape(&sample.lhs.name),
        sample.lhs.row_index,
        json_escape(&sample.rhs.name),
        sample.rhs.row_index,
        sample.dimension,
        sample.dot_product,
        sample.abs_sum,
        sample.lhs.decoded_checksum,
        sample.rhs.decoded_checksum
    )
}

fn prefix_logits_sample_json(sample: &GgufQuantizedPrefixLogitsSample) -> String {
    let first_logits = sample.logits.iter().take(8).cloned().collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_row\":{},",
            "\"output_tensor\":\"{}\",",
            "\"output_row_start\":{},",
            "\"output_row_count\":{},",
            "\"dimension\":{},",
            "\"logit_count\":{},",
            "\"logits_checksum\":{:.12},",
            "\"first_logits\":{},",
            "\"top_logits\":{}",
            "}}"
        ),
        json_escape(&sample.input.name),
        sample.input.row_index,
        json_escape(&sample.output_tensor_name),
        sample.output_row_start,
        sample.output_row_count,
        sample.dimension,
        sample.logits.len(),
        sample.logits_checksum,
        json_logit_array(&first_logits),
        json_logit_array(&sample.top_logits)
    )
}

fn normalized_logits_sample_json(sample: &GgufQuantizedNormalizedLogitsSample) -> String {
    let first_logits = sample.logits.iter().take(8).cloned().collect::<Vec<_>>();
    format!(
        concat!(
            "{{",
            "\"input_tensor\":\"{}\",",
            "\"input_row\":{},",
            "\"norm_tensor\":\"{}\",",
            "\"output_tensor\":\"{}\",",
            "\"output_row_start\":{},",
            "\"output_row_count\":{},",
            "\"dimension\":{},",
            "\"rms_epsilon\":{:.8},",
            "\"rms\":{:.12},",
            "\"norm_weight_checksum\":{:.8},",
            "\"normalized_input_checksum\":{:.8},",
            "\"logit_count\":{},",
            "\"logits_checksum\":{:.12},",
            "\"first_logits\":{},",
            "\"top_logits\":{}",
            "}}"
        ),
        json_escape(&sample.input.name),
        sample.input.row_index,
        json_escape(&sample.norm_tensor_name),
        json_escape(&sample.output_tensor_name),
        sample.output_row_start,
        sample.output_row_count,
        sample.dimension,
        sample.rms_epsilon,
        sample.rms,
        sample.norm_weight_checksum,
        sample.normalized_input_checksum,
        sample.logits.len(),
        sample.logits_checksum,
        json_logit_array(&first_logits),
        json_logit_array(&sample.top_logits)
    )
}

fn main() {
    let model_path = parse_arg("--model", "");
    if model_path.is_empty() {
        eprintln!(
            "usage: gguf_quantized_block_smoke --model <path> [--input-tensor <name>] [--output-tensor <name>] [--norm-tensor <name>] [--q4-row <index>] [--q6-row <index>]"
        );
        std::process::exit(2);
    }
    let input_tensor = parse_arg("--input-tensor", "token_embd.weight");
    let output_tensor = parse_arg("--output-tensor", "output.weight");
    let norm_tensor = parse_arg("--norm-tensor", "output_norm.weight");
    let q4_row = parse_u64_arg("--q4-row", 22177);
    let q6_row = parse_u64_arg("--q6-row", 100);
    let logit_start = parse_u64_arg("--logit-start", 0);
    let logit_rows = parse_u64_arg("--logit-rows", 256);
    let top_k = parse_usize_arg("--top-k", 5);

    let start = Instant::now();
    let header = GgufHeader::read(&model_path).expect("read GGUF header");
    let q4_sample = header
        .read_quantized_block_sample(&input_tensor)
        .expect("read input quantized block sample");
    let q6_sample = header
        .read_quantized_block_sample(&output_tensor)
        .expect("read output quantized block sample");
    let q4_row_sample = header
        .read_quantized_row_sample(&input_tensor, q4_row)
        .expect("read input quantized row sample");
    let q6_row_sample = header
        .read_quantized_row_sample(&output_tensor, q6_row)
        .expect("read output quantized row sample");
    let row_dot_sample = header
        .read_quantized_row_dot_sample(&input_tensor, q4_row, &output_tensor, q6_row)
        .expect("read quantized row dot sample");
    let prefix_logits_sample = header
        .read_quantized_prefix_logits_sample(
            &input_tensor,
            q4_row,
            &output_tensor,
            logit_start,
            logit_rows,
            top_k,
        )
        .expect("read quantized prefix logits sample");
    let normalized_logits_sample = header
        .read_quantized_normalized_logits_sample(
            &input_tensor,
            q4_row,
            &norm_tensor,
            &output_tensor,
            logit_start,
            logit_rows,
            top_k,
        )
        .expect("read quantized normalized logits sample");
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

    println!(
        concat!(
            "{{",
            "\"benchmark\":\"gguf_quantized_block_smoke\",",
            "\"model_path\":\"{}\",",
            "\"elapsed_ms\":{:.6},",
            "\"gguf_version\":{},",
            "\"tensor_count\":{},",
            "\"metadata_count\":{},",
            "\"file_size\":{},",
            "\"input_tensor\":\"{}\",",
            "\"output_tensor\":\"{}\",",
            "\"norm_tensor\":\"{}\",",
            "\"samples\":[{},{}],",
            "\"row_samples\":[{},{}],",
            "\"row_dot_samples\":[{}],",
            "\"prefix_logits_samples\":[{}],",
            "\"normalized_logits_samples\":[{}],",
            "\"limitations\":[",
            "\"selected-block selected-row selected-row-dot logit-range and configured RMS-norm CPU decode only\",",
            "\"not full transformer inference or generated-token logits\",",
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
        json_escape(&input_tensor),
        json_escape(&output_tensor),
        json_escape(&norm_tensor),
        sample_json(&q4_sample),
        sample_json(&q6_sample),
        row_sample_json(&q4_row_sample),
        row_sample_json(&q6_row_sample),
        row_dot_sample_json(&row_dot_sample),
        prefix_logits_sample_json(&prefix_logits_sample),
        normalized_logits_sample_json(&normalized_logits_sample)
    );
}
