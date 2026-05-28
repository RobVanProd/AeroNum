use aeronum_core::NdArray;
use std::time::Instant;

fn parse_arg(name: &str, default: usize) -> usize {
    let mut args = std::env::args();
    while let Some(arg) = args.next() {
        if arg == name {
            if let Some(value) = args.next() {
                return value.parse().unwrap_or(default);
            }
        }
    }
    default
}

fn softmax(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp = values
        .iter()
        .map(|value| (value - max).exp())
        .collect::<Vec<_>>();
    let sum = exp.iter().sum::<f32>();
    exp.into_iter().map(|value| value / sum).collect()
}

fn cross_entropy(logits: &[f32], target: usize) -> f32 {
    let probs = softmax(logits);
    -probs[target].max(1e-9).ln()
}

fn tiny_transformer_forward() -> (f32, f32) {
    let seq_len = 4usize;
    let dim = 8usize;
    let vocab = 16usize;
    let tokens = [3usize, 8, 13, 2];
    let targets = [8usize, 13, 2, 7];

    let embeddings = NdArray::from_list(
        (0..vocab * dim)
            .map(|i| ((i % 17) as f32 - 8.0) / 11.0)
            .collect(),
        Some(&[vocab, dim]),
    );
    let positions = NdArray::from_list(
        (0..seq_len * dim)
            .map(|i| ((i % 13) as f32 - 6.0) / 17.0)
            .collect(),
        Some(&[seq_len, dim]),
    );
    let wq = NdArray::from_list(
        (0..dim * dim)
            .map(|i| ((i % 19) as f32 - 9.0) / 23.0)
            .collect(),
        Some(&[dim, dim]),
    );
    let wk = NdArray::from_list(
        (0..dim * dim)
            .map(|i| ((i * 3 % 23) as f32 - 11.0) / 29.0)
            .collect(),
        Some(&[dim, dim]),
    );
    let wv = NdArray::from_list(
        (0..dim * dim)
            .map(|i| ((i * 5 % 29) as f32 - 14.0) / 31.0)
            .collect(),
        Some(&[dim, dim]),
    );
    let wo = NdArray::from_list(
        (0..dim * vocab)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 37.0)
            .collect(),
        Some(&[dim, vocab]),
    );

    let mut input_data = Vec::with_capacity(seq_len * dim);
    for pos in 0..seq_len {
        for d in 0..dim {
            input_data.push(
                embeddings.get(&[tokens[pos], d]).unwrap() + positions.get(&[pos, d]).unwrap(),
            );
        }
    }
    let input = NdArray::from_list(input_data, Some(&[seq_len, dim]));
    let q = input.matmul(&wq);
    let k = input.matmul(&wk);
    let v = input.matmul(&wv);

    let scale = (dim as f32).sqrt();
    let mut context = NdArray::zeros(&[seq_len, dim]);
    for row in 0..seq_len {
        let mut scores = Vec::with_capacity(row + 1);
        for col in 0..=row {
            let mut dot = 0.0f32;
            for d in 0..dim {
                dot += q.get(&[row, d]).unwrap() * k.get(&[col, d]).unwrap();
            }
            scores.push(dot / scale);
        }

        let weights = softmax(&scores);
        for d in 0..dim {
            let mut acc = 0.0f32;
            for col in 0..=row {
                acc += weights[col] * v.get(&[col, d]).unwrap();
            }
            context.set(&[row, d], acc);
        }
    }

    let logits = context.matmul(&wo);
    let mut loss = 0.0f32;
    let mut checksum = 0.0f32;
    for row in 0..seq_len {
        let row_logits = (0..vocab)
            .map(|col| logits.get(&[row, col]).unwrap())
            .collect::<Vec<_>>();
        loss += cross_entropy(&row_logits, targets[row]);
        checksum += row_logits
            .iter()
            .enumerate()
            .map(|(idx, value)| *value * (idx as f32 + 1.0))
            .sum::<f32>();
    }

    (loss / seq_len as f32, checksum)
}

fn main() {
    let repeats = parse_arg("--repeats", 1000);
    let start = Instant::now();
    let mut loss = 0.0f32;
    let mut checksum = 0.0f32;
    for _ in 0..repeats {
        let result = tiny_transformer_forward();
        loss = result.0;
        checksum = result.1;
    }
    let elapsed_seconds = start.elapsed().as_secs_f64();
    let tokens = repeats * 4;
    let tokens_per_second = tokens as f64 / elapsed_seconds.max(1e-9);

    println!(
        "{{\"benchmark\":\"aeronum_tiny_transformer_forward\",\"training_scope\":\"causal_self_attention_lm_forward_not_gpt2_training\",\"repeats\":{},\"sequence_len\":4,\"dim\":8,\"vocab\":16,\"tokens_processed\":{},\"loss\":{:.8},\"checksum\":{:.8},\"elapsed_seconds\":{:.8},\"tokens_per_second\":{:.6}}}",
        repeats,
        tokens,
        loss,
        checksum,
        elapsed_seconds,
        tokens_per_second
    );
}
