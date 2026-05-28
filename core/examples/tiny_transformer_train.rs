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

fn cross_entropy(probs: &[f32], target: usize) -> f32 {
    -probs[target].max(1e-9).ln()
}

fn fixed_attention_context() -> NdArray {
    let seq_len = 4usize;
    let dim = 8usize;
    let vocab = 16usize;
    let tokens = [3usize, 8, 13, 2];

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
    context
}

fn evaluate_and_train_epoch(
    context: &NdArray,
    projection: &mut NdArray,
    targets: &[usize],
    learning_rate: f32,
) -> (f32, f32) {
    let seq_len = targets.len();
    let dim = context.shape()[1];
    let vocab = projection.shape()[1];
    let mut loss = 0.0f32;

    for row in 0..seq_len {
        let mut logits = vec![0.0f32; vocab];
        for (vocab_idx, logit) in logits.iter_mut().enumerate() {
            for d in 0..dim {
                *logit +=
                    context.get(&[row, d]).unwrap() * projection.get(&[d, vocab_idx]).unwrap();
            }
        }

        let mut probs = softmax(&logits);
        loss += cross_entropy(&probs, targets[row]);
        probs[targets[row]] -= 1.0;

        for d in 0..dim {
            let hidden = context.get(&[row, d]).unwrap();
            for (vocab_idx, grad) in probs.iter().enumerate() {
                let old = projection.get(&[d, vocab_idx]).unwrap();
                projection.set(&[d, vocab_idx], old - learning_rate * hidden * grad);
            }
        }
    }

    let logits = context.matmul(projection);
    let checksum = logits
        .to_vec()
        .iter()
        .enumerate()
        .map(|(idx, value)| *value * (idx as f32 + 1.0))
        .sum::<f32>();
    (loss / seq_len as f32, checksum)
}

fn main() {
    let epochs = parse_arg("--epochs", 80);
    let runs = parse_arg("--runs", 5);
    let dim = 8usize;
    let vocab = 16usize;
    let targets = [8usize, 13, 2, 7];
    let learning_rate = 0.2f32;

    let context = fixed_attention_context();
    let mut projection = NdArray::from_list(
        (0..dim * vocab)
            .map(|i| ((i * 7 % 31) as f32 - 15.0) / 37.0)
            .collect(),
        Some(&[dim, vocab]),
    );

    let start = Instant::now();
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;
    let mut checksum = 0.0f32;
    for run in 0..runs {
        for epoch in 0..epochs {
            let (loss, current_checksum) =
                evaluate_and_train_epoch(&context, &mut projection, &targets, learning_rate);
            if run == 0 && epoch == 0 {
                first_loss = loss;
            }
            last_loss = loss;
            checksum = current_checksum;
        }
    }

    let elapsed_seconds = start.elapsed().as_secs_f64();
    let total_tokens = epochs * runs * targets.len();
    let tokens_per_second = total_tokens as f64 / elapsed_seconds.max(1e-9);

    println!(
        "{{\"benchmark\":\"aeronum_tiny_transformer_train\",\"training_scope\":\"causal_self_attention_lm_output_projection_training_not_gpt2\",\"epochs\":{},\"runs\":{},\"sequence_len\":4,\"dim\":8,\"vocab\":16,\"total_tokens\":{},\"first_loss\":{:.8},\"last_loss\":{:.8},\"loss_decreased\":{},\"checksum\":{:.8},\"elapsed_seconds\":{:.8},\"tokens_per_second\":{:.6}}}",
        epochs,
        runs,
        total_tokens,
        first_loss,
        last_loss,
        last_loss < first_loss,
        checksum,
        elapsed_seconds,
        tokens_per_second
    );
}
