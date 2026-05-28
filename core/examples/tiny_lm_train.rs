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

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.into_iter().map(|x| x / sum).collect()
}

fn cross_entropy(probs: &[f32], target: usize) -> f32 {
    -probs[target].max(1e-9).ln()
}

fn main() {
    let epochs = parse_arg("--epochs", 80);
    let runs = parse_arg("--runs", 5);
    let vocab = 16usize;
    let dim = 8usize;
    let learning_rate = 0.05f32;
    let sequence: Vec<usize> = (0..64).map(|i| (i * 5 + 3) % vocab).collect();
    let total_tokens = epochs * sequence.len() * runs;

    let embeddings = NdArray::from_list(
        (0..vocab * dim)
            .map(|i| ((i % 7) as f32 - 3.0) / 7.0)
            .collect(),
        Some(&[vocab, dim]),
    );
    let mut projection = NdArray::from_list(
        (0..dim * vocab)
            .map(|i| ((i % 11) as f32 - 5.0) / 20.0)
            .collect(),
        Some(&[dim, vocab]),
    );

    let start = Instant::now();
    let mut first_loss = 0.0f32;
    let mut last_loss = 0.0f32;

    for run in 0..runs {
        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            for pos in 0..sequence.len() {
                let token = sequence[pos];
                let prev = if pos == 0 {
                    sequence[sequence.len() - 1]
                } else {
                    sequence[pos - 1]
                };
                let target = sequence[(pos + 1) % sequence.len()];

                let mut hidden = vec![0.0f32; dim];
                for (d, value) in hidden.iter_mut().enumerate() {
                    let current = embeddings.get(&[token, d]).unwrap();
                    let previous = embeddings.get(&[prev, d]).unwrap();
                    *value = 0.75 * current + 0.25 * previous;
                }

                let hidden_row = NdArray::from_list(hidden.clone(), Some(&[1, dim]));
                let logits = hidden_row.matmul(&projection).to_vec();
                let mut probs = softmax(&logits);
                epoch_loss += cross_entropy(&probs, target);
                probs[target] -= 1.0;

                for (d, hidden_value) in hidden.iter().enumerate() {
                    for (v, grad_prob) in probs.iter().enumerate() {
                        let old = projection.get(&[d, v]).unwrap();
                        projection.set(&[d, v], old - learning_rate * hidden_value * grad_prob);
                    }
                }
            }

            let mean_loss = epoch_loss / sequence.len() as f32;
            if run == 0 && epoch == 0 {
                first_loss = mean_loss;
            }
            last_loss = mean_loss;
        }
    }

    let elapsed_seconds = start.elapsed().as_secs_f64();
    let tokens_per_second = total_tokens as f64 / elapsed_seconds.max(1e-9);

    println!(
        "aeronum_tiny_lm_train: epochs={} runs={} total_tokens={} first_loss={:.6} last_loss={:.6} tokens_per_second={:.6}",
        epochs, runs, total_tokens, first_loss, last_loss, tokens_per_second
    );
    println!(
        "{{\"benchmark\":\"aeronum_tiny_lm_train\",\"epochs\":{},\"runs\":{},\"vocab\":{},\"dim\":{},\"sequence_len\":{},\"total_tokens\":{},\"first_loss\":{:.6},\"last_loss\":{:.6},\"loss_decreased\":{},\"elapsed_seconds\":{:.6},\"tokens_per_second\":{:.6},\"training_scope\":\"explicit_gradient_tiny_language_model_not_gpt2\"}}",
        epochs,
        runs,
        vocab,
        dim,
        sequence.len(),
        total_tokens,
        first_loss,
        last_loss,
        last_loss < first_loss,
        elapsed_seconds,
        tokens_per_second
    );
}
