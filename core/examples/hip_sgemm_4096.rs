use aeronum_core::{HipBlas, HipRuntime};
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

fn summarize(values: &[f64]) -> (f64, f64, f64, f64) {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) * 0.5
    } else {
        sorted[sorted.len() / 2]
    };
    let min = sorted[0];
    let max = sorted[sorted.len() - 1];
    (mean, median, min, max)
}

fn main() {
    let n = parse_arg("--n", 4096);
    let runs = parse_arg("--runs", 10);
    let warmup = parse_arg("--warmup", 3);

    let runtime = HipRuntime::new(0).expect("create HIP runtime");
    let blas = HipBlas::new(&runtime).expect("create hipBLAS handle");
    let elements = n * n;

    let a = vec![1.0f32; elements];
    let b = vec![1.0f32; elements];
    let c = vec![0.0f32; elements];

    let d_a = runtime.copy_to_device(&a).expect("copy A to device");
    let d_b = runtime.copy_to_device(&b).expect("copy B to device");
    let d_c = runtime.copy_to_device(&c).expect("copy C to device");

    for _ in 0..warmup {
        blas.sgemm(n as i32, n as i32, n as i32, &d_a, &d_b, &d_c)
            .expect("warmup SGEMM");
    }
    runtime.synchronize().expect("warmup synchronize");

    let mut run_ms = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        blas.sgemm(n as i32, n as i32, n as i32, &d_a, &d_b, &d_c)
            .expect("measured SGEMM");
        runtime.synchronize().expect("measured synchronize");
        run_ms.push(start.elapsed().as_secs_f64() * 1000.0);
    }

    let mut out = vec![0.0f32; elements];
    runtime
        .copy_to_host(&d_c, &mut out)
        .expect("copy output to host");
    let expected = n as f32;
    let stride = (elements / 4096).max(1);
    let valid = out
        .iter()
        .step_by(stride)
        .all(|value| (*value - expected).abs() < 1e-3);
    assert!(valid, "sampled SGEMM output mismatch");

    let (mean_ms, median_ms, min_ms, max_ms) = summarize(&run_ms);
    let flops = 2.0 * (n as f64).powi(3);
    let median_tflops = (flops / (median_ms / 1000.0)) / 1e12;

    println!(
        "aeronum_core_hipblas_sgemm: n={} runs={} median_ms={:.6} median_tflops={:.6}",
        n, runs, median_ms, median_tflops
    );
    println!(
        "{{\"backend\":\"aeronum_core_hipblas\",\"kernel\":\"sgemm\",\"n\":{},\"runs\":{},\"warmup\":{},\"mean_ms\":{:.6},\"median_ms\":{:.6},\"min_ms\":{:.6},\"max_ms\":{:.6},\"median_tflops\":{:.6},\"validation\":\"sampled_all_ones_expected_n\"}}",
        n, runs, warmup, mean_ms, median_ms, min_ms, max_ms, median_tflops
    );
}
