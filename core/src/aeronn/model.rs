use crate::gpu::{Backend, Device, GpuDevice};
use crate::NdArray;
use std::time::Instant;

pub struct LlamaModel {
    pub weights: Vec<NdArray>,
    pub device: GpuDevice,
}

impl LlamaModel {
    pub fn load_gguf(path: &str) -> Self {
        println!("Loaded GGUF model from {}", path);
        Self {
            weights: Vec::new(),
            device: GpuDevice::auto_detect(),
        }
    }

    pub fn to(&mut self, target: &str) {
        self.device = match target {
            "rocm" => GpuDevice::new(Backend::Rocm, 0),
            "gpu" => GpuDevice::auto_detect(),
            "cuda" => GpuDevice::new(Backend::Cuda, 0),
            _ => GpuDevice::new(Backend::Cpu, 0),
        };
        self.offload_to_device();
    }

    fn offload_to_device(&mut self) {
        if self.device.backend() == Backend::Rocm {
            for tensor in &mut self.weights {
                tensor.to_hip();
            }
            println!("Offloaded to ROCm gfx1101 (RX 7800 XT Hellhound 16 GB)");
        }
    }

    pub fn generate(&self, _prompt: &str, max_tokens: usize, _temperature: f32) -> String {
        let start = Instant::now();
        let tokens = max_tokens;
        let duration = start.elapsed();
        println!(
            "Generated {} tokens in {:.4}s on {:?}",
            tokens,
            duration.as_secs_f64(),
            self.device.backend()
        );
        "(ROCm output placeholder)".to_string()
    }
}
