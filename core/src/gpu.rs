use std::process::Command;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Cuda,
    Rocm,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HipDeviceProp {
    pub name: [u8; 256],
    pub gcn_arch: u32,
    pub _padding: [u8; 1024],
}

pub trait Device {
    fn backend(&self) -> Backend;
    fn target_triple(&self) -> &'static str;
    fn mcpu(&self) -> &'static str;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuDevice {
    pub backend: Backend,
    pub device_id: i32,
}

impl GpuDevice {
    pub fn new(backend: Backend, device_id: i32) -> Self {
        Self { backend, device_id }
    }

    pub fn auto_detect() -> Self {
        if let Ok(force_backend) = std::env::var("AERONUM_GPU_BACKEND") {
            match force_backend.trim().to_ascii_lowercase().as_str() {
                "rocm" | "amd" => return Self::new(Backend::Rocm, 0),
                "cuda" | "nvidia" => return Self::new(Backend::Cuda, 0),
                _ => {}
            }
        }

        if command_exists("hipconfig") || command_exists("rocminfo") {
            return Self::new(Backend::Rocm, 0);
        }

        Self::new(Backend::Cpu, 0)
    }
}

impl Device for GpuDevice {
    fn backend(&self) -> Backend {
        self.backend
    }

    fn target_triple(&self) -> &'static str {
        match self.backend {
            Backend::Rocm => "amdgcn-amd-amdhsa",
            Backend::Cuda => "nvptx64-nvidia-cuda",
            Backend::Cpu => {
                if cfg!(target_os = "windows") {
                    "x86_64-pc-windows-msvc"
                } else if cfg!(target_os = "macos") {
                    "x86_64-apple-darwin"
                } else {
                    "x86_64-pc-linux-gnu"
                }
            }
        }
    }

    fn mcpu(&self) -> &'static str {
        match self.backend {
            Backend::Rocm => "gfx1101",
            Backend::Cuda => "sm_89",
            Backend::Cpu => "x86_64",
        }
    }
}

fn command_exists(command: &str) -> bool {
    Command::new(command).arg("--version").output().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rocm_profile_metadata_is_stable() {
        let dev = GpuDevice::new(Backend::Rocm, 0);
        assert_eq!(dev.target_triple(), "amdgcn-amd-amdhsa");
        assert_eq!(dev.mcpu(), "gfx1101");
    }
}
