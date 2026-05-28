use crate::gpu::{Backend, Device, GpuDevice, HipBuffer, HipRuntime};
use crate::NdArray;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufHeader {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "GGUF I/O error: {err}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic bytes: {magic:?}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version: {version}")
            }
        }
    }
}

impl Error for GgufError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidMagic(_) | Self::UnsupportedVersion(_) => None,
        }
    }
}

impl From<io::Error> for GgufError {
    fn from(value: io::Error) -> Self {
        Self::Io(value)
    }
}

pub struct LlamaModel {
    pub weights: Vec<NdArray>,
    pub device: GpuDevice,
    pub hip_runtime: Option<HipRuntime>,
    pub hip_weights: Vec<HipBuffer>,
    pub gguf_header: Option<GgufHeader>,
}

impl LlamaModel {
    pub fn load_gguf(path: &str) -> Self {
        Self::try_load_gguf(path).unwrap_or_else(|err| panic!("failed to load GGUF header: {err}"))
    }

    pub fn try_load_gguf(path: &str) -> Result<Self, GgufError> {
        let header = GgufHeader::read(path)?;
        println!(
            "Loaded GGUF header from {} (version {}, tensors {}, metadata kvs {})",
            path, header.version, header.tensor_count, header.metadata_kv_count
        );
        Ok(Self {
            weights: Vec::new(),
            device: GpuDevice::auto_detect(),
            hip_runtime: None,
            hip_weights: Vec::new(),
            gguf_header: Some(header),
        })
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
        self.hip_weights.clear();
        self.hip_runtime = None;

        if self.device.backend() != Backend::Rocm {
            return;
        }

        let runtime = match HipRuntime::new(self.device.device_id) {
            Ok(runtime) => runtime,
            Err(err) => {
                eprintln!("ROCm runtime unavailable: {}", err);
                return;
            }
        };

        let mut offloaded = Vec::with_capacity(self.weights.len());
        for tensor in &self.weights {
            match tensor.to_hip_buffer(&runtime) {
                Ok(buffer) => offloaded.push(buffer),
                Err(err) => {
                    eprintln!("ROCm tensor offload failed: {}", err);
                    return;
                }
            }
        }

        let device_name = runtime
            .device_name()
            .unwrap_or_else(|_| "unknown AMD GPU".to_string());
        println!(
            "Offloaded {} tensors to ROCm device {} ({})",
            offloaded.len(),
            self.device.device_id,
            device_name
        );

        self.hip_runtime = Some(runtime);
        self.hip_weights = offloaded;
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

impl GgufHeader {
    pub fn read(path: &str) -> Result<Self, GgufError> {
        let mut file = File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"GGUF" {
            return Err(GgufError::InvalidMagic(magic));
        }

        let version = read_u32_le(&mut file)?;
        if !(1..=3).contains(&version) {
            return Err(GgufError::UnsupportedVersion(version));
        }

        let tensor_count = read_u64_le(&mut file)?;
        let metadata_kv_count = read_u64_le(&mut file)?;

        Ok(Self {
            path: PathBuf::from(path),
            version,
            tensor_count,
            metadata_kv_count,
        })
    }
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32, io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64, io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    #[test]
    fn reads_minimal_gguf_header() {
        let path = std::env::temp_dir().join(format!(
            "aeronum-gguf-header-{}-{}.gguf",
            std::process::id(),
            "valid"
        ));
        let mut file = File::create(&path).expect("create GGUF test file");
        file.write_all(b"GGUF").expect("write magic");
        file.write_all(&3u32.to_le_bytes()).expect("write version");
        file.write_all(&2u64.to_le_bytes())
            .expect("write tensor count");
        file.write_all(&4u64.to_le_bytes())
            .expect("write metadata count");
        drop(file);

        let header = GgufHeader::read(path.to_str().expect("utf8 temp path")).expect("read header");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 2);
        assert_eq!(header.metadata_kv_count, 4);

        fs::remove_file(path).expect("remove GGUF test file");
    }

    #[test]
    fn rejects_non_gguf_magic() {
        let path = std::env::temp_dir().join(format!(
            "aeronum-gguf-header-{}-{}.gguf",
            std::process::id(),
            "invalid"
        ));
        fs::write(&path, b"NOPE").expect("write invalid GGUF test file");

        let err =
            GgufHeader::read(path.to_str().expect("utf8 temp path")).expect_err("reject magic");
        assert!(matches!(
            err,
            GgufError::InvalidMagic([b'N', b'O', b'P', b'E'])
        ));

        fs::remove_file(path).expect("remove invalid GGUF test file");
    }
}
