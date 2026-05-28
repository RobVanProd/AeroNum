use crate::gpu::{Backend, Device, GpuDevice, HipBuffer, HipRuntime};
use crate::NdArray;
use std::error::Error;
use std::fmt;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Clone, Debug, PartialEq)]
pub struct GgufHeader {
    pub path: PathBuf,
    pub version: u32,
    pub tensor_count: u64,
    pub metadata_kv_count: u64,
    pub metadata: Vec<GgufMetadataEntry>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: u64,
    pub data_offset: u64,
    pub file_size: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufMetadataEntry {
    pub key: String,
    pub value: GgufMetadataValue,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GgufMetadataValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array {
        element_type: GgufValueType,
        len: u64,
        string_samples: Vec<String>,
    },
    U64(u64),
    I64(i64),
    F64(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgufValueType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    F32,
    Bool,
    String,
    Array,
    U64,
    I64,
    F64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: u32,
    pub offset: u64,
    pub absolute_offset: u64,
    pub nbytes: Option<u64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufTensorByteSample {
    pub name: String,
    pub tensor_type: u32,
    pub absolute_offset: u64,
    pub tensor_nbytes: u64,
    pub bytes_read: usize,
    pub byte_checksum: u64,
    pub first_bytes_hex: Vec<String>,
    pub f32_samples: Vec<f32>,
}

#[derive(Debug)]
pub enum GgufError {
    Io(io::Error),
    InvalidMagic([u8; 4]),
    UnsupportedVersion(u32),
    InvalidUtf8(String),
    UnsupportedValueType(u32),
    InvalidArrayElementType(u32),
    TensorNotFound(String),
    UnknownTensorByteSize(String),
    InvalidTensorRange(String),
}

impl fmt::Display for GgufError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(err) => write!(f, "GGUF I/O error: {err}"),
            Self::InvalidMagic(magic) => write!(f, "invalid GGUF magic bytes: {magic:?}"),
            Self::UnsupportedVersion(version) => {
                write!(f, "unsupported GGUF version: {version}")
            }
            Self::InvalidUtf8(context) => write!(f, "invalid UTF-8 in GGUF {context}"),
            Self::UnsupportedValueType(value_type) => {
                write!(f, "unsupported GGUF metadata value type: {value_type}")
            }
            Self::InvalidArrayElementType(value_type) => {
                write!(f, "unsupported GGUF array element type: {value_type}")
            }
            Self::TensorNotFound(name) => write!(f, "GGUF tensor not found: {name}"),
            Self::UnknownTensorByteSize(name) => {
                write!(f, "cannot determine byte size for GGUF tensor: {name}")
            }
            Self::InvalidTensorRange(name) => write!(f, "invalid GGUF tensor byte range: {name}"),
        }
    }
}

impl Error for GgufError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidMagic(_)
            | Self::UnsupportedVersion(_)
            | Self::InvalidUtf8(_)
            | Self::UnsupportedValueType(_)
            | Self::InvalidArrayElementType(_)
            | Self::TensorNotFound(_)
            | Self::UnknownTensorByteSize(_)
            | Self::InvalidTensorRange(_) => None,
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
            "Loaded GGUF directory from {} (version {}, tensors {}, metadata kvs {})",
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
        let mut metadata = Vec::with_capacity(metadata_kv_count.min(usize::MAX as u64) as usize);
        for _ in 0..metadata_kv_count {
            metadata.push(GgufMetadataEntry::read(&mut file)?);
        }

        let alignment = metadata
            .iter()
            .find(|entry| entry.key == "general.alignment")
            .and_then(|entry| entry.value.as_u64())
            .unwrap_or(32);

        let mut tensors = Vec::with_capacity(tensor_count.min(usize::MAX as u64) as usize);
        for _ in 0..tensor_count {
            tensors.push(GgufTensorInfo::read(&mut file)?);
        }
        let directory_end = file.stream_position()?;
        let data_offset = align_to(directory_end, alignment);
        let file_size = file.metadata()?.len();

        for tensor in &mut tensors {
            tensor.absolute_offset = data_offset + tensor.offset;
            tensor.nbytes = tensor_nbytes(tensor.tensor_type, &tensor.dimensions);
        }

        Ok(Self {
            path: PathBuf::from(path),
            version,
            tensor_count,
            metadata_kv_count,
            metadata,
            tensors,
            alignment,
            data_offset,
            file_size,
        })
    }

    pub fn metadata_value(&self, key: &str) -> Option<&GgufMetadataValue> {
        self.metadata
            .iter()
            .find(|entry| entry.key == key)
            .map(|entry| &entry.value)
    }

    pub fn read_tensor_prefix(
        &self,
        tensor_name: &str,
        max_bytes: usize,
    ) -> Result<GgufTensorByteSample, GgufError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or_else(|| GgufError::TensorNotFound(tensor_name.to_string()))?;
        let tensor_nbytes = tensor
            .nbytes
            .ok_or_else(|| GgufError::UnknownTensorByteSize(tensor_name.to_string()))?;
        let tensor_end = tensor
            .absolute_offset
            .checked_add(tensor_nbytes)
            .ok_or_else(|| GgufError::InvalidTensorRange(tensor_name.to_string()))?;
        if tensor_end > self.file_size {
            return Err(GgufError::InvalidTensorRange(tensor_name.to_string()));
        }

        let bytes_to_read = max_bytes.min(tensor_nbytes as usize);
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(tensor.absolute_offset))?;
        let mut bytes = vec![0u8; bytes_to_read];
        file.read_exact(&mut bytes)?;
        let byte_checksum = bytes
            .iter()
            .enumerate()
            .map(|(idx, byte)| (idx as u64 + 1) * (*byte as u64))
            .sum();
        let first_bytes_hex = bytes
            .iter()
            .take(16)
            .map(|byte| format!("{byte:02x}"))
            .collect();
        let f32_samples = if tensor.tensor_type == 0 {
            bytes
                .chunks_exact(4)
                .take(8)
                .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect()
        } else {
            Vec::new()
        };

        Ok(GgufTensorByteSample {
            name: tensor.name.clone(),
            tensor_type: tensor.tensor_type,
            absolute_offset: tensor.absolute_offset,
            tensor_nbytes,
            bytes_read: bytes_to_read,
            byte_checksum,
            first_bytes_hex,
            f32_samples,
        })
    }
}

impl GgufMetadataEntry {
    fn read(reader: &mut impl Read) -> Result<Self, GgufError> {
        let key = read_gguf_string(reader, "metadata key")?;
        let value_type = GgufValueType::read(reader)?;
        let value = GgufMetadataValue::read(reader, value_type)?;
        Ok(Self { key, value })
    }
}

impl GgufMetadataValue {
    fn read(reader: &mut impl Read, value_type: GgufValueType) -> Result<Self, GgufError> {
        Ok(match value_type {
            GgufValueType::U8 => Self::U8(read_u8(reader)?),
            GgufValueType::I8 => Self::I8(read_i8(reader)?),
            GgufValueType::U16 => Self::U16(read_u16_le(reader)?),
            GgufValueType::I16 => Self::I16(read_i16_le(reader)?),
            GgufValueType::U32 => Self::U32(read_u32_le(reader)?),
            GgufValueType::I32 => Self::I32(read_i32_le(reader)?),
            GgufValueType::F32 => Self::F32(read_f32_le(reader)?),
            GgufValueType::Bool => Self::Bool(read_u8(reader)? != 0),
            GgufValueType::String => Self::String(read_gguf_string(reader, "metadata value")?),
            GgufValueType::Array => {
                let element_type_raw = read_u32_le(reader)?;
                let element_type = GgufValueType::from_u32(element_type_raw)
                    .ok_or(GgufError::InvalidArrayElementType(element_type_raw))?;
                if element_type == GgufValueType::Array {
                    return Err(GgufError::InvalidArrayElementType(element_type_raw));
                }
                let len = read_u64_le(reader)?;
                let mut string_samples = Vec::new();
                for _ in 0..len {
                    if element_type == GgufValueType::String {
                        let value = read_gguf_string(reader, "array string value")?;
                        if string_samples.len() < 8 {
                            string_samples.push(value);
                        }
                    } else {
                        skip_value(reader, element_type)?;
                    }
                }
                Self::Array {
                    element_type,
                    len,
                    string_samples,
                }
            }
            GgufValueType::U64 => Self::U64(read_u64_le(reader)?),
            GgufValueType::I64 => Self::I64(read_i64_le(reader)?),
            GgufValueType::F64 => Self::F64(read_f64_le(reader)?),
        })
    }

    pub fn summary(&self) -> String {
        match self {
            Self::U8(value) => value.to_string(),
            Self::I8(value) => value.to_string(),
            Self::U16(value) => value.to_string(),
            Self::I16(value) => value.to_string(),
            Self::U32(value) => value.to_string(),
            Self::I32(value) => value.to_string(),
            Self::F32(value) => value.to_string(),
            Self::Bool(value) => value.to_string(),
            Self::String(value) => value.clone(),
            Self::Array {
                element_type,
                len,
                string_samples,
            } => {
                if string_samples.is_empty() {
                    format!("array<{element_type:?}>[{len}]")
                } else {
                    format!(
                        "array<{element_type:?}>[{len}] sample_count={}",
                        string_samples.len()
                    )
                }
            }
            Self::U64(value) => value.to_string(),
            Self::I64(value) => value.to_string(),
            Self::F64(value) => value.to_string(),
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(*value as u64),
            Self::U16(value) => Some(*value as u64),
            Self::U32(value) => Some(*value as u64),
            Self::U64(value) => Some(*value),
            _ => None,
        }
    }
}

impl GgufValueType {
    fn read(reader: &mut impl Read) -> Result<Self, GgufError> {
        let raw = read_u32_le(reader)?;
        Self::from_u32(raw).ok_or(GgufError::UnsupportedValueType(raw))
    }

    fn from_u32(raw: u32) -> Option<Self> {
        match raw {
            0 => Some(Self::U8),
            1 => Some(Self::I8),
            2 => Some(Self::U16),
            3 => Some(Self::I16),
            4 => Some(Self::U32),
            5 => Some(Self::I32),
            6 => Some(Self::F32),
            7 => Some(Self::Bool),
            8 => Some(Self::String),
            9 => Some(Self::Array),
            10 => Some(Self::U64),
            11 => Some(Self::I64),
            12 => Some(Self::F64),
            _ => None,
        }
    }
}

impl GgufTensorInfo {
    fn read(reader: &mut impl Read) -> Result<Self, GgufError> {
        let name = read_gguf_string(reader, "tensor name")?;
        let n_dimensions = read_u32_le(reader)?;
        let mut dimensions = Vec::with_capacity(n_dimensions as usize);
        for _ in 0..n_dimensions {
            dimensions.push(read_u64_le(reader)?);
        }
        let tensor_type = read_u32_le(reader)?;
        let offset = read_u64_le(reader)?;
        Ok(Self {
            name,
            dimensions,
            tensor_type,
            offset,
            absolute_offset: 0,
            nbytes: None,
        })
    }
}

fn align_to(value: u64, alignment: u64) -> u64 {
    if alignment == 0 {
        return value;
    }
    value.div_ceil(alignment) * alignment
}

fn tensor_nbytes(tensor_type: u32, dimensions: &[u64]) -> Option<u64> {
    let elements = dimensions
        .iter()
        .try_fold(1u64, |acc, dim| acc.checked_mul(*dim))?;
    let (block_size, type_size) = ggml_type_layout(tensor_type)?;
    Some(elements.div_ceil(block_size) * type_size)
}

fn ggml_type_layout(tensor_type: u32) -> Option<(u64, u64)> {
    match tensor_type {
        0 => Some((1, 4)),      // F32
        1 => Some((1, 2)),      // F16
        2 => Some((32, 18)),    // Q4_0
        3 => Some((32, 20)),    // Q4_1
        6 => Some((32, 22)),    // Q5_0
        7 => Some((32, 24)),    // Q5_1
        8 => Some((32, 34)),    // Q8_0
        9 => Some((32, 40)),    // Q8_1
        10 => Some((256, 84)),  // Q2_K
        11 => Some((256, 110)), // Q3_K
        12 => Some((256, 144)), // Q4_K
        13 => Some((256, 176)), // Q5_K
        14 => Some((256, 210)), // Q6_K
        15 => Some((256, 292)), // Q8_K
        16 => Some((1, 8)),     // I8
        17 => Some((1, 2)),     // I16
        18 => Some((1, 4)),     // I32
        _ => None,
    }
}

fn skip_value(reader: &mut impl Read, value_type: GgufValueType) -> Result<(), GgufError> {
    match value_type {
        GgufValueType::U8 | GgufValueType::I8 | GgufValueType::Bool => {
            let _ = read_u8(reader)?;
        }
        GgufValueType::U16 | GgufValueType::I16 => {
            let _ = read_u16_le(reader)?;
        }
        GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => {
            let _ = read_u32_le(reader)?;
        }
        GgufValueType::String => {
            let _ = read_gguf_string(reader, "array string value")?;
        }
        GgufValueType::U64 | GgufValueType::I64 | GgufValueType::F64 => {
            let _ = read_u64_le(reader)?;
        }
        GgufValueType::Array => return Err(GgufError::InvalidArrayElementType(9)),
    }
    Ok(())
}

fn read_gguf_string(reader: &mut impl Read, context: &str) -> Result<String, GgufError> {
    let len = read_u64_le(reader)?;
    let len: usize = len
        .try_into()
        .map_err(|_| GgufError::InvalidUtf8(context.to_string()))?;
    let mut bytes = vec![0u8; len];
    reader.read_exact(&mut bytes)?;
    String::from_utf8(bytes).map_err(|_| GgufError::InvalidUtf8(context.to_string()))
}

fn read_u8(reader: &mut impl Read) -> Result<u8, io::Error> {
    let mut bytes = [0u8; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_i8(reader: &mut impl Read) -> Result<i8, io::Error> {
    Ok(read_u8(reader)? as i8)
}

fn read_u16_le(reader: &mut impl Read) -> Result<u16, io::Error> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_i16_le(reader: &mut impl Read) -> Result<i16, io::Error> {
    let mut bytes = [0u8; 2];
    reader.read_exact(&mut bytes)?;
    Ok(i16::from_le_bytes(bytes))
}

fn read_u32_le(reader: &mut impl Read) -> Result<u32, io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32_le(reader: &mut impl Read) -> Result<i32, io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_f32_le(reader: &mut impl Read) -> Result<f32, io::Error> {
    let mut bytes = [0u8; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_u64_le(reader: &mut impl Read) -> Result<u64, io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64_le(reader: &mut impl Read) -> Result<i64, io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f64_le(reader: &mut impl Read) -> Result<f64, io::Error> {
    let mut bytes = [0u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn write_gguf_string(file: &mut File, value: &str) {
        file.write_all(&(value.len() as u64).to_le_bytes())
            .expect("write string len");
        file.write_all(value.as_bytes()).expect("write string");
    }

    #[test]
    fn reads_minimal_gguf_directory() {
        let path = std::env::temp_dir().join(format!(
            "aeronum-gguf-header-{}-{}.gguf",
            std::process::id(),
            "valid"
        ));
        let mut file = File::create(&path).expect("create GGUF test file");
        file.write_all(b"GGUF").expect("write magic");
        file.write_all(&3u32.to_le_bytes()).expect("write version");
        file.write_all(&1u64.to_le_bytes())
            .expect("write tensor count");
        file.write_all(&3u64.to_le_bytes())
            .expect("write metadata count");

        write_gguf_string(&mut file, "general.architecture");
        file.write_all(&8u32.to_le_bytes())
            .expect("write string type");
        write_gguf_string(&mut file, "llama");

        write_gguf_string(&mut file, "general.quantization_version");
        file.write_all(&4u32.to_le_bytes()).expect("write u32 type");
        file.write_all(&2u32.to_le_bytes())
            .expect("write u32 value");

        write_gguf_string(&mut file, "tokenizer.ggml.tokens");
        file.write_all(&9u32.to_le_bytes())
            .expect("write array type");
        file.write_all(&8u32.to_le_bytes())
            .expect("write array string element type");
        file.write_all(&2u64.to_le_bytes())
            .expect("write array len");
        write_gguf_string(&mut file, "<s>");
        write_gguf_string(&mut file, "</s>");

        write_gguf_string(&mut file, "token_embd.weight");
        file.write_all(&2u32.to_le_bytes())
            .expect("write tensor dims");
        file.write_all(&32000u64.to_le_bytes())
            .expect("write tensor dim 0");
        file.write_all(&4096u64.to_le_bytes())
            .expect("write tensor dim 1");
        file.write_all(&15u32.to_le_bytes())
            .expect("write tensor type");
        file.write_all(&0u64.to_le_bytes())
            .expect("write tensor offset");
        drop(file);

        let header = GgufHeader::read(path.to_str().expect("utf8 temp path")).expect("read header");
        assert_eq!(header.version, 3);
        assert_eq!(header.tensor_count, 1);
        assert_eq!(header.metadata_kv_count, 3);
        assert_eq!(header.metadata.len(), 3);
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(
            header.metadata_value("general.architecture"),
            Some(&GgufMetadataValue::String("llama".to_string()))
        );
        assert_eq!(
            header.metadata_value("general.quantization_version"),
            Some(&GgufMetadataValue::U32(2))
        );
        assert_eq!(
            header.metadata_value("tokenizer.ggml.tokens"),
            Some(&GgufMetadataValue::Array {
                element_type: GgufValueType::String,
                len: 2,
                string_samples: vec!["<s>".to_string(), "</s>".to_string()]
            })
        );
        assert_eq!(header.tensors[0].name, "token_embd.weight");
        assert_eq!(header.tensors[0].dimensions, vec![32000, 4096]);
        assert_eq!(header.tensors[0].tensor_type, 15);

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
