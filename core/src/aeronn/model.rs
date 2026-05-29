use crate::gpu::{Backend, Device, GpuDevice, HipBuffer, HipRuntime};
use crate::NdArray;
use std::collections::HashMap;
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
        string_values: Vec<String>,
        i32_samples: Vec<i32>,
        i32_values: Vec<i32>,
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GgufTokenizerIndex {
    pub token_count: usize,
    pub token_to_id: HashMap<String, u32>,
    pub id_to_token: Vec<String>,
    pub merge_ranks: HashMap<(String, String), usize>,
    pub unknown_token_id: Option<u32>,
    pub bos_token_id: Option<u32>,
}

impl GgufTokenizerIndex {
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    pub fn encode_exact_pieces<'a>(
        &self,
        pieces: impl IntoIterator<Item = &'a str>,
    ) -> Option<Vec<u32>> {
        pieces
            .into_iter()
            .map(|piece| self.token_id(piece))
            .collect()
    }

    pub fn decode_ids(&self, ids: &[u32]) -> Option<Vec<String>> {
        ids.iter()
            .map(|id| {
                usize::try_from(*id)
                    .ok()
                    .and_then(|idx| self.id_to_token.get(idx))
                    .cloned()
            })
            .collect()
    }

    pub fn encode_byte_bpe(&self, text: &str, add_bos: bool) -> Option<Vec<u32>> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_token_id?);
        }

        for piece in byte_level_pieces(text) {
            for token in self.byte_bpe_piece(&piece) {
                ids.push(self.token_id(&token).or(self.unknown_token_id)?);
            }
        }

        Some(ids)
    }

    fn byte_bpe_piece(&self, piece: &str) -> Vec<String> {
        let mut parts = piece.chars().map(|ch| ch.to_string()).collect::<Vec<_>>();
        if parts.len() < 2 || self.merge_ranks.is_empty() {
            return parts;
        }

        loop {
            let Some((merge_index, _)) = parts
                .windows(2)
                .enumerate()
                .filter_map(|(idx, pair)| {
                    self.merge_ranks
                        .get(&(pair[0].clone(), pair[1].clone()))
                        .map(|rank| (idx, *rank))
                })
                .min_by_key(|(_, rank)| *rank)
            else {
                break;
            };

            let merged = format!("{}{}", parts[merge_index], parts[merge_index + 1]);
            parts.splice(merge_index..=merge_index + 1, [merged]);
            if parts.len() < 2 {
                break;
            }
        }

        parts
    }
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
    UnsupportedTensorType { name: String, tensor_type: u32 },
    TensorShapeTooLarge(String),
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
            Self::UnsupportedTensorType { name, tensor_type } => {
                write!(
                    f,
                    "unsupported GGUF tensor type {tensor_type} for tensor: {name}"
                )
            }
            Self::TensorShapeTooLarge(name) => write!(f, "GGUF tensor shape too large: {name}"),
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
            | Self::InvalidTensorRange(_)
            | Self::UnsupportedTensorType { .. }
            | Self::TensorShapeTooLarge(_) => None,
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
    pub weight_names: Vec<String>,
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
            weight_names: Vec::new(),
            device: GpuDevice::auto_detect(),
            hip_runtime: None,
            hip_weights: Vec::new(),
            gguf_header: Some(header),
        })
    }

    pub fn load_f32_weight(&mut self, tensor_name: &str) -> Result<usize, GgufError> {
        let header = self
            .gguf_header
            .as_ref()
            .ok_or_else(|| GgufError::TensorNotFound(tensor_name.to_string()))?;
        let tensor = header.load_f32_tensor(tensor_name)?;
        self.weights.push(tensor);
        self.weight_names.push(tensor_name.to_string());
        Ok(self.weights.len() - 1)
    }

    pub fn load_all_f32_weights(&mut self) -> Result<usize, GgufError> {
        let tensor_names = self
            .gguf_header
            .as_ref()
            .ok_or_else(|| GgufError::TensorNotFound("*.f32".to_string()))?
            .f32_tensor_names();
        for tensor_name in &tensor_names {
            let tensor = self
                .gguf_header
                .as_ref()
                .ok_or_else(|| GgufError::TensorNotFound(tensor_name.clone()))?
                .load_f32_tensor(tensor_name)?;
            self.weights.push(tensor);
            self.weight_names.push(tensor_name.clone());
        }
        Ok(tensor_names.len())
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

    pub fn load_f32_tensor(&self, tensor_name: &str) -> Result<NdArray, GgufError> {
        let tensor = self
            .tensors
            .iter()
            .find(|tensor| tensor.name == tensor_name)
            .ok_or_else(|| GgufError::TensorNotFound(tensor_name.to_string()))?;
        if tensor.tensor_type != 0 {
            return Err(GgufError::UnsupportedTensorType {
                name: tensor_name.to_string(),
                tensor_type: tensor.tensor_type,
            });
        }
        let tensor_nbytes = tensor
            .nbytes
            .ok_or_else(|| GgufError::UnknownTensorByteSize(tensor_name.to_string()))?;
        let tensor_end = tensor
            .absolute_offset
            .checked_add(tensor_nbytes)
            .ok_or_else(|| GgufError::InvalidTensorRange(tensor_name.to_string()))?;
        if tensor_end > self.file_size || tensor_nbytes % 4 != 0 {
            return Err(GgufError::InvalidTensorRange(tensor_name.to_string()));
        }

        let shape = tensor
            .dimensions
            .iter()
            .map(|dim| usize::try_from(*dim))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| GgufError::TensorShapeTooLarge(tensor_name.to_string()))?;
        let element_count = shape
            .iter()
            .try_fold(1usize, |acc, dim| acc.checked_mul(*dim))
            .ok_or_else(|| GgufError::TensorShapeTooLarge(tensor_name.to_string()))?;
        if element_count * 4 != tensor_nbytes as usize {
            return Err(GgufError::InvalidTensorRange(tensor_name.to_string()));
        }

        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(tensor.absolute_offset))?;
        let mut bytes = vec![0u8; tensor_nbytes as usize];
        file.read_exact(&mut bytes)?;
        let values = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect::<Vec<_>>();
        Ok(NdArray::from_list(values, Some(&shape)))
    }

    pub fn f32_tensor_names(&self) -> Vec<String> {
        self.tensors
            .iter()
            .filter(|tensor| tensor.tensor_type == 0)
            .map(|tensor| tensor.name.clone())
            .collect()
    }

    pub fn string_array_values(&self, key: &str) -> Option<&[String]> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::Array {
                element_type: GgufValueType::String,
                string_values,
                ..
            }) => Some(string_values),
            _ => None,
        }
    }

    pub fn i32_array_values(&self, key: &str) -> Option<&[i32]> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::Array {
                element_type: GgufValueType::I32,
                i32_values,
                ..
            }) => Some(i32_values),
            _ => None,
        }
    }

    pub fn u32_value(&self, key: &str) -> Option<u32> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::U8(value)) => Some(*value as u32),
            Some(GgufMetadataValue::U16(value)) => Some(*value as u32),
            Some(GgufMetadataValue::U32(value)) => Some(*value),
            Some(GgufMetadataValue::U64(value)) => u32::try_from(*value).ok(),
            _ => None,
        }
    }

    pub fn f32_value(&self, key: &str) -> Option<f32> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::F32(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn bool_value(&self, key: &str) -> Option<bool> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    pub fn string_value(&self, key: &str) -> Option<&str> {
        match self.metadata_value(key) {
            Some(GgufMetadataValue::String(value)) => Some(value),
            _ => None,
        }
    }

    pub fn tokenizer_index(&self) -> Option<GgufTokenizerIndex> {
        let tokens = self.string_array_values("tokenizer.ggml.tokens")?;
        let token_to_id = tokens
            .iter()
            .enumerate()
            .filter_map(|(idx, token)| u32::try_from(idx).ok().map(|id| (token.clone(), id)))
            .collect::<HashMap<_, _>>();
        let merge_ranks = self
            .string_array_values("tokenizer.ggml.merges")
            .map(|merges| {
                merges
                    .iter()
                    .enumerate()
                    .filter_map(|(rank, merge)| {
                        let (left, right) = merge.split_once(' ')?;
                        Some(((left.to_string(), right.to_string()), rank))
                    })
                    .collect::<HashMap<_, _>>()
            })
            .unwrap_or_default();
        Some(GgufTokenizerIndex {
            token_count: tokens.len(),
            token_to_id,
            id_to_token: tokens.to_vec(),
            merge_ranks,
            unknown_token_id: self.u32_value("tokenizer.ggml.unknown_token_id"),
            bos_token_id: self.u32_value("tokenizer.ggml.bos_token_id"),
        })
    }
}

fn byte_level_pieces(text: &str) -> Vec<String> {
    pretokenize(text)
        .into_iter()
        .map(|piece| {
            piece
                .as_bytes()
                .iter()
                .map(|byte| byte_level_char(*byte))
                .collect()
        })
        .collect()
}

fn pretokenize(text: &str) -> Vec<&str> {
    let mut pieces = Vec::new();
    let mut idx = 0;
    while idx < text.len() {
        if let Some(end) = contraction_end(text, idx) {
            pieces.push(&text[idx..end]);
            idx = end;
            continue;
        }

        let ch = text[idx..].chars().next().expect("valid char boundary");
        let mut start = idx;
        let mut scan = idx;
        if ch == ' ' {
            let next_idx = idx + ch.len_utf8();
            if next_idx < text.len() {
                let next = text[next_idx..]
                    .chars()
                    .next()
                    .expect("valid char boundary");
                if !next.is_whitespace() {
                    scan = next_idx;
                }
            }
        }

        let first = text[scan..].chars().next().expect("valid char boundary");
        let class = char_class(first);
        if scan != idx && class == CharClass::Whitespace {
            start = idx;
            scan = idx;
        }
        let mut end = scan + first.len_utf8();
        while end < text.len() {
            if contraction_end(text, end).is_some() {
                break;
            }
            let next = text[end..].chars().next().expect("valid char boundary");
            if char_class(next) != class {
                break;
            }
            end += next.len_utf8();
        }

        pieces.push(&text[start..end]);
        idx = end;
    }
    pieces
}

fn contraction_end(text: &str, idx: usize) -> Option<usize> {
    ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
        .iter()
        .find_map(|suffix| {
            let end = idx + suffix.len();
            text.get(idx..end)
                .is_some_and(|value| value.eq_ignore_ascii_case(suffix))
                .then_some(end)
        })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CharClass {
    Letter,
    Number,
    Whitespace,
    Other,
}

fn char_class(ch: char) -> CharClass {
    if ch.is_alphabetic() {
        CharClass::Letter
    } else if ch.is_numeric() {
        CharClass::Number
    } else if ch.is_whitespace() {
        CharClass::Whitespace
    } else {
        CharClass::Other
    }
}

fn byte_level_char(byte: u8) -> char {
    match byte {
        33..=126 | 161..=172 | 174..=255 => char::from(byte),
        _ => {
            let mut offset = 0u32;
            for value in 0u16..=255 {
                let value = value as u8;
                if matches!(value, 33..=126 | 161..=172 | 174..=255) {
                    continue;
                }
                if value == byte {
                    return char::from_u32(256 + offset).expect("valid byte-level unicode scalar");
                }
                offset += 1;
            }
            unreachable!("all byte values are covered")
        }
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
                let mut string_values = Vec::new();
                let mut i32_samples = Vec::new();
                let mut i32_values = Vec::new();
                for _ in 0..len {
                    if element_type == GgufValueType::String {
                        let value = read_gguf_string(reader, "array string value")?;
                        if string_samples.len() < 8 {
                            string_samples.push(value.clone());
                        }
                        string_values.push(value);
                    } else if element_type == GgufValueType::I32 {
                        let value = read_i32_le(reader)?;
                        if i32_samples.len() < 8 {
                            i32_samples.push(value);
                        }
                        i32_values.push(value);
                    } else {
                        skip_value(reader, element_type)?;
                    }
                }
                Self::Array {
                    element_type,
                    len,
                    string_samples,
                    string_values,
                    i32_samples,
                    i32_values,
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
                i32_samples,
                ..
            } => {
                if !string_samples.is_empty() {
                    format!(
                        "array<{element_type:?}>[{len}] sample_count={}",
                        string_samples.len()
                    )
                } else if !i32_samples.is_empty() {
                    format!(
                        "array<{element_type:?}>[{len}] sample_count={}",
                        i32_samples.len()
                    )
                } else {
                    format!("array<{element_type:?}>[{len}]")
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
    use std::io::{Seek, Write};

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
        file.write_all(&5u64.to_le_bytes())
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
        file.write_all(&13u64.to_le_bytes())
            .expect("write array len");
        write_gguf_string(&mut file, "<s>");
        write_gguf_string(&mut file, "</s>");
        write_gguf_string(&mut file, "<unk>");
        write_gguf_string(&mut file, "H");
        write_gguf_string(&mut file, "i");
        write_gguf_string(&mut file, "Hi");
        write_gguf_string(&mut file, "\u{0120}");
        write_gguf_string(&mut file, ".");
        write_gguf_string(&mut file, "c");
        write_gguf_string(&mut file, "p");
        write_gguf_string(&mut file, "cp");
        write_gguf_string(&mut file, "cpp");
        write_gguf_string(&mut file, ".cpp");

        write_gguf_string(&mut file, "tokenizer.ggml.merges");
        file.write_all(&9u32.to_le_bytes())
            .expect("write array type");
        file.write_all(&8u32.to_le_bytes())
            .expect("write array string element type");
        file.write_all(&4u64.to_le_bytes())
            .expect("write array len");
        write_gguf_string(&mut file, "H i");
        write_gguf_string(&mut file, "c p");
        write_gguf_string(&mut file, "cp p");
        write_gguf_string(&mut file, ". cpp");

        write_gguf_string(&mut file, "tokenizer.ggml.token_type");
        file.write_all(&9u32.to_le_bytes())
            .expect("write array type");
        file.write_all(&5u32.to_le_bytes())
            .expect("write array i32 element type");
        file.write_all(&13u64.to_le_bytes())
            .expect("write array len");
        for token_type in [3i32, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1] {
            file.write_all(&token_type.to_le_bytes())
                .expect("write token type");
        }

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
        assert_eq!(header.metadata_kv_count, 5);
        assert_eq!(header.metadata.len(), 5);
        assert_eq!(header.tensors.len(), 1);
        assert_eq!(
            header.metadata_value("general.architecture"),
            Some(&GgufMetadataValue::String("llama".to_string()))
        );
        assert_eq!(
            header.metadata_value("general.quantization_version"),
            Some(&GgufMetadataValue::U32(2))
        );
        assert_eq!(header.u32_value("general.quantization_version"), Some(2));
        assert_eq!(
            header.metadata_value("tokenizer.ggml.tokens"),
            Some(&GgufMetadataValue::Array {
                element_type: GgufValueType::String,
                len: 13,
                string_samples: vec![
                    "<s>".to_string(),
                    "</s>".to_string(),
                    "<unk>".to_string(),
                    "H".to_string(),
                    "i".to_string(),
                    "Hi".to_string(),
                    "\u{0120}".to_string(),
                    ".".to_string()
                ],
                string_values: vec![
                    "<s>".to_string(),
                    "</s>".to_string(),
                    "<unk>".to_string(),
                    "H".to_string(),
                    "i".to_string(),
                    "Hi".to_string(),
                    "\u{0120}".to_string(),
                    ".".to_string(),
                    "c".to_string(),
                    "p".to_string(),
                    "cp".to_string(),
                    "cpp".to_string(),
                    ".cpp".to_string()
                ],
                i32_samples: Vec::new(),
                i32_values: Vec::new()
            })
        );
        assert_eq!(
            header.i32_array_values("tokenizer.ggml.token_type"),
            Some(&[3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1][..])
        );
        let tokenizer_index = header.tokenizer_index().expect("tokenizer index");
        assert_eq!(tokenizer_index.token_count, 13);
        assert_eq!(tokenizer_index.token_to_id.get("<s>"), Some(&0));
        assert_eq!(tokenizer_index.token_to_id.get("</s>"), Some(&1));
        assert_eq!(
            tokenizer_index
                .merge_ranks
                .get(&("H".to_string(), "i".to_string())),
            Some(&0)
        );
        assert_eq!(
            tokenizer_index.encode_exact_pieces(["<s>", "</s>"]),
            Some(vec![0, 1])
        );
        assert_eq!(
            tokenizer_index.decode_ids(&[0, 1]),
            Some(vec!["<s>".to_string(), "</s>".to_string()])
        );
        assert_eq!(tokenizer_index.encode_byte_bpe("Hi", false), Some(vec![5]));
        assert_eq!(
            tokenizer_index.encode_byte_bpe(" Hi", false),
            Some(vec![6, 5])
        );
        assert_eq!(
            tokenizer_index.encode_byte_bpe(".cpp", false),
            Some(vec![7, 11])
        );
        assert_eq!(header.tensors[0].name, "token_embd.weight");
        assert_eq!(header.tensors[0].dimensions, vec![32000, 4096]);
        assert_eq!(header.tensors[0].tensor_type, 15);

        fs::remove_file(path).expect("remove GGUF test file");
    }

    #[test]
    fn llama_model_loads_f32_weight_from_gguf() {
        let path = std::env::temp_dir().join(format!(
            "aeronum-gguf-header-{}-{}.gguf",
            std::process::id(),
            "f32-weight"
        ));
        let mut file = File::create(&path).expect("create GGUF test file");
        file.write_all(b"GGUF").expect("write magic");
        file.write_all(&3u32.to_le_bytes()).expect("write version");
        file.write_all(&2u64.to_le_bytes())
            .expect("write tensor count");
        file.write_all(&0u64.to_le_bytes())
            .expect("write metadata count");

        write_gguf_string(&mut file, "output_norm.weight");
        file.write_all(&1u32.to_le_bytes())
            .expect("write tensor dims");
        file.write_all(&2u64.to_le_bytes())
            .expect("write tensor dim 0");
        file.write_all(&0u32.to_le_bytes())
            .expect("write F32 tensor type");
        file.write_all(&0u64.to_le_bytes())
            .expect("write tensor offset");

        write_gguf_string(&mut file, "blk.0.attn_norm.weight");
        file.write_all(&1u32.to_le_bytes())
            .expect("write tensor dims");
        file.write_all(&1u64.to_le_bytes())
            .expect("write tensor dim 0");
        file.write_all(&0u32.to_le_bytes())
            .expect("write F32 tensor type");
        file.write_all(&8u64.to_le_bytes())
            .expect("write tensor offset");

        let directory_end = file.stream_position().expect("directory end");
        let padding = align_to(directory_end, 32) - directory_end;
        file.write_all(&vec![0u8; padding as usize])
            .expect("write data padding");
        file.write_all(&1.25f32.to_le_bytes())
            .expect("write tensor value 0");
        file.write_all(&2.5f32.to_le_bytes())
            .expect("write tensor value 1");
        file.write_all(&3.75f32.to_le_bytes())
            .expect("write tensor value 2");
        drop(file);

        let mut model =
            LlamaModel::try_load_gguf(path.to_str().expect("utf8 temp path")).expect("load model");
        let header = model.gguf_header.as_ref().expect("header present");
        assert_eq!(
            header.f32_tensor_names(),
            vec![
                "output_norm.weight".to_string(),
                "blk.0.attn_norm.weight".to_string()
            ]
        );
        let weight_index = model
            .load_f32_weight("output_norm.weight")
            .expect("load f32 weight");
        assert_eq!(weight_index, 0);
        assert_eq!(model.weight_names, vec!["output_norm.weight".to_string()]);
        assert_eq!(model.weights[0].shape(), &[2]);
        assert_eq!(model.weights[0].to_vec(), vec![1.25, 2.5]);

        let mut model =
            LlamaModel::try_load_gguf(path.to_str().expect("utf8 temp path")).expect("load model");
        let weight_count = model.load_all_f32_weights().expect("load all f32 weights");
        assert_eq!(weight_count, 2);
        assert_eq!(
            model.weight_names,
            vec![
                "output_norm.weight".to_string(),
                "blk.0.attn_norm.weight".to_string()
            ]
        );
        assert_eq!(model.weights[0].to_vec(), vec![1.25, 2.5]);
        assert_eq!(model.weights[1].to_vec(), vec![3.75]);

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
