//! GGUF Model Format Loader for RuvLLM
//!
//! This module provides support for loading llama.cpp compatible GGUF model files.
//! GGUF (GGML Universal File) is a binary format that stores model weights along
//! with metadata and tokenizer information.
//!
//! ## Features
//!
//! - **Parser**: Complete GGUF v3 format parsing with memory-mapped file support
//! - **Quantization**: All llama.cpp quantization types (Q4_0, Q4_K, Q8_0, etc.)
//! - **Streaming**: Chunk-based tensor loading for large models
//! - **Metadata**: Automatic extraction of model architecture parameters
//!
//! ## Supported Quantization Types
//!
//! | Type | Bits | Block Size | Memory (7B) | Quality |
//! |------|------|------------|-------------|---------|
//! | F32 | 32 | 1 | 28 GB | Best |
//! | F16 | 16 | 1 | 14 GB | Excellent |
//! | Q8_0 | 8.5 | 32 | 7.5 GB | Very Good |
//! | Q4_K | 4.5 | 256 | 4 GB | Good |
//! | Q4_0 | 4.5 | 32 | 4 GB | Acceptable |
//! | Q2_K | 2.6 | 256 | 2.3 GB | Experimental |
//!
//! ## Example
//!
//! ```rust,ignore
//! use ruvllm::gguf::{GgufFile, GgufModelLoader};
//! use std::path::Path;
//!
//! // Load a GGUF file
//! let file = GgufFile::open(Path::new("model.gguf"))?;
//!
//! // Read model metadata
//! println!("Architecture: {:?}", file.architecture());
//! println!("Context length: {:?}", file.context_length());
//! println!("Layers: {:?}", file.layer_count());
//!
//! // Load a specific tensor
//! let weights = file.load_tensor_f32("model.layers.0.attention.wq.weight")?;
//!
//! // Or use memory-mapped loading for efficiency
//! let mmap_file = GgufFile::open_mmap(Path::new("model.gguf"))?;
//! let tensor_info = mmap_file.get_tensor("model.embed_tokens.weight").unwrap();
//! let data = mmap_file.tensor_data(&tensor_info);
//! ```
//!
//! ## Backend Integration
//!
//! The GGUF loader integrates seamlessly with RuvLLM backends:
//!
//! ```rust,ignore
//! use ruvllm::backends::LlmBackend;
//! use std::path::Path;
//!
//! // Load from GGUF file directly
//! let backend = LlmBackend::from_gguf(
//!     Path::new("model-Q4_K_M.gguf"),
//!     BackendConfig::default()
//! )?;
//! ```

pub mod loader;
pub mod model_init;
pub mod parser;
pub mod quantization;
pub mod tensors;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

#[cfg(unix)]
use std::os::unix::fs::FileExt;

use crate::backends::ModelArchitecture;
use crate::error::{Result, RuvLLMError};

pub use loader::{
    GgufLoader, LoadConfig, LoadProgress, LoadedTensor, LoadedWeights, ProgressCallback,
    StreamingLoader, TensorCategory, TensorNameMapper,
};
pub use model_init::{
    LayerWeights, ModelInitializer, ModelWeights, ProgressModelBuilder, QuantizedWeight,
    WeightTensor,
};
pub use parser::{parse_header, parse_metadata, parse_tensor_infos, GgufHeader, GgufValue};
pub use quantization::{dequantize_block, GgufQuantType, QuantizedTensor};
pub use tensors::TensorInfo;

// ============================================================================
// GGUF File Magic and Constants
// ============================================================================

/// GGUF magic number (little-endian: "GGUF")
pub const GGUF_MAGIC: u32 = 0x46554747;

/// Current GGUF version supported
pub const GGUF_VERSION: u32 = 3;

/// Default alignment for tensor data
pub const DEFAULT_ALIGNMENT: usize = 32;

// ============================================================================
// GgufFile - Main Interface
// ============================================================================

/// GGUF file reader with optional memory-mapping support.
///
/// This struct provides the main interface for reading GGUF model files.
/// It supports both traditional file I/O and memory-mapped access for
/// improved performance with large models.
///
/// # Memory Mapping
///
/// Memory mapping is recommended for large models as it allows the OS
/// to manage memory efficiently and enables lazy loading of tensor data.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::gguf::GgufFile;
/// use std::path::Path;
///
/// // Standard file access
/// let file = GgufFile::open(Path::new("model.gguf"))?;
///
/// // Memory-mapped access (recommended for large models)
/// let mmap_file = GgufFile::open_mmap(Path::new("model.gguf"))?;
/// ```
pub struct GgufFile {
    /// GGUF header information
    pub header: GgufHeader,
    /// Key-value metadata
    pub metadata: HashMap<String, GgufValue>,
    /// Tensor information array
    pub tensors: Vec<TensorInfo>,
    /// File path
    path: std::path::PathBuf,
    /// Optional memory-mapped data
    mmap: Option<MmapData>,
    /// Data section offset in file
    data_offset: u64,
    /// Alignment for tensor data
    alignment: usize,
}

/// Memory-mapped file data
struct MmapData {
    /// Memory-mapped region
    #[cfg(feature = "mmap")]
    mmap: memmap2::Mmap,
    #[cfg(not(feature = "mmap"))]
    data: Vec<u8>,
}

impl GgufFile {
    /// Open a GGUF file for reading.
    ///
    /// This method reads the file header, metadata, and tensor information
    /// but does not load tensor data into memory.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Returns
    ///
    /// A `GgufFile` instance ready for tensor loading
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be opened
    /// - The file is not a valid GGUF file
    /// - The GGUF version is not supported
    pub fn open(path: &Path) -> Result<Self> {
        let file = File::open(path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to open GGUF file: {}", e)))?;
        let mut reader = BufReader::new(file);

        // Parse header
        let header = parse_header(&mut reader)?;

        // Validate magic and version
        if header.magic != GGUF_MAGIC {
            return Err(RuvLLMError::Model(format!(
                "Invalid GGUF magic: expected 0x{:08X}, got 0x{:08X}",
                GGUF_MAGIC, header.magic
            )));
        }

        if header.version != GGUF_VERSION && header.version != 2 {
            return Err(RuvLLMError::Model(format!(
                "Unsupported GGUF version: {} (supported: 2, 3)",
                header.version
            )));
        }

        // Parse metadata
        let metadata = parse_metadata(&mut reader, header.metadata_kv_count)?;

        // Get alignment from metadata or use default
        let alignment = metadata
            .get("general.alignment")
            .and_then(|v| v.as_u64())
            .unwrap_or(DEFAULT_ALIGNMENT as u64);
        let alignment = usize::try_from(alignment)
            .map_err(|_| RuvLLMError::Model("GGUF alignment does not fit usize".to_owned()))?;

        // Parse tensor infos
        let tensors = parse_tensor_infos(&mut reader, header.tensor_count)?;

        // Calculate data offset (aligned)
        let current_pos = reader
            .stream_position()
            .map_err(|e| RuvLLMError::Model(format!("Failed to get stream position: {}", e)))?;
        let data_offset = align_offset(current_pos, alignment as u64)?;
        let source_len = reader
            .get_ref()
            .metadata()
            .map_err(|e| RuvLLMError::Model(format!("Failed to get GGUF file length: {e}")))?
            .len();

        let gguf = Self {
            header,
            metadata,
            tensors,
            path: path.to_path_buf(),
            mmap: None,
            data_offset,
            alignment,
        };
        gguf.validate_tensor_ranges(source_len)?;
        Ok(gguf)
    }

    /// Open a GGUF file with memory mapping.
    ///
    /// Memory mapping provides efficient access to tensor data for large
    /// models by letting the operating system manage memory paging.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    ///
    /// # Returns
    ///
    /// A `GgufFile` instance with memory-mapped tensor data access
    ///
    /// # Errors
    ///
    /// Returns an error if memory mapping fails or the file is invalid
    #[cfg(feature = "mmap")]
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let mut gguf = Self::open(path)?;

        let file = File::open(path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to open file for mmap: {}", e)))?;

        let mmap = unsafe {
            memmap2::Mmap::map(&file)
                .map_err(|e| RuvLLMError::Model(format!("Failed to memory map file: {}", e)))?
        };

        gguf.validate_tensor_ranges(mmap.len() as u64)?;
        gguf.mmap = Some(MmapData { mmap });
        Ok(gguf)
    }

    /// Open with memory mapping (fallback when mmap feature is disabled)
    #[cfg(not(feature = "mmap"))]
    pub fn open_mmap(path: &Path) -> Result<Self> {
        let mut gguf = Self::open(path)?;

        // Read entire file into memory as fallback
        let data = std::fs::read(path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to read file: {}", e)))?;

        gguf.validate_tensor_ranges(data.len() as u64)?;
        gguf.mmap = Some(MmapData { data });
        Ok(gguf)
    }

    /// Get tensor information by name.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name (e.g., "model.layers.0.attention.wq.weight")
    ///
    /// # Returns
    ///
    /// Reference to the tensor info if found
    pub fn get_tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Load tensor data as FP32 (dequantizing if necessary).
    ///
    /// This method reads the tensor from disk and converts it to FP32
    /// format, dequantizing quantized data as needed.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name
    ///
    /// # Returns
    ///
    /// Vector of FP32 values
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not found or cannot be read
    pub fn load_tensor_f32(&self, name: &str) -> Result<Vec<f32>> {
        let info = self
            .get_tensor(name)
            .ok_or_else(|| RuvLLMError::NotFound(format!("Tensor not found: {}", name)))?;

        let raw_data = self.read_tensor_bytes(info)?;
        let num_elements: usize = info.shape.iter().product();

        // Dequantize based on type
        let output = quantization::dequantize_tensor(&raw_data, info.dtype, num_elements)?;
        Ok(output)
    }

    /// Load tensor as a quantized tensor (preserving quantization).
    ///
    /// This method reads the tensor without dequantizing, preserving
    /// the original quantization format for efficient inference.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name
    ///
    /// # Returns
    ///
    /// A `QuantizedTensor` containing the raw quantized data
    pub fn load_tensor_quantized(&self, name: &str) -> Result<QuantizedTensor> {
        let info = self
            .get_tensor(name)
            .ok_or_else(|| RuvLLMError::NotFound(format!("Tensor not found: {}", name)))?;

        let data = self.read_tensor_bytes(info)?;
        let num_elements: usize = info.shape.iter().product();

        Ok(QuantizedTensor {
            data,
            dtype: info.dtype,
            shape: info.shape.clone(),
            num_elements,
        })
    }

    /// Get direct access to tensor data bytes (for memory-mapped files).
    ///
    /// This method returns a slice to the raw tensor data without copying.
    /// Only available when the file was opened with `open_mmap`.
    ///
    /// # Arguments
    ///
    /// * `info` - Tensor information
    ///
    /// # Returns
    ///
    /// Slice of raw bytes for the tensor
    ///
    /// # Panics
    ///
    /// Panics if the file was not opened with memory mapping
    pub fn tensor_data(&self, info: &TensorInfo) -> &[u8] {
        let mmap = self.mmap.as_ref().expect("File not memory-mapped");
        let start = (self.data_offset + info.offset) as usize;
        let end = start + info.byte_size();

        #[cfg(feature = "mmap")]
        {
            &mmap.mmap[start..end]
        }
        #[cfg(not(feature = "mmap"))]
        {
            &mmap.data[start..end]
        }
    }

    /// Stream tensor data in chunks for memory-efficient processing.
    ///
    /// This method processes the tensor in chunks, calling the provided
    /// callback for each chunk. Useful for very large tensors.
    ///
    /// # Arguments
    ///
    /// * `name` - The tensor name
    /// * `chunk_size` - Number of FP32 elements per chunk
    /// * `f` - Callback function receiving each chunk
    ///
    /// # Returns
    ///
    /// Ok(()) if all chunks were processed successfully
    pub fn stream_tensor<F>(&self, name: &str, chunk_size: usize, mut f: F) -> Result<()>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let info = self
            .get_tensor(name)
            .ok_or_else(|| RuvLLMError::NotFound(format!("Tensor not found: {}", name)))?;

        let _num_elements: usize = info.shape.iter().product();

        // For simple types (F32, F16), we can stream directly
        match info.dtype {
            GgufQuantType::F32 => {
                self.stream_f32_tensor(info, chunk_size, &mut f)?;
            }
            GgufQuantType::F16 => {
                self.stream_f16_tensor(info, chunk_size, &mut f)?;
            }
            _ => {
                // For quantized types, load and dequantize in block-aligned chunks
                let block_size = info.dtype.block_size();
                let aligned_chunk = ((chunk_size + block_size - 1) / block_size) * block_size;
                let full_data = self.load_tensor_f32(name)?;

                for chunk in full_data.chunks(aligned_chunk) {
                    f(chunk)?;
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // Metadata Extraction Methods
    // ========================================================================

    /// Get the model architecture (llama, mistral, phi, etc.).
    pub fn architecture(&self) -> Option<&str> {
        self.metadata
            .get("general.architecture")
            .and_then(|v| v.as_str())
    }

    /// Get the model architecture as enum.
    pub fn architecture_type(&self) -> Option<ModelArchitecture> {
        self.architecture()
            .and_then(|arch| match arch.to_lowercase().as_str() {
                "llama" => Some(ModelArchitecture::Llama),
                "mistral" => Some(ModelArchitecture::Mistral),
                "phi" | "phi2" | "phi3" => Some(ModelArchitecture::Phi),
                "qwen" | "qwen2" => Some(ModelArchitecture::Qwen),
                "gemma" => Some(ModelArchitecture::Gemma),
                _ => None,
            })
    }

    /// Get the context length (max sequence length).
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.context_length", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the embedding dimension (hidden size).
    pub fn embedding_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.embedding_length", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the number of attention heads.
    pub fn head_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.attention.head_count", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the number of key-value heads (for GQA/MQA).
    pub fn head_count_kv(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.attention.head_count_kv", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .or_else(|| self.head_count()) // Default to head_count if not specified
    }

    /// Get the number of layers.
    pub fn layer_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.block_count", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> Option<usize> {
        // Try tokenizer.ggml.model first
        self.metadata
            .get("tokenizer.ggml.tokens")
            .and_then(|v| v.as_array())
            .map(|arr| arr.len())
            .or_else(|| {
                let arch = self.architecture()?;
                self.metadata
                    .get(&format!("{}.vocab_size", arch))
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
            })
    }

    /// Get the RoPE frequency base.
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.rope.freq_base", arch))
            .and_then(|v| v.as_f32())
    }

    /// Get the RoPE dimension count.
    pub fn rope_dimension_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.rope.dimension_count", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the feed-forward hidden dimension.
    pub fn feed_forward_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata
            .get(&format!("{}.feed_forward_length", arch))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
    }

    /// Get the model name.
    pub fn model_name(&self) -> Option<&str> {
        self.metadata.get("general.name").and_then(|v| v.as_str())
    }

    /// Get the model author.
    pub fn author(&self) -> Option<&str> {
        self.metadata.get("general.author").and_then(|v| v.as_str())
    }

    /// Get the quantization type description.
    pub fn quantization_version(&self) -> Option<&str> {
        self.metadata
            .get("general.quantization_version")
            .and_then(|v| v.as_str())
    }

    /// Get all tensor names.
    pub fn tensor_names(&self) -> impl Iterator<Item = &str> {
        self.tensors.iter().map(|t| t.name.as_str())
    }

    /// Get the total size of all tensors in bytes.
    pub fn total_tensor_size(&self) -> usize {
        self.tensors.iter().map(|t| t.byte_size()).sum()
    }

    // ========================================================================
    // Private Helper Methods
    // ========================================================================

    fn tensor_span(&self, info: &TensorInfo, source_len: u64) -> Result<(u64, usize)> {
        let overflow =
            |kind| RuvLLMError::Model(format!("GGUF tensor {kind} overflow: {}", info.name));
        let elements = info.shape.iter().try_fold(1usize, |n, &dimension| {
            n.checked_mul(dimension)
                .ok_or_else(|| overflow("element count"))
        })?;
        let byte_size = elements
            .div_ceil(info.dtype.block_size())
            .checked_mul(info.dtype.type_size())
            .ok_or_else(|| overflow("byte size"))?;
        let length = u64::try_from(byte_size).map_err(|_| overflow("byte size"))?;
        let start = self
            .data_offset
            .checked_add(info.offset)
            .ok_or_else(|| overflow("offset"))?;
        let end = start
            .checked_add(length)
            .ok_or_else(|| overflow("offset"))?;
        if end > source_len {
            return Err(RuvLLMError::Model(format!(
                "GGUF tensor range outside file: {} ({start}..{end}, length {source_len})",
                info.name
            )));
        }
        Ok((start, byte_size))
    }

    fn validate_tensor_ranges(&self, source_len: u64) -> Result<()> {
        for info in &self.tensors {
            self.tensor_span(info, source_len)?;
        }
        Ok(())
    }

    fn read_tensor_bytes(&self, info: &TensorInfo) -> Result<Vec<u8>> {
        if let Some(ref mmap) = self.mmap {
            #[cfg(feature = "mmap")]
            let data = &mmap.mmap[..];
            #[cfg(not(feature = "mmap"))]
            let data = &mmap.data[..];

            let (start, byte_size) = self.tensor_span(info, data.len() as u64)?;
            let start = usize::try_from(start).map_err(|_| {
                RuvLLMError::Model(format!(
                    "GGUF tensor offset does not fit usize: {}",
                    info.name
                ))
            })?;
            // The span is bounded by this slice's usize length before indexing.
            return Ok(data[start..start + byte_size].to_vec());
        }

        // Read from file
        let mut file = File::open(&self.path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to open file: {}", e)))?;

        let source_len = file
            .metadata()
            .map_err(|e| RuvLLMError::Model(format!("Failed to get GGUF file length: {e}")))?
            .len();
        let (start, byte_size) = self.tensor_span(info, source_len)?;
        file.seek(SeekFrom::Start(start))
            .map_err(|e| RuvLLMError::Model(format!("Failed to seek: {}", e)))?;

        let mut data = vec![0u8; byte_size];
        file.read_exact(&mut data)
            .map_err(|e| RuvLLMError::Model(format!("Failed to read tensor: {}", e)))?;

        Ok(data)
    }

    fn stream_f32_tensor<F>(&self, info: &TensorInfo, chunk_size: usize, f: &mut F) -> Result<()>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let num_elements: usize = info.shape.iter().product();
        let mut file = File::open(&self.path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to open file: {}", e)))?;

        file.seek(SeekFrom::Start(self.data_offset + info.offset))
            .map_err(|e| RuvLLMError::Model(format!("Failed to seek: {}", e)))?;

        let mut processed = 0;
        let mut buffer = vec![0u8; chunk_size * 4];

        while processed < num_elements {
            let remaining = num_elements - processed;
            let this_chunk = remaining.min(chunk_size);
            let byte_count = this_chunk * 4;

            file.read_exact(&mut buffer[..byte_count])
                .map_err(|e| RuvLLMError::Model(format!("Failed to read: {}", e)))?;

            let floats: Vec<f32> = buffer[..byte_count]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();

            f(&floats)?;
            processed += this_chunk;
        }

        Ok(())
    }

    fn stream_f16_tensor<F>(&self, info: &TensorInfo, chunk_size: usize, f: &mut F) -> Result<()>
    where
        F: FnMut(&[f32]) -> Result<()>,
    {
        let num_elements: usize = info.shape.iter().product();
        let mut file = File::open(&self.path)
            .map_err(|e| RuvLLMError::Model(format!("Failed to open file: {}", e)))?;

        file.seek(SeekFrom::Start(self.data_offset + info.offset))
            .map_err(|e| RuvLLMError::Model(format!("Failed to seek: {}", e)))?;

        let mut processed = 0;
        let mut buffer = vec![0u8; chunk_size * 2];

        while processed < num_elements {
            let remaining = num_elements - processed;
            let this_chunk = remaining.min(chunk_size);
            let byte_count = this_chunk * 2;

            file.read_exact(&mut buffer[..byte_count])
                .map_err(|e| RuvLLMError::Model(format!("Failed to read: {}", e)))?;

            let floats: Vec<f32> = buffer[..byte_count]
                .chunks_exact(2)
                .map(|b| {
                    let bits = u16::from_le_bytes([b[0], b[1]]);
                    half::f16::from_bits(bits).to_f32()
                })
                .collect();

            f(&floats)?;
            processed += this_chunk;
        }

        Ok(())
    }
}

// ============================================================================
// Model Loader for Backend Integration
// ============================================================================

/// GGUF model loader for backend integration.
///
/// This struct wraps a `GgufFile` and provides higher-level methods
/// for model loading and configuration extraction.
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::gguf::GgufModelLoader;
/// use std::path::Path;
///
/// let loader = GgufModelLoader::load(Path::new("model.gguf"))?;
///
/// println!("Architecture: {:?}", loader.architecture());
/// println!("Config: {:?}", loader.config());
///
/// // Convert to Candle model
/// let model = loader.to_candle_model(&device)?;
/// ```
pub struct GgufModelLoader {
    file: GgufFile,
}

impl GgufModelLoader {
    /// Load a GGUF file.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the GGUF file
    pub fn load(path: &Path) -> Result<Self> {
        let file = GgufFile::open_mmap(path)?;
        Ok(Self { file })
    }

    /// Get the underlying GGUF file.
    pub fn file(&self) -> &GgufFile {
        &self.file
    }

    /// Get the model architecture.
    pub fn architecture(&self) -> Option<ModelArchitecture> {
        self.file.architecture_type()
    }

    /// Get the model configuration.
    pub fn config(&self) -> ModelConfig {
        ModelConfig {
            architecture: self.file.architecture().map(|s| s.to_string()),
            context_length: self.file.context_length(),
            embedding_length: self.file.embedding_length(),
            head_count: self.file.head_count(),
            head_count_kv: self.file.head_count_kv(),
            layer_count: self.file.layer_count(),
            vocab_size: self.file.vocab_size(),
            rope_freq_base: self.file.rope_freq_base(),
            feed_forward_length: self.file.feed_forward_length(),
        }
    }

    /// Get list of tensor names that match a pattern.
    pub fn find_tensors(&self, pattern: &str) -> Vec<&str> {
        self.file
            .tensor_names()
            .filter(|name| name.contains(pattern))
            .collect()
    }

    /// Check if this is a quantized model.
    pub fn is_quantized(&self) -> bool {
        self.file.tensors.iter().any(|t| t.dtype.is_quantized())
    }

    /// Get the primary quantization type.
    pub fn quantization_type(&self) -> Option<GgufQuantType> {
        // Find the most common quantization type among weight tensors
        let mut counts: HashMap<GgufQuantType, usize> = HashMap::new();

        for tensor in &self.file.tensors {
            if tensor.name.contains("weight") {
                *counts.entry(tensor.dtype).or_insert(0) += 1;
            }
        }

        counts
            .into_iter()
            .max_by_key(|(_, count)| *count)
            .map(|(dtype, _)| dtype)
    }

    /// Convert to a Candle-compatible model (stub for integration).
    #[cfg(feature = "candle")]
    pub fn to_candle_model(&self, _device: &candle_core::Device) -> Result<()> {
        // This would be implemented based on the specific Candle model architecture
        Err(RuvLLMError::Model(
            "Candle model conversion not yet implemented".to_string(),
        ))
    }
}

/// Model configuration extracted from GGUF metadata.
#[derive(Debug, Clone, Default)]
pub struct ModelConfig {
    /// Model architecture name
    pub architecture: Option<String>,
    /// Maximum context/sequence length
    pub context_length: Option<usize>,
    /// Hidden/embedding dimension
    pub embedding_length: Option<usize>,
    /// Number of attention heads
    pub head_count: Option<usize>,
    /// Number of key-value heads (for GQA)
    pub head_count_kv: Option<usize>,
    /// Number of transformer layers
    pub layer_count: Option<usize>,
    /// Vocabulary size
    pub vocab_size: Option<usize>,
    /// RoPE frequency base
    pub rope_freq_base: Option<f32>,
    /// Feed-forward hidden dimension
    pub feed_forward_length: Option<usize>,
}

// ============================================================================
// Utility Functions
// ============================================================================

/// Align an offset to the specified alignment.
#[inline]
fn align_offset(offset: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        return Err(RuvLLMError::Model(
            "GGUF alignment must be nonzero".to_owned(),
        ));
    }
    offset
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
        .ok_or_else(|| RuvLLMError::Model("GGUF alignment overflow".to_owned()))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tensor_fixture(
        alignment: Option<(u32, u64)>,
        tensor_offset: u64,
        padding_alignment: usize,
    ) -> tempfile::NamedTempFile {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        bytes.extend_from_slice(&GGUF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&u64::from(alignment.is_some()).to_le_bytes());
        if let Some((value_type, value)) = alignment {
            let key = b"general.alignment";
            bytes.extend_from_slice(&(key.len() as u64).to_le_bytes());
            bytes.extend_from_slice(key);
            bytes.extend_from_slice(&value_type.to_le_bytes());
            match value_type {
                4 => bytes.extend_from_slice(&u32::try_from(value).unwrap().to_le_bytes()),
                10 => bytes.extend_from_slice(&value.to_le_bytes()),
                other => panic!("unsupported fixture metadata type {other}"),
            }
        }
        let name = b"test.weight";
        bytes.extend_from_slice(&(name.len() as u64).to_le_bytes());
        bytes.extend_from_slice(name);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&tensor_offset.to_le_bytes());
        bytes.resize(
            bytes.len().div_ceil(padding_alignment) * padding_alignment,
            0,
        );
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        let mut file = tempfile::NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        file.flush().unwrap();
        file
    }

    fn open_fixture(file: &tempfile::NamedTempFile, mapped: bool) -> GgufFile {
        let result = if mapped {
            GgufFile::open_mmap(file.path())
        } else {
            GgufFile::open(file.path())
        };
        match result {
            Ok(file) => file,
            Err(error) => panic!("valid GGUF fixture must open, mapped={mapped}: {error}"),
        }
    }

    fn assert_model_error(error: RuvLLMError, diagnostic: &str) {
        match error {
            RuvLLMError::Model(message) => assert!(
                message.contains(diagnostic),
                "expected {diagnostic:?} diagnostic, got {message:?}"
            ),
            other => panic!("expected Model error, got {other:?}"),
        }
    }

    #[test]
    fn valid_alignment_preserves_full_tensor_reads() {
        for (alignment, padding) in [(None, 32), (Some((4, 32)), 32), (Some((4, 3)), 3)] {
            let fixture = tensor_fixture(alignment, 0, padding);
            for mapped in [false, true] {
                let file = open_fixture(&fixture, mapped);
                assert_eq!(file.alignment, padding);
                assert_eq!(file.load_tensor_f32("test.weight").unwrap(), [1.0]);
                let quantized = file.load_tensor_quantized("test.weight").unwrap();
                assert_eq!(quantized.dtype, GgufQuantType::F32);
                assert_eq!(quantized.shape, [1]);
                assert_eq!(quantized.num_elements, 1);
                assert_eq!(quantized.data, 1.0f32.to_le_bytes());
                if mapped {
                    assert_eq!(
                        file.tensor_data(file.get_tensor("test.weight").unwrap()),
                        &1.0f32.to_le_bytes()
                    );
                }
            }
        }
    }

    #[test]
    fn malformed_alignment_returns_error_for_both_open_modes() {
        for alignment in [0, u64::MAX] {
            assert_model_error(align_offset(100, alignment).unwrap_err(), "alignment");
            let value_type = if alignment == 0 { 4 } else { 10 };
            let fixture = tensor_fixture(Some((value_type, alignment)), 0, 1);
            for mapped in [false, true] {
                let result = if mapped {
                    GgufFile::open_mmap(fixture.path())
                } else {
                    GgufFile::open(fixture.path())
                };
                assert_model_error(
                    result.err().expect("malformed alignment must refuse"),
                    "alignment",
                );
            }
        }
    }

    #[test]
    fn invalid_tensor_headers_are_refused_before_mapped_access() {
        for offset in [4096, 2] {
            let fixture = tensor_fixture(Some((4, 32)), offset, 32);
            for mapped in [false, true] {
                let result = if mapped {
                    GgufFile::open_mmap(fixture.path())
                } else {
                    GgufFile::open(fixture.path())
                };
                assert_model_error(
                    result
                        .err()
                        .expect("tensor range must refuse before access"),
                    "tensor range outside file",
                );
            }
        }
    }

    #[test]
    fn tensor_reads_reject_start_and_end_overflow() {
        let fixture = tensor_fixture(Some((4, 32)), 0, 32);
        for mapped in [false, true] {
            let file = open_fixture(&fixture, mapped);
            let mut info = file.get_tensor("test.weight").unwrap().clone();
            for offset in [
                u64::MAX - file.data_offset + 1,
                u64::MAX - 2 - file.data_offset,
            ] {
                info.offset = offset;
                assert_model_error(
                    file.tensor_span(&info, u64::MAX).unwrap_err(),
                    "tensor offset overflow",
                );
                assert_model_error(
                    file.read_tensor_bytes(&info).unwrap_err(),
                    "tensor offset overflow",
                );
            }
        }
    }

    #[test]
    fn tensor_reads_reject_ranges_past_source_end() {
        let fixture = tensor_fixture(Some((4, 32)), 0, 32);
        for mapped in [false, true] {
            let file = open_fixture(&fixture, mapped);
            let mut info = file.get_tensor("test.weight").unwrap().clone();
            for offset in [4096, 2] {
                info.offset = offset;
                assert_model_error(
                    file.read_tensor_bytes(&info).unwrap_err(),
                    "tensor range outside file",
                );
            }
        }
    }

    #[test]
    fn tensor_reads_reject_size_overflow() {
        let fixture = tensor_fixture(Some((4, 32)), 0, 32);
        for mapped in [false, true] {
            let file = open_fixture(&fixture, mapped);
            let mut info = file.get_tensor("test.weight").unwrap().clone();
            info.shape = vec![usize::MAX / 2 + 1, 2];
            assert_model_error(
                file.read_tensor_bytes(&info).unwrap_err(),
                "tensor element count overflow",
            );
            info.shape = vec![usize::MAX / 4 + 1];
            assert_model_error(
                file.read_tensor_bytes(&info).unwrap_err(),
                "tensor byte size overflow",
            );
        }
    }

    #[test]
    fn test_align_offset() {
        assert_eq!(align_offset(0, 32).unwrap(), 0);
        assert_eq!(align_offset(1, 32).unwrap(), 32);
        assert_eq!(align_offset(31, 32).unwrap(), 32);
        assert_eq!(align_offset(32, 32).unwrap(), 32);
        assert_eq!(align_offset(33, 32).unwrap(), 64);
    }

    #[test]
    fn test_gguf_magic() {
        // "GGUF" in little-endian
        assert_eq!(GGUF_MAGIC, 0x46554747);
        let bytes = GGUF_MAGIC.to_le_bytes();
        assert_eq!(&bytes, b"GGUF");
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfig::default();
        assert!(config.architecture.is_none());
        assert!(config.context_length.is_none());
    }
}
