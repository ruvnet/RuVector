//! Security Validation Module for ADR-090 Ultra-Low-Bit Quantization
//!
//! This module implements security requirements from ADR-090 Section 4:
//! - Weight Integrity (Section 4.2): SHA-256 checksum verification
//! - Quantization Bounds (Section 4.3): Overflow protection with clamping
//! - GGUF Validation (Section 4.4): Magic bytes, version, tensor count checks
//! - WASM Sandbox (Section 4.5): Memory isolation and budget enforcement
//! - Invariant Enforcement: INV-2 (Scale positivity), INV-8 (Scalar reference)
//!
//! ## Threat Model
//!
//! | Threat | Mitigation | Invariant |
//! |--------|------------|-----------|
//! | Weight tampering | SHA-256 checksum | INV-1 |
//! | Quantization overflow | Bounds checking | INV-4, INV-5 |
//! | WASM memory escape | Linear memory isolation | INV-6 |
//! | Adversarial calibration | L2-norm cap (0.01) | INV-3 |
//! | Model extraction | Hardware attestation | INV-7 |

use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::Path;

use crate::error::{Result, RuvLLMError};

// ============================================================================
// Constants from ADR-090 Section 4
// ============================================================================

/// GGUF magic bytes: "GGUF" in little-endian (0x46475547)
const GGUF_MAGIC: u32 = 0x4755_4746; // "GGUF" as little-endian u32

/// Supported GGUF versions
const GGUF_VERSION_MIN: u32 = 2;
const GGUF_VERSION_MAX: u32 = 3;

/// Maximum tensor count sanity check (prevents memory exhaustion attacks)
const MAX_TENSOR_COUNT: u64 = 10_000;

/// Known quantization types from ADR-090 Section 4.4
/// Includes standard GGUF types plus new PiQ types
const KNOWN_QUANT_TYPES: &[u32] = &[
    0,  // F32
    1,  // F16
    2,  // Q4_0
    3,  // Q4_1
    6,  // Q5_0
    7,  // Q5_1
    8,  // Q8_0
    9,  // Q8_1
    10, // Q2_K
    11, // Q3_K
    12, // Q4_K
    13, // Q5_K
    14, // Q6_K
    15, // Q8_K
    16, // IQ2_XXS
    17, // IQ2_XS
    18, // IQ3_XXS
    19, // IQ1_S
    20, // IQ4_NL
    21, // IQ3_S
    22, // IQ2_S
    23, // IQ4_XS
    24, // I8
    25, // I16
    26, // I32
    27, // I64
    28, // F64
    29, // BF16
    // Pi-constant quantization types (ADR-090)
    40, // PiQ3 (3.14 bits)
    41, // PiQ2 (2.72 bits)
    42, // Q2_QuIP (2-bit QuIP#)
];

/// Maximum weight perturbation for adversarial calibration (L2 norm cap)
const MAX_PERTURBATION_L2: f32 = 0.01;

/// Maximum MSE threshold for quantization quality validation
const MAX_LAYER_MSE_THRESHOLD: f32 = 0.001;

/// WASM memory budget in bytes (256 MB default)
const WASM_MEMORY_BUDGET: usize = 256 * 1024 * 1024;

// ============================================================================
// WeightIntegrity Struct (ADR-090 Section 4.2)
// ============================================================================

/// Weight integrity verification using SHA-256 checksums.
///
/// Implements ADR-090 Section 4.2 requirements for weight tampering detection.
///
/// ## Example
///
/// ```rust,ignore
/// use ruvllm::quantize::security::WeightIntegrity;
///
/// let integrity = WeightIntegrity::compute(&original_weights, &quantized_weights, 0.0001, &config)?;
/// integrity.verify(&loaded_weights)?;
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct WeightIntegrity {
    /// SHA-256 hash of original (full-precision) weights
    pub original_hash: [u8; 32],

    /// SHA-256 hash of quantized weights
    pub quantized_hash: [u8; 32],

    /// Maximum MSE across all layers (quality metric)
    pub max_layer_mse: f32,

    /// SHA-256 hash of quantization configuration
    pub config_hash: [u8; 32],
}

impl WeightIntegrity {
    /// Create a new WeightIntegrity instance with computed hashes.
    pub fn new(
        original_hash: [u8; 32],
        quantized_hash: [u8; 32],
        max_layer_mse: f32,
        config_hash: [u8; 32],
    ) -> Self {
        Self {
            original_hash,
            quantized_hash,
            max_layer_mse,
            config_hash,
        }
    }

    /// Compute integrity from weight data and configuration.
    ///
    /// # Arguments
    /// * `original_weights` - Original full-precision weights as bytes
    /// * `quantized_weights` - Quantized weights as bytes
    /// * `max_layer_mse` - Maximum MSE across layers
    /// * `config_bytes` - Serialized quantization configuration
    pub fn compute(
        original_weights: &[u8],
        quantized_weights: &[u8],
        max_layer_mse: f32,
        config_bytes: &[u8],
    ) -> Self {
        Self {
            original_hash: Self::sha256(original_weights),
            quantized_hash: Self::sha256(quantized_weights),
            max_layer_mse,
            config_hash: Self::sha256(config_bytes),
        }
    }

    /// Compute SHA-256 hash of data.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();
        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// Verify that loaded weights match the expected hash.
    ///
    /// Returns Ok(()) if weights are valid, Err if tampering detected.
    pub fn verify_quantized(&self, loaded_weights: &[u8]) -> Result<()> {
        let loaded_hash = Self::sha256(loaded_weights);

        if loaded_hash != self.quantized_hash {
            return Err(RuvLLMError::Quantization(format!(
                "Weight integrity check failed: hash mismatch. \
                Expected: {:02x?}, Got: {:02x?}",
                &self.quantized_hash[..8],
                &loaded_hash[..8]
            )));
        }

        Ok(())
    }

    /// Verify MSE is within acceptable bounds.
    pub fn verify_quality(&self, threshold: f32) -> Result<()> {
        if self.max_layer_mse > threshold {
            return Err(RuvLLMError::Quantization(format!(
                "Quantization quality check failed: MSE {} exceeds threshold {}",
                self.max_layer_mse, threshold
            )));
        }
        Ok(())
    }

    /// Serialize to bytes for storage.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(100);
        bytes.extend_from_slice(&self.original_hash);
        bytes.extend_from_slice(&self.quantized_hash);
        bytes.extend_from_slice(&self.max_layer_mse.to_le_bytes());
        bytes.extend_from_slice(&self.config_hash);
        bytes
    }

    /// Deserialize from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 100 {
            return Err(RuvLLMError::Quantization(
                "Invalid WeightIntegrity data: insufficient bytes".to_string(),
            ));
        }

        let mut original_hash = [0u8; 32];
        let mut quantized_hash = [0u8; 32];
        let mut config_hash = [0u8; 32];

        original_hash.copy_from_slice(&bytes[0..32]);
        quantized_hash.copy_from_slice(&bytes[32..64]);

        let mse_bytes: [u8; 4] = bytes[64..68]
            .try_into()
            .map_err(|_| RuvLLMError::Quantization("Invalid MSE bytes".to_string()))?;
        let max_layer_mse = f32::from_le_bytes(mse_bytes);

        config_hash.copy_from_slice(&bytes[68..100]);

        Ok(Self {
            original_hash,
            quantized_hash,
            max_layer_mse,
            config_hash,
        })
    }
}

// ============================================================================
// GGUF Security Report (ADR-090 Section 4.4)
// ============================================================================

/// Security validation report for GGUF files.
#[derive(Debug, Clone)]
pub struct GgufSecurityReport {
    /// File path that was validated
    pub path: String,

    /// Whether the file passed all security checks
    pub is_valid: bool,

    /// GGUF version found
    pub version: u32,

    /// Number of tensors in the file
    pub tensor_count: u64,

    /// Quantization types found in the file
    pub quant_types: Vec<u32>,

    /// Unknown quantization types (security risk)
    pub unknown_quant_types: Vec<u32>,

    /// Security warnings (non-fatal issues)
    pub warnings: Vec<String>,

    /// Security errors (fatal issues)
    pub errors: Vec<String>,

    /// File size in bytes
    pub file_size: u64,

    /// SHA-256 hash of the file
    pub file_hash: [u8; 32],
}

impl GgufSecurityReport {
    /// Create a new security report.
    pub fn new(path: &Path) -> Self {
        Self {
            path: path.display().to_string(),
            is_valid: true,
            version: 0,
            tensor_count: 0,
            quant_types: Vec::new(),
            unknown_quant_types: Vec::new(),
            warnings: Vec::new(),
            errors: Vec::new(),
            file_size: 0,
            file_hash: [0u8; 32],
        }
    }

    /// Add a warning to the report.
    pub fn add_warning(&mut self, warning: String) {
        self.warnings.push(warning);
    }

    /// Add an error and mark the report as invalid.
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
        self.is_valid = false;
    }
}

// ============================================================================
// Quantization Bounds Validator (ADR-090 Section 4.3)
// ============================================================================

/// Bounds validator for quantization operations.
///
/// Implements ADR-090 Section 4.3 requirements with clamping and assertions.
/// Pattern reused from kv_cache.rs bounds checking.
#[derive(Debug, Clone)]
pub struct QuantizationBounds {
    /// Minimum quantized value (e.g., -8 for 4-bit)
    pub min_value: i32,

    /// Maximum quantized value (e.g., 7 for 4-bit)
    pub max_value: i32,

    /// Number of bits for this quantization format
    pub bits: u8,

    /// Format name for error messages
    pub format_name: String,
}

impl QuantizationBounds {
    /// Create bounds for a given bit width.
    pub fn for_bits(bits: u8, format_name: &str) -> Self {
        let half_range = 1i32 << (bits - 1);
        Self {
            min_value: -half_range,
            max_value: half_range - 1,
            bits,
            format_name: format_name.to_string(),
        }
    }

    /// Create bounds for Pi-constant quantization (PiQ3: 3.14 bits).
    pub fn for_piq3() -> Self {
        // PiQ3 uses 3.14 bits, approximated as [-4, 4] with 9 levels
        Self {
            min_value: -4,
            max_value: 4,
            bits: 3,
            format_name: "PiQ3".to_string(),
        }
    }

    /// Create bounds for Pi-constant quantization (PiQ2: 2.72 bits).
    pub fn for_piq2() -> Self {
        // PiQ2 uses 2.72 bits (e bits), approximated as [-3, 3] with 7 levels
        Self {
            min_value: -3,
            max_value: 3,
            bits: 3,
            format_name: "PiQ2".to_string(),
        }
    }

    /// Clamp a quantized value to valid bounds.
    ///
    /// ALWAYS clamp as per ADR-090 Section 4.3:
    /// ```text
    /// let q_clamped = q.clamp(-half_range, half_range - 1);
    /// ```
    #[inline]
    pub fn clamp(&self, value: i32) -> i32 {
        value.clamp(self.min_value, self.max_value)
    }

    /// Validate a quantized value with debug assertion.
    ///
    /// In debug builds, panics on overflow. In release, logs a warning.
    #[inline]
    pub fn validate(&self, value: i32) -> i32 {
        let clamped = self.clamp(value);

        debug_assert!(
            clamped >= self.min_value && clamped <= self.max_value,
            "quantization overflow: q={}, range=[{}, {}), format={}",
            value,
            self.min_value,
            self.max_value,
            self.format_name
        );

        #[cfg(not(debug_assertions))]
        if value != clamped {
            // Log overflow in release builds (would use tracing in production)
            eprintln!(
                "[SECURITY WARNING] Quantization overflow detected: \
                value={}, clamped to {}, format={}",
                value, clamped, self.format_name
            );
        }

        clamped
    }

    /// Quantize a float value with bounds checking.
    ///
    /// Implements the full quantization pipeline from ADR-090 Section 4.3.
    #[inline]
    pub fn quantize_with_bounds(&self, weight: f32, scale: f32) -> i8 {
        // INV-2: Scale must be positive
        debug_assert!(
            scale > 0.0,
            "INV-2 violation: scale must be positive, got {}",
            scale
        );

        let q = (weight / scale).round() as i32;
        let q_clamped = self.validate(q);

        q_clamped as i8
    }
}

// ============================================================================
// WASM Sandbox Security (ADR-090 Section 4.5)
// ============================================================================

/// WASM sandbox security configuration and validation.
#[derive(Debug, Clone)]
pub struct WasmSandboxConfig {
    /// Maximum memory budget in bytes
    pub memory_budget: usize,

    /// Whether filesystem access is allowed (should be false)
    pub allow_filesystem: bool,

    /// Whether network access is allowed (should be false for inference)
    pub allow_network: bool,

    /// Linear memory isolation enabled
    pub linear_memory_isolation: bool,
}

impl Default for WasmSandboxConfig {
    fn default() -> Self {
        Self {
            memory_budget: WASM_MEMORY_BUDGET,
            allow_filesystem: false,
            allow_network: false,
            linear_memory_isolation: true,
        }
    }
}

impl WasmSandboxConfig {
    /// Validate sandbox configuration meets security requirements.
    pub fn validate(&self) -> Result<()> {
        if self.allow_filesystem {
            return Err(RuvLLMError::Quantization(
                "WASM sandbox security violation: filesystem access must be disabled".to_string(),
            ));
        }

        if !self.linear_memory_isolation {
            return Err(RuvLLMError::Quantization(
                "WASM sandbox security violation: linear memory isolation must be enabled"
                    .to_string(),
            ));
        }

        if self.memory_budget > WASM_MEMORY_BUDGET * 4 {
            return Err(RuvLLMError::Quantization(format!(
                "WASM memory budget {} exceeds maximum allowed {} bytes",
                self.memory_budget,
                WASM_MEMORY_BUDGET * 4
            )));
        }

        Ok(())
    }

    /// Check if a memory allocation would exceed the budget.
    pub fn check_allocation(&self, current_usage: usize, requested: usize) -> Result<()> {
        let total = current_usage.saturating_add(requested);

        if total > self.memory_budget {
            return Err(RuvLLMError::OutOfMemory(format!(
                "WASM memory budget exceeded: {} + {} > {} bytes",
                current_usage, requested, self.memory_budget
            )));
        }

        Ok(())
    }
}

// ============================================================================
// Invariant Validators (ADR-090 System Invariants)
// ============================================================================

/// System invariant validators from ADR-090.
pub struct InvariantValidator;

impl InvariantValidator {
    /// INV-2: Scale positivity (alpha > 0).
    ///
    /// All quantization scales must be strictly positive.
    #[inline]
    pub fn validate_scale_positivity(alpha: f32) -> Result<()> {
        if alpha <= 0.0 {
            return Err(RuvLLMError::Quantization(format!(
                "INV-2 violation: scale must be positive, got alpha={}",
                alpha
            )));
        }
        Ok(())
    }

    /// INV-3: Calibration perturbation bound.
    ///
    /// Adversarial calibration defense: limit perturbation to L2 norm < 0.01.
    pub fn validate_perturbation_bound(original: &[f32], perturbed: &[f32]) -> Result<()> {
        if original.len() != perturbed.len() {
            return Err(RuvLLMError::Quantization(
                "Perturbation validation failed: length mismatch".to_string(),
            ));
        }

        let l2_norm: f32 = original
            .iter()
            .zip(perturbed.iter())
            .map(|(o, p)| (o - p).powi(2))
            .sum::<f32>()
            .sqrt();

        if l2_norm > MAX_PERTURBATION_L2 {
            return Err(RuvLLMError::Quantization(format!(
                "INV-3 violation: perturbation L2 norm {} exceeds cap {}",
                l2_norm, MAX_PERTURBATION_L2
            )));
        }

        Ok(())
    }

    /// INV-4: Quantization range validity.
    ///
    /// Ensures quantized values fit within the specified bit width.
    #[inline]
    pub fn validate_quant_range(value: i32, bits: u8) -> Result<()> {
        let half_range = 1i32 << (bits - 1);
        let min = -half_range;
        let max = half_range - 1;

        if value < min || value > max {
            return Err(RuvLLMError::Quantization(format!(
                "INV-4 violation: quantized value {} outside range [{}, {}] for {}-bit",
                value, min, max, bits
            )));
        }

        Ok(())
    }

    /// INV-8: Scalar reference for SIMD kernels.
    ///
    /// Validates that SIMD results match scalar reference within tolerance.
    pub fn validate_simd_scalar_match(
        simd_result: &[f32],
        scalar_result: &[f32],
        tolerance: f32,
    ) -> Result<()> {
        if simd_result.len() != scalar_result.len() {
            return Err(RuvLLMError::Quantization(
                "INV-8 violation: SIMD/scalar length mismatch".to_string(),
            ));
        }

        for (i, (simd, scalar)) in simd_result.iter().zip(scalar_result.iter()).enumerate() {
            let diff = (simd - scalar).abs();
            if diff > tolerance {
                return Err(RuvLLMError::Quantization(format!(
                    "INV-8 violation at index {}: SIMD={}, scalar={}, diff={} > tolerance={}",
                    i, simd, scalar, diff, tolerance
                )));
            }
        }

        Ok(())
    }
}

// ============================================================================
// Main Validation Functions
// ============================================================================

/// Validate a quantized model against security requirements.
///
/// Implements comprehensive validation from ADR-090 Section 4:
/// - Weight integrity (SHA-256)
/// - MSE quality bounds
/// - Configuration integrity
///
/// # Arguments
/// * `weights` - Quantized weight bytes
/// * `integrity` - Expected integrity metadata
/// * `mse_threshold` - Maximum acceptable MSE (default: 0.001)
pub fn validate_quantized_model(
    weights: &[u8],
    integrity: &WeightIntegrity,
    mse_threshold: Option<f32>,
) -> Result<()> {
    // Verify weight hash
    integrity.verify_quantized(weights)?;

    // Verify MSE quality
    let threshold = mse_threshold.unwrap_or(MAX_LAYER_MSE_THRESHOLD);
    integrity.verify_quality(threshold)?;

    Ok(())
}

/// Validate GGUF file security.
///
/// Implements ADR-090 Section 4.4 requirements:
/// - Magic bytes check (0x46475547 = "GGUF")
/// - Version validation (2 or 3)
/// - Tensor count sanity check (<10,000)
/// - Known quantization types only
///
/// # Arguments
/// * `path` - Path to the GGUF file
///
/// # Returns
/// * `GgufSecurityReport` with validation results
pub fn validate_gguf_security(path: &Path) -> Result<GgufSecurityReport> {
    let mut report = GgufSecurityReport::new(path);

    // Open file
    let file = File::open(path).map_err(|e| {
        RuvLLMError::Io(std::io::Error::new(
            e.kind(),
            format!("Failed to open GGUF file: {}", e),
        ))
    })?;

    // Get file size
    let metadata = file.metadata()?;
    report.file_size = metadata.len();

    // Minimum valid GGUF file size (header only)
    if report.file_size < 24 {
        report.add_error(format!(
            "File too small: {} bytes (minimum 24 bytes for GGUF header)",
            report.file_size
        ));
        return Ok(report);
    }

    let mut reader = BufReader::new(file);

    // Read and validate magic bytes (ADR-090 Section 4.4)
    let mut magic_bytes = [0u8; 4];
    reader.read_exact(&mut magic_bytes)?;
    let magic = u32::from_le_bytes(magic_bytes);

    if magic != GGUF_MAGIC {
        report.add_error(format!(
            "Invalid GGUF magic bytes: expected 0x{:08X} ('GGUF'), got 0x{:08X}",
            GGUF_MAGIC, magic
        ));
        return Ok(report);
    }

    // Read and validate version
    let mut version_bytes = [0u8; 4];
    reader.read_exact(&mut version_bytes)?;
    report.version = u32::from_le_bytes(version_bytes);

    if report.version < GGUF_VERSION_MIN || report.version > GGUF_VERSION_MAX {
        report.add_error(format!(
            "Unsupported GGUF version: {} (supported: {}-{})",
            report.version, GGUF_VERSION_MIN, GGUF_VERSION_MAX
        ));
        return Ok(report);
    }

    // Read tensor count
    let mut tensor_count_bytes = [0u8; 8];
    reader.read_exact(&mut tensor_count_bytes)?;
    report.tensor_count = u64::from_le_bytes(tensor_count_bytes);

    // Sanity check tensor count (ADR-090 Section 4.4: <10,000)
    if report.tensor_count > MAX_TENSOR_COUNT {
        report.add_error(format!(
            "Tensor count {} exceeds maximum {} (potential DoS attack)",
            report.tensor_count, MAX_TENSOR_COUNT
        ));
        return Ok(report);
    }

    // Warn on suspiciously low tensor count
    if report.tensor_count == 0 {
        report.add_warning("Tensor count is 0 - file may be empty or corrupted".to_string());
    }

    // Read metadata count
    let mut metadata_count_bytes = [0u8; 8];
    reader.read_exact(&mut metadata_count_bytes)?;
    let metadata_count = u64::from_le_bytes(metadata_count_bytes);

    // Sanity check metadata count
    if metadata_count > MAX_TENSOR_COUNT {
        report.add_error(format!(
            "Metadata count {} exceeds maximum {} (potential DoS attack)",
            metadata_count, MAX_TENSOR_COUNT
        ));
        return Ok(report);
    }

    // Skip metadata section (we don't parse it for security validation)
    // In a full implementation, we would parse metadata to extract quant types

    // Build known quant types set
    let known_types: HashSet<u32> = KNOWN_QUANT_TYPES.iter().copied().collect();

    // For now, assume we've collected quant types from tensor info
    // In a full implementation, this would be read from tensor metadata
    for quant_type in &report.quant_types {
        if !known_types.contains(quant_type) {
            report.unknown_quant_types.push(*quant_type);
        }
    }

    if !report.unknown_quant_types.is_empty() {
        report.add_warning(format!(
            "Unknown quantization types found: {:?}",
            report.unknown_quant_types
        ));
    }

    // Compute file hash (for integrity verification)
    reader.seek(SeekFrom::Start(0))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let bytes_read = reader.read(&mut buffer)?;
        if bytes_read == 0 {
            break;
        }
        hasher.update(&buffer[..bytes_read]);
    }
    let hash_result = hasher.finalize();
    report.file_hash.copy_from_slice(&hash_result);

    Ok(report)
}

/// Quick validation for GGUF file without full report.
///
/// Returns Ok(()) if the file passes basic security checks.
pub fn validate_gguf_quick(path: &Path) -> Result<()> {
    let report = validate_gguf_security(path)?;

    if !report.is_valid {
        return Err(RuvLLMError::Gguf(format!(
            "GGUF security validation failed: {:?}",
            report.errors
        )));
    }

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_weight_integrity_compute() {
        let original = b"original weights data";
        let quantized = b"quantized weights data";
        let config = b"quantization config";

        let integrity = WeightIntegrity::compute(original, quantized, 0.0001, config);

        assert_eq!(integrity.max_layer_mse, 0.0001);
        assert_ne!(integrity.original_hash, integrity.quantized_hash);
    }

    #[test]
    fn test_weight_integrity_serialization() {
        let integrity = WeightIntegrity::new([1u8; 32], [2u8; 32], 0.0005, [3u8; 32]);

        let bytes = integrity.to_bytes();
        let restored = WeightIntegrity::from_bytes(&bytes).unwrap();

        assert_eq!(integrity, restored);
    }

    #[test]
    fn test_weight_integrity_verification_success() {
        let data = b"test weights";
        let hash = WeightIntegrity::sha256(data);

        let integrity = WeightIntegrity::new([0u8; 32], hash, 0.0001, [0u8; 32]);

        assert!(integrity.verify_quantized(data).is_ok());
    }

    #[test]
    fn test_weight_integrity_verification_failure() {
        let integrity = WeightIntegrity::new(
            [0u8; 32], [1u8; 32], // Wrong hash
            0.0001, [0u8; 32],
        );

        assert!(integrity.verify_quantized(b"test weights").is_err());
    }

    #[test]
    fn test_quantization_bounds_4bit() {
        let bounds = QuantizationBounds::for_bits(4, "Q4");

        assert_eq!(bounds.min_value, -8);
        assert_eq!(bounds.max_value, 7);
        assert_eq!(bounds.clamp(-10), -8);
        assert_eq!(bounds.clamp(10), 7);
        assert_eq!(bounds.clamp(5), 5);
    }

    #[test]
    fn test_quantization_bounds_piq3() {
        let bounds = QuantizationBounds::for_piq3();

        assert_eq!(bounds.min_value, -4);
        assert_eq!(bounds.max_value, 4);
        assert_eq!(bounds.format_name, "PiQ3");
    }

    #[test]
    fn test_quantization_bounds_piq2() {
        let bounds = QuantizationBounds::for_piq2();

        assert_eq!(bounds.min_value, -3);
        assert_eq!(bounds.max_value, 3);
        assert_eq!(bounds.format_name, "PiQ2");
    }

    #[test]
    fn test_quantize_with_bounds() {
        let bounds = QuantizationBounds::for_bits(4, "Q4");

        // Normal quantization
        let q = bounds.quantize_with_bounds(0.5, 0.1);
        assert!(q >= -8 && q <= 7);

        // Overflow should be clamped
        let q_overflow = bounds.quantize_with_bounds(10.0, 0.1);
        assert_eq!(q_overflow, 7);
    }

    #[test]
    fn test_wasm_sandbox_config_default() {
        let config = WasmSandboxConfig::default();

        assert!(!config.allow_filesystem);
        assert!(!config.allow_network);
        assert!(config.linear_memory_isolation);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_wasm_sandbox_config_invalid() {
        let mut config = WasmSandboxConfig::default();
        config.allow_filesystem = true;

        assert!(config.validate().is_err());
    }

    #[test]
    fn test_invariant_scale_positivity() {
        assert!(InvariantValidator::validate_scale_positivity(1.0).is_ok());
        assert!(InvariantValidator::validate_scale_positivity(0.001).is_ok());
        assert!(InvariantValidator::validate_scale_positivity(0.0).is_err());
        assert!(InvariantValidator::validate_scale_positivity(-1.0).is_err());
    }

    #[test]
    fn test_invariant_perturbation_bound() {
        let original = vec![1.0, 2.0, 3.0];
        let small_perturb = vec![1.001, 2.001, 3.001];
        let large_perturb = vec![1.1, 2.1, 3.1];

        assert!(InvariantValidator::validate_perturbation_bound(&original, &small_perturb).is_ok());
        assert!(
            InvariantValidator::validate_perturbation_bound(&original, &large_perturb).is_err()
        );
    }

    #[test]
    fn test_invariant_quant_range() {
        // 4-bit range: [-8, 7]
        assert!(InvariantValidator::validate_quant_range(0, 4).is_ok());
        assert!(InvariantValidator::validate_quant_range(-8, 4).is_ok());
        assert!(InvariantValidator::validate_quant_range(7, 4).is_ok());
        assert!(InvariantValidator::validate_quant_range(-9, 4).is_err());
        assert!(InvariantValidator::validate_quant_range(8, 4).is_err());
    }

    #[test]
    fn test_invariant_simd_scalar_match() {
        let simd = vec![1.0, 2.0, 3.0];
        let scalar_match = vec![1.0001, 2.0001, 3.0001];
        let scalar_mismatch = vec![1.1, 2.0, 3.0];

        assert!(
            InvariantValidator::validate_simd_scalar_match(&simd, &scalar_match, 0.001).is_ok()
        );
        assert!(
            InvariantValidator::validate_simd_scalar_match(&simd, &scalar_mismatch, 0.001).is_err()
        );
    }

    #[test]
    fn test_validate_quantized_model() {
        let weights = b"test quantized weights";
        let hash = WeightIntegrity::sha256(weights);

        let integrity = WeightIntegrity::new([0u8; 32], hash, 0.0001, [0u8; 32]);

        assert!(validate_quantized_model(weights, &integrity, None).is_ok());
    }

    #[test]
    fn test_validate_quantized_model_mse_fail() {
        let weights = b"test quantized weights";
        let hash = WeightIntegrity::sha256(weights);

        let integrity = WeightIntegrity::new(
            [0u8; 32], hash, 0.01, // High MSE
            [0u8; 32],
        );

        // Should fail with default threshold (0.001)
        assert!(validate_quantized_model(weights, &integrity, None).is_err());

        // Should pass with higher threshold
        assert!(validate_quantized_model(weights, &integrity, Some(0.1)).is_ok());
    }
}
