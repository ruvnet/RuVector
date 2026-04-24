//! TurboQuant sidecar profile loading (ADR-129)
//!
//! Loads `.turboquant.json` sidecar files that sit next to GGUF model files,
//! providing per-layer quantization overrides and eviction policy defaults.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::error::{Result, RuvLLMError};
use crate::quantize::turbo_quant::{TurboQuantBits, TurboQuantConfig};

// ============================================================================
// Profile types
// ============================================================================

/// Per-layer quantization override from the sidecar JSON.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerConfig {
    /// Bit-width for this layer (e.g. "2.5", "3.0", "3.5", "4.0")
    pub bits: String,
    /// Optional human-readable reason for the override
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

/// A `.turboquant.json` sidecar profile that can override the default
/// TurboQuant configuration on a per-layer basis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TurboQuantProfile {
    /// Schema version (must be 1)
    pub version: u32,
    /// Default bit-width applied to all layers unless overridden
    pub default_bits: String,
    /// Default eviction policy (e.g. "h2o", "fifo")
    #[serde(default = "default_eviction")]
    pub default_eviction: String,
    /// Whether to enable QJL residual correction globally
    #[serde(default = "default_use_qjl")]
    pub use_qjl: bool,
    /// Per-layer overrides keyed by `layer_N`
    #[serde(default)]
    pub per_layer_config: HashMap<String, LayerConfig>,
}

fn default_eviction() -> String {
    "h2o".to_string()
}

fn default_use_qjl() -> bool {
    true
}

// ============================================================================
// Implementation
// ============================================================================

impl TurboQuantProfile {
    /// Load a profile from a JSON file path.
    pub fn load(path: &Path) -> Result<Self> {
        let data = std::fs::read_to_string(path).map_err(|e| {
            RuvLLMError::Config(format!(
                "failed to read turboquant profile {}: {e}",
                path.display()
            ))
        })?;
        let profile: Self = serde_json::from_str(&data).map_err(|e| {
            RuvLLMError::Config(format!(
                "invalid turboquant profile {}: {e}",
                path.display()
            ))
        })?;
        if profile.version != 1 {
            return Err(RuvLLMError::Config(format!(
                "unsupported turboquant profile version: {}",
                profile.version
            )));
        }
        Ok(profile)
    }

    /// Discover a sidecar profile next to a GGUF file.
    ///
    /// Checks (in order):
    /// 1. `{gguf_path}.turboquant.json`  (e.g. `model.gguf.turboquant.json`)
    /// 2. `{stem}.turboquant.json`       (e.g. `model.turboquant.json`)
    ///
    /// Returns `None` if neither file exists.
    pub fn discover(gguf_path: &Path) -> Result<Option<Self>> {
        // Try {path}.turboquant.json
        let mut candidate = PathBuf::from(gguf_path);
        let mut name = candidate.file_name().unwrap_or_default().to_os_string();
        name.push(".turboquant.json");
        candidate.set_file_name(&name);

        if candidate.is_file() {
            return Self::load(&candidate).map(Some);
        }

        // Try {stem}.turboquant.json
        if let Some(stem) = gguf_path.file_stem() {
            let mut stem_candidate = gguf_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf();
            stem_candidate.push(format!("{}.turboquant.json", stem.to_string_lossy()));

            if stem_candidate.is_file() {
                return Self::load(&stem_candidate).map(Some);
            }
        }

        Ok(None)
    }

    /// Convert this profile to a [`TurboQuantConfig`] for a specific layer.
    ///
    /// Applies the per-layer override if one exists for `layer_{idx}`,
    /// otherwise uses the profile defaults.
    pub fn to_config(&self, layer: usize) -> Result<TurboQuantConfig> {
        let bits_str = self
            .per_layer_config
            .get(&format!("layer_{layer}"))
            .map(|lc| lc.bits.as_str())
            .unwrap_or(&self.default_bits);

        let bits = parse_bits(bits_str)?;

        Ok(TurboQuantConfig {
            bits,
            rotation_seed: 42,
            enable_qjl_residual: self.use_qjl,
            block_size: 128,
        })
    }

    /// Returns the default profile (3.5-bit, H2O eviction, QJL enabled).
    pub fn default_profile() -> Self {
        Self {
            version: 1,
            default_bits: "3.5".to_string(),
            default_eviction: "h2o".to_string(),
            use_qjl: true,
            per_layer_config: HashMap::new(),
        }
    }
}

/// Parse a bit-width string like "3.5" into a [`TurboQuantBits`] variant.
fn parse_bits(s: &str) -> Result<TurboQuantBits> {
    match s {
        "2.5" => Ok(TurboQuantBits::Bits2_5),
        "3.0" | "3" => Ok(TurboQuantBits::Bits3_0),
        "3.5" => Ok(TurboQuantBits::Bits3_5),
        "4.0" | "4" => Ok(TurboQuantBits::Bits4_0),
        other => Err(RuvLLMError::Config(format!(
            "unsupported turboquant bit-width: {other:?} (expected 2.5, 3.0, 3.5, or 4.0)"
        ))),
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn sample_json() -> &'static str {
        r#"{
            "version": 1,
            "default_bits": "3.5",
            "default_eviction": "h2o",
            "use_qjl": true,
            "per_layer_config": {
                "layer_0": { "bits": "4.0", "reason": "high entropy" },
                "layer_1": { "bits": "3.5" }
            }
        }"#
    }

    #[test]
    fn test_load_profile() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(sample_json().as_bytes()).unwrap();

        let profile = TurboQuantProfile::load(f.path()).unwrap();
        assert_eq!(profile.version, 1);
        assert_eq!(profile.default_bits, "3.5");
        assert_eq!(profile.default_eviction, "h2o");
        assert!(profile.use_qjl);
        assert_eq!(profile.per_layer_config.len(), 2);
        assert_eq!(
            profile.per_layer_config["layer_0"].reason.as_deref(),
            Some("high entropy")
        );
    }

    #[test]
    fn test_to_config_default_layer() {
        let profile = TurboQuantProfile::default_profile();
        let cfg = profile.to_config(99).unwrap();
        assert_eq!(cfg.bits, TurboQuantBits::Bits3_5);
        assert!(cfg.enable_qjl_residual);
    }

    #[test]
    fn test_to_config_per_layer_override() {
        let mut f = NamedTempFile::new().unwrap();
        f.write_all(sample_json().as_bytes()).unwrap();
        let profile = TurboQuantProfile::load(f.path()).unwrap();

        let cfg0 = profile.to_config(0).unwrap();
        assert_eq!(cfg0.bits, TurboQuantBits::Bits4_0);

        let cfg1 = profile.to_config(1).unwrap();
        assert_eq!(cfg1.bits, TurboQuantBits::Bits3_5);

        // Layer without override falls back to default
        let cfg2 = profile.to_config(2).unwrap();
        assert_eq!(cfg2.bits, TurboQuantBits::Bits3_5);
    }

    #[test]
    fn test_discover_with_suffix() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        std::fs::write(&gguf_path, b"fake").unwrap();

        // Write sidecar as model.gguf.turboquant.json
        let sidecar = dir.path().join("model.gguf.turboquant.json");
        std::fs::write(&sidecar, sample_json()).unwrap();

        let found = TurboQuantProfile::discover(&gguf_path).unwrap();
        assert!(found.is_some());
        assert_eq!(found.unwrap().default_bits, "3.5");
    }

    #[test]
    fn test_discover_with_stem() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        std::fs::write(&gguf_path, b"fake").unwrap();

        // Write sidecar as model.turboquant.json (stem-based)
        let sidecar = dir.path().join("model.turboquant.json");
        std::fs::write(&sidecar, sample_json()).unwrap();

        let found = TurboQuantProfile::discover(&gguf_path).unwrap();
        assert!(found.is_some());
    }

    #[test]
    fn test_discover_none() {
        let dir = tempfile::tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        let found = TurboQuantProfile::discover(&gguf_path).unwrap();
        assert!(found.is_none());
    }

    #[test]
    fn test_invalid_bits() {
        let profile = TurboQuantProfile {
            default_bits: "7.0".to_string(),
            ..TurboQuantProfile::default_profile()
        };
        assert!(profile.to_config(0).is_err());
    }
}
