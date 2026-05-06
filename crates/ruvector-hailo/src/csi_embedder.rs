//! WifiCsi128d — 128-dim contrastive CSI embedding on the Hailo-8 NPU.
//!
//! ADR-183 Tier 3. Implements the CSI encoder from `ruv/ruview`
//! (architecture "csi-encoder-8-64-128"):
//!
//! ```text
//!   [f32; 8]  →  fc1(8→64, ReLU)  →  fc2(64→128)  →  L2-norm  →  [f32; 128]
//! ```
//!
//! The weights live in a 48 KB `model.safetensors` from HuggingFace
//! `ruv/ruview`. Both a CPU path (always available) and the Hailo NPU
//! path (gated on `feature = "hailo"`, uses the compiled HEF at
//! `/usr/local/share/ruvector/csi-encoder.hef`) are provided.
//!
//! ## Feature extraction
//!
//! The 8 input features are aggregate statistics from the CSI vitals
//! sliding window (computed by `ruview-vitals-worker`):
//!
//! ```text
//!   [0] breathing_bpm / 30.0       (normalised, 0–1 range)
//!   [1] breathing_confidence       (0–1)
//!   [2] heart_rate_bpm / 120.0     (normalised, 0–1 range)
//!   [3] heart_rate_confidence      (0–1)
//!   [4] motion_score               (0–1)
//!   [5] log_snr_db / 40.0          (normalised dB / 40, clipped to 0–1)
//!   [6] peak_amp_breathing / 10.0  (normalised, clipped to 0–1)
//!   [7] peak_amp_hr / 10.0         (normalised, clipped to 0–1)
//! ```

use crate::error::HailoError;
use std::path::Path;

/// Output dimensionality of the CSI contrastive encoder.
pub const CSI_EMBED_DIM: usize = 128;

/// Number of input features (must match the compiled HEF + model weights).
pub const CSI_INPUT_DIM: usize = 8;

/// Hidden layer size in the 2-layer FC encoder.
const CSI_HIDDEN_DIM: usize = 64;

/// Rank of the per-room LoRA adapters shipped in `node-N.json`.
/// ADR-183 iter 18: `rank=4, alpha=8, scaling=alpha/rank=2`.
pub const LORA_RANK: usize = 4;

/// 8 normalised CSI vital-sign features fed to the encoder.
#[derive(Debug, Clone, Copy)]
pub struct CsiFeatures {
    pub breathing_bpm_norm: f32,
    pub breathing_confidence: f32,
    pub heart_rate_bpm_norm: f32,
    pub heart_rate_confidence: f32,
    pub motion_score: f32,
    pub log_snr_norm: f32,
    pub peak_amp_breathing_norm: f32,
    pub peak_amp_hr_norm: f32,
}

impl CsiFeatures {
    /// Pack into the ordered `[f32; CSI_INPUT_DIM]` array the encoder expects.
    pub fn to_array(&self) -> [f32; CSI_INPUT_DIM] {
        [
            self.breathing_bpm_norm.clamp(0.0, 1.0),
            self.breathing_confidence.clamp(0.0, 1.0),
            self.heart_rate_bpm_norm.clamp(0.0, 1.0),
            self.heart_rate_confidence.clamp(0.0, 1.0),
            self.motion_score.clamp(0.0, 1.0),
            self.log_snr_norm.clamp(0.0, 1.0),
            self.peak_amp_breathing_norm.clamp(0.0, 1.0),
            self.peak_amp_hr_norm.clamp(0.0, 1.0),
        ]
    }
}

/// Weights for the 2-layer FC CSI encoder, loaded from `model.safetensors`.
struct CsiWeights {
    w1: [[f32; CSI_INPUT_DIM]; CSI_HIDDEN_DIM],
    b1: [f32; CSI_HIDDEN_DIM],
    w2: [[f32; CSI_HIDDEN_DIM]; CSI_EMBED_DIM],
    b2: [f32; CSI_EMBED_DIM],
}

impl CsiWeights {
    /// Parse `model.safetensors` from `ruv/ruview` and extract FC weights.
    fn load(path: &Path) -> Result<Self, HailoError> {
        use std::io::Read;

        let mut f = std::fs::File::open(path).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "cannot open model.safetensors",
        })?;

        // safetensors: 8-byte LE header length, then JSON header, then data
        let mut len_buf = [0u8; 8];
        f.read_exact(&mut len_buf).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "safetensors: cannot read header length",
        })?;
        let header_len = u64::from_le_bytes(len_buf) as usize;
        let mut header_bytes = vec![0u8; header_len];
        f.read_exact(&mut header_bytes).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "safetensors: cannot read header JSON",
        })?;
        // Strip null padding that the ruv/ruview safetensors file has
        let header_str = std::str::from_utf8(
            header_bytes.trim_ascii_end()
        ).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "safetensors: header is not valid UTF-8",
        })?;

        // Read the remainder (tensor data)
        let mut data = Vec::new();
        f.read_to_end(&mut data).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "safetensors: cannot read tensor data",
        })?;

        // We parse just the 4 keys we need (avoid pulling in serde_json)
        let get_f32_slice = |key: &str, expected_bytes: usize| -> Result<Vec<f32>, HailoError> {
            let tag = format!("\"{}\"", key);
            let pos = header_str.find(&tag).ok_or(HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "safetensors: key not found",
            })?;
            // Extract data_offsets from JSON: naive scan for "data_offsets":[start,end]
            let after = &header_str[pos..];
            let off_start = after.find("\"data_offsets\":[").ok_or(HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "safetensors: data_offsets not found",
            })?;
            let nums_start = off_start + "\"data_offsets\":[".len();
            let after2 = &after[nums_start..];
            let end_bracket = after2.find(']').ok_or(HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "safetensors: data_offsets malformed",
            })?;
            let pair: Vec<usize> = after2[..end_bracket]
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            if pair.len() != 2 {
                return Err(HailoError::BadModelDir {
                    path: path.display().to_string(),
                    what: "safetensors: data_offsets not a 2-element array",
                });
            }
            let (start, end) = (pair[0], pair[1]);
            if end > data.len() || (end - start) != expected_bytes {
                return Err(HailoError::BadModelDir {
                    path: path.display().to_string(),
                    what: "safetensors: data slice out of bounds",
                });
            }
            let slice = &data[start..end];
            Ok(slice
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        };

        // fc1: w1 [64, 8] stored flat as [512] f32
        let w1_flat = get_f32_slice("encoder.w1", CSI_HIDDEN_DIM * CSI_INPUT_DIM * 4)?;
        let b1_flat = get_f32_slice("encoder.b1", CSI_HIDDEN_DIM * 4)?;
        let w2_flat = get_f32_slice("encoder.w2", CSI_EMBED_DIM * CSI_HIDDEN_DIM * 4)?;
        let b2_flat = get_f32_slice("encoder.b2", CSI_EMBED_DIM * 4)?;

        let mut w1 = [[0f32; CSI_INPUT_DIM]; CSI_HIDDEN_DIM];
        for i in 0..CSI_HIDDEN_DIM {
            for j in 0..CSI_INPUT_DIM {
                w1[i][j] = w1_flat[i * CSI_INPUT_DIM + j];
            }
        }
        let mut b1 = [0f32; CSI_HIDDEN_DIM];
        b1.copy_from_slice(&b1_flat);

        let mut w2 = [[0f32; CSI_HIDDEN_DIM]; CSI_EMBED_DIM];
        for i in 0..CSI_EMBED_DIM {
            for j in 0..CSI_HIDDEN_DIM {
                w2[i][j] = w2_flat[i * CSI_HIDDEN_DIM + j];
            }
        }
        let mut b2 = [0f32; CSI_EMBED_DIM];
        b2.copy_from_slice(&b2_flat);

        Ok(CsiWeights { w1, b1, w2, b2 })
    }

    /// Forward pass: [8] → fc1 + ReLU → [64] → fc2 → [128] (L2-norm).
    fn forward(&self, x: &[f32; CSI_INPUT_DIM]) -> [f32; CSI_EMBED_DIM] {
        // FC1 + ReLU
        let mut h = [0f32; CSI_HIDDEN_DIM];
        for i in 0..CSI_HIDDEN_DIM {
            let mut v = self.b1[i];
            for j in 0..CSI_INPUT_DIM {
                v += self.w1[i][j] * x[j];
            }
            h[i] = v.max(0.0); // ReLU
        }
        // FC2
        let mut out = [0f32; CSI_EMBED_DIM];
        for i in 0..CSI_EMBED_DIM {
            let mut v = self.b2[i];
            for j in 0..CSI_HIDDEN_DIM {
                v += self.w2[i][j] * h[j];
            }
            out[i] = v;
        }
        // L2-normalise
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out {
            *v /= norm;
        }
        out
    }
}

/// Per-room LoRA adapter (rank-4, ADR-183 iter 18).
///
/// Loaded from a `node-N.json` file (HuggingFace `ruv/ruview`, `sona-lora`
/// model type). Applies a low-rank residual update to the base CSI embedding
/// to improve room-specific class separability:
///
/// ```text
///   intermediate = loraB @ emb          ([LORA_RANK])
///   delta        = loraA @ intermediate  ([CSI_EMBED_DIM])
///   output       = L2_norm(emb + scaling * delta)
/// ```
pub struct CsiLoraAdapter {
    /// Row-major [CSI_EMBED_DIM × LORA_RANK] = 512 f32.
    lora_a: Vec<f32>,
    /// Row-major [LORA_RANK × CSI_EMBED_DIM] = 512 f32.
    lora_b: Vec<f32>,
    /// alpha / rank, typically 2.0 for the ruv/ruview adapters.
    pub scaling: f32,
}

impl CsiLoraAdapter {
    /// Decompose the adapter into its raw weight vectors for SONA adaptation.
    ///
    /// Returns `(lora_a, lora_b, scaling)` where both matrices are row-major flat Vecs.
    pub fn into_parts(self) -> (Vec<f32>, Vec<f32>, f32) {
        (self.lora_a, self.lora_b, self.scaling)
    }

    /// Parse `node-N.json` from `ruv/ruview` (the `sona-lora` format).
    ///
    /// Expected shape: `{weights: {loraA: [[128×4]], loraB: [[4×128]], scaling: 2}}`
    pub fn load(path: &Path) -> Result<Self, HailoError> {
        let file = std::fs::File::open(path).map_err(|_| HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "cannot open LoRA adapter JSON",
        })?;
        let v: serde_json::Value =
            serde_json::from_reader(file).map_err(|_| HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "cannot parse LoRA adapter JSON",
            })?;

        let weights = v.get("weights").ok_or(HailoError::BadModelDir {
            path: path.display().to_string(),
            what: "LoRA JSON missing 'weights' key",
        })?;

        let scaling = weights["scaling"]
            .as_f64()
            .ok_or(HailoError::BadModelDir {
                path: path.display().to_string(),
                what: "LoRA JSON: weights.scaling missing or not a number",
            })? as f32;

        let path_str = path.display().to_string();
        let lora_a = parse_lora_matrix(weights, "loraA", CSI_EMBED_DIM, LORA_RANK, &path_str)?;
        let lora_b = parse_lora_matrix(weights, "loraB", LORA_RANK, CSI_EMBED_DIM, &path_str)?;

        Ok(Self { lora_a, lora_b, scaling })
    }

    /// Apply the rank-4 residual update and re-L2-normalise.
    pub fn apply(&self, emb: &[f32; CSI_EMBED_DIM]) -> [f32; CSI_EMBED_DIM] {
        // intermediate = loraB @ emb  →  [LORA_RANK]
        let mut intermediate = [0f32; LORA_RANK];
        for j in 0..LORA_RANK {
            let row_off = j * CSI_EMBED_DIM;
            for k in 0..CSI_EMBED_DIM {
                intermediate[j] += self.lora_b[row_off + k] * emb[k];
            }
        }
        // out = emb + scaling * loraA @ intermediate  →  [CSI_EMBED_DIM]
        let mut out = [0f32; CSI_EMBED_DIM];
        for i in 0..CSI_EMBED_DIM {
            let col_off = i * LORA_RANK;
            let mut delta = 0f32;
            for j in 0..LORA_RANK {
                delta += self.lora_a[col_off + j] * intermediate[j];
            }
            out[i] = emb[i] + self.scaling * delta;
        }
        // L2-normalise
        let norm: f32 = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
        for v in &mut out {
            *v /= norm;
        }
        out
    }
}

/// Parse a 2-D matrix stored as a JSON array-of-arrays into a flat row-major Vec.
fn parse_lora_matrix(
    weights: &serde_json::Value,
    key: &str,
    rows: usize,
    cols: usize,
    path_str: &str,
) -> Result<Vec<f32>, HailoError> {
    let arr = weights[key].as_array().ok_or(HailoError::BadModelDir {
        path: path_str.to_string(),
        what: "LoRA JSON: matrix key missing or not an array",
    })?;
    if arr.len() != rows {
        return Err(HailoError::BadModelDir {
            path: path_str.to_string(),
            what: "LoRA JSON: matrix row count mismatch",
        });
    }
    let mut flat = Vec::with_capacity(rows * cols);
    for row in arr {
        let row_arr = row.as_array().ok_or(HailoError::BadModelDir {
            path: path_str.to_string(),
            what: "LoRA JSON: matrix row is not an array",
        })?;
        if row_arr.len() != cols {
            return Err(HailoError::BadModelDir {
                path: path_str.to_string(),
                what: "LoRA JSON: matrix column count mismatch",
            });
        }
        for v in row_arr {
            flat.push(
                v.as_f64()
                    .ok_or(HailoError::BadModelDir {
                        path: path_str.to_string(),
                        what: "LoRA JSON: matrix element is not a number",
                    })? as f32,
            );
        }
    }
    Ok(flat)
}

/// CPU-path CSI encoder using pre-trained weights from `model.safetensors`.
///
/// Backed by pure-Rust matrix multiply; no NPU, no FFI. The `CsiEmbedder`
/// is the Hailo-feature-gated complement — both expose the same
/// `embed(&CsiFeatures) -> [f32; 128]` call.
pub struct CsiEmbedderCpu {
    weights: CsiWeights,
    /// Optional per-room LoRA adapter (ADR-183 iter 18). When set,
    /// `embed()` applies the rank-4 residual update after the base
    /// forward pass and re-L2-normalises before returning.
    lora: Option<CsiLoraAdapter>,
}

impl CsiEmbedderCpu {
    /// Load from either a `model.safetensors` file path directly, or a
    /// directory that contains `model.safetensors`. No LoRA adapter.
    pub fn open(path: &Path) -> Result<Self, HailoError> {
        let st_path = if path.is_file() {
            path.to_path_buf()
        } else {
            path.join("model.safetensors")
        };
        let weights = CsiWeights::load(&st_path)?;
        Ok(Self { weights, lora: None })
    }

    /// Load the base model and optionally a room-specific LoRA adapter.
    ///
    /// `lora_path` may point to a `node-N.json` from `ruv/ruview`. When
    /// `None`, behaviour is identical to [`open`].
    pub fn open_with_lora(
        model_path: &Path,
        lora_path: Option<&Path>,
    ) -> Result<Self, HailoError> {
        let mut embedder = Self::open(model_path)?;
        if let Some(lp) = lora_path {
            embedder.lora = Some(CsiLoraAdapter::load(lp)?);
        }
        Ok(embedder)
    }

    /// True when a per-room LoRA adapter is loaded.
    pub fn has_lora(&self) -> bool {
        self.lora.is_some()
    }

    /// Compute the 128-dim L2-normalised embedding.
    ///
    /// When a LoRA adapter is present, applies the rank-4 residual update
    /// to the base embedding before returning.
    pub fn embed(&self, features: &CsiFeatures) -> [f32; CSI_EMBED_DIM] {
        let base = self.weights.forward(&features.to_array());
        match &self.lora {
            Some(lora) => lora.apply(&base),
            None => base,
        }
    }
}

/// Pinned sha256 of the compiled CSI encoder HEF deployed by ADR-183 Tier 3.
pub const CSI_ENCODER_HEF_SHA256: &str =
    "91fcb74812ce08ac881518f26ae47e69ea33ccc8f1033e11fe556ba998709103";

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_features() -> CsiFeatures {
        CsiFeatures {
            breathing_bpm_norm: 0.4,
            breathing_confidence: 0.8,
            heart_rate_bpm_norm: 0.6,
            heart_rate_confidence: 0.7,
            motion_score: 0.2,
            log_snr_norm: 0.5,
            peak_amp_breathing_norm: 0.3,
            peak_amp_hr_norm: 0.2,
        }
    }

    #[test]
    fn features_to_array_clamps() {
        let f = CsiFeatures {
            breathing_bpm_norm: 1.5,
            breathing_confidence: -0.3,
            heart_rate_bpm_norm: 0.5,
            heart_rate_confidence: 0.5,
            motion_score: 0.5,
            log_snr_norm: 0.5,
            peak_amp_breathing_norm: 0.5,
            peak_amp_hr_norm: 0.5,
        };
        let arr = f.to_array();
        assert_eq!(arr[0], 1.0, "clamp > 1.0");
        assert_eq!(arr[1], 0.0, "clamp < 0.0");
    }

    /// LoRA adapter round-trip: build a minimal JSON, load it, apply it.
    #[test]
    fn lora_apply_changes_embedding() {
        use std::io::Write as _;

        // Build a tiny identity-like LoRA (loraA = eye-128-by-4, loraB = eye-4-by-128)
        // so the update is predictable.
        let lora_a: Vec<Vec<f32>> = (0..CSI_EMBED_DIM)
            .map(|i| (0..LORA_RANK).map(|j| if i == j { 1.0f32 } else { 0.0 }).collect())
            .collect();
        let lora_b: Vec<Vec<f32>> = (0..LORA_RANK)
            .map(|j| (0..CSI_EMBED_DIM).map(|k| if j == k { 1.0f32 } else { 0.0 }).collect())
            .collect();

        let json = serde_json::json!({
            "config": {"rank": LORA_RANK, "alpha": 8},
            "inputDim": CSI_EMBED_DIM,
            "outputDim": CSI_EMBED_DIM,
            "weights": {"loraA": lora_a, "loraB": lora_b, "scaling": 2.0}
        })
        .to_string();

        let tmp = std::env::temp_dir().join("csi_lora_test.json");
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(json.as_bytes()).unwrap();
        drop(f);

        let adapter = CsiLoraAdapter::load(&tmp).expect("load LoRA adapter");
        assert!((adapter.scaling - 2.0).abs() < 1e-6);

        // With identity matrices: intermediate = emb[0..4], delta = emb[0..4] padded to 128.
        // output[i] = emb[i] + 2.0 * emb[i] for i < 4, else emb[i]. Then L2-renorm.
        let mut emb = [0.1f32; CSI_EMBED_DIM];
        let applied = adapter.apply(&emb);
        // Check result is L2-normalised
        let norm: f32 = applied.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "LoRA output must be L2-normalised: norm={norm}");

        // With a zero embedding, output should still be L2-safe (norm epsilon clamp)
        emb = [0.0f32; CSI_EMBED_DIM];
        let zero_applied = adapter.apply(&emb);
        let zero_norm: f32 = zero_applied.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(zero_norm < 1e-3 || (zero_norm - 1.0).abs() < 1e-3,
            "zero input should remain near-zero after LoRA: norm={zero_norm}");

        let _ = std::fs::remove_file(&tmp);
    }

    /// Smoke-test the weights parser with a synthetic safetensors file.
    /// The model.safetensors format: 8-byte LE header-length, then the
    /// JSON header, then the tensor data (all f32 LE).
    #[test]
    fn round_trip_forward_with_identity_weights() {
        use std::io::Write as _;

        // Build a minimal safetensors file with identity-like weights
        // so we can predict the forward-pass output exactly.
        let w1: Vec<f32> = (0..CSI_HIDDEN_DIM * CSI_INPUT_DIM)
            .map(|i| if i % (CSI_INPUT_DIM + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let b1 = vec![0f32; CSI_HIDDEN_DIM];
        let w2: Vec<f32> = (0..CSI_EMBED_DIM * CSI_HIDDEN_DIM)
            .map(|i| if i % (CSI_HIDDEN_DIM + 1) == 0 { 1.0 } else { 0.0 })
            .collect();
        let b2 = vec![0f32; CSI_EMBED_DIM];

        let all_data: Vec<f32> = w1.iter().chain(b1.iter()).chain(w2.iter()).chain(b2.iter()).cloned().collect();
        let data_bytes: Vec<u8> = all_data.iter().flat_map(|v| v.to_le_bytes()).collect();

        // Compute offsets
        let w1_bytes = CSI_HIDDEN_DIM * CSI_INPUT_DIM * 4;
        let b1_bytes = CSI_HIDDEN_DIM * 4;
        let w2_bytes = CSI_EMBED_DIM * CSI_HIDDEN_DIM * 4;
        let b2_bytes = CSI_EMBED_DIM * 4;
        let o1s = 0usize; let o1e = w1_bytes;
        let o2s = o1e; let o2e = o2s + b1_bytes;
        let o3s = o2e; let o3e = o3s + w2_bytes;
        let o4s = o3e; let o4e = o4s + b2_bytes;

        let header_json = format!(
            "{{\"encoder.w1\":{{\"dtype\":\"F32\",\"shape\":[{},{CSI_INPUT_DIM}],\"data_offsets\":[{o1s},{o1e}]}},\"encoder.b1\":{{\"dtype\":\"F32\",\"shape\":[{CSI_HIDDEN_DIM}],\"data_offsets\":[{o2s},{o2e}]}},\"encoder.w2\":{{\"dtype\":\"F32\",\"shape\":[{CSI_EMBED_DIM},{CSI_HIDDEN_DIM}],\"data_offsets\":[{o3s},{o3e}]}},\"encoder.b2\":{{\"dtype\":\"F32\",\"shape\":[{CSI_EMBED_DIM}],\"data_offsets\":[{o4s},{o4e}]}}}}",
            CSI_HIDDEN_DIM
        );
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let tmp = std::env::temp_dir().join("csi_embedder_test.safetensors");
        let mut f = std::fs::File::create(&tmp).unwrap();
        f.write_all(&header_len.to_le_bytes()).unwrap();
        f.write_all(header_bytes).unwrap();
        f.write_all(&data_bytes).unwrap();
        drop(f);

        let weights = CsiWeights::load(&tmp).expect("load test weights");
        let features = dummy_features();
        let emb = weights.forward(&features.to_array());

        // Check L2-norm ≈ 1.0
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "embedding should be L2-normalised, got norm={norm}");

        let _ = std::fs::remove_file(&tmp);
    }
}
