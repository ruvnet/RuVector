//! Neural name inference via ONNX Runtime or pure-Rust transformer.
//!
//! When a `.bin` weights file is provided, uses the pure-Rust transformer
//! encoder from `transformer.rs` (no external dependencies). Falls back to
//! ONNX Runtime for `.onnx` files.

use std::path::{Path, PathBuf};

use crate::inferrer::{infer_declaration_name, InferenceContext};
use crate::training::TrainingCorpus;
use crate::transformer::TransformerEncoder;
use crate::types::{InferredName, Module};

/// Backend for neural inference.
enum Backend {
    /// Pure Rust transformer loaded from `.bin` weights.
    Transformer(TransformerEncoder),
    /// ONNX Runtime session (requires `neural` feature).
    Onnx(std::cell::RefCell<ort::session::Session>),
    /// Recognized format but no inference available (GGUF/RVF stub).
    Stub,
}

/// Neural name inference using a trained deobfuscation model.
///
/// Supports three backends:
/// - `.bin` — pure-Rust transformer (preferred, always available)
/// - `.onnx` — ONNX Runtime (requires `neural` feature)
/// - GGUF/RVF — validated but inference is stubbed pending full support
pub struct NeuralInferrer {
    model_path: PathBuf,
    backend: Backend,
}

impl NeuralInferrer {
    const MAX_CONTEXT_LEN: usize = 256;
    const MAX_NAME_LEN: usize = 32;
    const MAX_OUTPUT_LEN: usize = 64;

    /// Load a deobfuscation model from `path`.
    ///
    /// Format detection:
    /// - `.bin` → pure-Rust transformer (always available)
    /// - `.onnx` → ONNX Runtime
    /// - GGUF (`0x46475547`) or RVF (`RVF\x01`) → stub
    pub fn load(path: &Path) -> Result<Self, crate::error::DecompilerError> {
        if !path.exists() {
            return Err(crate::error::DecompilerError::ModelError(format!(
                "model file not found: {}",
                path.display()
            )));
        }

        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");

        match ext.to_ascii_lowercase().as_str() {
            "bin" => Self::load_transformer(path),
            "onnx" => Self::load_onnx(path),
            _ => Self::load_legacy(path),
        }
    }

    fn load_transformer(path: &Path) -> Result<Self, crate::error::DecompilerError> {
        let encoder = TransformerEncoder::from_weights_bin(path)?;
        Ok(Self {
            model_path: path.to_path_buf(),
            backend: Backend::Transformer(encoder),
        })
    }

    fn load_onnx(path: &Path) -> Result<Self, crate::error::DecompilerError> {
        let session = ort::session::Session::builder()
            .and_then(|b| b.commit_from_file(path))
            .map_err(|e| {
                crate::error::DecompilerError::ModelError(format!(
                    "failed to load ONNX model: {e}"
                ))
            })?;

        Ok(Self {
            model_path: path.to_path_buf(),
            backend: Backend::Onnx(std::cell::RefCell::new(session)),
        })
    }

    fn load_legacy(path: &Path) -> Result<Self, crate::error::DecompilerError> {
        let data = std::fs::read(path).map_err(|e| {
            crate::error::DecompilerError::ModelError(format!(
                "failed to read model file: {e}"
            ))
        })?;

        if data.len() < 4 {
            return Err(crate::error::DecompilerError::ModelError(
                "model file too small".to_string(),
            ));
        }

        let magic = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let is_gguf = magic == 0x46475547;
        let is_rvf = &data[..4] == b"RVF\x01";

        if !is_gguf && !is_rvf {
            return Err(crate::error::DecompilerError::ModelError(
                "unrecognized model format (expected .bin, .onnx, GGUF, or RVF)".to_string(),
            ));
        }

        Ok(Self {
            model_path: path.to_path_buf(),
            backend: Backend::Stub,
        })
    }

    /// Predict the original name for a minified identifier.
    pub fn predict_name(
        &self,
        minified: &str,
        context: &InferenceContext,
    ) -> Option<InferredName> {
        match &self.backend {
            Backend::Transformer(encoder) => {
                let ctx_strings: Vec<&str> = context
                    .string_literals
                    .iter()
                    .chain(context.property_accesses.iter())
                    .map(|s| s.as_str())
                    .collect();
                let (predicted, confidence) = encoder.predict(minified, &ctx_strings);
                if predicted.is_empty() || (confidence as f64) < 0.3 {
                    return None;
                }
                Some(InferredName {
                    original: minified.to_string(),
                    inferred: predicted,
                    confidence: confidence as f64,
                    evidence: vec![format!(
                        "pure-Rust transformer prediction (confidence: {confidence:.3})"
                    )],
                })
            }
            Backend::Onnx(cell) => {
                let mut session = cell.borrow_mut();
                Self::run_onnx_inference(&mut session, minified, context)
            }
            Backend::Stub => None,
        }
    }

    fn run_onnx_inference(
        session: &mut ort::session::Session,
        minified: &str,
        context: &InferenceContext,
    ) -> Option<InferredName> {
        use ort::value::Tensor;

        let name_bytes: Vec<f32> = minified
            .bytes()
            .take(Self::MAX_NAME_LEN)
            .map(|b| b as f32)
            .chain(std::iter::repeat(0.0f32))
            .take(Self::MAX_NAME_LEN)
            .collect();

        let ctx_joined = [
            context.kind.as_str(),
            " ",
            &context.string_literals.join(" "),
            " ",
            &context.property_accesses.join(" "),
        ]
        .concat();
        let ctx_bytes: Vec<f32> = ctx_joined
            .bytes()
            .take(Self::MAX_CONTEXT_LEN)
            .map(|b| b as f32)
            .chain(std::iter::repeat(0.0f32))
            .take(Self::MAX_CONTEXT_LEN)
            .collect();

        let name_tensor = Tensor::from_array((
            vec![1i64, Self::MAX_NAME_LEN as i64],
            name_bytes,
        ))
        .ok()?;
        let ctx_tensor = Tensor::from_array((
            vec![1i64, Self::MAX_CONTEXT_LEN as i64],
            ctx_bytes,
        ))
        .ok()?;

        let outputs = session
            .run(ort::inputs![name_tensor, ctx_tensor])
            .ok()?;

        if outputs.len() < 2 {
            return None;
        }

        let (_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .ok()?;
        let (_cshape, conf_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .ok()?;

        let confidence = *conf_data.first()? as f64;
        if confidence < 0.5 {
            return None;
        }

        let decoded: String = out_data
            .iter()
            .take(Self::MAX_OUTPUT_LEN)
            .map(|&v| v.round() as u8)
            .take_while(|&b| b > 0)
            .filter(|b| b.is_ascii_alphanumeric() || *b == b'_')
            .map(|b| b as char)
            .collect();

        if decoded.is_empty() {
            return None;
        }

        Some(InferredName {
            original: minified.to_string(),
            inferred: decoded,
            confidence,
            evidence: vec![format!(
                "neural model prediction (ONNX, confidence: {confidence:.3})"
            )],
        })
    }

    /// Whether the neural model is loaded and ready for inference.
    pub fn is_active(&self) -> bool {
        !matches!(self.backend, Backend::Stub)
    }

    /// Path to the loaded model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Whether the inferrer uses the pure-Rust transformer backend.
    pub fn has_transformer(&self) -> bool {
        matches!(self.backend, Backend::Transformer(_))
    }

    /// Whether the inferrer has a live ONNX session.
    pub fn has_onnx_session(&self) -> bool {
        matches!(self.backend, Backend::Onnx(_))
    }
}

/// Infer names with optional neural model support.
///
/// Neural inference is attempted first; results with confidence > 0.8
/// are accepted directly. Otherwise falls through to corpus + heuristics.
pub fn infer_names_neural(
    modules: &[Module],
    model_path: Option<&Path>,
) -> Vec<InferredName> {
    let corpus = TrainingCorpus::builtin();
    let neural = model_path.and_then(|p| NeuralInferrer::load(p).ok());

    let mut inferred = Vec::new();

    for module in modules {
        for decl in &module.declarations {
            if let Some(ref model) = neural {
                let ctx = InferenceContext::from_declaration(decl);
                if let Some(name) = model.predict_name(&decl.name, &ctx) {
                    if name.confidence > 0.8 {
                        inferred.push(name);
                        continue;
                    }
                }
            }

            if let Some(inf) = infer_declaration_name(decl, &corpus) {
                inferred.push(inf);
            }
        }
    }

    inferred
}
