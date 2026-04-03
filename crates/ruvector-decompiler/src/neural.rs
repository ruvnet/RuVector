//! Neural name inference via ONNX Runtime (behind `neural` feature flag).
//!
//! Loads a trained deobfuscation model in ONNX, GGUF, or RVF format and
//! predicts human-readable names for minified JS identifiers.

use std::path::{Path, PathBuf};

use crate::inferrer::{infer_declaration_name, InferenceContext};
use crate::training::TrainingCorpus;
use crate::types::{InferredName, Module};

/// Neural name inference using a trained deobfuscation model.
///
/// When an ONNX model is loaded, inference runs through ONNX Runtime.
/// GGUF and RVF formats are validated but inference is a stub pending
/// RuvLLM integration.
pub struct NeuralInferrer {
    model_path: PathBuf,
    /// Uses `RefCell` so `predict_name` can keep `&self` for the caller.
    session: Option<std::cell::RefCell<ort::session::Session>>,
    active: bool,
}

impl NeuralInferrer {
    const MAX_CONTEXT_LEN: usize = 256;
    const MAX_NAME_LEN: usize = 32;
    const MAX_OUTPUT_LEN: usize = 64;

    /// Load a deobfuscation model from `path`.
    ///
    /// Supports `.onnx` (ONNX Runtime), GGUF (`0x46475547`), and
    /// RVF (`RVF\x01`) formats.
    pub fn load(path: &Path) -> Result<Self, crate::error::DecompilerError> {
        if !path.exists() {
            return Err(crate::error::DecompilerError::ModelError(format!(
                "model file not found: {}",
                path.display()
            )));
        }

        let is_onnx = path
            .extension()
            .map_or(false, |ext| ext.eq_ignore_ascii_case("onnx"));

        if is_onnx {
            return Self::load_onnx(path);
        }

        Self::load_legacy(path)
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
            session: Some(std::cell::RefCell::new(session)),
            active: true,
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
                "unrecognized model format (expected ONNX, GGUF, or RVF)".to_string(),
            ));
        }

        Ok(Self {
            model_path: path.to_path_buf(),
            session: None,
            active: true,
        })
    }

    /// Predict the original name for a minified identifier.
    pub fn predict_name(
        &self,
        minified: &str,
        context: &InferenceContext,
    ) -> Option<InferredName> {
        if !self.active {
            return None;
        }

        let cell = self.session.as_ref()?;
        let mut session = cell.borrow_mut();
        Self::run_onnx_inference(&mut session, minified, context)
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

    /// Whether the neural model is loaded and ready.
    pub fn is_active(&self) -> bool {
        self.active
    }

    /// Path to the loaded model file.
    pub fn model_path(&self) -> &Path {
        &self.model_path
    }

    /// Whether the inferrer has a live ONNX session for real inference.
    pub fn has_onnx_session(&self) -> bool {
        self.session.is_some()
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
