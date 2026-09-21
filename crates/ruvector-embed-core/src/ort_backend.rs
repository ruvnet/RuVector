//! Native ONNX embedder on `ort` 2.0.0-rc.13 (ADR-002 §2).
//!
//! Loads a hash-verified `.onnx` from a local directory (no network at
//! runtime), runs single-thread CPU inference, mean/CLS pools and L2
//! normalises. `Session::run` takes `&mut self`, so the session lives behind a
//! `Mutex` to keep `Embedder: Send + Sync`.

#![cfg(feature = "native")]

use std::path::Path;
use std::sync::Mutex;

use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;

use ruvector_typesafe_core::{Embedder, Result as CoreResult};

use crate::error::{EmbedError, Result};
use crate::manifest::ModelManifest;
use crate::pooling::{l2_normalize, pool};
use crate::tokenize::SharedTokenizer;

/// Output tensor names we recognise, in preference order.
const PREFERRED_OUTPUTS: &[&str] = &[
    "last_hidden_state",
    "sentence_embedding",
    "token_embeddings",
];

pub struct OrtEmbedder {
    session: Mutex<Session>,
    tokenizer: SharedTokenizer,
    manifest: ModelManifest,
    id: String,
    input_has_token_type: bool,
    output_name: String,
}

impl OrtEmbedder {
    /// Load from a directory holding the model + tokenizer named in `manifest`.
    /// Both files are SHA-256 verified against the manifest; a mismatch is a
    /// typed error and nothing is loaded (fail closed, ADR-005).
    pub fn from_manifest(dir: impl AsRef<Path>, manifest: &ModelManifest) -> Result<Self> {
        let dir = dir.as_ref();
        let model_bytes = read_file(&dir.join(&manifest.file))?;
        let tok_bytes = read_file(&dir.join(&manifest.tokenizer_file))?;
        // Verification happens in `from_bytes` — one pass, one gate.
        Self::from_bytes(&model_bytes, &tok_bytes, manifest)
    }

    /// Load from in-memory bytes. Both are SHA-256 verified against the manifest
    /// before anything is parsed; a mismatch is a typed error and nothing is
    /// loaded (fail closed, ADR-005). This is the single verification gate for
    /// the native path, mirroring `TractEmbedder::from_bytes`.
    pub fn from_bytes(
        model_bytes: &[u8],
        tok_bytes: &[u8],
        manifest: &ModelManifest,
    ) -> Result<Self> {
        manifest.verify_model(model_bytes)?;
        manifest.verify_tokenizer(tok_bytes)?;

        let session = Session::builder()
            .map_err(|e| EmbedError::Backend(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedError::Backend(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| EmbedError::Backend(e.to_string()))?
            .commit_from_memory(model_bytes)
            .map_err(|e| EmbedError::Backend(e.to_string()))?;

        let input_has_token_type = session
            .inputs()
            .iter()
            .any(|o| o.name() == "token_type_ids");

        let output_name = pick_output(&session);

        let tokenizer = SharedTokenizer::from_bytes(tok_bytes, manifest.max_tokens)?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            id: manifest.id(),
            manifest: manifest.clone(),
            input_has_token_type,
            output_name,
        })
    }

    fn embed_impl(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let batch = self.tokenizer.encode_batch(texts)?;
        let bsz = batch.batch_size();
        let seq = batch.seq_len();

        // Flatten [batch][seq] -> [batch*seq].
        let mut ids = Vec::with_capacity(bsz * seq);
        let mut mask = Vec::with_capacity(bsz * seq);
        let mut types = Vec::with_capacity(bsz * seq);
        for r in 0..bsz {
            ids.extend_from_slice(&batch.input_ids[r]);
            mask.extend_from_slice(&batch.attention_mask[r]);
            types.extend_from_slice(&batch.token_type_ids[r]);
        }

        let shape = vec![bsz, seq];
        let ids_t = Tensor::from_array((shape.clone(), ids.into_boxed_slice()))
            .map_err(|e| EmbedError::Backend(e.to_string()))?;
        let mask_t = Tensor::from_array((shape.clone(), mask.into_boxed_slice()))
            .map_err(|e| EmbedError::Backend(e.to_string()))?;

        let mut inputs: Vec<(&str, ort::value::DynValue)> = vec![
            ("input_ids", ids_t.into_dyn()),
            ("attention_mask", mask_t.into_dyn()),
        ];
        if self.input_has_token_type {
            let types_t = Tensor::from_array((shape.clone(), types.into_boxed_slice()))
                .map_err(|e| EmbedError::Backend(e.to_string()))?;
            inputs.push(("token_type_ids", types_t.into_dyn()));
        }

        let mut session = self
            .session
            .lock()
            .map_err(|_| EmbedError::Backend("session mutex poisoned".into()))?;
        let outputs = session
            .run(inputs)
            .map_err(|e| EmbedError::Backend(e.to_string()))?;

        let value = outputs
            .get(&self.output_name)
            .ok_or_else(|| EmbedError::Shape(format!("output '{}' absent", self.output_name)))?;
        let (out_shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedError::Backend(e.to_string()))?;
        let dims: Vec<usize> = out_shape.iter().map(|&d| d as usize).collect();

        pool_outputs(
            data,
            &dims,
            bsz,
            seq,
            &batch.attention_mask,
            self.manifest.pooling,
            self.manifest.dims,
        )
    }
}

impl Embedder for OrtEmbedder {
    fn embed(&self, texts: &[&str]) -> CoreResult<Vec<Vec<f32>>> {
        Ok(self.embed_impl(texts)?)
    }
    fn dims(&self) -> usize {
        self.manifest.dims
    }
    fn id(&self) -> &str {
        &self.id
    }
}

/// Convert a model output tensor into pooled, L2-normalised sentence vectors.
/// Shared shape logic used by [`OrtEmbedder`]; also unit-tested directly.
pub(crate) fn pool_outputs(
    data: &[f32],
    dims: &[usize],
    bsz: usize,
    seq: usize,
    attention_mask: &[Vec<i64>],
    strategy: crate::manifest::Pooling,
    expect_dims: usize,
) -> Result<Vec<Vec<f32>>> {
    let mut out = Vec::with_capacity(bsz);
    match dims.len() {
        3 => {
            // [batch, seq, hidden]
            let hidden = dims[2];
            for (r, mask_r) in attention_mask.iter().enumerate().take(bsz) {
                let start = r * seq * hidden;
                let row = &data[start..start + seq * hidden];
                let mut v = pool(strategy, row, mask_r, hidden);
                l2_normalize(&mut v);
                out.push(v);
            }
        }
        2 => {
            // [batch, hidden] — already pooled by the model.
            let hidden = dims[1];
            for r in 0..bsz {
                let start = r * hidden;
                let mut v = data[start..start + hidden].to_vec();
                l2_normalize(&mut v);
                out.push(v);
            }
        }
        _ => {
            return Err(EmbedError::Shape(format!("rank {} output", dims.len())));
        }
    }
    if let Some(first) = out.first() {
        if first.len() != expect_dims {
            return Err(EmbedError::Shape(format!(
                "model dim {} != manifest dim {}",
                first.len(),
                expect_dims
            )));
        }
    }
    Ok(out)
}

fn pick_output(session: &Session) -> String {
    let names: Vec<String> = session
        .outputs()
        .iter()
        .map(|o| o.name().to_string())
        .collect();
    for pref in PREFERRED_OUTPUTS {
        if let Some(n) = names.iter().find(|n| n.as_str() == *pref) {
            return n.clone();
        }
    }
    names.into_iter().next().unwrap_or_default()
}

fn read_file(path: &Path) -> Result<Vec<u8>> {
    if !path.exists() {
        return Err(EmbedError::NotFound(path.display().to_string()));
    }
    std::fs::read(path).map_err(|e| EmbedError::Io(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::Pooling;

    #[test]
    fn pool_outputs_rank3_mean() {
        // batch 1, seq 2, hidden 2; both tokens real.
        let data = vec![1.0, 1.0, 3.0, 3.0];
        let mask = vec![vec![1i64, 1]];
        let out = pool_outputs(&data, &[1, 2, 2], 1, 2, &mask, Pooling::Mean, 2).unwrap();
        assert_eq!(out.len(), 1);
        // mean [2,2] normalised -> [0.707,0.707]
        assert!((out[0][0] - 0.70710677).abs() < 1e-5);
    }

    #[test]
    fn pool_outputs_rank2_passthrough_normalises() {
        let data = vec![3.0, 4.0];
        let out = pool_outputs(&data, &[1, 2], 1, 1, &[vec![1i64]], Pooling::Cls, 2).unwrap();
        assert!((out[0][0] - 0.6).abs() < 1e-6);
        assert!((out[0][1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn pool_outputs_dim_mismatch_errors() {
        let data = vec![1.0, 0.0, 0.0];
        let err = pool_outputs(&data, &[1, 3], 1, 1, &[vec![1i64]], Pooling::Cls, 384).unwrap_err();
        assert!(matches!(err, EmbedError::Shape(_)));
    }

    #[test]
    fn from_bytes_fails_closed_on_hash_mismatch_without_a_model() {
        // The hash gate runs before any ONNX parse, so a mismatch is caught
        // with bogus bytes and no model file present.
        let m = crate::manifest::ModelManifest {
            name: "bge-small-en-v1.5".into(),
            file: "model.onnx".into(),
            sha256: crate::manifest::sha256_hex(b"the real weights"),
            dims: 384,
            license: "MIT".into(),
            source_url: String::new(),
            added: String::new(),
            review_by: String::new(),
            pooling: Pooling::Cls,
            tokenizer_file: "tokenizer.json".into(),
            tokenizer_sha256: None,
            max_tokens: 256,
        };
        // OrtEmbedder is not Debug, so match instead of unwrap_err().
        match OrtEmbedder::from_bytes(b"tampered", b"tok", &m) {
            Err(EmbedError::HashMismatch { .. }) => {}
            Err(other) => panic!("expected HashMismatch, got {other:?}"),
            Ok(_) => panic!("expected HashMismatch, model loaded"),
        }
    }
}
