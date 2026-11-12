//! WASM ONNX embedder on `tract-onnx` 0.23 (ADR-002 §3).
//!
//! `tract` compiles an optimised plan for a *fixed* input shape, so this keeps
//! the raw ONNX bytes and lazily builds one plan per distinct padded sequence
//! length (cached), running batch rows one at a time. That keeps timing honest
//! — a short batch is not padded to 256 — without symbolic-dimension plumbing.
//!
//! No `std::time`, `fs` or network here (ADR-005; repo `no-systemtime-in-wasm`
//! guard): the caller supplies model + tokenizer bytes and verifies hashes.

#![cfg(feature = "wasm")]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use tract_onnx::prelude::*;

use ruvector_typesafe_core::{Embedder, Result as CoreResult};

use crate::error::{EmbedError, Result};
use crate::manifest::ModelManifest;
use crate::pooling::{l2_normalize, pool};
use crate::tokenize::SharedTokenizer;

// `into_optimized().into_runnable()` yields an `Arc<TypedRunnableModel>`
// (= `SimplePlan<TypedFact, Box<dyn TypedOp>>`). `TypedRunnableModel` is the
// alias exported from the tract prelude.
type Plan = TypedRunnableModel;

pub struct TractEmbedder {
    model_bytes: Vec<u8>,
    num_inputs: usize,
    tokenizer: SharedTokenizer,
    manifest: ModelManifest,
    id: String,
    plans: Mutex<HashMap<usize, Arc<Plan>>>,
}

impl TractEmbedder {
    /// Build from already-verified model + tokenizer bytes. WASM has no fs, so
    /// hash verification (fail closed) happens here before any parse.
    pub fn from_bytes(
        model_bytes: &[u8],
        tokenizer_bytes: &[u8],
        manifest: &ModelManifest,
    ) -> Result<Self> {
        manifest.verify_model(model_bytes)?;
        manifest.verify_tokenizer(tokenizer_bytes)?;

        // Parse once to learn the input arity (2 or 3), and to fail early on a
        // model tract cannot even read.
        let parsed = tract_onnx::onnx()
            .model_for_read(&mut std::io::Cursor::new(model_bytes))
            .map_err(|e| EmbedError::Backend(format!("parse: {e}")))?;
        let num_inputs = parsed.inputs.len();

        let tokenizer = SharedTokenizer::from_bytes(tokenizer_bytes, manifest.max_tokens)?;

        Ok(Self {
            model_bytes: model_bytes.to_vec(),
            num_inputs,
            tokenizer,
            id: manifest.id(),
            manifest: manifest.clone(),
            plans: Mutex::new(HashMap::new()),
        })
    }

    fn plan_for(&self, seq_len: usize) -> Result<Arc<Plan>> {
        let mut plans = self
            .plans
            .lock()
            .map_err(|_| EmbedError::Backend("plan cache poisoned".into()))?;
        if let Some(p) = plans.get(&seq_len) {
            return Ok(p.clone());
        }
        let plan = build_plan(&self.model_bytes, self.num_inputs, seq_len)?;
        plans.insert(seq_len, plan.clone());
        Ok(plan)
    }

    fn embed_impl(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let batch = self.tokenizer.encode_batch(texts)?;
        let seq = batch.seq_len();
        let plan = self.plan_for(seq)?;

        let mut out = Vec::with_capacity(batch.batch_size());
        for r in 0..batch.batch_size() {
            let row = run_row(
                &plan,
                self.num_inputs,
                &batch.input_ids[r],
                &batch.attention_mask[r],
                &batch.token_type_ids[r],
                seq,
            )?;
            let mut v = if row.already_pooled {
                row.data
            } else {
                pool(
                    self.manifest.pooling,
                    &row.data,
                    &batch.attention_mask[r],
                    row.hidden,
                )
            };
            l2_normalize(&mut v);
            if v.len() != self.manifest.dims {
                return Err(EmbedError::Shape(format!(
                    "model dim {} != manifest dim {}",
                    v.len(),
                    self.manifest.dims
                )));
            }
            out.push(v);
        }
        Ok(out)
    }
}

impl Embedder for TractEmbedder {
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

/// Outcome of a staged tract load — the honest pass/fail the INT8 spike
/// (ADR-002 §3, ADR-194) needs. On failure it records the stage and the exact
/// error (0.21 died at `optimize` with `/Unsqueeze AddDims`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LoadOutcome {
    /// Parsed, shaped, optimised and made runnable — INT8 works on this tract.
    Runnable,
    FailedParse(String),
    FailedOptimize(String),
    FailedRunnable(String),
}

impl LoadOutcome {
    pub fn is_runnable(&self) -> bool {
        matches!(self, LoadOutcome::Runnable)
    }
}

/// Try to load `model_bytes` through parse -> optimise -> runnable at a fixed
/// `seq_len`, reporting how far it got. Input arity is read from the parsed
/// model. Never returns `Err`; the failure stage is part of the value.
pub fn diagnose_load(model_bytes: &[u8], seq_len: usize) -> LoadOutcome {
    let parsed = match tract_onnx::onnx().model_for_read(&mut std::io::Cursor::new(model_bytes)) {
        Ok(m) => m,
        Err(e) => return LoadOutcome::FailedParse(e.to_string()),
    };
    let num_inputs = parsed.inputs.len();
    let mut m = parsed;
    for i in 0..num_inputs {
        match m.with_input_fact(
            i,
            InferenceFact::dt_shape(i64::datum_type(), tvec![1, seq_len]),
        ) {
            Ok(next) => m = next,
            Err(e) => return LoadOutcome::FailedParse(format!("with_input_fact[{i}]: {e}")),
        }
    }
    let optimized = match m.into_optimized() {
        Ok(o) => o,
        Err(e) => return LoadOutcome::FailedOptimize(e.to_string()),
    };
    match optimized.into_runnable() {
        Ok(_) => LoadOutcome::Runnable,
        Err(e) => LoadOutcome::FailedRunnable(e.to_string()),
    }
}

fn build_plan(model_bytes: &[u8], num_inputs: usize, seq_len: usize) -> Result<Arc<Plan>> {
    let mut m = tract_onnx::onnx()
        .model_for_read(&mut std::io::Cursor::new(model_bytes))
        .map_err(|e| EmbedError::Backend(format!("parse: {e}")))?;
    for i in 0..num_inputs {
        m = m
            .with_input_fact(
                i,
                InferenceFact::dt_shape(i64::datum_type(), tvec![1, seq_len]),
            )
            .map_err(|e| EmbedError::Backend(format!("with_input_fact[{i}]: {e}")))?;
    }
    // `into_runnable` already returns an `Arc`.
    let plan = m
        .into_optimized()
        .map_err(|e| EmbedError::Backend(format!("optimize: {e}")))?
        .into_runnable()
        .map_err(|e| EmbedError::Backend(format!("runnable: {e}")))?;
    Ok(plan)
}

struct RowOutput {
    data: Vec<f32>,
    hidden: usize,
    already_pooled: bool,
}

fn run_row(
    plan: &Arc<Plan>,
    num_inputs: usize,
    ids: &[i64],
    mask: &[i64],
    types: &[i64],
    seq: usize,
) -> Result<RowOutput> {
    let ids_t: Tensor = tract_ndarray::Array2::from_shape_vec((1, seq), ids.to_vec())
        .map_err(|e| EmbedError::Backend(e.to_string()))?
        .into();
    let mask_t: Tensor = tract_ndarray::Array2::from_shape_vec((1, seq), mask.to_vec())
        .map_err(|e| EmbedError::Backend(e.to_string()))?
        .into();

    let inputs: TVec<TValue> = if num_inputs >= 3 {
        let types_t: Tensor = tract_ndarray::Array2::from_shape_vec((1, seq), types.to_vec())
            .map_err(|e| EmbedError::Backend(e.to_string()))?
            .into();
        tvec![ids_t.into(), mask_t.into(), types_t.into()]
    } else {
        tvec![ids_t.into(), mask_t.into()]
    };

    let outputs = plan
        .run(inputs)
        .map_err(|e| EmbedError::Backend(format!("run: {e}")))?;
    let out = outputs
        .first()
        .ok_or_else(|| EmbedError::Shape("no output tensor".into()))?;
    let view = out
        .to_plain_array_view::<f32>()
        .map_err(|e| EmbedError::Backend(e.to_string()))?;
    let shape = view.shape().to_vec();
    let flat: Vec<f32> = view.iter().copied().collect();

    // [1, seq, hidden] -> token embeddings; [1, hidden] -> already pooled.
    let (hidden, already_pooled) = match shape.len() {
        3 => (shape[2], false),
        2 => (shape[1], true),
        _ => return Err(EmbedError::Shape(format!("rank {} output", shape.len()))),
    };
    Ok(RowOutput {
        data: flat,
        hidden,
        already_pooled,
    })
}
