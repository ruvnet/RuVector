//! ONNX initializer transplant (ADR-008 §5, plan Step 2 "Transplant mechanics").
//!
//! 1. Decode the template `ModelProto` (tract_onnx::pb) for analysis only.
//! 2. Map every float initializer to exactly one ORIGINAL base safetensors
//!    tensor by value (directly, or transposed for the anonymous MatMul
//!    weights) within 1e-6. Unmatched or ambiguous → abort.
//! 3. Cross-check: named initializers must carry the tensor's own name;
//!    anonymous ones (`onnx::MatMul_*`) must be consumed by a node whose name
//!    contains the tensor's module path (`/encoder/layer.0/attention/self/query/`).
//! 4. Require a bijection onto the encoder tensors, then overwrite the payload
//!    bytes of each initializer in place (see `pbwalk`) and verify by decoding.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use anyhow::{bail, Context, Result};
use candle_core::Device;
use prost::Message;
use regex::Regex;
use serde::Serialize;
use tract_onnx::pb::{ModelProto, TensorProto};

use super::pbwalk::float_payload_spans;
use crate::model::BertConfig;
use crate::norm::sha256_hex;

pub const TOLERANCE: f32 = 1e-6;
const FLOAT: i32 = 1;

/// name → (shape, values), restricted to the encoder parameters of `cfg`.
pub type Named = BTreeMap<String, (Vec<usize>, Vec<f32>)>;

/// A non-float initializer left untouched: (name, ONNX data_type).
pub type Skipped = (String, i32);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Mapping {
    pub initializer: String,
    pub tensor: String,
    pub transposed: bool,
    /// `name` (initializer carries the tensor name) or `node:<consumer name>`.
    pub cross_check: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TransplantReport {
    pub template_sha256: String,
    pub output_sha256: String,
    pub float_initializers: usize,
    pub named: usize,
    pub anonymous_transposed: usize,
    pub skipped_non_float: Vec<Skipped>,
    pub mappings: Vec<Mapping>,
    /// max |new − base| over all transplanted tensors (0 for --identity).
    pub max_abs_delta_vs_base: f32,
    pub byte_identical_to_template: bool,
}

pub fn load_named(path: &Path, cfg: BertConfig) -> Result<Named> {
    crate::pins::check_extension(path)?;
    let t = candle_core::safetensors::load(path, &Device::Cpu)
        .with_context(|| format!("load {}", path.display()))?;
    let mut out = Named::new();
    for (name, shape) in cfg.param_shapes() {
        let x = t
            .get(&name)
            .with_context(|| format!("{} lacks {name}", path.display()))?;
        if x.dims() != shape.as_slice() {
            bail!("{name}: shape {:?} != {shape:?}", x.dims());
        }
        let v = x
            .to_dtype(candle_core::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        out.insert(name, (shape, v));
    }
    Ok(out)
}

fn init_values(t: &TensorProto) -> Result<Vec<f32>> {
    if !t.raw_data.is_empty() {
        if !t.raw_data.len().is_multiple_of(4) {
            bail!("{}: raw_data not a multiple of 4", t.name);
        }
        Ok(t.raw_data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| f32::from_le_bytes(*c))
            .collect())
    } else {
        Ok(t.float_data.clone())
    }
}

fn close(a: &[f32], b: &[f32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() <= TOLERANCE)
}

/// `a` (shape [c, r]) equals base `b` (shape [r, c]) transposed.
fn close_transposed(a: &[f32], b: &[f32], r: usize, c: usize) -> bool {
    if a.len() != r * c || b.len() != r * c {
        return false;
    }
    for i in 0..c {
        for j in 0..r {
            if (a[i * r + j] - b[j * c + i]).abs() > TOLERANCE {
                return false;
            }
        }
    }
    true
}

pub fn transpose(v: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut out = vec![0f32; r * c];
    for j in 0..r {
        for i in 0..c {
            out[i * r + j] = v[j * c + i];
        }
    }
    out
}

/// `encoder.layer.0.attention.self.query.weight` → `/encoder/layer.0/attention/self/query/`.
pub fn module_path(tensor: &str) -> String {
    let stem = tensor.rsplit_once('.').map(|(s, _)| s).unwrap_or(tensor);
    let slashed = stem.replace('.', "/");
    let re = Regex::new(r"layer/(\d+)").expect("static regex");
    format!("/{}/", re.replace_all(&slashed, "layer.$1"))
}

/// Steps 2–4 (analysis): the verified initializer → tensor mapping.
pub fn map_initializers(model: &ModelProto, base: &Named) -> Result<(Vec<Mapping>, Vec<Skipped>)> {
    let graph = model.graph.as_ref().context("ModelProto has no graph")?;
    let mut consumers: HashMap<&str, Vec<&str>> = HashMap::new();
    for n in &graph.node {
        for i in &n.input {
            consumers
                .entry(i.as_str())
                .or_default()
                .push(n.name.as_str());
        }
    }
    let mut maps = Vec::new();
    let mut skipped = Vec::new();
    let mut errors = Vec::new();
    for init in &graph.initializer {
        if init.data_type != FLOAT {
            skipped.push((init.name.clone(), init.data_type));
            continue;
        }
        let vals = init_values(init)?;
        let dims: Vec<usize> = init.dims.iter().map(|&d| d as usize).collect();
        let mut hits = Vec::new();
        for (tname, (shape, bv)) in base {
            if bv.len() != vals.len() {
                continue;
            }
            if dims == *shape && close(&vals, bv) {
                hits.push((tname.clone(), false));
            }
            if dims.len() == 2
                && shape.len() == 2
                && dims == [shape[1], shape[0]]
                && close_transposed(&vals, bv, shape[0], shape[1])
            {
                hits.push((tname.clone(), true));
            }
        }
        if hits.len() != 1 {
            errors.push(format!(
                "{} {:?}: {} base matches {:?}",
                init.name,
                dims,
                hits.len(),
                hits
            ));
            continue;
        }
        let (tensor, transposed) = hits.remove(0);
        let cross_check = if !init.name.starts_with("onnx::") {
            if init.name != tensor {
                errors.push(format!(
                    "{} value-matches {tensor} but is named differently",
                    init.name
                ));
                continue;
            }
            "name".to_string()
        } else {
            let want = module_path(&tensor);
            let users = consumers
                .get(init.name.as_str())
                .cloned()
                .unwrap_or_default();
            match users.iter().find(|u| u.contains(&want)) {
                Some(u) => format!("node:{u}"),
                None => {
                    errors.push(format!(
                        "{} → {tensor}: consumers {users:?} lack {want}",
                        init.name
                    ));
                    continue;
                }
            }
        };
        maps.push(Mapping {
            initializer: init.name.clone(),
            tensor,
            transposed,
            cross_check,
        });
    }
    let mut hit_count: BTreeMap<&str, usize> = base.keys().map(|k| (k.as_str(), 0)).collect();
    for m in &maps {
        *hit_count
            .get_mut(m.tensor.as_str())
            .expect("tensor from base") += 1;
    }
    for (t, n) in hit_count {
        if n != 1 {
            errors.push(format!("base tensor {t} mapped {n} times (need exactly 1)"));
        }
    }
    if !errors.is_empty() {
        bail!(
            "transplant mapping failed ({} problems):\n  {}",
            errors.len(),
            errors.join("\n  ")
        );
    }
    Ok((maps, skipped))
}

/// Full transplant: returns the patched ONNX bytes and the report.
pub fn transplant(
    template: &[u8],
    base: &Named,
    new: &Named,
) -> Result<(Vec<u8>, TransplantReport)> {
    let model = ModelProto::decode(template).context("decode template ONNX")?;
    let (maps, skipped) = map_initializers(&model, base)?;
    let spans = float_payload_spans(template)?;
    let mut out = template.to_vec();
    let mut max_delta = 0f32;
    for m in &maps {
        let (shape, v) = new
            .get(&m.tensor)
            .with_context(|| format!("weights lack {}", m.tensor))?;
        let (_, b) = &base[&m.tensor];
        if v.len() != b.len() {
            bail!("{}: fine-tuned shape {shape:?} differs from base", m.tensor);
        }
        for (x, y) in v.iter().zip(b) {
            max_delta = max_delta.max((x - y).abs());
        }
        let vals = if m.transposed {
            transpose(v, shape[0], shape[1])
        } else {
            v.clone()
        };
        let span = spans
            .get(&m.initializer)
            .with_context(|| format!("no payload span for {}", m.initializer))?;
        if span.len() != vals.len() * 4 {
            bail!(
                "{}: payload {} bytes, need {}",
                m.initializer,
                span.len(),
                vals.len() * 4
            );
        }
        for (dst, x) in out[span.clone()]
            .as_chunks_mut::<4>()
            .0
            .iter_mut()
            .zip(&vals)
        {
            dst.copy_from_slice(&x.to_le_bytes());
        }
    }
    verify(&out, &maps, new)?;
    let named = maps.iter().filter(|m| m.cross_check == "name").count();
    let rep = TransplantReport {
        template_sha256: sha256_hex(template),
        output_sha256: sha256_hex(&out),
        float_initializers: maps.len(),
        named,
        anonymous_transposed: maps.iter().filter(|m| m.transposed).count(),
        skipped_non_float: skipped,
        mappings: maps,
        max_abs_delta_vs_base: max_delta,
        byte_identical_to_template: out == template,
    };
    Ok((out, rep))
}

/// Decode the output and require every mapped initializer to hold exactly the
/// intended (possibly transposed) fine-tuned values.
fn verify(out: &[u8], maps: &[Mapping], new: &Named) -> Result<()> {
    let model = ModelProto::decode(out).context("decode transplanted ONNX")?;
    let graph = model.graph.as_ref().context("no graph")?;
    let by_name: HashMap<&str, &TensorProto> = graph
        .initializer
        .iter()
        .map(|t| (t.name.as_str(), t))
        .collect();
    for m in maps {
        let (shape, v) = &new[&m.tensor];
        let want = if m.transposed {
            transpose(v, shape[0], shape[1])
        } else {
            v.clone()
        };
        let got = init_values(by_name[m.initializer.as_str()])?;
        if got != want {
            bail!("post-write verification failed for {}", m.initializer);
        }
    }
    Ok(())
}
