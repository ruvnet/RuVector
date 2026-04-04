//! Pure Rust transformer encoder for deobfuscation inference.
//!
//! Architecture: N-layer transformer encoder (default 3 layers, 128 embed, 4 heads, 512 FFN).
//! Activation: GELU (matching PyTorch `TransformerEncoderLayer`).
//! Weight format: simple binary from `scripts/training/export-weights-bin.py`.

use std::io::{Cursor, Read as IoRead};
use std::path::Path;

use crate::error::DecompilerError;

const VOCAB_SIZE: usize = 256;
const PAD_TOKEN: u8 = 0;
const MAX_CONTEXT: usize = 64;
const MAX_NAME: usize = 32;
const TOTAL_SEQ: usize = MAX_CONTEXT + MAX_NAME;

struct TransformerLayer {
    in_proj_weight: Vec<f32>, // [3*D, D]
    in_proj_bias: Vec<f32>,   // [3*D]
    out_proj_weight: Vec<f32>, // [D, D]
    out_proj_bias: Vec<f32>,
    norm1_weight: Vec<f32>,
    norm1_bias: Vec<f32>,
    linear1_weight: Vec<f32>, // [FFN, D]
    linear1_bias: Vec<f32>,
    linear2_weight: Vec<f32>, // [D, FFN]
    linear2_bias: Vec<f32>,
    norm2_weight: Vec<f32>,
    norm2_bias: Vec<f32>,
}

/// Minimal transformer encoder for deobfuscation inference.
/// Pure Rust -- no Python, no ONNX Runtime, no external ML deps.
pub struct TransformerEncoder {
    embed_dim: usize,
    num_heads: usize,
    ffn_dim: usize,
    char_embed: Vec<f32>,        // [VOCAB_SIZE * D]
    pos_embed: Vec<f32>,         // [TOTAL_SEQ * D]
    layers: Vec<TransformerLayer>,
    final_norm_weight: Vec<f32>, // [D]
    final_norm_bias: Vec<f32>,
    output_weight: Vec<f32>,     // [VOCAB_SIZE * D]
    output_bias: Vec<f32>,       // [VOCAB_SIZE]
}

impl TransformerEncoder {
    /// Load from binary weights file (see `export-weights-bin.py`).
    pub fn from_weights_bin(path: &Path) -> Result<Self, DecompilerError> {
        let data = std::fs::read(path).map_err(|e| {
            DecompilerError::ModelError(format!("failed to read weights: {e}"))
        })?;
        Self::from_tensor_map(&parse_bin_tensors(&data)?)
    }

    /// Forward pass: context bytes + name bytes -> logits[MAX_NAME][VOCAB_SIZE].
    pub fn forward(&self, context: &[u8], name: &[u8]) -> Vec<Vec<f32>> {
        let d = self.embed_dim;
        let mut tokens = vec![PAD_TOKEN; TOTAL_SEQ];
        for (i, &b) in context.iter().take(MAX_CONTEXT).enumerate() { tokens[i] = b; }
        for (i, &b) in name.iter().take(MAX_NAME).enumerate() { tokens[MAX_CONTEXT + i] = b; }

        // Embedding: char_embed + pos_embed
        let mut x = vec![0.0f32; TOTAL_SEQ * d];
        for (pos, &tok) in tokens.iter().enumerate() {
            let (ce, pe, xo) = ((tok as usize) * d, pos * d, pos * d);
            for j in 0..d { x[xo + j] = self.char_embed[ce + j] + self.pos_embed[pe + j]; }
        }

        let pad_mask: Vec<bool> = tokens.iter().map(|&t| t == PAD_TOKEN).collect();
        for layer in &self.layers { x = self.layer_forward(layer, &x, &pad_mask); }

        // Final layer norm + output projection on last MAX_NAME positions
        let mut logits = Vec::with_capacity(MAX_NAME);
        for i in 0..MAX_NAME {
            let off = (MAX_CONTEXT + i) * d;
            let normed = layer_norm(&x[off..off + d], &self.final_norm_weight, &self.final_norm_bias);
            let mut out = vec![0.0f32; VOCAB_SIZE];
            for v in 0..VOCAB_SIZE {
                let wo = v * d;
                let mut s = self.output_bias[v];
                for j in 0..d { s += self.output_weight[wo + j] * normed[j]; }
                out[v] = s;
            }
            logits.push(out);
        }
        logits
    }

    /// Predict original name from minified name + context strings.
    pub fn predict(&self, minified: &str, context_strings: &[&str]) -> (String, f32) {
        let ctx: Vec<u8> = context_strings.join(" ").bytes().take(MAX_CONTEXT).collect();
        let nm: Vec<u8> = minified.bytes().take(MAX_NAME).collect();
        let logits = self.forward(&ctx, &nm);

        let mut predicted = String::new();
        let (mut total_conf, mut count) = (0.0f32, 0usize);
        for pos_logits in &logits {
            let probs = softmax(pos_logits);
            let (idx, &prob) = probs.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));
            if idx == 0 || idx == 2 { break; }  // PAD or EOS
            if idx == 1 { continue; }            // SOS
            let ch = idx as u8;
            if ch.is_ascii_alphanumeric() || ch == b'_' {
                predicted.push(ch as char);
                total_conf += prob;
                count += 1;
            }
        }
        (predicted, if count > 0 { total_conf / count as f32 } else { 0.0 })
    }

    /// Single encoder layer: self-attn + residual + norm + FFN + residual + norm (post-norm).
    fn layer_forward(&self, l: &TransformerLayer, x: &[f32], pad_mask: &[bool]) -> Vec<f32> {
        let (d, seq, ffn) = (self.embed_dim, TOTAL_SEQ, self.ffn_dim);

        let attn = mha(x, &l.in_proj_weight, &l.in_proj_bias,
                       &l.out_proj_weight, &l.out_proj_bias, seq, d, self.num_heads, pad_mask);

        // Residual + LayerNorm1
        let mut mid = vec![0.0f32; seq * d];
        for p in 0..seq {
            let o = p * d;
            let mut r = vec![0.0f32; d];
            for j in 0..d { r[j] = x[o + j] + attn[o + j]; }
            mid[o..o + d].copy_from_slice(&layer_norm(&r, &l.norm1_weight, &l.norm1_bias));
        }

        // FFN: Linear1 -> GELU -> Linear2, then residual + LayerNorm2
        let mut out = vec![0.0f32; seq * d];
        for p in 0..seq {
            let o = p * d;
            let h = &mid[o..o + d];
            let mut h1 = vec![0.0f32; ffn];
            for f in 0..ffn {
                let wo = f * d;
                let mut s = l.linear1_bias[f];
                for j in 0..d { s += l.linear1_weight[wo + j] * h[j]; }
                h1[f] = gelu(s);
            }
            let mut h2 = vec![0.0f32; d];
            for j in 0..d {
                let wo = j * ffn;
                let mut s = l.linear2_bias[j];
                for f in 0..ffn { s += l.linear2_weight[wo + f] * h1[f]; }
                h2[j] = s;
            }
            let mut r = vec![0.0f32; d];
            for j in 0..d { r[j] = mid[o + j] + h2[j]; }
            out[o..o + d].copy_from_slice(&layer_norm(&r, &l.norm2_weight, &l.norm2_bias));
        }
        out
    }

    fn from_tensor_map(
        t: &std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>,
    ) -> Result<Self, DecompilerError> {
        let get = |n: &str| -> Result<Vec<f32>, DecompilerError> {
            t.get(n).map(|(_, d)| d.clone())
                .ok_or_else(|| DecompilerError::ModelError(format!("missing tensor: {n}")))
        };
        let shape = |n: &str| -> Option<&Vec<usize>> { t.get(n).map(|(s, _)| s) };

        let embed_dim = shape("char_embed.weight")
            .and_then(|s| if s.len() == 2 { Some(s[1]) } else { None })
            .ok_or_else(|| DecompilerError::ModelError("char_embed.weight must be 2D".into()))?;

        // Count encoder layers
        let num_layers = t.keys()
            .filter_map(|k| k.strip_prefix("encoder.layers."))
            .filter_map(|r| r.split('.').next()?.parse::<usize>().ok())
            .max().map(|m| m + 1).unwrap_or(3);

        let ffn_dim = shape("encoder.layers.0.linear1.weight")
            .and_then(|s| if s.len() == 2 { Some(s[0]) } else { None })
            .unwrap_or(512);

        // Verify in_proj_weight exists; infer num_heads from embed_dim
        let _ = get("encoder.layers.0.self_attn.in_proj_weight")?;
        let num_heads = if embed_dim % 4 == 0 { 4 } else if embed_dim % 2 == 0 { 2 } else { 1 };

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let p = format!("encoder.layers.{i}");
            layers.push(TransformerLayer {
                in_proj_weight: get(&format!("{p}.self_attn.in_proj_weight"))?,
                in_proj_bias: get(&format!("{p}.self_attn.in_proj_bias"))?,
                out_proj_weight: get(&format!("{p}.self_attn.out_proj.weight"))?,
                out_proj_bias: get(&format!("{p}.self_attn.out_proj.bias"))?,
                norm1_weight: get(&format!("{p}.norm1.weight"))?,
                norm1_bias: get(&format!("{p}.norm1.bias"))?,
                linear1_weight: get(&format!("{p}.linear1.weight"))?,
                linear1_bias: get(&format!("{p}.linear1.bias"))?,
                linear2_weight: get(&format!("{p}.linear2.weight"))?,
                linear2_bias: get(&format!("{p}.linear2.bias"))?,
                norm2_weight: get(&format!("{p}.norm2.weight"))?,
                norm2_bias: get(&format!("{p}.norm2.bias"))?,
            });
        }

        Ok(Self {
            embed_dim, num_heads, ffn_dim,
            char_embed: get("char_embed.weight")?,
            pos_embed: get("pos_embed.weight")?,
            layers,
            final_norm_weight: get("layer_norm.weight")?,
            final_norm_bias: get("layer_norm.bias")?,
            output_weight: get("output_proj.weight")?,
            output_bias: get("output_proj.bias")?,
        })
    }
}

// -- Multi-head self-attention --

#[allow(clippy::too_many_arguments)]
fn mha(
    x: &[f32], ipw: &[f32], ipb: &[f32], ow: &[f32], ob: &[f32],
    seq: usize, d: usize, nh: usize, pad: &[bool],
) -> Vec<f32> {
    let hd = d / nh;
    let scale = 1.0 / (hd as f32).sqrt();

    // QKV projection
    let mut qkv = vec![0.0f32; seq * 3 * d];
    for p in 0..seq {
        let xo = p * d;
        for i in 0..(3 * d) {
            let wo = i * d;
            let mut s = ipb[i];
            for j in 0..d { s += ipw[wo + j] * x[xo + j]; }
            qkv[p * 3 * d + i] = s;
        }
    }

    let mut attn_out = vec![0.0f32; seq * d];
    for h in 0..nh {
        let ho = h * hd;
        let mut scores = vec![f32::NEG_INFINITY; seq * seq];
        for i in 0..seq {
            if pad[i] { continue; }
            for j in 0..seq {
                if pad[j] { continue; }
                let mut dot = 0.0f32;
                for k in 0..hd {
                    dot += qkv[i * 3 * d + ho + k] * qkv[j * 3 * d + d + ho + k];
                }
                scores[i * seq + j] = dot * scale;
            }
        }
        // Softmax per row
        for i in 0..seq {
            if pad[i] { continue; }
            let row = &mut scores[i * seq..(i + 1) * seq];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if mx == f32::NEG_INFINITY { continue; }
            let mut sum = 0.0f32;
            for v in row.iter_mut() {
                if *v == f32::NEG_INFINITY { *v = 0.0; }
                else { *v = (*v - mx).exp(); sum += *v; }
            }
            if sum > 0.0 { for v in row.iter_mut() { *v /= sum; } }
        }
        // Weighted sum of V
        for i in 0..seq {
            if pad[i] { continue; }
            for k in 0..hd {
                let mut s = 0.0f32;
                for j in 0..seq { s += scores[i * seq + j] * qkv[j * 3 * d + 2 * d + ho + k]; }
                attn_out[i * d + ho + k] = s;
            }
        }
    }

    // Output projection
    let mut result = vec![0.0f32; seq * d];
    for p in 0..seq {
        let ao = p * d;
        for j in 0..d {
            let wo = j * d;
            let mut s = ob[j];
            for k in 0..d { s += ow[wo + k] * attn_out[ao + k]; }
            result[p * d + j] = s;
        }
    }
    result
}

// -- Primitive ops --

fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv = 1.0 / (var + 1e-5f32).sqrt();
    x.iter().enumerate().map(|(i, &v)| (v - mean) * inv * w[i] + b[i]).collect()
}

fn gelu(x: f32) -> f32 {
    x * 0.5 * (1.0 + (0.7978845608028654f32 * (x + 0.044715 * x * x * x)).tanh())
}

fn softmax(x: &[f32]) -> Vec<f32> {
    let mx = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = x.iter().map(|&v| (v - mx).exp()).collect();
    let s: f32 = exps.iter().sum();
    if s > 0.0 { exps.iter().map(|v| v / s).collect() } else { vec![0.0; x.len()] }
}

// -- Binary weight parser --

fn parse_bin_tensors(
    data: &[u8],
) -> Result<std::collections::HashMap<String, (Vec<usize>, Vec<f32>)>, DecompilerError> {
    let mut tensors = std::collections::HashMap::new();
    let mut cur = Cursor::new(data);
    let mut buf4 = [0u8; 4];

    while (cur.position() as usize) < data.len() {
        if cur.read_exact(&mut buf4).is_err() { break; }
        let name_len = u32::from_le_bytes(buf4) as usize;
        if name_len == 0 || name_len > 1024 { break; }

        let mut name_buf = vec![0u8; name_len];
        cur.read_exact(&mut name_buf).map_err(|e|
            DecompilerError::ModelError(format!("truncated name: {e}")))?;
        let name = String::from_utf8(name_buf).map_err(|e|
            DecompilerError::ModelError(format!("invalid name: {e}")))?;

        cur.read_exact(&mut buf4).map_err(|e|
            DecompilerError::ModelError(format!("truncated ndim for {name}: {e}")))?;
        let ndim = u32::from_le_bytes(buf4) as usize;

        let mut shape = Vec::with_capacity(ndim);
        let mut numel = 1usize;
        for _ in 0..ndim {
            cur.read_exact(&mut buf4).map_err(|e|
                DecompilerError::ModelError(format!("truncated shape for {name}: {e}")))?;
            let dim = u32::from_le_bytes(buf4) as usize;
            numel *= dim;
            shape.push(dim);
        }

        let byte_len = numel * 4;
        let pos = cur.position() as usize;
        if pos + byte_len > data.len() {
            return Err(DecompilerError::ModelError(format!(
                "truncated data for {name}: need {byte_len} bytes")));
        }
        let mut float_data = vec![0.0f32; numel];
        for (i, chunk) in data[pos..pos + byte_len].chunks_exact(4).enumerate() {
            float_data[i] = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        cur.set_position((pos + byte_len) as u64);
        tensors.insert(name, (shape, float_data));
    }

    if tensors.is_empty() {
        return Err(DecompilerError::ModelError("no tensors in weights file".into()));
    }
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_layer_norm() {
        let out = layer_norm(&[1.0, 2.0, 3.0, 4.0], &[1.0; 4], &[0.0; 4]);
        assert!((out.iter().sum::<f32>()).abs() < 1e-4);
        let std = (1.25f32 + 1e-5).sqrt();
        assert!((out[0] - (-1.5 / std)).abs() < 1e-4);
    }

    #[test]
    fn test_gelu() {
        assert!(gelu(0.0).abs() < 1e-6);
        assert!((gelu(3.0) - 3.0).abs() < 0.01);
        assert!(gelu(-3.0).abs() < 0.01);
    }

    #[test]
    fn test_softmax() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn test_softmax_stability() {
        let p = softmax(&[1000.0, 1001.0, 1002.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bin_roundtrip() {
        let mut data = Vec::new();
        let name = b"test.weight";
        data.extend_from_slice(&(name.len() as u32).to_le_bytes());
        data.extend_from_slice(name);
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&3u32.to_le_bytes());
        for v in [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        let t = parse_bin_tensors(&data).unwrap();
        let (s, v) = &t["test.weight"];
        assert_eq!(s, &[2, 3]);
        assert_eq!(v, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }
}
