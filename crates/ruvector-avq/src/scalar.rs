//! Per-dimension int8 scalar quantization. Baseline #1.
//!
//! For each dimension we learn `[lo, hi]` from training data and
//! map to int8 with affine reconstruction `x ≈ lo + (hi-lo)*(c+0.5)/256`.
//! Score is computed by decoding on the fly — fine for a baseline.

use crate::traits::{Encoder, Scorer};

#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    dim: usize,
    lo: Vec<f32>,
    span: Vec<f32>, // (hi - lo)
}

impl ScalarQuantizer {
    pub fn fit(train: &[f32], dim: usize) -> Self {
        assert!(train.len() % dim == 0);
        let n = train.len() / dim;
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        for i in 0..n {
            for d in 0..dim {
                let v = train[i * dim + d];
                if v < lo[d] {
                    lo[d] = v;
                }
                if v > hi[d] {
                    hi[d] = v;
                }
            }
        }
        let span: Vec<f32> = lo
            .iter()
            .zip(hi.iter())
            .map(|(&l, &h)| (h - l).max(1e-12))
            .collect();
        ScalarQuantizer { dim, lo, span }
    }

    fn decode_into(&self, code: &[u8], out: &mut [f32]) {
        for d in 0..self.dim {
            // map to mid-bin reconstruction
            let c = code[d] as f32;
            out[d] = self.lo[d] + self.span[d] * (c + 0.5) / 256.0;
        }
    }
}

impl Encoder for ScalarQuantizer {
    fn code_size(&self) -> usize {
        self.dim
    }
    fn dim(&self) -> usize {
        self.dim
    }
    fn encode(&self, xs: &[f32], codes: &mut [u8]) {
        let n = xs.len() / self.dim;
        for i in 0..n {
            let row = &xs[i * self.dim..(i + 1) * self.dim];
            let crow = &mut codes[i * self.dim..(i + 1) * self.dim];
            for d in 0..self.dim {
                let t = ((row[d] - self.lo[d]) / self.span[d]) * 256.0;
                let c = t.floor().clamp(0.0, 255.0);
                crow[d] = c as u8;
            }
        }
    }
}

impl Scorer for ScalarQuantizer {
    fn score_ip(&self, query: &[f32], codes: &[u8], out: &mut [f32]) {
        let mut buf = vec![0.0f32; self.dim];
        let n = codes.len() / self.dim;
        for i in 0..n {
            self.decode_into(&codes[i * self.dim..(i + 1) * self.dim], &mut buf);
            let mut s = 0.0f32;
            for d in 0..self.dim {
                s += query[d] * buf[d];
            }
            out[i] = s;
        }
    }
}
