//! Versioned int8 scalar quantizer and asymmetric (f32 query × u8 code)
//! distance.
//!
//! A code dequantizes as `x̂ᵢ = offsetᵢ + scaleᵢ · cᵢ`. Two policies:
//! [`QuantKind::FixedRange`] (one range for every dimension, e.g. cosine's
//! `[-1, 1]` from ADR §6.1, needs no training) and [`QuantKind::PerDim`]
//! (per-dimension min/max trained from a sample such as the store's 1k
//! reservoir). Cosine vectors are L2-normalised before encoding and
//! training. The `epoch` versions the parameters: codes are only comparable
//! with the parameters that produced them (`vectors.q8_epoch`).

use crate::bytes::{Reader, Sink};
use crate::error::{DecodeError, IndexError};
use crate::metric::{norm, Metric};

/// Largest supported dimension (ADR §9 `dim: 1..1536`). Bounding it here
/// keeps every `slots × dim` product small enough for a 32-bit `usize`.
pub const MAX_DIM: usize = 1536;

/// Quantizer policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantKind {
    /// One `[lo, hi]` range shared by every dimension.
    FixedRange,
    /// Per-dimension `[min, max]` trained from samples.
    PerDim,
}

/// Quantizer parameters (immutable once built; a change is a new epoch).
#[derive(Debug, Clone, PartialEq)]
pub struct QuantParams {
    metric: Metric,
    kind: QuantKind,
    epoch: u64,
    offset: Vec<f32>,
    scale: Vec<f32>,
}

fn check_dim(dim: usize) -> Result<(), IndexError> {
    if dim == 0 || dim > MAX_DIM {
        return Err(IndexError::InvalidParams("dim must be in 1..=MAX_DIM"));
    }
    Ok(())
}

/// Step for `[lo, hi]`, or `None` when the range is degenerate.
fn step(lo: f32, hi: f32) -> Option<f32> {
    let s = (hi - lo) / 255.0;
    (s.is_finite() && s > 0.0).then_some(s)
}

/// Per-dimension `(offset, scale)`. A dimension that was constant in the
/// sample gets a data-derived step (the widest trained step, else
/// `max(|lo|, 1e-3)·2/255`) centred on the constant, so later variation in
/// it still encodes instead of rounding to a single code.
fn fit(lo: &[f32], hi: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let widest = lo
        .iter()
        .zip(hi)
        .filter_map(|(l, h)| step(*l, *h))
        .fold(None, |m: Option<f32>, s| Some(m.map_or(s, |m| m.max(s))));
    lo.iter()
        .zip(hi)
        .map(|(&l, &h)| match step(l, h) {
            Some(s) => (l, s),
            None => {
                let s = widest.unwrap_or(l.abs().max(1e-3) * 2.0 / 255.0);
                (l - 128.0 * s, s)
            }
        })
        .unzip()
}

impl QuantParams {
    /// Fixed range `[lo, hi]` for every dimension.
    pub fn fixed_range(
        metric: Metric,
        dim: usize,
        lo: f32,
        hi: f32,
        epoch: u64,
    ) -> Result<Self, IndexError> {
        check_dim(dim)?;
        let s = match (lo.is_finite() && hi.is_finite()).then(|| step(lo, hi)) {
            Some(Some(s)) => s,
            _ => {
                return Err(IndexError::InvalidParams(
                    "fixed range needs finite lo < hi",
                ))
            }
        };
        Ok(Self {
            metric,
            kind: QuantKind::FixedRange,
            epoch,
            offset: vec![lo; dim],
            scale: vec![s; dim],
        })
    }

    /// ADR §6.1 cosine policy: fixed `[-1, 1]` (no training, never re-encodes).
    pub fn cosine_fixed(dim: usize, epoch: u64) -> Result<Self, IndexError> {
        Self::fixed_range(Metric::Cosine, dim, -1.0, 1.0, epoch)
    }

    /// Per-dimension min/max over `samples` (row-major, `len % dim == 0`,
    /// at least one row, all finite). Cosine rows are normalised first;
    /// zero-norm cosine rows are skipped.
    pub fn train(
        metric: Metric,
        dim: usize,
        samples: &[f32],
        epoch: u64,
    ) -> Result<Self, IndexError> {
        check_dim(dim)?;
        if samples.is_empty() || samples.len() % dim != 0 {
            return Err(IndexError::InvalidParams(
                "samples must be whole non-empty rows",
            ));
        }
        if samples.iter().any(|x| !x.is_finite()) {
            return Err(IndexError::NonFinite);
        }
        let mut lo = vec![f32::INFINITY; dim];
        let mut hi = vec![f32::NEG_INFINITY; dim];
        let mut rows = 0usize;
        for row in samples.chunks_exact(dim) {
            let inv = match metric {
                Metric::Cosine => {
                    let n = norm(row);
                    if n <= 0.0 || !n.is_finite() {
                        continue;
                    }
                    1.0 / n
                }
                _ => 1.0,
            };
            for (i, x) in row.iter().enumerate() {
                let x = x * inv;
                lo[i] = lo[i].min(x);
                hi[i] = hi[i].max(x);
            }
            rows += 1;
        }
        if rows == 0 {
            return Err(IndexError::ZeroNorm);
        }
        let (offset, scale) = fit(&lo, &hi);
        if offset.iter().any(|o| !o.is_finite()) {
            return Err(IndexError::NonFinite);
        }
        Ok(Self {
            metric,
            kind: QuantKind::PerDim,
            epoch,
            offset,
            scale,
        })
    }

    /// Metric the codes serve.
    pub fn metric(&self) -> Metric {
        self.metric
    }
    /// Dimension.
    pub fn dim(&self) -> usize {
        self.offset.len()
    }
    /// Parameter version (`quant_epoch`).
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
    /// Policy.
    pub fn kind(&self) -> QuantKind {
        self.kind
    }
    /// Per-dimension offsets.
    pub fn offset(&self) -> &[f32] {
        &self.offset
    }
    /// Per-dimension scales.
    pub fn scale(&self) -> &[f32] {
        &self.scale
    }

    fn inv_norm(&self, v: &[f32]) -> f32 {
        match self.metric {
            Metric::Cosine => {
                let n = norm(v);
                if n > 0.0 {
                    1.0 / n
                } else {
                    1.0
                }
            }
            _ => 1.0,
        }
    }

    /// Encode `v` (already validated, `len == dim`) into `out`, clamping
    /// values outside the trained range.
    pub fn encode(&self, v: &[f32], out: &mut [u8]) {
        let inv = self.inv_norm(v);
        for (((o, x), off), s) in out.iter_mut().zip(v).zip(&self.offset).zip(&self.scale) {
            let q = ((x * inv - off) / s).round();
            *o = q.clamp(0.0, 255.0) as u8;
        }
    }

    /// Dequantize `code` into `out`.
    pub fn decode(&self, code: &[u8], out: &mut [f32]) {
        for (((o, c), off), s) in out.iter_mut().zip(code).zip(&self.offset).zip(&self.scale) {
            *o = off + s * f32::from(*c);
        }
    }

    /// Precompute per-dimension weights so each code costs one pass.
    pub fn prepare(&self, q: &[f32]) -> PreparedQuery {
        let inv = self.inv_norm(q);
        let d = self.dim();
        let mut w = Vec::with_capacity(d);
        let mut b = Vec::new();
        let mut c0 = 0.0f32;
        match self.metric {
            Metric::L2 => {
                b.reserve_exact(d);
                for ((x, off), s) in q.iter().zip(&self.offset).zip(&self.scale) {
                    w.push(*s);
                    b.push(off - x);
                }
            }
            Metric::Cosine | Metric::Dot => {
                for ((x, off), s) in q.iter().zip(&self.offset).zip(&self.scale) {
                    let x = x * inv;
                    w.push(x * s);
                    c0 += x * off;
                }
            }
        }
        PreparedQuery {
            metric: self.metric,
            w,
            b,
            c0,
        }
    }

    /// Prepare from a stored code (code↔code distance during construction).
    pub(crate) fn prepare_code(&self, code: &[u8], scratch: &mut Vec<f32>) -> PreparedQuery {
        scratch.resize(self.dim(), 0.0);
        self.decode(code, scratch);
        self.prepare(scratch)
    }

    pub(crate) fn write(&self, s: &mut dyn Sink) {
        s.put_u8(self.metric.code());
        s.put_u8(match self.kind {
            QuantKind::FixedRange => 1,
            QuantKind::PerDim => 2,
        });
        s.put_u32(self.dim() as u32);
        s.put_u64(self.epoch);
        s.put_f32s(&self.offset);
        s.put_f32s(&self.scale);
    }

    pub(crate) fn read(r: &mut Reader<'_>) -> Result<Self, DecodeError> {
        let metric = Metric::from_code(r.u8()?).ok_or(DecodeError::Malformed("metric"))?;
        let kind = match r.u8()? {
            1 => QuantKind::FixedRange,
            2 => QuantKind::PerDim,
            _ => return Err(DecodeError::Malformed("quant kind")),
        };
        let dim = r.u32()? as usize;
        if dim == 0 || dim > MAX_DIM {
            return Err(DecodeError::Malformed("quant dim"));
        }
        let epoch = r.u64()?;
        let offset = r.f32_vec(dim)?;
        let scale = r.f32_vec(dim)?;
        if offset.iter().any(|x| !x.is_finite())
            || scale.iter().any(|s| !(s.is_finite() && *s > 0.0))
        {
            return Err(DecodeError::Malformed("quant params"));
        }
        Ok(Self {
            metric,
            kind,
            epoch,
            offset,
            scale,
        })
    }
}

/// Query prepared against one [`QuantParams`]. Distances are approximate
/// and only comparable with each other (L2 returns the *squared* estimate).
#[derive(Debug, Clone)]
pub struct PreparedQuery {
    metric: Metric,
    w: Vec<f32>,
    b: Vec<f32>,
    c0: f32,
}

impl PreparedQuery {
    /// Approximate distance to one code (lower is closer).
    #[inline]
    pub fn distance(&self, code: &[u8]) -> f32 {
        match self.metric {
            Metric::L2 => l2w(&self.b, &self.w, code),
            Metric::Dot => -(self.c0 + wsum(&self.w, code)),
            Metric::Cosine => 1.0 - (self.c0 + wsum(&self.w, code)),
        }
    }
}

#[inline]
fn wsum(w: &[f32], c: &[u8]) -> f32 {
    let n = w.len().min(c.len());
    let (w, c) = (&w[..n], &c[..n]);
    let mut acc = [0.0f32; 8];
    let (cw, cc) = (w.chunks_exact(8), c.chunks_exact(8));
    let (rw, rc) = (cw.remainder(), cc.remainder());
    for (x, y) in cw.zip(cc) {
        for k in 0..8 {
            acc[k] += x[k] * f32::from(y[k]);
        }
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in rw.iter().zip(rc) {
        s += x * f32::from(*y);
    }
    s
}

#[inline]
fn l2w(b: &[f32], w: &[f32], c: &[u8]) -> f32 {
    let n = w.len().min(c.len()).min(b.len());
    let (b, w, c) = (&b[..n], &w[..n], &c[..n]);
    let mut acc = [0.0f32; 8];
    let mut i = 0;
    while i + 8 <= n {
        for k in 0..8 {
            let d = b[i + k] + w[i + k] * f32::from(c[i + k]);
            acc[k] += d * d;
        }
        i += 8;
    }
    let mut s = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    while i < n {
        let d = b[i] + w[i] * f32::from(c[i]);
        s += d * d;
        i += 1;
    }
    s
}

#[cfg(test)]
mod tests;
