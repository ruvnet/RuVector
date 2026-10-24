//! Precision allocation: turns per-vector coherence scores into a mix of
//! `ruvector-turboquant` codes (Turbo4 4-bit, 1-bit) plus, for one variant,
//! an exact f32 residual cache for the most incoherent outliers.
//!
//! Three variants, matching the nightly-harness required shape
//! (`baseline`, `candidate_A`, `candidate_B`):
//!
//! * [`Variant::Baseline`] — every vector gets a Turbo4 4-bit code. This is
//!   `ruvector-turboquant`'s actual production datatype (ADR-296), not a
//!   weakened straw man.
//! * [`Variant::CandidateA`] — the `high_precision_frac` least-coherent
//!   vectors (per block) keep Turbo4; the rest drop to the 1-bit plane.
//! * [`Variant::CandidateB`] — same split as A, plus the `residual_frac`
//!   *most* incoherent vectors additionally get an exact f32 residual, so
//!   their distance is computed exactly rather than estimated.

use std::collections::HashSet;

use ruvector_turboquant::{Bits1Query, Metric, Turbo4Codec, Turbo4Query};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    Bit1,
    Turbo4,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    Baseline,
    CandidateA,
    CandidateB,
}

pub struct Encoded {
    pub precision: Precision,
    pub turbo4: Option<Vec<u8>>,
    pub bits1: Option<Vec<u8>>,
    pub exact: Option<Vec<f32>>,
}

impl Encoded {
    pub fn bytes(&self) -> usize {
        self.turbo4.as_ref().map_or(0, |b| b.len())
            + self.bits1.as_ref().map_or(0, |b| b.len())
            + self.exact.as_ref().map_or(0, |v| v.len() * 4)
    }
}

/// Per-block (same grouping as `coherence::corpus_coherence`) selection of
/// the `frac` *least* coherent vectors, by exact rank rather than a value
/// threshold: coherence scores are coarse ratios of small integer counts
/// and tie heavily (e.g. a whole block with no cheap min-cut separation
/// scores uniformly 1.0), so comparing against a threshold *value* can
/// silently select "everyone tied at the cutoff" instead of the intended
/// fraction. Ranking with an index tiebreak always selects exactly
/// `ceil(frac * block_size)` vectors regardless of ties.
fn select_bottom_frac(coherence: &[f32], idxs: &[usize], frac: f32) -> HashSet<usize> {
    if idxs.is_empty() {
        return HashSet::new();
    }
    let mut ranked: Vec<usize> = idxs.to_vec();
    ranked.sort_by(|&a, &b| {
        coherence[a]
            .partial_cmp(&coherence[b])
            .unwrap()
            .then(a.cmp(&b))
    });
    let k = ((ranked.len() as f32 * frac).ceil() as usize).clamp(1, ranked.len());
    ranked.into_iter().take(k).collect()
}

pub fn encode_corpus(
    vectors: &[Vec<f32>],
    coherence: &[f32],
    block_of: &[u32],
    n_blocks: usize,
    variant: Variant,
    codec: &Turbo4Codec,
    high_precision_frac: f32,
    residual_frac: f32,
) -> Vec<Encoded> {
    let n = vectors.len();

    // Per-block "keep Turbo4" set and, for candidate B, the smaller
    // "also cache exact" subset of it.
    let mut hp_set: Vec<HashSet<usize>> = vec![HashSet::new(); n_blocks];
    let mut res_set: Vec<HashSet<usize>> = vec![HashSet::new(); n_blocks];
    if matches!(variant, Variant::CandidateA | Variant::CandidateB) {
        for b in 0..n_blocks {
            let idxs: Vec<usize> = (0..n).filter(|&i| block_of[i] as usize == b).collect();
            hp_set[b] = select_bottom_frac(coherence, &idxs, high_precision_frac);
            res_set[b] = select_bottom_frac(coherence, &idxs, residual_frac);
        }
    }

    vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let (t4, b1) = codec.encode_dual(v).expect("dim mismatch");
            match variant {
                Variant::Baseline => Encoded {
                    precision: Precision::Turbo4,
                    turbo4: Some(t4),
                    bits1: None,
                    exact: None,
                },
                Variant::CandidateA | Variant::CandidateB => {
                    let b = block_of[i] as usize;
                    let high_precision = hp_set[b].contains(&i);
                    if !high_precision {
                        return Encoded {
                            precision: Precision::Bit1,
                            turbo4: None,
                            bits1: Some(b1),
                            exact: None,
                        };
                    }
                    let needs_exact =
                        matches!(variant, Variant::CandidateB) && res_set[b].contains(&i);
                    Encoded {
                        precision: Precision::Turbo4,
                        turbo4: Some(t4),
                        bits1: None,
                        exact: if needs_exact { Some(v.clone()) } else { None },
                    }
                }
            }
        })
        .collect()
}

/// Estimated distance from a prepared query to one encoded entry. Exact
/// residuals (candidate B outliers) are scored exactly; everything else is
/// scored by whichever quantized plane it holds.
pub fn estimate_distance(
    metric: Metric,
    query_raw: &[f32],
    turbo4_query: &Turbo4Query,
    bits1_query: &Bits1Query,
    entry: &Encoded,
    dim: usize,
) -> f32 {
    if let Some(exact) = &entry.exact {
        return match metric {
            Metric::Euclidean => query_raw
                .iter()
                .zip(exact)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt(),
            _ => unimplemented!("benchmark only exercises Euclidean"),
        };
    }
    if let Some(code) = &entry.turbo4 {
        return ruvector_turboquant::asymmetric_distance(metric, &turbo4_query.blob, code, dim);
    }
    if let Some(code) = &entry.bits1 {
        return bits1_query.distance_to(metric, code);
    }
    unreachable!("encoded entry has no payload")
}
