//! `PqIvf` — the BET 5 contender: an IVF index with **product-quantized within-list pruning**
//! (IVFADC). Over the *same* `ruvector-rairs` k-means substrate as the plain-`IvfFlat` incumbent
//! and the BET-4 `BnBIvf`, it adds a product quantizer so a list can be scanned with cheap
//! **asymmetric distance computation (ADC)** — an `m`-entry table lookup-sum per member instead of a
//! full `D`-dim L2 — then recovers exactness with a small exact-L2 **re-rank** of the top-`R` ADC
//! candidates.
//!
//! This is the *different mechanism* ADR-205 left open: ADR-205's triangle-inequality bound competed
//! with `nprobe` on the **same axis** (which lists to scan) and was redundant (1.00×). PQ competes on
//! an **orthogonal axis** — the cost of *considering* a member — so a win is not structurally
//! impossible. Whether it pays is the amortization question the BET-5 pre-registration freezes.
//!
//! ## Cost accounting (one unit = one full `D`-dim L2 = "1 member-eval-equivalent")
//! - ADC table build (per query): `m·256·(D/m)/D = 256` equivalents — the fixed overhead.
//! - ADC member scan: `m/D` equivalents.
//! - exact re-rank member: `1` equivalent.
//!
//! The kernel returns raw counters; [`AdcCost::l2_equiv`] does the conversion so the harness charges
//! every operation in one honest unit (no free LUT, no free re-rank).

use crate::kernel::{build_ivf, IvfParts};
use crate::oracle::l2;
use ruvector_rairs::{kmeans, SearchResult};
use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// A product-quantized IVF index sharing its centroids/lists with [`crate::kernel::BnBIvf`]
/// (build with the same `nclusters`/`max_iter`/`seed` → identical k-means → genuinely shared index).
pub struct PqIvf {
    centroids: Vec<Vec<f32>>,
    /// Per cluster: `(id, vector)` of its members (full vectors retained for exact re-rank).
    lists: Vec<Vec<(usize, Vec<f32>)>>,
    /// `m` sub-quantizer codebooks; `codebooks[j]` is 256 sub-centroids of `dim/m` dims.
    codebooks: Vec<Vec<Vec<f32>>>,
    /// PQ codes indexed by original corpus id: `codes[id][j]` = sub-centroid index in subspace `j`.
    codes: Vec<[u8; MAX_M]>,
    m: usize,
    sub: usize,
    dim: usize,
}

/// Max sub-quantizers supported (fixed-size code array; `m ∈ {8,16}` in the pre-reg ≤ this).
const MAX_M: usize = 32;
const PQ_CENTROIDS: usize = 256;

/// Raw per-query counters from an ADC+re-rank search, converted to honest cost by [`Self::l2_equiv`].
#[derive(Clone, Copy, Debug, Default)]
pub struct AdcCost {
    /// Members touched by the cheap ADC scan.
    pub adc_members: usize,
    /// Members recomputed with exact `D`-dim L2 (the re-rank pool actually used).
    pub rerank: usize,
    pub m: usize,
    pub dim: usize,
}
impl AdcCost {
    /// Within-list cost in full-L2-equivalents: `256` (LUT) + `adc_members·m/D` + `rerank·1`.
    /// Routing (`nclusters` centroid evals) is charged separately and equally by the harness.
    pub fn l2_equiv(&self) -> f64 {
        let lut = (PQ_CENTROIDS * self.dim) as f64 / self.dim.max(1) as f64; // = 256
        let adc = self.adc_members as f64 * self.m as f64 / self.dim.max(1) as f64;
        lut + adc + self.rerank as f64
    }
}

// --- top-k accumulator (mirrors kernel.rs; kept local so the modules stay independent) ---
struct Cand {
    dist: f32,
    id: usize,
}
impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.dist.total_cmp(&o.dist)
    }
}
#[inline]
fn consider(heap: &mut BinaryHeap<Cand>, k: usize, id: usize, d: f32) {
    if heap.len() < k {
        heap.push(Cand { dist: d, id });
    } else if d < heap.peek().unwrap().dist {
        heap.pop();
        heap.push(Cand { dist: d, id });
    }
}
fn finalize(heap: BinaryHeap<Cand>) -> Vec<SearchResult> {
    let mut res: Vec<SearchResult> = heap
        .into_iter()
        .map(|c| SearchResult {
            id: c.id,
            distance: c.dist,
        })
        .collect();
    res.sort_by(|a, b| a.distance.total_cmp(&b.distance));
    res
}

/// Squared L2 over a dim slice — the ADC table metric (ranking-equivalent to L2, cheaper).
#[inline]
fn l2sq_slice(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

impl PqIvf {
    /// Build the IVF (shared k-means) **and** train an `m`-subquantizer product quantizer on top.
    /// `dim % m == 0` required. PQ codebooks use 256 sub-centroids (8-bit codes); training uses
    /// `seed + 1 + j` per subspace so the IVF seed (`seed`) reproduces [`BnBIvf`]'s centroids exactly.
    pub fn build(
        corpus: &[Vec<f32>],
        nclusters: usize,
        m: usize,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        Self::from_parts(&build_ivf(corpus, nclusters, max_iter, seed), corpus, m, max_iter, seed)
    }

    /// Construct from a pre-built shared [`IvfParts`] (skips re-clustering) and train the `m`-sub
    /// product quantizer on `corpus`. Reusing one `IvfParts` for `BnBIvf` + every `PqIvf(m)` pays
    /// the k-means once per cell while guaranteeing all contenders share an identical index.
    pub fn from_parts(
        parts: &IvfParts,
        corpus: &[Vec<f32>],
        m: usize,
        max_iter: usize,
        seed: u64,
    ) -> Self {
        assert!(!corpus.is_empty(), "empty corpus");
        let dim = corpus[0].len();
        assert!((1..=MAX_M).contains(&m), "m out of range");
        assert!(dim.is_multiple_of(m), "dim {dim} not divisible by m {m}");
        let sub = dim / m;

        let centroids = parts.centroids.clone();
        let lists = parts.lists.clone();

        // --- PQ: one k-means per subspace; assignments ARE the codes ---
        let n = corpus.len();
        let mut codes = vec![[0u8; MAX_M]; n];
        let mut codebooks: Vec<Vec<Vec<f32>>> = Vec::with_capacity(m);
        for j in 0..m {
            let lo = j * sub;
            let hi = lo + sub;
            let subvecs: Vec<Vec<f32>> = corpus.iter().map(|v| v[lo..hi].to_vec()).collect();
            let kc_pq = PQ_CENTROIDS.min(n).max(1);
            let (subcentroids, subassign) = kmeans::train(&subvecs, kc_pq, max_iter, seed + 1 + j as u64);
            for (code_row, &c) in codes.iter_mut().zip(subassign.iter()) {
                code_row[j] = c as u8;
            }
            codebooks.push(subcentroids);
        }

        Self {
            centroids,
            lists,
            codebooks,
            codes,
            m,
            sub,
            dim,
        }
    }

    pub fn num_lists(&self) -> usize {
        self.centroids.len()
    }
    pub fn m(&self) -> usize {
        self.m
    }
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Centroid clone for the shared-index assertion in the gate test.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }

    /// Build the per-query ADC lookup table: `lut[j][c] = ‖q_subj − codebook[j][c]‖²` over the
    /// `dim/m` dims of subspace `j`. `m × 256` entries; charged as 256 full-L2-equivalents.
    fn adc_lut(&self, q: &[f32]) -> Vec<[f32; PQ_CENTROIDS]> {
        let mut lut = vec![[0f32; PQ_CENTROIDS]; self.m];
        for (j, lut_j) in lut.iter_mut().enumerate() {
            let lo = j * self.sub;
            let qs = &q[lo..lo + self.sub];
            for (c, cb) in self.codebooks[j].iter().enumerate() {
                lut_j[c] = l2sq_slice(qs, cb);
            }
        }
        lut
    }

    #[inline]
    fn adc_dist(&self, lut: &[[f32; PQ_CENTROIDS]], id: usize) -> f32 {
        // `lut` has `m` entries ≤ `code`'s MAX_M; zip stops at `m` (the valid codes).
        let mut d = 0f32;
        for (lut_j, &cj) in lut.iter().zip(self.codes[id].iter()) {
            d += lut_j[cj as usize];
        }
        d
    }

    /// The `nprobe` nearest lists by centroid distance (the incumbent's list selection, shared).
    fn route(&self, q: &[f32], nprobe: usize) -> Vec<usize> {
        let mut cd: Vec<(f32, usize)> = (0..self.centroids.len())
            .map(|c| (l2(q, &self.centroids[c]), c))
            .collect();
        cd.sort_by(|a, b| a.0.total_cmp(&b.0));
        let np = nprobe.clamp(1, self.centroids.len());
        cd.into_iter().take(np).map(|(_, c)| c).collect()
    }

    /// **The BET-5 contender.** Scan the `nprobe` nearest lists with cheap ADC, keep the top-`R`
    /// candidates by ADC distance, then recompute **exact** L2 on those `R` and return the top-`k`.
    /// Returns `(top-k, AdcCost)`; routing evals are charged separately by the harness.
    pub fn search_adc_rerank(
        &self,
        q: &[f32],
        k: usize,
        nprobe: usize,
        r: usize,
    ) -> (Vec<SearchResult>, AdcCost) {
        let lists = self.route(q, nprobe);
        let lut = self.adc_lut(q);

        // ADC scan: collect (adc_dist, id, &vector) for every member of the probed lists.
        let mut scanned: Vec<(f32, usize, &[f32])> = Vec::new();
        for &c in &lists {
            for (id, v) in &self.lists[c] {
                scanned.push((self.adc_dist(&lut, *id), *id, v.as_slice()));
            }
        }
        let adc_members = scanned.len();

        // Keep the top-R candidates by ADC distance (partial sort; ascending).
        let rr = r.max(1).min(adc_members);
        if rr < adc_members {
            scanned.select_nth_unstable_by(rr - 1, |a, b| a.0.total_cmp(&b.0));
            scanned.truncate(rr);
        }
        let rerank = scanned.len();

        // Exact re-rank: recompute true L2 on the pooled candidates only.
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for (_adc, id, v) in &scanned {
            consider(&mut heap, k, *id, l2(q, v));
        }

        (
            finalize(heap),
            AdcCost {
                adc_members,
                rerank,
                m: self.m,
                dim: self.dim,
            },
        )
    }

    /// **Pure-ADC ceiling probe** (control): top-`k` by ADC distance with **no** re-rank. Measures how
    /// lossy the quantizer is on this data — the mechanistic explainer for the `R` re-rank needs.
    pub fn search_adc_only(&self, q: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        let lists = self.route(q, nprobe);
        let lut = self.adc_lut(q);
        let mut heap: BinaryHeap<Cand> = BinaryHeap::with_capacity(k + 1);
        for &c in &lists {
            for (id, _v) in &self.lists[c] {
                let d = self.adc_dist(&lut, *id);
                consider(&mut heap, k, *id, d);
            }
        }
        finalize(heap)
    }

    /// Members in the `nprobe` nearest lists (the working-set size the incumbent must full-scan).
    pub fn working_set(&self, q: &[f32], nprobe: usize) -> usize {
        self.route(q, nprobe)
            .iter()
            .map(|&c| self.lists[c].len())
            .sum()
    }
}
