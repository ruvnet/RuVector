//! Query policy for a stream that has not completed.
//!
//! Measured (B2, 100k x 128): `ProgressiveIndex` layers A and A+B answer worse
//! than an exact scan over the vectors that have already arrived (A 0.197 and
//! A+B 0.032 against 0.197 and 0.197 for the scan), and the scan is cheap
//! enough to run under a latency bound. So before Layer C is published the
//! default answer is the exact scan over the verified vectors, as long as the
//! scan fits `scan_latency_bound` (calibrated from the measured per-vector
//! cost on this machine, not assumed); beyond it, and for stores too big to
//! scan within the bound, the layers answer. Once Layer C is published it
//! answers (recall 0.95 at ~0.3 ms).
//!
//! Every answer states what it covers: `coverage_vectors` is the fraction of
//! the stream's vectors that had arrived, `QueryPath` says how it was made,
//! and `sealed` stays false until `Complete`.

use super::*;
use rvf_types::quality::BudgetReport;
use std::time::{Duration, Instant};

/// Which engine answers a query.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QueryMode {
    /// Scan the arrived vectors while that fits the latency bound and Layer C
    /// is absent; otherwise layers.
    #[default]
    Auto,
    /// Always the published layers (the pre-policy behaviour).
    LayersOnly,
    /// Always the exact scan over arrived vectors.
    ScanOnly,
}

/// Counts the work a layer search does and stops serving lookups once a budget
/// is spent (deterministic, in steps — not time). A hostile or merely huge
/// Layer A can then degrade one answer but not stall the client.
struct BudgetedStore<'a> {
    inner: &'a VerifiedVectors,
    steps: std::cell::Cell<u64>,
    limit: u64,
    exhausted: std::cell::Cell<bool>,
}

impl BudgetedStore<'_> {
    fn spend(&self, n: u64) -> bool {
        let used = self.steps.get().saturating_add(n);
        if used > self.limit {
            self.exhausted.set(true);
            return false;
        }
        self.steps.set(used);
        true
    }
}

impl rvf_index::VectorStore for BudgetedStore<'_> {
    fn get_vector(&self, id: u64) -> Option<&[f32]> {
        if !self.spend(1) {
            return None;
        }
        self.inner.get_vector(id)
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn for_each_in_range(&self, start: u64, end: u64, f: &mut dyn FnMut(u64, &[f32])) {
        if end <= start {
            return;
        }
        // A range costs at most one pass over the present ids.
        let cost = ((end - start) as u128).min(self.inner.len() as u128) as u64;
        if !self.spend(cost.max(1)) {
            return;
        }
        self.inner.for_each_in_range(start, end, f);
    }
}

/// Steps one layer search may spend: generous for honest layers (a search
/// touches a few thousand ids) and a hard ceiling for hostile ones.
fn default_work_budget(visible: usize) -> u64 {
    1_000_000 + 16 * visible as u64
}

/// Rows timed when calibrating the scan cost.
const CALIBRATION_ROWS: usize = 2048;
/// Smallest scan the Auto policy always allows, however slow the calibration.
const MIN_SCAN_ROWS: usize = 1024;

impl<'p> StreamIngest<'p> {
    /// Latency bound for the exact-scan path (default 25 ms).
    pub fn set_scan_latency_bound(&mut self, bound: Duration) {
        self.scan_bound = bound;
    }

    /// Largest number of visible vectors the exact scan is allowed to cover,
    /// from the measured cost per vector (calibrated on first use).
    pub fn scan_threshold(&self, q: &[f32]) -> usize {
        let mut ns = self.scan_ns_per_vec.get();
        if ns <= 0.0 {
            let rows = self.vectors.len().min(CALIBRATION_ROWS);
            if rows == 0 {
                return usize::MAX;
            }
            let t = Instant::now();
            let _ = self.vectors.exact_topk_rows(q, 10, rows, &|_| false);
            ns = (t.elapsed().as_nanos() as f64 / rows as f64).max(0.5);
            self.scan_ns_per_vec.set(ns);
        }
        (self.scan_bound.as_nanos() as f64 / ns) as usize
    }

    /// Override the per-query work budget (steps) of the layer search.
    pub fn set_query_work_budget(&mut self, steps: Option<u64>) {
        self.work_budget = steps;
    }

    /// k-NN over what is verified so far, using [`QueryMode::Auto`].
    pub fn query(&self, q: &[f32], k: usize, ef: usize) -> StreamAnswer {
        self.query_with(q, k, ef, QueryMode::Auto)
    }

    /// k-NN with an explicit engine choice (benchmarks, tests).
    pub fn query_with(&self, q: &[f32], k: usize, ef: usize, mode: QueryMode) -> StreamAnswer {
        let t0 = Instant::now();
        let layers = self.layers();
        let state = self.verifier.state();
        // `sealed` is a derived release: re-check the epoch pin now (ADR-353 section 7),
        // so a session superseded since completion is never reported as final.
        let sealed = state == StreamState::Complete
            && self
                .pins
                .is_none_or(|st| self.verifier.recheck_pin(st).is_ok());
        let visible = self.vectors.len();
        let ok_query = !self.unpublished && q.len() == self.vectors.dimension_hint() && visible > 0;
        let can_layers = layers.layer_a && ok_query;
        let scan = ok_query
            && match mode {
                QueryMode::ScanOnly => true,
                QueryMode::LayersOnly => false,
                QueryMode::Auto => {
                    !layers.layer_c && (!can_layers || visible <= self.scan_threshold(q))
                }
            };
        let mut truncated = false;
        let mut work_steps = 0u64;
        let mut scanned = visible;
        let (mut hits, path) = if scan {
            // Auto never scans more rows than the latency bound allows (F4-03):
            // without Layer A the scan covers the oldest rows only and says so.
            if mode == QueryMode::Auto {
                let cap = self.scan_threshold(q).max(MIN_SCAN_ROWS);
                if visible > cap {
                    scanned = cap;
                    truncated = true;
                }
            }
            let t = Instant::now();
            let h = self
                .vectors
                .exact_topk_rows(q, k, scanned, &|id| self.deleted.contains(&id));
            let per = t.elapsed().as_nanos() as f64 / scanned.max(1) as f64;
            let old = self.scan_ns_per_vec.get();
            self.scan_ns_per_vec.set(if old <= 0.0 {
                per
            } else {
                0.5 * old + 0.5 * per
            });
            (h, QueryPath::ArrivedScan)
        } else if can_layers {
            let limit = self
                .work_budget
                .unwrap_or_else(|| default_work_budget(visible));
            let ncent = self.index.layer_a.as_ref().map_or(0, |a| a.centroids.len()) as u64;
            let per_centroid = (self.vectors.dimension_hint() as u64).div_ceil(8).max(1);
            let b = BudgetedStore {
                inner: &self.vectors,
                steps: std::cell::Cell::new(0),
                limit,
                exhausted: std::cell::Cell::new(false),
            };
            // F4-02: the centroid scan and sort are charged to the budget; a
            // Layer A whose centroids alone exceed it is not searched.
            let h = if b.spend(ncent.saturating_mul(per_centroid)) {
                self.index.search(q, k + self.deleted.len().min(k), ef, &b)
            } else {
                Vec::new()
            };
            truncated = b.exhausted.get();
            work_steps = b.steps.get();
            (h, QueryPath::Layers)
        } else {
            (Vec::new(), QueryPath::Empty)
        };
        hits.retain(|(id, _)| !self.deleted.contains(id) && self.vectors.contains(*id));
        hits.truncate(k);
        let rq = match (path, layers.layer_b, layers.layer_c) {
            (QueryPath::ArrivedScan, _, _) => RetrievalQuality::BruteForceBudgeted,
            (_, _, true) => RetrievalQuality::Full,
            (_, true, false) => RetrievalQuality::Partial,
            _ => RetrievalQuality::LayerAOnly,
        };
        let full = sealed && layers.layer_c && path == QueryPath::Layers;
        let quality = if hits.is_empty() {
            ResponseQuality::Unreliable
        } else if truncated {
            ResponseQuality::Degraded
        } else if full {
            ResponseQuality::Verified
        } else {
            ResponseQuality::Usable
        };
        let degradation = (!full || truncated).then_some(DegradationReport {
            fallback_path: if truncated {
                FallbackPath::SafetyNetBudgetExhausted
            } else if path == QueryPath::ArrivedScan {
                FallbackPath::SafetyNetSelective
            } else {
                FallbackPath::None
            },
            reason: if truncated {
                DegradationReason::BudgetExhausted {
                    scanned: work_steps,
                    total: self.work_budget.unwrap_or_else(|| default_work_budget(visible)),
                    budget_type: rvf_types::quality::BudgetType::DistanceOps,
                }
            } else {
                DegradationReason::IndexNotLoaded { available: layers }
            },
            guarantee_lost: if truncated {
                "the layer search hit its work budget; results are partial"
            } else if path == QueryPath::ArrivedScan {
                "exact over the verified vectors received so far only (see coverage_vectors); not sealed"
            } else {
                "index layers and vectors still arriving; not sealed"
            },
        });
        let dim = self.vectors.dimension_hint();
        // Each VEC segment is a 64-byte header, a 6-byte `dim | count` prefix,
        // the records and padding; only the records are vectors.
        let expected = if dim > 0 {
            let rec = (8 + 4 * dim) as u64;
            self.vec_stored_lens
                .iter()
                .map(|&len| len.saturating_sub(64 + 6) / rec)
                .sum::<u64>() as f64
        } else {
            0.0
        };
        let all_arrived = matches!(
            self.verifier.state(),
            StreamState::Complete | StreamState::CompleteUnverified
        );
        let coverage_vectors = if all_arrived {
            1.0
        } else if expected > 0.0 {
            (scanned as f64 / expected).min(1.0)
        } else {
            f64::from(u8::from(sealed))
        };
        let body_len = self.verifier.descriptor().map_or(0, |d| d.header.body_len);
        let coverage_bytes = if body_len > 0 {
            (self.verifier.verified_bytes() as f64 / body_len as f64).min(1.0)
        } else {
            0.0
        };
        StreamAnswer {
            envelope: QualityEnvelope {
                results: hits
                    .into_iter()
                    .map(|(id, distance)| SearchResult {
                        id,
                        distance,
                        retrieval_quality: rq,
                        metadata: None,
                    })
                    .collect(),
                quality,
                evidence: SearchEvidenceSummary {
                    layers_used: layers,
                    safety_net_candidate_count: if path == QueryPath::ArrivedScan {
                        scanned as u32
                    } else {
                        0
                    },
                    ..Default::default()
                },
                budgets: BudgetReport {
                    total_us: t0.elapsed().as_micros() as u64,
                    linear_scan_count: if path == QueryPath::ArrivedScan {
                        scanned as u64
                    } else {
                        0
                    },
                    ..Default::default()
                },
                degradation,
            },
            sealed,
            state,
            verified_bytes: self.verifier.verified_bytes(),
            visible_vectors: visible,
            path,
            coverage_vectors,
            coverage_bytes,
            work_truncated: truncated,
            work_steps,
        }
    }
}
