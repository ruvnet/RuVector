//! Rank accumulation into MR / MRR / Hits@{1,3,10}.

use serde::{Deserialize, Serialize};

/// Aggregated link-prediction metrics over a set of ranked queries.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MetricSet {
    pub count: usize,
    /// Mean rank (lower is better).
    pub mr: f32,
    /// Mean reciprocal rank (higher is better).
    pub mrr: f32,
    pub hits1: f32,
    pub hits3: f32,
    pub hits10: f32,
}

impl MetricSet {
    pub fn empty() -> Self {
        Self {
            count: 0,
            mr: 0.0,
            mrr: 0.0,
            hits1: 0.0,
            hits3: 0.0,
            hits10: 0.0,
        }
    }
}

/// Running accumulator over integer ranks (1-based).
#[derive(Debug, Clone, Default)]
pub(crate) struct Accum {
    count: usize,
    sum_rank: f64,
    sum_rr: f64,
    h1: usize,
    h3: usize,
    h10: usize,
}

impl Accum {
    pub(crate) fn add(&mut self, rank: usize) {
        debug_assert!(rank >= 1);
        self.count += 1;
        self.sum_rank += rank as f64;
        self.sum_rr += 1.0 / rank as f64;
        if rank <= 1 {
            self.h1 += 1;
        }
        if rank <= 3 {
            self.h3 += 1;
        }
        if rank <= 10 {
            self.h10 += 1;
        }
    }

    /// Merge another accumulator (for the combined head+tail set).
    pub(crate) fn merge(&mut self, other: &Accum) {
        self.count += other.count;
        self.sum_rank += other.sum_rank;
        self.sum_rr += other.sum_rr;
        self.h1 += other.h1;
        self.h3 += other.h3;
        self.h10 += other.h10;
    }

    pub(crate) fn finish(&self) -> MetricSet {
        if self.count == 0 {
            return MetricSet::empty();
        }
        let c = self.count as f64;
        MetricSet {
            count: self.count,
            mr: (self.sum_rank / c) as f32,
            mrr: (self.sum_rr / c) as f32,
            hits1: (self.h1 as f64 / c) as f32,
            hits3: (self.h3 as f64 / c) as f32,
            hits10: (self.h10 as f64 / c) as f32,
        }
    }
}
