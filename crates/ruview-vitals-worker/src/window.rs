//! Sliding window over per-subcarrier residuals.
//!
//! The vital-sign extractors operate on a multi-second time series of
//! residuals; this module is the ring-buffer carrier that holds it
//! between frame ingestion and extraction.
//!
//! Layout: one [`VecDeque<f64>`] per subcarrier (the per-channel time
//! series), plus a parallel deque of microsecond timestamps. All
//! deques share the same capacity; pushing into a full window drops
//! the oldest sample on every channel atomically.

use std::collections::VecDeque;

use crate::types::NodeId;

/// Per-subcarrier ring-buffered residual window. Cheap to push (O(W)
/// across W subcarriers); cheap to read.
#[derive(Debug, Clone)]
pub struct CsiSlidingWindow {
    n_subcarriers: usize,
    capacity: usize,
    by_subcarrier: Vec<VecDeque<f64>>,
    timestamps_us: VecDeque<i64>,
    sample_rate_hz: f64,
    last_node_id: NodeId,
}

impl CsiSlidingWindow {
    #[must_use]
    pub fn new(n_subcarriers: usize, capacity: usize, sample_rate_hz: f64) -> Self {
        Self {
            n_subcarriers,
            capacity,
            by_subcarrier: (0..n_subcarriers)
                .map(|_| VecDeque::with_capacity(capacity))
                .collect(),
            timestamps_us: VecDeque::with_capacity(capacity),
            sample_rate_hz,
            last_node_id: 0,
        }
    }

    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.capacity
    }

    #[must_use]
    pub const fn n_subcarriers(&self) -> usize {
        self.n_subcarriers
    }

    #[must_use]
    pub const fn sample_rate_hz(&self) -> f64 {
        self.sample_rate_hz
    }

    #[must_use]
    pub const fn last_node_id(&self) -> NodeId {
        self.last_node_id
    }

    /// Number of samples currently in the window.
    #[must_use]
    pub fn len(&self) -> usize {
        self.timestamps_us.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity
    }

    /// Push residuals + their timestamp + the source node_id.
    ///
    /// `residuals.len()` is decoupled from `n_subcarriers` for
    /// robustness: extra entries are dropped, missing ones are
    /// zero-filled. This keeps the worker tolerant to per-frame
    /// subcarrier jitter (some ESP32 firmware variants emit slightly
    /// different counts on band-edge channels).
    pub fn push(&mut self, residuals: &[f64], ts_us: i64, node_id: NodeId) {
        for (sc, deq) in self.by_subcarrier.iter_mut().enumerate() {
            let v = residuals.get(sc).copied().unwrap_or(0.0);
            if deq.len() == self.capacity {
                deq.pop_front();
            }
            deq.push_back(v);
        }
        if self.timestamps_us.len() == self.capacity {
            self.timestamps_us.pop_front();
        }
        self.timestamps_us.push_back(ts_us);
        self.last_node_id = node_id;
    }

    /// Borrow the time series for one subcarrier.
    #[must_use]
    pub fn subcarrier(&self, sc: usize) -> Option<&VecDeque<f64>> {
        self.by_subcarrier.get(sc)
    }

    /// Most recent timestamp pushed (in microseconds since UNIX epoch).
    #[must_use]
    pub fn latest_timestamp_us(&self) -> Option<i64> {
        self.timestamps_us.back().copied()
    }

    /// Earliest timestamp still in the window.
    #[must_use]
    pub fn earliest_timestamp_us(&self) -> Option<i64> {
        self.timestamps_us.front().copied()
    }

    /// Window-center timestamp — useful as the canonical timestamp
    /// for an emitted [`crate::types::VitalReading`].
    #[must_use]
    pub fn center_timestamp_us(&self) -> Option<i64> {
        match (self.earliest_timestamp_us(), self.latest_timestamp_us()) {
            (Some(a), Some(b)) => Some(a + (b - a) / 2),
            _ => None,
        }
    }

    /// Mean residual across all subcarriers at frame index `t`.
    ///
    /// Used by zero-crossing-style extractors that fuse subcarriers
    /// via simple arithmetic mean (the breathing extractor variant).
    #[must_use]
    pub fn mean_amplitude(&self, t: usize) -> Option<f64> {
        if t >= self.len() {
            return None;
        }
        let mut sum = 0.0;
        let mut n = 0usize;
        for deq in &self.by_subcarrier {
            if let Some(v) = deq.get(t) {
                sum += *v;
                n += 1;
            }
        }
        if n == 0 {
            None
        } else {
            Some(sum / n as f64)
        }
    }

    /// Per-subcarrier sample variance over the current window.
    ///
    /// High-variance subcarriers carry most of the breathing /
    /// heart-rate signal; the breathing extractor uses these values
    /// (normalised) as fusion weights.
    #[must_use]
    pub fn subcarrier_variance(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.n_subcarriers];
        for (sc, deq) in self.by_subcarrier.iter().enumerate() {
            if deq.is_empty() {
                continue;
            }
            let n = deq.len() as f64;
            let mean = deq.iter().sum::<f64>() / n;
            let var = deq.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            out[sc] = var;
        }
        out
    }

    /// Per-subcarrier weights, normalised to sum to 1.0. Falls back
    /// to uniform when the variance vector is degenerate (all zero).
    #[must_use]
    pub fn variance_weights(&self) -> Vec<f64> {
        let var = self.subcarrier_variance();
        let total: f64 = var.iter().sum();
        if total <= 0.0 || self.n_subcarriers == 0 {
            return vec![
                1.0 / self.n_subcarriers.max(1) as f64;
                self.n_subcarriers
            ];
        }
        var.into_iter().map(|v| v / total).collect()
    }

    /// Drop all samples; window becomes empty. Allocations preserved.
    pub fn clear(&mut self) {
        for deq in &mut self.by_subcarrier {
            deq.clear();
        }
        self.timestamps_us.clear();
        self.last_node_id = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pushes_grow_then_evict_oldest() {
        let mut w = CsiSlidingWindow::new(2, 3, 30.0);
        for (i, ts) in [100_i64, 200, 300].iter().enumerate() {
            w.push(&[i as f64, (i as f64) * 2.0], *ts, 5);
            assert_eq!(w.len(), i + 1);
        }
        assert!(w.is_full());
        assert_eq!(w.last_node_id(), 5);
        // Push fourth: oldest (timestamp 100) is evicted.
        w.push(&[99.0, 99.0], 400, 5);
        assert_eq!(w.len(), 3);
        assert_eq!(w.earliest_timestamp_us(), Some(200));
        assert_eq!(w.latest_timestamp_us(), Some(400));
    }

    #[test]
    fn missing_subcarriers_zero_filled() {
        let mut w = CsiSlidingWindow::new(4, 2, 30.0);
        w.push(&[1.0, 2.0], 0, 1);
        let sc2 = w.subcarrier(2).unwrap();
        assert_eq!(*sc2.front().unwrap(), 0.0);
    }

    #[test]
    fn extra_subcarriers_dropped() {
        let mut w = CsiSlidingWindow::new(2, 2, 30.0);
        w.push(&[1.0, 2.0, 3.0, 4.0], 0, 1);
        assert_eq!(w.subcarrier(0).unwrap().front().copied(), Some(1.0));
        assert_eq!(w.subcarrier(1).unwrap().front().copied(), Some(2.0));
        assert!(w.subcarrier(2).is_none());
    }

    #[test]
    fn center_timestamp_is_midpoint() {
        let mut w = CsiSlidingWindow::new(1, 4, 30.0);
        w.push(&[0.0], 0, 0);
        w.push(&[0.0], 1000, 0);
        w.push(&[0.0], 2000, 0);
        w.push(&[0.0], 3000, 0);
        assert_eq!(w.center_timestamp_us(), Some(1500));
    }

    #[test]
    fn variance_weights_sum_to_one_when_signal_present() {
        let mut w = CsiSlidingWindow::new(3, 4, 30.0);
        // sc0 has zero variance; sc1 / sc2 vary.
        w.push(&[1.0, 0.0, 5.0], 0, 0);
        w.push(&[1.0, 1.0, 0.0], 1, 0);
        w.push(&[1.0, 0.0, 5.0], 2, 0);
        w.push(&[1.0, 1.0, 0.0], 3, 0);
        let wts = w.variance_weights();
        let s: f64 = wts.iter().sum();
        assert!((s - 1.0).abs() < 1e-9);
        // sc0 is the lowest-weighted (it's constant).
        assert!(wts[0] < wts[1] && wts[0] < wts[2]);
    }

    #[test]
    fn variance_weights_uniform_when_no_signal() {
        let mut w = CsiSlidingWindow::new(4, 3, 30.0);
        for ts in 0..3 {
            w.push(&[1.0; 4], ts, 0);
        }
        let wts = w.variance_weights();
        for &v in &wts {
            assert!((v - 0.25).abs() < 1e-9);
        }
    }

    #[test]
    fn mean_amplitude_at_index() {
        let mut w = CsiSlidingWindow::new(3, 4, 30.0);
        w.push(&[1.0, 2.0, 3.0], 0, 0);
        w.push(&[4.0, 5.0, 6.0], 1, 0);
        assert_eq!(w.mean_amplitude(0), Some(2.0));
        assert_eq!(w.mean_amplitude(1), Some(5.0));
        assert_eq!(w.mean_amplitude(2), None);
    }

    #[test]
    fn clear_drops_samples() {
        let mut w = CsiSlidingWindow::new(2, 3, 30.0);
        w.push(&[1.0, 2.0], 0, 9);
        w.clear();
        assert!(w.is_empty());
        assert_eq!(w.last_node_id(), 0);
    }
}
