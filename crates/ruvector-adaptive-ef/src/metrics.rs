//! Latency tracking and recall estimation utilities.

/// Rolling window of recent latency observations (microseconds).
pub struct LatencyWindow {
    buf: Vec<u64>,
    head: usize,
    filled: bool,
    cap: usize,
}

impl LatencyWindow {
    pub fn new(capacity: usize) -> Self {
        Self { buf: vec![0; capacity], head: 0, filled: false, cap: capacity }
    }

    pub fn push(&mut self, latency_us: u64) {
        self.buf[self.head] = latency_us;
        self.head = (self.head + 1) % self.cap;
        if self.head == 0 {
            self.filled = true;
        }
    }

    fn samples(&self) -> &[u64] {
        if self.filled { &self.buf } else { &self.buf[..self.head] }
    }

    pub fn mean_us(&self) -> f64 {
        let s = self.samples();
        if s.is_empty() { return 0.0; }
        s.iter().sum::<u64>() as f64 / s.len() as f64
    }

    pub fn percentile_us(&self, pct: f64) -> u64 {
        let s = self.samples();
        if s.is_empty() { return 0; }
        let mut sorted: Vec<u64> = s.to_vec();
        sorted.sort_unstable();
        let idx = ((pct / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[idx.min(sorted.len() - 1)]
    }

    pub fn len(&self) -> usize {
        self.samples().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Estimate recall by comparing approximate result set against a ground-truth set.
///
/// Returns the fraction of ground-truth ids found in the approximate result.
pub fn recall_at_k(approx: &[usize], ground_truth: &[usize]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let found = approx.iter().filter(|id| ground_truth.contains(id)).count();
    found as f32 / ground_truth.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_mean_and_percentile() {
        let mut w = LatencyWindow::new(10);
        for v in [100u64, 200, 300, 400, 500] {
            w.push(v);
        }
        assert!((w.mean_us() - 300.0).abs() < 1.0);
        assert_eq!(w.percentile_us(50.0), 300);
        assert_eq!(w.percentile_us(95.0), 500);
    }

    #[test]
    fn recall_exact_overlap() {
        let approx = vec![1, 2, 3, 4, 5];
        let gt = vec![1, 2, 3, 4, 5];
        assert!((recall_at_k(&approx, &gt) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_partial_overlap() {
        let approx = vec![1, 2, 6, 7, 8];
        let gt = vec![1, 2, 3, 4, 5];
        assert!((recall_at_k(&approx, &gt) - 0.4).abs() < 1e-6);
    }
}
