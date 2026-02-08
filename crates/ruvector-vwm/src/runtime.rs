//! Three-cadence loop controller for the Visual World Model runtime.
//!
//! Manages three concurrent loops at different rates:
//! - Fast loop (30-60 Hz): render, pose tracking, draw list construction
//! - Medium loop (2-10 Hz): Gaussian updates, entity tracking, delta writes
//! - Slow loop (0.1-1 Hz): GNN refinement, consolidation, pruning, keyframe publishing
//!
//! The [`LoopScheduler`] does **not** own threads. The caller drives ticks by
//! calling [`LoopScheduler::poll`] with the current wall-clock time and then
//! executing the returned cadences. After each cadence executes, the caller
//! reports the duration via [`LoopScheduler::record_tick`] so that budget
//! overrun tracking stays accurate.

/// Configuration for loop timing.
#[derive(Clone, Debug)]
pub struct LoopConfig {
    /// Target frequency for the fast loop in Hz (default 60.0).
    pub fast_target_hz: f32,
    /// Target frequency for the medium loop in Hz (default 5.0).
    pub medium_target_hz: f32,
    /// Target frequency for the slow loop in Hz (default 0.5).
    pub slow_target_hz: f32,
    /// Maximum milliseconds allowed per fast tick (default 16.0).
    pub fast_budget_ms: f32,
    /// Maximum milliseconds allowed per medium tick (default 50.0).
    pub medium_budget_ms: f32,
    /// Maximum milliseconds allowed per slow tick (default 500.0).
    pub slow_budget_ms: f32,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            fast_target_hz: 60.0,
            medium_target_hz: 5.0,
            slow_target_hz: 0.5,
            fast_budget_ms: 16.0,
            medium_budget_ms: 50.0,
            slow_budget_ms: 500.0,
        }
    }
}

impl LoopConfig {
    /// Compute the interval in milliseconds for a given frequency.
    fn interval_ms(hz: f32) -> f64 {
        if hz <= 0.0 {
            return f64::MAX;
        }
        1000.0 / hz as f64
    }

    /// Fast loop interval in ms.
    pub fn fast_interval_ms(&self) -> f64 {
        Self::interval_ms(self.fast_target_hz)
    }

    /// Medium loop interval in ms.
    pub fn medium_interval_ms(&self) -> f64 {
        Self::interval_ms(self.medium_target_hz)
    }

    /// Slow loop interval in ms.
    pub fn slow_interval_ms(&self) -> f64 {
        Self::interval_ms(self.slow_target_hz)
    }
}

/// Metrics collected per loop tick.
#[derive(Clone, Debug, Default)]
pub struct LoopMetrics {
    /// Total number of ticks recorded.
    pub tick_count: u64,
    /// Cumulative time spent in ticks (ms).
    pub total_time_ms: f64,
    /// Duration of the most recent tick (ms).
    pub last_tick_ms: f64,
    /// Maximum tick duration observed (ms).
    pub max_tick_ms: f64,
    /// Number of ticks that exceeded the budget.
    pub budget_overruns: u64,
    /// Running average tick duration (ms).
    pub avg_tick_ms: f64,
}

impl LoopMetrics {
    /// Record a completed tick with the given duration and budget.
    ///
    /// Updates all counters, tracks the maximum, and recomputes the running
    /// average. If `duration_ms` exceeds `budget_ms`, the overrun counter is
    /// incremented.
    pub fn record_tick(&mut self, duration_ms: f64, budget_ms: f64) {
        self.tick_count += 1;
        self.total_time_ms += duration_ms;
        self.last_tick_ms = duration_ms;
        if duration_ms > self.max_tick_ms {
            self.max_tick_ms = duration_ms;
        }
        if duration_ms > budget_ms {
            self.budget_overruns += 1;
        }
        self.avg_tick_ms = self.total_time_ms / self.tick_count as f64;
    }

    /// Reset all metrics to their default state.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Loop cadence identifier.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LoopCadence {
    /// Fast loop (~30-60 Hz).
    Fast,
    /// Medium loop (~2-10 Hz).
    Medium,
    /// Slow loop (~0.1-1 Hz).
    Slow,
}

/// Three-cadence loop scheduler.
///
/// Determines which loops should tick based on elapsed time.
/// Does **not** own threads -- the caller drives ticks.
pub struct LoopScheduler {
    config: LoopConfig,
    fast_metrics: LoopMetrics,
    medium_metrics: LoopMetrics,
    slow_metrics: LoopMetrics,
    last_fast_ms: f64,
    last_medium_ms: f64,
    last_slow_ms: f64,
}

impl LoopScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: LoopConfig) -> Self {
        Self {
            config,
            fast_metrics: LoopMetrics::default(),
            medium_metrics: LoopMetrics::default(),
            slow_metrics: LoopMetrics::default(),
            last_fast_ms: 0.0,
            last_medium_ms: 0.0,
            last_slow_ms: 0.0,
        }
    }

    /// Create a new scheduler with default configuration.
    pub fn with_defaults() -> Self {
        Self::new(LoopConfig::default())
    }

    /// Given current time in ms, return which loops should tick.
    ///
    /// Returns a `Vec` of `(LoopCadence, elapsed_since_last_tick_ms)` for each
    /// cadence whose interval has elapsed. The fast loop is checked first,
    /// then medium, then slow.
    pub fn poll(&mut self, now_ms: f64) -> Vec<(LoopCadence, f64)> {
        let mut ready = Vec::with_capacity(3);

        let fast_interval = self.config.fast_interval_ms();
        let elapsed_fast = now_ms - self.last_fast_ms;
        if elapsed_fast >= fast_interval {
            ready.push((LoopCadence::Fast, elapsed_fast));
            self.last_fast_ms = now_ms;
        }

        let medium_interval = self.config.medium_interval_ms();
        let elapsed_medium = now_ms - self.last_medium_ms;
        if elapsed_medium >= medium_interval {
            ready.push((LoopCadence::Medium, elapsed_medium));
            self.last_medium_ms = now_ms;
        }

        let slow_interval = self.config.slow_interval_ms();
        let elapsed_slow = now_ms - self.last_slow_ms;
        if elapsed_slow >= slow_interval {
            ready.push((LoopCadence::Slow, elapsed_slow));
            self.last_slow_ms = now_ms;
        }

        ready
    }

    /// Record that a loop tick completed with the given duration.
    pub fn record_tick(&mut self, cadence: LoopCadence, duration_ms: f64) {
        match cadence {
            LoopCadence::Fast => {
                self.fast_metrics
                    .record_tick(duration_ms, self.config.fast_budget_ms as f64);
            }
            LoopCadence::Medium => {
                self.medium_metrics
                    .record_tick(duration_ms, self.config.medium_budget_ms as f64);
            }
            LoopCadence::Slow => {
                self.slow_metrics
                    .record_tick(duration_ms, self.config.slow_budget_ms as f64);
            }
        }
    }

    /// Get metrics for a specific loop.
    pub fn metrics(&self, cadence: LoopCadence) -> &LoopMetrics {
        match cadence {
            LoopCadence::Fast => &self.fast_metrics,
            LoopCadence::Medium => &self.medium_metrics,
            LoopCadence::Slow => &self.slow_metrics,
        }
    }

    /// Get config reference.
    pub fn config(&self) -> &LoopConfig {
        &self.config
    }

    /// Check if any loop is over budget (has at least one overrun).
    pub fn any_overrun(&self) -> bool {
        self.fast_metrics.budget_overruns > 0
            || self.medium_metrics.budget_overruns > 0
            || self.slow_metrics.budget_overruns > 0
    }

    /// Get total overruns across all loops.
    pub fn total_overruns(&self) -> u64 {
        self.fast_metrics.budget_overruns
            + self.medium_metrics.budget_overruns
            + self.slow_metrics.budget_overruns
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- LoopConfig tests ----

    #[test]
    fn test_config_defaults() {
        let cfg = LoopConfig::default();
        assert!((cfg.fast_target_hz - 60.0).abs() < f32::EPSILON);
        assert!((cfg.medium_target_hz - 5.0).abs() < f32::EPSILON);
        assert!((cfg.slow_target_hz - 0.5).abs() < f32::EPSILON);
        assert!((cfg.fast_budget_ms - 16.0).abs() < f32::EPSILON);
        assert!((cfg.medium_budget_ms - 50.0).abs() < f32::EPSILON);
        assert!((cfg.slow_budget_ms - 500.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_config_intervals() {
        let cfg = LoopConfig::default();
        // 60 Hz -> ~16.667 ms
        assert!((cfg.fast_interval_ms() - 16.6667).abs() < 0.1);
        // 5 Hz -> 200 ms
        assert!((cfg.medium_interval_ms() - 200.0).abs() < 0.1);
        // 0.5 Hz -> 2000 ms
        assert!((cfg.slow_interval_ms() - 2000.0).abs() < 0.1);
    }

    #[test]
    fn test_config_zero_hz_returns_max() {
        assert_eq!(LoopConfig::interval_ms(0.0), f64::MAX);
        assert_eq!(LoopConfig::interval_ms(-1.0), f64::MAX);
    }

    // ---- LoopMetrics tests ----

    #[test]
    fn test_metrics_default() {
        let m = LoopMetrics::default();
        assert_eq!(m.tick_count, 0);
        assert!((m.total_time_ms - 0.0).abs() < f64::EPSILON);
        assert!((m.last_tick_ms - 0.0).abs() < f64::EPSILON);
        assert!((m.max_tick_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.budget_overruns, 0);
        assert!((m.avg_tick_ms - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_record_tick_within_budget() {
        let mut m = LoopMetrics::default();
        m.record_tick(10.0, 16.0);
        assert_eq!(m.tick_count, 1);
        assert!((m.total_time_ms - 10.0).abs() < f64::EPSILON);
        assert!((m.last_tick_ms - 10.0).abs() < f64::EPSILON);
        assert!((m.max_tick_ms - 10.0).abs() < f64::EPSILON);
        assert_eq!(m.budget_overruns, 0);
        assert!((m.avg_tick_ms - 10.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_record_tick_over_budget() {
        let mut m = LoopMetrics::default();
        m.record_tick(20.0, 16.0);
        assert_eq!(m.budget_overruns, 1);

        m.record_tick(10.0, 16.0);
        assert_eq!(m.budget_overruns, 1); // no new overrun

        m.record_tick(17.0, 16.0);
        assert_eq!(m.budget_overruns, 2);
    }

    #[test]
    fn test_metrics_accumulation() {
        let mut m = LoopMetrics::default();
        m.record_tick(10.0, 50.0);
        m.record_tick(20.0, 50.0);
        m.record_tick(30.0, 50.0);
        assert_eq!(m.tick_count, 3);
        assert!((m.total_time_ms - 60.0).abs() < f64::EPSILON);
        assert!((m.last_tick_ms - 30.0).abs() < f64::EPSILON);
        assert!((m.max_tick_ms - 30.0).abs() < f64::EPSILON);
        assert!((m.avg_tick_ms - 20.0).abs() < f64::EPSILON);
        assert_eq!(m.budget_overruns, 0);
    }

    #[test]
    fn test_metrics_max_tracking() {
        let mut m = LoopMetrics::default();
        m.record_tick(5.0, 100.0);
        m.record_tick(50.0, 100.0);
        m.record_tick(25.0, 100.0);
        assert!((m.max_tick_ms - 50.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_metrics_reset() {
        let mut m = LoopMetrics::default();
        m.record_tick(10.0, 16.0);
        m.record_tick(20.0, 16.0);
        m.reset();
        assert_eq!(m.tick_count, 0);
        assert!((m.total_time_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(m.budget_overruns, 0);
    }

    // ---- LoopScheduler tests ----

    #[test]
    fn test_scheduler_with_defaults() {
        let s = LoopScheduler::with_defaults();
        assert!((s.config().fast_target_hz - 60.0).abs() < f32::EPSILON);
        assert!((s.config().medium_target_hz - 5.0).abs() < f32::EPSILON);
        assert!((s.config().slow_target_hz - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_fast_loop_fires_at_correct_interval() {
        let mut s = LoopScheduler::with_defaults();

        // At t=0 all loops fire (elapsed from 0.0 >= interval)
        let ready = s.poll(0.0);
        // At t=0, elapsed is 0 for all, no loop should fire since 0 < interval
        assert!(ready.is_empty());

        // Fast interval is ~16.67 ms. At t=16.67 fast should fire.
        let ready = s.poll(16.67);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, LoopCadence::Fast);
        assert!((ready[0].1 - 16.67).abs() < 0.1);

        // At t=33.34 fast should fire again
        let ready = s.poll(33.34);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].0, LoopCadence::Fast);
    }

    #[test]
    fn test_medium_loop_fires_at_correct_interval() {
        let mut s = LoopScheduler::with_defaults();
        // Consume initial poll
        let _ = s.poll(0.0);

        // Medium interval is 200 ms. At t=100 only fast should fire.
        let ready = s.poll(100.0);
        for (cadence, _) in &ready {
            assert_ne!(*cadence, LoopCadence::Medium);
        }

        // At t=200 medium should fire (along with fast)
        let ready = s.poll(200.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Medium));
        assert!(cadences.contains(&LoopCadence::Fast));
    }

    #[test]
    fn test_slow_loop_fires_at_correct_interval() {
        let mut s = LoopScheduler::with_defaults();
        let _ = s.poll(0.0);

        // Slow interval is 2000 ms. At t=1000 slow should not fire.
        let ready = s.poll(1000.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(!cadences.contains(&LoopCadence::Slow));

        // At t=2000 slow should fire
        let ready = s.poll(2000.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Slow));
    }

    #[test]
    fn test_all_loops_fire_together() {
        let mut s = LoopScheduler::with_defaults();
        // Jump far enough that all intervals are exceeded
        let ready = s.poll(3000.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Fast));
        assert!(cadences.contains(&LoopCadence::Medium));
        assert!(cadences.contains(&LoopCadence::Slow));
    }

    #[test]
    fn test_no_double_fire_without_elapsed() {
        let mut s = LoopScheduler::with_defaults();
        // Fire at t=3000
        let _ = s.poll(3000.0);
        // Poll again at the same time -- nothing should fire
        let ready = s.poll(3000.0);
        assert!(ready.is_empty());
    }

    #[test]
    fn test_budget_overrun_tracking() {
        let mut s = LoopScheduler::with_defaults();
        assert!(!s.any_overrun());
        assert_eq!(s.total_overruns(), 0);

        // Record a fast tick that exceeds the 16ms budget
        s.record_tick(LoopCadence::Fast, 20.0);
        assert!(s.any_overrun());
        assert_eq!(s.total_overruns(), 1);

        // Record a medium tick within budget
        s.record_tick(LoopCadence::Medium, 30.0);
        assert_eq!(s.total_overruns(), 1);

        // Record a medium tick over budget
        s.record_tick(LoopCadence::Medium, 60.0);
        assert_eq!(s.total_overruns(), 2);

        // Record a slow tick over budget
        s.record_tick(LoopCadence::Slow, 600.0);
        assert_eq!(s.total_overruns(), 3);
    }

    #[test]
    fn test_metrics_per_cadence() {
        let mut s = LoopScheduler::with_defaults();
        s.record_tick(LoopCadence::Fast, 10.0);
        s.record_tick(LoopCadence::Fast, 12.0);
        s.record_tick(LoopCadence::Medium, 40.0);

        let fast = s.metrics(LoopCadence::Fast);
        assert_eq!(fast.tick_count, 2);
        assert!((fast.avg_tick_ms - 11.0).abs() < f64::EPSILON);

        let medium = s.metrics(LoopCadence::Medium);
        assert_eq!(medium.tick_count, 1);
        assert!((medium.last_tick_ms - 40.0).abs() < f64::EPSILON);

        let slow = s.metrics(LoopCadence::Slow);
        assert_eq!(slow.tick_count, 0);
    }

    #[test]
    fn test_custom_config() {
        let cfg = LoopConfig {
            fast_target_hz: 30.0,
            medium_target_hz: 2.0,
            slow_target_hz: 0.1,
            fast_budget_ms: 33.0,
            medium_budget_ms: 100.0,
            slow_budget_ms: 1000.0,
        };
        let mut s = LoopScheduler::new(cfg);

        // 30 Hz -> interval ~33.33 ms
        let ready = s.poll(33.34);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Fast));
        assert!(!cadences.contains(&LoopCadence::Medium)); // 500ms interval

        // At 500ms, medium should also fire
        let ready = s.poll(500.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Medium));
    }

    #[test]
    fn test_scheduler_rapid_polls() {
        let mut s = LoopScheduler::with_defaults();
        // Rapid polls at 1ms increments -- fast loop should not fire until ~16.67ms
        for ms in 1..16 {
            let ready = s.poll(ms as f64);
            for (cadence, _) in &ready {
                assert_ne!(*cadence, LoopCadence::Fast);
            }
        }
        // At 17ms, fast should fire
        let ready = s.poll(17.0);
        let cadences: Vec<LoopCadence> = ready.iter().map(|(c, _)| *c).collect();
        assert!(cadences.contains(&LoopCadence::Fast));
    }
}
