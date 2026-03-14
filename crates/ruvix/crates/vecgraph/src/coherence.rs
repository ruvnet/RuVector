//! Coherence tracking for kernel vector and graph stores.
//!
//! Coherence metadata is co-located with each vector and graph node,
//! enabling the scheduler and proof engine to make informed decisions.
//!
//! # Design (from ADR-087 Section 4.3)
//!
//! - coherence_score: Structural consistency score (0.0-1.0)
//! - last_mutation_epoch: Epoch of the last mutation
//! - proof_attestation_hash: Hash of the proof that authorized the last mutation
//!
//! The coherence-aware scheduler uses these signals to:
//! 1. Prioritize tasks processing genuinely new information
//! 2. Deprioritize tasks whose mutations would lower coherence
//! 3. Fast-path mutations within coherent partitions

use ruvix_types::CoherenceMeta;

/// Configuration for coherence tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct CoherenceConfig {
    /// Minimum acceptable coherence score (0-10000 = 0.0-1.0).
    /// Operations that would lower coherence below this threshold may be rejected.
    pub min_coherence_threshold: u16,

    /// Whether to enable coherence-aware scheduling hints.
    pub enable_scheduler_hints: bool,

    /// Whether to track coherence deltas for mutation planning.
    pub track_deltas: bool,

    /// Decay rate for coherence over time (0-10000 = 0.0-1.0 per epoch).
    /// Set to 0 for no decay.
    pub decay_rate: u16,

    /// Initial coherence score for new entries.
    pub initial_coherence: u16,
}

impl Default for CoherenceConfig {
    fn default() -> Self {
        Self {
            min_coherence_threshold: 5000, // 0.5
            enable_scheduler_hints: true,
            track_deltas: false,
            decay_rate: 0,
            initial_coherence: 10000, // 1.0 = fully coherent
        }
    }
}

impl CoherenceConfig {
    /// Creates a new coherence configuration.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            min_coherence_threshold: 5000,
            enable_scheduler_hints: true,
            track_deltas: false,
            decay_rate: 0,
            initial_coherence: 10000,
        }
    }

    /// Sets the minimum coherence threshold.
    #[inline]
    #[must_use]
    pub const fn with_min_threshold(mut self, threshold: f32) -> Self {
        self.min_coherence_threshold = (threshold.clamp(0.0, 1.0) * 10000.0) as u16;
        self
    }

    /// Enables or disables scheduler hints.
    #[inline]
    #[must_use]
    pub const fn with_scheduler_hints(mut self, enabled: bool) -> Self {
        self.enable_scheduler_hints = enabled;
        self
    }

    /// Enables or disables delta tracking.
    #[inline]
    #[must_use]
    pub const fn with_delta_tracking(mut self, enabled: bool) -> Self {
        self.track_deltas = enabled;
        self
    }

    /// Sets the coherence decay rate.
    #[inline]
    #[must_use]
    pub const fn with_decay_rate(mut self, rate: f32) -> Self {
        self.decay_rate = (rate.clamp(0.0, 1.0) * 10000.0) as u16;
        self
    }

    /// Returns the minimum threshold as a float.
    #[inline]
    #[must_use]
    pub fn min_threshold_f32(&self) -> f32 {
        self.min_coherence_threshold as f32 / 10000.0
    }
}

/// Tracks coherence state for a collection of entries.
#[derive(Debug, Clone)]
pub struct CoherenceTracker {
    /// Configuration for coherence tracking.
    config: CoherenceConfig,

    /// Global epoch counter for mutations.
    current_epoch: u64,

    /// Rolling average coherence score.
    average_coherence: u16,

    /// Number of entries tracked.
    entry_count: u32,

    /// Sum of all coherence scores (for average calculation).
    coherence_sum: u64,

    /// Number of mutations below threshold.
    low_coherence_mutations: u32,
}

impl CoherenceTracker {
    /// Creates a new coherence tracker with the given configuration.
    #[inline]
    #[must_use]
    pub const fn new(config: CoherenceConfig) -> Self {
        Self {
            config,
            current_epoch: 0,
            average_coherence: 10000,
            entry_count: 0,
            coherence_sum: 0,
            low_coherence_mutations: 0,
        }
    }

    /// Returns the current epoch.
    #[inline]
    #[must_use]
    pub const fn current_epoch(&self) -> u64 {
        self.current_epoch
    }

    /// Advances the epoch and returns the new value.
    #[inline]
    pub fn advance_epoch(&mut self) -> u64 {
        self.current_epoch = self.current_epoch.wrapping_add(1);
        self.current_epoch
    }

    /// Returns the configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &CoherenceConfig {
        &self.config
    }

    /// Creates initial coherence metadata for a new entry.
    ///
    /// Note: Uses current_epoch + 1 to ensure new entries always have
    /// a non-zero mutation_epoch, signifying they've been mutated at least once.
    #[inline]
    #[must_use]
    pub fn create_initial_meta(&mut self, proof_attestation_hash: [u8; 32]) -> CoherenceMeta {
        CoherenceMeta::new(
            self.config.initial_coherence,
            self.advance_epoch(),
            proof_attestation_hash,
        )
    }

    /// Updates coherence tracking when an entry is added.
    pub fn on_entry_added(&mut self, coherence_score: u16) {
        self.entry_count = self.entry_count.saturating_add(1);
        self.coherence_sum = self.coherence_sum.saturating_add(coherence_score as u64);
        self.update_average();
    }

    /// Updates coherence tracking when an entry is removed.
    pub fn on_entry_removed(&mut self, coherence_score: u16) {
        self.entry_count = self.entry_count.saturating_sub(1);
        self.coherence_sum = self.coherence_sum.saturating_sub(coherence_score as u64);
        self.update_average();
    }

    /// Updates coherence tracking when an entry is mutated.
    ///
    /// Returns the new coherence metadata for the entry.
    pub fn on_entry_mutated(
        &mut self,
        old_meta: &CoherenceMeta,
        new_coherence: u16,
        proof_attestation_hash: [u8; 32],
    ) -> CoherenceMeta {
        // Update sum
        self.coherence_sum = self
            .coherence_sum
            .saturating_sub(old_meta.coherence_score as u64)
            .saturating_add(new_coherence as u64);

        self.update_average();

        // Track low coherence mutations
        if new_coherence < self.config.min_coherence_threshold {
            self.low_coherence_mutations = self.low_coherence_mutations.saturating_add(1);
        }

        // Create new metadata
        CoherenceMeta::new(new_coherence, self.advance_epoch(), proof_attestation_hash)
    }

    /// Checks if a proposed coherence change would violate constraints.
    #[inline]
    #[must_use]
    pub fn would_violate_threshold(&self, proposed_coherence: u16) -> bool {
        proposed_coherence < self.config.min_coherence_threshold
    }

    /// Returns the average coherence score as a float.
    #[inline]
    #[must_use]
    pub fn average_coherence_f32(&self) -> f32 {
        self.average_coherence as f32 / 10000.0
    }

    /// Returns the number of entries tracked.
    #[inline]
    #[must_use]
    pub const fn entry_count(&self) -> u32 {
        self.entry_count
    }

    /// Returns the number of low-coherence mutations.
    #[inline]
    #[must_use]
    pub const fn low_coherence_mutations(&self) -> u32 {
        self.low_coherence_mutations
    }

    /// Updates the rolling average coherence.
    fn update_average(&mut self) {
        if self.entry_count > 0 {
            self.average_coherence = (self.coherence_sum / self.entry_count as u64) as u16;
        } else {
            self.average_coherence = self.config.initial_coherence;
        }
    }

    /// Applies coherence decay based on elapsed epochs.
    ///
    /// This is called periodically to age coherence scores.
    pub fn apply_decay(&mut self, epochs_elapsed: u64) {
        if self.config.decay_rate == 0 || epochs_elapsed == 0 {
            return;
        }

        // Calculate decay factor: (1 - decay_rate) ^ epochs_elapsed
        // Simplified: subtract decay_rate * epochs_elapsed from average
        let decay_amount =
            ((self.config.decay_rate as u64) * epochs_elapsed).min(self.average_coherence as u64);

        self.average_coherence = self.average_coherence.saturating_sub(decay_amount as u16);

        // Also decay the sum proportionally
        if self.entry_count > 0 {
            let sum_decay = decay_amount * self.entry_count as u64;
            self.coherence_sum = self.coherence_sum.saturating_sub(sum_decay);
        }
    }
}

impl Default for CoherenceTracker {
    fn default() -> Self {
        Self::new(CoherenceConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coherence_config_defaults() {
        let config = CoherenceConfig::default();
        assert_eq!(config.initial_coherence, 10000);
        assert_eq!(config.min_coherence_threshold, 5000);
        assert!(config.enable_scheduler_hints);
    }

    #[test]
    fn test_coherence_config_builder() {
        let config = CoherenceConfig::new()
            .with_min_threshold(0.7)
            .with_decay_rate(0.01)
            .with_delta_tracking(true);

        assert_eq!(config.min_coherence_threshold, 7000);
        assert_eq!(config.decay_rate, 100);
        assert!(config.track_deltas);
    }

    #[test]
    fn test_coherence_tracker_epoch() {
        let mut tracker = CoherenceTracker::default();

        assert_eq!(tracker.current_epoch(), 0);

        let epoch1 = tracker.advance_epoch();
        assert_eq!(epoch1, 1);

        let epoch2 = tracker.advance_epoch();
        assert_eq!(epoch2, 2);
    }

    #[test]
    fn test_coherence_tracker_average() {
        let mut tracker = CoherenceTracker::default();

        // Add entries with different coherence
        tracker.on_entry_added(10000); // 1.0
        tracker.on_entry_added(8000); // 0.8
        tracker.on_entry_added(6000); // 0.6

        // Average should be (10000 + 8000 + 6000) / 3 = 8000
        assert_eq!(tracker.entry_count(), 3);
        assert!((tracker.average_coherence_f32() - 0.8).abs() < 0.01);
    }

    #[test]
    fn test_coherence_tracker_mutation() {
        let mut tracker = CoherenceTracker::default();

        let initial_meta = tracker.create_initial_meta([0u8; 32]);
        tracker.on_entry_added(initial_meta.coherence_score);

        assert_eq!(initial_meta.coherence_score, 10000);
        assert_eq!(initial_meta.mutation_epoch, 1); // Epoch 1 for new entry

        let new_meta = tracker.on_entry_mutated(&initial_meta, 9000, [1u8; 32]);

        assert_eq!(new_meta.coherence_score, 9000);
        assert_eq!(new_meta.mutation_epoch, 2); // Epoch 2 after mutation
        assert_eq!(new_meta.proof_attestation_hash, [1u8; 32]);
    }

    #[test]
    fn test_coherence_threshold_violation() {
        let tracker = CoherenceTracker::new(CoherenceConfig::new().with_min_threshold(0.5));

        assert!(!tracker.would_violate_threshold(6000)); // 0.6 > 0.5
        assert!(tracker.would_violate_threshold(4000)); // 0.4 < 0.5
    }

    #[test]
    fn test_coherence_decay() {
        let config = CoherenceConfig::new().with_decay_rate(0.1); // 10% decay per epoch
        let mut tracker = CoherenceTracker::new(config);

        tracker.on_entry_added(10000);
        assert_eq!(tracker.average_coherence, 10000);

        tracker.apply_decay(1);
        // Should decay by 10% = 1000
        assert_eq!(tracker.average_coherence, 9000);
    }
}
