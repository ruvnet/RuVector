//! Novelty tracking for coherence-aware scheduling.
//!
//! The novelty tracker measures how different new inputs are from recent inputs,
//! providing a priority boost for tasks processing genuinely new information.
//! This prevents the system from starving exploration in favor of exploitation.
//!
//! # Algorithm
//!
//! Novelty is computed as the normalized distance from a new input vector to the
//! centroid of recent inputs:
//!
//! ```text
//! novelty = clamp(distance(input, centroid) / max_distance, 0.0, 1.0)
//! ```
//!
//! The centroid is maintained as an exponential moving average of recent inputs.

// Math helper for no_std
#[inline]
fn sqrt_f32(x: f32) -> f32 {
    // Newton-Raphson method for square root
    if x <= 0.0 {
        return 0.0;
    }
    let mut guess = x;
    for _ in 0..8 {
        guess = (guess + x / guess) * 0.5;
    }
    guess
}

#[inline]
fn tanh_f32(x: f32) -> f32 {
    // Approximation of tanh using exp approximation
    if x > 4.0 {
        return 1.0;
    }
    if x < -4.0 {
        return -1.0;
    }
    // tanh(x) = (e^2x - 1) / (e^2x + 1)
    let e2x = exp_f32(2.0 * x);
    (e2x - 1.0) / (e2x + 1.0)
}

#[inline]
fn exp_f32(x: f32) -> f32 {
    // Fast approximation of exp for small values
    if x > 10.0 {
        return f32::MAX;
    }
    if x < -10.0 {
        return 0.0;
    }
    // Taylor series approximation
    let mut sum = 1.0f32;
    let mut term = 1.0f32;
    for i in 1..12 {
        term *= x / i as f32;
        sum += term;
    }
    sum
}

/// Configuration for the novelty tracker.
#[derive(Debug, Clone, Copy)]
pub struct NoveltyConfig {
    /// Dimensionality of input vectors.
    pub dimensions: usize,

    /// Exponential moving average factor for centroid updates.
    /// Higher values give more weight to recent inputs.
    pub ema_alpha: f32,

    /// Maximum expected distance (used for normalization).
    pub max_distance: f32,

    /// Minimum novelty threshold (values below this are treated as zero).
    pub min_novelty_threshold: f32,

    /// Decay factor per tick for novelty values.
    pub decay_factor: f32,
}

impl Default for NoveltyConfig {
    fn default() -> Self {
        Self {
            dimensions: 768, // Common embedding dimension
            ema_alpha: 0.1,
            max_distance: 10.0,
            min_novelty_threshold: 0.01,
            decay_factor: 0.95,
        }
    }
}

impl NoveltyConfig {
    /// Creates a configuration for a specific dimensionality.
    #[inline]
    #[must_use]
    pub const fn with_dimensions(mut self, dim: usize) -> Self {
        self.dimensions = dim;
        self
    }

    /// Creates a configuration with a custom EMA alpha.
    #[inline]
    #[must_use]
    pub const fn with_ema_alpha(mut self, alpha: f32) -> Self {
        self.ema_alpha = alpha;
        self
    }

    /// Creates a configuration with a custom max distance.
    #[inline]
    #[must_use]
    pub const fn with_max_distance(mut self, dist: f32) -> Self {
        self.max_distance = dist;
        self
    }
}

/// Novelty tracker using a centroid-based approach.
///
/// This tracker maintains an exponential moving average of recent input vectors
/// and computes novelty as the distance from new inputs to this centroid.
///
/// # Type Parameters
///
/// * `N` - Maximum dimensionality of input vectors (compile-time constant).
///
/// # Note
///
/// In a no_std environment, the centroid is stored in a fixed-size array.
/// When the `alloc` feature is enabled, a dynamically-sized vector is used.
#[derive(Debug)]
pub struct NoveltyTracker<const N: usize> {
    /// Configuration for the tracker.
    config: NoveltyConfig,

    /// Running centroid (EMA of recent inputs).
    centroid: [f32; N],

    /// Number of inputs seen (for initialization).
    input_count: u64,

    /// Running variance estimate for adaptive max distance.
    variance_estimate: f32,

    /// Whether the tracker has been initialized with at least one input.
    initialized: bool,
}

impl<const N: usize> NoveltyTracker<N> {
    /// Creates a new novelty tracker with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if `config.dimensions > N`.
    #[must_use]
    pub fn new(config: NoveltyConfig) -> Self {
        assert!(
            config.dimensions <= N,
            "dimensions {} exceeds maximum {}",
            config.dimensions,
            N
        );

        Self {
            config,
            centroid: [0.0; N],
            input_count: 0,
            variance_estimate: config.max_distance,
            initialized: false,
        }
    }

    /// Creates a new novelty tracker with default configuration.
    #[inline]
    #[must_use]
    pub fn with_default_config() -> Self {
        Self::new(NoveltyConfig::default().with_dimensions(N.min(768)))
    }

    /// Computes the novelty score for an input vector.
    ///
    /// Returns a value in the range `[0.0, 1.0]`, where higher values indicate
    /// more novel inputs.
    ///
    /// # Arguments
    ///
    /// * `input` - The input vector to evaluate.
    ///
    /// # Returns
    ///
    /// The novelty score, or 1.0 if this is the first input.
    pub fn compute_novelty(&self, input: &[f32]) -> f32 {
        if !self.initialized {
            // First input is always maximally novel
            return 1.0;
        }

        let distance = self.euclidean_distance(input);

        // Treat very small distances as zero (floating-point precision threshold)
        if distance < 0.01 {
            return 0.0;
        }

        let normalized = distance / self.variance_estimate.max(0.001);

        // Apply sigmoid-like clamping for smooth transitions
        let novelty = tanh_f32(normalized * 2.0);

        if novelty < self.config.min_novelty_threshold {
            0.0
        } else {
            novelty.clamp(0.0, 1.0)
        }
    }

    /// Updates the tracker with a new input vector.
    ///
    /// This updates the running centroid using exponential moving average.
    ///
    /// # Arguments
    ///
    /// * `input` - The input vector to incorporate.
    pub fn update(&mut self, input: &[f32]) {
        let dims = self.config.dimensions.min(input.len()).min(N);

        if !self.initialized {
            // Initialize centroid with first input
            for (i, &val) in input.iter().take(dims).enumerate() {
                self.centroid[i] = val;
            }
            self.initialized = true;
            self.input_count = 1;
            return;
        }

        // Compute distance before update (for variance estimation)
        let distance = self.euclidean_distance(input);

        // Update centroid using EMA
        let alpha = self.config.ema_alpha;
        for (i, &val) in input.iter().take(dims).enumerate() {
            self.centroid[i] = (1.0 - alpha) * self.centroid[i] + alpha * val;
        }

        // Update variance estimate using EMA
        self.variance_estimate =
            (1.0 - alpha) * self.variance_estimate + alpha * distance.max(0.001);

        self.input_count = self.input_count.saturating_add(1);
    }

    /// Computes novelty and updates the tracker in one step.
    ///
    /// This is more efficient than calling `compute_novelty` and `update` separately.
    pub fn process(&mut self, input: &[f32]) -> f32 {
        let novelty = self.compute_novelty(input);
        self.update(input);
        novelty
    }

    /// Returns the current centroid.
    #[inline]
    #[must_use]
    pub fn centroid(&self) -> &[f32] {
        &self.centroid[..self.config.dimensions.min(N)]
    }

    /// Returns the number of inputs processed.
    #[inline]
    #[must_use]
    pub const fn input_count(&self) -> u64 {
        self.input_count
    }

    /// Returns true if the tracker has been initialized.
    #[inline]
    #[must_use]
    pub const fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Resets the tracker to its initial state.
    pub fn reset(&mut self) {
        self.centroid = [0.0; N];
        self.input_count = 0;
        self.variance_estimate = self.config.max_distance;
        self.initialized = false;
    }

    /// Returns the current variance estimate.
    #[inline]
    #[must_use]
    pub const fn variance_estimate(&self) -> f32 {
        self.variance_estimate
    }

    /// Computes the Euclidean distance from input to centroid.
    fn euclidean_distance(&self, input: &[f32]) -> f32 {
        let dims = self.config.dimensions.min(input.len()).min(N);
        let mut sum_sq = 0.0f32;

        for i in 0..dims {
            let diff = input[i] - self.centroid[i];
            sum_sq += diff * diff;
        }

        sqrt_f32(sum_sq)
    }
}

/// A novelty tracker optimized for small vectors (up to 16 dimensions).
pub type SmallNoveltyTracker = NoveltyTracker<16>;

/// A novelty tracker for medium vectors (up to 128 dimensions).
pub type MediumNoveltyTracker = NoveltyTracker<128>;

/// A novelty tracker for embedding vectors (up to 768 dimensions).
pub type EmbeddingNoveltyTracker = NoveltyTracker<768>;

/// A novelty tracker for large vectors (up to 1536 dimensions).
pub type LargeNoveltyTracker = NoveltyTracker<1536>;

/// Computes cosine similarity between two vectors.
///
/// Returns a value in the range `[-1.0, 1.0]`.
#[must_use]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let norm_product = (sqrt_f32(norm_a) * sqrt_f32(norm_b)).max(1e-10);
    dot / norm_product
}

/// Computes novelty from cosine similarity.
///
/// Maps cosine similarity `[-1, 1]` to novelty `[0, 1]`, where orthogonal
/// or opposite vectors are considered more novel.
#[inline]
#[must_use]
pub fn novelty_from_similarity(similarity: f32) -> f32 {
    ((1.0 - similarity) / 2.0).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_novelty_tracker_creation() {
        let tracker: NoveltyTracker<64> =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(64));

        assert!(!tracker.is_initialized());
        assert_eq!(tracker.input_count(), 0);
    }

    #[test]
    fn test_first_input_is_novel() {
        let tracker: NoveltyTracker<8> =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(8));

        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let novelty = tracker.compute_novelty(&input);

        assert!((novelty - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_same_input_not_novel() {
        let mut tracker: NoveltyTracker<4> =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(4));

        let input = [1.0, 2.0, 3.0, 4.0];

        // Process first input
        let first_novelty = tracker.process(&input);
        assert!((first_novelty - 1.0).abs() < 0.001);

        // Same input should be less novel
        let second_novelty = tracker.process(&input);
        assert!(
            second_novelty < first_novelty,
            "second {} < first {}",
            second_novelty,
            first_novelty
        );

        // After many identical inputs, novelty should be very low
        for _ in 0..100 {
            tracker.update(&input);
        }
        let final_novelty = tracker.compute_novelty(&input);
        assert!(
            final_novelty < 0.1,
            "final novelty {} should be low",
            final_novelty
        );
    }

    #[test]
    fn test_different_input_is_novel() {
        let mut tracker: NoveltyTracker<4> =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(4));

        let input1 = [1.0, 0.0, 0.0, 0.0];
        let input2 = [0.0, 0.0, 0.0, 10.0]; // Very different

        tracker.update(&input1);

        let novelty = tracker.compute_novelty(&input2);
        assert!(novelty > 0.5, "novelty {} should be high", novelty);
    }

    #[test]
    fn test_centroid_update() {
        let mut tracker: NoveltyTracker<4> = NoveltyTracker::new(
            NoveltyConfig::default()
                .with_dimensions(4)
                .with_ema_alpha(0.5),
        );

        tracker.update(&[1.0, 0.0, 0.0, 0.0]);
        tracker.update(&[0.0, 1.0, 0.0, 0.0]);

        let centroid = tracker.centroid();

        // With alpha=0.5, centroid should be mix of both inputs
        assert!(centroid[0] > 0.0 && centroid[0] < 1.0);
        assert!(centroid[1] > 0.0 && centroid[1] < 1.0);
    }

    #[test]
    fn test_reset() {
        let mut tracker: NoveltyTracker<4> =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(4));

        tracker.update(&[1.0, 2.0, 3.0, 4.0]);
        assert!(tracker.is_initialized());

        tracker.reset();
        assert!(!tracker.is_initialized());
        assert_eq!(tracker.input_count(), 0);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = [1.0, 0.0, 0.0, 0.0];
        let b = [1.0, 0.0, 0.0, 0.0];

        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 0.001);

        let c = [-1.0, 0.0, 0.0, 0.0];
        let sim_opposite = cosine_similarity(&a, &c);
        assert!((sim_opposite + 1.0).abs() < 0.001);

        let d = [0.0, 1.0, 0.0, 0.0];
        let sim_orthogonal = cosine_similarity(&a, &d);
        assert!(sim_orthogonal.abs() < 0.001);
    }

    #[test]
    fn test_novelty_from_similarity() {
        // Identical vectors (similarity = 1) -> novelty = 0
        assert!((novelty_from_similarity(1.0) - 0.0).abs() < 0.001);

        // Opposite vectors (similarity = -1) -> novelty = 1
        assert!((novelty_from_similarity(-1.0) - 1.0).abs() < 0.001);

        // Orthogonal vectors (similarity = 0) -> novelty = 0.5
        assert!((novelty_from_similarity(0.0) - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_type_aliases() {
        let _small: SmallNoveltyTracker =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(16));
        let _medium: MediumNoveltyTracker =
            NoveltyTracker::new(NoveltyConfig::default().with_dimensions(128));
    }
}
