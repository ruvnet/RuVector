//! Generic feature vector trait for EML model inputs.
//!
//! Any struct that can produce a slice of `f64` values can implement
//! [`FeatureVector`] to be used as input to an [`EmlModel`].

/// Trait for types that can produce a fixed-length feature vector
/// suitable as EML model input.
///
/// Implementors should normalize features to roughly [0, 1] for
/// best numerical stability.
///
/// # Example
///
/// ```
/// use eml_core::FeatureVector;
///
/// struct SensorReading {
///     temperature: f64,
///     humidity: f64,
///     pressure: f64,
/// }
///
/// impl FeatureVector for SensorReading {
///     fn as_features(&self) -> Vec<f64> {
///         vec![
///             self.temperature / 100.0,  // normalize to ~[0,1]
///             self.humidity / 100.0,
///             self.pressure / 1100.0,
///         ]
///     }
///
///     fn feature_count() -> usize {
///         3
///     }
/// }
/// ```
pub trait FeatureVector {
    /// Return the feature values as a Vec of f64.
    ///
    /// Values should be normalized to roughly [0, 1] for best
    /// training convergence.
    fn as_features(&self) -> Vec<f64>;

    /// The number of features this type produces.
    ///
    /// Must be constant for all instances of the same type.
    fn feature_count() -> usize;
}
