//! EML (exp-ln) universal function approximation.
//!
//! This crate provides the EML operator and learning machinery for
//! O(1) learned functions from data. Based on Odrzywolel 2026,
//! "All elementary functions from a single operator".
//!
//! # Core Idea
//!
//! The EML operator `eml(x, y) = exp(x) - ln(y)` is the continuous-
//! mathematics analog of the NAND gate: combined with the constant 1,
//! it can reconstruct all elementary functions.
//!
//! # Components
//!
//! - [`eml`] / [`eml_safe`] / [`softmax3`] — primitive operators
//! - [`EmlTree`] — depth-configurable evaluation tree
//! - [`EmlModel`] — multi-head model with training
//! - [`FeatureVector`] — trait for types that produce `&[f64]` inputs
//!
//! # Example
//!
//! ```
//! use eml_core::EmlModel;
//!
//! // Create a depth-4 model with 3 inputs and 1 output head
//! let mut model = EmlModel::new(4, 3, 1);
//!
//! // Record training data (y = x0 + x1 + x2)
//! for i in 0..100 {
//!     let x = [i as f64 / 100.0, i as f64 / 50.0, i as f64 / 200.0];
//!     let y = x[0] + x[1] + x[2];
//!     model.record(&x, &[Some(y)]);
//! }
//!
//! // Train
//! let _converged = model.train();
//!
//! // Predict
//! let prediction = model.predict_primary(&[0.5, 1.0, 0.25]);
//! assert!(prediction.is_finite());
//! ```

pub mod features;
pub mod model;
pub mod operator;
pub mod tree;

// Re-export public API
pub use features::FeatureVector;
pub use model::EmlModel;
pub use operator::{eml, eml_safe, softmax3};
pub use tree::EmlTree;
