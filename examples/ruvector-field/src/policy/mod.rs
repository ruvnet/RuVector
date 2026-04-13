//! Policy registry and axis constraints.
//!
//! A [`Policy`] declares a bitmask and per-axis floors / ceilings. The
//! registry's [`PolicyRegistry::policy_fit`] returns the product of axis
//! constraint satisfaction scores for a node, and `policy_risk` is the
//! complement.
//!
//! # Example
//!
//! ```
//! use ruvector_field::policy::{AxisConstraint, AxisConstraints, Policy, PolicyRegistry};
//! use ruvector_field::model::AxisScores;
//! let mut reg = PolicyRegistry::new();
//! reg.register(Policy {
//!     id: 1,
//!     name: "safety".into(),
//!     mask: 0b0001,
//!     required_axes: AxisConstraints {
//!         limit: AxisConstraint::min(0.5),
//!         care: AxisConstraint::min(0.5),
//!         bridge: AxisConstraint::any(),
//!         clarity: AxisConstraint::min(0.3),
//!     },
//! });
//! let axes = AxisScores::new(0.8, 0.7, 0.4, 0.9);
//! let fit = reg.policy_fit(&axes, 0b0001);
//! assert!(fit > 0.9);
//! ```

pub mod registry;

pub use registry::{AxisConstraint, AxisConstraints, Policy, PolicyRegistry};
