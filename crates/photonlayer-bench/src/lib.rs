//! # PhotonLayer Bench
//!
//! Reproducible benchmarks plus the in-Rust mask **learner** and digital
//! **decoder** that turn the optical core into an end-to-end, trainable
//! hybrid system (ADR-260 Phase 2 & 4). Exposed as a library so the CLI and
//! examples can reuse the learner without duplicating it.
//!
//! Variants (ADR-260 §16.1): digital baseline, random optical mask, learned
//! optical mask. The headline, defensible claim is **not** state-of-the-art
//! accuracy but: *a learned optical frontend preserves task-useful information
//! while shrinking the sensor / decoder vs. a direct pixel pipeline.*

pub mod baselines;
pub mod decoder;
pub mod diffdetect;
pub mod learn;
pub mod mnist;
pub mod mnist_bench;
pub mod pipeline;
pub mod privacy;
pub mod synthetic;
pub mod verification;

pub use baselines::{run_classification, run_compression, BenchReport, VariantResult};
pub use decoder::{frame_features, NearestCentroid};
pub use diffdetect::{DiffDetector, Region};
pub use learn::{learn_mask, LearnConfig, LearnOutcome};
pub use mnist::{load_test, load_train, subset, MnistError, RawMnist, MNIST_CLASSES};
pub use mnist_bench::{run_mnist_differential, MnistBenchConfig, MnistBenchResult};
pub use privacy::{privacy_leakage, PrivacyReport};
pub use synthetic::{class_names, make_dataset, Sample, NUM_CLASSES};
pub use verification::{verify_eer, VerificationReport};
