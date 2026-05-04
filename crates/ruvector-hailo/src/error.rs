//! Error type for the Hailo embedding backend.

use thiserror::Error;

/// All failure modes surfaced by `ruvector-hailo`. Maps cleanly onto
/// `ruvector_core::EmbeddingError` once that trait lands (iteration 2).
#[derive(Debug, Error)]
pub enum HailoError {
    /// The crate was built without the `hailo` feature; the NPU path is
    /// not compiled in. Build with `--features hailo` on a Pi 5 + AI HAT+.
    #[error(
        "ruvector-hailo built without `hailo` feature — recompile with \
         `--features hailo` on a Pi 5 + AI HAT+"
    )]
    FeatureDisabled,

    /// Iteration N hasn't landed yet for this code path. Should never
    /// reach a release build by the time the loop completes.
    #[error("not yet implemented: {0}")]
    NotYetImplemented(&'static str),

    /// `/dev/hailo*` not present or not enumerable; usually means the
    /// kernel `hailo_pci` module didn't load (no HAT, PCIe disabled,
    /// firmware missing).
    #[error("no Hailo device found: {0}")]
    NoDevice(String),

    /// HailoRT C library returned a non-success status. The numeric code
    /// matches the `hailo_status` enum in `hailort.h`.
    #[error("HailoRT error: status={status}, where={where_}")]
    Hailort { status: i32, where_: &'static str },

    /// Model dir layout missing a required artifact (HEF / vocab.txt).
    #[error("model directory `{path}` is missing `{what}`")]
    BadModelDir { path: String, what: &'static str },

    /// Tokenizer rejected the input (e.g. WordPiece vocab corrupt).
    #[error("tokenizer error: {0}")]
    Tokenizer(String),

    /// Output vector shape didn't match the configured `dim`.
    #[error("output shape mismatch: expected {expected}, got {actual}")]
    Shape { expected: usize, actual: usize },

    /// `HailoEmbedder::open` succeeded (vdevice is alive) but no
    /// HEF / model graph has been loaded into it yet — the worker
    /// can't perform inference. Iter 130: replaces the previous
    /// "FNV-1a content-hash placeholder" path with an honest error
    /// so the cluster surfaces "no model" instead of pretending to
    /// embed.
    ///
    /// Resolution: drop a compiled `model.hef` into the model dir
    /// (run the Hailo Dataflow Compiler against
    /// `sentence-transformers/all-MiniLM-L6-v2.onnx`) and restart
    /// the worker. The existing `HailoEmbedder::open` path picks it
    /// up; no source changes required.
    #[error(
        "no Hailo model graph loaded — drop a compiled `model.hef` into \
         the worker's model dir and restart"
    )]
    NoModelLoaded,
}
