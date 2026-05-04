//! Cluster-side errors. iter-218 wired
//! `impl EmbeddingProvider for HailoClusterEmbedder`, which folds
//! `ClusterError → ruvector_core::RuvectorError::ModelInferenceError`
//! at the trait boundary; the iter-209 `is_terminal()` helper drives
//! the retry-loop short-circuit on deterministic gRPC statuses.

use thiserror::Error;

/// Errors surfaced by `HailoClusterEmbedder` and its supporting modules.
/// Distinguishes worker-level failures (`Transport`, `FingerprintMismatch`,
/// `DimMismatch`) from coordinator-level failures (`NoWorkers`,
/// `AllWorkersFailed`) so callers can map to user-facing categories.
#[derive(Debug, Error)]
pub enum ClusterError {
    /// Coordinator built with zero workers.
    #[error("HailoClusterEmbedder requires at least one worker")]
    NoWorkers,

    /// Iteration N hasn't landed for this code path.
    #[error("not yet implemented: {0}")]
    NotYetImplemented(
        /// Description of the missing functionality.
        &'static str,
    ),

    /// Every worker we tried failed (after retry budget exhausted).
    #[error("all workers failed: {0}")]
    AllWorkersFailed(
        /// Aggregated reason — typically the last seen error.
        String,
    ),

    /// Worker refused due to model fingerprint mismatch — never silently
    /// fan out across a heterogeneous fleet.
    #[error("worker {worker} fingerprint {actual} != expected {expected}")]
    FingerprintMismatch {
        /// Name of the worker (per `WorkerEndpoint::name`).
        worker: String,
        /// Fingerprint string the worker reports.
        actual: String,
        /// Fingerprint string the coordinator was configured to require.
        expected: String,
    },

    /// Transport-layer failure (gRPC connect / RPC error).
    #[error("transport error to {worker}: {reason}")]
    Transport {
        /// Worker name from the transport call site.
        worker: String,
        /// Free-form failure reason — gRPC status, IO error, etc.
        reason: String,
    },

    /// Worker returned a vector with the wrong dimensionality.
    #[error("worker {worker}: expected dim {expected}, got {actual}")]
    DimMismatch {
        /// Worker name from the dispatch call site.
        worker: String,
        /// Dimensionality the coordinator was configured to accept.
        expected: usize,
        /// Dimensionality the worker actually returned.
        actual: usize,
    },
}

impl ClusterError {
    /// Iter 209 — true when the error is deterministic and won't change
    /// on retry to the same fleet. Used by `HailoClusterEmbedder`'s
    /// dispatch loop to short-circuit instead of burning the entire
    /// retry budget on errors that will repeat verbatim.
    ///
    /// Recognized terminal statuses (string-matched on the wrapped
    /// gRPC `Status::code()` name, which tonic includes in
    /// `Display`):
    ///   * `OutOfRange`        — iter-180 byte cap or iter-190 encode cap
    ///   * `InvalidArgument`   — iter-199 batch-size cap
    ///   * `ResourceExhausted` — iter-104/200 rate limit denied
    ///
    /// Plus structural failures that retry can't fix:
    ///   * `DimMismatch` — coordinator/worker dim contract broken
    ///   * `FingerprintMismatch` — model drift; worker is wrong fleet
    ///
    /// Network-level failures (`Transport` with non-terminal status,
    /// timeouts, connect errors, etc.) are NOT terminal — those are
    /// the legitimate retry candidates.
    pub fn is_terminal(&self) -> bool {
        match self {
            ClusterError::DimMismatch { .. } | ClusterError::FingerprintMismatch { .. } => true,
            ClusterError::Transport { reason, .. } => {
                // tonic's Display for Status starts with `status: <Code>`.
                // Match conservatively — only the three deterministic
                // codes we know retry can't fix.
                reason.contains("status: OutOfRange")
                    || reason.contains("status: InvalidArgument")
                    || reason.contains("status: ResourceExhausted")
            }
            ClusterError::NoWorkers
            | ClusterError::NotYetImplemented(_)
            | ClusterError::AllWorkersFailed(_) => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transport(reason: &str) -> ClusterError {
        ClusterError::Transport {
            worker: "w".into(),
            reason: reason.into(),
        }
    }

    #[test]
    fn is_terminal_recognizes_byte_cap_rejection() {
        // iter-180 + iter-190 byte caps surface as OutOfRange.
        let e = transport(
            "embed RPC: status: OutOfRange, message: \"Error, decoded \
             message length too large: found 102432 bytes, the limit \
             is: 65536 bytes\"",
        );
        assert!(e.is_terminal());
    }

    #[test]
    fn is_terminal_recognizes_batch_cap_rejection() {
        // iter-199 batch-size cap surfaces as InvalidArgument.
        let e = transport(
            "embed_stream RPC: status: InvalidArgument, message: \
             \"batch size 300 exceeds max 256 (ADR-172 §3a iter 199)\"",
        );
        assert!(e.is_terminal());
    }

    #[test]
    fn is_terminal_recognizes_rate_limit_denial() {
        // iter-104/200 rate limit surfaces as ResourceExhausted.
        let e = transport(
            "embed RPC: status: ResourceExhausted, message: \
             \"rate limit exceeded for ip:10.0.0.7 (ADR-172 §3b)\"",
        );
        assert!(e.is_terminal());
    }

    #[test]
    fn is_terminal_does_not_match_transient_failures() {
        // Connect timeout, deadline_exceeded, generic transport
        // hiccups — all retry-worthy.
        assert!(!transport("connect: timeout after 5s").is_terminal());
        assert!(!transport("embed RPC: status: DeadlineExceeded, message: \"\"").is_terminal());
        assert!(
            !transport("embed RPC: status: Cancelled, message: \"operation was canceled\"")
                .is_terminal()
        );
        assert!(
            !transport("embed RPC: status: Internal, message: \"embed: NPU stuck\"").is_terminal()
        );
    }

    #[test]
    fn is_terminal_includes_structural_mismatches() {
        // DimMismatch + FingerprintMismatch can't be cured by retry.
        let dim = ClusterError::DimMismatch {
            worker: "w".into(),
            expected: 384,
            actual: 8,
        };
        assert!(dim.is_terminal());
        let fp = ClusterError::FingerprintMismatch {
            worker: "w".into(),
            actual: "fp:bad".into(),
            expected: "fp:good".into(),
        };
        assert!(fp.is_terminal());
    }

    #[test]
    fn is_terminal_does_not_match_aggregate_or_setup_errors() {
        assert!(!ClusterError::NoWorkers.is_terminal());
        assert!(!ClusterError::AllWorkersFailed("aggregate".into()).is_terminal());
        assert!(!ClusterError::NotYetImplemented("foo").is_terminal());
    }
}
