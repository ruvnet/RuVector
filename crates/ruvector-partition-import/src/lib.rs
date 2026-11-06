//! Signature-verified cut-partition import for cross-agent memory transfer.
//!
//! ## Hypothesis
//!
//! ```text
//! Given a weighted graph G (synthetic agent-memory graph: nodes = memory
//! items, edges = semantic/temporal association weights, generated
//! deterministically from a fixed seed) and a correct global-minimum-cut
//! partition P of G with cut value C (computed once by a trusted baseline
//! algorithm),
//!
//! when agent A signs a certificate over
//! digest(canonical_encoding(G), canonical_encoding(P), C) with Ed25519 and
//! sends (G, P, certificate) to agent B,
//!
//! then B's import latency (verify + accept) should be substantially lower
//! than B's baseline import latency (ignore P entirely, recompute the
//! global mincut from scratch on G),
//!
//! subject to: (a) zero false negatives — any single-byte tamper to the
//! serialized G or P after signing must be detected by signature
//! verification in 100% of trials; (b) when the certificate is valid and P
//! is genuinely optimal, B's accepted cut value must exactly equal the
//! baseline's recomputed cut value; (c) build green, tests green.
//! ```
//!
//! ## The adversarial finding this crate exists to demonstrate
//!
//! A valid Ed25519 signature over `(G, P, C)` proves only that the holder
//! of A's signing key produced that exact triple. It proves nothing about
//! whether `P` is actually a correct minimum-cut partition. An adversary
//! who legitimately holds A's key (or A itself, buggy or compromised) can
//! sign a genuinely suboptimal `P'` with a worse cut value `C'`, and a
//! verifier that only checks the signature ([`candidates::candidate_a`])
//! will accept it every time. [`candidates::candidate_b`] adds a cheap,
//! *necessary-but-not-sufficient* local-optimality spot-check — sampling a
//! fraction of vertices and confirming that moving each one to the other
//! side would not strictly decrease the cut — which catches some, but not
//! all, suboptimal partitions, in proportion to how many vertices were
//! perturbed versus how many were sampled. See the nightly research report
//! at `docs/research/nightly/2026-10-22-verified-partition-import/README.md`
//! for the measured numbers and the full discussion.
//!
//! ## Modules
//!
//! - [`graph`]: deterministic synthetic weighted-graph generator plus
//!   canonical (hash-stable) encodings of graphs and partitions.
//! - [`baseline`]: a self-contained, sparse Stoer-Wagner global minimum-cut
//!   reference implementation — the "recompute from scratch" slow path and
//!   the ground truth for correctness. Deliberately *not* integrated with
//!   `ruvector-mincut`; see that module's doc comment for why.
//! - [`certificate`]: the Ed25519 cut certificate, following the
//!   `MSG_TAG || ... || digest` domain-separated signing convention used
//!   elsewhere in this workspace (`ruvector-agent-memory::witness_signing`,
//!   `ruvector-retrieval-receipt::signing`).
//! - [`candidates`]: `candidate_a` ("trust-the-signature") and
//!   `candidate_b` ("verify + spot-check") import strategies.

pub mod baseline;
pub mod candidates;
pub mod certificate;
pub mod graph;

pub use baseline::{global_min_cut, MinCutResult};
pub use candidates::{candidate_a, candidate_b, ImportDecision, ImportError, ImportOutcome};
pub use certificate::{verify_against, verify_signature, CutCertificate, Signer};
pub use graph::{generate_graph, Partition, WeightedGraph};
