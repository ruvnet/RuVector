//! # latentmesh-thoughtmap — DeAR-pattern decentralized capability-grounded reasoning
//!
//! PIR **WP23** (RuVector#882), implementing **ADR-326** over the LatentMesh
//! fabric (ADR-309 transport, ADR-310 causal gate, ADR-311 quarantine).
//!
//! Informed by **DeAR** — *Decentralized Agentic Reasoning via Capability
//! Grounding and Collaborative Thought Navigation* (arXiv:2608.17282), graded
//! **B+, not A**, with **no released code** (the paper's repo is a placeholder
//! `https://open_upon_acceptance` URL). This crate is therefore a **first-party
//! build from the paper's description**, not a port. Under the program's
//! preprint-reproduction rule, DeAR's reported numbers are hypotheses, not the
//! acceptance bar.
//!
//! ## What this first shippable slice provides
//!
//! 1. **Capability vectors replace named roles** ([`capability`]): for a triple
//!    `(agent i, peer j, query q)`, `C(i,j,q) = competence × trust × locality ×
//!    freshness ÷ cost`, recomputed *per query*. Each agent selects its next peer
//!    **locally** — no central planner ([`capability::LocalView::select_peer`]).
//! 2. **A shared, navigable thought map** ([`thoughtmap`]): nodes are reasoning
//!    steps, edges are peer interactions; success **reinforces** edges, failure
//!    **weakens** them. First-party in-memory graph today, with a documented
//!    RuVector-integration seam ([`thoughtmap::ThoughtGraphBackend`]).
//! 3. **LatentMesh transport carrying compressed latent state** ([`transport`]),
//!    with **causal verification before acceptance** ([`causal`]) — a received
//!    latent message must pass ADR-310's controlled-replacement causal audit or
//!    it is never incorporated.
//! 4. **Dead-end handling** ([`topology`]): change topology and *continue from
//!    the current state* rather than restart — **explicitly marked an unverified
//!    design choice (ADR-326 §6), not a reproduced DeAR claim**, and kept
//!    swappable via [`topology::DeadEndPolicy`].
//! 5. **Security** ([`quarantine`], [`engine`]): every incorporated message is
//!    causally attributable, and a peer that degrades the map is dampened /
//!    quarantined so it cannot churn the topology.
//!
//! ## Deferred (honest scope)
//!
//! Real multi-agent execution, the 9-benchmark DeAR evaluation, and full
//! LatentMesh **wire** transport (compression codec + `ruvnet/LatentMesh`
//! wire-format coordination) are out of scope for this slice. The transport is a
//! defined seam with an in-process stub; a WP5 crate implements
//! [`transport::LatentTransport`] later without changing callers.
//!
//! ## ⚠️ Unverified-behavior marker (binding, ADR-326)
//!
//! The dead-end **"continue, not restart"** behavior is this program's own
//! working interpretation of the abstract's four-word *"topology update for
//! adaptive error correction"* — **not** a verified paper claim. It must be
//! re-verified against the full paper or released code before being treated as a
//! fixed contract. See [`topology`] for the full marker.

pub mod capability;
pub mod causal;
pub mod engine;
pub mod quarantine;
pub mod thoughtmap;
pub mod topology;
pub mod transport;
mod vecmath;

pub use capability::{CapabilityVector, LocalView, PeerId, PeerProfile, PeerSelection, Query};
pub use causal::{CausalVerdict, CausalVerifier, ControlledReplacementVerifier, RejectReason};
pub use engine::{Incorporation, ReasoningEngine, RejectionKind, ROOT_STEP};
pub use quarantine::{Anomaly, PeerReputation, Quarantine};
pub use thoughtmap::{StepId, ThoughtEdge, ThoughtGraphBackend, ThoughtMap, ThoughtNode};
pub use topology::{DeadEndPolicy, TopologyCause, TopologyChange};
pub use transport::{
    integrity_tag, CausalTag, InProcessTransport, LatentMessage, LatentState, LatentTransport,
    TransportError,
};
