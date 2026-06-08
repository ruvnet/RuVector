//! # ruvector-perception — the layer *under* classification
//!
//! Current WiFi/edge SOTA is racing toward better *classifiers* (CSI foundation
//! models, self-supervised CSI representations, adaptive fusion). This crate
//! deliberately does **not** build a better classifier. It builds the substrate
//! underneath one:
//!
//! ```text
//! classification → confidence → alert        (today)
//! delta → boundary → coherence → proof → action   (here)
//! ```
//!
//! Instead of asking *"what is this?"* it asks *"what changed, where did the
//! boundary move, and is the change coherent enough to act on?"* — and it
//! requires **evidence**, not confidence, before it grants any authority.
//!
//! ## Pipeline
//!
//! 1. **Delta** ([`state`], [`engine`]) — every reading becomes a delta against a
//!    rolling multi-modal baseline. No fixed task label (fall/gesture/leak).
//! 2. **Boundary** ([`coherence`]) — zones form a coherence graph; dynamic
//!    min-cut isolates the side that broke away (the moved boundary).
//! 3. **Coherence + contradiction** — a modality that *usually* reacts in a zone
//!    but stayed silent is a first-class contradiction (disagreement is
//!    information), weighted by the modality's physical spoof-resistance.
//! 4. **Proof** ([`witness`]) — a proof gate turns novelty/coherence/
//!    contradiction into *bounded authority* (Ignore → Observe → Alert →
//!    Mutate) and emits an auditable SHA-256 evidence chain.
//! 5. **Action** — only evidence that is novel, coherent, and uncontradicted may
//!    escalate; contradicted evidence is capped at *Observe*.
//!
//! Plus [`absence`]: a *missing* expected continuation (e.g. a bedtime routine
//! that never returns) is detected as structural incompleteness, not a threshold.
//!
//! ## Honest scope
//!
//! This is the **mechanism** (a trusted-physical-memory engine), demonstrated on
//! synthetic multi-modal deltas and reusing [`ruvector_mincut`] for boundary
//! detection. It is not validated on real CSI hardware, and it is not a
//! classifier — it is the auditable perception layer a classifier (or an agent)
//! would sit on top of.
//!
//! ## Example
//!
//! ```
//! use ruvector_perception::{DeltaEngine, EngineConfig, Reading, Modality, Action};
//!
//! let mut eng = DeltaEngine::new(EngineConfig::default());
//! // (warm up baselines first in real use)
//! let w = eng.observe(&[
//!     Reading::new("table_left_zone", Modality::Rf, 3.0),
//!     Reading::new("table_left_zone", Modality::Vibration, 3.0),
//! ], 0);
//! assert_eq!(w.changed_boundary, "table_left_zone");
//! let _ = Action::Observe;
//! ```

#![forbid(unsafe_code)]

pub mod absence;
pub mod captcha;
pub mod coherence;
pub mod custody;
pub mod engine;
pub mod hypothesis;
pub mod identity;
pub mod modality;
pub mod node;
pub mod predict;
pub mod reality;
pub mod state;
pub mod swarm;
pub mod topology;
pub mod witness;

pub use absence::{Absence, SequenceMonitor};
pub use captcha::{CaptchaVerifier, ChallengeResponse, ObservedResponse, RealityProof, Stimulus};
pub use coherence::{detect_boundary, Boundary};
pub use custody::{CustodyError, CustodyLedger, CustodyRecord};
pub use engine::{DeltaEngine, EngineConfig};
pub use hypothesis::{rank_hypotheses, DisagreementInput, Hypothesis, RankedHypothesis};
pub use identity::{IdentityDrift, IdentityMemory};
pub use modality::{Modality, Physics};
pub use node::{NervousSystemNode, NodeEvent};
pub use predict::{BoundaryForecast, BoundaryObservation, BoundaryPredictor};
pub use reality::{GroundedAnswer, Query, RealityGraph};
pub use state::{Reading, WorldState};
pub use swarm::{FacilityGraph, FragilityReport};
pub use topology::{NodeAssessment, NodeRole, TopologyManager};
pub use witness::{evidence_hash, novelty_level, Action, DeltaWitness, ProofGate};

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
