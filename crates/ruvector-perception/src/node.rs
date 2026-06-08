//! Ambient nervous-system node — the appliance surface.
//!
//! Wires the perception substrate into one local "coherence node" for a room /
//! machine / building. It does **not** stream raw sensor data; it ingests
//! readings and emits **deltas, boundaries, coherence, proof-gated witnesses,
//! forecasts, and an auditable custody chain** — and answers grounded agent
//! queries. Not a camera, not an IoT hub, not a dashboard.

use crate::custody::{CustodyError, CustodyLedger};
use crate::engine::{DeltaEngine, EngineConfig};
use crate::predict::{BoundaryForecast, BoundaryObservation, BoundaryPredictor};
use crate::reality::{GroundedAnswer, Query, RealityGraph};
use crate::state::Reading;
use crate::witness::DeltaWitness;

/// What the node emits per observed window — structure, never raw signal.
#[derive(Debug, Clone)]
pub struct NodeEvent {
    /// The proof-gated delta witness for this window.
    pub witness: DeltaWitness,
    /// Where coherence is forecast to break next (if anywhere).
    pub forecast: Option<BoundaryForecast>,
}

/// A self-contained ambient perception node.
pub struct NervousSystemNode {
    engine: DeltaEngine,
    reality: RealityGraph,
    ledger: CustodyLedger,
    predictor: BoundaryPredictor,
}

impl NervousSystemNode {
    /// Build a node. `predict_window` is the per-zone history length used by the
    /// boundary-break forecaster.
    pub fn new(config: EngineConfig, predict_window: usize) -> Self {
        Self {
            engine: DeltaEngine::new(config),
            reality: RealityGraph::new(),
            ledger: CustodyLedger::new(),
            predictor: BoundaryPredictor::new(predict_window),
        }
    }

    /// Observe one window of multi-modal readings. Runs the full pipeline
    /// (delta → boundary → coherence → proof → action), appends the witness to
    /// the custody chain, grounds it into the reality graph, updates the
    /// forecaster, and returns the emitted [`NodeEvent`].
    pub fn observe(&mut self, readings: &[Reading], t: u64) -> NodeEvent {
        let witness = self.engine.observe(readings, t);
        // Maintain the auditable chain (the engine produces a linked witness
        // chain, so append links cleanly).
        let _ = self.ledger.append(witness.clone());
        self.reality.ingest(&witness);
        if !witness.changed_boundary.is_empty() {
            self.predictor.observe(&BoundaryObservation::new(
                witness.changed_boundary.clone(),
                witness.coherence,
                witness.contradiction,
                t,
            ));
        }
        let forecast = self.predictor.next_break();
        NodeEvent { witness, forecast }
    }

    /// Answer a grounded agent query from physical memory.
    pub fn query(&self, q: &Query) -> GroundedAnswer {
        self.reality.query(q)
    }

    /// The auditable custody ledger (chain of every emitted witness).
    pub fn ledger(&self) -> &CustodyLedger {
        &self.ledger
    }

    /// Verify the integrity of the custody chain.
    pub fn verify_custody(&self) -> Result<(), CustodyError> {
        self.ledger.verify()
    }

    /// The grounding reality graph.
    pub fn reality(&self) -> &RealityGraph {
        &self.reality
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modality::Modality;

    fn warm(node: &mut NervousSystemNode) {
        // Three zones so the changed-boundary is unambiguous (2-zone min-cut
        // splits are symmetric and the minority side is arbitrary).
        for i in 0..8u64 {
            let hi = (i % 2) as f32;
            node.observe(
                &[
                    Reading::new("zone_a", Modality::Rf, hi),
                    Reading::new("zone_a", Modality::Vibration, hi),
                    Reading::new("zone_a", Modality::Thermal, 20.0 + hi),
                    Reading::new("zone_b", Modality::Rf, 0.0),
                    Reading::new("zone_c", Modality::Rf, 0.0),
                ],
                i,
            );
        }
    }

    #[test]
    fn node_emits_witness_chain_and_grounds_queries() {
        let mut node = NervousSystemNode::new(EngineConfig::default(), 16);
        warm(&mut node);
        let n_before = node.ledger().len();

        // An RF/vibration event in zone_a (thermal silent).
        let ev = node.observe(
            &[
                Reading::new("zone_a", Modality::Rf, 5.0),
                Reading::new("zone_a", Modality::Vibration, 5.0),
                Reading::new("zone_a", Modality::Thermal, 20.5),
                Reading::new("zone_b", Modality::Rf, 0.0),
                Reading::new("zone_c", Modality::Rf, 0.0),
            ],
            100,
        );
        assert_eq!(ev.witness.changed_boundary, "zone_a");

        // Custody chain grew and verifies.
        assert_eq!(node.ledger().len(), n_before + 1);
        assert!(node.verify_custody().is_ok());

        // The agent can query reality, grounded in a witness evidence hash.
        let presence = node.query(&Query::Presence {
            zone: "zone_a".into(),
        });
        assert!(presence.yes);
        assert!(!presence.evidence.is_empty());
        // A zone with no memory is honestly unknown.
        assert!(
            !node
                .query(&Query::Presence {
                    zone: "unknown".into()
                })
                .yes
        );
    }

    #[test]
    fn empty_node_is_safe() {
        let node = NervousSystemNode::new(EngineConfig::default(), 8);
        assert!(node.ledger().is_empty());
        assert!(node.verify_custody().is_ok());
        assert!(!node.query(&Query::Presence { zone: "x".into() }).yes);
    }
}
