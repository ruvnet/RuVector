//! The causal episodic graph — the cluster-layer fusion structure of ADR-320
//! (PIR WP18), built over [`AtomicObservation`]s.
//!
//! Informed by **MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`)**'s
//! event-layer → cluster-layer architecture, and **explicitly distinct from the
//! unrelated pre-existing `memfuse/memfuse` open-source project**. Atomic,
//! source-tagged observations (the event layer) fuse into
//! [`FusedCluster`]s (the cluster layer), and **every fused/derived node
//! resolves back to the atomic source observations it was built from** — the
//! load-bearing provenance guarantee of this layer (see
//! [`CausalEpisodicGraph::resolve_provenance`]).
//!
//! ## Security / validation gates (ADR-320)
//!
//! - **Per-observation signature** verified before an observation enters the
//!   graph ([`CausalEpisodicGraph::ingest`]).
//! - **Causal-parents integrity**: every `causal_parents` reference must
//!   resolve to an already-present observation; an unresolvable parent, or a
//!   self-referential (cyclic) parent, is a hard rejection at ingest, not a
//!   warning. (Because parents must pre-exist, the parent relation is
//!   necessarily a DAG.)
//! - **Tenant isolation**: a [`CausalEpisodicGraph`] is bound to one
//!   [`Tenant`]; an observation from any other tenant is rejected and never
//!   fuses in.
//!
//! ## Transactional integrity (ADR-320 §3, reusing WP4)
//!
//! [`CausalEpisodicGraph::ingest_governed`] routes an observation's admission
//! through the WP4 [`TransactionalLedger`] (ADR-307 TARL): the observation
//! enters memory as a governed `add` transition that emits an ADR-134 witness
//! record ("no witness, no mutation"), with its `causal_parents` mapped to the
//! ledger's `depends_on` edges so the ledger's poisoning-containment cascade
//! extends to observation lineage. This crate does **not** rebuild the ledger;
//! it interfaces to it.

use std::collections::{BTreeMap, BTreeSet};

use crate::ledger::{ProofGate, TransactionalLedger};
use crate::observation::{AtomicObservation, ObservationId, Tenant};
use crate::ops::{LedgerError, WitnessSink};

/// Identifier for a cluster-layer [`FusedCluster`] (graph-local, monotonic).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClusterId(pub u64);

/// A reference to a node in the causal episodic graph: either an event-layer
/// atomic observation or a cluster-layer fused cluster. A cluster may fuse
/// other clusters, so provenance resolution is transitive down to atomic
/// observations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NodeRef {
    Observation(ObservationId),
    Cluster(ClusterId),
}

/// A cluster-layer node: related observations (and/or sub-clusters) fused into
/// one unit, carrying aggregated confidence and the (single) tenant its members
/// share.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedCluster {
    pub id: ClusterId,
    /// Event-layer observations and/or sub-clusters fused into this node.
    pub members: Vec<NodeRef>,
    /// The tenant all members share (equal to the owning graph's tenant).
    pub tenant: Tenant,
    /// Aggregated confidence: the **weakest-link minimum** over member
    /// confidences. A fused cluster is only as trustworthy as its least
    /// confident evidence.
    pub confidence: f32,
    /// Human-readable fusion key / topic label.
    pub label: String,
}

/// Errors from fusion-graph operations.
///
/// `#[non_exhaustive]`: new gates get new variants (this enum gained
/// `MemberConfidenceInvalid` in ADR-330), so external consumers must carry a
/// wildcard arm rather than have a future gate break their build.
#[derive(Debug)]
#[non_exhaustive]
pub enum FusionError {
    /// The observation's signature did not verify (per-observation gate).
    SignatureInvalid(ObservationId),
    /// The observation's tenant does not match the graph's tenant boundary.
    TenantMismatch { expected: Tenant, got: Tenant },
    /// A `causal_parents` reference does not resolve to a known observation.
    UnresolvedParent {
        observation: ObservationId,
        missing: ObservationId,
    },
    /// The observation lists its own id as a causal parent (a trivial cycle).
    CausalCycle(ObservationId),
    /// An observation with this id is already present.
    DuplicateObservation(ObservationId),
    /// A referenced observation is not in the graph.
    UnknownObservation(ObservationId),
    /// A referenced cluster is not in the graph.
    UnknownCluster(ClusterId),
    /// A fusion was requested with no members.
    EmptyCluster,
    /// A fusion member's confidence was not finite in `[0, 1]`, so the
    /// weakest-link fold would have produced a value outside this layer's
    /// documented contract (or silently dropped a NaN).
    MemberConfidenceInvalid(NodeRef),
    /// A governed ingest referenced a parent that has no ledger entry (it was
    /// not itself admitted through the ledger).
    ParentNotGoverned(ObservationId),
    /// The underlying WP4 ledger refused the governed transition.
    Ledger(LedgerError),
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusionError::SignatureInvalid(id) => {
                write!(
                    f,
                    "observation {} failed signature verification",
                    id.to_hex()
                )
            }
            FusionError::TenantMismatch { expected, got } => write!(
                f,
                "tenant isolation violation: graph tenant {:?}, observation tenant {:?}",
                expected.as_str(),
                got.as_str()
            ),
            FusionError::UnresolvedParent {
                observation,
                missing,
            } => write!(
                f,
                "observation {} references unresolved causal parent {}",
                observation.to_hex(),
                missing.to_hex()
            ),
            FusionError::CausalCycle(id) => {
                write!(f, "observation {} is its own causal parent", id.to_hex())
            }
            FusionError::DuplicateObservation(id) => {
                write!(f, "observation {} already present", id.to_hex())
            }
            FusionError::UnknownObservation(id) => {
                write!(f, "unknown observation {}", id.to_hex())
            }
            FusionError::UnknownCluster(id) => write!(f, "unknown cluster {}", id.0),
            FusionError::EmptyCluster => write!(f, "cannot fuse an empty member set"),
            FusionError::MemberConfidenceInvalid(node) => write!(
                f,
                "fusion member {node:?} has non-finite or out-of-range confidence"
            ),
            FusionError::ParentNotGoverned(id) => write!(
                f,
                "governed ingest parent {} has no ledger entry",
                id.to_hex()
            ),
            FusionError::Ledger(e) => write!(f, "ledger refused governed ingest: {e}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// A tenant-scoped causal episodic graph: an event layer of atomic
/// observations and a cluster layer of fused clusters over them.
pub struct CausalEpisodicGraph {
    tenant: Tenant,
    observations: BTreeMap<ObservationId, AtomicObservation>,
    clusters: BTreeMap<ClusterId, FusedCluster>,
    /// Ledger entry ids for observations admitted through [`Self::ingest_governed`].
    ledger_ids: BTreeMap<ObservationId, u64>,
    next_cluster: u64,
}

impl CausalEpisodicGraph {
    /// A new, empty graph bound to `tenant`.
    pub fn new(tenant: Tenant) -> Self {
        Self {
            tenant,
            observations: BTreeMap::new(),
            clusters: BTreeMap::new(),
            ledger_ids: BTreeMap::new(),
            next_cluster: 0,
        }
    }

    /// The tenant boundary this graph enforces.
    pub fn tenant(&self) -> &Tenant {
        &self.tenant
    }

    /// Admit an observation into the event layer after running every
    /// ADR-320 ingest gate (signature, tenant, causal-parents integrity,
    /// duplicate). Returns the observation's content address on success.
    ///
    /// This is the pure-graph path; [`Self::ingest_governed`] additionally
    /// routes admission through the WP4 ledger.
    pub fn ingest(&mut self, obs: AtomicObservation) -> Result<ObservationId, FusionError> {
        let id = self.validate_for_ingest(&obs)?;
        self.observations.insert(id, obs);
        Ok(id)
    }

    /// Admit an observation as a **governed transition** through the WP4 TARL
    /// ledger (ADR-320 §3). Runs the same ingest gates, then records the
    /// admission in `ledger` as an `add` (emitting an ADR-134 witness record),
    /// wiring `causal_parents` to the ledger's `depends_on` edges. On ledger
    /// refusal nothing is inserted into the graph.
    ///
    /// Returns the observation id and its ledger entry id.
    pub fn ingest_governed<S: WitnessSink, G: ProofGate>(
        &mut self,
        obs: AtomicObservation,
        ledger: &mut TransactionalLedger<S, G>,
        actor: &str,
    ) -> Result<(ObservationId, u64), FusionError> {
        let id = self.validate_for_ingest(&obs)?;

        // Map causal parents to ledger dependency edges. Parents must have been
        // admitted through the ledger too, else there is no governed dependency
        // to record.
        let mut depends_on = Vec::with_capacity(obs.causal_parents.len());
        for parent in &obs.causal_parents {
            let ledger_id = self
                .ledger_ids
                .get(parent)
                .ok_or(FusionError::ParentNotGoverned(*parent))?;
            depends_on.push(*ledger_id);
        }

        // Witness-first governed transition: on refusal nothing mutates here.
        let entry_id = ledger
            .add(
                id.to_hex(),
                &depends_on,
                actor,
                "ADR-320 atomic observation admitted to continuous-latent-state tier",
            )
            .map_err(FusionError::Ledger)?;

        self.ledger_ids.insert(id, entry_id);
        self.observations.insert(id, obs);
        Ok((id, entry_id))
    }

    /// Fuse `members` (event-layer observations and/or sub-clusters, all of
    /// this graph's tenant) into a new cluster-layer node. Confidence is the
    /// weakest-link minimum over members. Rejects an empty set, any member not
    /// present in the graph, and any member whose confidence is not finite in
    /// `[0, 1]`.
    ///
    /// The finiteness check is load-bearing, not decorative: `f32::min` returns
    /// the *non-NaN* operand per IEEE 754, so folding with an `INFINITY` seed
    /// would give a single-NaN-member cluster `confidence = +inf` (violating
    /// this type's documented `[0, 1]` contract) and would let a NaN member of
    /// `[NaN, 0.9]` vanish without trace into a `0.9` cluster. Because
    /// [`AtomicObservation::new_signed`]'s range check can be bypassed — the
    /// field is public and a signature over a NaN bit-pattern verifies — the
    /// rejection has to happen here, before the fold.
    pub fn fuse(
        &mut self,
        members: &[NodeRef],
        label: impl Into<String>,
    ) -> Result<ClusterId, FusionError> {
        if members.is_empty() {
            return Err(FusionError::EmptyCluster);
        }
        let mut confidence = 1.0f32;
        for member in members {
            let c = self.node_confidence(*member)?;
            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                return Err(FusionError::MemberConfidenceInvalid(*member));
            }
            if c < confidence {
                confidence = c;
            }
        }
        let id = ClusterId(self.next_cluster);
        self.next_cluster += 1;
        self.clusters.insert(
            id,
            FusedCluster {
                id,
                members: members.to_vec(),
                tenant: self.tenant.clone(),
                confidence,
                label: label.into(),
            },
        );
        Ok(id)
    }

    /// Resolve `node` to the set of **atomic source observations** it derives
    /// from — the load-bearing provenance guarantee. For an observation this is
    /// the observation itself; for a cluster it is the transitive union of its
    /// members' atomic sources.
    ///
    /// Traversal is **iterative** (an explicit heap-allocated worklist, not the
    /// call stack), so a deeply nested fuse chain (`C1=fuse([obs])`,
    /// `C2=fuse([C1])`, …) is bounded by heap, not stack depth, and cannot
    /// overflow the stack / abort the process on this load-bearing query. It is
    /// also **cycle-safe**: the `visited_clusters` guard makes resolution total
    /// even though the cluster layer is acyclic by construction.
    pub fn resolve_provenance(
        &self,
        node: NodeRef,
    ) -> Result<BTreeSet<ObservationId>, FusionError> {
        let mut atomic = BTreeSet::new();
        let mut visited_clusters = BTreeSet::new();
        let mut worklist: Vec<NodeRef> = vec![node];
        while let Some(current) = worklist.pop() {
            match current {
                NodeRef::Observation(id) => {
                    if !self.observations.contains_key(&id) {
                        return Err(FusionError::UnknownObservation(id));
                    }
                    atomic.insert(id);
                }
                NodeRef::Cluster(id) => {
                    if !visited_clusters.insert(id) {
                        continue; // already expanded; cycle-safe
                    }
                    let cluster = self
                        .clusters
                        .get(&id)
                        .ok_or(FusionError::UnknownCluster(id))?;
                    for member in &cluster.members {
                        worklist.push(*member);
                    }
                }
            }
        }
        Ok(atomic)
    }

    /// Look up an event-layer observation.
    pub fn observation(&self, id: ObservationId) -> Option<&AtomicObservation> {
        self.observations.get(&id)
    }

    /// Look up a cluster-layer node.
    pub fn cluster(&self, id: ClusterId) -> Option<&FusedCluster> {
        self.clusters.get(&id)
    }

    /// The ledger entry id for an observation admitted via
    /// [`Self::ingest_governed`], if any.
    pub fn ledger_id(&self, id: ObservationId) -> Option<u64> {
        self.ledger_ids.get(&id).copied()
    }

    /// Number of event-layer observations.
    pub fn observation_count(&self) -> usize {
        self.observations.len()
    }

    /// Number of cluster-layer nodes.
    pub fn cluster_count(&self) -> usize {
        self.clusters.len()
    }

    // ── Internals ────────────────────────────────────────────────────────────

    /// Run every ingest gate and return the validated content address without
    /// mutating the graph.
    fn validate_for_ingest(&self, obs: &AtomicObservation) -> Result<ObservationId, FusionError> {
        // Tenant isolation (hard boundary).
        if obs.tenant != self.tenant {
            return Err(FusionError::TenantMismatch {
                expected: self.tenant.clone(),
                got: obs.tenant.clone(),
            });
        }
        // Per-observation signature.
        if !obs.verify() {
            return Err(FusionError::SignatureInvalid(obs.id()));
        }
        let id = obs.id();
        if self.observations.contains_key(&id) {
            return Err(FusionError::DuplicateObservation(id));
        }
        // Causal-parents integrity: acyclic (no self-parent) and resolvable.
        for parent in &obs.causal_parents {
            if *parent == id {
                return Err(FusionError::CausalCycle(id));
            }
            if !self.observations.contains_key(parent) {
                return Err(FusionError::UnresolvedParent {
                    observation: id,
                    missing: *parent,
                });
            }
        }
        Ok(id)
    }

    /// Confidence of a node, for weakest-link aggregation.
    fn node_confidence(&self, node: NodeRef) -> Result<f32, FusionError> {
        match node {
            NodeRef::Observation(id) => self
                .observations
                .get(&id)
                .map(|o| o.confidence)
                .ok_or(FusionError::UnknownObservation(id)),
            NodeRef::Cluster(id) => self
                .clusters
                .get(&id)
                .map(|c| c.confidence)
                .ok_or(FusionError::UnknownCluster(id)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observation::{ObservationSource, SourceKind};
    use rand::rngs::OsRng;
    use rvf_types::Ed25519Keypair;

    fn signed(
        keypair: &Ed25519Keypair,
        kind: SourceKind,
        src_id: &str,
        tenant: &str,
        confidence: f32,
        parents: Vec<ObservationId>,
        payload: &[u8],
    ) -> AtomicObservation {
        AtomicObservation::new_signed(
            ObservationSource {
                kind,
                id: src_id.to_string(),
                public_key: [0u8; 32],
            },
            0,
            confidence,
            Tenant::new(tenant),
            parents,
            payload.to_vec(),
            keypair,
        )
        .unwrap()
    }

    #[test]
    fn cross_tenant_observation_rejected() {
        let kp = Ed25519Keypair::generate(&mut OsRng);
        let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
        let foreign = signed(
            &kp,
            SourceKind::AgentObservation,
            "a",
            "globex",
            0.9,
            vec![],
            b"x",
        );
        match graph.ingest(foreign) {
            Err(FusionError::TenantMismatch { .. }) => {}
            other => panic!("expected TenantMismatch, got {other:?}"),
        }
    }

    #[test]
    fn unresolved_parent_rejected() {
        let kp = Ed25519Keypair::generate(&mut OsRng);
        let mut graph = CausalEpisodicGraph::new(Tenant::new("acme"));
        let ghost = ObservationId([9u8; 32]);
        let child = signed(
            &kp,
            SourceKind::AgentObservation,
            "a",
            "acme",
            0.8,
            vec![ghost],
            b"c",
        );
        match graph.ingest(child) {
            Err(FusionError::UnresolvedParent { .. }) => {}
            other => panic!("expected UnresolvedParent, got {other:?}"),
        }
    }
}
