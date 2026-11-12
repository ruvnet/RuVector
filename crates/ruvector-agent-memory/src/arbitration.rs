//! Correlation-aware memory arbitration (ADR-330, PIR WP27), informed by
//! **CAMA — "Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent
//! Memory Arbitration" (arXiv:2608.19701)**. No upstream code exists and the
//! paper's abstract reports no quantitative results, so this is a first-party
//! build of the described *mechanism* with this program's own acceptance bar,
//! not a port and not a number-chase.
//!
//! # Memory Correlation Bias
//!
//! N retrieved memories that all **declare** derivation from one original
//! source are **one observation repeated N times, not N independent
//! observations**. A naive confidence vote (`sum of supporting memories`) lets
//! twenty agents relaying one rumor outvote three genuinely independent
//! witnesses. This module arbitrates a retrieved memory set by clustering it
//! into **evidence lineages** using the causal provenance the ADR-320 layer
//! already records, then scoring effective evidence per *lineage*, not per
//! memory.
//!
//! # Known limitation: declared derivation only (READ BEFORE USE)
//!
//! **Correlation is detected from declared `causal_parents`, never from
//! content.** A memory that copies another memory's *content* while declaring
//! no parent is, to this module, a fresh root — it mints an independent
//! lineage. Twenty agents that each re-assert one rumor's content without
//! citing it therefore arbitrate to twenty lineages with **no *correlation*
//! downgrade** against the naive vote, and a sufficiency check at any
//! threshold ≤ 20 returns [`ArbitrationOutcome::Sufficient`].
//!
//! **Do not read a smaller arbitrated score as correlation having been
//! caught.** Any reduction observed with a reliability or freshness factor
//! below `1.0` comes from *those multipliers*, which carry no correlation
//! information whatsoever. Concretely, under the shipped
//! [`ReliabilityModel::default`] (`0.9`) that same twenty-agent astroturfed
//! set scores naive `18.0` vs arbitrated `16.2` — a ~10% reduction that is
//! entirely the reliability multiplier and reflects *zero* detected
//! correlation. **The diagnostic to watch is the lineage count, not the score
//! delta**: [`ArbitrationVerdict::independent_lineages`] equalling the number
//! of arbitrated memories means nothing was clustered, whatever the score
//! says.
//!
//! The shipped scope deliberately **under-detects** correlation rather than
//! pretending otherwise (ADR-330, Negative consequences). The consequences
//! for an integrator:
//!
//! - [`ArbitrationConfig::sufficiency_threshold`] and
//!   [`ArbitrationOutcome::InsufficientIndependentEvidence`] are **not a trust
//!   gate against repeated-claim or sybil attacks**. They answer "how many
//!   *declared-independent* lineages support this?", which is a corroboration
//!   signal under cooperative provenance, not an adversarial one.
//!   Wiring them as an anti-astroturfing control is materially wrong.
//! - The guarantee holds only as far as provenance discipline holds upstream.
//!   Deployments that need adversarial robustness must pair this with content-
//!   level near-duplicate detection or authenticated citation at ingest, so
//!   that copied content cannot enter as a parentless observation.
//!
//! The bench in `agent-memory-bench` (900 derived memories collapsing to 100
//! origin lineages) exercises **declared-parent chains — the mechanism's best
//! case** — and is a demonstration of the clustering math, not evidence of
//! adversarial robustness.
//!
//! # The lineage-sharing rule (precise definition)
//!
//! Every memory node resolves to its atomic observations
//! ([`CausalEpisodicGraph::resolve_provenance`]); every atomic observation has
//! a **root set**: the parentless ancestors reachable through
//! `causal_parents` (a parentless observation is its own root). Two memories
//! belong to the same lineage **iff** their root sets are connected — i.e.
//! lineages are the connected components of the bipartite memory↔root graph.
//! A memory whose ancestry bridges two roots merges those roots into one
//! lineage (conservative: shared ancestry anywhere collapses independence).
//!
//! # Downgrade-only effective evidence (binding invariant)
//!
//! `arbitrated_confidence = Σ_lineages max_member(confidence × reliability ×
//! freshness)` with every factor in `[0, 1]`. Because each lineage contributes
//! at most its single strongest member, and the naive score is the sum of raw
//! confidences over *all* members, the arbitrated score can only be **less
//! than or equal to** the naive score — correlation adjustment can never
//! manufacture evidence (Wave-3 lesson: metric-integrity gates must be
//! downgrade-only + fail-loud). The invariant is re-checked at runtime and a
//! violation is a hard error, never a silent clamp.
//!
//! # Non-finite choke point
//!
//! Every confidence and reliability is checked finite-in-`[0,1]` **before**
//! any comparison or aggregation (the Wave-3 #888 lesson: every `<` against
//! NaN is false, so a single NaN silently defeats comparison-based gates).
//! [`AtomicObservation::new_signed`] already rejects non-finite confidence,
//! but the field is public and a signature over a NaN bit-pattern *does*
//! verify — so ingest alone is not a sufficient guard, and this module
//! re-checks at its own boundary.
//!
//! # Tenant isolation
//!
//! Arbitration operates on one tenant-bound [`CausalEpisodicGraph`]; a node id
//! from another tenant's graph is simply unknown here and is a hard error.
//! Cross-tenant lineage clustering is impossible by construction.
//!
//! # Input sizing (caller's responsibility)
//!
//! [`arbitrate`] bounds neither the memory-set size nor total ancestry-
//! traversal work: cost scales with the retrieved set and the size of its
//! ancestry closure. Traversal is iterative and heap-bounded (no stack
//! overflow on deep chains — a 10 000-deep chain measures ~35 ms, a
//! 1 000-root/2 000-derived merge ~27 ms, union-find memory O(distinct
//! roots)), so this is a throughput consideration, not a crash risk. A service
//! that arbitrates per retrieval should still impose its own cap on the number
//! of memories admitted to one call.

use std::collections::{BTreeMap, BTreeSet};

use crate::fusion::{CausalEpisodicGraph, ClusterId, FusionError, NodeRef};
use crate::observation::ObservationId;

/// Per-source-kind reliability model, keyed by [`SourceKind::code`]
/// (`crate::observation::SourceKind`). All values must be finite in `[0, 1]`;
/// [`arbitrate`] validates this before use.
#[derive(Debug, Clone)]
pub struct ReliabilityModel {
    /// Reliability applied to any source kind without an explicit override.
    pub default: f64,
    /// Overrides keyed by `SourceKind::code()`.
    pub overrides: BTreeMap<u16, f64>,
}

impl Default for ReliabilityModel {
    /// Conservative default: every source kind at `0.9` — reliability shades
    /// the score, the lineage structure does the heavy lifting.
    fn default() -> Self {
        ReliabilityModel {
            default: 0.9,
            overrides: BTreeMap::new(),
        }
    }
}

impl ReliabilityModel {
    fn for_code(&self, code: u16) -> f64 {
        *self.overrides.get(&code).unwrap_or(&self.default)
    }

    fn validate(&self) -> Result<(), ArbitrationError> {
        let ok = |v: f64| v.is_finite() && (0.0..=1.0).contains(&v);
        if !ok(self.default) {
            return Err(ArbitrationError::InvalidConfig(
                "reliability default must be finite in [0, 1]",
            ));
        }
        if !self.overrides.values().all(|v| ok(*v)) {
            return Err(ArbitrationError::InvalidConfig(
                "every reliability override must be finite in [0, 1]",
            ));
        }
        Ok(())
    }
}

/// Configuration for one arbitration pass.
#[derive(Debug, Clone)]
pub struct ArbitrationConfig {
    /// "Now" in nanoseconds, same clock as `AtomicObservation::time_ns`.
    /// Observation timestamps in the future clamp to age zero (freshness 1.0).
    pub now_ns: u64,
    /// Exponential freshness half-life: `freshness = 0.5^(age_ns / half_life_ns)`.
    /// Must be non-zero.
    pub half_life_ns: u64,
    /// Minimum number of independent lineages for the evidence to count as
    /// sufficient; below it the outcome is
    /// [`ArbitrationOutcome::InsufficientIndependentEvidence`] — the CAMA
    /// active-search hook, telling the caller to go find evidence *outside*
    /// the over-represented lineages rather than silently passing.
    pub sufficiency_threshold: usize,
    /// Per-source-kind reliability.
    pub reliability: ReliabilityModel,
}

/// One independent evidence lineage: the memories that share causal ancestry,
/// the provenance roots that bind them, and the lineage's effective weight
/// (its single strongest member's `confidence × reliability × freshness`).
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceLineage {
    /// Parentless ancestor observations this lineage traces to (sorted).
    pub roots: Vec<ObservationId>,
    /// The arbitrated memory nodes in this lineage (sorted, de-duplicated).
    pub members: Vec<NodeRef>,
    /// Effective weight in `[0, 1]`: max over members of the member's
    /// `confidence × min_atomic(reliability) × min_atomic(freshness)`.
    pub weight: f64,
}

/// The full arbitration result.
#[derive(Debug, Clone, PartialEq)]
pub struct ArbitrationVerdict {
    /// Lineages, sorted by their minimal root id (deterministic).
    pub lineages: Vec<EvidenceLineage>,
    /// What a correlation-blind vote would score: Σ raw member confidences.
    pub naive_confidence: f64,
    /// Σ lineage weights. Guaranteed ≤ `naive_confidence` (downgrade-only).
    pub arbitrated_confidence: f64,
}

impl ArbitrationVerdict {
    /// Number of independent evidence lineages.
    pub fn independent_lineages(&self) -> usize {
        self.lineages.len()
    }
}

/// Outcome of an arbitration pass.
#[derive(Debug, Clone, PartialEq)]
pub enum ArbitrationOutcome {
    /// At least `sufficiency_threshold` independent lineages support the claim.
    Sufficient(ArbitrationVerdict),
    /// Fewer independent lineages than required. Carries the roots of every
    /// lineage holding more than one member — the over-represented origins the
    /// caller's evidence search should look *away from* (CAMA's active
    /// independent-evidence recovery step).
    InsufficientIndependentEvidence {
        verdict: ArbitrationVerdict,
        required: usize,
        over_represented_roots: Vec<ObservationId>,
    },
}

impl ArbitrationOutcome {
    /// The verdict regardless of sufficiency.
    pub fn verdict(&self) -> &ArbitrationVerdict {
        match self {
            ArbitrationOutcome::Sufficient(v) => v,
            ArbitrationOutcome::InsufficientIndependentEvidence { verdict, .. } => verdict,
        }
    }
}

/// Errors from [`arbitrate`]. Every error is a hard refusal — this module
/// never degrades to a partial or naive score.
#[derive(Debug)]
pub enum ArbitrationError {
    /// No memories were supplied.
    EmptyMemorySet,
    /// A supplied node id is not in this graph (wrong tenant or never
    /// ingested).
    UnknownObservation(ObservationId),
    /// A supplied cluster id is not in this graph.
    UnknownCluster(ClusterId),
    /// A confidence reached the choke point non-finite or outside `[0, 1]`.
    ConfidenceInvalid(NodeRef),
    /// Configuration failed validation.
    InvalidConfig(&'static str),
    /// The downgrade-only invariant was violated — a bug, surfaced loudly
    /// rather than clamped silently.
    DowngradeInvariantViolated { naive: f64, arbitrated: f64 },
    /// A memory resolved to no provenance root, or to a root with no lineage
    /// component. Unreachable while the ADR-320 layer's guarantees hold;
    /// refused rather than panicked so a future change degrades safely.
    ProvenanceRootMissing(NodeRef),
    /// Ancestry traversal exceeded its cycle/size guard — the `causal_parents`
    /// relation is not the DAG the fusion layer guarantees.
    AncestryNotAcyclic(ObservationId),
    /// Provenance resolution failed inside the graph.
    Graph(FusionError),
}

impl std::fmt::Display for ArbitrationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArbitrationError::EmptyMemorySet => write!(f, "cannot arbitrate an empty memory set"),
            ArbitrationError::UnknownObservation(id) => {
                write!(f, "unknown observation {}", id.to_hex())
            }
            ArbitrationError::UnknownCluster(id) => write!(f, "unknown cluster {}", id.0),
            ArbitrationError::ConfidenceInvalid(node) => {
                write!(f, "non-finite or out-of-range confidence at {node:?}")
            }
            ArbitrationError::InvalidConfig(msg) => write!(f, "invalid arbitration config: {msg}"),
            ArbitrationError::DowngradeInvariantViolated { naive, arbitrated } => write!(
                f,
                "downgrade-only invariant violated: arbitrated {arbitrated} > naive {naive}"
            ),
            ArbitrationError::ProvenanceRootMissing(node) => {
                write!(f, "memory {node:?} resolved to no provenance root")
            }
            ArbitrationError::AncestryNotAcyclic(id) => write!(
                f,
                "ancestry traversal from {} exceeded its acyclicity guard",
                id.to_hex()
            ),
            ArbitrationError::Graph(e) => write!(f, "provenance resolution failed: {e}"),
        }
    }
}

impl std::error::Error for ArbitrationError {}

/// Arbitrate a retrieved memory set against its causal provenance.
///
/// Duplicate node refs in `memories` are de-duplicated first (the same memory
/// listed twice is still one memory). See the module docs for the lineage
/// rule, the downgrade-only invariant, and the non-finite choke point.
pub fn arbitrate(
    graph: &CausalEpisodicGraph,
    memories: &[NodeRef],
    config: &ArbitrationConfig,
) -> Result<ArbitrationOutcome, ArbitrationError> {
    // ── Config choke point ──────────────────────────────────────────────────
    if config.half_life_ns == 0 {
        return Err(ArbitrationError::InvalidConfig(
            "half_life_ns must be non-zero",
        ));
    }
    config.reliability.validate()?;

    let mut nodes: Vec<NodeRef> = memories.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    if nodes.is_empty() {
        return Err(ArbitrationError::EmptyMemorySet);
    }

    // ── Per-memory provenance + weight (choke point on every confidence) ────
    let mut root_memo: BTreeMap<ObservationId, BTreeSet<ObservationId>> = BTreeMap::new();
    struct MemoryFacts {
        node: NodeRef,
        roots: BTreeSet<ObservationId>,
        confidence: f64,
        weight: f64,
    }
    let mut facts: Vec<MemoryFacts> = Vec::with_capacity(nodes.len());

    for node in &nodes {
        let atomics = graph.resolve_provenance(*node).map_err(map_fusion)?;

        let confidence = node_confidence(graph, *node)?;
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(ArbitrationError::ConfidenceInvalid(*node));
        }

        // Weakest-link reliability and freshness over the node's atomic set.
        // The per-atomic confidence check below is NOT redundant with the
        // node-level check above: for a `NodeRef::Cluster` the node-level value
        // is `FusedCluster::confidence`, a *folded* number. `fuse()` now
        // rejects non-finite members at the source, but `confidence` is a
        // public field on a public struct, so a cluster carrying a laundered
        // NaN can still be constructed or mutated outside that constructor.
        // This loop is the last check between such a value and the score.
        let mut reliability = 1.0f64;
        let mut freshness = 1.0f64;
        let mut roots: BTreeSet<ObservationId> = BTreeSet::new();
        for atomic in &atomics {
            let obs = graph
                .observation(*atomic)
                .ok_or(ArbitrationError::UnknownObservation(*atomic))?;
            let c = obs.confidence as f64;
            if !c.is_finite() || !(0.0..=1.0).contains(&c) {
                return Err(ArbitrationError::ConfidenceInvalid(NodeRef::Observation(
                    *atomic,
                )));
            }
            reliability = reliability.min(config.reliability.for_code(obs.source.kind.code()));
            freshness = freshness.min(freshness_of(obs.time_ns, config));
            roots.extend(root_set(graph, *atomic, &mut root_memo)?);
        }

        // Defense in depth: the factors above are each finite in [0, 1] by
        // construction, so the product must be too — refuse loudly if not.
        let weight = confidence * reliability * freshness;
        if !weight.is_finite() {
            return Err(ArbitrationError::ConfidenceInvalid(*node));
        }
        facts.push(MemoryFacts {
            node: *node,
            roots,
            confidence,
            weight,
        });
    }

    // ── Union-find over roots: connected components = lineages ──────────────
    let all_roots: Vec<ObservationId> = facts
        .iter()
        .flat_map(|f| f.roots.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let root_index: BTreeMap<ObservationId, usize> =
        all_roots.iter().enumerate().map(|(i, r)| (*r, i)).collect();
    // `root_index` is built from exactly these root sets a few lines above, so
    // every lookup below is present by construction; they still resolve
    // through `ok_or` so this module has no panic sites at all.
    let index_of = |root: &ObservationId, node: NodeRef| {
        root_index
            .get(root)
            .copied()
            .ok_or(ArbitrationError::ProvenanceRootMissing(node))
    };
    let mut uf = UnionFind::new(all_roots.len());
    for f in &facts {
        let mut iter = f.roots.iter();
        if let Some(first) = iter.next() {
            let a = index_of(first, f.node)?;
            for other in iter {
                let b = index_of(other, f.node)?;
                uf.union(a, b);
            }
        }
    }

    // ── Assemble lineages ───────────────────────────────────────────────────
    let mut by_component: BTreeMap<usize, EvidenceLineage> = BTreeMap::new();
    for (root, idx) in &root_index {
        let comp = uf.find(*idx);
        by_component
            .entry(comp)
            .or_insert_with(|| EvidenceLineage {
                roots: Vec::new(),
                members: Vec::new(),
                weight: 0.0,
            })
            .roots
            .push(*root);
    }
    let mut naive = 0.0f64;
    for f in &facts {
        naive += f.confidence;
        // Any root works: the memory's whole root set is one component. An
        // empty root set is unreachable (every resolved atomic contributes at
        // least itself), but this returns an error rather than panicking so a
        // future change to the provenance layer degrades to a refusal.
        let first_root = f
            .roots
            .iter()
            .next()
            .ok_or(ArbitrationError::ProvenanceRootMissing(f.node))?;
        let comp = uf.find(index_of(first_root, f.node)?);
        let lineage = by_component
            .get_mut(&comp)
            .ok_or(ArbitrationError::ProvenanceRootMissing(f.node))?;
        lineage.members.push(f.node);
        lineage.weight = lineage.weight.max(f.weight);
    }
    let mut lineages: Vec<EvidenceLineage> = by_component
        .into_values()
        .filter(|l| !l.members.is_empty()) // roots only reachable via members, so always true
        .collect();
    for l in &mut lineages {
        l.roots.sort_unstable();
        l.members.sort_unstable();
        l.members.dedup();
    }
    // Deterministic order by minimal root. Every retained lineage has at least
    // one root by construction; `first()` keeps that from being a panic if the
    // construction above ever changes.
    lineages.sort_by_key(|l| l.roots.first().copied());

    let arbitrated: f64 = lineages.iter().map(|l| l.weight).sum();

    // ── Fail-loud downgrade-only invariant ──────────────────────────────────
    // Deliberately the negated form: `arbitrated > bound` is false for NaN and
    // would let a (theoretically unreachable) non-finite sum pass silently;
    // `!(arbitrated <= bound)` sends NaN into the error path instead.
    #[allow(clippy::neg_cmp_op_on_partial_ord)]
    if !(arbitrated <= naive + 1e-9) {
        return Err(ArbitrationError::DowngradeInvariantViolated { naive, arbitrated });
    }

    let verdict = ArbitrationVerdict {
        naive_confidence: naive,
        arbitrated_confidence: arbitrated,
        lineages,
    };

    if verdict.independent_lineages() >= config.sufficiency_threshold {
        Ok(ArbitrationOutcome::Sufficient(verdict))
    } else {
        let over_represented_roots: Vec<ObservationId> = verdict
            .lineages
            .iter()
            .filter(|l| l.members.len() > 1)
            .flat_map(|l| l.roots.iter().copied())
            .collect();
        let required = config.sufficiency_threshold;
        Ok(ArbitrationOutcome::InsufficientIndependentEvidence {
            verdict,
            required,
            over_represented_roots,
        })
    }
}

/// `0.5^(age / half_life)` in `f64`; future timestamps clamp to age zero.
fn freshness_of(time_ns: u64, config: &ArbitrationConfig) -> f64 {
    let age = config.now_ns.saturating_sub(time_ns) as f64;
    (0.5f64).powf(age / config.half_life_ns as f64)
}

fn node_confidence(graph: &CausalEpisodicGraph, node: NodeRef) -> Result<f64, ArbitrationError> {
    match node {
        NodeRef::Observation(id) => graph
            .observation(id)
            .map(|o| o.confidence as f64)
            .ok_or(ArbitrationError::UnknownObservation(id)),
        NodeRef::Cluster(id) => graph
            .cluster(id)
            .map(|c| c.confidence as f64)
            .ok_or(ArbitrationError::UnknownCluster(id)),
    }
}

/// Memoized, iterative root-set computation over `causal_parents`. Iteration
/// keeps deep chains off the call stack, same posture as `resolve_provenance`.
///
/// The fusion layer guarantees the parent relation is a DAG (parents must
/// pre-exist at ingest, and content addressing makes a cycle cryptographically
/// infeasible to mint), and `resolve_provenance` nonetheless carries its own
/// `visited` guard — this walk matches that posture rather than resting on the
/// invariant: a node whose parents are *still* unresolved after it has already
/// been expanded once can only mean a cycle, and is refused as
/// [`ArbitrationError::AncestryNotAcyclic`] instead of spinning forever.
fn root_set(
    graph: &CausalEpisodicGraph,
    id: ObservationId,
    memo: &mut BTreeMap<ObservationId, BTreeSet<ObservationId>>,
) -> Result<BTreeSet<ObservationId>, ArbitrationError> {
    let mut stack = vec![id];
    let mut expanded: BTreeSet<ObservationId> = BTreeSet::new();
    while let Some(top) = stack.last().copied() {
        if memo.contains_key(&top) {
            stack.pop();
            continue;
        }
        let obs = graph
            .observation(top)
            .ok_or(ArbitrationError::UnknownObservation(top))?;
        if obs.causal_parents.is_empty() {
            memo.insert(top, BTreeSet::from([top]));
            stack.pop();
            continue;
        }
        let unresolved: Vec<ObservationId> = obs
            .causal_parents
            .iter()
            .filter(|p| !memo.contains_key(*p))
            .copied()
            .collect();
        if unresolved.is_empty() {
            let mut roots = BTreeSet::new();
            for p in &obs.causal_parents {
                let parent_roots = memo
                    .get(p)
                    .ok_or(ArbitrationError::UnknownObservation(*p))?;
                roots.extend(parent_roots.iter().copied());
            }
            memo.insert(top, roots);
            stack.pop();
        } else if !expanded.insert(top) {
            // Second visit with parents still unresolved ⇒ the walk is going in
            // a circle: this node is (transitively) its own ancestor.
            return Err(ArbitrationError::AncestryNotAcyclic(top));
        } else {
            stack.extend(unresolved);
        }
    }
    memo.get(&id)
        .cloned()
        .ok_or(ArbitrationError::UnknownObservation(id))
}

/// Minimal union-find with path compression + union by size.
struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind {
            parent: (0..n).collect(),
            size: vec![1; n],
        }
    }
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }
        x
    }
    fn union(&mut self, a: usize, b: usize) {
        let (mut ra, mut rb) = (self.find(a), self.find(b));
        if ra == rb {
            return;
        }
        if self.size[ra] < self.size[rb] {
            std::mem::swap(&mut ra, &mut rb);
        }
        self.parent[rb] = ra;
        self.size[ra] += self.size[rb];
    }
}

fn map_fusion(e: FusionError) -> ArbitrationError {
    match e {
        FusionError::UnknownObservation(id) => ArbitrationError::UnknownObservation(id),
        FusionError::UnknownCluster(id) => ArbitrationError::UnknownCluster(id),
        other => ArbitrationError::Graph(other),
    }
}
