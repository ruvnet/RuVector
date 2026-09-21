# ADR-004: The self-optimization loop — HPO-weighted, gated, reversible

## Status
Implemented and measured (2026-09-21); optimize campaign pending

## Date
2026-09-21

## Context

The single most load-bearing finding for a self-improving KGE engine is that
**hyperparameters and training strategy matter more than model architecture**
(Ruffinelli et al., ICLR 2020; KGTuner, arXiv:2205.02460 — a well-tuned RESCAL
matches later models, and good configs are found from few random samples). This
reorders the priority typesafe ADR-004's own Loop 2 used (which ranked
model-arm selection above HPO): **for KGE, HPO arms must weigh at least as
heavily as model-family arms.**

The gating machinery this loop needs already exists and is deliberately reused
rather than restated: typesafe ADR-004 (frozen splits, paired anytime-valid
sequential test, transfer holdout, permanent control arm, per-day budget, trust
tiers A/B/C, receipts) and ruvector ADR-276 (its evidence base). Continual
updates reuse `ruvector-gnn`'s `ElasticWeightConsolidation` (ewc.rs).

## Decision

Loops as *proposals*, each passing the one promotion gate defined in **typesafe
ADR-004 §"The promotion gate"** — this ADR does not restate that gate; it adopts
it by reference and records only the KGE-specific deviations.

### Loop 1 — HPO (ranked first, the KGE deviation)

- Arms: embedding dimension `d`, negative-sampling strategy (self-adversarial vs
  1-vs-all), loss, regularizer (N3 vs L2), learning-rate schedule, optimizer
  (Adam vs Adagrad). **These are weighted at least as heavily as the model-family
  arm** — the direct consequence of the LibKGE/KGTuner finding.
- Thompson sampling with a cost-aware reward `MRR − λ·(train_time + score_ms)`,
  seeded by a one-time offline sweep on the deployment's frozen validation split
  so cold start is not uniform.

### Loop 2 — model arm (HolE vs RotatE)

- Two arms only in v1: the HolE default and the RotatE composition scorer.
  Selection is bandit-driven but subordinate to Loop 1's HPO arms. Split per
  relation-type only after a minimum sample count (ADR-271's per-category
  finding: a per-category router helps with data, hurts when scarce).

### Loop 3 — continual update on graph growth (EWC)

- New/changed triples are a delta; fine-tune on the delta with
  `ElasticWeightConsolidation` (`crates/ruvector-gnn/src/ewc.rs`) as-is — entity
  and relation matrices are just parameter tensors to its Fisher-diagonal
  penalty `L_EWC = λ/2 Σ Fᵢ(θᵢ − θ*ᵢ)²`.
- **Stated v1 limitation, not silently absorbed:** standard EWC only penalizes
  drift in *existing* entity embeddings; *new* entities from graph growth can
  interfere with old ones by competing for score mass — a forgetting mode the
  Fisher penalty does not cover (arXiv:2604.19401, 2026). v1 documents this;
  v2 revisits with incremental distillation (arXiv:2405.04453) or incremental
  LoRA (arXiv:2407.05705) if EWC-only proves insufficient in our own benchmarks.
- Promotion evaluates **both** old-graph and new-graph filtered MRR before
  accepting — the loop cannot trade old accuracy for new.

### Loop 4 — criteria/schema mutation (v2, off by default)

- Relation-description or namespace text as a genome, the trained scorer frozen;
  highest-risk loop, strictest budget, off in v1. Same posture as typesafe
  ADR-004 Loop 4.

### KGE-specific gate additions

The typesafe gate applies unchanged, plus:

1. **The bandit/anytime-test marriage is a transplant, not a validated KGE
   result** — no KGE-specific literature marries Thompson sampling with
   anytime-valid testing (brief Q4 gap). Decoupling exploration policy from the
   promotion gate is the *safer* design, not a proven one. It is gated by:
   transfer-holdout non-regression (promote on FB15k-237, transfer-check a
   CoDEx-M slice) **and** a CI test that injects a known regression into a
   candidate arm and asserts the permanent control arm alarms.
2. **Receipts** carry the model manifest hash, scorer id, HPO config, val/
   transfer MRR with the test statistic, EWC λ and Fisher snapshot id, and an
   RVF lineage link via `rvf-manifest`'s `lineage_depth`
   (`crates/rvf/rvf-manifest/src/level0.rs`) so `kge eval --history` and
   rollback are exact.

## Consequences

- Ranking HPO first means most promotions are tuning wins, matching the
  literature — the engine improves by search, not by swapping architectures.
- The EWC path keeps continual updates cheap and reversible but carries a named,
  benchmarked limitation rather than a hidden one.
- Two extra splits (transfer) and a control arm consume evaluation budget;
  ADR-006 sets the minimum graph/label sizes below which loops stay propose-only.

## Alternatives considered

- **Weight model-family arms above HPO** (typesafe's own Loop 2 ordering).
  Rejected for KGE: contradicts the LibKGE/KGTuner finding that tuning dominates.
- **Greedy accept-if-better on validation MRR.** Rejected: the poisoning and
  continual-KGE literature both show it is exploitable and forgetting-prone; the
  paired anytime-valid test and control arm exist to refuse it.
- **Non-regularized fine-tune for continual updates.** Rejected: catastrophic
  forgetting; EWC is the documented base case with a stated gap for new entities.

## Evidence

- HPO dominance: Ruffinelli et al., ICLR 2020 (LibKGE); KGTuner
  arXiv:2205.02460. Continual KGE: arXiv:2604.19401 (EWC new-entity gap),
  arXiv:2405.04453 (distillation), arXiv:2407.05705 (LoRA).
- Gate machinery adopted by reference: **typesafe ADR-004** at
  `/home/ruvultra/projects/ruvector-worktrees/feat-typesafe/npm/packages/typesafe/docs/adr/ADR-004-self-optimization-loop.md`;
  ruvector ADR-276, ADR-271.
- Repo reuse: `crates/ruvector-gnn/src/ewc.rs`
  (`ElasticWeightConsolidation`); `crates/rvf/rvf-manifest/src/level0.rs`
  (`lineage_depth`); `crates/sona/src/darwin_guard.rs`.
- Unmeasured: Thompson-sampling-plus-anytime-testing has no KGE-specific
  validation (brief Q4) — treated as a gated transplant.
