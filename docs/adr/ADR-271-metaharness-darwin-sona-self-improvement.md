# ADR-271: Metaharness-Darwin for SONA Self-Improvement — EWC Config Evolution, the weightAdapter Gene, and Ornith-1.0 Reward-Hacking Defenses

- **Status**: Proposed (all four components prototyped — PR #615)
- **Date**: 2026-06-27
- **Extends**: ADR-266 (metaharness-Darwin ANN optimization), ADR-269/270 (mragent graph-memory Darwin)
- **External anchor**: Ornith-1.0 "Self-Scaffolding LLMs for Agentic Coding" (DeepReinforce, Jun 2026)

---

## Context

SONA (`crates/sona`) is a self-learning substrate: online LoRA adaptation + **EWC++** (`EwcPlusPlus`) to resist catastrophic forgetting, recording `QueryTrajectory` rewards. Its behaviour is governed by hand-tuned hyper-parameters (`EwcConfig`: lambda schedule, Fisher decay, task-boundary threshold; LoRA rank; MoE gate). These are **non-differentiable, workload-dependent** meta-parameters — gradients tune the weights, but nothing tunes the *config that governs how the weights are tuned*.

The metaharness line (ADR-266/269/270) established "freeze the model, evolve the harness" via **external evolutionary search**. This ADR applies that line to SONA's continual-learning layer and hardens it using the structure of **Ornith-1.0**, which does the same thing the *other* way (in-weights RL co-optimizing scaffold + solution).

## Decision

Apply metaharness-Darwin to SONA across four components. The **frozen model** is the EWC++/LoRA *algorithm*; the **evolved harness** is its config genome. Weights stay on the gradient path — Darwin only ever touches the meta-layer.

### 1. EWC++ config evolution (implemented — PR #615)
Evolve the `EwcConfig` genome (GA + coordinate-descent polish) on a continual-learning benchmark with **no replay** and **self-detected task boundaries**, train/test split over task-sequence seeds. Measured (held-out): **35% lower final loss, 98.6% less forgetting** than `EwcConfig::default()` (the crate's hand-tuned "OPTIMIZED" values) — a strict Pareto win that generalizes to unseen sequences. (`examples/darwin_ewc.rs`)

### 2. The `weightAdapter` gene (implemented — PR #615)
Expose a fine-tuned adapter delta (a LoRA) as a gene `(which_adapter, alpha)` so evolutionary selection decides *whether/how much* to apply it (`w_eff = w_base + alpha·Δw`) rather than assuming new weights are better. **Key finding** (`examples/darwin_weightadapter.rs`): "selection prunes overfit adapters" holds **only under per-domain (no-regression Pareto) evaluation**. A volume-weighted aggregate fitness is fooled by an adapter whose in-dist gain (where the eval pool concentrates) hides an out-dist regression. → **Score every adapter per-repository.**

### 3. The Autonomous Data Engine (proposed)
Darwin's archive is **execution-verified** preference data — the label is a *passing test suite*, not a noisy human judgment (higher-signal than RLHF). The plumbing exists (`ruv_brain_export_pairs`/`write_preference_pair`; the SWE-bench `solve*.mjs` trajectory archive). Two refinements: (a) DPO negatives should be **plausible-but-failed** patches on the same issue (teach "this reasoning was wrong"), not empty no-ops; (b) the archive is biased toward *solvable* issues — balance/weight it.

### 4. Ornith-1.0 borrows (method, not model)
Ornith bakes scaffold-evolution into weights via RL; we keep it external (cheaper, model-agnostic, no training). We borrow its *structure*:

- **3-layer reward-hacking defense** (the `darwin-guard` module): (i) **immutable outer boundary** — the verifier/eval is frozen and outside what evolves; (ii) **deterministic monitor** — gated variants (new imports/network/shell/env, reading withheld paths, touching the verifier) are **excluded from the advantage/Pareto computation**, not merely zero-scored, so they cannot bias selection; (iii) **frozen LLM judge as a veto** (local GPU `qwen`) on intent-level Goodharting inside the allowed surface — a veto on top of the verifier, never the primary reward.
- **Per-task-category specialization**: evolve a router `task-class → genome` instead of one global genome (Ornith's main empirical result is per-category strategies emerging).
- **Two-stage reward credit**: credit the *mutation/scaffold-proposal* that produced a winning genome, not just the outcome — turning the random `mutate()` into a learned write-layer (and the `(proposal → outcome)` pairs are themselves data-engine preference pairs).
- **Staleness-weighted replay** `w(d_t)` (1 if fresh → exp-decay → drop past threshold) for the online auto-tuner over SONA's live trajectory stream; maps onto `fisher_ema_decay` and is itself evolvable.

## Consequences

**Positive**: out-tunes hand-tuning on held-out continual learning (measured); model-agnostic and training-free (vs Ornith's GPU-scale RL); the reward-hacking defenses make the loop rigorous and Goodhart-resistant; the same Darwin genome co-optimizes "adopt this fine-tune?" with "how hard to protect old knowledge?".

**Negative / risks**: a beyond-SOTA number is only as real as the benchmark — the immutable-verifier boundary (borrow #4-i) is what keeps it honest; meta-optimization cost scales with benchmark realism (real nets ⇒ GPU + parallelism); generalization across *workload distributions* (not just task sequences) likely needs the per-category router (#4-ii), not one frozen genome.

## Relationship to Ornith-1.0
| | Ornith-1.0 | This ADR |
|---|---|---|
| Harness optimization | in-weights RL (gradients), two-stage | external evolutionary (GA/Pareto) |
| Cost | frontier RL training (9B–397B) | training-free, any frozen model |
| Reward-hack defense | immutable boundary + monitor + judge | **borrowed verbatim** (darwin-guard) |
| Specialization | per-task-category (emergent) | per-task-category router (borrowed) |

Complementary, not competing: external-Darwin is the no-training counterpart to Ornith's in-weights approach.

## Implementation status
- ✅ EWC config evolution + weightAdapter gene (PR #615, `feat/sona-darwin-ewc-evolve`).
- ✅ darwin-guard reward-hacking module (`crates/sona/src/darwin_guard.rs`, 4 tests; wired into `darwin_ewc`).
- ✅ per-task-category router (`examples/darwin_router.rs`): beats the single best global config on held-out (~2%), with the **data-efficiency caveat** — the gain *reverses* when per-class data is scarce (a specialized config overfits while the pooled global generalizes), so routing needs enough per-category samples (Ornith's regime).
- ✅ online auto-tuner with staleness-weighted replay `w(d_t)` (`crates/sona/src/auto_tuner.rs` — `StalenessSchedule`/`StalenessWindow`, 4 tests; `examples/darwin_autotuner.rs` — a (1+1)-ES that adapts a deployed config to workload drift, beating the static config ~3% post-drift). Modest margin on synthetic regimes; the durable win is the reusable staleness machinery + the online-adaptation principle (a fixed offline-tuned config goes stale under drift).

## References
- Ornith-1.0: "Self-Scaffolding LLMs for Agentic Coding", DeepReinforce, 2026-06.
- ADR-266 metaharness-Darwin; ADR-269/270 mragent; PR #615.
