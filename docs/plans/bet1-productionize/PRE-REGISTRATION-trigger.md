# BET 1 follow-up — Sampled-recall rebuild trigger vs fixed periodic-K

**Status:** Pre-registered (gate frozen before any contender run) · **Date:** 2026-06-04 ·
**Research line:** SepRAG (ruvnet/RuVector issue #534) · **Extends:** ADR-202 (BET 1
productionized WIN), ADR-200 next-step #2 · **Self-contained:** `ruvector-diskann` +
`ruvector-gnn` only · **Outcome:** ADR-202 addendum (WIN *or* KILL).

> Pre-registration, committed before the harness runs. A loss is acceptable and reportable
> (ADR-200's own Frobenius trigger lost — that is the precedent). Editing the gate after seeing
> results voids the bet. Plumbing (`DriftingIndex::force_rebuild` + harness) may precede freeze;
> the contender run may not.

> **OUTCOME: WIN** (2026-06-04) — see [ADR-202 addendum](../../adr/ADR-202-reuse-under-drift-real-gnn-trajectory.md#addendum-2026-06-04-sampled-recall-trigger--win).
> On bursty drift (n=20k, 89% end churn), `Recall{floor=0.95}` = 97.2% recall @ 7 rebuilds beat
> `Periodic{k=2}` (96.8% @ 12) on both axes and the best `Frobenius` (97.3% @ 9) on rebuilds;
> probe cost (~1s) was <2% of the ~73s rebuild time saved. Productionized as
> `ruvector_diskann::reuse::RecallTrigger`. **Note:** the first run was VOID (plain-SGD trajectory
> drifted 0%); switched the generator to Adam and enforced the ≥15% churn precondition — the
> WIN/KILL gate itself was unchanged.

## Prove-not-hype protocol (all five)

1. One claim, one number. 2. Beat the strongest in-repo incumbent (here: `Periodic{k}`, the
ADR-202 winner) tuned. 3. Public data + ground truth (ogbn-arxiv). 4. Pre-register WIN + KILL.
5. Adversarial check (here: the **probe-cost honesty trap** — the trigger's own measurement cost
is counted, so it can't win by ignoring it).

## Thesis (one claim, one number)

> Under **variable-rate** drift, a sampled-recall-triggered rebuild matches `Periodic{k}`'s
> recall floor (within 1%) at **≥ 25% fewer rebuilds**, with the probe's own distance-eval cost
> counted — and uses fewer rebuilds at matched recall than the **Frobenius-norm monitor** ADR-200
> found wanting.

## Why variable-rate drift is the honest stage (central insight)

`Periodic{k}` is near-optimal under **steady** drift (ADR-202). A trigger can only earn its keep
when drift is **bursty**: calm stretches where a fixed cadence over-rebuilds, bursts where it
under-rebuilds. The trajectory therefore alternates high-lr bursts (3 epochs, lr 0.03) and
low-lr calm (5 epochs, lr 0.002) on the same arxiv contrastive objective. If the trigger cannot
beat periodic *there*, it cannot beat it anywhere — clean KILL.

**Mechanism (falsifiable):** Frobenius measures *how much the metric moved*; recall measures
*whether the move broke navigability*. ADR-202 showed those decouple (40% churn cost ~0 recall),
so a recall probe should track the thing we care about and the norm monitor should not.

## Contenders

| Trigger | Role |
|---|---|
| `Recall{floor}` (sweep {0.97, 0.95, 0.93}) | **the bet** — rebuild when a probe-set recall estimate drops below `floor` |
| `Periodic{k}` (sweep {2, 3, 4, 6}) | incumbent (ADR-202 winner) |
| `Frobenius{τ}` (sweep {0.15, 0.25, 0.40}) | the monitor ADR-200 found wanting — must be beaten |
| `Always` (k=1) | cost ceiling reference |

Index built once on `E₀` (`ReweightOnly` so `on_metric_update` never auto-rebuilds);
`force_rebuild` driven by each trigger. Production Vamana R=32/L=64/α=1.2; recall@10; 200 scored
queries; **30 disjoint probe queries** (no leakage into the scored set). n=10k (ADR-202 already
established scale-robustness; this bet isolates *cadence*, where rebuild count is the signal).

## Pre-registered gate

- **Honest comparison = the (rebuilds, recall) Pareto frontier**, not a cherry-picked single
  config. For each `Recall{floor}`, find the cheapest `Periodic{k}` matching its recall (within
  0.5%); the trigger wins that cell iff it used **≥ 25% fewer rebuilds**.
- **Probe-cost honesty trap (counted):** the recall probe costs `probe_size × n` distance-evals
  per step. Reported in the trigger's ledger; a rebuild-count win whose probe cost exceeds the
  saved rebuild cost is **not** a WIN.
- **WIN:** some `Recall{floor}` is within 1% recall of the best `Periodic{k}` at ≥ 25% fewer
  rebuilds, net cost (rebuilds + probes) below that periodic, **and** strictly fewer rebuilds
  than the best `Frobenius{τ}` at matched recall.
- **KILL (reportable, like ADR-200's Frobenius result):** no `Recall{floor}` cell beats periodic
  by ≥ 25% fewer rebuilds at matched recall, **or** the probe cost eats the savings, **or** it
  merely ties Frobenius. Then ADR-200's "periodic-K is the recommended knob" stands, reinforced.

## Where it lives

- Primitive: `DriftingIndex::force_rebuild(vectors)` (shipped in `ruvector-diskann::reuse`, the
  clean mechanism an external trigger drives). The `Recall` trigger stays in the harness until it
  earns productionization — `RebuildPolicy` keeps only self-contained policies for now.
- Harness: `crates/ruvector-gnn/examples/triggered_rebuild.rs`.
- Same branch / PR #537; outcome as an ADR-202 addendum.

## Out of scope

- Steady-drift regime (periodic already owns it — ADR-202).
- Productionizing the trigger as a `RebuildPolicy` variant (only if it WINS).
- Larger n (scale is ADR-202's domain; this is the cadence question).
