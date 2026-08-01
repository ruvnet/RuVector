---
adr: 278
title: "rvAgent Self-Learning: Adopt the metaharness Flywheel; Shift from Memory to Policy"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-271, ADR-273, ADR-275, ADR-276, ADR-277]
external: [metaharness ADR-226, metaharness ADR-228, metaharness ADR-236]
tags: [rvagent, self-learning, flywheel, gepa, policy-evolution, metaharness, promotion, sota]
---

# ADR-278 — rvAgent Self-Learning: Adopt the metaharness Flywheel; Shift from Memory to Policy

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Sources: `@metaharness/flywheel@0.1.7` (`/workspace/metaharness/packages/flywheel`),
metaharness ADR-226 / ADR-228 / ADR-236, and
`docs/research/rvagent-hermes-harness/04-sota-landscape.md`.

## 1. Decision

1. **Adopt `@metaharness/flywheel` as rvAgent's promotion engine.** Do not build
   one. Do not port it to Rust.
2. **Shift self-learning investment from memory accumulation to policy
   evolution.** ReasoningBank/SONA stays gated per ADR-276; GEPA-style policy
   evolution becomes the primary self-learning mechanism.
3. **Adopt `noopRate` as a first-class score axis.**
4. **Contribute anytime-valid sequential testing back to the flywheel gate**
   rather than only consuming it.
5. **Record the metaharness ADR-226 null as a binding constraint** on rvAgent's
   subagent design (see ADR-275 amendment).

## 2. Why not build our own

ADR-276 §3.4 specified a promotion apparatus from scratch. `@metaharness/flywheel`
already implements it, and more rigorously.

It is deliberately host-agnostic — a stated design rule forbids any host, model,
or benchmark from leaking into the package. Its entire vocabulary is
`Policy = Record<string, string>`, `Score`, `PromotionEvidence`, and a
`PromotionRule`; everything rvAgent-specific enters through an injected
`Evaluator`. There is no adapter impedance to pay.

What it already provides that ADR-276 was specifying:

| ADR-276 requirement | Flywheel |
|---|---|
| Promotion gate | `meetsPromotionRule` — **frozen, conjunctive**, every clause load-bearing |
| Proof the gate did not move | `gateFingerprint()` — SHA-256 over the rule source |
| Transfer holdout | `HoldoutSuite` **plus** a frozen `AnchorSuite` never optimized against |
| Audit trail | Ed25519 `PromotionReceipt` + `verifyReceipt()` |
| Independent verification | `verifyReplayBundle()` — reviewer trusts the signature, not us |
| Compounding, not scattering | Lineage DAG re-basing on the promoted winner; `computeLiftCurve()` |

The anchor deserves emphasis: ADR-276 §4 asked only for a transfer holdout. The
flywheel requires a candidate to clear a holdout **and** a frozen suite it is
never optimized against. That is a strictly stronger anti-Goodhart guard than we
specified.

`verifyReplayBundle()` is also, in substance, the replay-verification story
ADR-277 §3.2 positions on — working, in JS, today.

## 3. The `noopRate` clause

The default gate's second clause requires the no-op rate to **strictly** improve:

> a policy earns a promotion by making the executor COMMIT more, not just score higher

This is non-obvious and load-bearing. A policy that raises the primary metric
while leaving the executor more likely to end empty has not improved the agent;
it has found a scoring artifact.

**Adopt this axis in rvAgent's own scoring.** It pairs naturally with ADR-273's
reliability framing — "never end empty" is a reliability property, and the
+54.3-point patch-application result is the same phenomenon measured a different
way.

## 4. Memory versus policy — the reframe

These are different objects with different evidence:

| | Object | 2026 evidence |
|---|---|---|
| Flywheel / GEPA | **Policy text** — a genome of named string levers | Positive; GEPA is the best-evidenced optimizer in the sweep |
| ReasoningBank / SONA | **Episodic memory** — accumulated trajectories | Negative; inverted-U, confound-sized gains, self-memory underperforming plain retrieval |

RuVector's self-learning weight currently sits on the memory side, which is the
side the evidence argues against. **Move the weight to policy evolution.**

metaharness ADR-228 reaches the same conclusion from its own measurements:
redirect strong-model judgment *offline into the executor's standing operating
policy* rather than injecting it as runtime advice. It notes GEPA's candidate is
a `dict[str,str]` of named text components, matching the flywheel's `Policy`
exactly — the same shape rvAgent would supply.

ADR-276 is not repealed. Its gating, trust tiers, and inverted-U regression
metric remain the conditions under which memory may ever be enabled. This ADR
changes where *new* effort goes.

## 5. Two nulls we must respect

### 5.1 ADR-226 — the read-only advisor is dead

A frontier read-only advisor over a cheap executor produced **zero marginal
gold-scored resolves at 5.4× cost**. The advisor was genuinely active — 33
advisories and 3 vetoes across the slice — not silently disabled. The track was
killed.

This **independently corroborates** the figure in ADR-275 §3.2 from the public
literature (a frontier model in the read-only slot: +0.4 pp at 5.8× cost). Two
independent measurements, near-identical cost multiple. Treat the conclusion as
established rather than provisional: **do not put an expensive model in a
read-only slot.**

It also constrains ADR-275 §3.1 — see §7.

### 5.2 ADR-236 — a promotion engine cannot rescue a weak loop

The flywheel mechanism was proven end-to-end on real SWE-bench and still
produced **no compounding lift, because the base solver was too weak.** Recorded
as an honest null rather than buried.

**Consequence for sequencing:** adopting the flywheel does not shorten ADR-273.
The reliability floor comes first; the flywheel amplifies a loop that already
works and does nothing for one that does not. This validates the ordering
ADR-273 already set.

## 6. What we contribute back

The flywheel's gate is a **single-shot** conjunctive comparison. Running many
generations against the same holdout is uncontrolled multiple testing — the
regime where PACE (arXiv 2606.08106) measured **30–42% false commits**, and 13–21
spurious modifications even when no true gains existed.

The conjunctive gate plus frozen anchor mitigates this with multiple hurdles,
which is real but is not anytime-valid. **Frozen conjunctive gate ∧ anytime-valid
sequential test is strictly stronger than either**, and PACE reported ~18% lower
evaluation cost as a side effect.

This is an upstream contribution to `ruvnet/metaharness`, offered as an optional
`PromotionRule` plus a sequential-evidence accumulator — not a change to the
default gate, whose stability is itself the product.

## 7. Amendment to ADR-275

ADR-275 §3.1 adopted a fresh-context reviewer subagent on the strength of
Cognition's production data (~2 bugs/PR, 58% severe). ADR-226 is the closest
*measured null* to that design and was not considered when §3.1 was written.

The distinction that may preserve it: **ADR-226's advisor received the full
transcript**, whereas the fresh-context reviewer receives only the diff — and the
Cognition finding is specifically that reviewers perform *better* without shared
context. ADR-226 therefore does not refute §3.1, but it is the strongest nearby
negative result.

**ADR-275 §3.1 is downgraded from adopted to gated.** The reviewer must
demonstrate marginal lift over a no-reviewer control on the same instances
before it goes on the default path, and ADR-226's design is the specific null it
must beat. §3.2 (read-only gatherer on a cheap model) is unaffected and is in
fact strengthened by §5.1.

## 8. Integration

**No Rust port.** Promotion is offline; the flywheel is not on the hot path.

- Run it in CI at the existing ruflo/metaharness seam (roadmap Phase 3).
- rvAgent supplies an `Evaluator` mapping a run onto the four `Score` axes, and
  a `Proposer` for the mutation seam.
- rvAgent's policy genome is the natural `Policy`: system-prompt components,
  compaction rubric (ADR-274 §3.4), `loop_repeat_threshold`, masking
  `keep_last_observations`, tool-surface composition, per-role model tiers.
- metaharness ships `crates/kernel-napi` as the in-process bridge pattern if we
  later need it. We do not need it now.

**Gap this closes:** ADR-271 (`metaharness-darwin-sona-self-improvement`) does
not reference the flywheel at all. This seam was previously unrecorded on our
side.

## 9. Consequences

**Positive.** Deletes the largest unbuilt subsystem in ADR-276 — we consume a
maintained engine instead. Inherits an audit and replay story that already
exists. Moves self-learning onto the side of the evidence. Turns a
one-directional dependency into a two-way exchange (§6).

**Negative.** A cross-repo dependency on a package at `0.1.x`. Mitigated because
the flywheel is thin, runtime-dependency-free (Node `crypto` only), fully typed,
and — being offline — a version pin is low-risk.

**Risk.** rvAgent's four `Score` axes must be projected honestly. `primary`,
`noopRate`, `costPerWin`, and `regressed` are where all host meaning lands, and
a dishonest projection defeats every downstream guarantee. The Evaluator is the
trust boundary.

## 10. Implementation order

1. Record the ADR-275 §3.1 downgrade (§7) — documentation only
2. Add `noopRate` to rvAgent's score axes (§3)
3. Define the rvAgent policy genome (§8)
4. Evaluator mapping a headless run onto the four axes
5. Wire `runFlywheelGenerations` into CI at the ruflo seam
6. Upstream the sequential-testing `PromotionRule` to metaharness (§6)
