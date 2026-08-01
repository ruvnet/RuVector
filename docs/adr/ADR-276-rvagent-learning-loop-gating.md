---
adr: 276
title: "rvAgent Learning Loop: Gating, Trust Tiers and Measurement"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-271, ADR-273, ADR-274, ADR-275, ADR-277, ADR-323]
tags: [rvagent, harness, memory, reasoningbank, sona, self-improvement, evaluation, security, sota]
---

# ADR-276 — rvAgent Learning Loop: Gating, Trust Tiers and Measurement

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Evidence base: `docs/research/rvagent-hermes-harness/04-sota-landscape.md` §5–6.

**This ADR constrains a standing design bet.** `03-roadmap.md` Phase 2.4 puts
"SONA on the default path". The 2026 literature turned substantially against
this class of system. The component is not cancelled — it is gated behind
measurement it must earn.

## 1. Decision

1. **Trajectory-learning memory ships feature-gated OFF by default.**
2. **The measurement apparatus is a precondition for enabling it**, not a
   follow-up.
3. **A permanent memory-off control arm** runs for the life of the system.
4. **Episodic storage is immutable**; distilled artifacts are derived.
5. **Consolidation is gated and delta-only** — never end-to-end rewrites.
6. **Trust tiers by verifier**, with untrusted-derived memories barred from
   influencing permission or destructive-action decisions.
7. **Promotion uses anytime-valid sequential testing**, never greedy
   accept-if-better.

## 2. Why the constraint

The case *for* is real and peer-reviewed: ReasoningBank reports +4.6 to +8.3
points on WebArena across three backbones (ICLR 2026); ACE reports +10.6% on
agents. Distilling from *failures* as well as successes is a genuine
contribution over success-only baselines.

The case *against* is now stronger:

- **The gains are confound-sized.** MemDelta (arXiv 2606.29914): swapping the
  embedding model alone shifts accuracy **±6.2 pp** — comparable to
  ReasoningBank's entire headline gain. Without controlled ablation you cannot
  distinguish "our memory design works" from "we picked a better embedder." In
  the same work, **agent self-memory (42%) underperformed plain retrieval
  (47%)**, and one system reached parity with cloud RAG at **50× the cost**.
- **Memory utility is an inverted U.** "Useful Memories Become Faulty"
  (arXiv 2605.12978): utility rises, then degrades *below* the no-memory
  baseline. GPT-5.4 failed **54% of previously-solved ARC-AGI problems** when
  using consolidated memory. **Episodic-only management doubled accuracy** vs
  forced consolidation — the consolidation step is the bug, not the storage.
- **Gains are benchmark-local.** MemoryArena specifically names ReasoningBank's
  procedural memory as performing poorly on interdependent multi-session tasks
  — the setting closest to real work.
- **No automatic self-evolution method sustains positive gain across settings**
  (EvoAgentBench, arXiv 2607.05202). *Curated* ability content transfers across
  model families; *automatic extraction* is the failure point.
- **Greedy acceptance is uncontrolled multiple testing.** PACE (arXiv
  2606.08106): "keep it if the score improved" committed **30–42% false
  edits**, and made 13–21 spurious modifications when *no true gains existed*,
  degrading one agent by 4.9 points.

Meanwhile **Live-SWE-agent reaches 79.2% on SWE-bench Verified with zero
persistent memory** — on-the-fly tool synthesis from the current trajectory,
discarded after use. It sidesteps every failure mode above. That is not a
coincidence: nothing persistent means nothing to poison, stale, or collapse.

**Conclusion.** Trajectory-learning memory is a nice-to-have with fragile
upside, not a differentiator (see ADR-277 for what the differentiators are).

## 3. Design

### 3.1 Two-tier storage

**Episodic is immutable.** Append-only raw trajectory store; never rewritten,
never overwritten. Distilled playbook items are *derived artifacts* carrying
pointers back to their source episodes.

Rationale: consolidation is the documented failure point, and episodic-only
management doubled accuracy against forced consolidation. Raw episodes remain
primary evidence.

### 3.2 Gated, delta-only consolidation

Never run consolidation automatically after each task. Never rewrite the
playbook end-to-end. Append or amend individual items with structured deltas.

Hard per-item length cap (~1,500 chars) and a hard total cap. ACE documents a
single end-to-end rewrite collapsing **18,282 tokens → 122**, dropping
performance *below* the no-adaptation baseline. Production experience
independently shows unconstrained growth past 5,000 chars overfits, and that
length regularization is nearly free (4× compression for −0.8%).

More data made it worse: 500 samples grew prompt length +75% and *dropped*
performance 2% versus a 20–100 sample sweet spot.

### 3.3 Trust tiers

Every candidate memory carries provenance: source episode, verdict source,
verifier type, timestamp.

| Tier | Backed by | Retrieval |
|---|---|---|
| **A — active** | Programmatic verifier: tests pass, type check, schema validation, invariant assertion | Full weight |
| **B — quarantine** | LLM-judge verdict only | Reduced weight, or withheld until promoted |
| **C — tainted** | Derived from untrusted content: fetched pages, tool output, user-supplied text | Separate namespace. **Never** allowed to influence tool-permission or destructive-action decisions |

Tier C is a security boundary, not a quality heuristic. See §5.

### 3.4 Promotion by sequential testing

Promotion B → A, and any prompt or scaffold edit, requires a **paired
anytime-valid sequential test** (e-process / testing-by-betting) against the
current version on identical held-out instances. Commit only when evidence is
decisive.

PACE achieved comparable accuracy at **~18% lower evaluation cost** than greedy
acceptance while eliminating the false-commit rate. This is the single
highest-value component of the learning loop.

### 3.5 Retrieval discipline

- Inject retrieved memories framed explicitly as **"references, not rules"**.
  Nearly free, and drops attack success **20.6% → 13.1%** while raising the
  refusal rate 54.4% → 66.9%.
- Relevance-gated top-k with a threshold. **Never concatenate the whole bank** —
  ExpeL's documented scaling failure.
- Hard token budget on injected memory; over budget, drop lowest-trust first.
- TTL, decay, and eviction on realized contribution.

## 4. Measurement — the precondition

The gate does not open until these run.

**Primary metric — paired net lift:**
`lift = P(success | memory) − P(success | no memory)` on *the same instances*,
with PACE's e-process providing the stopping rule. Report a confidence
interval, not a point estimate.

**The regression metric that matters most:** rate of **previously-solved tasks
that now fail with memory on**. This is the 54%-on-ARC-AGI signal and it is the
earliest warning that consolidation has gone bad. Track per consolidation
event; above threshold, roll back and quarantine the items it produced.

**Permanent control arm.** A fraction of traffic — or a shadow run on a frozen
held-out suite — always executes with memory disabled. Not a one-time
ablation: without a live control the inverted-U crossover is undetectable.

**Confound controls (run before believing any result):**

- Hold the embedding model **fixed** across arms; report sensitivity separately.
  A ±6.2 pp swing from the embedder alone masquerades as an architecture win.
- Hold the backbone LLM fixed; re-verify on a second backbone.
- **Log refusal rates per arm.** A 63%-refusal arm is not comparable to a 5% one.
- Always include a **plain-retrieval baseline** (BM25 or vanilla embedding RAG
  over raw episodes). Failing to beat it is a 50×-cost parity result, not a win.

**Cost-normalized:** tokens and dollars per *additional* success, not raw
accuracy.

**Transfer holdout:** a task set from a *different distribution* than the
memories were written from. In-distribution gain is expected and tells you
almost nothing.

**Per-item attribution:** track retrieval count and conditional lift when
retrieved; evict items with negative or non-significant contribution. This
makes the bank self-pruning and gives an audit trail when something poisons it.

## 5. Security: the shared brain is the highest-risk surface

Cross-agent shared memory multiplies the blast radius. Measured
memory-poisoning work reports **~50% attack success and ~41% relapse success**,
with **contextual assimilation** as the primary vector — poisoned entries work
best when they look like ordinary preferences, constraints, or workflow
requirements. Reported >90% of tested agents vulnerable, with **100% relapse**
when teams tried to fix it conversationally.

Implications:

- Tier C (§3.3) is mandatory and must be enforced structurally, not by prompt.
- Never write raw credentials, PHI, or secrets to shared memory (already policy).
- Sleeper entries may lie dormant until triggered — per-item attribution (§4)
  is the detection mechanism.
- "Misevolution" affects top-tier models; the cheapest known mitigation is the
  references-not-rules framing of §3.5.

## 6. Verifier quality is the binding constraint

Any learning loop is only as good as its verdict signal, and LLM-as-judge
evidence is poor: a judge surfaced **under 25%** of human-confirmed systematic
problems and **flagged zero** issues in a batch where humans confirmed 23
distinct defects — implying a **3–6× undercount**. Blind spots are structural:
it catches turn-local problems and is severely blind to cross-turn state.
Calibration drifts — one judge at 0.91 agreement shifted four points after a
model update.

**Rules:**

- Programmatic and execution-grounded signals first (tests, type checks, schema
  validation, invariants).
- LLM judge is a **secondary, quarantined** signal only (Tier B).
- Treat the judge as a **regression floor, never a promotion authority.**
- Re-anchor against a rolling human-labeled sample after **every** model update.

## 7. Consequences

**Positive.** The component ships honestly. If it works we can prove it; if it
degrades we detect the crossover instead of shipping a silent regression. The
measurement apparatus (§4) is reusable for every other harness change and
overlaps with the eval-loop investment ADR-273 §6 already requires.

**Negative.** Slower to enable than "SONA on the default path". Significant
work lands before any measured benefit. This is the correct trade given §2 —
the alternative is enabling a component whose own literature says it may go
below baseline.

**Relationship to ADR-271.** ADR-271's Darwin/SONA self-improvement direction
remains valid as *mechanism*. This ADR supplies the gating and acceptance
criteria it lacked, and replaces greedy fitness acceptance with §3.4.

## 8. Implementation order

1. Episodic append-only store (immutable, §3.1)
2. Measurement harness: paired lift, previously-solved regression rate, control
   arm, plain-retrieval baseline (§4)
3. Trust tiers with structural Tier-C enforcement (§3.3, §5)
4. Retrieval discipline with references-not-rules framing (§3.5)
5. Gated delta-only consolidation with caps (§3.2)
6. PACE-style sequential-test promotion (§3.4)
7. Only then: consider default-on, if and only if §4 shows sustained positive
   paired lift on the transfer holdout
