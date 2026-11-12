---
adr: 273
title: "rvAgent Harness Reliability Floor"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-103, ADR-139, ADR-274, ADR-275, ADR-276, ADR-277]
tags: [rvagent, harness, reliability, agent-loop, tools, error-recovery, sota]
---

# ADR-273 — rvAgent Harness Reliability Floor

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Evidence base: `docs/research/rvagent-hermes-harness/04-sota-landscape.md`.

## 1. Decision

**Sequence harness work by measured reliability impact, not by architectural
ambition.** A defined "reliability floor" of seven mechanisms ships before
event-streaming, cache-tiering, evolution, or any other Phase 1+ item.

This ADR reorders `03-roadmap.md` Phase 1. It does not remove any item from it.

## 2. Context

The cleanest harness ablation available (Claw-SWE-Bench, Jun 2026) rebuilds a
harness from bare model-emits-diff to full scaffolding: **19.1% → 73.4%,
+54.3 points**. Nearly all of that delta is patch-apply failures falling from
69.1% to under 1.5%.

That result generalizes across the technique literature: **the dominant wins
come from eliminating mechanical failure modes, not from better reasoning.**
Patches that don't apply, tool-call loops, context rot, flaky tests.

Two structural qualifiers:

- **Harness value scales inversely with model strength.** Same harness set:
  12.5-point spread on GLM-5.1, 27.4-point spread on Qwen-3.6-flash. Targeting
  frontier models only should be expected to yield roughly half the ROI the
  literature reports.
- **Effect sizes are mostly single-source preprints.** The figures below are
  hypotheses to re-measure in our own harness, not constants. What is
  multiply-corroborated is the *direction* and the *ranking*.

This also settles the project's positioning argument. The case for Rust is not
speed — it is that the failure modes dominating these ablations can be made
**type-unrepresentable** rather than merely rare. See ADR-277.

## 3. The floor

Seven mechanisms, ordered by measured effect per unit of effort.

| # | Mechanism | Reported effect | Status |
|---|---|---|---|
| 1 | Reliable patch application | +54.3 pp | Partial |
| 2 | Observation-window management | +3 pp, prevents long-run collapse | Planned (ADR-274) |
| 3 | Loop / stuck detection | Removes most common catastrophic failure | **Done** |
| 4 | Actionable tool errors + response size caps | Part of the reliability delta | Partial |
| 5 | Tool surface held to 8–15 tools | Avoids −16 to −23 pt routing collapse | Holding at 9 |
| 6 | Environment bootstrap injection | Meta-Harness @ 76.4% TB2.0 | Planned |
| 7 | Persisted thinking across tool calls | +2.2 pp coding | Planned |

### 3.1 Reliable patch application

Real workspace, file-based edits, git-based diff extraction, and
**verify-after-write** — re-read the file and confirm the edit landed before
reporting success. Offer `str_replace` with fuzzy-failure diagnostics and
`write_file` side by side; **do not** offer unified diff, where line numbers,
hunk headers, and trailing newlines dominate apply failures.

Edit-tool ergonomics are load-bearing, not incidental: in the Qwen-3.6
reproduction only the SWE-agent `str_replace_editor` flavor moved the number,
while a different `edit`/`write_file` pair gave *zero* improvement.

Post-edit gate: run `cargo check` (not a full build) inline. This is the
analogue of SWE-agent's linter guardrail, which its ablations found essential
for recovering from bad edits.

### 3.2 Loop / stuck detection — **implemented**

Fingerprint each tool call by `(name, args)`; refuse it once it repeats
consecutively past a threshold (default 3), substituting an actionable message.

**Counting is consecutive, not windowed.** An agent re-running the same check
between edits is doing legitimate work; a windowed counter refuses it. Only an
unbroken run of identical calls trips the detector. Alternating cycles are not
caught — `max_iterations` remains the backstop, and the limitation is
documented on the type.

Raising `max_iterations` does **not** fix loops; it makes them more expensive.
`max_iterations` is a cost cap, not a loop guard.

Refused calls still emit exactly one tool result each, in the model's original
call order, so provider `tool_use`/`tool_result` pairing stays in sync.

### 3.3 Actionable errors and output caps

Tool errors must state what failed *and what to do differently*. An opaque
error code causes the model to retry the identical call, which is the input
condition for §3.2.

Cap tool responses (~25k tokens, matching Claude Code's default) with explicit
truncation markers. An uncapped tool result can consume the context window in
one call.

### 3.4 Tool surface budget

**Hold the builtin surface at 8–15 tools.** Routing accuracy degrades 16–23
points across large catalogs; vendors document degradation past 30–50 tools.
We currently ship 9 builtins — this is a constraint to *defend*, not a target
to grow toward.

MCP servers must therefore be gated behind explicit per-session enablement.
Exposing an MCP firehose directly into the tool list forfeits this.

### 3.5 Environment bootstrap injection

Before the loop starts, snapshot the workspace and inject it into the initial
prompt: cwd, file listing, toolchain versions, `cargo metadata` summary,
workspace members, the test command, and whether `cargo check` currently
passes. This eliminates early exploration turns. ~100 lines of code.

### 3.6 Persisted thinking

Do not strip prior-turn thinking blocks from history. Pure protocol plumbing;
+2.2 pp on coding (the smallest of the reported deltas — coding benefits least
because tool results are self-explanatory — but free).

## 4. Explicitly not in the floor

Rejected for v1 on evidence, not on effort:

- **Few-shot demonstrations and explicit CoT instructions** for reasoning
  models — zero-shot ≥ few-shot; exemplars can contradict native reasoning.
- **Ungrounded self-reflection loops** — can degrade already-correct answers.
  Only execution-grounded critique ("tests failed, here is the output") works.
- **Semantic/embedding code index** — vendor-only evidence, high maintenance,
  and stale by construction on a repo the agent is actively editing. Layer
  ripgrep → structural search → semantic, and only if a conceptual query
  demands it.
- **Elaborate system-prompt frameworks** — the widely-quoted "20–30%
  improvement" claims have no published methodology. Keep rulebooks under ~60
  lines.
- **Context windows beyond ~128k** — sweeps plateau around 114k; documented
  ceilings sit at 96–112k. Buying more window buys nothing.
- **Learned/RL-trained components** — the hand-written 80% is available for 5%
  of the effort.

## 5. Consequences

**Positive.** The largest measured deltas land first. Five of seven mechanisms
are days of work. The floor is testable end-to-end without a live provider,
which is how the P0 exit gate is already structured.

**Negative.** Phase 1's architectural items (event-streaming loop, cache-first
prompt tiers) are deferred behind less glamorous work. This is deliberate: the
evidence does not support them being the biggest lever.

**Risk.** Effect sizes are largely single-source. Mitigation: §6.

## 6. Measurement obligation

Before trusting any A/B of a harness change, **verify test determinism** — run
F2P/P2P repeatedly under gold and base patches. Weak tests inflate resolve
rates by ~6.4 pp, and 1 in 5 "solved" patches on a saturated benchmark are
semantically incorrect. A 3-point improvement sits inside the flaky-test noise
band and means nothing.

Invest in eval-loop speed. Validation that cannot run in minutes will not get
run, and every mechanism in §3 needs it.

## 7. Implementation status

- §3.2 loop detection — **shipped** (`rvagent-core/src/graph.rs`, commit `e709e1a`)
- §3.1 partial — tools write real files; verify-after-write and git diff
  extraction outstanding
- §3.4 — holding at 9 builtins
- §3.3, §3.5, §3.6 — outstanding
