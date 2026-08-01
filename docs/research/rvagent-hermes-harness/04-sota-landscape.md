# SOTA Landscape (August 2026) — and what it changes

Research sweep across five areas: benchmarks and credible-claim criteria,
harness technique literature with effect sizes, competing harness
architectures, long-horizon context methods, and self-improving harnesses.

This document exists to correct the roadmap, not to decorate it. Where the
evidence contradicts `03-roadmap.md`, the contradiction is stated plainly and
the roadmap change is specified.

**Evidence discipline.** Most 2026 material is single-source arXiv preprints
or vendor blogs. Individual percentage-point figures are hypotheses to
re-measure in our own harness, not constants. The findings below are ranked by
corroboration, and single-source claims are marked. Four findings are
multiply-corroborated and safe to build on:

1. Harness choice moves results 10–30 points at fixed model.
2. Harness value scales *inversely* with model strength.
3. Context **management** beats context **size**.
4. Parallel writer agents fail on coding tasks specifically.

---

## 1. The headline: most of the delta is reliability, not intelligence

The cleanest ablation available (Claw-SWE-Bench, Jun 2026) strips a harness to
bare model-emits-diff and rebuilds it: **19.1% → 73.4%, +54.3 points**. Nearly
all of that is patch-apply failures going from 69.1% to under 1.5%.

That reframes the entire project. The biggest measured wins in every ablation
come from eliminating mechanical failure modes — patches that don't apply,
tool-call loops, context rot, flaky tests — not from smarter reasoning.

**This is the strongest argument for building the harness in Rust**, and it is
not the argument we have been making. The case is not "Rust is fast." It is
that the failure modes which dominate these ablations can be made
*type-unrepresentable* rather than merely rare.

Two structural findings that should shape sequencing:

- **Harness variance is larger for weaker models.** Same harness set, GLM-5.1:
  12.5-point spread. Qwen-3.6-flash: 27.4-point spread. If we target frontier
  models only, we should expect roughly *half* the harness ROI the literature
  reports.
- **The harness is now a disclosed experimental variable.** Two 2026 papers
  (arXiv 2605.23950, 2607.04528) argue benchmark results are substantially
  determined by undisclosed harness choices, and that harnesses induce
  systematically different agent *beliefs* on logically equivalent tasks. Any
  claim we publish must disclose the harness or it is not a claim.

---

## 2. Competitive position: the Rust field is crowded at the top

The premise that a Rust harness is differentiating is **false as of July 2026**.
Three of the major harnesses are already Rust:

| Harness | Scale | License | Notes |
|---|---|---|---|
| **Codex CLI** (OpenAI) | ~70–80 crates | Apache 2.0 | Rewritten *from* TypeScript for single-binary distribution + native sandbox bindings |
| **Grok Build** (xAI) | ~844k LOC | Apache 2.0 | Open-sourced **2026-07-15**. ACP, checkpoints, TUI |
| **Goose** (Block) | — | Apache 2.0 | MCP-native, subagents via `Agent::new()` |

OpenAI's stated reasons for the Rust rewrite are exactly the ones in our
positioning: zero-dependency install (Node 22+ blocked enterprise/air-gapped),
no GC pauses in long-running processes, memory-safe sandbox bindings without
FFI shims. That ground is taken.

**Two positions remain genuinely open:**

1. **No Rust harness is usable as a library.** Codex's own `AGENTS.md`
   discourages extending `codex-core`; Grok Build has issues and PRs
   *disabled* (source-visible, not open governance); Goose is app-first; and
   Anthropic's Agent SDK is Python/TypeScript only — the docs instruct other
   languages to shell out to the CLI. A stable, semver'd, embeddable harness
   crate with open governance is unoccupied.
2. **Deterministic replay is a named gap** in the Rust agent ecosystem and
   thin everywhere. Our append-only witness/segment infrastructure is already
   most of the way there.

**Honesty constraint on the replay claim.** Even at temperature 0, hosted
inference is not reproducible — floating-point non-associativity and
batch-size-dependent kernels produce run-to-run variation. The defensible
claim is *replay of the harness*, with **action-match rate** reported, not
byte-exact reproduction of the model.

**Convergence warning.** Grok Build's tool implementations are documented ports
— `apply_patch`, `grep_files`, `list_dir`, `read_file` from Codex; `bash`,
`edit`, `glob`, `grep`, `read`, `skill`, `todowrite`, `write` from opencode. A
frontier lab with a million lines of Rust chose to port the tool surface rather
than design one. **Tool-surface novelty is not available as a differentiator.**

---

## 3. Ranked technique priorities

Ordered by measured effect per unit of engineering effort. Tier 1 items are
days of work each and carry the largest deltas in the literature.

### Tier 1 — do first

| # | Technique | Effect | Status in rvagent |
|---|---|---|---|
| 1 | Reliable patch application (real workspace, file-based edits, git diff extraction, verify-after-write) | **+54.3 pp** | Partial — tools write real files; no verify-after-write, no git extraction |
| 2 | Observation-window management (keep last N tool outputs in full, elide older) | +3 pp, prevents long-run collapse | **Missing** |
| 3 | Loop/stuck detection (3-strike tool-call fingerprint → inject warning, skip) | Removes most common catastrophic failure | **Missing** |
| 4 | Actionable structured tool errors + response size caps (~25k tokens) | Part of the reliability delta | Partial — errors feed back (P0.4), uncapped |
| 5 | Tool surface held to 8–15 tools | Avoids −16 to −23 pt routing collapse | **OK** — 9 builtins. Protect this. |
| 6 | Environment bootstrap injection (cwd, tree, toolchain, test command, current check status) | Stanford Meta-Harness @ 76.4% TB2.0 | **Missing** |
| 7 | Persist thinking blocks across tool calls | +2.2 pp coding | **Missing** |

Item 3 deserves emphasis: raising max-iteration counts does **not** fix loops,
it makes them more expensive. Our `max_iterations: 100` is a cost cap, not a
loop guard.

### Tier 2 — clear ROI, ~1–2 weeks each

- **`str_replace` edit tool with fuzzy-failure diagnostics + `cargo check` lint
  gate.** +10 to +23 pp for mid-tier models, ~0–5 pp frontier. The specific
  ergonomics matter: in the Qwen reproduction only the SWE-agent
  `str_replace_editor` flavor moved the number; a different `edit`/`write_file`
  pair gave *zero* improvement. Offer `write_file` alongside (+2.1 pp, −17.9%
  cost). Skip unified diff — apply failures dominate.
- **Reproduction-test-first loop** (+8 to +13% relative). Critical caveat:
  adding "write tests first" to the prompt *without* targeted context made
  regressions **worse** (6.08% → 9.94%). The gain is in *executing*
  reproduction tests, not the TDD ritual.
- **Summarized grep/glob** — return paths + match counts, require a second call
  for contents. `ripgrep` as a library crate.
- **Fresh-context reviewer subagent** — ~2 bugs/PR, 58% severe, in Cognition
  production. Counterintuitively, reviewers perform **better with no shared
  context**: shorter context, less rot, deeper analysis.
- **Read-only context-gathering subagent** returning a summary string. +2.1 pp,
  −34.5% main-agent input tokens. Use a *cheap* model here — a frontier model
  in this slot gave +0.4 pp at 5.8× cost.

### Tier 3 — real but expensive or conditional

Rubric-guided compaction (the rubric is load-bearing, not the tool);
best-of-N with test-based filtering then deterministic patch fusion (+7 to
+9.4 pp at N≈8, ≈8× cost); shadow-git checkpointing; coordinator delegation
(only after single-agent is solid).

### Skip list — evidence says no

- **Parallel writer swarms** for coding
- **Few-shot demos / explicit CoT instructions** for reasoning models — zero-shot ≥ few-shot; exemplars can contradict native reasoning
- **Ungrounded self-reflection loops** — can degrade already-correct answers; only execution-grounded critique works
- **Unified-diff edit format**
- **Semantic/embedding code index in v1** — vendor-only evidence, high maintenance, stale-by-construction on a repo the agent is editing
- **Elaborate system-prompt frameworks** — the "20–30% improvement" claims have no methodology; keep rulebooks under ~60 lines
- **Context windows beyond ~128k** — sweeps plateau ~114k; documented ceiling ~96–112k
- **Learned/RL-trained components** (adaptive edit-format selectors, RL compaction) — the hand-written 80% is available for 5% of the effort

---

## 4. Long-horizon execution: what actually breaks

Ranked by when it bites in a multi-hour run:

1. **Wasted-context accumulation → attention dilution.** Earliest, most
   universal, invisible. Unaided coding agents waste ~1-in-3 file reads.
2. **History error accumulation.** The largest single failure driver —
   process-level failures are **72.5%** of long-horizon failures (HORIZON,
   arXiv 2604.11978). Errors compound *between* steps, not within them.
   Non-linear: sharp collapse past a domain-specific threshold.
3. **Compaction-induced information loss** — self-inflicted, caused by the
   mitigation for #1.
4. **Goal/identity drift** — *downstream* of 1–3, not an independent disease.
5. **Hallucinated state** — phantom invoices, fabricated history.
6. **Memory staleness / negative transfer.**

The 2026 evidence does **not** support treating goal drift as the primary
problem. It is the observable end-stage of context and error problems.

### The compaction finding that contradicts our plan

**Simple observation masking matches or beats LLM summarization at roughly
half the cost** (JetBrains, 250-turn SWE-bench trajectories, NeurIPS 2025
workshop). Mechanism: LLM summarization **extended trajectories 13–15%** by
destroying natural stopping signals — the agent loses the cue that it already
finished something.

Better still, **Addressable Recall Compaction** (arXiv 2607.25066): mask the
observation but leave an ID the agent can dereference on demand. Beat
full-context, sliding window, LLM summary, structured state, *and* RAG memory
(NIAH 99.40% vs 88.12% best baseline).

And the single highest-leverage technique found anywhere in this sweep:
**programmatic tool calling** — the model writes code that orchestrates tools,
so intermediate outputs never enter context. On LOCA-bench it was the **only**
strategy positive across all four models tested (+6 to +13.3 points). It is
absent from our roadmap entirely.

Two further hard requirements:

- **Re-inject invariants verbatim after every compaction; never summarize
  them.** "Governance Decay" (arXiv 2606.22528) shows safety constraints and
  system-prompt instructions erode through successive compaction cycles with
  no failure signal. This is architectural, not jailbreaking.
- **Capability-gate context features.** On LOCA-bench a memory tool made a
  weaker model *worse* (10.7% → 8.0%), and context-budget awareness made it
  much worse (10.7% → 4.0%), while both helped stronger models. Multi-model
  support makes per-tier feature flags mandatory.

---

## 5. Self-improvement: the literature turned against us

RuVector already has ReasoningBank-style trajectory learning and a shared
brain. The 2026 evidence on this class of system is substantially negative.

**The case for** is real: ReasoningBank reports +4.6 to +8.3 points on WebArena
across three backbones (ICLR 2026); ACE reports +10.6% on agents. Distilling
from *failures* as well as successes is a genuine contribution.

**The case against is now stronger:**

- **The gains are confound-sized.** MemDelta (arXiv 2606.29914) shows swapping
  the embedding model alone shifts accuracy ±6.2pp — comparable to
  ReasoningBank's entire headline gain. In the same work, **agent self-memory
  (42%) underperformed plain retrieval (47%)**, and one system reached parity
  with cloud RAG at **50× the cost**.
- **Memory utility is an inverted U.** "Useful Memories Become Faulty"
  (arXiv 2605.12978): utility rises, then degrades *below* the no-memory
  baseline. GPT-5.4 failed **54% of previously-solved ARC-AGI problems** when
  using consolidated memory. Episodic-only management **doubled** accuracy vs
  forced consolidation — **the consolidation step is the bug, not the storage.**
- **Benchmark-local.** MemoryArena specifically names ReasoningBank's
  procedural memory as performing poorly on interdependent multi-session tasks
  — the setting closest to real work.
- **No automatic self-evolution method sustains positive gain across settings**
  (EvoAgentBench, arXiv 2607.05202). *Curated* ability content transfers
  across model families; *automatic extraction* is the failure point.
- **Greedy acceptance is uncontrolled multiple testing.** PACE
  (arXiv 2606.08106): "keep it if the score improved" committed **30–42% false
  edits**, and made 13–21 spurious modifications when *no true gains existed*,
  degrading one agent by 4.9 points.

Meanwhile **Live-SWE-agent reaches 79.2% on SWE-bench Verified with zero
persistent memory** — on-the-fly tool synthesis from the current trajectory,
discarded after use. It sidesteps every failure mode above, which is not a
coincidence.

**Verdict:** trajectory-learning memory is a nice-to-have with fragile upside,
not a differentiator. It must ship as a **gated optimization with a measured
contribution**, never on the default path unmeasured.

Minimum viable discipline if we keep it:

- Two-tier storage: **episodic immutable**, distilled artifacts *derived* with
  pointers back to source episodes.
- Gated, delta-only consolidation with hard length caps. Never end-to-end
  rewrites (ACE documents a single step collapsing 18,282 tokens → 122).
- Trust tiers by verifier: programmatic (tests/typecheck) = active;
  LLM-judge-only = quarantined; **derived from untrusted content = never
  allowed to influence tool-permission or destructive-action decisions.**
- **PACE-style anytime-valid sequential testing for promotion**, never greedy.
- Retrieved memories framed as **"references, not rules"** — nearly free, and
  drops attack success 20.6% → 13.1%.
- **A permanent memory-off control arm.** Not a one-time ablation — without a
  live control we cannot detect the inverted-U crossover.

Primary metric: **paired net lift on the same instances**, with the regression
metric that matters most being *rate of previously-solved tasks that now fail
with memory on*.

**Shared-brain caveat.** Cross-agent shared memory multiplies the poisoning
blast radius. Measured memory-poisoning work reports ~50% attack success and
~41% *relapse* success, with contextual assimilation as the primary vector —
poisoned entries work best when they look like ordinary preferences. The
shared brain needs the strongest gates in the system, not the weakest.

---

## 6. Verifier quality is the binding constraint

Any self-improving loop is only as good as its verdict signal, and the 2026
evidence on LLM-as-judge is poor:

- A judge surfaced **under 25%** of human-confirmed systematic problems, and
  **flagged zero** issues in a batch where humans confirmed 23 distinct
  defects. Implied **3–6× undercount** of true defect rates.
- Blind-spot structure is systematic: catches **turn-local** problems, severely
  blind to **cross-turn state**.
- Calibration drifts: one judge at 0.91 agreement shifted four points after a
  model update — the signal stopped meaning what it meant.

**Design rule:** programmatic and execution-grounded signals first (tests, type
checks, schema validation, invariants). LLM judge as a *secondary, quarantined*
signal only. Treat the judge as a **regression floor, never a promotion
authority**. Re-anchor against a rolling human-labeled sample after every model
update.

---

## 7. Benchmark hygiene

**SWE-bench Verified is saturated and unreliable as a claim target.**

- UTBoost (ACL 2025): resolve rates inflated **~6.4 pp** by weak tests; **1 in
  5** "solved" patches semantically incorrect; augmenting tests changed
  leaderboard ranks in **24.4%** of Verified submissions.
- OpenAI's own audit found **59.4%** of the hardest unsolved Verified problems
  had flawed test cases, and OpenAI stopped reporting Verified in early 2026.

This directly threatens our Phase 4 exit gate (`≥70% on a 350-instance
fixed-model set`) — the gate is stated against a saturated benchmark whose
noise band exceeds the effect sizes we would be claiming.

**Consequence for our own A/B testing:** before trusting any measurement of a
harness change, verify test determinism (run F2P/P2P repeatedly under gold and
base patches). A 3-point "improvement" sits inside the flaky-test noise band.
Invest in eval-loop speed — validation that cannot run in minutes will not get
run.

---

## 8. Roadmap corrections

Specific, and each traceable to a finding above.

### Phase 1 — reorder and add

**Add a new Phase 1a "reliability floor" ahead of everything else**, comprising
Tier 1 items 1–7 (§3). These are days of work with the largest measured
deltas, and five of seven are currently missing. The present Phase 1 leads with
event-streaming, cache-first prompts, and compaction — all defensible, none of
them the biggest lever.

**Add programmatic tool calling** (§4). Highest-leverage single technique in
the sweep; absent from the plan.

### Phase 1.3 / Phase 2.5 — change the compaction bet

Both currently bet on summarization (middle-turn summarization; ADR-252
coherence-weighted compaction). Evidence says summarization is close to the
*worst* measured option and inflates trajectories 13–15%.

**Replace with:** observation masking as the default, plus addressable recall
(masked entries keep a dereferenceable ID). Keep summarization as a fallback
behind a rubric. Add mandatory verbatim invariant re-injection post-compaction.

Note this also affects the shipped default pipeline, which currently includes a
`summarization` middleware.

### Phase 1.6 — subagents: drop the CRDT merge

Currently "CoW fork/merge, CRDT join." This is parallel-writer architecture,
which is the one multi-agent pattern with strong negative evidence for coding.
Cognition's 2026 revision — after a year of production data — is **one writer,
augmented by auxiliary intelligence; never parallel writes.**

**Replace with** the two patterns that have production evidence: a
**fresh-context reviewer** (no shared context — it performs *better* without
it) and a **read-only context-gatherer** returning a summary string. Model the
subagent boundary as *a tool that spawns an isolated context and returns a
String*, not as peers with a message bus. That buys nearly all demonstrated
upside at a fraction of the complexity.

### Phase 2 — gate the learning loop

"SONA on the default path" contradicts §5. Move behind a feature gate with the
measurement apparatus (paired lift, previously-solved regression rate, control
arm) as the *precondition* for enabling it, not a follow-up.

The existing exit gate (≥30% token reduction on a repeated task suite) is
well-formed — keep it, and add the control arm.

### Phase 3 — MCP spec migration is now urgent

MCP **2026-07-28** landed days ago and is breaking: protocol-level sessions and
`Mcp-Session-Id` removed, `_meta` on every request, `server/discover` added,
Tasks extension replaces long-running `tools/call`, and Roots/Sampling/Logging
deprecated. Building against the 2025-11-25 shape means a rewrite within
months. The official Rust SDK (`rmcp`) already implements the new spec.

**Also add ACP as a first-class target.** It went from Zed-only to headline
feature of Zed 1.0, built into JetBrains, a public registry, and 25+ agents —
and its reference implementation is Rust. We have an `rvagent-acp` crate
already; this is closer than it looks.

### Phase 4 — re-ground the SOTA claim

Drop the SWE-bench-Verified-based gate. Retarget to non-saturated benchmarks
and to the axes where we can actually win. The differentiator claims should be
**library API + deterministic replay** (§2), not speed.

Report action-match rate for replay, not reproducibility. Keep the honesty
apparatus — given §7, retraction discipline is the feature.

---

## 9. Open items

- Benchmark/leaderboard sweep (current top scores by model+harness,
  cost-normalized Pareto, credible-claim conformance criteria) is still
  outstanding; §7 covers hygiene but not target selection.
- `codex-rs` internals in §2 come from a third-party architecture writeup, not
  the repo. Worth reading `codex-rs/core/src/` directly before copying the
  `Op`/`EventMsg` design.
- Grok Build's governance may change. If xAI opens PRs, the "no community Rust
  harness" gap closes fast — the moat must be the library API and replay, not
  merely that a Rust harness exists.
