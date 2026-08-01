---
adr: 277
title: "rvAgent Positioning, Protocols and Benchmark Claims"
status: accepted
date: 2026-08-01
authors: [Reuven Cohen]
project: "rvAgent Harness"
related: [ADR-159, ADR-267, ADR-273, ADR-274, ADR-275, ADR-276]
tags: [rvagent, harness, positioning, mcp, acp, replay, benchmarks, sota, honesty]
---

# ADR-277 — rvAgent Positioning, Protocols and Benchmark Claims

## Status

**Accepted.** Owner: Reuven Cohen. Project: rvAgent Harness. Date: 2026-08-01.

Evidence base: `docs/research/rvagent-hermes-harness/04-sota-landscape.md` §2, §7.

**This ADR invalidates a premise.** The roadmap's implicit positioning — that a
Rust-native harness is itself differentiating — was true when the research
began and is false as of July 2026.

## 1. Decision

1. **Do not position on "Rust is fast."** That ground is taken.
2. **Position on two open gaps:** a stable embeddable **library API with open
   governance**, and **deterministic replay** as a core primitive.
3. **Migrate to MCP 2026-07-28** before building further on the tool protocol.
4. **Treat ACP as a first-class target.**
5. **Retire SWE-bench Verified as a claim target.**
6. **No claim ships without harness disclosure.**

## 2. The field is crowded at the top

Three of the major 2026 harnesses are already Rust:

| Harness | Scale | License | Notes |
|---|---|---|---|
| **Codex CLI** (OpenAI) | ~70–80 crates | Apache 2.0 | Rewritten *from* TypeScript |
| **Grok Build** (xAI) | ~844k LOC | Apache 2.0 | Open-sourced **2026-07-15** |
| **Goose** (Block) | — | Apache 2.0 | MCP-native |

OpenAI's stated reasons for the rewrite are exactly our positioning:
zero-dependency install (Node 22+ blocked enterprise and air-gapped
deployments), no GC pauses in long-running agentic processes, memory-safe
sandbox bindings without FFI shims.

**Tool-surface novelty is also unavailable.** Grok Build's tools are documented
ports — `apply_patch`, `grep_files`, `list_dir`, `read_file` from Codex;
`bash`, `edit`, `glob`, `grep`, `read`, `skill`, `todowrite`, `write` from
opencode. A frontier lab with a million lines of Rust ported the tool surface
rather than designing one. Convergence is complete.

## 3. What remains open

### 3.1 No Rust harness is usable as a library

- Codex's own `AGENTS.md` **discourages** adding to `codex-core` — it is an
  app, not a published SDK.
- **Grok Build has issues and PRs disabled.** Contributions explicitly
  rejected; xAI develops internally and syncs a mirror. This is *source
  transparency, not open governance.*
- Goose is app-first.
- **Anthropic's Agent SDK is Python and TypeScript only** — the docs instruct
  other languages to shell out to the CLI with `-p --output-format json`.

A stable, semver'd, embeddable harness crate with open governance is
unoccupied. This is the primary position.

### 3.2 Deterministic replay

A named gap in the Rust agent ecosystem and thin everywhere. Our existing
append-only witness/segment infrastructure is most of the way there, and
ADR-274's deterministic masking (no model call on the compaction path) and
ADR-275's `Fn(prompt) -> String` subagent boundary both make it tractable.

**Honesty constraint — this is binding.** Even at temperature 0, hosted
inference is not reproducible: floating-point non-associativity and
batch-size-dependent kernels produce run-to-run variation, with reported
accuracy swings up to 15% across runs.

Therefore the claim is **replay of the harness, not of the model**, and the
reported metric is **action-match rate**, never "reproducible" without
qualification. Record every LLM call, tool response, and timestamp; replay to
reproduce harness behavior; promote incidents to test fixtures.

Claiming byte-exact reproducibility would be false and would be caught.

### 3.3 Why Rust, restated honestly

Not speed. The defensible argument is ADR-273's: the failure modes that
dominate every harness ablation — patch-apply failures, tool-call loops,
desynced `tool_use`/`tool_result` pairing, unbounded tool output — can be made
**type-unrepresentable** rather than merely rare. Plus structured concurrency
(`JoinSet` + `CancellationToken`) making mid-run interrupt and subagent
lifecycle nearly free, where they are hard in Python.

**Where Rust is a liability — state these plainly:**

- **Provider coverage.** LiteLLM's 100+ providers is a moat. Rust's best is
  ~20. Permanent maintenance tax.
- **Iteration speed on what matters most.** Prompts, tool descriptions, and
  compaction rubrics are where harness performance lives, and they want a
  REPL. Mitigation: keep prompts and templates in **hot-reloadable external
  files**, never `const &str`.
- **The eval ecosystem is Python.** We will shell out for evaluation.
- **Compile times** on a large workspace are a daily cost.
- **Extension authors don't write Rust.** Mitigation: **the extension language
  is MCP, not Rust** — Goose's key insight. Pi has 2,143 third-party extensions
  because they are TypeScript.

## 4. Protocols

### 4.1 MCP 2026-07-28 — migrate now

Landed 2026-07-28 and is breaking:

- **Stateless core.** Protocol-level sessions and `Mcp-Session-Id` **removed**.
- Protocol version, client info, and capabilities now travel in `_meta` on
  **every** request.
- New `server/discover` method.
- **Tasks extension:** `tools/call` returns a task handle driven via
  `tasks/get` / `tasks/update` / `tasks/cancel`. `tasks/list` removed.
- **Deprecated:** Roots, Sampling, Logging; HTTP+SSE reclassified deprecated.
- Auth aligns with real OAuth 2.0/OIDC; clients must validate `iss` per
  RFC 9207 (mix-up attack mitigation).

Building against the 2025-11-25 shape means a rewrite within months. The
official Rust SDK (`rmcp`) already implements the new spec while remaining
compatible with older ones — start there rather than hand-rolling.

MCP servers remain gated behind per-session enablement (ADR-273 §3.4): the
tool-count ceiling is not negotiable for protocol convenience.

### 4.2 ACP — first-class

ACP went from Zed-only (Jun 2025) to headline feature of Zed 1.0 (2026-04-29),
built into JetBrains since Dec 2025, a public registry (2026-01-28), and 25+
agents by March 2026. **Its reference implementation is Rust.**

We already have an `rvagent-acp` crate. Being ACP-native rather than
ACP-bolted-on is closer than it looks and is the natural distribution channel
for a library-shaped harness.

Division of labor as settled in 2026: **MCP = tools, A2A = agent discovery,
ACP = editor↔agent, AG-UI = agent↔UI.** ADR-159's A2A work sits in the third
slot and remains valid.

## 5. Benchmark claims

### 5.1 SWE-bench Verified is retired as a claim target

- **UTBoost (ACL 2025):** resolve rates inflated **~6.4 pp** by weak tests;
  **1 in 5** "solved" patches semantically incorrect; augmenting tests changed
  leaderboard ranks in **24.4%** of Verified submissions (40.9% on Lite).
- **OpenAI's own audit:** **59.4%** of the hardest unsolved Verified problems
  had flawed test cases. OpenAI stopped reporting Verified in early 2026.

`03-roadmap.md` Phase 4's exit gate (`≥70% on a 350-instance fixed-model set`)
is stated against a benchmark whose **noise band exceeds the effect sizes we
would be claiming.** It is withdrawn pending replacement.

Verified may still be used as an internal regression signal — with test
determinism verified per ADR-273 §6 — but not as a published claim.

### 5.2 Claim conformance

Two 2026 papers establish that harness choices substantially determine
benchmark results (arXiv 2605.23950) and that harnesses induce systematically
different agent *beliefs* on logically equivalent tasks (arXiv 2607.04528).

Therefore every published claim must disclose: environment setup, tool
implementations, the full harness configuration, and the evaluation procedure.
**A result without harness disclosure is not a result.**

Retain the existing honesty apparatus (ADR-267): fixed-model comparison,
conformant packaged submissions, Wilson confidence intervals, and retraction
discipline. Given §5.1, retraction discipline is a feature.

### 5.3 Differentiator claims

Claim the axes from §3, not throughput:

- Stable embeddable library API with open governance (§3.1)
- Harness replay with **action-match rate** reported (§3.2)
- Startup latency, memory footprint, single-binary distribution — real, but
  **already claimed by Codex and Grok Build.** Supporting evidence, not the
  headline.

## 6. Consequences

**Positive.** Positioning now rests on gaps that are actually open and on
claims that survive scrutiny. The library-API framing also improves the
internal architecture — it forces a clean core/app split that ADR-275's
subagent-as-tool boundary and §3.2's replay both need.

**Negative.** The "first fast Rust harness" story is gone. Phase 4's headline
gate is withdrawn without a replacement in hand (§7).

**Risk — Grok Build's governance may change.** If xAI opens PRs, the "no
community Rust harness" gap closes quickly. The moat must be the library API
and replay, **not merely that a Rust harness exists.**

## 7. Open

Benchmark target selection is unresolved. Required before Phase 4 can restate
its gate: current top scores by model *and* harness, cost-normalized Pareto
position, and conformance criteria for a credible 2026 claim on a
non-saturated benchmark. Tracked in
`04-sota-landscape.md` §9.

Also outstanding: read `codex-rs/core/src/` directly before adopting its
`Op`/`EventMsg` submit/event design — the survey's account of it is
third-party.
