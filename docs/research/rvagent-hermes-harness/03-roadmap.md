# Implementation Roadmap

Ordering rule: nothing in P1+ lands until P0 is green, because every
feature in P1+ is blocked by the type unification and the tools-to-model
fix. Each phase has a falsifiable exit gate.

---

## Phase 0 — Foundation Repair (rvagent-core/backends/middleware)

The loop must become a real agent before it can become a great harness.

1. **Unify types.** `rvagent-core`'s `AgentState`/`Message`/`Tool`/
   `TodoItem`/`RunnableConfig` become canonical; delete the middleware and
   subagents duplicates; add conversion impls only at the WASM/serde
   boundary. (Blocks everything.)
2. **Send tools to the model.** `ChatModel::complete(&[Message],
   &[ToolDefinition])`; `tools` field on Anthropic `ApiRequest`;
   `functionDeclarations` for Gemini; parse `tool_use` into the canonical
   `ToolCall`.
3. **Async middleware + wiring.** `wrap_model_call` → async;
   `build_default_pipeline()` constructed and installed in CLI, ACP, and
   the future A2A runner. Delete the CLI's duplicate `LocalFsBackend`
   grep/glob/execute in favor of `rvagent-backends`.
4. **Loop correctness.** Tool errors feed back as tool results (no `?`
   abort); real parallel execution via `parallel_execute`; per-turn `Usage`
   flows into `BudgetEnforcer` and `Metrics`.
5. **Honest docs.** Remove unsupported claims (parallel exec, HNSW speedups,
   streaming) from README/architecture.md until true; docs list all 10
   crates.

**Exit gate:** `rvagent run "create and run a failing test, then fix it"`
completes end-to-end against the live Anthropic API with ≥2 tool round
trips, budget accounting, and a green `cargo test` across the workspace.

## Phase 1 — Hermes-Class Loop Mechanics

1. **Event-streaming loop** (`AgentEvent` stream; ADR-139 direction) with
   SSE streaming backends and incremental TUI render.
2. **Cache-first prompt builder** (stable/context/volatile tiers,
   `cache_control` emission, cache-hit-rate metric).
3. **Compaction with lineage** (sentinel pre-limit, middle-turn
   summarization, durable-fact extraction under hard budget, parent/child
   session chains, real tokenizer).
4. **Full-state checkpoints** (`SegmentType::Checkpoint`), resume/fork.
5. **Hooks + permissions** (10-event taxonomy with Allow/Deny/Ask/Defer
   merge mirrored from metaharness kernel; HITL wired to
   `PermissionRequest` events; permission modes; per-tool rules).
6. **Real subagents** (JoinSet spawn, semaphore, CoW fork/merge, CRDT join,
   per-subagent model override).
7. **Headless contract** (`-p --output-format stream-json
   --max-budget-usd --policy-file`).
8. **Docker sandbox backend** (read-only root, dropped caps) alongside the
   local shell.

**Exit gate:** rvagent completes a 20-instance SWE-bench-Lite smoke slice
via the headless contract inside metaharness's runner, with measured cache
hit rate >85% and zero loop aborts on tool errors.

## Phase 2 — Memory & The Learning Loop

1. **Kill fake HNSW**; episodic memory = session store + FTS, with real
   `ruvector` embeddings behind the feature gate; optional shared
   `.swarm/memory.db` (AgentDB bindings, ADR-323 provenance, WAL guard).
2. **Prompt-memory write-back** with hard char budget + periodic curation
   nudges.
3. **Skill synthesis**: trajectory triggers (5+ tool calls / error recovery
   / user correction) → SKILL.md emission via auxiliary model; progressive
   disclosure; `skill_manage` tool with string-patch default. (Closes
   ruflo's #1 roadmap gap; skills interop with Claude Code/Codex via the
   existing `skills_bridge`.)
4. **SONA on the default path** (feature-gated on, trajectories from the
   witness chain feeding ReasoningBank; ADR-271 EwcConfig as genome).
5. **ADR-252 coherence-weighted compaction** as the summarization upgrade.

**Exit gate:** on a 3×-repeated task suite, the skilled agent shows ≥30%
token reduction vs a fresh instance (Hermes's measured ~40% is the bar).

## Phase 3 — Ecosystem Integration

1. **ruflo:** NAPI package (`@ruvector/rvagent-native`) as a sibling to
   `rvagent-wasm` behind `agent-wasm.ts`; `wasm_agent_compose` tool
   allowlist honored; Capability Brain registration; hook-handler
   subprocess shim; bandit outcome reporting. ruflo's worker daemon gains
   an `rvagent` executor option beside headless Claude.
2. **metaharness:** `packages/host-rvagent` adapter; rvagent registered as
   a Darwin solver backend via the headless contract; witness format
   alignment (kernel ↔ rvagent ↔ RVF per ADR-106 Phase 4).
3. **A2A goes live:** real `TaskRunner` backed by the loop replaces
   `InMemoryRunner`; `rvagent a2a serve` advertises real skills; federated
   rvagent↔rvagent task delegation demo with budgets + recursion guards.
4. **Eval in CI:** TBLite-style smoke slice + frozen-eval gate on PRs
   (metaharness harness-loop gates: QUALIFY→BENCHMARK→VERIFY→CANARY).

**Exit gate:** one command (`npx ruflo swarm ... --executor rvagent` or
equivalent) runs a swarm where rvagent instances execute, ruflo
coordinates/remembers, and the run emits signed witness manifests.

## Phase 4 — Evolution & SOTA Claims

1. **Policy-genome evolution**: Darwin mutates policy text, temperature
   schedules, compaction thresholds, tool configs; flywheel promotion with
   held-out benchmarks and signed champion manifests.
2. **Skill-population evolution** (GEPA-style over execution traces).
3. **weight-eft distillation** of gold trajectories into cheap-tier LoRA
   (ADR-271's SFT/DPO on/off-policy recipe) → escalate to frontier models
   less often; cost-Pareto tracking.
4. **Publish**: Claw-SWE-Bench-style fixed-model comparison vs Hermes/
   OpenClaw, Terminal-Bench 2.0, with conformant packaged submissions and
   Wilson CIs. No claim ships without the metaharness honesty apparatus
   (the ecosystem's retraction discipline is a feature — keep it).

**Exit gate:** rvagent within the Hermes/OpenClaw band (≥70%) on the
350-instance fixed-model set, and demonstrably ahead on ≥2 of: startup
latency, token throughput, parallel-tool wall clock, federated multi-agent
tasks, in-browser deployment.

---

## Proposed ADRs

| ADR | Title | Decides |
|---|---|---|
| A | rvAgent Harness Core Repair (supersedes parts of ADR-095/097) | canonical types, tools-to-model, async middleware, loop recovery semantics |
| B | rvAgent Event-Streaming Loop & Headless Contract (implements ADR-139) | `AgentEvent` taxonomy, stream-json format, budget/policy flags |
| C | rvAgent Cache-First Prompts, Compaction & Lineage | prompt tiers, cache_control, sentinel compaction, session lineage |
| D | rvAgent Skills & Learning Loop | synthesis triggers, SKILL.md interop, SONA/witness wiring, char budgets |
| E | rvAgent ⇄ ruflo Execution Seam | NAPI sibling, memory co-tenancy rules (ADR-323/WAL), hook shim, Capability Brain entry |
| F | rvAgent ⇄ metaharness Evolution Seam | host adapter, Darwin solver contract, policy genome surfaces, witness alignment |

## Risks

- **Scope gravity.** The ecosystem's pattern (documented in all three
  audits) is protocol/ADR surface outrunning the loop. Mitigation: P0/P1
  exit gates are executable, not documentary; no new crate until the gate
  passes.
- **Type unification churn** touches every crate at once. Mitigation: land
  as one PR series with the workspace green at each step; WASM API kept
  stable via serde boundary.
- **Benchmark credibility.** Any SOTA claim without the gold-eval +
  Wilson-CI + packaged-submission discipline damages the whole ecosystem's
  (currently strong) honesty record. Mitigation: Phase 4 gates are
  metaharness-conformant by construction.
- **Optionality invariants** (ADR-150/256) must hold in both directions or
  the three projects become a distributed monolith. Mitigation: CI runs
  each project's `--ignore-optional` path with rvagent absent/present.
