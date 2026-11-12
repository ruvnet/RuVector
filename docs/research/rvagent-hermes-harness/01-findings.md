# Findings: Hermes, rvAgent, MetaHarness, RuFlo

Research conducted 2026-08-01 across four parallel investigations:
web research on the Hermes harness, and deep code audits of
`crates/rvAgent/` (this repo), `ruvnet/metaharness`, and `ruvnet/ruflo`.

---

## 1. The Hermes Harness (NousResearch/hermes-agent)

MIT-licensed Python harness from Nous Research, released Feb 2026; ~175K
GitHub stars in four months, the most-used agent on OpenRouter by mid-2026.
Fully model-agnostic (18+ providers, 3 API modes, mid-session failover).
Sources: [repo](https://github.com/NousResearch/hermes-agent) ·
[architecture docs](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture) ·
[self-evolution companion](https://github.com/NousResearch/hermes-agent-self-evolution).

### Architecture

- **Loop:** one synchronous `AIAgent` class (`run_agent.py`) serves CLI,
  messaging gateway, ACP, batch, and API-server modes. Task ID → prompt
  build → preflight compression check → provider resolution → API call →
  tool dispatch loop → SQLite persistence.
- **Cache-first prompts:** ordered tiers `stable` (identity, tool guidance,
  skill summaries) → `context` (user context files) → `volatile`
  (memory/profile/timestamp). Invariant: *the system prompt never changes
  mid-conversation*. Only model switch or memory/context file change breaks
  the prefix cache → **91–97% measured cache hit rates**.
- **Tools:** 70+ tools/28 toolsets, self-registering; MCP for extension.
  Benchmark configs run well with **only `terminal` + `file` enabled**.
- **Sandboxing:** 7 terminal backends (local, Docker, SSH, Singularity,
  Modal, Daytona, Vercel) behind one interface; Docker defaults to
  read-only root FS + dropped capabilities. Zero telemetry.
- **Context management:** a sentinel triggers compaction *before* hard
  limits; an auxiliary model extracts durable facts into memory (hard
  3,575-char budget) and **summarizes middle turns instead of dropping
  them**; compressed sessions keep parent/child **lineage** in SQLite.
- **Memory, four separated layers:**
  1. *Prompt*: MEMORY.md / USER.md, always injected, hard char budget
  2. *Episodic*: SQLite + FTS5 over all past sessions, retrieved on demand
  3. *Procedural*: **skills** — markdown in `~/.hermes/skills/`
     (agentskills.io standard), progressive disclosure
  4. *User model*: optional passively-built profile
- **The learning loop (signature feature):** skill creation triggers on 5+
  tool calls, error recovery, or user correction → trajectory distilled
  into a named skill; skills self-patch during use (string-patch default);
  offline, GEPA (Genetic-Pareto, ICLR 2026 Oral, ~35× fewer rollouts than
  GRPO) rewrites underperforming skills from execution traces.
- **Subagents:** `delegate_tool.py` spawns isolated subagents, multi-model
  routing per subtask, programmatic tool-calling via `execute_code`.

### Benchmark evidence

- **Claw-SWE-Bench** (arXiv 2606.12344): harness choice alone swings pass@1
  by **12.5 pts** (strong model) to **27.4 pts** (weak model). Hermes 71.1%
  with GLM 5.1.
- **Harness-Bench** (arXiv 2605.27922): Hermes 71.2% overall, 100% security;
  paper's conclusion — "execution alignment" (model beliefs ↔ workspace
  state ↔ tool feedback ↔ verification) is the dominant success factor.
- **Learning-loop payoff:** agents with 20+ self-created skills complete
  similar tasks **~40% faster / 40% fewer tokens** (Nous internal,
  independently corroborated; domain-specific).
- **Agentic Harness Engineering** (arXiv 2604.25850): *structure transfers,
  prose doesn't* — tools, middleware, memory architecture generalize across
  models; prompt wording tweaks don't. An observability-driven self-evolving
  harness beat human-designed Codex-CLI on Terminal-Bench 2 (77.0% vs 71.9%).

Hermes's reputation rests on **amortized performance via the learning loop
and ecosystem dominance**, not one-shot leaderboard wins — it is top-tier
but not #1 on frozen single-run benchmarks.

---

## 2. rvAgent Current State (crates/rvAgent/, ~45K LoC, 10 crates)

Declared as a 100%-fidelity Rust port of LangChain DeepAgents (ADR-093..103),
extended with MCP (ADR-104/105/112), A2A (ADR-159), RVF (ADR-106), WASM.

### Production-grade parts

- **`rvagent-a2a`** — the best crate: Ed25519-signed AgentCards, global
  rolling budgets, per-task policy, peer routing with EWMA + circuit
  breaker, recursion guard, W3C trace context, typed/versioned artifacts,
  SSE + signed webhooks. 24 integration test files. *But no real runner —
  `InMemoryRunner` echoes; not connected to the agent loop.*
- **`rvagent-mcp`** — complete JSON-RPC 2.0 MCP server/client, stdio/SSE,
  tool groups, skills bridge (Claude Code + Codex formats). Implemented
  per ADR-112.
- **Security primitives** — path confinement (`virtual_mode`), env
  sanitization, Unicode/BiDi/homoglyph detection, tool-output sanitizer,
  AES-256-GCM session encryption, SHA3-256 witness chains.

### Blocking defects (the loop is scaffold-grade)

| # | Defect | Evidence |
|---|---|---|
| D1 | **Tool schemas are never sent to the model.** `ChatModel::complete(&[Message])` has no tools param; Anthropic/Gemini request structs have no `tools`/`functionDeclarations` field. The loop's `Tools` node is unreachable in production — rvagent is currently a chat client, not an agent. | `rvagent-core/src/models.rs:140`, `rvagent-backends/src/anthropic.rs:69-80`, `gemini.rs:32-35` |
| D2 | **The 19-module middleware pipeline is never wired into the loop.** `build_default_pipeline` is called only from benches/tests; the CLI constructs `AgentGraph::new(model, tools)` directly. Memory, skills, summarization, prompt caching, witness, HITL, SONA, HNSW all dormant. | `rvagent-cli/src/app.rs:643-728` |
| D3 | **Three incompatible type systems.** `AgentState`, `Message`, `Tool`, `TodoItem`, `RunnableConfig` each defined 2–3× (core enum `Message` vs middleware struct vs subagents `HashMap<String, Value>`), no conversion layer. This is the structural blocker for D2. | `core/src/state.rs:81`, `middleware/src/lib.rs:130`, `subagents/src/lib.rs:39` |
| D4 | **Subagents, parallelism, streaming are stubs.** `spawn_sync` returns a formatted string; "parallel" tool execution awaits sequentially in a loop (README's "true concurrency" claim is false); `stream()` returns "not yet implemented"; middleware hooks are sync (can't make an HTTP call without blocking). | `subagents/src/orchestrator.rs:44-110`, `core/src/graph.rs:183-193`, `anthropic.rs:378-385` |

Additional gaps: tool errors abort the loop via `?` instead of feeding back
as tool results (the single most important recovery behavior); session
persistence is messages-only (no todos/files/middleware state); no hooks
infrastructure at all; HITL has no approval transport; the "HNSW" middleware
uses **hash-based pseudo-embeddings with no semantic properties** and no
connection to real RuVector crates; prompt caching types exist but nothing
is emitted to the API; docs claim capabilities the code doesn't have.

ADR-139 (decompiled Claude Code intelligence: async-generator loop yielding
13 event types, 6 permission modes, per-subagent model override) and ADR-107
(`rvagent-swarm`) are proposed but unimplemented.

---

## 3. MetaHarness (ruvnet/metaharness)

A **harness factory + evolution lab**, not a harness runtime. Node/TS
(~50K LoC) + a small Rust kernel (2,259 LoC → WASM/NAPI). 19 published
`@metaharness/*` packages, 223 ADRs, exemplary CI/release engineering.

- **Generator:** `npx metaharness` mints branded harnesses for **nine hosts**
  — Claude Code, Codex, pi.dev, **Hermes**, OpenClaw, RVM, Copilot,
  OpenCode, GitHub Actions — via a `HostAdapter.generateConfig(spec) →
  {path: contents}` interface. The Hermes adapter is verified against the
  real `cli-config.yaml` and mirrors ruflo's `scrubReasoningBlocks`.
  **Generated harnesses contain no agent loop** (`init` + `doctor` only) —
  the loop is provided by the host.
- **Rust kernel (`crates/kernel`):** MCP spec validation + `ToolRegistry`,
  claim-checked `dispatch()`, **10-event hook taxonomy with
  Allow/Deny/Ask/Defer decision-merge** (modeled on Claude Code), 3-tier
  routing heuristic, Ed25519 witness, cost, federation. `#![forbid(unsafe_code)]`,
  serde-typed, no async/no I/O — a library rvagent could consume or mirror.
  (Memory module is a 34-line stub; real memory delegates to `@ruvector/*`.)
- **Darwin Mode (the credible asset):** a real DI-tested ReAct loop
  (`bench/swebench/agentic-loop.mjs`: text-JSON and native function-calling
  variants, anti-thrash state hashing, observation caps, escalation
  cascades) plus the full measurement apparatus — official SWE-bench Docker
  gold eval (**Verified 55.6%**, Lite 51.3%, conformant packaged
  submissions), Terminal-Bench, LiveCodeBench, GAIA/FRAMES, DRACO, with
  Wilson CIs and documented retractions of its own failed claims.
- **The flywheel policy seam:** `SWE_POLICY_SYSTEM` env appends an evolved
  operating policy to the solver's system prompt without touching loop code
  — the template for how Darwin should drive rvagent.
- **Key structural fact:** the marketed control plane (`@metaharness/harness`
  `HarnessKernel`, ADR-047, still Proposed) and the loop that produced every
  measured number (`agentic-loop.mjs`) **share no code**. The runtime slot
  is open.

RuVector's own ADR-256 already ruled: **borrow metaharness concepts, do not
take it as a runtime dependency** — and ruflo's ADR-150 enforces the same
invariant in the other direction (metaharness must stay removable).

---

## 4. RuFlo (ruvnet/ruflo = claude-flow v3, TS, ~173K LoC in the CLI)

Self-described *"agent meta-harness for Claude Code and Codex … Ruflo is the
harness — the execution layer around Claude Code."* `Agent = Model + Harness`.

- **The execution split:** ruflo **coordinates; Claude Code executes.**
  `agent_spawn` is registry metadata; multi-turn tool-use execution is
  delegated to Claude Code's Task tool or `claude -p` subprocesses. Ruflo
  itself executes only: single-turn LLM calls with bandit-fed model routing
  (`agent-execute-core.ts`), headless Claude subprocesses (worker daemon,
  fable-harness LLM-judge), and — the one real in-house turn loop —
  **rvagent's WASM build** via `@ruvector/rvagent-wasm`
  (`ruvector/agent-wasm.ts`, 27 `wasm_agent_*`/`wasm_gallery_*` MCP tools).
- **Memory substrate (battle-tested):** `.swarm/memory.db` (SQLite/AgentDB),
  `.swarm/hnsw.index`, `.rvf` containers; hybrid retrieval (BM25 +
  cross-encoder rerank + RaBitQ + graph edges); **ADR-323 provenance typing**
  (`user_claim|agent_output|system_observation|tool_result`) that any
  co-writing runtime must honor; ReasoningBank with HNSW-backed pattern
  promotion; SONA/EWC++/LoRA distillation pipeline
  (RETRIEVE→JUDGE→DISTILL→CONSOLIDATE).
- **Hooks bus:** all 8 Claude Code lifecycle events funnel through
  `.claude/helpers/hook-handler.cjs` — a subprocess contract a Rust binary
  could shim or replace. 17 hooks + 14 background workers.
- **Model routing:** 3-tier (codemods $0 / Haiku / Sonnet-Opus) with a
  persisted Thompson-sampling bandit closed-loop from execution outcomes.
- **MCP:** hand-rolled stdio JSON-RPC server, 305 tools; the **Capability
  Brain** (typed maturity/authority/risk/health metadata per domain) is the
  best-designed integration seam for advertising a new runtime.
- **Governance:** `.harness/mcp-policy.json` default-deny; ADR-150's
  invariant — metaharness may augment, never be required.
- **Honesty:** ruflo's docs audit their own claims (e.g. "150x-12,500x NOT
  reproduced — was brute-force fallback"). Its IMPROVEMENT-ROADMAP names
  **"skill synthesis vs Hermes-class agents"** the highest-leverage missing
  capability — DISTILL exists but never emits shareable SKILL.md artifacts.
- **rvagent is already first-class:** `plugins/ruflo-agent` wires WASM
  rvagent + `wasm_agent_compose` hands agents a **safety-gated allowlist of
  ruflo MCP tools** (destructive-tool gate included); plugins opt in via a
  `rvagent.exposeSkillsAsTools` manifest field.

---

## 5. Synthesis: the opening

| Harness | Loop | Learning loop | Evolution apparatus | Coordination plane | Native speed / WASM |
|---|---|---|---|---|---|
| Hermes | ✅ mature | ✅ (skills+GEPA) | partial (offline) | ❌ | ❌ (Python) |
| Claude Code | ✅ mature | partial (manual skills) | ❌ | partial (Task tool) | ❌ |
| metaharness | ❌ (bench-only) | ❌ | ✅ (Darwin/flywheel) | ❌ | kernel only |
| ruflo | ❌ (delegates) | partial (DISTILL, no skills) | partial (harness-loop) | ✅ (305 tools, memory, swarm) | via rvagent-wasm |
| **rvagent today** | ❌ broken | ❌ | ❌ | via A2A (no runner) | ✅ (Rust+WASM+NAPI) |
| **rvagent target** | ✅ Hermes-class | ✅ (skills+SONA+witness) | ✅ (via metaharness) | ✅ (via ruflo) | ✅ |

Every column of the target row has real code behind it somewhere in the
rUv ecosystem. The work is repair (rvagent's loop) plus wiring (the seams),
not greenfield invention. See [02-target-architecture.md](02-target-architecture.md).
