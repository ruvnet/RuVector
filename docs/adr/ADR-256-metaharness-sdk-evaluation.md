---
adr: 256
title: "Borrowing `metaharness` concepts into `npx ruvector` using primitives we already ship (router, witness, MCP, SONA, memory)"
status: proposed
date: 2026-06-16
authors: [ruvnet, claude-flow]
related: [ADR-026, ADR-103, ADR-122, ADR-134, ADR-252, ADR-255]
tags: [cli, npx, metaharness, harness, mcp, agent-sdk, router, witness-chain, startup-latency, dependency-policy, tooling]
---

# ADR-256 — Borrowing `metaharness` concepts into `npx ruvector`

> **Decision in one line.** `metaharness` is a real, same-ecosystem (rUv) npm
> package — but a **v0.1.x harness *generator*, not a runtime SDK** — and the
> capabilities it advertises (cost-optimal model routing, agentic tool surface,
> signed releases, memory + learning loops) are concepts **ruvector already
> ships natively** (Tiny Dancer cost-router, semantic router, MCP server,
> witness chain, SONA, hooks routing). Notably the dependency runs the *other*
> way: `metaharness` optionally depends on `@ruvector/ruvllm` and links
> `@ruvector/emergent-time`. So we **borrow the good concepts and implement them
> with the primitives we already use — we do NOT take the package as a runtime
> dependency.**

## Context

`npx ruvector` is the CLI published as the `ruvector` npm package (currently
**v0.2.31**, source at `npm/packages/ruvector/`). Its entry point
`bin/cli.js` is a `commander@11` CLI that **lazy-loads** every heavy dependency
specifically to protect cold-start latency: the Rust/WASM core (`../dist/index.js`),
`@ruvector/gnn`, `@ruvector/attention`, `@ruvector/sona`, `@ruvector/rvf`,
`@ruvector/tiny-dancer`, and `@modelcontextprotocol/sdk`. It already ships an
**MCP server** (`bin/mcp-server.js`) that exposes its commands as an agentic
tool surface to any MCP host (Claude Code, etc.).

The question raised: *could we improve `npx ruvector` by adopting the "npm
metaharness SDK"?* The term `metaharness` does **not** appear in this monorepo
(only an unrelated `meta-harness` string in tiny-dancer training), so the first
job was to establish what is actually being referred to before deciding.

A multi-source, evidence-graded research pass (npm registry + GitHub APIs + web)
established the following.

### Identity finding (Confidence: High, Evidence: H)

There is an npm package literally named **`metaharness`** (v0.1.14, published
**2026-06-16**), authored by **rUv (ruv@ruv.net)** — the same author/ecosystem
as `ruvector`. Repo: [`ruvnet/agent-harness-generator`](https://github.com/ruvnet/agent-harness-generator).

It is a **harness generator / factory**, not an embeddable runtime SDK. Tagline:
*"mint a custom AI agent harness from any repo… The model is replaceable. The
harness is the product."* You point it at a repo and it scaffolds an
npm-publishable, branded agent CLI (custom `npx <name>`, multi-agent pods, an MCP
server with default-deny security, Ed25519 witness-signed releases, a memory
namespace + learning loops), targeting six interactive hosts (Claude Code,
Codex, pi.dev, Hermes, OpenClaw, RVM).

- Bins: `metaharness` → `dist/bin.js`, `harness` → `dist/harness-bin.js`
- Direct deps are tiny (`kolorist`, `prompts`) — no Anthropic SDK, no MCP SDK,
  no `commander` bundled.
- Peer: `@metaharness/kernel@0.1.0` (Rust + WASM + NAPI, optional). Optional:
  `@ruvector/ruvllm`. Its kernel links `@ruvector/emergent-time` — corroborated
  locally by the untracked `docs/research/emergent-time/` dir in git status.
- Maturity: pre-1.0 beta (GitHub tag v0.1.3 vs npm 0.1.14 diverge), 117 stars,
  ~3,305 downloads/last month, CI green on Node 20/22 (download count/package.json
  Evidence H; feature/maturity claims single-source Evidence M).

**Name collisions ruled out** (all exist, none are the npm package): Superagentic
`metaharness` (Python/PyPI harness *optimizer*), Databricks **Omnigent**
(meta-harness for *composing/governing* agents), Strands **`harness-sdk`** (a
genuine TS/Python agent-harness SDK — the closest match had the user meant "a
harness SDK" generically), and the **Claude Agent SDK** (Anthropic's agent-loop
SDK). Given the author identity, the `@ruvector/*` linkage, the `npx` framing,
and the local `emergent-time` work, the intended referent is rUv's own npm
`metaharness` (Confidence: High). The "SDK" label is imprecise — the package is a
CLI/generator; the runtime pieces are separate `@metaharness/kernel` /
`@metaharness/router` packages.

### What `npx ruvector` already ships (the "concepts we use")

The CLI is not a thin wrapper — it already exposes most of metaharness's harness
concepts through primitives this repo built and maintains:

- **Cost-optimal model router (Tiny Dancer).** `ruvector tiny-dancer train`
  trains a FastGRNN router from a DRACO dataset (rows of `{embedding, scores}`)
  into a `.safetensors` model; `tiny-dancer route` scores a query embedding —
  *high = the cheap model is good enough → route cheap, else route stronger*
  (`bin/cli.js:1879–1950`, ADR-252). This **is** metaharness's "learned
  cost-optimal routing," already native and shipping for linux/macos/windows.
- **Semantic router.** `ruvector router --route` over `@ruvector/router` /
  `ruvector-router-core` ("Core vector database and neural routing inference
  engine") — text → best-matching intent (`bin/cli.js:1832–1879`).
- **Hooks / intelligence routing.** `npx ruvector hooks route "<task>"` routes a
  task to an agent/model via the intelligence engine's `route()` (the 3-tier
  model routing of ADR-026: Agent-Booster → Haiku → Sonnet/Opus).
- **Agentic tool surface.** The bundled **MCP server** (`bin/mcp-server.js`,
  `@modelcontextprotocol/sdk`) already exposes commands as LLM-drivable tools.
- **Witness-signed provenance.** Existing witness chain / manifest machinery
  (ADR-103, ADR-134) — the basis for signed releases.
- **Memory + learning loops.** SONA (LoRA/EWC++), ReasoningBank, pi-brain shared
  intelligence — persistent, self-learning state.

### Concept → ruvector primitive (what we borrow vs. what we already have)

| metaharness concept | ruvector primitive we already use | Action |
|---|---|---|
| Learned cost-optimal model routing | **Tiny Dancer** FastGRNN cost-router (ADR-252) | Reuse; surface as the harness router |
| Semantic intent routing | `@ruvector/router` / `ruvector-router-core` | Reuse |
| Multi-tier model routing | `hooks route` + 3-tier routing (ADR-026) | Reuse |
| Agentic tool exposure (MCP) | Bundled MCP server (`bin/mcp-server.js`) | Reuse; harden to default-deny |
| Ed25519 witness-signed releases | Witness chain / manifest (ADR-103, ADR-134) | Extend to release signing |
| Memory namespace + learning loops | SONA + ReasoningBank + pi-brain | Reuse; document a stable namespace |
| Default-deny security posture | (gap) | **Borrow** as a convention |
| "Harness is the product" framing | (gap) | **Borrow** as documentation framing |

## Decision

**Borrow the metaharness *concepts* and deliver them by composing ruvector
primitives we already ship — do NOT take `metaharness`/`@metaharness/*` as a
runtime dependency.**

Concretely:

1. **Do NOT add `metaharness`, `@metaharness/kernel`, or `@metaharness/router`
   as a runtime dependency of `ruvector`.** It is a build-time generator that
   re-scaffolds a *new* CLI, not a retrofit; its kernel is a Rust+WASM+NAPI beta
   whose init cost conflicts with the CLI's ms-startup invariant (Evidence H —
   grounded in the lazy-load architecture in `cli.js`). The dependency direction
   also runs *toward* ruvector (`metaharness` → `@ruvector/ruvllm` /
   `@ruvector/emergent-time`), so taking it as a dep would be circular.
2. **Promote the router we already have as the harness's cost-optimal router.**
   Use **Tiny Dancer** (ADR-252) as the canonical model-routing primitive and
   the **semantic router** for intent dispatch — this is the headline
   metaharness feature, already native. No `@metaharness/router` needed.
3. **Treat the shipped MCP server as the canonical agentic surface** and harden
   it toward the **default-deny allowlist** posture metaharness advocates.
4. **Borrow three metaharness conventions, building on our own machinery:**
   (a) default-deny MCP tool allowlist; (b) Ed25519 witness-signed releases by
   *extending* the existing witness chain (ADR-103/ADR-134), not replacing it;
   (c) a stable, documented `ruvector` memory namespace over SONA/ReasoningBank.
5. **If an embedded agent loop inside `npx ruvector` is ever wanted** (the CLI
   *drives* an LLM — a capability MCP does not provide), prefer the **mature
   Claude Agent SDK directly**, behind a lazy-loaded optional subcommand (e.g.
   `ruvector agent`), wired to the Tiny Dancer router for model selection.
6. **Re-evaluate a `@metaharness/*` dependency only** if it reaches ≥1.0 with a
   stable embeddable contract *and* a benchmark shows it beating Tiny Dancer.

This is a **borrow-concepts + dependency-policy** decision, not a refactor.
Items 2–4 reuse shipping primitives and are near-zero blast radius.

## Consequences

### Positive
- **We get the metaharness feature set by reusing what we own.** The headline
  capability — learned cost-optimal routing — is already shipping as Tiny Dancer
  (ADR-252); the agentic surface is the bundled MCP server; provenance is the
  witness chain. Borrowing concepts costs assembly, not a new engine.
- **Zero startup regression and zero new audit burden** — the ms-startup
  invariant verified in `cli.js` is preserved; no Rust+WASM+NAPI beta kernel on
  the load path.
- **No circular dependency.** Since `metaharness` depends on `@ruvector/ruvllm`
  and `@ruvector/emergent-time`, keeping ruvector dependency-free of
  `@metaharness/*` avoids an inverted/circular coupling.
- Reuses prior decisions: ADR-026 (3-tier routing), ADR-103/ADR-134 (witness),
  ADR-122 (RVF container), ADR-252 (Tiny Dancer cost-router).

### Negative
- **Borrowing is integration work, not free.** Surfacing Tiny Dancer + semantic
  router as a coherent "harness router," hardening MCP to default-deny, and
  extending witness signing to releases are real tasks that must be tracked, not
  left implicit.
- We forgo whatever a future shared `@metaharness/router` *might* offer; gated by
  decision item 6 (must beat Tiny Dancer on benchmark).
- Two routers (Tiny Dancer cost-router + semantic router) under one "harness
  router" framing risks confusion; the docs must state which routes what
  (cost/tier vs. intent).

### Neutral
- No new runtime dependency; `metaharness` stays an external, same-ecosystem
  reference whose good ideas we adopt as conventions. Revisit cheaply at ≥1.0.

## Alternatives considered

| Option | Verdict | Why |
|---|---|---|
| **Borrow concepts, reuse our primitives** | **Adopt** | Tiny Dancer router + MCP + witness + SONA already deliver the feature set; zero dep, zero startup cost, no circular coupling. |
| Adopt npm `metaharness` generator | Reject | Re-scaffolds a *new* CLI; does not improve a mature one. Headline benefit already native. |
| Depend on `@metaharness/kernel`/`router` | Reject (now) | Rust+WASM+NAPI 0.1.0 beta; overlaps Tiny Dancer + `@ruvector/router` + witness; threatens ms-startup; circular dep. Re-evaluate ≥1.0. |
| Generic harness SDK (Strands / Claude Agent SDK) | Defer | Embedded agent loop is a *new capability*. If wanted, Claude Agent SDK direct, lazy + optional, fed by Tiny Dancer routing. |
| Stricter commander / oclif | Reject | Off-target — CLI ergonomics, not the agentic-harness question. |

## Rollout sketch (borrow-concepts adoption)

0. **Now:** document the "harness router" framing — Tiny Dancer (cost/tier) +
   semantic router (intent) + `hooks route` as one coherent surface; harden the
   MCP server to a default-deny allowlist; document the `ruvector` memory
   namespace over SONA/ReasoningBank.
1. **Next:** extend the witness chain (ADR-103/ADR-134) to cover release
   signing (Ed25519), matching metaharness's signed-release pattern with our own
   machinery.
2. **Guardrail:** add a CI assertion that cold `npx ruvector --help` startup
   stays within its current ms budget, so no borrowed feature regresses startup.
3. **Optional future capability:** an embedded agent loop (`ruvector agent`) via
   the Claude Agent SDK, lazy + optional, model selection driven by Tiny Dancer.
4. **Gate for any `@metaharness/*` dep:** ≥1.0 + stable embeddable contract +
   benchmark beating Tiny Dancer; never bundled into the default load path.

## Implementation status (as shipped)

Tracked in issue #574 / PR #575 (branch `adr-256-harness-router`).

| Rollout item | Status | Where |
|---|---|---|
| 0. Harness router surface | **Done** | `ruvector harness status [--json]` (`bin/cli.js`) — unifies cost router, semantic router, hooks routing, MCP, witness, memory; degrades gracefully |
| 1. Default-deny MCP allowlist | **Done** | `bin/mcp-policy.js` + `mcp-server.js` gate; `RUVECTOR_MCP_ALLOW/DENY/PROFILE`; verified end-to-end over MCP stdio |
| 3. Memory namespace | **Done** | `RUVECTOR_MEMORY_NAMESPACE` (default `ruvector`), surfaced as `memory.namespace` |
| 4. Startup-budget guard | **Done** | `test/startup-budget.js` — abs ceiling + relative delta (harness adds ≈ +0–3ms) |
| Benchmarked / verified | **Done** | full `npm test` green (cli 73/0, mcp-policy 8/0, startup 2/0, db-workflow, integration, sigterm) |
| 2. Witness-signed releases | **Convention** | documented extension of the existing witness chain (ADR-103/ADR-134); no new signing engine built |
| 5. Embedded `ruvector agent` loop | **Deferred** | optional future capability via Claude Agent SDK, per the decision |

No `@metaharness/*` runtime dependency was added; the borrowed capabilities are
delivered entirely by primitives ruvector already ships.

## Links
- npm `metaharness` (v0.1.14, 2026-06-16, author rUv):
  https://registry.npmjs.org/metaharness/latest ·
  downloads: https://api.npmjs.org/downloads/point/last-month/metaharness
- Generator repo: https://github.com/ruvnet/agent-harness-generator
- Name collisions (NOT this package): Superagentic metaharness
  (https://superagenticai.github.io/metaharness/) · Databricks Omnigent
  (https://www.databricks.com/blog/introducing-omnigent-meta-harness-combine-control-and-share-your-agents) ·
  Strands harness-sdk (https://github.com/strands-agents/harness-sdk) ·
  awesome-harness-engineering (https://github.com/ai-boost/awesome-harness-engineering)
- Local grounding: `npm/packages/ruvector/bin/cli.js` — lazy-load startup
  architecture, Tiny Dancer cost-router (`:1879–1950`), semantic router
  (`:1832–1879`), `hooks route` (`:4391`); `ruvector-router-core` ("Core vector
  database and neural routing inference engine"); `package.json` (v0.2.31,
  `@modelcontextprotocol/sdk` + `@ruvector/router` peer dep); `bin/mcp-server.js`
  (shipped MCP surface); git status (`docs/research/emergent-time/` untracked,
  corroborating the `@ruvector/emergent-time` linkage).
- Prior decisions: ADR-026 (3-tier model routing), ADR-103 (witness manifest),
  ADR-122 (RVF container), ADR-134 (witness chain), ADR-252 (Tiny Dancer
  FastGRNN cost-router), ADR-255 (OIA alignment profile).
- Source: deep-research brief (npm/GitHub registry APIs + web) cross-referenced
  with the ruvector CLI source.
