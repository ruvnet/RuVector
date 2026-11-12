# Program Plan — RuV Perpetual Intelligence Runtime

Status: draft for ADR seeding, swarm work-package assignment, and GitHub issue filing. Compiled 2026-08-19, revised same day after a second, deeper asset-inventory pass (asset-scout) grounded several sections in verified file paths, LOC counts, and ADR numbers — see `02-asset-map.md`'s corrections. Depends on `01-evidence-review.md` (evidence grades) and `02-asset-map.md` (component map). Read the asset map's headline finding first: this program extends `ruvnet/LatentMesh` ADR-009's "online causal control loop" (2026-08-18) rather than designing from scratch.

**Revision note**: the second inventory pass materially changed two things worth knowing before reading the rest of this plan. First, "Dream Machine" is *not* a from-scratch build — real, CI-wired, cryptographically-signed statistical promotion machinery already exists (`scripts/research-gate/`, `crates/ruvector-sota-bench/harness`, ADR-282) and should be adopted/renamed, not reinvented; this meaningfully de-risks what was previously flagged as the top program risk. Second, ADR numbering in this repo is not the "ADR-001 through ADR-118" stated in the root `CLAUDE.md` — the true count is 283 files with duplicated numbers up to ADR-304 (see asset map §6 of the cross-cutting notes) — every new ADR in this program must be numbered above 304.

## Governing invariants

Invariants 1–6 are carried from the brief, unchanged. Invariant 7 was added
during ADR authoring (PR #847, ADR-305), adopted from ruflo ADR-322B's
proposer/promotion separation-of-powers rule — see ADR-305 for the source
citation.

1. Every observation may change memory.
2. Every memory change is transactional.
3. Every agent communication is attributable.
4. Every behavioral mutation is tested.
5. Every promoted mutation must outperform its parent.
6. Every physical action produces new evidence.
7. A proposer produces untrusted candidates only; it cannot issue promotion
   decisions or mutate active policy (adopted from ruflo ADR-322B, binding
   on ADR-306, ADR-313, ADR-315).

Risk control: Darwin proposes, RVM gates what can change, Dream Machine requires statistically credible improvement before promotion.

## Scope decision

Build **1 + 2 + 3** (self-evolving physical intelligence, persistent transactional memory, verified latent communication) as the combined "Perpetual Intelligence Runtime" branch. Fast-follow with **4** (KV-cache cross-model migration in ruvLLM — grade-A evidence, cheapest to ship). Treat **5** (universal RF/CSI vocabulary) as a stretch research bet inside `RuView` with no external prior art to lean on (see evidence review item 8) — sequence it last and fund it separately if 1–4 land on schedule.

---

## Bounded contexts (DDD)

| Context | Owns | Primary repo(s) |
|---|---|---|
| **Physical Skill Evolution** | Skill/harness genome, mutation surfaces, rollout execution | `agent-harness-generator` (Darwin core), `ruvector` (`crates/sona`, `crates/ruvllm` mutator backend) |
| **World Verification** | Reversible-action verification, dream-replay pre-filter, promotion scoring | `ruvector` (`crates/ruvector-nervous-system`, SONA dream engine, `ruvector-proof-gate`) |
| **Persistent Memory Governance** | Three-level memory (working context → continuous latent state → transactional RuVector memory), TARL-style ledger, RVF/RVM provenance | `ruvector` (`crates/rvf`, `crates/rvm`, `ruvector-core`), `agentdb` |
| **Latent Communication Fabric** | Latent transport, causal-edge verification, quarantine, capability-governed execution | `ruvnet/LatentMesh` (primary), `ruvnet/rvm` (`rvm-witness`, `rvm-cap`) |
| **Cross-Model Cognition Migration** *(fast-follow)* | KV-cache mapping, transfer-quality prediction, routing gate | `ruvector` (`crates/ruvllm`) |
| **RF Sensing Substrate** *(stretch)* | Hardware-independent CSI representation, pose-semantic embedding | `ruvnet/RuView`, `ruvector` (`crates/ruvector-hailo-cluster`) |
| **Governance & Constitution** *(cross-cutting)* | Admission gates, capability ceilings, rollback, cross-repo witness convergence | `ruvnet/autogenous`, `ruvnet/rvm` |

---

## ADR list

Numbers are placeholders — assign real numbers **above ADR-304** via `ruflo-adr:adr-create` at kickoff (the repo's true max, not the stale "ADR-118" in root `CLAUDE.md` — see asset map §6), and confirm against both `ruvector`'s and `LatentMesh`'s ADR sequences before merging (see asset map note on ADR-103 ambiguity). ADR-create should also register each new ADR in `ruflo-adr:adr-index` immediately to avoid adding to the duplicate-number problem it's meant to fix.

**This list has eleven ADRs (1–11 below); the two work packages below numbered
WP0a and WP0b are process/bug-fix work with no corresponding ADR** — they
were mistakenly written as "ADR — …" entries in an earlier draft of this
plan, which claimed thirteen ADRs against the eleven PR #847 actually
shipped (ADR-305 through ADR-315). Reworded here as work-package-only items
so this plan and the shipped ADR set agree:

- **WP0a — ADR numbering hygiene remediation** (no ADR). Audit + fix pass
  over `docs/adr/` to resolve the ~15+ duplicate ADR numbers found (ADR-272
  ×5, ADR-264/252/194/144/040 ×3 each, ~12 more ×2), by renumbering the
  later-created duplicate in each collision to the next free number above
  304 and updating all in-repo references to match. Process hygiene,
  tracked as a separate issue, not a PIR ADR.

1. **ADR — Adopt LatentMesh ADR-009 as the Perpetual Intelligence Runtime's control-loop spine.** Decision: rather than defining a new cross-mechanism architecture, this program formally adopts the `execute → transfer → causal audit → measure → update authority → persist → evolve topology` loop from LatentMesh ADR-009, and commits to closing the specific gap that ADR names ("statistical primitive and admission gate implemented; closed loop across live components not wired"). Coordination channel opened with LatentMesh maintainers before any conflicting ADR numbers are assigned.

2. **ADR — Adopt `research-gate`/`ruvector-sota-bench` as the Dream Machine equivalent, and wire it to Darwin.** Decision: rather than building a new evaluation service, formally adopt the existing `scripts/research-gate/` + `.github/workflows/research-*.yml` + `crates/ruvector-sota-bench/harness` (paired-bootstrap `statistics.ts`, hard-veto `vetoes.ts`, Ed25519-signed `flywheel.ts` replay bundles), documented in ADR-282, as the Dream Machine role in this program. The concrete work is wiring it as the mandatory evaluation stage between Darwin's mutation proposals and RVM's proof-gated promotion — optionally with SONA's dream-replay engine (`05-MEMORY-DREAMS.md`) as a pre-filter feeding candidates into it — not writing a parallel system. This significantly reduces the risk previously flagged as the program's biggest gap.

3. **ADR — Three-level persistent memory architecture (LiveMem + TARL pattern) on RuVector's existing memory stack.** Decision: implement working-context / continuous-latent-state / transactional-RuVector-memory as three explicit tiers mapped onto components that mostly already exist: working context → `ruvllm`'s `working_memory` module; continuous latent state → `episodic_memory`/`semantic_cache`/`agentic_memory` (following LiveMem's fixed-capacity recurrent-state design, arXiv:2608.02515); transactional memory → `ruvector-agent-memory` (ADR-252) + `reasoning_bank` + `ruvector-temporal-coherence` (ADR-211), extended with TARL's five-operation ledger (add/ignore/revise/reject/defer, arXiv:2608.03699). Proof-gated writes for this tier **already exist** (ADR-194, ADR-047) — the net-new work is the TARL-style ledger states and operation semantics layered on top of the existing proof-gate, not the gate itself.

4. **ADR — WorldCycle-style verification for the physical action loop.** Decision: adopt reversible-action-sequence verification (arXiv:2608.04964) as the WorldCycle-verification stage between "observe consequences" and "Dream Machine evaluation" in the physical loop, targeting the paper's reported 44% long-horizon drift reduction and ~4x composite-action accuracy as the acceptance bar for Phase 3's world-model component.

5. **ADR — Build LatentMesh integration inside `ruvector` as new crates, coordinated with the external design.** Decision: since `ruvector` has zero existing wiring to `ruvnet/LatentMesh` today (only an unmerged `origin/docs/link-latentmesh` branch stub), build the `ruvector`-side integration as new crates under `crates/rvAgent/` (alongside the existing `rvagent-a2a`, ADR-159) or a new `latentmesh` crate family — not as "finishing an integration" that doesn't yet exist. Implement the network transport, RVF packaging (model identity, transform, permitted recipients, provenance, witness history as artifact metadata per LatentMesh ADR-008), and RVM `rvm-cap` admission enforcement that LatentMesh's own ADRs mark as "not implemented" externally too. This requires an explicit coordination channel with the `ruvnet/LatentMesh` maintainers so the two sides converge on the same wire format rather than diverging.

6. **ADR — Causal-attribution gate for latent communication.** Decision: every latent-channel deployment must pass a controlled-replacement causal audit modeled on arXiv:2607.26773 before its performance claims can be used to justify further rollout — this becomes a required CI gate on any change touching `latentmesh-align` or downstream consumers, not just a one-time paper reproduction.

7. **ADR — Anomaly quarantine for latent channels (net-new, not "LATTE").** Decision: since no verified prior art named "LATTE" exists (evidence review item 6, UNVERIFIED), build quarantine as a first-party contribution combining arXiv:2606.28958's HMAC-manifest integrity check with `rvm-witness`/`autogenous witness` provenance chains — explicitly documented as original work, not an implementation of a paper that doesn't exist under that name.

8. **ADR — Resolve the RVM/Autogenous witness-crate duplication.** Decision: `rvm-witness` becomes the canonical witness-chain implementation; `autogenous`'s `witness` crate is refactored to depend on it rather than maintaining a parallel cryptographic-provenance implementation. Both repos' maintainers must sign off before this lands.

9. **ADR — SHAPER-pattern skill/harness evolution loop (frozen weights).** Decision: implement the physical-intelligence evolution loop with foundation-model weights frozen throughout, following arXiv:2608.11350's pattern — the same frozen model serves as planner and optimizer, evolving only skills, context, and the execution harness (mapped to Darwin's mutation surfaces), never the weights themselves. This is the acceptance test's central frozen-weights constraint and must be enforced structurally (no fine-tuning code path reachable from the promotion pipeline), not just by policy.

10. **ADR — KV-cache cross-model migration in ruvLLM (fast-follow).** Decision: implement arXiv:2608.03893's closed-form linear KV-cache mapper for same-family model migration, with the nonlinear MLP fallback for the pairs it identifies as degrading, plus a routing gate that predicts transfer quality before migrating (never migrate blind). Ships independently of the Phase 1–3 branch since it depends only on `ruvllm`.

11. **ADR — Governance constitution for capability expansion.** Decision: adopt `autogenous`'s constitution/admission-gate pattern as the enforcement point for "zero unapproved capability expansion" (acceptance-test requirement); every mutation that would expand an agent's capability set (new tool access, new physical action class, new communication peer) requires explicit constitutional approval logged to the witness chain, distinct from ordinary behavioral mutation promotion. Note `autogenous`'s own README self-labels it "research prototype" status — treat its APIs as unstable and budget time for API churn, don't assume production-grade stability.

- **WP0b — MetaHarness dependency-compliance remediation** (no ADR). Fix the
  confirmed bug where `crates/ruvector-sota-bench/harness`'s nine
  `@metaharness/*` npm dependencies are declared as plain (hard)
  dependencies while `METAHARNESS-README.md` claims `optionalDependencies`
  compliance, attributed there to "**ADR-150**: MetaHarness Integration
  Surfaces (**upstream**)." Neither `ruvector`'s own ADR-150
  (`pi-brain-ruvltra-tailscale`) nor `metaharness`'s own ADR-150
  (`tailscale-local-frontier-concurrent-benchmarks`) is the right document
  — both unrelated. **The upstream document is `ruflo` ADR-150**
  (`v3/docs/adr/ADR-150-metaharness-integration-surfaces.md`, "MetaHarness
  Integration Surfaces in `npx ruflo`," Status **Implemented**, 2026-06-16),
  whose rule 2 is verbatim the policy: *"`@metaharness/*` packages MUST
  appear in `optionalDependencies` or `peerDependencies` (optional), never
  in `dependencies`."* Fix the bug directly against that source — make the
  nine packages genuinely optional per ruflo ADR-150 rule 2, or correct the
  documentation to state the real (hard-dependency) install requirement —
  and adopt rule 4 (a CI job on the `--ignore-optional` install path,
  "the only structural defense against accidentally promoting an optional
  dep to required") as the acceptance criterion, stronger than a plain
  successful `npm install`. **The previously-tracked HTTP-307 redirect bug in
  `ruvllm`'s model-download path is already fixed on `main`** (commit
  `946275a61`, PR #590, 2026-06-18); it is not part of this work package.
  Verifying that fix surfaced the actual remaining download blocker: a GGUF
  glob/alias bug in `ruvllm-cli`'s `get_files_to_download()`
  (`download.rs:193`'s glob pattern, `models.rs:65`'s alias resolution),
  which this work package tracks and fixes instead.

---

## Work packages (sized for 6–8 agent swarm teams, per project anti-drift config)

| # | Package | Bounded context | Team composition | Depends on |
|---|---|---|---|---|
| WP0a | ADR numbering hygiene remediation | Governance | coordinator, adr-architect | — |
| WP0b | MetaHarness dependency-compliance + ruvllm-cli GGUF glob/alias fix | Physical Skill Evolution | coordinator, backend-dev, tester | — |
| WP1 | LatentMesh coordination & ADR alignment | Governance | coordinator, adr-architect, system-architect | WP0a |
| WP2 | Adopt research-gate/sota-bench as Dream Machine; wire to Darwin | World Verification | coordinator, system-architect, coder ×2, tester | WP1 |
| WP3 | Three-level memory tiers on RuVector (LiveMem + TARL ledger on top of existing proof-gated writes) | Persistent Memory Governance | coordinator, backend-dev, memory-specialist, tester | WP1 |
| WP4 | TARL ledger states wired into existing ADR-194/047 proof-gate | Persistent Memory Governance | coordinator, security-architect, coder | WP3 |
| WP5 | LatentMesh `ruvector`-side crates (new, under `crates/rvAgent/`) + cross-repo wire-format coordination | Latent Communication Fabric | coordinator, system-architect (cross-repo), coder ×2, tester | WP1, WP4 |
| WP6 | Causal-audit CI gate | Latent Communication Fabric | coordinator, security-auditor, tester | WP5 |
| WP7 | Anomaly quarantine (net-new — no "LATTE" prior art exists) | Latent Communication Fabric | coordinator, security-architect, coder, tester | WP5, WP6 |
| WP8 | Witness-crate convergence (rvm ↔ autogenous) | Governance & Constitution | coordinator, system-architect (cross-repo), coder | WP1 |
| WP9 | SHAPER-pattern skill/harness evolution loop (Darwin via `@metaharness/darwin`) | Physical Skill Evolution | coordinator, system-architect, coder ×2, tester, reviewer | WP0b, WP2 |
| WP10 | WorldCycle verification stage | World Verification | coordinator, coder, tester | WP9 |
| WP11 | Constitutional capability-expansion gate (autogenous, research-prototype status) | Governance & Constitution | coordinator, security-architect, coder | WP8 |
| WP12 | 30-day acceptance harness | Cross-cutting | coordinator, performance-engineer, tester, observability-engineer | WP2, WP4, WP9, WP10, WP11 |
| WP13 *(fast-follow)* | ruvLLM KV-cache cross-model migration (crates/ruvllm: kv_cache.rs, paged_attention.rs, serving/kv_cache_manager.rs) | Cross-Model Cognition Migration | coordinator, coder ×2, tester | none — parallel track |
| WP14 *(stretch)* | RuView pose-semantic embedding + universal CSI vocabulary (pick up ADR-178 gaps C & D) | RF Sensing Substrate | coordinator, ml-developer, coder, tester | independent — fund separately |

Use `hierarchical` topology, `max-agents 8`, `specialized` strategy per project config for each work package; spawn WP1–WP11 teams as concurrent swarms once WP1 clears (they gate on each other per the dependency column, not on serial scheduling).

---

## Security / validation gates

- **Structural frozen-weights enforcement (WP9)**: the promotion pipeline must have no code path that can write to foundation-model weight files — verified by a CI check that fails the build if any mutation surface imports a training/fine-tuning API.
- **Witness-chain requirement**: every state transition in Persistent Memory Governance, Latent Communication Fabric, and Governance & Constitution contexts must emit an RVM witness record (ADR-134 schema) before it's considered committed. No RVF write without a corresponding witness entry.
- **Proof-gated promotion**: `ruvector-proof-gate`/`rvm-proof` must approve every mutation promotion; Dream Machine's verdict is an input to the proof gate, not a bypass of it.
- **Causal-audit CI gate (WP6)**: any PR touching `latentmesh-align`, the quarantine module, or downstream consumers must pass a controlled-replacement causal audit before merge — modeled on arXiv:2607.26773's methodology, run against the same benchmark families (GSM8K/ARC-C/MATH-500-equivalent internal tasks).
- **Constitutional admission gate (WP11)**: capability-expanding mutations (new tools, new physical action classes, new communication peers) require a separate, higher-bar approval than ordinary behavioral mutations, logged distinctly in the witness chain so the acceptance test can query "zero unapproved capability expansion" directly.
- **Hosted-RVM honesty discipline**: per RVM's own ADR-285, any Cloud Run–hosted component of this program must not claim bare-metal isolation strength it doesn't have — carry that same claims-honesty discipline into this program's own status reporting.
- **Standard repo gates**: `npx @claude-flow/cli@latest security scan` after any change touching auth, capability tables, or witness signing, per repo-root `CLAUDE.md`.

---

## GCP deployment / publishing surface

- **Dream Machine control plane**: new Cloud Run service in project **`ruv-dev`**, region `us-central1` (mirror `mcp-brain-server`'s deployment as `ruvbrain` — same project, same region, session-affinity pattern, reuse its `cloudbuild` files as a template), fronting WP2's adopted research-gate/sota-bench evaluation service; secrets pulled from the existing Secret Manager entries (`ANTHROPIC_API_KEY`, `GOOGLE_AI_API_KEY`, etc.) rather than new ones where possible.
- **LatentMesh transport gateway**: if network transport (WP5) needs a rendezvous/relay point beyond direct P2P, host it as a second Cloud Run service, session-affinity enabled, in the same region for latency parity with the memory/brain services it talks to.
- **Cloud Scheduler jobs**: extend the existing 7-job pattern (train, drift, transfer, graph, attractor, cleanup, full) with new jobs for: nightly causal-audit re-runs (WP6), witness-chain integrity verification sweeps (WP8), and 30-day acceptance-harness daily checkpoint aggregation (WP12).
- **npm packages** (`@ruvector` scope): `@ruvector/dream-machine` (WP2), `@ruvector/latent-quarantine` (WP7), `@ruvector/kv-migrate` (WP13) — each following the existing `@ruvector/*` publish flow (see the user's own remembered npm release procedure: worktree flow, `--force --ignore-scripts` install, no-OTP publish, main-vs-registry drift check).
- **Crates**: new crates land under `crates/` in `ruvector` for anything not owned by a sibling repo (e.g., `ruvector-dream-machine`, `ruvector-worldcycle-verify`); cross-repo crates (`latentmesh-*`, `rvm-*`, `autogenous-*`) are published from their own repos per their existing release process.
- **Cross-repo coordination**: since LatentMesh, RVM, RuView, and Autogenous are sibling repos, this program needs a lightweight release-train coordination doc (owned by WP1) tracking version compatibility across the five repos involved — do not let `ruvector` pin to unreleased commits of the others without an explicit compatibility ADR.

---

## 30-day acceptance test harness design

**Precondition**: foundation model weights frozen and hash-verified at day 0; the structural enforcement from WP9's gate confirms no write path exists to weight files for the duration of the run.

**Daily cycle** (automated, via WP12):
1. Run the full physical/latent/memory loop for a fixed batch of tasks (held-out task set, not seen during any prior day's mutation testing, to prevent overfitting to the eval set itself).
2. Log per-task: latency, task success (binary + graded where applicable), sensing accuracy (RuView/mmWave ground-truth comparison where available), token/compute cost.
3. Any proposed mutation that day goes through: Darwin propose → Dream Machine evaluate (dream-replay pre-filter + sandbox score) → proof-gate check → constitutional gate (if capability-expanding) → promote or reject, each step witness-logged.
4. Nightly causal-audit and witness-integrity sweeps (Cloud Scheduler jobs above) run against that day's promoted mutations.

**Day-30 evaluation**:
- Statistical comparison of day-30 vs. day-1 metrics using a paired bootstrap or equivalent test per metric (latency, task success, sensing accuracy, cost) — "outperforms" must be a stated significance threshold (e.g., p < 0.05 with a pre-registered minimum effect size), not a raw mean comparison, to avoid the exact "gains don't prove causation" failure mode the causal-audit evidence (item 5) warns against.
- **Full mutation provenance**: every promoted mutation across the 30 days must be traceable through the witness chain from proposal → evaluation → promotion, queryable end-to-end.
- **Zero unapproved capability expansion**: diff the capability/tool/action tables at day 0 vs. day 30; any expansion must have a matching constitutional-gate approval record (WP11) or the acceptance test fails outright, independent of the performance numbers.
- **Frozen-weights verification**: re-hash the foundation model weights at day 30 and confirm bit-identical to day 0.

**Failure handling**: any day where a promoted mutation later fails its causal audit or proof-gate re-check triggers automatic rollback via `autogenous`'s existing promotion/rollback controller (asset map §9) — the acceptance test should include at least one injected failure scenario to confirm rollback actually fires during the 30-day window, not just in isolated unit tests.

---

## Top risks (revised after second-pass grounding)

1. **Cross-repo coordination overhead, not Dream Machine, is now the top risk.** Four of the seven bounded contexts depend on sibling repos (`LatentMesh`, `rvm`, `autogenous`, `RuView`) maintained outside this program's direct control — LatentMesh's own ADR-009 shows its maintainers are still actively revising scope (twice in one day, per its own text), `autogenous` self-labels "research prototype," and `ruvector` has zero existing wiring to either LatentMesh or Autogenous today. This program's timeline is exposed to their churn and to genuinely new (not integration) engineering effort on the `ruvector` side. *(Dream Machine was the top risk in the first pass of this plan — the second inventory pass found real, CI-wired promotion machinery already exists under `research-gate`/`ruvector-sota-bench`, which downgrades that specific risk substantially.)*
2. **Two of eight founding claims are unverified, and one previously-"external" component (LATTE) turns out to have zero prior art anywhere.** Treating WP7 (quarantine) and WP14 (universal CSI vocabulary) as "implement the paper" instead of "build it for the first time" will produce schedule and credibility risk if that distinction gets lost downstream (e.g., in an ADR that cites "LATTE" as if it were a real paper).
3. **ADR numbering hygiene is worse than expected and could actively cause confusion mid-program.** 283 files, true max ADR-304, with ~15+ genuine duplicate numbers (one number reused 5 times). If WP0a isn't run early, this program's own new ADRs risk colliding with existing (possibly still-duplicate) numbers, and cross-references to "ADR-X" throughout this plan and the asset map need the reader to know which of several same-numbered documents is meant.
4. **A real dependency/compliance bug is already blocking part of the pipeline this program needs.** The MetaHarness `optionalDependencies` non-compliance (hard dependency on nine `@metaharness/*` packages) and the `ruvllm-cli` GGUF glob/alias bug in `get_files_to_download()` (blocks Darwin's local-mutator live-serve e2e testing) are concrete, small, already-known issues — cheap to fix (WP0b) but currently unfixed, and WP9 (the SHAPER-pattern evolution loop, arguably the program's centerpiece) depends on both being resolved first. (A third bug this plan previously tracked here, an HTTP-307 redirect in `ruvllm`'s model-download path, is already fixed on `main` — commit `946275a61`, PR #590, 2026-06-18 — and is not part of WP0b's remaining scope.)

---

## GitHub issue breakdown

Every work package above maps to one or more fileable GitHub issues, one per repo it touches (a cross-repo WP gets a linked issue in each repo rather than one issue pretending to speak for two codebases). **These are ready-to-file specs only — do not create them; the coordinator files them with `gh issue create` once this plan is approved.** File the five epics first, then child issues referencing their epic number, then edit each epic body to link the child issue numbers back in (`gh issue create` → capture number → `gh issue edit <epic> --body-file`).

Suggested label set to create in each repo before filing (`gh label create`): `pir` (every issue in this program), `epic`, `adr` (issue produces/updates an ADR), `security` (touches witness/proof-gate/capability/quarantine), `cross-repo` (coordination-dependent), and phase labels `phase-0`, `phase-1`, `phase-2`, `phase-3`, `fast-follow`, `stretch`.

### Epics (file first, one per repo)

| Epic | Repo | Title | Labels | Links to |
|---|---|---|---|---|
| E1 | `ruvnet/ruvector` | `[PIR] Epic: Perpetual Intelligence Runtime — ruvector work packages` | `pir`, `epic` | WP0a, WP0b, WP2, WP3, WP4, WP5-ruvector, WP6, WP7, WP9, WP10, WP12, WP13, WP14-companion |
| E2 | `ruvnet/LatentMesh` | `[PIR] Epic: Perpetual Intelligence Runtime — LatentMesh coordination` | `pir`, `epic` | WP1-latentmesh, WP5-latentmesh |
| E3 | `ruvnet/rvm` | `[PIR] Epic: Perpetual Intelligence Runtime — witness-chain convergence` | `pir`, `epic` | WP8-rvm |
| E4 | `ruvnet/autogenous` | `[PIR] Epic: Perpetual Intelligence Runtime — governance integration` | `pir`, `epic` | WP8-autogenous, WP11-autogenous |
| E5 | `ruvnet/RuView` | `[PIR] Epic: Perpetual Intelligence Runtime — RF sensing substrate` | `pir`, `epic` | WP14-ruview |

Each epic body: one paragraph linking to `docs/research/perpetual-intelligence-runtime/03-program-plan.md` in `ruvnet/ruvector` (the canonical plan document — sibling repos should link to it rather than duplicating the plan), plus a checklist of its child issue numbers to be filled in after filing.

### Child issues

**WP0a — ADR numbering hygiene remediation**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `adr`, `phase-0`.
- Title: `[PIR][WP0a] Resolve duplicate ADR numbers in docs/adr/ (ADR-272 ×5, ADR-264/252/194/144/040 ×3, ~12 more ×2)`
- Body — Goal: audit `docs/adr/` and resolve every duplicate ADR number by renumbering the later-created file in each collision to the next free number above the true max (currently ADR-304), updating all in-repo references. Acceptance criteria: `ruflo-adr:adr-index` reports zero duplicate numbers; every renumbered ADR's old-number references (code comments, other ADRs, MCP tool descriptions) are updated; a CI check is added that fails on future duplicate ADR numbers. Dependencies: none — do this first, it blocks every other ADR this program creates.

**WP0b — MetaHarness dependency compliance + ruvllm-cli GGUF glob/alias fix**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `phase-0`.
- Title: `[PIR][WP0b] Fix MetaHarness optionalDependencies non-compliance and ruvllm-cli GGUF glob/alias bug`
- Body — Goal: (1) make the nine `@metaharness/*` packages in `crates/ruvector-sota-bench/harness` genuinely optional per ruflo ADR-150 rule 2 (`v3/docs/adr/ADR-150-metaharness-integration-surfaces.md`, "MetaHarness Integration Surfaces in `npx ruflo`," Implemented 2026-06-16 — the real source `METAHARNESS-README.md` attributes its policy to; neither `ruvector`'s nor `metaharness`'s own ADR-150 is the right document, both unrelated), or correct the documentation to state the real hard-dependency requirement; (2) fix the GGUF glob/alias mismatch in `ruvllm-cli`'s `get_files_to_download()` (`download.rs:193`, `models.rs:65`) blocking Darwin's local-mutator live-serve e2e tests (ADR-259). Note: the HTTP-307 redirect bug this issue previously also tracked is already fixed on `main` (commit `946275a61`, PR #590, 2026-06-18) and is out of scope here. Acceptance criteria: a CI job passes on the `--ignore-optional` install path per ruflo ADR-150 rule 4 (stronger than a plain successful `npm install`) without the `@metaharness/*` packages present (if made optional), or documentation matches reality (if not); `ruvllm`-backed Darwin mutator passes a live-serve end-to-end test. Dependencies: none. **Blocks WP9.**

**WP1 — LatentMesh coordination & ADR alignment**
- Repo: `ruvnet/ruvector` (primary) + linked issue in `ruvnet/LatentMesh`. Labels: `pir`, `adr`, `cross-repo`, `phase-0`.
- Title (ruvector): `[PIR][WP1] Adopt LatentMesh ADR-009's control loop as the runtime spine; open cross-repo coordination`
- Title (LatentMesh): `[PIR][WP1] Coordination request: ruvector is building a Perpetual Intelligence Runtime on top of ADR-009`
- Body — Goal: formally adopt LatentMesh ADR-009's `execute → transfer → causal audit → measure → update authority → persist → evolve topology` loop as this program's architecture; open a standing coordination thread with LatentMesh maintainers so `ruvector`-side work (WP5) doesn't diverge from LatentMesh's own evolving design (it revised scope twice in one day per ADR-009's own text). Acceptance criteria: a coordination doc/thread exists and is linked from both epics; this program's new ADR-hygiene-corrected ADR (WP0a) explicitly cites and doesn't contradict LatentMesh ADR-001–009. Dependencies: WP0a.

**WP2 — Adopt research-gate/sota-bench as Dream Machine; wire to Darwin**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `adr`, `phase-1`.
- Title: `[PIR][WP2] Wire scripts/research-gate + ruvector-sota-bench/harness as the mutation-promotion gate (Dream Machine role)`
- Body — Goal: connect Darwin's mutation proposals to the existing `research-gate`/`ruvector-sota-bench/harness` statistical promotion machinery (paired bootstrap `statistics.ts`, hard vetoes `vetoes.ts`, Ed25519-signed replay bundles `flywheel.ts`, ADR-282) as the mandatory evaluation stage before RVM proof-gated promotion; optionally feed SONA's dream-replay engine in as a pre-filter. Acceptance criteria: no mutation reaches RVM's proof gate without passing a `research-gate` verdict; verdict + replay bundle are witness-logged. Dependencies: WP1.

**WP3 — Three-level memory tiers on RuVector**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `adr`, `phase-1`.
- Title: `[PIR][WP3] Map LiveMem/TARL three-tier memory onto ruvllm context modules + ruvector-agent-memory`
- Body — Goal: formalize working-context (ruvllm `working_memory`) / continuous-latent-state (`episodic_memory`/`semantic_cache`/`agentic_memory`) / transactional-memory (`ruvector-agent-memory` ADR-252 + `reasoning_bank` + `ruvector-temporal-coherence` ADR-211) as an explicit three-tier architecture per LiveMem's design (arXiv:2608.02515). Acceptance criteria: each tier has a documented interface and a test proving state survives context eviction (LiveMem's core claim). Dependencies: WP1.

**WP4 — TARL ledger states on existing proof-gate**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `adr`, `security`, `phase-1`.
- Title: `[PIR][WP4] Add TARL five-operation ledger (add/ignore/revise/reject/defer) onto ADR-194/047 proof-gated memory writes`
- Body — Goal: implement TARL's (arXiv:2608.03699) five executable memory operations and accepted/pending/rejected ledger states as a layer on top of the memory writes already proof-gated by ADR-194/047 — this is a smaller scope than originally planned since the underlying gate already exists. Acceptance criteria: a poisoning-style adversarial test (per TARL's own eval) shows rejected/deferred writes never reach transactional memory. Dependencies: WP3.

**WP5 — LatentMesh `ruvector`-side crates**
- Repo: `ruvnet/ruvector` (new crates) + linked issue in `ruvnet/LatentMesh` (wire-format sign-off). Labels: `pir`, `cross-repo`, `phase-2`.
- Title (ruvector): `[PIR][WP5] Build new crates/rvAgent/latentmesh-* crates (greenfield — no existing ruvector wiring)`
- Title (LatentMesh): `[PIR][WP5] Wire-format compatibility review for ruvector's new LatentMesh client crates`
- Body — Goal: `ruvector` has zero existing LatentMesh integration (only an unmerged `origin/docs/link-latentmesh` branch) — build new crates under `crates/rvAgent/` (alongside `rvagent-a2a`, ADR-159) implementing network transport, RVF packaging, and `rvm-cap` admission enforcement against LatentMesh's `latentmesh-core`/`latentmesh-align`/`latentmesh-gate` wire format. Acceptance criteria: a live multi-agent task runs LatentMesh's causal-edge verification (ADR-003) end-to-end for the first time, from `ruvector`; LatentMesh maintainers sign off on wire-format compatibility. Dependencies: WP1, WP4.

**WP6 — Causal-audit CI gate**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `security`, `phase-2`.
- Title: `[PIR][WP6] CI gate: controlled-replacement causal audit (arXiv:2607.26773) required before latent-channel merge`
- Body — Goal: require every PR touching `latentmesh-align`-consuming code or the quarantine module (WP7) to pass a controlled-replacement causal audit before merge, modeled on the July 2026 causal-audit paper's methodology. Acceptance criteria: CI blocks merge on audit failure; audit report is witness-logged. Dependencies: WP5.

**WP7 — Anomaly quarantine (net-new)**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `security`, `adr`, `phase-2`.
- Title: `[PIR][WP7] Build latent-channel anomaly quarantine (net-new — "LATTE" has no verified prior art)`
- Body — Goal: build quarantine for anomalous latent-channel updates as an original contribution, combining arXiv:2606.28958's HMAC-manifest integrity approach with `rvm-witness` provenance — explicitly document in the ADR that this is not an implementation of a paper called "LATTE" (none was found to exist). Acceptance criteria: injected-anomaly test suite shows tampered/anomalous latent payloads are quarantined before reaching a receiving agent. Dependencies: WP5, WP6.

**WP8 — Witness-crate convergence (rvm ↔ autogenous)**
- Repo: `ruvnet/rvm` (primary, canonical implementation) + linked issue in `ruvnet/autogenous` (consumer-side migration). Labels: `pir`, `security`, `cross-repo`, `phase-2`.
- Title (rvm): `[PIR][WP8] Confirm rvm-witness as canonical; document external-consumer API for autogenous convergence`
- Title (autogenous): `[PIR][WP8] Migrate autogenous's witness crate to depend on rvm-witness instead of a parallel implementation`
- Body — Goal: resolve the duplicated cryptographic-provenance implementation — `rvm-witness` becomes canonical (more hypervisor-integrated; `autogenous` is self-labeled "research prototype"), and `autogenous`'s `witness` crate is refactored to depend on it. Acceptance criteria: `autogenous`'s witness records are verifiably interchangeable with `rvm-witness` output; both maintainer teams sign off. Dependencies: WP1.

**WP9 — SHAPER-pattern skill/harness evolution loop**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `adr`, `phase-3`.
- Title: `[PIR][WP9] Frozen-weight skill/harness evolution loop via @metaharness/darwin (SHAPER pattern, arXiv:2608.11350)`
- Body — Goal: implement the physical-intelligence evolution loop with foundation weights frozen throughout (planner and optimizer are the same frozen model; only skills/context/harness evolve via Darwin's mutation surfaces at `harness/src/darwin.ts`, `examples/mragent` `scorePolicy`, `crates/sona/src/darwin_guard.rs`). Acceptance criteria: a CI check proves no promotion-reachable code path can write to foundation-model weight files; VLABench/ESI-Bench-style eval shows skill improvement without weight changes. Dependencies: WP0b, WP2.

**WP10 — WorldCycle verification stage**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `phase-3`.
- Title: `[PIR][WP10] Reversible-action verification stage (WorldCycle pattern, arXiv:2608.04964)`
- Body — Goal: add reversible-action-sequence verification between "observe consequences" and Dream Machine evaluation in the physical loop, targeting the paper's reported 44% long-horizon drift reduction and ~4x composite-action accuracy as the internal acceptance bar. Acceptance criteria: CycleBench-equivalent internal benchmark shows the target improvement over a no-WorldCycle baseline. Dependencies: WP9.

**WP11 — Constitutional capability-expansion gate**
- Repo: `ruvnet/autogenous` (primary) + linked integration issue in `ruvnet/ruvector`. Labels: `pir`, `security`, `adr`, `cross-repo`, `phase-3`.
- Title (autogenous): `[PIR][WP11] Expose constitution/admission-gate API for external capability-expansion approval`
- Title (ruvector): `[PIR][WP11] Integrate autogenous's constitutional gate for capability-expanding mutations`
- Body — Goal: every mutation expanding an agent's capability set (new tool access, new physical action class, new communication peer) requires explicit constitutional approval logged to the witness chain, distinct from ordinary behavioral-mutation promotion (WP2). Acceptance criteria: acceptance harness (WP12) can query "zero unapproved capability expansion" directly against this gate's log. Dependencies: WP8.

**WP12 — 30-day acceptance harness**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `phase-3`.
- Title: `[PIR][WP12] Build the 30-day continuous-run acceptance test harness`
- Body — Goal: implement the daily-cycle harness described in this plan's acceptance-test section — frozen-weight hash verification, per-task metric logging, statistical day-30-vs-day-1 comparison (paired bootstrap, pre-registered significance threshold), full witness-chain mutation provenance query, capability-table diff for zero-unapproved-expansion, and an injected-failure rollback test via autogenous's promotion/rollback controller. Acceptance criteria: a full 30-day dry run completes and produces a pass/fail report against every acceptance criterion in the brief. Dependencies: WP2, WP4, WP9, WP10, WP11.

**WP13 — ruvLLM KV-cache cross-model migration (fast-follow)**
- Repo: `ruvnet/ruvector`. Labels: `pir`, `fast-follow`.
- Title: `[PIR][WP13] Implement closed-form KV-cache cross-model migration in ruvllm (arXiv:2608.03893)`
- Body — Goal: implement the paper's closed-form linear KV-cache mapper for same-family model migration in `crates/ruvllm` (`kv_cache.rs`, `paged_attention.rs`, `serving/kv_cache_manager.rs`), with the nonlinear MLP fallback for degrading pairs, plus a routing gate that predicts transfer quality before migrating. Acceptance criteria: reproduces the paper's reported 2.7-25x speedup over re-prefill on an internal same-family model pair; routing gate correctly refuses/downgrades for a known-degrading pair. Dependencies: none — independent parallel track, can start immediately.

**WP14 — RuView pose-semantic embedding + universal CSI vocabulary (stretch)**
- Repo: `ruvnet/RuView` (primary) + linked companion issue in `ruvnet/ruvector` (for `ruvector-perception` validation). Labels: `pir`, `stretch`.
- Title (RuView): `[PIR][WP14] Real-CSI validation + pose-semantic embedding pipeline (pick up ADR-178 gaps C & D)`
- Title (ruvector): `[PIR][WP14] Validate ruvector-perception (3.8K LOC) against real CSI captures, not synthetic-only`
- Body — Goal: close ADR-178's two still-open gaps — gap C (CSI bridge I/Q → pose-semantics conversion) and gap D (mcp-brain-server-side cluster consumer) — and validate `ruvector-perception` against real, not synthetic, CSI data. If time/budget allows, extend into the universal/heterogeneous-chipset CSI vocabulary research bet (evidence item #8 — no prior art exists anywhere, first-party research). Acceptance criteria: `ruvector-perception` achieves a documented accuracy baseline on real CSI captures (not just synthetic); gap C/D are closed per ADR-178's own definition of done. Dependencies: none — independent, recommend funding separately from the WP0-WP12 critical path.

---
