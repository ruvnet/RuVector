# Wave 4 Program Plan — Six Aug 20, 2026 Papers + NVIDIA Security-Stack Items

Status: approved for immediate execution (see "Approval status" below —
this wave differs procedurally from Waves 1–3). Compiled 2026-08-22.
Depends on `10-wave4-evidence-review.md` (evidence grades and artifact
availability) and, transitively, on `01`–`09` and the merged ADR-305–321 /
ADR-323–327 (Waves 1–3 complete on `origin/main`, confirmed via
`git ls-tree` and `gh pr list` — see `10`'s repo-state check).

## Approval status — how this wave differs procedurally

Waves 1–3 held issue-filing until the coordinator approved the drafted
plan. **Wave 4 arrives pre-approved**: ruv's 2026-08-22 briefing supplies
the priority order, per-item difficulty/value scoring, concrete
implementation sketches, a verbatim seven-day acceptance test, and the
directive "update upstream as needed, build, test, benchmark, secure,
publish, merge." That is coordinator approval in advance; Wave-4 issues
are therefore filed together with this plan rather than after a second
approval round. Everything else — the preprint-reproduction rule, the
research-gate promotion bar, the security gates — is unchanged.

## How this wave differs from Waves 1–3

Wave 1 added six bounded contexts; Wave 2 extended six existing ones;
Wave 3 was mixed. **Wave 4 is again mixed, with two firsts**:

1. **It is the first wave with a benchmark whose entire purpose is to
   measure this program's own core claim** — AI4AI-Bench
   (arXiv:2608.20318) scores whether an agent genuinely rewrites a
   learning algorithm rather than tuning parameters, which is precisely
   the "genuine substrate improvement vs. benchmark adaptation"
   distinction ruv's Wave-3 acceptance test drew. WP25 wires it in as an
   externally-authored, externally-evaluated benchmark surface — external
   grounding by construction, exactly what ADR-324's REQUIRED invariant
   demands of any evaluation the evolution loop trusts.
2. **It is the first wave with an industry item rather than only papers**:
   NVIDIA's Aug 21 security-stack essay and OpenShell (both verified real,
   see `10` §NVIDIA) externally validate the program's frozen-model /
   evolving-harness / enforcement-below-the-mutable-layer posture. WP30
   responds with differentiation, not imitation, and is a cross-repo
   posture ADR (RVM), not a ruvector implementation item.

Context mapping: WP25 extends the Darwin/MetaHarness benchmark surface
(ADR-306/313/324 context); WP27 extends Persistent Memory Governance
(ADR-307/320/325 context); WP28 extends routing/inference-economics (new
mechanism in an existing crate); WP26 creates a new schema-resource-cache
surface (no prior tool-schema cache exists anywhere in the repo — checked,
see `10`); WP29 (RF modality router) is a stretch bet adjacent to RuView,
dataset-only artifact; WP30 is cross-repo (RVM).

**Preprint-reproduction rule, applied uniformly, unchanged from Waves
2–3**: every one of these six papers is a **candidate mutation**, run
through the existing Darwin → dream-machine/`research-gate` → proof-gate
pipeline (ADR-306, ADR-282, both merged) like any other proposed change.
Paper numbers are hypotheses; **our bar is our own research-gate delta,
independently recomputed, never the paper's own reported figure.** This
applies with special force in Wave 4 because two of the six papers ship
live author repos (AI4AI-Bench, ReCache) — an available repo changes how
much code needs writing, not what counts as evidence.

**Invariant 7, carried unchanged and binding on every WP below** (adopted
Wave 2, from ruflo ADR-322B): a proposer produces untrusted candidates
only; it cannot issue promotion decisions or mutate active policy.

**Wave-4 acceptance test — provided verbatim by ruv, 2026-08-22, carried
here unmodified**:

> "within seven days, MetaHarness should run at least one AI4AI task,
> Ruflo should demonstrate cached reusable tool schemas with at least 2×
> lower time to first token, and RuVector should correctly treat ten agent
> memories derived from one source as one effective evidence lineage
> rather than ten independent votes. If those three pass, this research
> has produced measurable substrate improvements rather than architectural
> speculation."

The three clauses map 1:1 onto WP25, WP26, and WP27 — which is why those
three share the wave's first phase and everything else is sequenced after.
Two honesty constraints on measuring it, applying this program's standing
metric-integrity discipline: (a) the "2× lower TTFT" clause must be
measured against a real uncached baseline on the same assembly path, with
downgrade-only accounting (a cache miss or partial hit may never be
scored as a hit); (b) the "one effective lineage" clause is
downgrade-only by design — arbitration may only ever *reduce* effective
evidence relative to the naive count, never inflate it (see ADR-330).

**ADR numbering**: per `10-wave4-evidence-review.md`'s repo-state check,
INDEX.md's generated header ("next available: 328") matches a direct
recount of the live tree. **This wave allocates ADR-328 through ADR-333**,
confirmed free at kickoff. ADR-322 remains permanently skipped (it names
ruflo's ADR-322 family). Re-run `node scripts/adr-index.mjs` in the PR
that lands these ADRs.

**WP numbering**: continues from WP24 (Wave 3's highest). This wave adds
**WP25 through WP30**.

---

## Priority order (ruv's stated sequence, carried verbatim)

> AI4AI Bench first, ReCache plus cache aware routing second, CAMA third,
> Value of Information Routing fourth, RF modality routing fifth, and RVM
> plus OpenShell interoperability sixth.

Mapped to phases: **Phase W4-1** = WP25 + WP26 + WP27 (the three
acceptance-test items — run as parallel tracks; they share no code paths).
**Phase W4-2** = WP28 (VoI routing). **Phase W4-3** = WP29 (RF modality
router, stretch, ADR-only this wave). **Phase W4-4** = WP30 (RVM ↔
OpenShell posture, cross-repo, ADR-only in this repo; RVM-side work
requires maintainer review — USER ACTION for any merge there).

---

## ADR list and mapping

| # | Item | Proposed ADR | Extends (merged) | Genuinely new |
|---|---|---|---|---|
| 1 | AI4AI-Bench | **ADR-328** — AI4AI-Bench (arXiv:2608.20318) recursive-improvement benchmark adapter for MetaHarness | ADR-306 (dream-machine/research-gate), ADR-313 (SHAPER), ADR-324 (external-grounding invariant — this benchmark is external grounding by construction) | An externally-authored, externally-evaluated benchmark whose scoring specifically rewards rewriting the learning algorithm itself (0.1 = shipped algorithm, 1.0 = optimum), wired in through the harness's existing constructor-injected benchmark-adapter seam without touching the research-artifact-emission restriction |
| 2 | ReCache | **ADR-329** — Content-addressed schema-resource cache (ReCache-pattern, arXiv:2608.19662) | ADR-301 (semantic query cache — prior art for the caching mechanics, different object), ADR-320 (content-addressed identity discipline) | Stable content-addressed resource identity for every tool/skill/agent-role/policy/MCP schema, compiled once and assembled from cached position-independent blocks — no tool-schema/prefix cache exists anywhere in the repo today (checked). Context-assembly scope now; ruvllm KV-block reuse is follow-up. Carries CacheRoute (arXiv:2608.19677) shadow-replay caveat |
| 3 | CAMA | **ADR-330** — CAMA-pattern (arXiv:2608.19701) correlation-aware memory arbitration | ADR-307 (memory tiers), ADR-320 (AtomicObservation + causal graph — the provenance substrate arbitration clusters over), ADR-325 (stage-level gate — arbitration is a retrieval/filtering-stage mechanism under its taxonomy) | Effective-independent-source counting: retrieved memories clustered by causal ancestry before voting; confidence = independent evidence groups × source reliability × freshness, strictly downgrade-only vs. the naive count; ten memories from one root = one lineage |
| 4 | Pandora Router | **ADR-331** — Value-of-information cost-aware routing (Pandora-pattern, arXiv:2608.20316) | Nothing merged in-crate (tiny-dancer has uncertainty quantification but zero cost model — checked); harness `routeModel` price-aware routing is adjacent prior art | The buy-information decision rule (purchase a better estimate iff expected quality gain × value of success > estimator cost + latency) as one reusable primitive, closed-form under a Gaussian signal model, landing in `ruvector-tiny-dancer-core`; corroborated by the independent judge-panels result (arXiv:2608.19802) |
| 5 | RF comparison | **ADR-332** — RF sensing modality router (stretch; arXiv:2608.20322, under review) | Nothing merged (RuView is a sibling repo; `ruvector-mmwave` exists as a sensing-adjacent crate) | Treat sensing modality as a routed decision, not a fixed architectural choice: Wi-Fi for ubiquitous presence/coarse behavior, IR-UWB for fine activity at lowest cost, FMCW for unseen-environment robustness, with a common representation above all three. ADR-only this wave |
| 6 | NVIDIA stack/OpenShell | **ADR-333** — RVM semantic-authority layer above OpenShell-class secure runtimes (cross-repo posture) | ADR-312 (witness/anchoring contract), ADR-315 (capability-expansion gate) | Differentiation, not imitation: RVM moves one layer up in semantic authority (signed agent identity, signed memory, mutation rights, transferable capabilities, provenance chains, revocation, rollback, portable execution receipts) and gains an OpenShell interop adapter rather than duplicating kernel isolation. ADR-only in this repo; RVM-side merge is USER ACTION |

Bounded-context assignment: **Evolution Benchmarking** (328, extending
306/313/324's context), **Agent Context Economics** (329 — new context),
**Persistent Memory Governance** (330, extending 307/320/325's context),
**Inference Economics** (331 — new mechanism, existing crate), **RF
Sensing** (332, stretch, RuView-adjacent), **Secure Runtime Interop**
(333, cross-repo).

---

## Work packages (sized for 6–8 agent swarm teams, per project anti-drift config)

| # | Package | Extends | Team composition | Depends on |
|---|---|---|---|---|
| WP25 | Add an AI4AI-Bench (arXiv:2608.20318) adapter behind the harness's existing constructor-injected benchmark seam (`RuvectorFlywheelOptions.benchmark` / `RuvectorGepaOptions.benchmark` / `runControlledBenchmark`'s runner param), adapting the released Apache-2.0 suite ([github.com/Einsia/AI4AI-Bench](https://github.com/Einsia/AI4AI-Bench)) — task-suite interface + evaluator contract + smoke-scale local execution; the `flywheel.ts` research-artifact-emission restriction (native-runner-only) stays intact (ADR-328) | WP2, WP9, WP21 | coordinator, system-architect, coder ×2, tester | WP2 (promotion pipeline), WP21 (external-grounding veto — AI4AI tasks satisfy it by construction and must be registered as such) |
| WP26 | Build a content-addressed schema-resource cache: stable SHA-256 resource identity for tool/skill/role/policy/MCP schemas, compile-once reusable blocks, context assembly from cached blocks, hit/miss accounting, and a TTFT-proxy benchmark with an uncached baseline (ADR-329). Pattern-reimplementation from the paper — **no code copying: `EIT-NLP/ReCache` has no LICENSE file** | ADR-301 prior art | coordinator, backend-dev, coder, perf-analyzer, tester | none hard (new surface in `crates/mcp-gate`) |
| WP27 | Implement correlation-aware arbitration in `crates/ruvector-agent-memory`: cluster retrieved memories by causal ancestry (via `fusion.rs`'s provenance resolution), count effective independent evidence groups, score confidence = groups × source reliability × freshness, downgrade-only vs. naive count, non-finite inputs rejected at the choke point (ADR-330). From-scratch — no CAMA code exists | WP4, WP18, WP22 | coordinator, memory-specialist, backend-dev, tester | WP4 (`ledger.rs`/`observation.rs`), WP18 (`fusion.rs`), WP22 (`diagnostic.rs` — arbitration registers as a retrieval/filtering-stage policy under its gate) |
| WP28 | Add a VoI module to `crates/ruvector-tiny-dancer-core`: closed-form Gaussian value-of-information, the buy-better-estimate decision rule, integration with the existing conformal-uncertainty surface and `RoutingDecision` (ADR-331). Centralized policy only — the paper's own decentralized variant carries a stated negative result and is explicitly out of scope | tiny-dancer existing routing | coordinator, perf-engineer, coder, tester | none hard |
| WP29 | *(Stretch, ADR-only this wave)* RF modality-router posture: corrected sensor attributions, dataset pointer, modality-abstraction decision recorded; implementation deferred pending RuView coordination (ADR-332) | none | adr-architect only | none |
| WP30 | *(Cross-repo, ADR-only in this repo)* RVM semantic-authority posture + OpenShell interop adapter spec (ADR-333); any `ruvnet/rvm` code lands under that repo's maintainer-review requirement — **USER ACTION for merge there** | ADR-312, ADR-315 | adr-architect, security-architect | none in-repo |

Use `hierarchical` topology, `max-agents 8`, `specialized` strategy per
project config, same as every prior PIR work package.

---

## Repo assignments

- **WP25** (ADR-328): `crates/ruvector-sota-bench/harness/src/` — new
  `ai4aiBench.ts` adapter module + tests; no change to `benchmark.ts`'s
  closed lever set or `flywheel.ts`'s research-emission restriction.
- **WP26** (ADR-329): `crates/mcp-gate/` (the repo's schema-serving gate)
  — new schema-resource-cache module; `crates/ruvector-query-cache` is
  mechanics prior art, not the landing site (it caches ANN query results,
  a different object).
- **WP27** (ADR-330): `crates/ruvector-agent-memory/src/` — new
  `arbitration.rs` over `observation.rs`/`fusion.rs`; registers under
  `diagnostic.rs`'s stage taxonomy.
- **WP28** (ADR-331): `crates/ruvector-tiny-dancer-core/src/` — new
  `voi.rs` + wiring into `router.rs`/`uncertainty.rs`.
- **WP29** (ADR-332): docs only this wave.
- **WP30** (ADR-333): docs only in this repo; `ruvnet/rvm` for any code.

---

## Security / validation gates (in addition to those governing WP1–WP24)

- **WP26 is this wave's highest-security-sensitivity item**: a schema
  cache is a new trust surface — a poisoned cached schema block would be
  replayed into every context assembled from it. Two binding requirements,
  both direct applications of Wave-3 lessons: (a) resource identity must
  bind to **resolved content**, never to a self-declared name or hash
  claim (the Wave-3 #887 lesson: hashing self-declared strings is a false
  safety claim); (b) cache-hit accounting is downgrade-only — misses and
  partial hits can never score as hits, so the 2× TTFT claim cannot be
  faked by accounting.
- **WP27's arbitration is downgrade-only by construction**: effective
  evidence ≤ naive count, always; and non-finite confidence/reliability/
  freshness inputs are rejected at the choke point before any comparison,
  per the Wave-3 #888 NaN-bypass lesson (comparison-based gates over
  float inputs silently pass NaN).
- **WP25 executes nothing from the third-party benchmark in-process**:
  the AI4AI suite's tasks and evaluators run only in contained
  subprocesses (the same posture `research-gate` takes toward candidate
  code), and the adapter cannot reach research-artifact emission
  (`flywheel.ts`'s native-runner-only restriction is load-bearing and
  untouched).
- **No wholesale adoption without a measured delta**: none of WP25–WP30's
  mechanisms may be merged on the strength of the source papers' own
  numbers — `research-gate`'s independent recomputation governs, per the
  standing rule.
- **Name-collision citation discipline**: "CacheRoute" must always carry
  arXiv:2608.19677 and the explicit note that `AstraNetLab/CacheRoute` is
  an unrelated system (see `10` §3 — the collision is worse than SPADE's
  because the colliding repo is the top search hit). Never adopt
  "CacheRoute," "ReCache," "CAMA," or "Pandora" as package/crate/module
  names.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan`
  after WP26 and WP27 land (both touch trust/evidence surfaces).

---

## Top risks

1. **The acceptance test's 2× TTFT clause is the easiest to fake and the
   most visible.** Without a real uncached baseline on the identical
   assembly path and downgrade-only hit accounting, a cache can "pass" by
   measurement artifact. WP26's benchmark harness must be built to the
   same metric-integrity standard as HarnessRisk (shadow mapping can only
   worsen the treated run) before any number is reported.
2. **Preprint-reproduction risk, amplified**: AI4AI-Bench and ReCache
   have live repos, which invites "the repo already shows it works."
   Unchanged mitigation: research-gate's recomputed delta is the only
   citable bar. Additionally, ReCache's figures are at Qwen3-1.7B/4B
   scale only, and its metric is Inv-F1, not accuracy.
3. **Three of six papers have no code at all (CacheRoute, CAMA, Pandora)**
   — CAMA moreover reports *zero quantitative results in its abstract*.
   WP27 and WP28 are reproduction-from-description efforts guided by
   mechanism, not by any promised number; size and evaluate accordingly.
4. **ReCache's repo has no LICENSE file.** Until one appears, WP26 is a
   pattern reimplementation from the paper text only — no code copying,
   no vendoring. Flag for re-check before any future adaptation.
5. **AI4AI-Bench's official scoring assumes a 4h budget on one B300 plus
   a 12h from-scratch rerun.** WP25 delivers the adapter, contract tests,
   and smoke-scale local runs; an official-scale scored run is an
   infrastructure decision (USER ACTION), and "beat 0.250" is a program
   target, not a this-wave deliverable. Note the score scale: 0.1 = the
   shipped algorithm, so 0.250 closes under a fifth of the
   ship-to-optimum distance.
6. **CacheRoute's own authors recommend shadow-replay gating before any
   deployment** (two 32B counterexample workloads erase its gain). Any
   future cache-aware placement work adopting its pattern inherits that
   recommendation as a requirement, not a suggestion — recorded in
   ADR-329 so it survives into whichever wave implements placement.

---

## GitHub issue breakdown

Filed with this plan (see "Approval status"). Labels: `pir`, `adr`,
`wave-4`, plus phase labels `phase-w4-1..4`, `security` for WP26,
`stretch` for WP29, `cross-repo` for WP30.

| WP | Title | Depends on |
|---|---|---|
| WP25 | `[PIR][WP25] AI4AI-Bench (arXiv:2608.20318) recursive-improvement benchmark adapter (ADR-328)` | WP2, WP21 |
| WP26 | `[PIR][WP26] ReCache-pattern content-addressed schema-resource cache + TTFT benchmark (ADR-329)` | none |
| WP27 | `[PIR][WP27] CAMA-pattern correlation-aware memory arbitration — effective evidence lineages (ADR-330)` | WP4, WP18, WP22 |
| WP28 | `[PIR][WP28] Pandora-pattern value-of-information cost-aware routing (ADR-331)` | none |
| WP29 | `[PIR][WP29] RF sensing modality router posture (ADR-332, stretch, ADR-only)` | none |
| WP30 | `[PIR][WP30] RVM semantic-authority layer above OpenShell-class runtimes (ADR-333, cross-repo, ADR-only here)` | none in-repo |

Each issue body links `10-wave4-evidence-review.md` for evidence grades
and this document for the WP/ADR mapping — same pattern as prior waves.
