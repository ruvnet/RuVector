# Wave 3 Program Plan — Four Aug 17–19, 2026 Papers

Status: draft for ADR seeding, swarm work-package assignment, and GitHub
issue filing. Compiled 2026-08-21. Depends on `08-wave3-evidence-review.md`
(evidence grades and artifact availability) and, transitively, on `01-07`
and the merged ADR-305–321/323 (Wave 1 + Wave 2 complete on `origin/main`,
confirmed via `git ls-tree` and `gh pr list` — see `08`'s repo-state check).
**No Wave-3 issues are filed by this document** — same convention as Waves 1
and 2: filing happens only after the coordinator approves this plan.

## How this wave differs from Waves 1 and 2

Wave 1 added six bounded contexts for genuinely new mechanisms. Wave 2
extended six already-defined contexts with no new context needed. **Wave 3
is mixed**: three of the four papers (SPADE, D²ACCI, DeAR) extend contexts
that already exist (World Verification / Physical Skill Evolution,
Persistent Memory Governance, Latent Communication Fabric respectively);
the fourth (Zetta) has **no existing context to extend**, because — per
`08-wave3-evidence-review.md`'s novelty mapping — no repo in the `ruvnet`
org currently operates physical robots or a robot-rollout harness. Zetta is
scoped below as a **stretch item**, sequenced last, the same posture Wave 1
gave the universal-CSI bet (WP14).

**Preprint-reproduction rule, applied uniformly, unchanged from Wave 2**:
every one of these four papers is a **candidate mutation**, run through the
existing Darwin → dream-machine/`research-gate` → proof-gate pipeline
(ADR-306, ADR-282, both merged) like any other proposed change. Paper
numbers are hypotheses about what might work; **our bar is our own
research-gate delta, independently recomputed, never the paper's own
reported figure.** This applies even to SPADE and Zetta despite both having
confirmed-live reference implementations — an available repo changes how
much code needs writing from scratch, it does not change what counts as
evidence of a promotable improvement.

**REQUIRED invariant carried from ruv's brief, binding on ADR-324
specifically and on every future mutation-candidate-generation mechanism
this program adopts**: every generated environment, skill, or harness
mutation must trace to external evidence — repository tests, real observed
failures, physical-world observations, or independently published
benchmarks — and must never be validated solely by a closed loop where the
same system that generated the artifact is also the sole judge of its
value. This is not a new invention for this program: it is SPADE's own
ablation finding (external corpus grounding is "critical to success," per
the paper's abstract) generalized into a standing design rule, because the
failure mode it prevents — a self-play loop drifting away from any
real-world distribution while still reporting internally-measured gains —
is exactly the risk this program's own invariant 5 ("every promoted
mutation must outperform its parent") is not, by itself, sufficient to
catch. A mutation can outperform its parent *on a self-generated benchmark*
and still be worthless against reality. `research-gate`'s existing
hard-veto mechanism (`crates/ruvector-sota-bench/harness/src/statistics.ts`,
adopted by ADR-306) is the natural enforcement point: add an external-
grounding veto alongside the existing paired-bootstrap statistical veto,
so that no self-play-generated environment can reach promotion without a
traceable link to something outside the generating loop.

**Wave-3 acceptance test — provided verbatim by ruv, 2026-08-20, carried
here unmodified rather than reconstructed** (contrast with Wave 2, where no
verbatim wave-specific test could be located and the plan had to say so
explicitly):

> "freeze the underlying model for 30 days; allow only harness, memory,
> topology, tools, and generated environments to evolve; hold out 20% of
> real tasks the evolution system never sees; if held-out success improves
> at least 10% while capability violations remain zero and every promoted
> change has reproducible provenance, you have demonstrated genuine
> substrate-level learning rather than benchmark adaptation."

This composes cleanly with the existing WP12 30-day acceptance harness
(`03-program-plan.md`) and ADR-315's capability-expansion gate ("capability
violations remain zero" is exactly ADR-315's zero-unapproved-capability-
expansion requirement, already queryable via its witness-anchored approval
log) — it adds two concrete new requirements on top of what's already
built: (a) a genuinely **held-out** 20% task split the evolution system
never trains or self-evaluates against, and (b) a **≥10% held-out-success**
threshold, measured the same paired-bootstrap way `research-gate` already
measures everything else. No new acceptance-harness infrastructure is
needed; WP12's existing harness is extended with the held-out split and the
10% threshold as its Wave-3-specific pass condition.

**ADR numbering**: per `08-wave3-evidence-review.md`'s direct filesystem
check of `origin/main` (not the stale `INDEX.md` header, which still reads
"317" after PR #868 and the WP15–20 PRs went in without a re-generation),
the highest real ADR file is `ADR-323-governed-pipeline-shard-placement.md`
and ADR-322 does not exist (deliberately skipped, per Wave 2). **The next
available number is ADR-324.** This plan assigns ADR-324 through ADR-327
and confirms each against the live filename list at kickoff, exactly as
Waves 1 and 2 required — treat the numbers below as placeholders subject to
that check. Re-run `node scripts/adr-index.mjs` as part of whichever PR
lands the first Wave-3 ADR, so the generated header stops lagging reality.

**WP numbering**: continues from the highest existing number. WP0a/0b,
WP1–WP20 already exist (WP20 = pipeline-shard placement, the highest
Wave-2 number). This wave adds **WP21 through WP24**.

---

## Priority order (ruv's stated sequence, carried verbatim)

> SPADE → D²ACCI → DeAR → Zetta

Mapped to phases below: **Phase W3-1** = WP21 (SPADE, highest priority,
carries the REQUIRED external-grounding invariant — do not sequence
anything ahead of getting that veto correct). **Phase W3-2** = WP22
(D²ACCI). **Phase W3-3** = WP23 (DeAR). **Phase W3-4** = WP24 (Zetta,
stretch, no hard dependency on the others, can start whenever resourcing
allows the same way Wave 1's WP14 and Wave 2's WP20 were both explicitly
scoped as independent parallel tracks — except WP24 additionally has no
existing bounded context to attach to, which the ADR below addresses by
scoping it as an explicit stretch context, not by silently expanding an
existing one).

---

## ADR list and mapping

| # | Paper | Proposed ADR | Extends (merged) | Genuinely new |
|---|---|---|---|---|
| 1 | SPADE | **ADR-324** — Self-play synthetic-environment generation as a Darwin mutation-candidate source, with a REQUIRED external-grounding veto | ADR-306 (dream-machine/research-gate promotion gate), ADR-313 (SHAPER frozen-weight loop, WP9), ADR-321 (SkillForge-pattern candidate source, WP19 — same "one more candidate-generation strategy feeding the existing pipeline" shape) | An LLM-authored, executable Gym-style (`reset()`/`step()`) training environment, generated by the same frozen model under a regret signal, as a new type of mutation candidate — **paired with a new hard veto in `research-gate` requiring every generated environment to trace to external evidence** (repo tests, real incident logs, or an independently published benchmark), never validated solely by the same closed self-play loop that generated it |
| 2 | D²ACCI | **ADR-325** — Stage-level diagnostic gate and DCR observability metric for the memory pipeline | ADR-307 (three-level persistent memory, WP3/WP4), ADR-320 (MemFuse-pattern atomic observation + causal graph, WP18), ADR-282/ADR-306 (`research-gate` paired-bootstrap promotion gate, which this ADR sits *underneath*, not replaces) | A fault-localization layer that identifies *which* of the four memory-pipeline stages (ingestion, retrieval, filtering, generation — the abstract's own enumeration; "supplement extraction" is an ablated intervention, not a fifth stage) changed behavior when a memory-system mutation's end-to-end score moves — something `research-gate`'s existing outcome-level statistics do not do today — plus a graded DCR (a metric the abstract describes only as "a graded observability metric," with no expansion given) attached to every memory-mutation's promotion record |
| 3 | DeAR | **ADR-326** — Decentralized capability grounding and shared thought-map navigation for the Latent Communication Fabric, with dead-end-triggered topology update | ADR-309/310/311 (LatentMesh greenfield fabric, causal-attribution gate, anomaly quarantine) | Query-dependent peer specialization with no central role-assigner, a shared thought-map data structure agents navigate directly rather than exchanging only point-to-point messages, and a topology-mutation trigger fired specifically on dead-end detection (continue, not restart) — **the continue-vs-restart behavior is this program's working interpretation of the abstract's "topology update for adaptive error correction," not a verified quotation** (see `08-wave3-evidence-review.md` §3); confirm against the full paper (still not available today, see artifact status) or the eventual code release before this ADR's Decision section treats it as settled |
| 4 | Zetta | **ADR-327** — Frozen-policy runtime critic/recovery evolution for physical embodied harnesses (stretch context — no existing `ruvnet` home) | Nothing merged — see Context below | The entirety of Zetta's mechanism (three-timescale critic/recovery evolution over a frozen base policy, validated against held-out seeds before promotion) is new relative to everything in `ruvnet` today, **because no repo in the org currently operates physical robots or a robot-simulation rollout harness** (RVM = agentic runtime/hosting, not robotics; RuView = RF sensing, not actuation) — this ADR is scoped as a bounded-context proposal, not an extension, and is explicitly a stretch bet the same way Wave 1's WP14 (universal CSI vocabulary) was |

None of the four maps onto a context that needs re-litigating from scratch
except Zetta, which needs one created. Bounded-context assignment: **World
Verification / Physical Skill Evolution** (324, extending 313's context),
**Persistent Memory Governance** (325, extending 307/320's context),
**Latent Communication Fabric** (326, extending 309/310/311's context),
**Physical Embodiment** (327 — new, stretch-only context, no merged ADR
occupies it yet).

---

## Work packages (sized for 6–8 agent swarm teams, per project anti-drift config)

| # | Package | Extends | Team composition | Depends on |
|---|---|---|---|---|
| WP21 | Implement an Environment-Designer mutation-candidate source in Darwin's proposal surface, adapting the released implementation ([github.com/spade-rl/spade](https://github.com/spade-rl/spade), confirmed live) rather than rebuilding from the paper alone; add the REQUIRED external-grounding hard veto to `research-gate` before this source is allowed to reach the promotion gate at all (ADR-324) | WP9, WP2, WP19 | coordinator, security-architect, system-architect, coder ×2, tester | WP9 (SHAPER loop), WP2 (dream-machine/research-gate promotion), WP19 (SkillForge candidate-source pattern to mirror) |
| WP22 | Build a stage-level diagnostic gate over the four memory-pipeline stages (ingestion, retrieval, filtering, generation) in `crates/ruvector-agent-memory`, plus a DCR-style observability metric attached to every memory-mutation promotion record (no upstream code — build from paper description); wire it as a layer underneath `research-gate`'s existing outcome-level statistics, not a replacement for them (ADR-325) | WP3, WP4, WP18 | coordinator, memory-specialist, backend-dev, tester | WP4 (TARL ledger must exist — `ledger.rs`, `observation.rs`), WP18 (MemFuse causal-graph fusion — `fusion.rs`) |
| WP23 | Add decentralized capability grounding, shared thought-map data structure, and dead-end-triggered topology mutation to the Latent Communication Fabric (no upstream code — build from paper description, and explicitly re-verify the continue-vs-restart behavioral detail against the full paper or a future code release before finalizing, per the grade-B+ caveat in `08-wave3-evidence-review.md`) (ADR-326) | WP5, WP6, WP7 | coordinator, system-architect, coder ×2, tester | WP5 (greenfield `latentmesh-*` crates), WP6 (causal-audit gate — topology changes must remain attributable), WP7 (anomaly quarantine — a misbehaving peer must not be able to trigger spurious topology churn) |
| WP24 | *(Stretch, sequenced last, no hard blocking dependency)* Stand up a minimal physical/simulated embodied-harness surface (LIBERO-Pro/RoboCasa-class rollout infrastructure) and a frozen-policy critic/recovery evolution loop, adapting the released implementation ([github.com/air-embodied-brain/Zetta-Embodiment](https://github.com/air-embodied-brain/Zetta-Embodiment), confirmed live) as the starting reference rather than building the rollout infrastructure from nothing (ADR-327) | Nothing merged (new context) | coordinator, system-architect, backend-dev, coder ×2, tester | None hard; soft-coordinates with RuView if/when a real-world (not just simulated) embodiment path is pursued, the same "sensing side only, no actuation" boundary `08`'s novelty mapping identified |

Use `hierarchical` topology, `max-agents 8`, `specialized` strategy per
project config, same as every prior PIR work package.

---

## Repo assignments

Three of four stay inside `ruvnet/ruvector`; DeAR's topology/thought-map
mechanism is scoped primarily to `LatentMesh` (a greenfield crate context
this program already coordinates cross-repo per WP5/ADR-309); Zetta has no
natural `ruvector` home at all and is scoped as either a new crate/example
inside `ruvector` or, if the coordinator judges the physical-robotics scope
too far outside this repo's remit, a new sibling repo epic mirroring how
Wave 1 treated LatentMesh/rvm/autogenous/RuView as separate epics — **this
plan does not decide that placement question**, it only flags that it is
open, unlike WP21–23 which have unambiguous `ruvector`-internal homes.

- **WP21** (ADR-324): `crates/ruvector-sota-bench/harness/src/{darwin.ts,
  dreamMachine.ts, shaperLoop.ts, skillForge.ts, statistics.ts}` — new
  `environmentDesigner.ts` module alongside the existing candidate-source
  files, plus a new veto function in `statistics.ts`'s hard-veto set
- **WP22** (ADR-325): `crates/ruvector-agent-memory/src/{ledger.rs,
  observation.rs, fusion.rs, scoring.rs}`, `scripts/research-gate/
  research_gate.py` (new diagnostic-trace consumer, layered under the
  existing paired-bootstrap statistics)
- **WP23** (ADR-326): `crates/rvAgent/latentmesh-*` (per ADR-305's WP5
  greenfield crates — likely a new `latentmesh-thoughtmap` or extension of
  `latentmesh-align`), coordinates with `latentmesh-gate`'s causal.rs per
  the existing ADR-310 CI gate
- **WP24** (ADR-327): placement open — candidate: a new
  `crates/ruvector-embodiment` (or sibling-repo epic; see note above);
  soft-coordinates with `ruvnet/RuView` only on sensing/state-estimation,
  never on actuation

---

## Security / validation gates (in addition to the ones already governing WP1–WP20)

- **WP21 is the highest-security-sensitivity item in this wave, and higher
  than any single Wave-2 item**: it is the first PIR mechanism whose
  mutation candidates are not just *proposed* by an untrusted generator
  (already true of every Darwin candidate source) but are entire
  **executable environments** — code that defines state transitions,
  rewards, and verification logic, which the Reasoning Agent then trains
  against. This is strictly more capability-sensitive than SkillForge's
  synthetic *issues* (WP19) or TRUSS's shadow-executed *skills* (WP17): a
  generated environment that is subtly wrong doesn't just produce a bad
  skill, it can produce a bad **measurement of what "better" means**,
  corrupting every downstream promotion decision that trusts it. Apply
  ADR-315's constitutional gate to any generated environment that would be
  used to certify a promotion, not only to environments used for training,
  and require the REQUIRED external-grounding veto to fire *before* Dream
  Machine's evaluation stage ever runs, not after.
- **No wholesale adoption without a measured delta**: per the framing
  above, none of WP21–24's mechanisms may be merged on the strength of the
  source papers' own reported numbers alone — `research-gate`'s independent
  paired-bootstrap recomputation must show a delta over the pre-WP baseline
  before proof-gate promotion, the same rule Wave 2 applied uniformly.
- **Name-collision citation discipline**: per `08-wave3-evidence-review.md`,
  every reference to SPADE in code comments, ADRs, or issue text must spell
  out the arXiv ID on first use — SPADE collides with a live, long-
  established, same-domain multi-agent Python framework of the identical
  name. This is the same severity of discipline Wave 2 required for MemFuse
  and TRUSS, and higher-severity than Wave 1's TARL/LiveMem flags.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  WP21 and WP23 land, given both touch mutation-proposal/topology-mutation
  surfaces analogous to Wave 2's WP17/WP19 gate requirement.

---

## Top risks

1. **SPADE's closed-epistemic-loop risk is this wave's most important
   structural risk, and it is evidenced, not hypothetical.** The source
   paper's own ablation found external corpus grounding "critical to
   success" — meaning even the paper's authors observed the ungrounded
   variant underperforming. A self-play loop where the same model designs
   environments and is scored inside them can silently reward-hack its own
   evaluation without the REQUIRED external-grounding veto (ADR-324,
   WP21). This is not a generic "be careful" risk; it is the single
   highest-priority engineering item in this wave, ahead of any feature
   work in WP21 itself.
2. **The preprint-reproduction risk, unchanged from Wave 2**: every one of
   these four papers reports numbers on its own benchmark, evaluated by
   its own authors (SPADE and Zetta on their own repos' eval harnesses;
   D²ACCI and DeAR with no repo to check against at all). None of those
   numbers may be cited as the acceptance bar for a promoted PIR change —
   only `research-gate`'s independently recomputed delta counts. The risk
   is schedule pressure causing WP21/24 (which have a working reference
   implementation to point at) to shortcut this because "the paper's repo
   already shows it works" — the mitigation is unchanged: ADR-306's
   existing gate, and WP21–24 must actually route through it.
3. **Two of four papers have no confirmed upstream code (D²ACCI, DeAR) —
   both are top-3 priority.** This mirrors Wave 2's TRUSS/StagedWorkspace
   risk exactly: WP22 and WP23 are reproduction-from-description efforts, a
   materially larger scope than WP21/24's "adapt a released repo" pattern.
   Size them accordingly.
4. **DeAR's dead-end topology behavior (continue vs. restart) is an
   unverified interpretation, not a confirmed paper claim.** If the eventual
   full-text read or code release contradicts this program's working
   interpretation, ADR-326's Decision section needs revision before WP23
   locks in a specific behavioral contract. Do not let WP23's
   implementation harden around an assumption `08-wave3-evidence-review.md`
   flagged as unconfirmed.
5. **Zetta has no existing bounded context, unlike every other paper in
   Waves 1–3.** WP24 is, structurally, the same kind of first-party bet
   Wave 1's WP14 (universal CSI) was — except WP14 at least had RuView as a
   plausible eventual home (RF sensing is RuView's actual domain); Zetta's
   embodied-harness mechanism has **no** plausible existing home anywhere
   in `ruvnet` (RVM hosts agentic runtimes generically, not robots; RuView
   senses, it does not actuate). Do not schedule WP24 as if a bounded
   context already exists to receive it — ADR-327 is proposing one, and
   the repo-placement question (new crate in `ruvector` vs. new sibling
   repo) is explicitly left open above for the coordinator to decide.
6. **SPADE's name collision is the most severe of any wave so far** and the
   easiest one for a hurried commit message or status update to get wrong,
   because "SPADE" alone, with no arXiv ID, reads as a completely
   reasonable reference to the pre-existing, decades-old multi-agent
   framework to anyone who has taken a multi-agent-systems course. The
   mitigation (spell out the arXiv ID every time) is cheap; the failure
   mode if skipped is a genuinely confusing document, not just an
   unattributed citation.

---

## GitHub issue breakdown (ready-to-file specs only — do not create them)

Per the master tracking issue's stated convention, Wave-3 issues are filed
only after the coordinator approves this plan. Suggested labels: `pir`,
`wave-3`, `adr`, plus `security` for WP21 (mirrors WP17's label from Wave
2), `stretch` for WP24 (mirrors WP14's label from Wave 1).

| WP | Title | Depends on |
|---|---|---|
| WP21 | `[PIR][WP21] SPADE-pattern self-play environment generation + REQUIRED external-grounding veto (ADR-324)` | WP9, WP2, WP19 |
| WP22 | `[PIR][WP22] D²ACCI-pattern stage-level memory diagnostic gate + DCR metric (ADR-325)` | WP4, WP18 |
| WP23 | `[PIR][WP23] DeAR-pattern decentralized capability grounding + thought-map navigation (ADR-326)` | WP5, WP6, WP7 |
| WP24 | `[PIR][WP24] Zetta-pattern frozen-policy embodied critic/recovery evolution (ADR-327, stretch, new context)` | none (soft: RuView, sensing only) |

Each issue body should link `08-wave3-evidence-review.md` for the evidence
grade and artifact-availability status, and this document for the WP/ADR
mapping — same pattern Waves 1 and 2 used.
