# Wave-3 Evidence Review — Four Aug 17–19, 2026 Papers

Status: draft for ADR seeding. Compiled 2026-08-21 by direct web-verification
against primary sources (arXiv abstracts fetched directly, comments/version
fields checked, and every claimed GitHub repository fetched via the GitHub
API to distinguish "code released" from "code claimed") — the same
discipline `06-wave2-evidence-review.md` applied, and per ADR-305 §6's
fix-history rule: every "no system does X" / "component absent" claim below
was checked at source via `gh`, not inferred from a prior pass's summary.

**Repo-state check, done before anything else, because Wave 1 got this
wrong twice**: this program's Wave-1 and Wave-2 ADRs (ADR-305 through
ADR-321, and ADR-323 — 322 deliberately skipped, see ADR-306/313/315 and
`07-wave2-program-plan.md`) **are confirmed merged on `origin/main`** via
`git ls-tree origin/main -- docs/adr/`. The corresponding work packages
(WP0a/b, WP1–WP4, WP9, WP13, WP15–WP20) are also confirmed merged via `gh pr
list` (PRs #847, #854–878). The local working tree this review runs from
(branch `ci/nextest-heavy-timeout`) does **not** have these files checked
out — a plain `ls docs/adr/` on this branch shows nothing past ADR-304 — but
`origin/main` does. **Do not repeat Wave-1's "component is absent" mistake
by trusting a local directory listing over the actual remote state.**

**Hygiene finding, not previously flagged**: `docs/adr/INDEX.md` on
`origin/main` still reads "Next available ADR number: 317" — it was
generated once, at PR #857 (ADR-316, WP0a), and has not been regenerated
since PR #868 and the WP15–20 PRs added ADR-317 through ADR-321 and
ADR-323. The header is now stale. Confirmed by direct file listing: the
highest real file is `ADR-323-governed-pipeline-shard-placement.md`, ADR-322
does not exist (deliberately skipped per Wave-2's numbering-collision
avoidance), so **the next available number is 324** — matching the
program's own stated number for this wave, but arrived at by recounting the
filesystem, not by trusting the generated header. `node scripts/adr-index.mjs`
should be re-run before any Wave-3 ADR is committed, both to refresh the
header and to re-run the CI collision gate ADR-316 established.

## Summary table

| # | Paper (brief's shorthand) | Found | Source | Grade | Numbers verified | Code/artifact available now |
|---|---|---|---|---|---|---|
| 1 | SPADE | Yes | arXiv:2608.19197 | A | all figures exact match | **Yes** — `spade-rl/spade` (GitHub, real content, 31 stars, MIT) |
| 2 | D²ACCI | Yes | arXiv:2608.17756 | A | all figures exact match, plus one bonus figure not in the brief | **No** — no code/data link anywhere found |
| 3 | DeAR | Yes | arXiv:2608.17282 | B+ | headline mechanism + benchmark count match; one specific mechanism detail (dead-end topology behavior) not independently confirmable from the abstract alone | **No** — abstract states code ships only "upon acceptance" |
| 4 | Zetta | Yes | arXiv:2608.16590 | A | all figures exact match | **Yes** — `air-embodied-brain/Zetta-Embodiment` (GitHub, real content, 224 stars, confirmed author repo) |

All four papers exist, exactly as named, dated Aug 17–19 2026 as claimed —
a clean hit rate matching Wave 2 (six of six) and better than Wave 1 (six of
eight). **Two of four have no available code today (D²ACCI, DeAR)** — a
higher no-code ratio than Wave 2's one-in-six (TRUSS) plus one-under-review
(StagedWorkspace); size the corresponding work packages as
reproduction-from-description efforts, not integration efforts, the same
posture Wave 2 required for WP16/WP17.

**Most important finding of this wave**: SPADE collides, in the *same
adjacent domain* (multi-agent systems), with a well-established, widely
taught academic framework of the identical name — more serious than Wave
1's LiveMem/iOS-app collision, comparable in severity to Wave 2's
MemFuse/memfuse collision. See §1 below.

---

## 1. SPADE

- **Found**: Yes — [arXiv:2608.19197](https://arxiv.org/abs/2608.19197),
  "SPADE: Self-Play in Adaptive Synthetic Executable Environments,"
  submitted 2026-08-19. Authors: Bo Liu, Simon Yu, Yiding Jiang, Ao Qu,
  Andrew Zhao, Zichen Liu, Junsu Kim, Zijian Zhou, Seungone Kim, Tongzheng
  Ren, Mickel Liu, Hanfei Yu, Zhaorun Chen, Weiyan Shi, Paul Pu Liang, Luke
  Zettlemoyer, Yejin Choi, Natasha Jaques. Subjects: cs.CL, cs.AI.
- **Grade**: A.
- **Claims vs. source** (confirmed verbatim against the abstract):
  - Single LLM plays both an **Environment Designer** ("writes complete,
    long-horizon training environments as executable code with an OpenAI
    Gym-style `reset()`/`step()` interface") and a **Reasoning Agent** —
    matches the brief exactly, including the "executable" framing (state
    transitions, reward functions, verification code all authored as code).
  - **Regret signal**: "The Reasoning Agent's regret is estimated using the
    gap between its reward with and without privileged hints" — matches.
  - **30B-parameter results**: abstract, verbatim — "SPADE improves over the
    strongest fixed-environment baseline by +5.3 on average across eight
    held-out math, science, code, and reasoning benchmarks, and lifts the
    tool-use setting by +5.7 on BFCL-v4 multi-turn and +13.9 on
    ACEBench-Agent." Exact match to the brief's three figures (5.3 / 5.7 /
    13.9).
  - **External-grounding ablation**: abstract, verbatim — "we find several
    components critical to success: grounding the Environment Designer on
    documents sampled from a large pretraining corpus, and giving it an
    accumulated environment memory." This confirms the brief's claim that
    external corpus grounding is called out as load-bearing. **Caveat**:
    the abstract does not itself use the phrase "closed-epistemic-loop
    safeguard" — that framing is the program's own interpretation of *why*
    the ablation matters (an ungrounded Environment Designer could drift
    into self-referential, non-representative environments), not a
    quotation from the paper. State it as our framing in any ADR, not as a
    paper claim.
- **Artifact availability — checked, not assumed**: arXiv Comments field,
  verbatim: "Work in progress. Project page: https://spade-rl.github.io ;
  Code: https://github.com/spade-rl/spade." Verified directly via the
  GitHub API: `spade-rl/spade` is public, organization-owned, **1,389 KB,
  31 stars, 3 forks, MIT-licensed, last pushed 2026-08-20T05:20:09Z**, and
  contains real content — training scripts for 4B/8B/30B models,
  Slime/SGLang and Megatron-LM distributed-training integration, a Tinker
  framework alternative, and an evaluation harness. Not a placeholder.
  **The reference implementation is genuinely available today.**
- **Name collision — the most severe of this wave, flag for citation
  care**: **SPADE (Smart Python multi-Agent Development Environment)** is a
  real, long-established, widely used open-source multi-agent systems
  framework for Python — FIPA-compliant agent messaging, behavior models
  (finite state machines, periodic behaviors), presence notification, and a
  web GUI. It predates this arXiv paper by years and is the subject of
  dozens of academic course/lab repositories (confirmed via GitHub search:
  `spade-labs`, `spade-agent-lab`, `spade_agent_communication`, and similar,
  spanning multiple universities). This is **not** the same domain overlap
  as Wave 1's LiveMem (unrelated iOS photo app — low confusion risk); it is
  the *same* general field (multi-agent systems), the way Wave 2's MemFuse
  collision was "memory fusion" colliding with "memory fusion." **Always
  cite as "SPADE (arXiv:2608.19197, Self-Play in Adaptive Synthetic
  Executable Environments)" in full, and never as a bare "SPADE" in any
  ADR, package name, crate name, or module name in this program's own
  deliverables** — the same discipline applied to TARL, TRUSS, and MemFuse
  in prior waves.

## 2. D²ACCI

- **Found**: Yes — [arXiv:2608.17756](https://arxiv.org/abs/2608.17756),
  "D²ACCI: A Dual-Loop Diagnostic Protocol for Evidence-Preserving Agent
  Memory," submitted 2026-08-18 (v1), revised 2026-08-19 (v2). Authors: Xule
  Liu, Yijun Liu, Chao Li, Shao Kun. Subjects: cs.AI.
- **Grade**: A.
- **Claims vs. source** (confirmed verbatim against the abstract):
  - Dual-loop structure: an inner loop is "the actual execution of a
    memory-augmented agent," an outer "diagnostic gate promotes,
    feature-flags, or rejects memory interventions based on paired
    evidence, protected-slice monitoring, and trace-level localizability" —
    matches the brief's "localize which memory stage changed" framing.
  - **93.59% LoCoMo, 90.93% LongMemEval** — abstract, verbatim: "achieving
    93.59% on LoCoMo, 90.93% on LongMemEval." Exact match.
  - **Bonus figure beyond the brief**: the abstract also reports **57.20% on
    PersonaMem-V2** — a third benchmark not named in the original brief.
    Cite all three if citing this paper's headline numbers; omitting
    PersonaMem-V2 would understate the paper's own reported scope.
  - **Interventions worth 1.9–3.7pp at p≤.003** — abstract, verbatim: "Five
    paired ablations show that supplement extraction, session-memory
    retrieval, and Forget Guard yield statistically significant gains (+1.9
    to +3.7pp, all p ≤ .003)." Exact match. Note the abstract also states a
    *negative* result the brief did not mention: "BM25/RRF is retained as a
    monitored feature flag" (i.e., one of the five ablations did *not*
    clear the bar for full promotion) — worth carrying into any ADR as an
    example of the gate correctly not-promoting a weak intervention.
  - **98–100% DCR@3 vs. 0% for result-only logs** — abstract, verbatim:
    "Diagnostic artifacts reach 98–100% DCR@3 versus 0% for results-only
    logs." Exact match.
- **Artifact availability — checked, not assumed**: no Comments field, no
  code/data URL anywhere on the arXiv abstract page, and no submission-history
  entry pointing to a repository. A targeted GitHub search for "MemStack" (the
  paper's own instantiation vehicle, per its abstract: "We instantiate the
  protocol in MemStack") and for "D2ACCI"/"D2ACCI-Eval" returned only
  unrelated projects (a Claude Code skill framework, a C memory-allocation
  library, a flash-card app, several student memory-management exercises —
  none related to LLM agent memory) plus two research-digest blog posts
  (`dmoliveira/ai-research-briefs`, `vincentxuu/quidproquo`) that summarize
  the paper's own abstract but are not the authors' repository. **No code or
  benchmark artifact for D²ACCI is available today.** Treat as a
  from-scratch reproduction, the same posture Wave 2 required for TRUSS and
  StagedWorkspace.
- **Name collision**: "D²ACCI" itself is distinctive and low-risk. "MemStack"
  (the paper's instantiation name, not the protocol's own name) collides
  with several unrelated, low-profile open-source projects — low confusion
  risk since none are in the LLM-memory domain, but this program should
  never adopt "MemStack" as its own package/crate name given the generic
  collisions already present, independent of any specific domain overlap.

## 3. DeAR

- **Found**: Yes — [arXiv:2608.17282](https://arxiv.org/abs/2608.17282),
  "DeAR: Decentralized Agentic Reasoning via Capability Grounding and
  Collaborative Thought Navigation," submitted 2026-08-18. Authors: Xing
  Wei, Changmeng Zheng, XiaoYong Wei, Xiufen Ye, Qing Li. Subjects: cs.AI.
- **Grade**: B+ (mechanism and headline claim confirmed verbatim; one
  specific behavioral detail from the brief is not independently
  verifiable from the abstract alone — see below).
- **Claims vs. source**:
  - **No central planner, three mechanisms** — abstract, verbatim: "(1)
    decentralized capability grounding for query-dependent agent
    specialization, (2) thought map navigation for targeted peer
    interactions, and (3) topology update for adaptive error correction."
    Matches the brief's three-part mechanism description closely.
  - **9 benchmarks** — abstract, verbatim: "Evaluations across 9 diverse
    multimodal reasoning and text-based QA benchmarks." Exact match to the
    brief's "9 multimodal+text benchmarks."
  - **Improves across benchmarks** — abstract, verbatim: "DeAR consistently
    outperforms recent baseline methods." Qualitative only, same posture as
    Wave 2's SkillForge — no percentage figure is given in the abstract, so
    do not invent one.
  - **Not independently confirmed from the abstract**: the brief's specific
    claim that on a dead-end "topology change" causes the system to
    "continue not restart" is a plausible reading of "topology update for
    adaptive error correction," but the abstract's four-word phrase does
    not itself state the continue-vs-restart distinction. This is graded
    down to B+ rather than A for that reason — the mechanism-*category*
    (adaptive topology change on error) is confirmed, the specific
    behavioral nuance is not, without reading the full paper body (not done
    in this pass, since no full-text or code artifact is available to
    check it against). **Do not cite "continue not restart" as a verified
    quotation in any ADR** — cite only "topology update for adaptive error
    correction" verbatim, and describe the continue-vs-restart framing as
    this program's working interpretation pending a full-text read.
- **Artifact availability — checked, not assumed**: the abstract's own text
  states, verbatim, "The source code will be available at
  https://open_upon_acceptance" — a literal placeholder URL, not a working
  link. No Comments field with a real repository link exists on the arXiv
  page. **No code is available today; this is explicit in the paper's own
  wording, not an absence this review had to search for.** Treat as a
  from-scratch reproduction.
- **Name collision**: no meaningful collision found in the agentic-reasoning
  domain. "DeAR" the bare acronym is heavily used elsewhere in unrelated
  software (Dear ImGui and its many bindings/extensions, `dear-github`,
  `DeArrow`) but none of these are agent-reasoning systems and none use the
  exact "DeAR" capitalization in the same sense — low collision risk,
  lower than SPADE's, comparable to Wave 2's SkillForge.

## 4. Zetta

- **Found**: Yes — [arXiv:2608.16590](https://arxiv.org/abs/2608.16590),
  "Zetta ζ: An Efficient Closed-Loop Embodied Harness for Self-Evolving
  Physical Intelligence," submitted 2026-08-17. Authors: Xin Ding, Liang Mi,
  Mingzhe Huang, Zixuan Wang, Chao Zhang, Zixu Hao, Fu Chen, Xiangyu Li,
  Yikai Zheng, Yaoyu Guo, Weijun Wang, Kun Li, Hao Wu, Yunxin Liu, Ting Cao.
  Subjects: cs.RO.
- **Grade**: A.
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **Frozen base policy, evolves runtime critics + recovery skills** —
    abstract, verbatim: "Zetta, a closed-loop embodied harness that evolves
    code-based runtime critics and recovery skills online while keeping the
    base policy frozen." Matches the brief and the same frozen-weight
    posture ADR-313 already adopted for SHAPER.
  - **Three timescales** — abstract, verbatim: "Through three
    timescale-separated loops, Zetta provides action-frequency governance,
    rollout-level critic-recovery proposal, and validation-gated skill
    updates." This confirms three distinct timescales exist and their
    *category* (immediate action governance / rollout-level proposal /
    validated skill promotion), matching the brief's ms/seconds/
    minutes-hours framing in spirit. **Caveat**: the abstract does not
    itself attach literal "ms," "seconds," or "minutes–hours" labels to the
    three loops — those specific units are the brief's gloss on the
    qualitative ordering, confirmed as directionally correct via the
    project's own README (see below) but not verbatim in the abstract.
  - **90.8% LIBERO Pro, 93.6% RoboCasa, 11.1× inference speedup** — abstract,
    verbatim: "Zetta achieves state-of-the-art success on LIBERO-Pro and
    RoboCasa under our current rollout budget, reaching 90.8% and 93.6%,
    with an 11.1x inference speedup." Exact match on all three figures.
- **Artifact availability — checked, not assumed**: the arXiv abstract page
  itself shows **no Comments field and no code link** — a WebFetch of the
  page directly confirmed this. Rather than concluding "no code available"
  from that absence alone (the mistake to avoid per ADR-305 §6), a targeted
  GitHub search for the paper's own name plus "embodiment" surfaced
  `air-embodied-brain/Zetta-Embodiment`, owned by an organization whose
  GitHub profile reads "Embodied Brain Team at Institute for AI Industry
  Research (AIR), Tsinghua University" — created 2026-08-18 (one day after
  the paper's submission), currently **public, 4,224 KB, 224 stars, 13
  forks, pushed 2026-08-19T12:20:51Z**. Its README opens: "Zetta is an
  efficient closed-loop embodied harness for self-evolving physical
  intelligence. It evolves code-based runtime critics and recovery skills
  online while keeping the base policy frozen, achieving state-of-the-art
  success on LIBERO-Pro (90.8%) and RoboCasa (93.6%) with an 11.1x inference
  speedup" — a verbatim match to the arXiv abstract's own numbers, and its
  repository layout names the exact evolution-loop stages ("Failure
  Cluster → Stage 1 causal Diagnose → Stage 2 Critic-Recovery Candidates →
  Shadow Replay → paired Same-seed Gate → Held-out seeds → Reject/Promote")
  that give the abstract's "validation-gated skill updates" its concrete
  shape. **This is confirmed as the authors' own reference implementation,
  not a coincidental name match, and the code is genuinely available
  today** — the arXiv page's own metadata omission does not mean no code
  exists; always check GitHub directly, per ADR-305 §6.
- **Name collision**: "Zetta" collides with `zettajs` (an IoT device
  platform, unrelated domain, dormant since ~2016 per its repo activity)
  and several unrelated small projects (a ZFS replication tool, a Minecraft
  mod, a note-taking app). None are in the embodied-agent domain — low
  collision risk, comparable to Wave 1's LiveMem.

---

## Novelty mapping — verdict per paper, checked against `gh`, not assumed

**The general caveat carried from Wave 2, restated**: a negative-existence
claim ("no system does X") over a fast-moving space is never fully provable
by a bounded search. Every verdict below was checked by fetching real
repository contents (file listings, READMEs, ADR titles) rather than
trusting descriptions, per the Wave-1 lesson (Dream Machine, Autogenous)
and the Wave-2 lesson (re-fetching `ruvnet/autogenous` ADR-393 directly).

### SPADE → dream-machine / MetaHarness / Darwin / Ruflo / RuVector

- `ruvnet/dream-machine`'s own description ("Freeze the model. Evolve the
  harness. Evaluation is not promotion — the machine never merges; a human
  does.") and ADR-306's adoption of it are philosophically adjacent — both
  are frozen-model, evidence-gated evolution loops — but dream-machine
  evolves the **harness/skills around an existing task set**, not the
  **task set itself**. A targeted search of `ruvector`'s own ADR corpus and
  of `dream-machine`'s repository for "self-play," "environment designer,"
  or "regret signal" returned no hits.
- **Genuinely new relative to everything currently in the org**: an LLM
  that authors whole executable `reset()`/`step()` environments (state
  transitions, rewards, verification code) as a *mutation-candidate source*
  for Darwin, driven by a regret signal comparing performance with/without
  privileged hints — this exists nowhere in `ruvector`, `dream-machine`,
  `metaharness`, or `ruflo` today. It is the same *category* of addition
  Wave 2's SkillForge (ADR-321) was: one more candidate-generation strategy
  feeding the existing Darwin/dream-machine promotion pipeline, not a new
  pipeline.
- **The load-bearing risk this program must actually design against**: an
  Environment Designer that is *also* the Reasoning Agent, self-play,
  training itself on its own generated environments, is structurally a
  closed epistemic loop unless something outside the loop grounds it. The
  paper's own ablation (external corpus grounding is "critical to success")
  is direct evidence this risk is real even in the source paper's own
  results, not merely a hypothetical this program is inventing. See the
  program plan's REQUIRED invariant below.

### D²ACCI → RuVector / Darwin / MetaHarness / dream-machine

- `ruvector` **already has** a paired-bootstrap statistical promotion gate
  (`scripts/research-gate/research_gate.py`,
  `crates/ruvector-sota-bench/harness/src/statistics.ts`, ADR-282, adopted
  as the promotion mechanism by ADR-306). This is conceptually close to
  D²ACCI's *outer loop* (a gate that promotes/feature-flags/rejects based
  on paired evidence) — but `research-gate` operates at the level of "does
  this whole mutation beat its parent," not at D²ACCI's granularity of
  "which pipeline *stage* (ingestion/extraction/retrieval/filtering/
  generation) is responsible for a memory failure." No existing `ruvector`
  or sibling-repo component does stage-level fault localization for the
  memory pipeline specifically, and none ships a DCR-style graded
  observability metric.
- **Genuinely new**: a diagnostic *inner* layer that sits underneath the
  existing promotion gate and localizes which of the (already-real, already
  five-stage in `crates/ruvector-agent-memory`) memory operations changed
  behavior — this is additive to, not a duplicate of, `research-gate` and
  the WP4 TARL ledger (`crates/ruvector-agent-memory/src/{ledger.rs,
  observation.rs}`) and the WP18 MemFuse causal-graph fusion
  (`crates/ruvector-agent-memory/src/fusion.rs`).

### DeAR → Autogenous / LatentMesh / Ruflo / MidStream / RuVector

- `ruvnet/autogenous`'s `packages/radio-moe/src/` already implements a
  peer-expert mesh (`mesh.ts`, `mesh-evolve.ts`, `mixture.ts`, `failover.ts`,
  `reputation.ts`, `relevance.ts`) — confirmed by direct file listing. This
  is a **mixture-of-experts routing mesh with reputation-weighted
  selection and failover**, not a peer-to-peer collaborative-reasoning
  system with a shared "thought map" and topology changes triggered
  specifically by dead-end detection. A targeted search of `radio-moe`'s
  source for "topology update" and "thought map" returned no hits.
- `ruflo`'s swarm topologies (hierarchical/mesh/ring/star/hybrid, per this
  repo's own `CLAUDE.md`) are **static configuration choices selected at
  swarm-init time**, not a topology that is dynamically restructured mid-run
  in response to one agent hitting a reasoning dead-end.
- **Genuinely new**: query-dependent decentralized capability grounding (no
  central planner assigns roles), a shared thought-map data structure
  peers navigate rather than exchange messages through a broker, and
  topology mutation triggered by dead-end detection specifically — none of
  the three exists in `autogenous`, `LatentMesh`, or `ruflo` today. This
  complements rather than competes with LatentMesh's ADR-309/310/311 latent
  communication fabric (which governs *whether a communication channel is
  attributable*, not *how peers are selected or how the collaboration graph
  reshapes itself*).

### Zetta → RuView / Ruflo / MetaHarness / dream-machine / RVM

- **`RVM`** (confirmed via its own GitHub description: "The Virtual Machine
  Built for the Agentic Age, in Rust") is a runtime/hosting/execution-
  sandboxing layer with no robotics or physical-actuation scope anywhere in
  its ADR corpus (checked by filename search for
  robot/embodied/physical/actuat* terms — no matches). **RVM is not a
  plausible near-term home for Zetta's critic/recovery loop**; it is, at
  most, a witness/provenance-anchoring surface for promoted skill updates,
  the same relationship it has to every other PIR mutation-promotion path.
- **`RuView`** (confirmed via its own GitHub description: turns WiFi CSI
  into spatial intelligence / vital-sign monitoring / presence detection —
  a **sensing**, not **actuation**, system) has no physical-robot rollout
  infrastructure either.
- **No repo in the `ruvnet` org currently operates physical robots or a
  robot-simulation rollout harness** (LIBERO/RoboCasa-class environments).
  This is the same structural gap Wave 1 identified for the universal-CSI
  stretch bet (WP14/RuView) — Zetta is, if pursued, **a first-party bet
  into a domain the org does not yet occupy**, not an extension with an
  existing home to slot into. The genuinely new content is entirely the
  paper's own mechanism (frozen-policy + evolving critics/recovery at three
  timescales); there is no internal duplication risk because there is
  nothing internal yet to duplicate.

---

## Bottom line for ADR-seeding

- **SPADE and Zetta are the strongest-evidenced items**: grade A, verbatim
  figure matches, and — checked directly via the GitHub API rather than
  assumed from the papers' own text — both have real, live, non-placeholder
  reference implementations today (`spade-rl/spade`; `air-embodied-brain/
  Zetta-Embodiment`). These are safe to adapt-from-repo, the same posture
  Wave 2 applied to HarnessRisk, MemFuse, SkillForge, and pipeline-shards.
- **D²ACCI and DeAR have no available code today** — DeAR says so
  explicitly in its own abstract; D²ACCI simply has no code/data link
  anywhere findable. Both must be built from the paper's description as
  from-scratch reproductions, the same posture Wave 2 required for TRUSS
  and StagedWorkspace, and the corresponding work packages below are sized
  accordingly (larger than SPADE's/Zetta's adapt-from-repo scope).
- **DeAR's grade is B+, not A**, because one specific behavioral claim in
  the brief (dead-end topology causing "continue not restart") could not be
  confirmed from the abstract alone and no code/full-text was available to
  check it against in this pass — flagged rather than guessed, the same
  discipline Wave 2 applied to SkillForge's unquantified "beats baselines"
  claim.
- **SPADE carries this wave's most serious name-collision risk** — a live,
  long-established, same-domain multi-agent framework (SPADE — Smart Python
  multi-Agent Development Environment) shares the exact name. Every ADR,
  package name, and doc referencing the arXiv paper must disambiguate
  explicitly, the same discipline Wave 2 required for MemFuse and TRUSS.
- **The ADR-index hygiene finding** (INDEX.md's stale "next available: 317"
  header) should be fixed as a small housekeeping item alongside whichever
  PR lands the Wave-3 ADRs — regenerate via `node scripts/adr-index.mjs`
  rather than hand-editing the header, consistent with ADR-316's stated
  policy that the generated file is the counter's source of truth.
