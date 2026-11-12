# ADR-327: Zetta-Pattern Three-Timescale Physical-Evolution Harness (New Bounded Context, Explicit Stretch)

- **Status**: Proposed
- **Date**: 2026-08-21
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-313 (PIR, extends — SHAPER frozen-weight harness evolution, WP9; this ADR is a timescale-structured variant of the same frozen-base pattern); see `docs/research/perpetual-intelligence-runtime/08-wave3-evidence-review.md` and `09-wave3-program-plan.md`
- **Tags**: pir, wave-3, zetta, embodied, timescale-evolution, stretch, new-context

## Context

Wave-3 evidence review grades this paper **A** —
[arXiv:2608.16590](https://arxiv.org/abs/2608.16590), "Zetta ζ: An Efficient
Closed-Loop Embodied Harness for Self-Evolving Physical Intelligence,"
submitted 2026-08-17.

Confirmed mechanism, verbatim against the abstract: "Zetta, a closed-loop
embodied harness that evolves code-based runtime critics and recovery skills
online while keeping the base policy frozen. Through three
timescale-separated loops, Zetta provides action-frequency governance,
rollout-level critic-recovery proposal, and validation-gated skill updates."
**The literal "ms local critic / seconds recovery / minutes-hours validated
skill evolution" labels are this program's own gloss on those three
loops** — directionally confirmed by the repo's own README (see below) but
not verbatim in the abstract itself, which names the loops functionally
(action-frequency governance, rollout-level critic-recovery proposal,
validation-gated skill updates) without attaching literal time-unit labels.

Numbers (verbatim, exact match to the program brief): "Zetta achieves
state-of-the-art success on LIBERO-Pro and RoboCasa under our current
rollout budget, reaching 90.8% and 93.6%, with an 11.1x inference speedup."

**Artifact availability — checked, not assumed, and non-trivial to
confirm.** The arXiv abstract page itself shows no Comments field and no
code link (confirmed via direct fetch of the page) — an absence that, on its
own, would suggest no code exists. **It does not**: a GitHub search located
`github.com/air-embodied-brain/Zetta-Embodiment`, owned by the "Embodied
Brain Team at Institute for AI Industry Research (AIR), Tsinghua
University," created 2026-08-18 — one day after the paper's submission
date — public, 4,224 KB, 13 forks, last pushed 2026-08-19T12:20:51Z (star
count omitted here — it drifts continuously and is not load-bearing
evidence; size, forks, and push date are). Its README opens with text
matching the abstract
verbatim, including the exact 90.8%/93.6%/11.1x figures, and names the
evolution loop's concrete stages: Failure Cluster → Stage 1 causal Diagnose
→ Stage 2 Critic-Recovery Candidates → Shadow Replay → paired Same-seed Gate
→ Held-out seeds → Reject/Promote. This is confirmed as the real author
repository, not a coincidental match — this ADR adapts the confirmed-live
implementation, the same posture ADR-320, ADR-321, and ADR-324 apply to
their own confirmed-live sources. **The lesson generalized from this
check**: an arXiv page's own missing Comments field is not sufficient
evidence that no code exists; a targeted search is required before
concluding "no artifact," the same discipline this program already applies
to citation verification generally.

**Name collision — low risk.** "Zetta" collides with `zettajs` (a dormant
IoT platform, unrelated domain), a ZFS-related tool, a Minecraft mod, and a
note-taking app — none in the embodied-agent space.

**CRITICAL HONESTY, verified directly rather than assumed — this is the
load-bearing fact for this ADR's scoping decision.** No repository in
`ruvnet` operates physical robots today:

- `ruvnet/rvm`'s own description is "The Virtual Machine Built for the
  Agentic Age, in Rust" — a runtime/hosting layer, not a robotics platform. A
  filename search across `rvm` for `robot`/`embodied`/`physical`/`actuat*`
  returned no matches: **zero robotics ADRs**.
- `ruvnet/RuView`'s own description is WiFi CSI → spatial
  intelligence/vital-signs/presence detection — **sensing, not actuation**.
  It has no actuation surface to target.

**Zetta therefore has no existing actuation home anywhere in `ruvnet`.**
This is structurally worse than Wave 1's WP14 CSI work package, which at
least had `RuView` as a plausible eventual sensing-side home; Zetta's target
domain (physical robot control) has no analogous placeholder anywhere in
this org today. This ADR does not pretend otherwise, and does not scope a
physical deployment.

**Preprint-reproduction rule** (applies uniformly across this program, per
`09-wave3-program-plan.md`): Zetta's reported 90.8%/93.6%/11.1x figures are
the source paper's and reference implementation's own measurements on
LIBERO-Pro and RoboCasa, physical/simulated-robot benchmarks this program
has no way to run today. This ADR does not claim, and no derived document
may claim, that this program has reproduced or will imminently reproduce
those figures — the pattern this ADR actually adopts (see Decision) is
implemented and validated against a synthetic harness, not a physical or
simulated-robot benchmark, and carries no expectation of matching Zetta's
own reported numbers until an actual actuation target exists.

## Decision

**This is an explicit stretch bet, scoped honestly, not a physical
deployment.** Zetta's genuine contribution — a frozen base policy with
critics, recovery, and skills evolving at three separated timescales — is a
timescale-structured variant of ADR-313's already-adopted SHAPER frozen-weight
harness-evolution loop. This program adopts the **pattern**, not the robot:

1. **Implement the timescale-separated evolution harness pattern**, informed
   by Zetta's mechanism (arXiv:2608.16590) and adapting the confirmed-live
   `air-embodied-brain/Zetta-Embodiment` implementation's evolution-loop
   structure (Failure Cluster → causal Diagnose → Critic-Recovery Candidates
   → Shadow Replay → paired Same-seed Gate → Held-out seeds →
   Reject/Promote), in `crates/ruvector-sota-bench/harness` — its real home,
   beside WP9's existing SHAPER loop (`src/shaperLoop.ts`, `src/genome.ts`,
   `src/darwin.ts`) — as a **new bounded context**, not folded into ADR-313's
   existing single-timescale loop.
2. **Map Zetta's frozen-base constraint onto ADR-313's existing structural
   enforcement.** The same CI-enforced check ADR-313 already requires — no
   mutation surface reachable from the promotion pipeline may import a
   training/fine-tuning API — applies unchanged to this timescale-separated
   harness's frozen base policy. This ADR does not invent a second
   frozen-weights mechanism; it reuses ADR-313's existing one.
3. **Three timescale-separated evolution loops, structurally distinct
   evaluation tiers**: a fast local-critic loop (informing action-frequency
   governance), a mid-tier recovery-proposal loop (rollout-level
   critic-recovery candidate generation), and a slow, validation-gated
   skill-evolution loop (skill updates promoted only after the
   Shadow-Replay/paired-Same-seed-Gate/Held-out-seeds sequence the confirmed
   implementation names). Each tier's promotion criteria are distinct and
   independently gated — a candidate promoted at the fast tier is not
   thereby promoted at the slow tier.
4. **Synthetic/stubbed environment only.** Because no `ruvnet` repo has an
   actuation surface, this work package implements and validates the
   timescale-separated harness against a **synthetic, stubbed environment**
   inside `crates/ruvector-sota-bench/harness` — analogous in spirit to
   ADR-321's own stubbed SELECT/REPAIR/persistence slices
   (`src/skillForge.ts`) — proving the pattern's mechanics (three
   independently-gated timescales, shadow-replay-then-promote flow) without
   claiming any physical or simulated-robot result.
5. **Explicitly defer the physical/RuView actuation target.** Wiring this
   harness to an actual robot, a physical simulator, or `RuView`'s
   (currently sensing-only) surface is out of scope for this ADR and is
   deferred to a future ADR, contingent on both hardware/simulator access
   this program does not have today and a separate architectural decision
   about where an actuation surface would live. This ADR does not overclaim
   a physical deployment timeline.
6. **Repo placement left open.** Whether the timescale-separated harness
   pattern eventually grows into a new `ruvector` crate, a sibling-repo
   epic, or stays inside `crates/ruvector-sota-bench/harness` long-term is
   an open decision for the program coordinator, not fixed by this ADR.

## Consequences

### Positive

- Extracts Zetta's genuinely reusable contribution — timescale-separated,
  frozen-base evolution with a shadow-replay-then-promote gate — as an
  abstract harness pattern, independent of any particular robot or
  simulator, so this program benefits from the paper's design even with zero
  actuation surfaces available today.
- Mapping the frozen-base constraint onto ADR-313's existing structural
  enforcement avoids building a second, parallel frozen-weights mechanism —
  one CI-enforced check governs both the single-timescale SHAPER loop and
  this new timescale-separated variant.
- The confirmed-live `air-embodied-brain/Zetta-Embodiment` repository names
  a concrete, citable evolution-loop structure (Failure Cluster → Diagnose →
  Critic-Recovery Candidates → Shadow Replay → Same-seed Gate → Held-out
  seeds → Reject/Promote) this program can adapt directly rather than
  reconstructing from the abstract alone.
- Explicit, honest scoping (synthetic environment, deferred actuation
  target) sets accurate expectations for reviewers and avoids the specific
  credibility risk of an ADR implying a physical-robot capability this
  program does not have — the same evidence-honesty discipline ADR-311
  applies to correcting the "LATTE" citation and ADR-321 applies to
  SkillForge's unquantified claim.

### Negative

- This ADR delivers no physical or simulated-robot result and cannot be
  cited as evidence this program has reproduced Zetta's 90.8%/93.6%/11.1x
  figures — a synthetic-environment validation of the pattern's mechanics is
  a genuinely weaker deliverable than what this wave's other three ADRs
  produce against real `ruvector` subsystems.
- Structurally worse actuation-home problem than Wave 1's WP14: WP14 at
  least had `RuView` as a plausible eventual sensing-side home; this ADR's
  target domain (physical actuation) has no analogous placeholder anywhere
  in `ruvnet` today, and this ADR does not resolve that gap — it defers it.
- A three-tier, timescale-separated evolution harness is materially more
  complex to implement and validate than ADR-313's existing single-timescale
  loop, even scoped to a synthetic environment — three independently-gated
  promotion criteria, not one, each needing its own test coverage.
- Because the mapping from Zetta's literal "ms/seconds/minutes-hours" gloss
  to this program's own tier definitions is this program's own
  interpretation (the abstract itself does not attach those units), the
  three tiers' actual timing/frequency parameters are an implementation
  decision this ADR does not fix, leaving room for a mismatch between this
  program's synthetic harness and any future real actuation target's actual
  timing requirements.

## Security / Validation Gates

- **Structural frozen-weights enforcement (ADR-313's existing mechanism,
  reused unchanged)**: CI fails the build if any mutation surface reachable
  from this harness's promotion pipeline imports a training/fine-tuning API
  — applies to the frozen base policy exactly as it applies to ADR-313's
  SHAPER loop.
- **Independent per-tier gating**: a candidate's promotion at the fast
  local-critic tier, the mid-tier recovery-proposal loop, or the slow
  validation-gated skill-evolution tier are evaluated and logged
  independently; promotion at one tier never implies promotion at another.
- **Shadow-replay-before-promote discipline**: following the confirmed
  implementation's own gate sequence, a skill-evolution candidate passes
  Shadow Replay and a paired Same-seed Gate against held-out seeds before
  promotion — mirrored from the confirmed-live repo's structure, not
  invented independently.
- **Separation-of-powers invariant** (ADR-305, ADR-313, adopted unchanged):
  candidates proposed at any of the three timescales never gain promotion
  authority themselves; every promotion still routes through ADR-306's
  adopted evaluation pipeline.
- **No overclaimed deployment target**: no code, comment, design document,
  or status report produced under this ADR may state or imply that this
  program operates, or is imminently about to operate, a physical robot or a
  live physical-simulator integration; the synthetic/stubbed environment
  scope is stated explicitly wherever this work is described.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  this module lands, per this program's uniform practice for new harness
  surfaces.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-sota-bench/harness` (new
  timescale-separated evolution harness module, beside WP9's existing
  `src/shaperLoop.ts`/`src/genome.ts`/`src/darwin.ts`), synthetic/stubbed
  environment only. Candidate long-term placement (open, per Decision §6):
  a new `crates/ruvector-embodiment` crate, or a sibling-repo epic mirroring
  how Wave 1 treated LatentMesh/rvm/autogenous/RuView as separate epics —
  this ADR does not decide that placement question. No `RuView` code
  changes in this ADR's scope — the actuation target is explicitly
  deferred, not touched; any future coordination with `RuView` is
  soft-scoped to its existing sensing/state-estimation surface only, never
  actuation.

## Dependencies

Extends ADR-313 (the frozen-weight harness-evolution loop this ADR's
timescale-separated variant reuses the structural frozen-weights enforcement
of, and sits beside as a distinct bounded context). No dependency on
ADR-324, ADR-325, or ADR-326 — this ADR is independently scoped and is not
part of any combined Wave-3 acceptance test those three share.

## Alternatives Considered

- **Fold this pattern directly into ADR-313's existing single-timescale
  SHAPER loop instead of a new bounded context**: rejected — Zetta's
  three-tier, independently-gated structure is a genuinely different shape
  from ADR-313's existing loop; conflating them would either force
  ADR-313's simpler mutations through unnecessary three-tier gating or
  silently weaken this ADR's tier-independence guarantee.
- **Wire this harness directly to `RuView` now, treating its sensing
  capability as sufficient for an initial actuation integration**: rejected
  outright — `RuView` has no actuation surface today; describing this as
  even a partial physical integration would misrepresent a sensing-only
  system's capability, exactly the overclaiming this ADR's honesty
  discipline exists to prevent.
- **Defer this ADR entirely until a physical or simulated-robot target
  exists in `ruvnet`**: rejected — the timescale-separated evolution-harness
  pattern is independently valuable and testable against a synthetic
  environment today; waiting for an actuation target to appear before
  capturing the architectural pattern would lose the opportunity to validate
  the pattern's mechanics early.
- **Claim this ADR reproduces Zetta's reported LIBERO-Pro/RoboCasa
  results**: rejected outright, per the preprint-reproduction rule — this
  program has no actuation surface to run those benchmarks against; any
  such claim would be fabricated.
