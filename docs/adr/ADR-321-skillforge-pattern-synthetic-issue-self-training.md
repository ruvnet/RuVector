# ADR-321: SkillForge-Pattern Synthetic-Issue Self-Training in the Darwin Loop

- **Status**: Proposed
- **Date**: 2026-08-20
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-313 (PIR, extends — SHAPER frozen-weight harness evolution, WP9), directly; see `docs/research/perpetual-intelligence-runtime/06-wave2-evidence-review.md` §4
- **Tags**: pir, wave-2, skillforge, self-training, darwin, entity-grounded-skills

## Context

Wave-2 evidence review grades this paper **B+**, not A —
[arXiv:2608.18933](https://arxiv.org/abs/2608.18933), "SkillForge:
Self-Distilling Agents for Project-Specific Issue Resolution," submitted
2026-08-19. The B+ grade reflects a specific, load-bearing gap: the paper's
mechanism claims are confirmed verbatim, but its headline "beats baselines"
claim is under-specified — the abstract states it qualitatively only
("consistently improves issue resolution performance over strong
baselines"), with **no percentage-point figure given**, so the magnitude of
improvement is not independently verifiable from the abstract alone.

Confirmed claims (verbatim against the abstract):

- **Synthesizes project-specific issues from test-covered functionality**:
  "SkillForge synthesizes project-specific issues by re-implementing
  test-covered core functionalities of the repository."
- **Distills entity-grounded skills**: "SkillForge distills reusable
  project-specific knowledge into entity-grounded skills and associates
  them with relevant repository entities for future issue resolution."
- **Beats issue-resolution baselines**: confirmed as a real claim, not
  fabricated — but qualitative only. **This ADR does not cite, and no
  derived document may cite, a specific percentage for this claim**; the
  evidence review is explicit that reading the paper's results tables
  directly (not done in this evidence pass) would be required before any
  number could be attributed, and this ADR does not do that here.

**Artifact availability — checked, not assumed**: the arXiv comments field
states plainly, "Our code and data are available at
`github.com/cslsolow/SkillForge`." Verified directly via the GitHub API:
public, 2.5 MB, last pushed 2026-08-20T01:28:03Z (the same day as the
evidence review), containing real content (`distilling/`, `synthesis/`,
`src/`, `evaluation_result/`, `pyproject.toml`) — not a placeholder. This ADR
adapts the confirmed-live implementation, the same posture ADR-320 applies
to MemFuse.

**Name collision**: no major pre-existing project found in a targeted
search. "SkillForge" follows a common "X-Forge" naming pattern shared by
many unrelated tools and games, so it is not a unique name, but no specific
conflicting project surfaced — lower risk than ADR-319's TRUSS or ADR-320's
MemFuse. Still worth a final targeted check before this program uses
"SkillForge" as its own package or crate name.

ADR-313 already gives `ruvector` a real evolutionary loop (Darwin,
`@metaharness/darwin`, called from `harness/src/darwin.ts`,
`examples/mragent`'s `scorePolicy`, `crates/sona/src/darwin_guard.rs`) with
frozen model weights and multiple approved mutation surfaces. That loop
today waits for mutation-candidate sources — real incidents, manual
proposals, or other ADR-313-approved surfaces. SkillForge's contribution is
a **self-supplied candidate source**: synthesizing issues from the target
repository's own test-covered functionality, rather than waiting for a real
incident to occur.

## Decision

Add a new Darwin mutation-candidate source that synthesizes, repairs, and
distills project-specific skills, informed by SkillForge's mechanism
(arXiv:2608.18933) and adapting the confirmed-live `cslsolow/SkillForge`
implementation:

1. **Synthesize project-specific regressions from test-covered functions.**
   Re-implement test-covered core functionality of the target repository
   (per SkillForge's confirmed mechanism) to generate realistic,
   repository-specific issues, rather than relying solely on real incidents
   or manually authored mutation candidates as ADR-313's existing sources do.
2. **Repair the synthesized issue** using the same frozen-weight harness
   this program already requires (ADR-313) — the repair step does not
   introduce a fine-tuning path; it is a harness/skill mutation like any
   other candidate ADR-313 governs.
3. **Distill the successful repair strategy into an entity-grounded skill.**
   Skills are stored in RuVector, **attached to the specific files,
   functions, or ADRs they concern** — not a flat, undifferentiated skill
   list — so a future issue-resolution attempt on one of those entities can
   retrieve directly relevant skills instead of searching an undifferentiated
   corpus. This mirrors SkillForge's confirmed entity-grounding mechanism.
4. **Feed this source into ADR-313's existing frozen-weight harness-evolution
   loop as one more candidate-mutation source**, alongside its existing
   surfaces. This does not bypass ADR-306's evaluation pipeline or ADR-305's
   separation-of-powers invariant: a synthesized-and-repaired candidate is
   still an untrusted proposal until it clears ADR-306's promotion gate,
   exactly like any other Darwin-proposed mutation.
5. **Evidence grade B+ governs this ADR's citation discipline**: the
   mechanism (synthesis from test coverage, entity-grounded distillation) is
   cited with confidence, per the confirmed-verbatim abstract claims above.
   The "beats baselines" claim is cited qualitatively only — **no
   percentage figure is invented or attributed to SkillForge in this ADR**,
   and this program's own `research-gate`-recomputed delta, not a citation
   of the paper's own claim, is what determines whether any given synthesized-
   and-distilled skill is promoted.

## Consequences

### Positive

- Gives Darwin a self-supplied mutation-candidate source that does not
  require waiting for a real incident — the harness-evolution loop can
  exercise itself continuously against synthesized, test-covered-derived
  issues.
- Entity-grounded skill storage (attached to files/functions/ADRs) makes
  future retrieval targeted rather than requiring a search over an
  undifferentiated skill corpus — directly useful to every other ADR-313
  mutation surface, not just this one.
- The confirmed-live `cslsolow/SkillForge` implementation gives WP19 a
  genuine port-and-adapt scope, materially smaller than ADR-318's or
  ADR-319's from-scratch builds.
- The B+ grade's explicit "do not invent a percentage" discipline models the
  same evidence-honesty standard this program applies elsewhere (e.g.
  ADR-311's "LATTE is not real" correction) — citing a real B+-grade
  mechanism accurately is preferred over inflating it to match an A-grade
  item's citation style.

### Negative

- The B+ grade means this ADR has a genuinely weaker acceptance bar to point
  to than the wave's five A-grade items — this program cannot cite an
  expected magnitude of improvement from the source paper at all, only that
  an improvement exists qualitatively; the internal benchmark delta carries
  correspondingly more weight in the promotion decision for this source.
- Synthesizing issues from test-covered functionality is itself a mutation
  surface that must be validated as producing *realistic* issues, not just
  syntactically valid ones — a synthesis step that generates unrealistic or
  trivial issues would distill low-value skills without failing any existing
  gate designed for that failure mode.
- Depends on ADR-313's SHAPER-pattern loop existing first, and inherits
  ADR-313's own existing blocking dependency (WP0b's GGUF glob/alias bug)
  before live-serve testing of this new candidate source can begin.

## Security / Validation Gates

- **No fine-tuning path**: the repair step reuses ADR-313's existing
  structural (CI-enforced) frozen-weights check — synthesized-issue repair
  is a harness/skill mutation, never a weight update.
- **Separation-of-powers invariant** (ADR-305, ADR-313): a synthesized,
  repaired, and distilled skill is a candidate proposal only; it gains no
  promotion authority and must clear ADR-306's evaluation pipeline like any
  other Darwin-proposed mutation.
- **Entity-attachment integrity**: a skill's attachment to a specific file,
  function, or ADR should reference that entity's exact content hash where
  ADR-318 has landed (interop, not a hard dependency of this ADR) — so a
  skill does not silently misattach after its target entity changes.
- **No invented magnitude**: this ADR, and every document derived from it,
  cites SkillForge's "beats baselines" claim qualitatively only; a specific
  percentage may only be cited once independently confirmed from the
  paper's results tables or from this program's own benchmark.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  this module lands, per `07-wave2-program-plan.md`'s call-out that WP19
  touches Darwin's mutation-proposal surface.

## Affected Repos

- `ruvnet/ruvector` only — `harness/src/darwin.ts`, `examples/mragent`
  `scorePolicy` (same surfaces WP9/ADR-313 already own). Single-repo scope
  per `07-wave2-program-plan.md`.

## Dependencies

Depends on ADR-313 (the frozen-weight harness-evolution loop this ADR
supplies a new candidate source into) directly. No dependency on ADR-317,
ADR-318, ADR-319, ADR-320, or ADR-323 — this ADR is not part of the Wave-2
combined acceptance test.

## Alternatives Considered

- **Wait for real incidents only, without a synthetic-issue source** (the
  pre-Wave-2 status quo for this candidate-source dimension): rejected —
  this leaves the harness-evolution loop idle between real incidents;
  SkillForge's synthesis mechanism gives it a continuous, self-supplied
  exercise source instead.
- **Cite a specific improvement percentage for SkillForge's "beats
  baselines" claim, inferred or estimated from context**: rejected outright
  — the evidence review is explicit that no percentage exists in the
  abstract and that inventing one would misrepresent a B+-grade source as
  if it carried A-grade quantitative backing.
- **Store distilled skills as a flat, unattached corpus rather than
  entity-grounded**: rejected — this discards SkillForge's confirmed
  entity-grounding mechanism, which is precisely what makes future retrieval
  targeted rather than requiring an undifferentiated search.
