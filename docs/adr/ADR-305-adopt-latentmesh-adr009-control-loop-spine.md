# ADR-305: Adopt Autogenous ADR-401 and LatentMesh ADR-009 as the Perpetual Intelligence Runtime's Definition and Control-Loop Spine

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-306, ADR-307, ADR-309, ADR-312, ADR-313, ADR-315 (PIR, this program); LatentMesh ADR-009 "Online Causal Control Loop" (LatentMesh repo, 2026-08-18, status Proposed); autogenous ADR-401 "The Perpetual Intelligence Machine" (autogenous repo, 2026-08-16, status Accepted/Partial) and ADR-400 "Self-Evolving Mesh Perpetual Loop" (autogenous repo, status Accepted — Implemented); ruflo ADR-322/322A/322B/322C "Metaharness Flywheel Integration" (ruflo repo, `v3/docs/adr/`, 2026-07-28, status Accepted); see `docs/research/perpetual-intelligence-runtime/04-verification-addendum.md`
- **Tags**: pir, latentmesh, autogenous, control-loop, governance, cross-repo

## Context

This program's brief asked for a new cross-mechanism architecture combining
self-evolving physical intelligence, persistent transactional memory, and
verified latent agent communication. Research for this ADR set (see
`docs/research/perpetual-intelligence-runtime/02-asset-map.md`, compiled
2026-08-19) found that architecture already designed, one day earlier, inside
`ruvnet/LatentMesh`'s own
[`docs/adr/009-online-causal-control-loop.md`](https://github.com/ruvnet/LatentMesh/blob/main/docs/adr/009-online-causal-control-loop.md)
(dated 2026-08-18, status "Proposed"). That ADR names a role for every
component in this program's brief — RuFlo, MetaHarness/Darwin, RuVector,
MidStream, Radio, RVF/RVM, RuView, and Autogenous — inside one loop:

```
execute → transfer latent state → counterfactual audit (LatentMesh ADR-003)
        → measure causal value
        → update edge value/authority ceiling (LatentMesh ADR-008)
        → persist (LatentMesh ADR-005)
        → change topology (LatentMesh ADR-006, Darwin)
        → next execution
```

LatentMesh ADR-009 states plainly what is built and what is not: *"The
statistical primitive (LatentMesh ADR-003) and admission gate (LatentMesh
ADR-008) are implemented; the closed loop across live components is not
wired."* That is the actual gap this program should close — not a greenfield
design problem.

**Verification note (updated)**: a second agent (asset-scout) subsequently
cloned `ruvnet/LatentMesh` directly (HEAD `922ea196`, 2026-08-18) and
confirmed the loop diagram and the "not wired" quote above verbatim against
source, including the per-stage "not implemented" markers on ADR-004 through
ADR-008. **One material correction to the asset map's framing: LatentMesh is
a small research prototype, not a substantial codebase.** The entire
workspace is 1,407 LOC across four crates (`latentmesh-core` 272,
`latentmesh-align` 454, `latentmesh-gate` 534, `latentmesh-bench` 147; 23
tests total) — roughly 1/23rd the size of `ruvector-core` alone. **There is
no network transport crate anywhere in the workspace.** "Adopt LatentMesh
ADR-009 as the spine" means adopting a *design contract* and ~1.4K LOC of
tested primitives (the causal-edge and admission-gate math), not inheriting a
mature system. Every later PIR ADR and work package must be scoped against
that reality — WP5 (wire LatentMesh live, ADR-309) in particular is building
a transport layer that does not exist today, not integrating one.

LatentMesh's maintainers are also still actively revising scope — the asset
map notes ADR-009 itself was revised twice in one day — so this program's
timeline is exposed to their churn (see `03-program-plan.md`, Top Risks §2).

## Reconciliation with autogenous ADR-401 (material finding, added post-draft)

Independent verification of `ruvnet/autogenous` (asset-scout, HEAD `b5c6e838`,
2026-08-19) surfaced a second, materially different finding the original
asset map did not capture: autogenous's ADR sequence runs **391 through 403**
(13 ADRs), not the four the asset map cited (391/392/393/397). The missing
ones include
[`ADR-401-perpetual-intelligence-machine.md`](https://github.com/ruvnet/autogenous/blob/main/docs/adr/ADR-401-perpetual-intelligence-machine.md)
— **"The Perpetual Intelligence Machine," status Accepted (framing +
capability map) · Partial (implementation), dated 2026-08-16** — and
[`ADR-400-self-evolving-mesh-perpetual-loop.md`](https://github.com/ruvnet/autogenous/blob/main/docs/adr/ADR-400-self-evolving-mesh-perpetual-loop.md),
status "Accepted — Implemented (first flywheel turn measured)." This author
independently confirmed ADR-401's content by reading it directly.

ADR-401 defines a 10-capability map (observe→diagnose→propose→execute→measure→revise,
governed self-improvement, cross-organization intelligence, and more), an
operating loop, and a V1 acceptance test — implemented mostly in
`packages/radio-moe/src/*.ts` (6,171 LOC, 3,161 LOC of tests), not the Rust
crates the asset map described. Several of its capabilities are already
**Built and measured** (e.g. peer-loss recovery at 30% mesh loss, p50 0.34ms;
fused-mixture decision quality beating best-single agent 100% vs 66.7%),
while others are honestly marked Partial, Gap, or Narrative — ADR-401 itself
models the same claims-honesty discipline this program has been trying to
adopt from LatentMesh. It also references a third ADR series it labels
"metaharness ADR-322" (flywheel receipts/promotion) — **that citation is a
misattribution**. A fourth research pass located the actual document at
**`ruvnet/ruflo`, `v3/docs/adr/ADR-322-metaharness-flywheel-integration.md`**
— "Adopt `@metaharness/{flywheel,darwin}` as pluggable engines behind
ruflo's ADR-176 self-improvement flywheel," status Accepted (phases 0–2
implemented), dated 2026-07-28. `ruvnet/metaharness` (the renamed
`agent-harness-generator`) has its own 230-ADR series topping out at
ADR-250 — no ADR-322 exists there. **This program cites the dependency as
"ruflo ADR-322" going forward; every reference to it elsewhere in this
program's documents should be corrected to match** (see
`04-verification-addendum.md` §5).

ruflo ADR-322 has a directly load-bearing implementation detail for this
program: its child ADR-322B states, verbatim, *"A proposer produces
untrusted candidates only. It cannot issue promotion decisions or mutate
active policy."* This is the same separation-of-powers boundary autogenous
ADR-401's capability-5 table row describes as "not yet one checked
predicate" — **but that row is stale relative to ADR-401's own Decision
section.** ADR-401's **Update 1 §3, "Converge the promotion invariant to one
predicate," is marked DONE**: `mesh-evolve.ts` exports
`promoteAuthorized(candidate, champion, { authorized, reversible }) →
PromotionDecision`, the single gate `Promote = Better ∧ Safe ∧ Authorized ∧
Reversible`, with each conjunct independently blocking and
`evolveMesh` routing every promotion through it — proven by
`test/promote-authorized.test.ts` (all-four promotes; any three-of-four does
not). This program cites the Decision section, not the stale capability
table, as ADR-401's authoritative status on this point (see ADR-315 for the
corrected scope this implies). ruflo ADR-322A independently implements the
same separation as one atomic transaction (`promoteFlywheelCandidate` as
sole promotion authority, verified under 100 concurrent promotion attempts
producing exactly one commit). This program
adopts that separation-of-powers invariant as a **governing invariant**,
alongside the six carried from the original brief (see Decision §3 below).

**This changes the decision below**: LatentMesh ADR-009 is a *proposed,
unwired, ~1.4K-LOC integration contract*. Autogenous ADR-400/401 are
**Accepted, partially-to-fully implemented, with measured results**, and
ADR-401 is, on its face, largely the same product this program's brief
describes. WP1 must read ADR-401 in full and choose explicitly between
adopting/extending it or stating in writing why the PIR program defines a
parallel Perpetual Intelligence Machine — silently re-deriving an Accepted
ADR-401 across this program's other ten ADRs would be the worst outcome this
program could produce.

## Decision

The Perpetual Intelligence Runtime does **not** define its own product
framing or its own cross-mechanism architecture. It adopts two existing,
already-Proposed-or-Accepted decisions from sibling repos, at two different
levels:

1. **Program definition of record**: PIR adopts autogenous ADR-401's
   10-capability map, its "perpetual means operational continuity, not
   infallibility" framing, and its V1/30-day acceptance-test structure as the
   program's own definition of what is being built and how it will be judged
   done. Where a PIR ADR in this set (ADR-306 through ADR-315) covers ground
   ADR-401 already scores as Built or Partial, that PIR ADR must say so
   explicitly and describe itself as closing ADR-401's named gap, not as
   introducing a new capability.
2. **Communication-fabric design contract**: PIR adopts LatentMesh ADR-009's
   seven-stage online causal control loop (execute → transfer → counterfactual
   audit → measure → update authority → persist → evolve topology) as the
   design contract specifically for the Latent Communication Fabric bounded
   context (ADR-309, ADR-310, ADR-311) — the one part of ADR-401's capability
   map (capability 6/7 boundary-crossing communication) that autogenous's own
   repo does not itself implement.
3. PIR work packages are scoped to close the gaps **both** ADRs already
   declare — LatentMesh ADR-009's "closed loop not wired" and autogenous
   ADR-401's own Partial/Gap rows — not to re-derive either document's
   claims. autogenous ADR-400/401 cite "metaharness ADR-322" for flywheel
   receipts/promotion; that citation is a misattribution corrected above —
   the actual document is **ruflo ADR-322** (Accepted, phases 0–2
   implemented), and its child ADR-322B's separation-of-powers invariant
   ("a proposer produces untrusted candidates only; it cannot issue
   promotion decisions or mutate active policy") is adopted here as a
   seventh governing invariant alongside the six carried from the brief,
   binding on every PIR ADR that defines a mutation-proposal or
   promotion-decision mechanism (ADR-306, ADR-313, ADR-315).
4. A coordination channel is opened with both the `ruvnet/LatentMesh` and
   `ruvnet/autogenous` maintainers before any ADR in this program assigns a
   number, or makes a claim, that could conflict with either repo's own ADR
   sequence. Every cross-repo ADR reference in this program's documents must
   name the owning repo explicitly (e.g. "LatentMesh ADR-009", "autogenous
   ADR-401", "ruvector ADR-134") — the asset map documents an ADR-103
   numbering collision across repos (ruvector's ADR-103 is unrelated to the
   witness/fix-manifest ADR-103 that lives in claude-flow/Ruflo), and this
   program must not create a second instance of that ambiguity.
5. The program's committed first work package (WP1) is, before any other PIR
   work starts: (a) read autogenous ADR-401 in full and produce the
   explicit adopt/diverge decision required above, (b) confirm LatentMesh
   ADR-009's loop against source (done — see Verification note), (c)
   confirm ruflo ADR-322/322A/322B/322C against source (**done**: cloned
   directly, HEAD `fa13ee4`, 2026-08-15; the separation-of-powers quote,
   the 322C canonical-encoding/signature stack including its three signing
   domains, and the evidence-grading vocabulary all check out verbatim —
   see `04-verification-addendum.md` §8), and (d) apply the **fix-history
   verification rule** below to every remaining inherited claim this
   program has not yet independently checked.
6. **Fix-history verification rule (added after PR #847 review)**: an
   inherited "known bug," "gap," or "not yet implemented" claim from any
   source document — the program brief, an upstream ADR, or a prior research
   pass in this program itself — must be checked against that path's actual
   fix history (`git log` on the named file/module, the owning repo's
   release notes or merged PRs) before being repeated in a PIR ADR. It is
   not sufficient that the asserting document's prose says the bug is open.
   This rule exists because all three of this ADR set's blocking review
   findings (ADR-401's promotion predicate, the "metaharness ADR-251"
   citation, and the ADR-150 misattribution — see ADR-315, ADR-306, ADR-313)
   shared the same root cause: a claim was carried forward from an upstream
   document without checking whether upstream's own state had since moved
   past it, or whether the citation resolved to a real document at all. The
   `ruvllm` HTTP-307 bug ADR-313 originally cited as open (later found
   already fixed on `main`, commit `946275a61`) is the concrete instance
   that surfaced this pattern.

## Consequences

### Positive

- Avoids duplicating a design that already exists and is already
  cross-component-aware; every later PIR ADR gets a concrete loop stage to
  attach to instead of inventing placement.
- Forces early cross-repo coordination instead of discovering a conflict
  after multiple ADRs and work packages are already built on a diverging
  architecture.
- Inherits LatentMesh ADR-009's own honesty discipline (narrowing its novelty
  claims against StateBridge/LatentMAS/MANTA/E2-Explainer) as the model for
  how this program's own ADRs should be written.

### Negative

- Couples this program's schedule to a repo it does not control, whose
  maintainers are still actively revising scope (twice in one day, per the
  asset map).
- The loop's accuracy has not been independently verified by this program as
  of this ADR — see the Verification note above. If WP1's direct read of
  LatentMesh ADR-009 contradicts the asset map's summary, every downstream
  PIR ADR that assumes this loop shape needs re-review.
- Adds an explicit dependency: no PIR work package that assigns cross-repo
  ADR numbers may proceed until the LatentMesh coordination channel exists.

## Security / Validation Gates

- **Cross-repo ADR-numbering discipline**: every reference to a non-`ruvector`
  ADR in any PIR document must name the owning repo. This is a documentation
  gate, not a code gate, but it is treated as a blocking requirement for WP1
  sign-off given the asset map's documented ADR-103 collision precedent.
- **Hosted-RVM honesty discipline** (carried from ruvector ADR-285): any
  claim this program makes about LatentMesh's or RVM's isolation/verification
  strength must match what has actually been tested, not what is aspired to.

## Affected Repos

- `ruvnet/LatentMesh` (primary — architecture spine, coordination target)
- `ruvnet/ruvector` (RuVector, RVF, RVM, ruvLLM components named in the loop)
- `ruvnet/rvm` (witness/capability enforcement referenced by the loop)
- `ruvnet/autogenous` (governance role named in the loop)
- `ruvnet/RuView` (sensing role named in the loop)

## Dependencies

None — this is the foundational ADR for the PIR program. ADR-306, ADR-307,
ADR-309, and ADR-312 each depend on this ADR's adoption decision.

## Alternatives Considered

- **Design a new cross-mechanism architecture from scratch**: rejected — the
  asset map's headline finding is that this would duplicate work already done
  one day earlier in a sibling repo, and would create exactly the kind of
  cross-repo architectural conflict this ADR is written to avoid.
- **Fork LatentMesh ADR-009's loop into a `ruvector`-local copy instead of
  coordinating with LatentMesh**: rejected — forking invites drift between
  the two repos' understanding of the loop, which is the same failure mode
  the asset map warns against for the RVM/Autogenous witness-crate
  duplication (see ADR-312).
