# ADR-318: StagedWorkspace-Pattern Content-Hash State Binding as a RuV Invariant

- **Status**: Proposed
- **Date**: 2026-08-20
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-307 (PIR, extends — three-level persistent memory, WP3/WP4); ADR-312 (PIR, extends — shared witness schema + anchoring contract, WP8); ADR-317, ADR-319 (PIR, combined-effect acceptance test); see `docs/research/perpetual-intelligence-runtime/06-wave2-evidence-review.md` §5
- **Tags**: pir, wave-2, stagedworkspace, content-hash, rvf, witness, memory

## Context

Wave-2 evidence review grades StagedWorkspace **A** —
[arXiv:2608.18050](https://arxiv.org/abs/2608.18050), "StagedWorkspace: A
Versioned Workspace for Knowledge-Work Agents," submitted 2026-08-18. Both
claims the Wave-2 brief cited check out verbatim against the abstract:

- **Binds views to content hashes**: "The workspace binds parsed records and
  review diffs to content hashes of the native files as they change."
- **OfficeQA Pass@1 improvement of +8.3–12.1 points**: "dual parsed/native
  access has the highest point estimate for every tested model; relative to
  the more limiting single view, it improves OfficeQA Pass@1 by 8.3-12.1
  points."

**Benchmark provenance, checked**: OfficeQA is not a benchmark
StagedWorkspace introduces — it is a pre-existing, independently published
Databricks benchmark (`databricks/officeqa`, also documented separately as
"OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning,"
[arXiv:2603.08655](https://arxiv.org/abs/2603.08655)) for enterprise
document-grounded reasoning over U.S. Treasury Bulletin data. StagedWorkspace
reports results on someone else's independent benchmark, which is a point in
its favor on evidence rigor.

**Artifact availability — checked, not assumed, and the operative constraint
for this ADR's scope**: the arXiv comments field reads only "Under Review" —
no code or data link. A targeted search for a standalone "StagedWorkspace"
repository found nothing. **No code is available today.** This ADR is
therefore a from-scratch implementation of the paper's described mechanism,
not a port of a released reference implementation — the same posture this
program applied to LATTE's quarantine work in Wave 1 (ADR-311) and applies
to TRUSS in this wave (ADR-319).

`ruvector` already has the durable-artifact and provenance primitives this
pattern needs: RVF (`crates/rvf`, append-only crash-safe segments with
canonical format, ruvector ADR-029) and RVM's witness records
(`crates/rvm`, hash-chained per privileged action, ruvector ADR-134
(witness-schema-log-format)),
formalized as the transactional memory tier in ADR-307 and anchored across
layers via ADR-312's shared witness schema and cross-layer anchoring
contract. ADR-312 was designed for mutation/promotion records specifically;
it does not today generalize to arbitrary knowledge-work artifacts
(documents, spreadsheets, notebooks, workspace files) that this runtime's
agents read, parse, and write outside the mutation/promotion pipeline. That
gap is what this ADR closes.

**Preprint-reproduction rule**: StagedWorkspace's own +8.3–12.1pp OfficeQA
figure is the *paper's* result on its own benchmark and authors' own
reference implementation, which does not exist for this program to inherit.
This ADR does not adopt that figure as an expected outcome; it is
implemented as a candidate mutation and benchmarked against an internal
OfficeQA-equivalent task set, with promotion gated by ADR-306's
`research-gate`-recomputed delta, exactly like every other Wave-2 item.

## Decision

Adopt **content-hash + revision-id state binding** as a RuV invariant across
every artifact this runtime's agents produce or consume, generalizing
ADR-312's anchoring contract from mutation/promotion records to arbitrary
knowledge-work artifacts:

1. **Every artifact gets a content hash and a revision id at write time.**
   "Artifact" includes RVF-backed records (ADR-307's transactional memory
   tier), workspace files, generated documents, and any other durable output
   this runtime's agents write. The content hash is computed over the
   artifact's canonical byte representation (reusing RVF's existing
   canonical-format discipline, ruvector ADR-029); the revision id is a
   monotonic counter scoped to that artifact's lineage.
2. **Every downstream reference binds to that exact hash, not just the
   artifact's identity.** A parser result, a tool call's input, a review
   diff, an approval decision, or a generated output that reads or acts on
   an artifact must record the content hash (and revision id) of the exact
   version it read — modeled on StagedWorkspace's binding of parsed records
   and review diffs to native-file content hashes (arXiv:2608.18050),
   implemented from the paper's described mechanism since no upstream code
   exists to port.
3. **Stale state is automatically invalid.** If an artifact's live content
   hash no longer matches a downstream reference's recorded hash, that
   reference is stale by construction: any parser result, diff, approval, or
   output built against it must be recomputed against the current hash
   before it can be used or trusted. This is enforced structurally at read
   time (a hash mismatch is a hard rejection, not a warning), not left as a
   convention for callers to honor voluntarily.
4. This binding is anchored using **ADR-312's shared witness schema and
   cross-layer anchoring contract** — the content-hash/revision-id record for
   a knowledge-work artifact uses the same canonical-encoding,
   domain-separated Ed25519 signature discipline (ruflo ADR-322C, adopted
   via ADR-312) that mutation/promotion witness records already use, rather
   than inventing a second anchoring format for this class of artifact.
5. Benchmark the delta this binding produces on an internal
   OfficeQA-equivalent task set (dual parsed/native access vs. single view),
   independently measured by `research-gate`'s paired-bootstrap recomputation
   per ADR-306 — not cited as StagedWorkspace's own +8.3–12.1pp figure, which
   describes the paper's own benchmark and models, not this runtime's.

## Consequences

### Positive

- Closes a real gap ADR-312 didn't originally scope: today's anchoring
  contract only covers mutation/promotion records, leaving arbitrary
  knowledge-work artifacts (documents, spreadsheets, notebooks) without a
  structural staleness guarantee.
- Reuses RVF's existing canonical-format and ADR-312's existing
  cross-layer anchoring rather than inventing a parallel hashing/signing
  scheme for this artifact class.
- Gives every downstream consumer (parser, tool, reviewer, approver) a hard
  guarantee that what it is acting on is the exact version it thinks it is
  — directly supporting ADR-317's "successful recovery" condition and
  ADR-319's shadow-execution receipts, both of which need to know precisely
  which artifact version a given action touched.

### Negative

- No upstream reference implementation exists (StagedWorkspace's own repo is
  "Under Review," not published) — this is a genuine from-scratch build
  against the paper's described mechanism, with correspondingly higher
  implementation and validation cost than ADR-320's or ADR-321's
  port-and-adapt work.
- Structural staleness rejection is a real behavior change: any workflow
  that previously tolerated acting on a slightly-stale artifact view now
  hard-fails and must recompute — this could surface latent assumptions
  elsewhere in the runtime that quietly relied on eventual consistency.
- The internal OfficeQA-equivalent benchmark this ADR requires does not yet
  exist and must itself be built before a delta can be measured — an
  additional deliverable this ADR's own promotion gate depends on.

## Security / Validation Gates

- **Hash-binding enforcement**: every downstream reference to an artifact
  must carry that artifact's content hash and revision id; a reference
  without one is rejected at write time.
- **Stale-reference rejection**: a hash mismatch between a downstream
  reference and the artifact's current live hash is a hard rejection at read
  time, never a soft warning.
- **Cross-layer anchoring**: content-hash/revision-id records use ADR-312's
  canonical-encoding, domain-separated Ed25519 signature scheme; fail-closed
  verification (unknown fields, non-finite numbers, negative zero rejected)
  applies here exactly as it does to mutation/promotion records.
- **Benchmark-delta gate**: promotion requires a `research-gate`-recomputed
  delta on the internal OfficeQA-equivalent task set — StagedWorkspace's own
  +8.3–12.1pp figure is never cited as this program's acceptance bar.
- **Wave-2 combined acceptance test**: this ADR supplies the "RVF bound
  workspace states" component of ruv's verbatim Wave-2 acceptance criterion
  (see ADR-317's Security/Validation Gates for the full text) — its own
  promotion does not by itself satisfy that combined test, which requires
  ADR-317 and ADR-319 to land and be benchmarked together.
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan` after
  any change to the content-hashing or anchoring write path.

## Affected Repos

- `ruvnet/ruvector` only — `crates/rvf`, `crates/rvm` (extends the ADR-312
  anchoring contract), `ruvector-agent-memory`. Single-repo scope per
  `07-wave2-program-plan.md`.

## Dependencies

Depends on ADR-307 (the three-level memory tiers this binding applies to
must exist first, along with the TARL ledger) and ADR-312 (the shared
witness schema and anchoring contract this ADR's records use). No
dependency on ADR-317 or ADR-319 for its own promotion, but the combined
Wave-2 acceptance test names all three.

## Alternatives Considered

- **Bind only mutation/promotion records to content hashes, leave other
  knowledge-work artifacts unbound** (the pre-Wave-2 status quo): rejected —
  this is exactly the gap StagedWorkspace's pattern is adopted to close;
  leaving it unaddressed means parsers, tools, and reviewers can silently act
  on stale artifact state outside the mutation/promotion path.
- **Invent a new hashing/signing scheme specific to knowledge-work artifacts
  instead of reusing ADR-312's anchoring contract**: rejected — ADR-312
  already adopted ruflo ADR-322C's canonical-encoding/signature scheme
  specifically so every PIR ADR that emits witness records has one
  anchoring contract to target; a second scheme for this artifact class
  would repeat the coordination-surface cost ADR-312 was written to avoid.
- **Treat a hash mismatch as a warning rather than a hard rejection**:
  rejected — a soft warning is exactly the "aspirational, not enforceable"
  failure mode ADR-307's TARL-ledger decision already rejected for memory
  transactions; this ADR applies the same enforceability standard to
  knowledge-work artifacts.
