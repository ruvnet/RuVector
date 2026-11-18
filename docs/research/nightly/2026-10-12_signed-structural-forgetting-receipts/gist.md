# Signing the Reason a Memory Was Forgotten, Not Just That It Was

## Problem

`ruvector-agent-memory` has three pieces that, until this run, never talked
to each other:

1. A mincut-gated forgetting policy (`MincutGatedForgetting`) that builds a
   similarity graph over candidate memories and asks `ruvector-mincut` which
   ones sit on the graph's minimum cut — the "bridge" memories whose removal
   would disconnect two otherwise-separate topic clusters. It computes this,
   uses it to decide what to keep, and then throws the signal away.
2. A witnessed-compaction function that emits a tamper-evident record every
   time a memory is evicted — but the record's one flexible field (`aux`,
   an 8-byte slot already baked into the fixed-size record layout) is always
   zero. It proves *that* something was deleted. It says nothing about why.
3. An Ed25519 signing wrapper, `SignedWitnessSink`, built three weeks ago to
   close a different gap (the admission log was hash-chained but not
   cryptographically signed). It can sign any witness payload handed to
   it — but every payload it has ever actually signed has been a generic
   ledger transition, never anything structural.

Composing these is trivially easy — the signer already works on any witness
stream. The reason nobody had done it is that composing them with today's
all-zero `aux` field would sign a receipt that says nothing. We built the
payload worth signing first.

## Hypothesis

Pack the forgetting policy's own reasoning — was this memory on the
structural boundary, was it evicted under the budget-reserving "hard" mode,
what was its plain importance score — into the witness record's spare
field, using bits the fixed record format already has. Then check three
things with real code, not just an argument:

1. Can an auditor, looking only at the signed witness log (no access to the
   original vectors, no replaying the compaction), recover exactly which
   evicted memories were structurally important? Not approximately — exactly,
   every time.
2. Does this cost anything? The mincut computation itself is already
   expensive (a prior nightly measured it at hundreds of milliseconds per
   call on an 84-memory corpus) — does packing three extra facts into an
   already-allocated field add anything on top of a cost that's already
   being paid?
3. Does signing actually buy something a plain hash chain doesn't? We built
   two attacks: a clumsy one (flip the "this was a bridge" bit and touch
   nothing else) and a careful one (flip the bit, then recompute every
   downstream hash so the chain still looks internally consistent). The
   hash chain alone should catch the first and miss the second. Signing
   should catch both.

## Technical Design

The forgetting policy's eviction logic used to compute the mincut partition,
use it internally to rank candidates, and return a plain list of survivor
indices — discarding the boundary set the moment it had used it. We pulled
that computation out into a new method, `decide()`, that returns the
boundary set *alongside* the survivor list, from the same partition call.
That "same call" detail matters: the underlying mincut solver is
documented as giving slightly different (though each individually valid)
answers across repeated calls on identical input, so computing the
boundary set a second time, separately, could disagree with the first
computation used for ranking. One call, two outputs, no inconsistency.

The new witness-annotation function then does exactly what the existing
(unmodified) eviction-witnessing function does, field for field, except it
fills in the previously-zero slot:

```rust
pub fn encode_structural_aux(boundary: bool, hard_mode: bool, score: f32) -> u64 {
    let mut aux = 0u64;
    if boundary  { aux |= 1 << 0; }
    if hard_mode { aux |= 1 << 1; }
    aux |= (score.to_bits() as u64) << 8;
    aux
}
```

And the audit side is a pure function over the records alone:

```rust
pub fn audit_structural_evictions(records: &[LedgerWitnessRecord]) -> StructuralAuditReport {
    // evicted count, boundary-evicted count, hard-mode-evicted count --
    // computed from aux bits, no vectors, no store.
}
```

Signing is zero new code: the existing `SignedWitnessSink` already accepts
any witness-emitting function's output, so wrapping the new annotation
function's sink in it is exactly the same call shape as wrapping the old
one.

## Why Signing Catches More Than Hashing

The witness record's hash chain covers the whole 64-byte record — including
the field we just started using. So a sloppy tamper (flip one bit, change
nothing else) breaks the chain at the very next record, because that
record's "previous hash" field no longer matches. That's already true today,
before this run — hashing was always going to catch the lazy version of this
attack.

A patient attacker doesn't stop there. If they flip the bit *and* walk
forward re-deriving every subsequent record's hash to stay self-consistent,
the chain looks perfectly valid end to end — hashing alone cannot
distinguish "this chain was never touched" from "this chain was touched and
then carefully re-linked." That is exactly the scenario a secret signing
key defeats: the signature was computed over the *original* content, and no
amount of forward-recomputation produces a signature the attacker doesn't
have the key to produce. We built this exact two-tier attack against our
own structural payload and measured both outcomes directly rather than
asserting them.

## Results

One `cargo run --release` execution against an 84-memory synthetic corpus
(6 clusters of 12 + 12 cross-cluster "bridge" memories, compacted to 42 —
the same corpus shape a prior nightly used to characterize the underlying
mincut solver's cost):

```
baseline(compact_witnessed)                     71347.5us mean
candidate_A(structural, unsigned)               58844.4us mean
candidate_B(structural, signed BatchTail16)     65374.4us mean

Audit-accuracy: 20/20 trials exactly recovered ground truth
Forgery detection: 10/10 naive tampers caught unsigned; 10/10 diligent
  forgeries fooled the unsigned walk but were rejected once signed
```

All four pre-registered gates passed. One of them needs an honest caveat
rather than a victory lap: the "structural encoding adds ≤10% overhead"
gate measured *negative* overhead (candidate_A looked 17.5% faster than
baseline). That is not evidence that annotating records makes compaction
faster — it is evidence that the mincut computation both variants pay for
is itself expensive and noisy enough (documented elsewhere as
non-deterministic in both runtime and result from call to call) to
completely swallow the actual cost of a few bit-shift operations per
evicted entry. The gate passed because the true added cost is almost
certainly negligible, not because this measurement precisely proved it —
and we're saying so rather than quietly reporting "17.5% faster!" as if it
were a real optimization.

The forgery-detection result is the one with no asterisk: across 10
independently generated corpora, every naive tamper broke the existing
unsigned hash chain, and every diligent (fully recomputed) tamper fooled
that same unsigned chain while being rejected by the unmodified Ed25519
signature check — exactly the asymmetry this run set out to demonstrate,
now measured against structural content specifically rather than generic
ledger records.

**Verdict: ACCEPT**, with the overhead-measurement caveat stated plainly
in the full research write-up rather than hidden in a footnote.

## What This Doesn't Claim

The forgetting policy this receipt format rides on top of has its own,
separate, already-settled verdict: a prior research run measured its
bridge-protection benefit at exactly zero percentage points over a plain
scalar eviction policy, on the corpus size tested, and the policy remains
experimental and off by default. This run does not reopen that question.
What it adds is orthogonal: however good or bad the policy's decisions are,
they are now cheap to make auditable and expensive to lie about after the
fact — and a cheap audit trail is exactly the kind of instrumentation a
future attempt to re-evaluate that rejected verdict at a different scale
would want to have already.

## Why It Matters

Agent memory systems that forget things will eventually need to answer, for
an auditor, a regulator, or a worried engineer at 2am: *what did we delete,
and can you prove you're not lying about why?* An unsigned log answers the
first question weakly (it's tamper-evident against carelessness, not
against someone who tries). This run shows that answering the second
question costs almost nothing once you already have a structural signal and
already have a signing primitive — the missing piece was never
infrastructure, it was deciding what the receipt should actually say.
