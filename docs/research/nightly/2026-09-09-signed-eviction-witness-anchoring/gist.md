# Your hash-chained audit log is lying to you (and how to catch it for ~80µs)

## Problem

RuVector's `ruvector-agent-memory` crate keeps a tamper-evident log of
every memory it evicts: an FNV-1a hash chain, one 64-byte record per
eviction, each record's hash folded into the next. It's fast (no
allocation, no crypto), and it catches the failure mode people usually
worry about — accidental corruption, a dropped write, a bit flip on disk.

It does not catch the failure mode that actually matters for an audit
log: an adversary who can write to the log. If you can mutate record #50
in a 4,096-record chain, you can also recompute records #51 through
#4,096's hashes to match, and the chain will verify perfectly. The
"tamper-evident" log is not evident to anyone who controls the storage.

This isn't a novel observation — it's exactly why Certificate Transparency
uses signed tree heads, why TPM event logs support PCR extension, and it's
already written into this repository's own witness-schema ADR from April.
What hadn't happened yet was closing the loop for this *specific* chain —
memory eviction — and measuring the actual cost of doing so, rather than
asserting it's fine.

## Hypothesis

Sign the chain head periodically with Ed25519, using a primitive the
crate already depends on (no new crypto, no new crate). Specifically:
after every `K` evicted records, sign `(sequence, chain_head)`. An
auditor who holds that signature — and *only* that signature, not trust
in the log's storage — can detect any tamper at or before the signed
point, because forging a matching signature requires the private key.

The interesting empirical questions were: does it actually catch a real
attack (not a hypothetical one), what does it cost, and how does the
cost trade off against `K`.

## Technical Design

Three pieces:

1. **The attack, implemented for real.** `relink_tampered_suffix` takes a
   slice of witness records, mutates one, and recomputes every downstream
   hash so the result still passes the existing `verify_chain()` check.
   This isn't a described vulnerability — it's a function you can call and
   watch the "tamper-evident" log stay green.

2. **The anchor.** `EvictionAnchorLog::note_record` is called once per
   evicted record. Every `K` records, it signs `(sequence, chain_head)`
   with `rvf_types::ed25519_sign` — the exact Ed25519 implementation this
   crate already uses elsewhere for per-observation signatures. Domain
   separation (`"ruvector:agent-memory:eviction-anchor:v1:"`) stops a
   signature from this chain being replayed as valid for a different one.

3. **The check.** `verify_anchor_against_chain` takes a signed anchor and
   a chain head an auditor independently recomputed at the anchor's
   sequence number, and returns whether they match *and* the signature is
   valid. An auditor who only has the log (no anchor) can't tell honest
   history from a relinked one. An auditor who also has one signed anchor
   can — for everything up to that anchor's sequence.

## Implementation

`crates/ruvector-agent-memory/src/eviction_witness_signing.rs` (new
module) and `crates/ruvector-agent-memory/examples/signed_eviction_witness_bench.rs`
(benchmark). No `Cargo.toml` changes — the Ed25519 dependency (`rvf-types`,
`ed25519` feature) was already unconditional in this crate.

## Actual Benchmark Evidence

4,096 real evicted records, real `compact_witnessed` output, release
build, 4 independent process runs:

- **Signing cost**: ~77,000-89,000 ns per Ed25519 sign. High for
  Ed25519 (which is usually tens of microseconds *total*, not per
  operation, in optimized implementations) — almost certainly because
  `rvf_types::ed25519_sign` re-derives the expanded `SigningKey` from raw
  bytes on *every call* instead of caching it. Measured, not fixed this
  run; flagged as the obvious next optimization.
- **Verify cost**: ~48,500-49,000 ns per verify, steady across runs.
- **Memory overhead** (88-byte signed anchor vs. 64-byte raw record):
  signing every record costs **137.5%** overhead — the audit trail
  literally outgrows the thing it's auditing. Signing every 16th record
  costs **8.6%**. Signing every 4,096th (once per 4,096-record chain)
  costs **0.034%**.
- **Adversarial detection**: sampled 111 tamper positions across the
  chain, ran the real relink attack at each, checked detection against
  the next anchor at or after the tamper point. **111/111 detected
  (100%)**. Zero false positives on anchors that predated the tamper (as
  expected — an anchor only authenticates what it covers). The FNV-1a-only
  baseline, checked against the same attack: **0% detected** —
  `verify_chain()` reports `true` on the fully relinked log, every time.

Acceptance was fixed in the benchmark source before it ran: 100%
detection once covered, 0 false positives. Both held, all 4 runs. Verdict:
**ACCEPT**.

## Limitations

- Tested at one corpus size (4,096 records). Ed25519 signing is O(1) per
  call regardless of chain length, so this should scale, but that's an
  inference from reading the code, not a measurement at larger N.
  - 111 sampled tamper positions, not an exhaustive sweep of all 4,096.
  The theoretical argument (can't forge a signature without the key) is
  stronger than the sample; the sample is what was actually run.
- No key-management story. Who holds the signing key, how it's rotated,
  and how an auditor gets the public key are all unaddressed — this ships
  the primitive, not a deployment.
- Not wired into any production call site. `compact_witnessed` doesn't
  call this yet.

## Production Relevance

This is not a novel cryptographic technique — periodic signed checkpoints
over a hash chain is the same idea Certificate Transparency and TPM event
logs use, and this exact repository already validated the same
interval/staleness/cost tradeoff shape for a different chain (signed
retrieval receipts) a week earlier. What's new here is closing a
specific, previously-named gap in agent-memory's deletion audit trail
with the crate's own existing crypto dependency, and getting honest
numbers on what it costs: cheap at `K≈16` (single-digit percent memory
overhead, sub-16-record staleness), expensive at `K=1` (more anchor bytes
than log bytes).

## RuVector Ecosystem Implications

Deletion is the third and last stage of `ruvector-agent-memory`'s
lifecycle to get witness coverage (admission and retrieval already have
it, in this crate and in `ruvector-retrieval-receipt` respectively). The
anchor's domain-separated statement format is also a natural fit for
RVF's "signed lineage" metadata and RVM's proof-referenced-witness
pattern — plausible, not built this run.

## Future Direction

1. Determine whether caching the expanded Ed25519 signing key (instead of
   re-deriving it from raw bytes every call, as `rvf_types::ed25519_sign`
   currently does) closes most of the gap between the measured ~80µs and
   ADR-134's <10µs software-signing budget.
2. Design the key-management story, then actually wire this into a
   `WitnessSink`.
3. Widen the adversarial sample to exhaustive or statistically-justified
   coverage.

## References

- `docs/adr/ADR-134-witness-schema-log-format.md` — the witness schema
  and the `WitnessSigner` follow-up this closes.
- `docs/adr/ADR-342-periodic-state-root-anchoring.md` — the anchor-policy
  shape this reuses, applied here to a different chain.
- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md` —
  the eviction-witness chain this signs, and the "Next Research" item this
  run addresses.
- Full methodology and all 4 raw benchmark runs:
  `docs/research/nightly/2026-09-09-signed-eviction-witness-anchoring/README.md`.
- `docs/adr/ADR-346-signed-eviction-witness-anchoring.md` — the formal
  decision record for this work.
