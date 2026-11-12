# ADR-311: Anomaly Quarantine for Latent Channels (Net-New Work — Not "LATTE")

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-309, ADR-310 (PIR, depends on both); ADR-312 (PIR, shares provenance chain)
- **Tags**: pir, quarantine, latent-communication, security, net-new

## Context

**"LATTE" is not a real paper.** Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md` item 6,
this program's original brief cited "LATTE" as prior art for anomaly
quarantine of multi-agent latent communication. An extensive search effort —
direct phrase search across all major combinations — found eight or more
unrelated papers using "LATTE" as an acronym (a latent diffusion transformer
for video, atomic environment descriptors, hyperbolic Lorentz attention for
EEG, a robotics trajectory transformer, quantum error-correction decoding,
federated test-time adaptation, bank-transaction embeddings, linear-time
attention) — **none of which relate to multi-agent latent-communication
anomaly quarantine.** No paper under this name, in this domain, was found to
exist. This ADR states that explicitly and permanently: **any future
reference to "LATTE" as prior art for this quarantine mechanism is
incorrect, and this ADR's decision must never be described as "implementing
LATTE."**

The closest genuine match on topic is grade-graded separately in the
evidence review:

- [arXiv:2606.28958](https://arxiv.org/abs/2606.28958), "When Latent Agents
  Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration" — confirmed to
  use an HMAC-SHA256 manifest-based integrity/tamper-detection scheme (774
  honest payloads accepted, 295 tampered rejected in the paper's own
  evaluation). This is related in spirit — it is about detecting
  compromised latent-channel payloads — but it is a **different mechanism**:
  cryptographic integrity verification via HMAC manifest, not statistical
  anomaly quarantine, and it does not use the name "LATTE" anywhere.

**This ADR's quarantine mechanism is original work informed by
arXiv:2606.28958, not an implementation of "LATTE" or any other paper.**
Where this program needs statistical/behavioral anomaly detection beyond
what a cryptographic integrity manifest alone provides, that half of the
mechanism has no external prior art to lean on and is being built for the
first time by this program.

## Decision

Build anomaly quarantine for latent-channel updates as a first-party
contribution, combining two elements this program can cite concretely:

1. **Integrity verification**: adopt arXiv:2606.28958's HMAC-manifest
   approach for detecting tampered latent-channel payloads — a manifest
   binds a payload to an expected hash, and a mismatch is rejected before
   the payload reaches a receiving agent.
2. **Provenance chaining**: bind quarantine decisions into this program's
   shared witness/anchoring contract (ADR-312), specifically anchoring
   quarantine verdicts through `rvm-witness`/autogenous `witness` provenance
   chains, so a quarantine decision is auditable alongside the rest of the
   mutation/promotion history.
3. **Statistical anomaly detection** (the genuinely novel half): design and
   implement statistical/behavioral anomaly scoring for latent-channel
   updates that HMAC integrity checking alone cannot catch (e.g. a
   cryptographically valid but behaviorally anomalous payload from a
   compromised-but-still-signing agent). This is explicitly documented, in
   this ADR and in every derived design document, as original work with no
   existing paper to validate it against — its effectiveness must be
   established empirically by this program's own test suite, not cited from
   a source that does not exist.
4. Every quarantine ADR, design doc, code comment, or status report produced
   by this program must describe this mechanism as "net-new anomaly
   quarantine informed by arXiv:2606.28958," never as "LATTE" or an
   implementation of it.

## Consequences

### Positive

- Gives the quarantine mechanism a concrete, citable foundation for its
  integrity-verification half (arXiv:2606.28958's measured 774/295
  accept/reject split) rather than an unsupported claim.
- Being explicit that the statistical-anomaly half is novel work sets
  accurate expectations for validation effort — this program budgets time
  to build and test an anomaly-scoring mechanism, not to integrate an
  existing one.
- Prevents a specific, identified credibility risk: an ADR or status report
  citing "LATTE" as if it were real prior art would be discovered as false
  by any reviewer who searches for the paper, undermining trust in this
  program's other, genuinely grade-A-cited claims.

### Negative

- The statistical-anomaly-detection half has no published baseline to
  benchmark against — this program must define its own success criteria and
  validation methodology from scratch, which is a real, unbounded design
  task rather than a scoped integration.
- Combining a cryptographic mechanism (HMAC manifest, deterministic) with a
  statistical mechanism (anomaly scoring, probabilistic) in one quarantine
  pipeline introduces two different failure/false-positive regimes that need
  separate tuning and separate testing.
- This work package (WP7, per `03-program-plan.md`) explicitly depends on
  ADR-309 and ADR-310 both landing first — there is no latent-channel
  transport to quarantine, and no causal-attribution baseline to compare
  against, until those exist.

## Security / Validation Gates

- **Integrity gate**: every latent-channel payload is checked against its
  HMAC-SHA256 manifest before delivery; a mismatch quarantines the payload
  and never delivers it to the receiving agent.
- **Statistical anomaly gate**: payloads that pass integrity verification
  but score above the anomaly threshold are also quarantined, pending
  further review — the exact threshold and scoring methodology are an
  implementation detail for the owning work package, not fixed by this ADR.
- **Witness-chain requirement**: every quarantine decision (pass, integrity
  failure, or anomaly hold) is anchored via this program's shared witness
  contract (ADR-312), making quarantine history auditable.
- **Causal-audit interaction**: a quarantined payload does not count toward
  a latent channel's causal-attribution evidence (ADR-310) — quarantine and
  causal audit are complementary, not substitutes for each other.

## Affected Repos

- `ruvnet/ruvector` (new quarantine module, built on ADR-309's crates)
- `ruvnet/rvm` (`rvm-witness` provenance anchoring)
- `ruvnet/autogenous` (`witness` provenance anchoring, per ADR-312's shared contract)

## Dependencies

Depends on ADR-309 (latent-channel transport must exist before it can be
quarantined) and ADR-310 (causal-attribution gate — quarantine and causal
audit are complementary controls on the same channel). Shares its provenance
mechanism with ADR-312.

## Alternatives Considered

- **Describe this mechanism as an implementation of "LATTE"** (the
  program's original framing): rejected outright — no such paper exists;
  doing so would be a factually false citation in a permanent architecture
  record.
- **Skip statistical anomaly detection and rely on HMAC integrity checking
  alone**: rejected — integrity checking only catches *tampered* payloads
  signed incorrectly; it does not catch a validly-signed but behaviorally
  anomalous payload from a compromised agent that still holds valid signing
  credentials, which is exactly the gap statistical anomaly scoring is meant
  to close.
- **Wait for external prior art on multi-agent latent-channel anomaly
  quarantine before building this**: rejected — the evidence review already
  confirms none exists after a genuine search effort; waiting indefinitely
  is not compatible with invariant 3's enforcement needs.
