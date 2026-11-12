# ADR-310: Causal-Attribution Gate for Latent Communication

- **Status**: Proposed
- **Date**: 2026-08-19
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ADR-309 (PIR, depends on); ADR-311 (PIR, downstream consumer); ruflo PR #2956 (anytime-valid sequential-evidence mechanism); ruflo ADR-381 (Proposed — stream identity + budget-exhaustion recovery over that mechanism)
- **Tags**: pir, causal-audit, ci-gate, latent-communication, security

## Context

Invariant 3 of the program requires: *"every agent communication is
attributable."* Per
`docs/research/perpetual-intelligence-runtime/01-evidence-review.md`, this is
grounded in a grade-A source, cited here with a corrected date:

- **Causal audit of latent multi-agent communication** —
  [arXiv:2607.26773](https://arxiv.org/abs/2607.26773), "Do Latent Channels
  Actually Communicate? A Causal Audit of Latent Multi-Agent LLM
  Communication." **Submission date: 2026-07-29** (arXiv's own timestamp).
  The program brief's original citation of "Aug 5" is incorrect and is not
  used anywhere in this program's documents. Grade A. The paper's core
  finding: end-task performance gains alone do not establish that a receiver
  actually used task-relevant information passed through a latent (KV-cache)
  channel — a controlled-replacement causal audit is needed to distinguish
  genuine communication from confounded correlation. Tested on Qwen3-4B/8B
  over GSM8K, ARC-C, and MATH-500.

This paper is the evidentiary basis for invariant 3: without a causal audit,
"agent A's latent message caused agent B's behavior change" is exactly the
kind of unfalsifiable claim the paper's methodology exists to test.

## Decision

Every latent-channel deployment inside this program's Latent Communication
Fabric context (built per ADR-309) must pass a controlled-replacement causal
audit, modeled on arXiv:2607.26773's methodology, before its performance
claims can justify further rollout. Concretely:

1. This becomes a **required CI gate** on any pull request touching
   `latentmesh-align` (or its `ruvector`-side consumers built under ADR-309),
   the anomaly-quarantine module (ADR-311), or any downstream consumer of
   latent-channel output — not a one-time paper reproduction run once and
   forgotten.
2. The audit runs against benchmark families equivalent in structure to the
   paper's own (GSM8K/ARC-C/MATH-500-style internal tasks), using a
   controlled-replacement methodology: replace the latent channel's content
   with a control (unrelated or null) payload and confirm the receiving
   agent's behavior changes accordingly when the real payload is restored.
3. Statistical claims produced by this gate (e.g. "the latent channel's
   effect on receiver behavior is significant") follow ruflo PR #2956's
   anytime-valid e-process scheme (`α_k = α_total·6/(π²k²)` per-test
   allocation, so `Σα_k = α_total` holds across arbitrarily many
   adaptively-chosen PRs) when the gate runs repeatedly over time, rather
   than treating each PR's audit as an independent, uncorrected significance
   test. Ruflo ADR-381 (Proposed) is the governance layer over that
   mechanism — per-project stream identity and an audited,
   `confirm: true`-gated evidence-epoch reset for budget exhaustion. **The
   false-promotion bound this composition provides is per-epoch, not
   global**: after a reset, the guarantee is family-wise false-promotion
   probability ≤ `α_total` for that epoch, and this gate's own statistical
   claims must be stated the same way.
4. A CI failure on this gate blocks merge; it does not merely warn.

## Consequences

### Positive

- Gives invariant 3 an enforcement mechanism that runs on every relevant
  change, not just once at program kickoff — a latent channel that "worked"
  at audit time but silently stops actually communicating (only correlating)
  after a later change would otherwise go undetected.
- Reuses a peer-reviewed, grade-A methodology instead of inventing an
  in-house causal-attribution test from scratch.
- Composing with ruflo PR #2956's sequential-evidence scheme (governed by
  ADR-381) keeps the gate's statistical claims sound, per epoch, even as it
  runs many times over the program's lifetime, rather than accumulating
  false-positive risk across repeated single tests.

### Negative

- A controlled-replacement causal audit is more expensive to run than a
  simple end-task performance comparison; this adds CI latency to every PR
  touching the gated modules.
- The gate's benchmark families are internal analogues of GSM8K/ARC-C/
  MATH-500, not the exact published benchmarks — fidelity to the paper's
  original methodology needs independent validation before the gate's
  results are treated as equivalent-strength evidence to the paper's own.
- Gating on this gate alone does not itself catch anomalous or malicious
  latent payloads — that is ADR-311's scope; a channel can pass causal
  attribution and still carry a tampered payload.

## Security / Validation Gates

- **Causal-audit CI gate** (this ADR's core mechanism): blocking, not
  advisory, on every PR touching the scoped modules.
- **Witness-chain requirement**: every audit run and its verdict are
  witness-logged (this program's shared anchoring contract, ADR-312), so
  the audit history for any given latent-channel change is queryable
  end-to-end.
- **Sequential-evidence discipline**: statistical claims from repeated gate
  runs follow ruflo PR #2956's e-process scheme (governed by ADR-381) rather
  than compounding uncorrected significance tests, and are stated as
  per-epoch bounds, not global ones.

## Affected Repos

- `ruvnet/ruvector` (CI workflow, the new LatentMesh-integration crates from ADR-309)
- `ruvnet/LatentMesh` (`latentmesh-align`, if the gate is also adopted upstream — coordination item, not required by this ADR)

## Dependencies

Depends on ADR-309 (there is no latent-channel code to gate until the
greenfield transport/RVF-packaging work exists). ADR-311 (anomaly
quarantine) is a downstream consumer — quarantine decisions are informed by,
but distinct from, this gate's causal-attribution verdict.

## Alternatives Considered

- **A one-time causal-audit reproduction at program kickoff, not a
  standing CI gate**: rejected — a static audit cannot catch regressions
  introduced by later changes to `latentmesh-align` or its consumers, which
  is exactly the failure mode invariant 3 is meant to prevent on an ongoing
  basis.
- **Use end-task performance improvement alone as the attribution signal**:
  rejected — this is precisely the confounded-correlation failure mode
  arXiv:2607.26773 documents; performance gains do not establish that the
  latent channel itself was the causal mechanism.
