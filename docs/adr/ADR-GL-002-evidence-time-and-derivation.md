# ADR-GL-002: Temporal evidence and bounded derivation

**Status**: Implemented, validation recorded in the acceptance report
**Date**: 2026-10-16

## Decision

Snapshots contain nodes, binary edges, observations and registered rules.
Observations carry an immutable ID, subject, predicate, explicit Boolean value,
source, observation time and expiry. The host supplies the evaluation clock.
Evidence is live when observed_at <= now < expires_at and its age is within the
program's requested window. Future observations and expired evidence do not
support a conclusion. Unknown is distinct from explicit negative evidence.

Assess classifies each selected subject/predicate domain as supported, refuted,
unknown or conflicting. A domain remains present even if no live observation
matches. Both signs produce conflicting, not an arbitrary winner. These states
refer to admitted evidence, not independently established empirical truth.

Infer applies explicit same-subject signed Horn rules to a finite fixed point.
Every required predicate/value must have support. An absent predicate is never
treated as false. Rules may explicitly conclude false, but a failed positive
rule does not refute its conclusion. Cycles without premises produce nothing.
Inference returns input evidence plus supported derived facts and registers
rule conclusion domains so unsupported conclusions can be reported unknown.

Input observations retain their separate witnesses. A newly supported signed
fact retains one canonical derived witness, selected by deterministic ordering.
The witness preserves the union of original observation IDs, sources and rule
IDs. Its expiry is the earliest premise expiry, including the observation
freshness deadline. This is a bounded witness, not exhaustive proof enumeration.
Two sources may be correlated; no independence or probability is inferred.

## Retraction and consistency

Each execution validates and hashes an immutable snapshot and recomputes its
result. Deleting an observation or changing the clock removes its support on
the next execution. An alternative surviving witness may then be selected.
No incremental cache is used in version 1. This pays validation and hashing
cost on every run but makes stale-cache behavior impossible within the core.
Snapshot revision labels are not trusted content hashes.

A future incremental engine must be differential-tested against full snapshot
recomputation under insertion, deletion, expiry, conflict and model replacement.

## Acceptance

Tests cover all four outcomes, boundary timestamps, contradictory sources,
multi-premise rules, recursive chains, unsupported cycles, deletion and alternate
witnesses. No learned score may enter this evidence domain through type coercion.

## Prior art

[Scallop provenance](https://www.scallop-lang.org/doc/probabilistic/provenance.html),
[DeepGraphLog](https://arxiv.org/abs/2509.07665), and
[Differential Dataflow](https://github.com/TimelyDataflow/differential-dataflow)
inform the design; this implementation is exact Boolean evidence reasoning,
not their differentiable or probabilistic reasoning semantics.
