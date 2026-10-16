# ADR-GL-003: Host capabilities, deterministic budgets and trust boundaries

**Status**: Implemented, validation recorded in the acceptance report
**Date**: 2026-10-16

## Assets and threat model

Protect host authority, bounded CPU/memory use, evidence provenance and model
identity from malformed or adversarial programs and artifacts. The program is
untrusted. Snapshot sources, inference rules, model admission and capability
grants are trusted host decisions. The CLI operator owns the local context file;
a network service must authenticate its caller and construct Context itself.
Accepting arbitrary client-supplied Context as authority would defeat the model.

## Decision

Expose only graph.read, model.predict and decision.emit in version 1. Static
checking requires appropriate declared effects. Runtime checks all declarations
against host-owned grants before execution. A program cannot grant itself an
effect. The runtime has no shell, filesystem, network or ambient-clock access.
External side effects and privileged action dispatch are not exposed.

The effective deterministic work budget is the smaller of program budget and
host max_steps. Validation, scans, expansions, vector arithmetic, inference and
prediction work consume budget. Exhaustion is an error, not a successful partial
answer or unknown knowledge state. Native predictors preflight their cost and
the runtime verifies returned metadata and charges reported work. Custom Rust
Predictor implementations are trusted host code, not a sandbox boundary.

Hard limits bound source size, nesting, program steps, dimensions, result count,
snapshot records and retained intermediate data. Model loading has additional
table and training limits. A work unit is not a millisecond; deployments must
also impose process deadlines and memory limits. The SDKs use process timeouts.

Strict JSON rejects duplicate keys at every decoded object. The frontend rejects
unknown fields and versions. Nonfinite floats, invalid vector shapes and invalid
identifiers are rejected rather than normalized into a plausible answer.

## Trust boundaries

| Boundary | Enforcement | Residual responsibility |
|---|---|---|
| Source to IR | Bounded parser and type/effect checker | Host chooses supported language version |
| Snapshot to execution | Shape/referential validation and content hash | Host authenticates data sources |
| Model to prediction | Bounded tables, pinned hash/version, exact scoring | Host decides whether model is suitable |
| Runtime to SDK | Structured JSON, no shell invocation | Executable path must be trusted |
| Receipt to consumer | SHA256 integrity and replay | Hashes do not establish authenticity |

## Acceptance

Tests include duplicate/unknown fields, forward references, oversized and deeply
nested values, missing capabilities, budget exhaustion, malformed provider output
and attempted prediction-to-evidence conversion. A failed run emits no receipt.
No secrets or credential access are needed for the acceptance suite.

Typed effects and authority confinement have prior art, including
[Hard-KLAIM](https://doi.org/10.1016/j.jlamp.2026.101191). The implementation does
not claim that paper's formal proof or confinement guarantee.
