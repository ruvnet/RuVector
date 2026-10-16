# ADR-GL-005: Replay receipts and reproducible delivery gates

**Status**: Implemented, validation recorded in the acceptance report
**Date**: 2026-10-16

## Decision

Every successful execution produces a structured receipt containing hashes of
the program, full snapshot and context, explicit clock, work count and typed
output. Prediction output additionally identifies its model. The receipt hash
excludes its own hash field. Hashes detect changed bytes under the documented
canonicalization; they are not signatures, authentication or proof of truth.

The verify CLI checks receipt integrity only. Replay reexecutes the request
using the selected native model state, verifies the supplied receipt, and
requires full receipt equality. A changed clock, grant, snapshot, program or
model participating in the emitted predictions therefore invalidates replay.
Unused model bundles are not receipt inputs. Reusing an old revision string
cannot hide changed content. Program and property array ordering remain
significant; snapshot records and rule premises are canonically sorted.

CI runs locked core and native builds, unit/integration/adversarial tests, both
process SDK suites, packaged-client smoke tests, formatting and targeted lint.
Fixtures are synthetic and labelled accordingly. The benchmark records workload,
machine and sample counts. A measured result must never be substituted with a
target, source comment or provider success message.

## Proposed product gates

The research proposal suggested at least 30% less application code and no more
than 10% overhead relative to equivalent handwritten implementations. These are
product hypotheses, not automatically satisfied release claims. Comparing a
validated, hashed interpreter with an unchecked loop is a diagnostic, not an
equivalent-work speed contest. Report the distinction and optimize only after
correctness is established.

## Rollout and rollback

Deliver additive source, examples, docs, CI and a reviewable PR. Existing APIs
remain available. Package publication, default-branch merge and deployment are
separate release decisions. A user can install the native command from a pinned
checkout and roll back by replacing that binary and its model registry with the
previous versions. Immutable model artifacts support that rollback.

## Acceptance trace

The final trace lives in docs/research/graph-language/ACCEPTANCE.md and maps each
in-scope requirement to executed evidence. Missing gates must be named; a passing
targeted suite does not imply all unrelated repository crates pass.
