# ADR-GL-004: Native RuVector adapters and explicit model lifecycle

**Status**: Implemented, validation recorded in the acceptance report
**Date**: 2026-10-16

## Decision

The native language package imports real ruvector-graph and ruvector-kge crates.
Graph capture requires the caller to hold its writer gate: GraphDB exposes
concurrent individual operations, not a language-level multi-object snapshot
transaction. The adapter's quiescence precondition is explicit. The language
snapshot supports one label per node and binary edges; unsupported graph data
is rejected rather than silently dropped.

The KGE provider uses RuVector Tables and HolE/RotatE scorers. It performs exact,
bounded exhaustive candidate ranking with deterministic tie order. Version 1
does not use ANN or assert calibrated probability. Predictions carry model name,
revision, content hash, exactness and charged work. Predictions may refer to an
entity in the model vocabulary that has no node in the snapshot; they remain
tagged predictions and cannot become observations or graph nodes implicitly.

Model artifacts are inert data. The offline model tool separates training,
evaluation and explicit promotion. Training creates a candidate, not a change
to a live registry. Split boundaries and measured metrics belong to the artifact
and report. Promotion is a host decision against declared acceptance conditions.
Randomly initialized weights are not described as learned or accurate.

The core binary is rvg-core and fails if a model provider is requested. The
native rvg binary supplies the real provider. Both share one parser, runtime and
CLI. TypeScript and Python clients launch the selected executable with argument
arrays and shell disabled. They perform no installation or network activity.

## Alternatives

Embedding Python or JavaScript evaluation into the runtime would widen the
authority boundary and duplicate evidence semantics. Direct native bindings
could reduce process startup cost but add build and ABI complexity. Process
bindings are selected for the initial version and their overhead is reported
separately from in-process execution.

## Acceptance

Native tests compare scorer output with direct RuVector APIs, capture a real
GraphDB and execute a language program against it, reject incompatible model
shapes and unsupported graph content, and exercise the offline model lifecycle.
These validate software contracts, not accuracy on a customer dataset. Published
KGE quality or cross-graph generalization requires separate held-out experiments.
