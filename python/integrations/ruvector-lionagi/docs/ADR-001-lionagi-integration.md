# ADR 001: Native RuVector retrieval in LionAGI

Status: implemented candidate, subject to integrated validation.

## Context

LionAGI 0.35.3 supplies a structural MemoryStore protocol and callable Tool registration. A native vector backend can add semantic retrieval while preserving ordinary Branch orchestration. The inspected source revision is f2663bcf33183e512a9a35b3f6dda8a507f867d8 in ohdearquant/lionagi. Published SDK behavior is tested separately from source inspection.

## Decision

Ship an independently versioned ruvector-lionagi distribution under python/integrations. Depend on the published RuVector Python SDK and LionAGI public interfaces. Keep embeddings supplied by the host through an async callback. The integration uses a real Rust HNSW index and bounded Python validation, with explicit thread offloading for synchronous native calls. Keep mutation ownership inside the store.

Offer Node retrieval, a read-only retrieval Tool, and LionAGI MemoryStore integration. Preserve UUIDs as native string IDs. Reconstruct only known fixed classes through explicit constructors, never arbitrary class names from serialized metadata. Strings are the supported content type for this first release.

Keep this distribution independent of the active top-ten portfolio release to avoid changing its reviewed manifest or pinned supervisor commit. Reuse the repository's protected PyPI environment only after this integration's independent release gates pass.

## Alternatives

A Python-only replacement would lose the native backend and its measured performance characteristics. A provider-specific embedding integration would introduce credentials, network behavior and model assumptions into the storage contract. Modifying LionAGI Branch internals would couple unrelated lifecycles and make maintenance harder.

## Consequences

Users supply suitable embeddings and application access control. Approximate search must report measured recall. Resource limits bound individual stores but do not establish a multitenant security sandbox. Read-only tool exposure does not prevent prompt injection through retrieved text; applications retain tool governance and content trust policies. Initial compatibility is intentionally narrow and broader releases need their own tests.
