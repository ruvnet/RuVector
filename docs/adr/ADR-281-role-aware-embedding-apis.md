# ADR-281: Role-Aware Embedding APIs for Asymmetric Retrieval

- **Status**: Proposed
- **Date**: 2026-07-29
- **Deciders**: RuVector Architecture Team
- **Related**: ADR-194, ADR-210, issue #694
- **Tags**: embeddings, retrieval, bge, e5, qwen, api-contract

## Context

RuVector's central Rust abstraction is:

```rust
pub trait EmbeddingProvider {
    fn embed(&self, text: &str) -> Result<Vec<f32>>;
    fn dimensions(&self) -> usize;
    fn name(&self) -> &str;
}
```

That contract is sufficient for symmetric encoders such as MiniLM. It is not
sufficient for asymmetric retrieval models:

- E5 distinguishes `query: ` from `passage: `.
- BGE applies a retrieval instruction to queries and no instruction to
  passages.
- Qwen3-Embedding uses task and role instructions for retrieval.

`LatticeEmbedding` already provides an inherent `embed_query` method while
its trait implementation uses passage semantics for `embed`. Once the
provider is stored as `Arc<dyn EmbeddingProvider>`, the query method is no
longer available. High-level search code can therefore call `embed` for both
indexing and querying, producing valid-looking vectors with degraded
retrieval quality and no error.

The TypeScript model registry already carries parts of the same knowledge,
but naming and enforcement differ across Rust, N-API, WASM, and TypeScript.
Correctness must not depend on callers knowing which concrete provider is
behind a trait object.

## Decision

Embedding role is a first-class API concept. Every high-level retrieval path
must explicitly request either query or passage semantics.

### 1. Canonical role, policy, and embedding-space identity

The shared concepts are:

```rust
pub enum EmbeddingRole {
    Query,
    Passage,
}

pub enum EmbeddingRolePolicy {
    Symmetric,
    Asymmetric,
}

pub struct EmbeddingSpaceIdentity {
    pub schema_version: u16,
    pub provider: String,
    pub model_id: String,
    pub model_artifact_sha256: [u8; 32],
    pub model_graph_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    pub prompt_template_sha256: [u8; 32],
    pub pooling_strategy: PoolingStrategy,
    pub normalize: bool,
    pub truncation_tokens: u32,
    pub output_dimension: u32,
    pub output_dtype: OutputDtype,
    pub runtime_revision: String,
    pub distance_metric: DistanceMetric,
    pub role_policy: EmbeddingRolePolicy,
    pub prefix_policy: PrefixPolicy,
    pub prefix_policy_version: u32,
}
```

The TypeScript role and policy equivalents are the string unions:

```ts
type EmbeddingRole = 'query' | 'passage';
type EmbeddingRolePolicy = 'symmetric' | 'asymmetric';
```

`EmbeddingRole` describes the current operation. `EmbeddingRolePolicy`
describes whether the loaded model requires the distinction.

`EmbeddingSpaceIdentity` is the canonical provenance contract shared by this
ADR, ADR-280, and ADR-282. It identifies the complete vector space, not just
the marketing name of a model. The prompt hash covers the canonical query and
passage template bundle, including whitespace and separators. Pooling and
distance enums use stable wire values; custom strategies include an
implementation revision.

All fields are present when hashing. Strings are UTF-8 and normalized to NFC.
The deterministic identity is:

```text
embedding_space_id =
  SHA-256(
    "ruvector.embedding-space.v1\0"
    || RFC8785_JCS(EmbeddingSpaceIdentity)
  )
```

Hashes use lowercase hexadecimal in JSON and raw 32-byte values in RVF.
`embedding_space_id` governs RVF corpus compatibility, embedding caches,
research manifests, benchmark identity, and re-embedding decisions.

Changing only a query prompt changes `prompt_template_sha256` and therefore
changes the identity. This is intentionally conservative even when stored
passage vectors happen to be byte-identical: query and passage behavior
together define the retrieval space.

### 2. Make the model registry and constructor authoritative

Registered models are keyed by exact artifact revision, not model family
name. Each registry entry pins the complete expected
`EmbeddingSpaceIdentity`, including model/graph and tokenizer hashes, exact
query and passage templates, pooling, normalization, truncation, dtype,
runtime revision, distance metric, and role/prefix policies.

Provider construction computes identity fields from the loaded artifacts and
compares them with the registry entry. Construction fails if the provider's
capabilities, templates, or hashes disagree. BGE, E5, and Qwen behavior is
never inferred from a family-name substring.

Custom models are allowed only with an explicit, complete identity supplied
by the caller. There is no default `Symmetric` classification for an unknown
provider.

### 3. Make `embed_for` authoritative

The Rust trait is inverted so implementations cannot silently inherit query
behavior from a generic passage method:

```rust
pub trait EmbeddingProvider: Send + Sync {
    fn embed_for(
        &self,
        role: EmbeddingRole,
        text: &str,
    ) -> Result<Vec<f32>>;

    fn embedding_space(&self) -> &EmbeddingSpaceIdentity;

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(EmbeddingRole::Query, text)
    }

    fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(EmbeddingRole::Passage, text)
    }

    #[deprecated(note = "use embed_query or embed_passage")]
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        metrics::record_legacy_embed(self.embedding_space());
        self.embed_for(EmbeddingRole::Passage, text)
    }

    fn dimensions(&self) -> usize;
    fn name(&self) -> &str;
}
```

Implementations must handle both enum variants. An asymmetric provider that
cannot produce one role returns `EmbeddingRoleUnsupported`. A legacy
symmetric provider is migrated through an explicit
`SymmetricEmbeddingAdapter` whose constructor requires a complete identity;
it is never classified implicitly.

The deprecated `embed` wrapper preserves a migration path while making
legacy instrumentation implementable. Direct legacy calls pass through that
wrapper; intentional `embed_passage` calls do not.

Batch APIs mirror the authoritative single-item contract:

```rust
embed_batch_for(role, texts)
embed_query_batch(texts)
embed_passage_batch(texts)
```

They must not strip the role or route through generic `embed`.

### 4. Providers own prompt and prefix application

Callers provide raw user text. The exact model registry fixture owns all
model-specific instructions.

- A pinned MiniLM fixture may declare identical query and passage templates.
- A pinned E5 fixture carries its exact query and passage templates.
- A pinned BGE fixture carries its exact retrieval instruction and passage
  template.
- A pinned Qwen3-Embedding fixture carries its exact task and role templates.

Prefixes are applied exactly once. High-level databases, CLIs, and SDKs must
not prepend them independently. The policy version and prompt hash change
whenever template text or role behavior changes, even if model weights do
not.

### 5. Retrieval boundaries make the role unambiguous

The following operations always use passage semantics:

- document, memory, node, or catalog ingestion;
- index rebuild and re-embedding;
- stored corpus batch embedding; and
- any vector persisted for later query-to-document retrieval.

The following operations always use query semantics:

- text-to-vector search;
- filtered text search;
- RAG retrieval;
- agent-memory recall from a user/task query; and
- evaluation queries in retrieval benchmarks.

Similarity utilities comparing two peer texts may use a symmetric-only API.
They reject asymmetric providers unless the caller explicitly assigns roles.

Internal APIs do not accept `Option<EmbeddingRole>` at a retrieval boundary.
Absence of a role would recreate the silent-default failure this ADR closes.

### 6. Cross-language API alignment

Rust:

```text
embed_query
embed_passage
embed_for
```

TypeScript/N-API/WASM:

```text
embedQuery
embedPassage
embedFor
```

The generic `embed` method remains an alias for passage embedding during the
compatibility period. Documentation and examples use the explicit names.

Serialized requests use the stable wire value `role: "query" | "passage"`.
Unknown roles are rejected.

### 7. Provenance prevents incompatible mixing

RVF stores and research manifests carry the complete
`EmbeddingSpaceIdentity` and its deterministic `embedding_space_id`.

- Passage vectors produced under different identities may not be inserted
  into the same store.
- On identity mismatch, text embedding operations and corpus mutation are
  disabled. Existing vector-only queries, inspection, verification, and
  export remain available. An explicit migration operation may read source
  text and write a new corpus under the new identity.
- Query vectors are not persisted as corpus vectors unless explicitly marked
  as a separate vector space.
- A cache key includes `embedding_space_id`, role, and text hash.

Dimension equality alone is never treated as proof of embedding
compatibility.

### 8. Observability makes role visible

Embedding statistics and debug traces report:

```text
provider
model_id
embedding_space_id
role
role_policy
prefix_policy
policy_version
```

They never log raw user text or full model prompts by default. A metric counts
role errors. Legacy generic-`embed` calls are counted at the trait wrapper and
public SDK boundaries, not inferred inside `embed_passage`.

## Migration

1. Add the canonical identity, exact registry validation, and deterministic
   hash.
2. Introduce authoritative `embed_for` plus the explicit legacy symmetric
   adapter.
3. Change AgenticDB and retrieval consumers to call `embed_query` for search
   and `embed_passage` for ingestion.
4. Expose aligned N-API, WASM, and TypeScript methods.
5. Update examples and mark generic `embed` as passage-side compatibility.
6. After two minor releases, remove generic `embed` from
   retrieval-oriented high-level APIs. The deprecated provider wrapper may
   remain for ecosystem migration.

## Acceptance Criteria

1. Provider construction fails when loaded artifacts or declared
   capabilities disagree with the exact registry fixture.
2. The same spy proves ingestion and re-index paths invoke `embed_passage`.
3. A spy asymmetric provider proves every text-search path invokes
   `embed_query`; an asymmetric provider that lacks query support returns
   `EmbeddingRoleUnsupported`; it never falls back to `embed`.
4. A pinned symmetric MiniLM fixture produces query and passage vectors with
   `1 - cosine <= 1e-7` and maximum component difference `<= 1e-6` in one
   backend.
5. Each pinned BGE, E5, and Qwen registry fixture produces the vectors
   expected for its exact artifact revision and template hashes; no
   family-name inference is used.
6. Prefixes are applied exactly once in single and batch APIs.
7. Trait-object use (`Arc<dyn EmbeddingProvider>`) preserves both roles.
8. Rust, N-API, WASM, and TypeScript fixtures achieve
   `1 - cosine <= 1e-5` for the same identity, role, and text, with backend-specific
   component tolerances recorded in the fixture. The top-k retrieval fixture
   must also return the same ordered identifiers, allowing documented
   tie-equivalence.
9. Cache entries never cross `embedding_space_id` or role boundaries.
10. A store refuses corpus mutation when embedding-space identity differs
    while preserving vector-only read, inspection, verification, and export.
11. Retrieval benchmarks fail configuration validation if query text is
    routed through passage embedding.
12. Given one corpus, changing only the query template while retaining the
    model ID produces a new `embedding_space_id`, rejects cache reuse,
    disables text embedding and corpus mutation, preserves vector-only reads,
    and requires a new experimental revision under ADR-282.

## Consequences

### Positive

- Asymmetric retrieval correctness survives trait objects and SDK boundaries.
- High-level callers no longer need concrete-model knowledge.
- Missing query support becomes an explicit error.
- Cache and persistence provenance reflect the actual embedding space.
- Model switches among MiniLM, BGE, E5, and Qwen become safer.

### Negative

- Provider and SDK surfaces grow additional methods.
- Retrieval consumers must classify embedding call sites during migration.
- Existing caches and stores without embedding-space identities require conservative
  compatibility handling.
- Cross-language conformance fixtures add CI cost.

## Alternatives Considered

- **Keep `embed_query` only as an inherent concrete-provider method**:
  rejected because it disappears behind trait objects.
- **Have callers prepend model instructions**: rejected because it duplicates
  model knowledge, invites double-prefixing, and cannot be versioned safely.
- **Make `embed` mean query for search callers**: rejected because the same
  method is also used for stored documents and would merely reverse the bug.
- **Automatically infer role from text or call stack**: rejected as
  non-deterministic and impossible to audit.
- **Keep source-compatible defaults on the provider trait**: rejected because
  an asymmetric provider can forget to override the default and silently
  become symmetric. The explicit adapter is the migration boundary instead.

## Implementation Surfaces

- `schemas/embedding-space-identity-v1.json` (canonical cross-language schema)
- `crates/ruvector-core/src/embeddings.rs`
- `crates/ruvector-core/src/agenticdb.rs`
- `examples/ruvLLM/src/embedding.rs` and N-API bindings
- `npm/packages/ruvector/src/core/embedding-provenance.ts`
- `npm/packages/ruvector/src/core/onnx-embedder.ts`
- RVF embedding provenance from ADR-280
