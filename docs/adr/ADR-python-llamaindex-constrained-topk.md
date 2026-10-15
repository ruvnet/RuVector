# Apply LlamaIndex identity constraints before nearest-neighbor selection

## Context

VectorIndexRetriever forwards node_ids and doc_ids to VectorStoreQuery as search restrictions. RuVectorStore previously discarded disallowed node IDs only after selecting the global top-k and ignored document restrictions. A nearer disallowed node could therefore suppress every allowed result, or a query could return a node from an excluded document.

## Decision

Compose node and document restrictions with the existing metadata predicate before Collection.search selects top-k. Recover node identity through LlamaIndex's metadata_dict_to_node helper; use the ref_doc_id written by node_to_metadata_dict for document identity. None leaves a restriction absent and an empty list permits no nodes. All supplied restrictions intersect.

## Consequences

Constrained queries use the existing Python callable filtering path. Node constraints reconstruct framework nodes while evaluating candidate metadata, adding parsing/allocation cost rather than claiming native filter acceleration. Unconstrained queries retain their existing behavior. This does not change Collection's approximate search or introduce a new persistence format.

## Validation

The real llama-index-core 0.14.25 consumer is exercised with public RuVectorStore.query calls and deterministic TextNode embeddings. Four regressions fail on original source: node restriction after global top-k, ignored document restriction, combined restrictions, and empty document restriction. Fifteen controls pass on original source; all nineteen pass after the fix. The full Python package passes 300 tests with two optional-dependency skips using published 0.1.2 native bindings; no fresh native build is claimed by these results.

## Optional metadata constraints

The same retrieval boundary must retain framework filter semantics. IS_EMPTY now runs before the general null-value guard, so missing and explicit-null metadata match alongside empty strings and lists. Official llama-index-core handles IS_EMPTY separately for this reason. Real TextNode/query regressions cover missing/null values, with empty/nonempty string and list controls; this does not change NE/NIN null handling.
