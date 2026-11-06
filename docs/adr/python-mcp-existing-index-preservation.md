# Preserve existing index files during MCP collection creation

## Context
A partially restored collection can have a native index but no metadata sidecar. The MCP create tool checks only the sidecar and overwrites the index with an empty collection. This destroys the available vectors instead of reporting that the name is already occupied.

## Decision
Reject creation when either the native index or its metadata sidecar exists. Use the existing collection-already-exists error. Recovery and restoration remain explicit operator actions. This follows the existing no-overwrite create contract.

## Evidence
The public MCP tool regression creates actual HNSW and RaBitQ collections, removes their sidecars, and compares stored bytes after attempting creation. Both regressions fail on unchanged upstream: HNSW index bytes are overwritten, while RaBitQ creation writes an empty replacement sidecar against the retained index. All 57 existing MCP controls pass. With the change, the full Python suite passes 284 tests with 2 optional-integration skips using current Python source, MCP SDK, and the official published 0.1.2 macOS native extension.

## Limits
This does not repair a missing sidecar or introduce atomic create across separate processes. Native-source builds and other operating systems are not covered by this local evidence.
