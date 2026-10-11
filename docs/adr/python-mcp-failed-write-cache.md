# Evict MCP collection state after a failed save

Status: proposed

## Context

MCP insert, batch insert, and delete mutate an in-process cached Collection before saving it. If saving raises, the tool reports failure but subsequent search/stats calls still expose the rejected changes. A retry can save those changes together with its new mutation, producing duplicate writes or persisting an unacknowledged deletion.

## Decision

Remove the collection from the cache whenever save raises, then propagate the original error. The next operation loads the persisted state. All mutators already call this helper under the existing lock, so the policy applies consistently without changing successful writes.

## Evidence and limits

Three regressions call the actual MCP SDK tools and published native index, injecting a filesystem failure before saving. Unchanged upstream fails all three while 56 existing MCP cases pass with one existing skip. Tests assert byte-identical persisted files, correct subsequent search/count, and a successful retry followed by cache-clear/reload. Full source Python tests pass 284 cases with three optional-integration skips. Native execution uses the official published macOS 0.1.2 extension; this does not prove a freshly compiled Rust wheel or other platforms.

This fixes cache exposure and retry behavior. It does not roll back a partially replaced on-disk pair, provide multi-process transactions, or guarantee durability beyond Collection.save. If loading an incomplete pair fails, the server continues reporting that failure instead of masking it with cached state.
