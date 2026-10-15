# Apply the shared collection capacity to Agentforce writes

## Context
The Python server exposes MCP and opt-in Agentforce routes backed by the same collections. Its configured per-collection vector limit is checked by MCP inserts, but Agentforce upserts append rows without consulting it. A deployment selecting a capacity limit can therefore grow beyond it through the other supported write interface.

## Decision
Use the existing _check_capacity helper under the shared collection lock before an Agentforce insertion. Its ValueError maps to the existing HTTP 400 response. Preserve successful writes up to the cap and do not change standalone in-process collection behavior.

## Evidence
Real ASGI action requests use actual HNSW and RaBitQ collections with a one-vector cap. Both accept the first row, reject the second, preserve index/sidecar bytes and cached count, then reload one row. On unchanged upstream both regressions fail with 16 existing tests passing. Fixed full Python tests using the published wheel pass 284 tests with 2 skips; the current-source native extension built at pristine upstream SHA07933730 passes 297 tests with 2 optional-integration skips including real LlamaIndex consumers.

## Limits
The cap counts physical/tombstoned rows using the same existing helper as MCP; this does not establish multi-process coordination, remote Salesforce deployment evidence, or cross-platform native compatibility.
