# ADR-063: WASM Executable Nodes — Deterministic Compute at the Edge

**Status**: Accepted
**Date**: 2026-02-27
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-062 (Brainpedia Architecture)

## 1. Context

The Shared Brain stores knowledge as embeddings and serves MicroLoRA weights. But the embedding pipeline — structured hash features, LoRA transform, L2 normalization — runs only on the server and in the local MCP client. Browser users, edge appliances, and third-party integrations cannot produce compatible embeddings without reimplementing the pipeline.

WASM solves this. A signed WASM module that implements feature extraction can run identically in the browser, on the appliance, and on the server. The same bytecode, the same results, cryptographically verified.

This ADR defines the WASM node architecture: how nodes are built, signed, served, executed, and integrated with the Shared Brain and MicroLoRA pipeline.

## 2. What WASM Nodes Enable

### 2.1 Executable Knowledge Pages

A Brainpedia entry can include a small WASM module that validates, scores, normalizes, or transforms content locally. Knowledge becomes executable, not just readable.

### 2.2 Deterministic Verification

The same scoring and feature extraction runs in browser, on the appliance, and on the server with identical results. No "works on my machine" for embeddings.

### 2.3 Safer Public Compute

WASM with capability gating lets untrusted users run brain logic without server-side code execution. Pure compute only — no filesystem, no network, no clock access.

## 3. Node Architecture

### 3.1 WASM Segment Type

WASM nodes are signed RVF artifacts. They use the existing RVF segment layout with a new segment type for WASM bytecode.

A node RVF container contains:
- `MANIFEST (0x05)`: Segment directory
- `META_SEG (0x07)`: Node name, version, description, interface schema
- `WASM_SEG (0x40)`: WASM bytecode (compiled, not WAT)
- `WITNESS_SEG (0x0A)`: SHAKE-256 hash chain (build → sign → publish)
- `CRYPTO_SEG (0x0C)`: Ed25519 signature over all segments

### 3.2 Hash Choice Consistency

Two hash functions serve distinct purposes:

**SHA-256** for HTTP-layer integrity: `X-Node-SHA256` header, `wasm_sha256` in signed manifests, GCS content-addressed storage paths, browser cache keys. SHA-256 is conventional for HTTP caching and CDN infrastructure.

**SHAKE-256** for RVF witness chains: all WITNESS_SEG entries use SHAKE-256 to maintain consistency with the global RVF witness chain convention (ADR-058). Witness chains are internal to the RVF container format, not exposed via HTTP.

These do not conflict. SHA-256 is the identity hash visible to HTTP clients. SHAKE-256 is the provenance hash visible to RVF verifiers.

### 3.3 V1 ABI Specification

The v1 ABI is `feature_extract` only. `score` and `validate` are deferred to v2 after the determinism guarantee is proven on the simpler single-function interface. Keeping v1 to one function reduces the ABI surface that must be verified across three runtimes.

**ABI version**: `1`

**Required exports**:
- `memory`: Linear memory export (must declare a maximum size, rejected if absent or > 16 pages)
- `malloc(size: i32) -> i32`: Allocate memory for input
- `feature_extract_dim() -> i32`: Return the output dimension (must be constant, e.g., 128)
- `feature_extract(in_ptr: i32, in_len: i32, out_ptr: i32) -> i32`: Extract features from text at `in_ptr`/`in_len`, write f32 vector to `out_ptr`, return dimension written

**Calling convention**:
1. Caller invokes `feature_extract_dim()` to learn output size
2. Caller allocates input buffer via `malloc(input_bytes_len)`
3. Caller writes UTF-8 text into the allocated buffer
4. Caller allocates output buffer via `malloc(dim * 4)` (f32 = 4 bytes)
5. Caller invokes `feature_extract(in_ptr, in_len, out_ptr)` → returns `dim`
6. Caller reads `dim` f32 values directly from `memory` at `out_ptr`

No global result state. No `read_f32` indirection. The caller owns both input and output pointers. This is the fastest path in both browser and wasmtime.

**Embedded lookup tables**: Nodes MAY include constant data (token maps, n-gram tables, normalization tables) compiled into the WASM data section. This is fine — the data section is part of the signed bytecode and does not violate the pure-compute constraint. The data section is immutable at runtime.

### 3.4 Determinism Constraints

"Bit-identical" is a strong claim. These constraints make it true in practice:

1. **No f64 in v1 nodes**: All arithmetic uses f32 only. f64 intermediate results are prohibited because fused multiply-add behavior varies across engines for f64.
2. **No SIMD for v1 nodes**: The WASM module must not use the SIMD proposal. SIMD instruction selection and NaN propagation differ across engines.
3. **Deterministic compilation flags**: Nodes must be compiled with `no-fast-math` (Rust: `-C target-feature=-fast-math`, or simply do not opt in). No `-ffast-math`, no `reassociate`, no `reciprocal`.
4. **NaN scrubbing before output**: The last operation before writing each f32 to the output buffer must check for NaN and replace with 0.0f32. This prevents non-deterministic NaN propagation from producing environment-dependent results.
5. **Rounding mode**: IEEE 754 round-to-nearest-even (the WASM default). No explicit rounding mode changes.

These constraints are verified at publish time where possible (SIMD detection, f64 usage detection via module inspection) and at runtime via conformance test vectors.

### 3.5 Capability Model

**V1: zero imports**. The WASM module's import section must be empty. No `env`, no WASI, no custom imports. If the import section is non-empty, the module is rejected at publish time. This is the cleanest rule for pure compute.

If future nodes require host capabilities (v2+), they are served from a separate endpoint (`/v1/nodes/privileged/`) with explicit capability declarations. Privileged nodes require reputation >= 0.7 to publish.

### 3.6 Resource Limits

Resource limits are defined differently per environment because browser Workers lack instruction-level metering.

**Server and appliance (wasmtime)**:
- Max fuel: calibrated to 25ms on reference CPU (equivalent to ~100M simple instructions)
- Max memory: module must declare maximum <= 16 pages (1MB)
- Epoch interruption as backstop

**Browser (WebAssembly.instantiate in Worker)**:
- Max wall clock: 25ms (Worker `setTimeout` termination)
- Max input size: 8KB UTF-8
- Worker isolation: `postMessage` only, no DOM, no fetch, no storage

**All environments**:
- Max WASM binary size: 1MB
- Max input size: 8KB UTF-8

The 25ms budget is aligned across environments. Server fuel is calibrated against reference CPU to approximate 25ms of compute.

## 4. Signing and Verification

### 4.1 Canonical Signed Manifest

The Ed25519 signature covers a canonical binary manifest, not raw JSON. This prevents serialization differences from breaking verification.

**Signed manifest fields** (fixed-order binary encoding):

| Field | Type | Bytes |
|-------|------|-------|
| `abi_version` | u8 | 1 |
| `id_len` | u16 LE | 2 |
| `id` | UTF-8 | variable |
| `version_len` | u16 LE | 2 |
| `version` | UTF-8 | variable |
| `dim` | u16 LE | 2 |
| `wasm_sha256` | [u8; 32] | 32 |
| `created_at` | i64 LE (Unix epoch seconds) | 8 |
| `allowed_imports_hash` | [u8; 32] | 32 (SHA-256 of empty string for v1) |
| `compiler_tag_len` | u16 LE | 2 |
| `compiler_tag` | UTF-8 | variable (e.g., "rustc-1.77-wasm32") |

The signature is `Ed25519.sign(signing_key, manifest_bytes)`. Verification is `Ed25519.verify(public_key, manifest_bytes, signature)`.

This encoding is deterministic: fixed field order, little-endian integers, explicit length prefixes. No JSON, no CBOR, no ambiguity.

### 4.2 Verification Flow

1. Client fetches node metadata (`GET /v1/nodes/{id}/{version}`)
2. Client fetches WASM binary (`GET /v1/nodes/{id}/{version}/node.wasm`)
3. Client computes SHA-256 of WASM bytes, compares to `wasm_sha256` in metadata
4. Client reconstructs manifest bytes from metadata fields
5. Client verifies Ed25519 signature over manifest bytes
6. Client runs conformance test vectors (Section 5)
7. If all pass: instantiate and use. If any fail: reject.

## 5. Conformance Test Vectors

Conformance test vectors are first-class policy, not optional documentation.

### 5.1 Test Vector Format

Every node's metadata includes a `conformance` object:

```json
{
  "abi_version": 1,
  "dim": 128,
  "test_vectors": [
    {
      "input": "deadlock tokio spawn rwlock",
      "expected_output_sha256": "a1b2c3d4..."
    },
    {
      "input": "",
      "expected_output_sha256": "e5f6a7b8..."
    },
    {
      "input": "日本語テスト Unicode edge case",
      "expected_output_sha256": "c9d0e1f2..."
    }
  ]
}
```

**20 mandatory test vectors** including:
- Empty string
- Single character
- ASCII only
- Unicode (CJK, emoji, combining characters, RTL)
- Maximum length (8KB)
- Repeated tokens
- Whitespace-only
- Mixed case sensitivity test pairs

### 5.2 Verification Method

For each test vector:
1. Run `feature_extract` with the input text
2. Read the output f32 vector as raw bytes (little-endian)
3. Compute SHA-256 of the raw output bytes
4. Compare to `expected_output_sha256`

**Output byte hash, not float comparison.** Float printing can hide differences. Raw byte SHA-256 catches any bit-level divergence.

### 5.3 Runtime Self-Test

Every runtime (browser, server, appliance) runs all conformance vectors on first load of a node. If any vector fails, the node is rejected and not used. The server logs conformance failures as security events.

## 6. Endpoints

### 6.1 Node Registry

| Method | Path | Purpose | Auth |
|--------|------|---------|------|
| GET | `/v1/nodes` | List available nodes | Public |
| GET | `/v1/nodes/{id}/{version}` | Get node metadata + conformance vectors | Public |
| GET | `/v1/nodes/{id}/{version}/node.wasm` | Download WASM binary | Public |
| POST | `/v1/nodes` | Publish a new node | Identity + Reputation >= 0.5 |
| DELETE | `/v1/nodes/{id}/{version}` | Revoke a node (original publisher only) | Identity |

### 6.2 Version Resolution

There is no `GET /v1/nodes/{id}.wasm` (no implicit "latest"). Clients must specify both `id` and `version`. This prevents supply-chain attacks where a compromised node replaces a trusted version.

The `GET /v1/nodes` listing includes all versions of all nodes. Clients filter by `id` and select the desired version.

### 6.3 Response Headers

**GET /v1/nodes/{id}/{version}/node.wasm**:
- `Content-Type: application/wasm`
- `Cache-Control: public, immutable, max-age=31536000`
- `X-Node-SHA256: {hex-encoded SHA-256 of WASM bytes}`

Immutable caching is safe because the version is in the path. New versions get new paths.

### 6.4 Revocation

`DELETE /v1/nodes/{id}/{version}` marks a node as revoked:
- Only the original publisher can revoke
- Revocation writes an audit record to the witness chain
- WASM bytes remain in storage (for forensic analysis), but the registry stops serving the node
- `GET /v1/nodes/{id}/{version}` returns 410 Gone with a revocation reason
- `GET /v1/nodes/{id}/{version}/node.wasm` returns 410 Gone

### 6.5 Content-Addressed Storage

GCS stores WASM binaries by content hash, not by ID:

```
gs://ruvector-brain-{region}/nodes/sha256/{hash}.wasm
```

The registry maps `(id, version)` → `hash`. Multiple nodes referencing the same bytecode share storage. This makes bytes immutable at the storage layer — the registry controls visibility, not the blob store.

## 7. Integration with Shared Brain

### 7.1 Embedding Pipeline in Browser

The clean split for browser-side semantic search:

1. **WASM node** (`sona_feature/1.0.0`): runs `feature_extract()` to produce structured hash features (128-dim f32 vector)
2. **Brain API** (`GET /v1/lora/latest`): browser fetches current MicroLoRA weights (2KB JSON)
3. **JS or second WASM node**: applies MicroLoRA transform `output = L2_norm(features + scale * (features @ down) @ up)`
4. **Brain API** (`GET /v1/memories/search`): browser sends the transformed embedding for server-side ranked search

This gives semantic retrieval in the browser with no heavy model download.

### 7.2 Brainpedia Integration

Brainpedia pages (ADR-062) can reference WASM nodes:
- A debug page can link to a `feature_extract` node for domain-specific embedding
- Evidence links can include deterministic WASM validation results
- Pages declare which node version they were indexed with for reproducibility

### 7.3 Appliance Mode

Edge appliances run the same WASM nodes as the browser and server:

1. Load node from local cache or `GET /v1/nodes/{id}/{version}/node.wasm`
2. Run conformance self-test (Section 5.3)
3. Load MicroLoRA weights from local cache or `GET /v1/lora/latest`
4. Run locally: `feature_extract()` → LoRA transform → L2 normalize
5. Search: send embedding to brain, or search local cache

Offline: steps 1-4 use cached artifacts. Step 5 searches local cache only.

## 8. Runtime Selection

| Environment | Runtime | Resource Control | Isolation |
|-------------|---------|-----------------|-----------|
| Browser | `WebAssembly.instantiate` in Worker | 25ms wall clock + 8KB input cap | Worker sandbox, no imports |
| Server | wasmtime 18+ | Fuel (~25ms ref CPU) + 16 page memory | No imports, epoch interruption |
| Appliance | wasmtime 18+ | Fuel (~25ms ref CPU) + 16 page memory | No imports, fuel metering |
| Test | wasmtime | Fuel + conformance vector check | No imports |

All environments produce bit-identical results for the same input and node SHA, verified by output byte hash comparison (Section 5.2).

## 9. Security Summary

### 9.1 Module Validation at Publish Time

1. WASM binary is parsed and validated (valid WASM module)
2. Import section must be empty (no imports of any kind)
3. Memory must declare a maximum <= 16 pages
4. No SIMD instructions (module inspection)
5. No f64 usage (module inspection)
6. Required exports present: `memory`, `malloc`, `feature_extract_dim`, `feature_extract`
7. Ed25519 signature verified over canonical binary manifest
8. SHA-256 computed and stored
9. All conformance test vectors pass on server-side wasmtime
10. Contributor reputation >= 0.5
11. Size <= 1MB

### 9.2 Client-Side Verification

1. SHA-256 hash match (WASM bytes vs metadata)
2. Ed25519 signature verification (canonical manifest)
3. Conformance test vectors pass locally
4. Only then: instantiate module

## 10. Implementation Status

### Shipped (in this branch)

| Component | Status |
|-----------|--------|
| `WasmNode` types in server | Shipped |
| `GET /v1/nodes` list endpoint | Shipped |
| `GET /v1/nodes/{id}` metadata endpoint | Shipped |
| `GET /v1/nodes/{id}.wasm` binary endpoint | Shipped |
| `POST /v1/nodes` publish endpoint | Shipped |
| Node storage in `FirestoreClient` | Shipped |

### Deferred (Future Work)

| Component | Priority | Dependency |
|-----------|----------|------------|
| Versioned URL scheme (`/v1/nodes/{id}/{version}/...`) | High | Route refactor |
| `DELETE /v1/nodes/{id}/{version}` revocation | High | Audit logging |
| Module inspection (SIMD/f64 detection) | High | WASM parser |
| Canonical binary manifest signing | High | Manifest encoder |
| Conformance test vector validation | High | Reference node |
| Content-addressed GCS storage | Medium | GCS integration |
| `sona_feature` v1 reference node | High | WASM compilation |
| Browser demo HTML | Medium | Reference node |
| Worker isolation wrapper | Medium | Browser demo |
| wasmtime server-side runner | Medium | Server integration |
| MCP tools (`brain_node_list`, `brain_node_get`) | Low | All endpoints |

## 11. Answers to Design Questions

**Q: Do you want these WASM nodes to run in browser only, or also on the appliance and server with the same bytecode?**

A: All three. Browser, appliance, and server execute the same WASM bytecode with bit-identical results, verified by output byte hash comparison across environments.

**Q: Should nodes be public read like memory search, or should node download be gated because it encodes proprietary heuristics?**

A: Public read. The proprietary value is in the MicroLoRA weights (learned from quality signals via federation), not in the deterministic feature extraction logic. Gating node download would kill the browser demo path for zero competitive benefit.

**Q: Do you want the v1 node ABI to support only feature_extract, or do you want score and validate in v1 as well?**

A: `feature_extract` only for v1. One function, one ABI, one determinism proof. `score` and `validate` are v2 after the cross-platform guarantee is proven on the simpler interface.

**Q: Should nodes be allowed to include a tiny embedded lookup table, or must they be pure hashing with no internal data beyond constants?**

A: Lookup tables are allowed. Token maps, n-gram tables, and normalization tables compiled into the WASM data section are fine — the data section is part of the signed bytecode and is immutable at runtime. This is necessary for practical feature extraction.

## 12. Acceptance Criteria

### Deterministic Cross-Platform Verification

- Pick 20 canonical inputs (including Unicode edge cases, empty string, max length)
- Store expected output SHA-256 values in node metadata `conformance.test_vectors`
- Browser demo shows green for all 20 vectors
- Server and appliance startup self-tests pass the same 20 vectors
- Compare output **byte hashes** (SHA-256 of raw f32 bytes), not float arrays
- p95 execution under 25ms for 128-dim output on all three environments

### Integration

- Browser produces embedding via WASM `feature_extract` + JS MicroLoRA transform
- Browser embedding and server-produced embedding match **byte for byte** given the same node SHA and the same LoRA epoch
- `brain_search` returns identical top-5 results for browser-produced vs server-produced embeddings

### Security

- Module with non-empty import section: rejected at publish time
- Module without memory maximum: rejected at publish time
- Module with SIMD or f64: rejected at publish time (when module inspection is implemented)
- Module exceeding 1MB: rejected at publish time
- Module without valid Ed25519 signature: rejected by client
- Conformance vector failure: node rejected at runtime
- Revoked node: returns 410 Gone

## 13. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-058 | Hash Security Optimization — the feature extraction algorithm nodes implement |
| ADR-059 | Shared Brain Google Cloud — infrastructure serving node binaries |
| ADR-060 | Shared Brain Capabilities — MicroLoRA weights that complement WASM features |
| ADR-062 | Brainpedia Architecture — pages can reference WASM nodes for validation |
