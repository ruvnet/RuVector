# Security Audit Report: RuVector DNA Analyzer

**Date:** 2026-02-11
**Scope:** Full codebase security audit (1,519 Rust source files across 80+ crates)
**Auditor:** Automated Security Audit (claude-opus-4-6)
**Version:** 2.0.2

---

## Executive Summary

The RuVector codebase demonstrates a generally security-conscious development approach, with path traversal protections in storage layers, proper use of modern cryptographic libraries (Ed25519, AES-256-GCM, Argon2id), and SHA-256 integrity verification on snapshots. However, the audit identified several findings that warrant attention, ranging from a medium-severity overly permissive CORS configuration in the REST API server to numerous `unsafe` blocks in WASM and SIMD code that, while often justified for performance, carry inherent risk. No critical production secrets or private keys were found hardcoded in source files.

**Finding Summary:**

| Severity | Count |
|----------|-------|
| Critical | 0 |
| High | 3 |
| Medium | 5 |
| Low | 4 |
| Informational | 4 |
| **Total** | **16** |

---

## Findings

### HIGH Severity

#### H-01: Overly Permissive CORS Configuration in REST API Server

**File:** `crates/ruvector-server/src/lib.rs:85-89`

```rust
let cors = CorsLayer::new()
    .allow_origin(Any)
    .allow_methods(Any)
    .allow_headers(Any);
```

**Description:** The REST API server (`ruvector-server`) configures CORS to allow requests from any origin, with any method and any headers. When this server is deployed in a production environment, this configuration allows any website to make cross-origin requests to the API, potentially enabling CSRF-like attacks where a malicious page could query or modify vector data on behalf of an authenticated user.

**Impact:** An attacker could craft a malicious webpage that performs unauthorized operations against the vector database via cross-origin requests from any browser session that has network access to the server.

**Remediation:**
- Replace `allow_origin(Any)` with a configurable allowlist of trusted origins.
- Only allow specific HTTP methods that are actually used (`GET`, `POST`, `PUT`, `DELETE`).
- Only allow specific headers needed by the API.
- Add the configuration to `Config` so operators can restrict origins in production.

---

#### H-02: No Authentication or Rate Limiting on REST API Server

**File:** `crates/ruvector-server/src/lib.rs:69-93`
**File:** `crates/ruvector-server/src/routes/points.rs` (all endpoints)
**File:** `crates/ruvector-server/src/routes/collections.rs` (all endpoints)

**Description:** The main REST API server (`ruvector-server`) has no authentication middleware, no authorization checks, and no rate limiting on any endpoint. Any network-reachable client can create/delete collections, insert vectors, and perform searches without restriction. While the `ruvector-tiny-dancer-core` admin API does have optional bearer token authentication, the primary vector database server does not.

**Impact:** Unauthenticated users can perform denial-of-service attacks by inserting massive volumes of vectors, deleting collections, or overwhelming the server with search queries. For genomic data applications, this means sensitive DNA embeddings could be exfiltrated or tampered with by anyone with network access.

**Remediation:**
- Add configurable authentication middleware (API key or bearer token at minimum).
- Add rate limiting middleware (e.g., `tower::limit::RateLimitLayer`).
- Add request size limits to prevent resource exhaustion via oversized vector payloads.

---

#### H-03: No Input Validation on API Vector Dimensions and Search Parameters

**File:** `crates/ruvector-server/src/routes/points.rs:17-34`
**File:** `crates/ruvector-server/src/routes/collections.rs:17-24`

**Description:** The REST API accepts `SearchRequest` with an unbounded `k` value and `vector` of any size, and `CreateCollectionRequest` with an unbounded `dimension`. There are no server-side validation checks on:
- Maximum `k` (number of nearest neighbors) - an attacker could request `k = usize::MAX`
- Maximum vector dimensions - could cause memory exhaustion
- Maximum number of points in an upsert batch
- Vector dimension consistency with the collection

While `ruvector-core` has an internal `MAX_DIMENSIONS = 65536` check in `cache_optimized.rs`, this is enforced via `assert!` (panic), not via a graceful error return at the API boundary.

**Impact:** Resource exhaustion, denial of service, or panics in production from malformed requests.

**Remediation:**
- Add validation at the API layer for `k`, vector dimensions, and batch sizes.
- Return `400 Bad Request` for invalid inputs rather than relying on internal panics.
- Enforce dimension consistency between vectors and their collection.

---

### MEDIUM Severity

#### M-01: Extensive Use of `static mut` in WASM Module (Thread Safety)

**File:** `crates/micro-hnsw-wasm/src/lib.rs:90-139`

```rust
static mut HNSW: MicroHnsw = MicroHnsw { ... };
static mut QUERY: [f32; MAX_DIMS] = [0.0; MAX_DIMS];
static mut INSERT: [f32; MAX_DIMS] = [0.0; MAX_DIMS];
static mut RESULTS: [SearchResult; 16] = [...];
// ... 20+ static mut declarations
```

**Description:** The `micro-hnsw-wasm` module uses 20+ `static mut` global variables for its WASM FFI layer. While this is a common pattern for `no_std` WASM modules that run single-threaded, `static mut` is inherently unsound in Rust and has been soft-deprecated. Every access requires `unsafe`. If this module were ever used in a multi-threaded context, it would produce undefined behavior due to data races.

**Impact:** Low immediate risk for single-threaded WASM use, but creates unsound foundations. The `#[no_std]` attribute and WASM target mitigate thread-safety concerns in practice.

**Remediation:**
- Consider using `core::cell::UnsafeCell` wrapped in a safe abstraction, or `static` with `Cell`/`RefCell` if single-threaded use is guaranteed.
- Add prominent documentation that this module is not thread-safe.
- Consider using `#[thread_local]` or `wasm_bindgen`'s thread-local alternatives.

---

#### M-02: Hardcoded Password in Test Code

**File:** `examples/edge-net/src/identity/mod.rs:340`

```rust
let password = "secure_password_123";
```

**Description:** A hardcoded password string exists in test code within the `examples/edge-net` identity module. While this is in a `#[cfg(target_arch = "wasm32")]` test function, hardcoded test credentials can become problematic if the pattern is copied to production code or if the test data is used to populate real environments.

**Impact:** Low direct impact as it is in test-only code, but sets a bad pattern.

**Remediation:**
- Use environment variables or test fixtures for credentials.
- Add a comment clearly marking this as test-only data.

---

#### M-03: Hardcoded Database Passwords in CI/CD Workflows

**File:** `.github/workflows/benchmarks.yml:183`
**File:** `.github/workflows/docker-publish.yml:130,192`

```yaml
POSTGRES_PASSWORD: postgres
POSTGRES_PASSWORD: ruvector
POSTGRES_PASSWORD=secret
```

**Description:** Multiple CI/CD workflows contain hardcoded PostgreSQL passwords. While these are for CI environments and ephemeral containers, the `docker-publish.yml` example command in the summary step prints `POSTGRES_PASSWORD=secret`, which persists in GitHub Action logs and step summaries.

**Impact:** CI/CD credential exposure. If similar patterns leak into deployment scripts, they could expose production databases.

**Remediation:**
- Use GitHub Secrets for all passwords, even in CI.
- Remove hardcoded passwords from example commands printed in summaries.
- Consider using randomly generated passwords for ephemeral CI databases.

---

#### M-04: Snapshot Storage Path Not Validated for Traversal

**File:** `crates/ruvector-snapshot/src/storage.rs:36-48`

```rust
fn snapshot_path(&self, id: &str) -> PathBuf {
    self.base_path.join(format!("{}.snapshot.gz", id))
}

fn metadata_path(&self, id: &str) -> PathBuf {
    self.base_path.join(format!("{}.metadata.json", id))
}
```

**Description:** The `LocalStorage` snapshot backend constructs file paths by directly joining a user-supplied `id` parameter with the base path. If an attacker can control the snapshot `id` value (e.g., `"../../etc/important"`), the resulting path would escape the base directory. There is no path traversal check or sanitization on the `id` parameter.

This contrasts with `ruvector-router-core/src/storage.rs` and `ruvector-graph/src/storage.rs`, which both include explicit path traversal detection.

**Impact:** If the snapshot ID is derived from user input (API call), an attacker could read/write arbitrary files on the filesystem within the process's permissions.

**Remediation:**
- Validate that `id` contains only alphanumeric characters, hyphens, and underscores.
- Verify the resolved path starts with `base_path` after canonicalization.
- Apply the same traversal protections used in `ruvector-router-core/src/storage.rs`.

---

#### M-05: Non-Constant-Time Token Comparison (Partial Mitigation)

**File:** `crates/ruvector-tiny-dancer-core/src/api.rs:537-544`

```rust
let mut result = token_bytes.len() == expected_bytes.len();
let min_len = std::cmp::min(token_bytes.len(), expected_bytes.len());
for i in 0..min_len {
    result &= token_bytes[i] == expected_bytes[i];
}
```

**Description:** The bearer token comparison in the admin API attempts constant-time comparison but has a subtle issue: the loop only iterates over `min_len` bytes. If the token lengths differ, the comparison exits early after only comparing the shorter length. While the length check on the first line partially mitigates this, a timing side-channel could reveal whether the token length is correct before the content comparison occurs. A dedicated constant-time comparison function would be more robust.

**Impact:** Theoretical timing side-channel attack could reveal token length, though exploitability is low.

**Remediation:**
- Use the `subtle` crate's `ConstantTimeEq` trait or `ring::constant_time::verify_slices_are_equal` for cryptographic token comparison.
- Alternatively, HMAC the token and compare HMACs.

---

### LOW Severity

#### L-01: Unchecked Integer Casts (`as usize`) Throughout Codebase

**Files:** Numerous files across crates (80+ instances found in WASM crates alone)

Example from `crates/ruvector-dag-wasm/src/lib.rs:49`:
```rust
let id = self.nodes.len() as u32;
```

Example from `crates/ruvector-node/src/lib.rs:97`:
```rust
max_elements: config.max_elements.unwrap_or(10_000_000) as usize,
```

**Description:** The codebase uses `as` casts extensively to convert between integer types (e.g., `u32` to `usize`, `usize` to `u32`, `u64` to `usize`). These casts silently truncate on 32-bit platforms or if values exceed the target type's range. While most values in practice are small, the lack of checked conversions means unexpected truncation could occur.

**Impact:** Low practical risk on 64-bit platforms, but could cause logic errors or panics on 32-bit targets.

**Remediation:**
- Use `try_into()` or `usize::try_from()` with proper error handling at API boundaries.
- Use `saturating_cast` patterns where truncation is acceptable.
- Note: 269 instances of checked/saturating arithmetic were found, showing awareness of this issue in critical paths.

---

#### L-02: `clippy::all` Suppressed in PostgreSQL Extension

**File:** `crates/ruvector-postgres/src/lib.rs:13`

```rust
#![allow(clippy::all)] // Allow all clippy warnings for development
```

**Description:** The PostgreSQL extension crate suppresses all Clippy warnings. This disables important safety lints including those for `unsafe` code patterns, potential panics, and correctness issues. The PostgreSQL extension has significant `unsafe` usage (79+ pointer operations) for FFI with PostgreSQL internals.

**Impact:** Clippy would catch potential issues in the extensive `unsafe` FFI code used for PostgreSQL integration.

**Remediation:**
- Replace with targeted `#[allow(...)]` annotations on specific items that need exemptions.
- At minimum, keep `clippy::correctness` and `clippy::suspicious` enabled.

---

#### L-03: Debug/Example API Keys in Documentation and Tests

**File:** `crates/ruvector-core/src/embeddings.rs:22`
**File:** `crates/ruvector-core/tests/embeddings_test.rs:127`
**File:** `crates/ruvector-core/src/agenticdb.rs:156`

```rust
// In doc comments and tests:
let api_provider = ApiEmbedding::openai("sk-...", "text-embedding-3-small");
let openai_small = ApiEmbedding::openai("sk-test", "text-embedding-3-small");
```

**Description:** Placeholder API keys appear in documentation comments and test code. While these are clearly placeholder values (`sk-...`, `sk-test`) and not real credentials, they could encourage copy-paste patterns where developers insert real API keys in code.

**Impact:** Minimal direct impact. Risk of establishing patterns that lead to real key exposure.

**Remediation:**
- Use environment variable patterns in documentation: `std::env::var("OPENAI_API_KEY")`.
- Add comments warning against hardcoding real keys.

---

#### L-04: `panic = "abort"` in Release Profile

**File:** `Cargo.toml:156`

```toml
[profile.release]
panic = "abort"
```

**Description:** The release profile uses `panic = "abort"`, which terminates the process immediately on any panic rather than unwinding the stack. While this is common for performance, it means that any `assert!`, `unwrap()`, or index out-of-bounds in production code will crash the entire process without running destructors or cleanup handlers.

**Impact:** In a server context, a single bad input that triggers a panic will take down the entire server process. Combined with H-03 (no input validation), a malicious request could crash the server.

**Remediation:**
- Ensure all public API boundaries use `Result` returns rather than panicking operations.
- Consider using `panic = "unwind"` for the server crate specifically to allow graceful error recovery.
- Implement a process supervisor or restart policy for production deployments.

---

### INFORMATIONAL

#### I-01: Cryptographic Libraries Are Modern and Well-Chosen

**Files:** Various `Cargo.toml` across crates

**Description:** The project uses industry-standard cryptographic libraries:
- `ed25519-dalek 2.1` for signatures
- `x25519-dalek 2.0` for key exchange
- `aes-gcm 0.10` for authenticated encryption
- `argon2 0.5` for password-based key derivation
- `sha2 0.10` for hashing
- `chacha20poly1305 0.10` for AEAD

No deprecated or weak algorithms (MD5, SHA1, DES, RC4, ECB mode) were found in security-relevant code paths. This is a positive finding.

---

#### I-02: Path Traversal Protection Present in Core Storage Layers

**Files:**
- `crates/ruvector-router-core/src/storage.rs:47-65` (explicit traversal check)
- `crates/ruvector-graph/src/storage.rs:75-89` (explicit traversal check)

**Description:** The router and graph storage layers include explicit path traversal detection that validates `..` components and ensures resolved paths remain within the working directory. This is a good security practice, though it should be applied consistently to all storage layers (see M-04).

---

#### I-03: `cargo audit` Not Available in Build Environment

**Description:** `cargo-audit` is not installed in the current environment, preventing automated dependency vulnerability scanning. The `Cargo.lock` file is 289KB and commits exact dependency versions, which is good for reproducibility.

**Remediation:**
- Add `cargo audit` to CI/CD pipeline.
- Consider adding `cargo-deny` for policy-based dependency checking.
- The project uses a patched `hnsw_rs` (via `[patch.crates-io]`), which means upstream security patches for that dependency must be manually tracked.

---

#### I-04: Extensive Justified `unsafe` Usage in Performance-Critical Code

**Files:**
- `crates/ruvector-core/src/simd_intrinsics.rs` (SIMD intrinsics - justified)
- `crates/ruvector-core/src/arena.rs` (arena allocator - justified)
- `crates/ruvector-postgres/` (PostgreSQL FFI - justified)
- `crates/micro-hnsw-wasm/src/lib.rs` (WASM FFI - justified, see M-01)

**Description:** The codebase contains 620+ raw pointer operations and significant `unsafe` code blocks. The majority of these are in:
1. **SIMD intrinsics** (`simd_intrinsics.rs`): Required for AVX2/AVX-512/NEON acceleration. Guarded by runtime feature detection.
2. **Arena allocator** (`arena.rs`): Custom allocator with proper alignment checks and size validation (line 66-72).
3. **PostgreSQL FFI**: Required for `pgrx` integration.
4. **WASM FFI**: Required for `no_std` WASM modules.

The SIMD and arena code includes appropriate safety documentation and bounds checking. The PostgreSQL FFI is inherently unsafe but follows `pgrx` conventions.

---

## Summary Table

| ID | Severity | Category | Finding | File(s) |
|----|----------|----------|---------|---------|
| H-01 | High | Configuration | Permissive CORS (`allow_origin(Any)`) | `ruvector-server/src/lib.rs:85-89` |
| H-02 | High | Authentication | No auth or rate limiting on REST API | `ruvector-server/src/lib.rs`, `/routes/*.rs` |
| H-03 | High | Input Validation | Unbounded vector dimensions and search k | `ruvector-server/src/routes/points.rs:17-34` |
| M-01 | Medium | Memory Safety | 20+ `static mut` globals in WASM module | `micro-hnsw-wasm/src/lib.rs:90-139` |
| M-02 | Medium | Secrets | Hardcoded password in test code | `examples/edge-net/src/identity/mod.rs:340` |
| M-03 | Medium | Secrets | Hardcoded DB passwords in CI workflows | `.github/workflows/benchmarks.yml:183` |
| M-04 | Medium | Path Traversal | Snapshot ID not sanitized | `ruvector-snapshot/src/storage.rs:36-48` |
| M-05 | Medium | Cryptography | Imperfect constant-time token comparison | `ruvector-tiny-dancer-core/src/api.rs:537-544` |
| L-01 | Low | Integer Safety | Unchecked `as` casts across codebase | Multiple files (80+ instances) |
| L-02 | Low | Code Quality | Clippy suppressed in PostgreSQL extension | `ruvector-postgres/src/lib.rs:13` |
| L-03 | Low | Secrets | Placeholder API keys in docs/tests | `ruvector-core/src/embeddings.rs:22` |
| L-04 | Low | Availability | `panic = "abort"` with insufficient input validation | `Cargo.toml:156` |
| I-01 | Info | Cryptography | Modern crypto libraries (positive) | Various `Cargo.toml` |
| I-02 | Info | Path Traversal | Traversal protection in core storage (positive) | `ruvector-router-core/src/storage.rs` |
| I-03 | Info | Dependencies | cargo audit not in CI pipeline | Build environment |
| I-04 | Info | Memory Safety | Justified unsafe in SIMD/arena/FFI (positive) | `ruvector-core/src/simd_intrinsics.rs` |

---

## Genomic Data Security Considerations

Given that RuVector may process genomic/DNA data embeddings, the following additional recommendations apply:

1. **Data at Rest Encryption**: Vector embeddings of genomic sequences could potentially be reversed to recover sensitive genetic information. The storage layer (redb, memmap2) does not encrypt data at rest. Consider adding transparent encryption for genomic collections.

2. **Access Control for Sensitive Collections**: Genomic data falls under HIPAA (PHI), GDPR (special category personal data), and the GINA Act. The current server has no collection-level access controls. Implement per-collection authentication and audit logging.

3. **Audit Logging**: No audit trail exists for data access operations. For genomic data compliance, log all search/read operations with timestamps and requester identity.

4. **Data Retention and Deletion**: The snapshot system creates persistent copies but lacks a secure deletion mechanism. Implement secure erasure (overwrite before delete) for genomic data snapshots.

5. **Vector Embedding Inversion Risk**: Research has shown that embedding vectors can sometimes be inverted to recover original text. For DNA sequences, this means vector embeddings of genomic data should be treated with the same sensitivity as the raw sequences.

---

## Recommendations Priority

1. **(Immediate)** Add authentication and rate limiting to `ruvector-server` (H-02)
2. **(Immediate)** Add input validation at all API boundaries (H-03)
3. **(Short-term)** Restrict CORS configuration with a configurable allowlist (H-01)
4. **(Short-term)** Add path traversal validation to snapshot storage (M-04)
5. **(Short-term)** Use `subtle::ConstantTimeEq` for token comparison (M-05)
6. **(Medium-term)** Add `cargo audit` and `cargo deny` to CI pipeline (I-03)
7. **(Medium-term)** Remove hardcoded CI passwords and use GitHub Secrets (M-03)
8. **(Long-term)** Evaluate `static mut` alternatives in WASM module (M-01)
9. **(Long-term)** Add data-at-rest encryption for sensitive genomic collections
10. **(Long-term)** Implement collection-level access controls and audit logging
