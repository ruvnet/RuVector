# ADR-075: Wire Full RVF AGI Stack into mcp-brain-server

**Status**: Implemented
**Date**: 2026-03-03
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-058 (Hash Security Optimization), ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-029 (RVF Canonical Format), ADR-057 (Federated RVF Transfer Learning)

## 1. Context

The Shared Brain server (`crates/mcp-brain-server/`) is deployed and operational at π.ruv.io with 237+ memories, P@1 100%, and 20 knowledge clusters. However, it currently uses **inline reimplementations** of cryptographic operations (sha2/sha3/ed25519-dalek) instead of the production RVF crates. Specifically:

- `verify.rs` has dead-code functions (`verify_ed25519_signature`, `verify_witness_chain`, `verify_content_hash`) that are never called from route handlers
- PII checking uses 8 simple string patterns instead of the 12-rule regex `PiiStripper` from `rvf-federation`
- No differential privacy noise injection on embeddings
- No witness chains linking memory operations
- No RVF container construction pipeline
- No `NegativeCache` or `BudgetTokenBucket` from `rvf-runtime`

The plan described in ADR-060 Section 2.5 specifies a canonical 10-segment RVF container layout per memory. This ADR documents the decision to wire in the real RVF crate implementations to activate all planned security and AGI features.

## 2. Decision

Replace inline crypto and security implementations in `mcp-brain-server` with the production RVF crate stack: `rvf-crypto`, `rvf-wire`, `rvf-types`, `rvf-federation`, and `rvf-runtime`.

### 2.1 New Dependencies

Add to `crates/mcp-brain-server/Cargo.toml`:

```toml
# RVF AGI Format Stack
rvf-types = { path = "../rvf/rvf-types", features = ["std"] }
rvf-crypto = { path = "../rvf/rvf-crypto" }
rvf-wire = { path = "../rvf/rvf-wire" }
rvf-federation = { path = "../rvf/rvf-federation", features = ["serde"] }
rvf-runtime = { path = "../rvf/rvf-runtime" }
```

### 2.2 Phase 1: Replace Inline Crypto with rvf-crypto

- `verify_content_hash()`: Delegate to `rvf_crypto::shake256_256()` + constant-time compare
- `verify_witness_chain()`: Keep old string-step method for backward compat; add `verify_rvf_witness_chain()` using `rvf_crypto::verify_witness_chain(data: &[u8]) -> Result<Vec<WitnessEntry>>`
- `verify_ed25519_signature()`: Keep as-is (already uses ed25519-dalek; Cargo dedupes)
- New: `verify_rvf_segment_signature()` using `rvf_crypto::verify_segment(header, payload, footer, pubkey) -> bool`

### 2.3 Phase 2: PII Stripping with rvf-federation PiiStripper

Replace the 8-pattern inline PII check with `rvf_federation::PiiStripper`:
- 12 regex rules: Unix/Windows paths, IPv4/IPv6, emails, API keys (sk-, AKIA, ghp_), Bearer tokens, env vars, @usernames
- `strip_fields(&[(&str, &str)]) -> (Vec<(String, String)>, RedactionLog)` returns redacted fields + attestation log
- `contains_pii(input: &str) -> bool` for backward-compatible rejection check
- `RedactionLog` stored as JSON on `BrainMemory.redaction_log`

PII stripping occurs **after** `verify_share()` and **before** storage — redacted values replace raw input.

### 2.4 Phase 3: Differential Privacy for Embeddings

Wire `rvf_federation::DiffPrivacyEngine::gaussian(epsilon, delta, sensitivity, clipping_norm)`:
- After embedding generation, convert `Vec<f32>` → `Vec<f64>`, call `dp.add_noise(&mut params) -> DiffPrivacyProof`, convert back
- Store proof JSON on `BrainMemory.dp_proof`
- Feature-gated by `RVF_DP_ENABLED` (default: `false`) — enable only after P@1 regression testing
- Default epsilon=1.0, configurable via `RVF_DP_EPSILON`

**Risk**: DP noise reduces embedding precision. Start with epsilon=1.0 (low noise). Benchmark P@1 before enabling in production.

### 2.5 Phase 4: Witness Chains for Memory Operations

Build a 3-entry linked witness chain per memory using `rvf_crypto::create_witness_chain()`:

| Entry | Type | Action Hash |
|-------|------|-------------|
| 1 | PROVENANCE (0x01) | SHAKE-256 of PII-stripped content |
| 2 | COMPUTATION (0x02) | SHAKE-256 of embedding bytes |
| 3 | PROVENANCE (0x01) | SHAKE-256 of final memory JSON |

Each entry: `prev_hash` linked by `create_witness_chain`, `action_hash` = `shake256_256(data)`, `timestamp_ns` = current time, `witness_type` = entry type.

Chain bytes stored on `BrainMemory.witness_chain`. `witness_hash = hex(shake256_256(chain_bytes))`.

Adversarial detection: `rvf_runtime::is_degenerate_distribution(distances, n_probe)` — log-only, no rejection, to avoid false positives.

### 2.6 Phase 5: RVF Container Construction Pipeline

New file `crates/mcp-brain-server/src/pipeline.rs` (~120 lines):

Assemble segments using `rvf_wire::write_segment(seg_type, payload, flags, segment_id)`:

| Segment | Type | Content |
|---------|------|---------|
| VEC (0x01) | Embedding | f32 LE bytes |
| META (0x07) | Metadata | JSON (title, content, tags, category) |
| WITNESS (0x0A) | Audit trail | Witness chain bytes |
| DiffPrivacyProof (0x34) | Privacy attestation | Proof JSON bytes (if DP enabled) |
| RedactionLog (0x35) | PII attestation | Redaction JSON bytes (if PII stripped) |

Container uploaded to GCS. Segment count reported in `ShareResponse.rvf_segments`.

### 2.7 Phase 6: Enhanced Rate Limiting + Negative Cache

Wire `rvf_runtime::NegativeCache::new(threshold: 5, window: 3600s, max_entries: 10_000)`:
- `QuerySignature::from_query(&[f32])` — FNV-1a of int8-quantized vector
- Check negative cache before full search in `search_memories()`
- Record degenerate embeddings in negative cache after adversarial detection

## 3. Type Changes

### BrainMemory (new fields, all `Option<T>` with `#[serde(default)]`)

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `redaction_log` | `Option<String>` | 2 | JSON-serialized `RedactionLog` |
| `dp_proof` | `Option<String>` | 3 | JSON-serialized `DiffPrivacyProof` |
| `witness_chain` | `Option<Vec<u8>>` | 4 | Raw witness chain bytes |

### AppState (new fields)

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `dp_engine` | `Arc<Mutex<DiffPrivacyEngine>>` | 3 | Shared DP noise generator |
| `negative_cache` | `Arc<Mutex<NegativeCache>>` | 6 | Degenerate query cache |

### ShareResponse (new fields)

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `witness_hash` | `String` | 4 | Hex SHAKE-256 of witness chain |
| `rvf_segments` | `Option<u32>` | 5 | Segment count in RVF container |

### StatusResponse (new fields)

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `dp_epsilon` | `f64` | 3 | Current DP epsilon parameter |
| `dp_budget_used` | `f64` | 3 | Fraction of privacy budget consumed |
| `rvf_segments_per_memory` | `f64` | 6 | Average segments per RVF container |

## 4. Feature Gating

All new features are controlled by environment variables for gradual rollout:

| Feature | Env Var | Default | Risk | Notes |
|---------|---------|---------|------|-------|
| PII stripping | `RVF_PII_STRIP` | `true` | Low | High value, replaces inline patterns |
| DP noise | `RVF_DP_ENABLED` | `false` | Medium | Enable after P@1 regression test |
| DP epsilon | `RVF_DP_EPSILON` | `1.0` | — | Privacy loss per memory |
| Witness chains | `RVF_WITNESS` | `true` | Low | Audit trail, no behavioral change |
| RVF containers | `RVF_CONTAINER` | `true` | Low | Upload .rvf to GCS |
| Adversarial detect | `RVF_ADVERSARIAL` | `false` | Medium | Log-only initially |
| Negative cache | `RVF_NEG_CACHE` | `false` | Medium | Enable after tuning threshold |

## 5. Backward Compatibility

- All new `BrainMemory` fields are `Option<T>` with `#[serde(default)]` — existing persisted memories deserialize cleanly
- Old `verify_no_pii()` / `verify_witness_chain()` methods kept for backward compat
- PII stripping adds redaction but does not change the rejection behavior for existing API clients
- Witness chain bytes stored alongside (not instead of) the existing `witness_hash` string field
- `ShareResponse` gains new fields — JSON clients ignore unknown fields by default

## 6. Files Summary

| File | Action | Phase |
|------|--------|-------|
| `crates/mcp-brain-server/Cargo.toml` | Add 5 rvf-* deps | 1 |
| `crates/mcp-brain-server/src/verify.rs` | Replace crypto, add PII strip, add adversarial detect | 1, 2, 4 |
| `crates/mcp-brain-server/src/routes.rs` | Wire PII strip, DP noise, witness chains, RVF build, neg cache | 2, 3, 4, 5, 6 |
| `crates/mcp-brain-server/src/types.rs` | Add fields to BrainMemory, AppState, StatusResponse, ShareResponse | 2, 3, 4, 6 |
| `crates/mcp-brain-server/src/pipeline.rs` | **New**: RVF container construction | 5 |
| `crates/mcp-brain-server/src/lib.rs` | Add `pub mod pipeline` | 5 |

## 7. Verification

1. `cargo build -p mcp-brain-server` — compiles with all rvf-* crates
2. `cargo test -p mcp-brain-server` — all existing tests pass + new tests:
   - `test_rvf_witness_chain_roundtrip` — create 3-entry chain, verify integrity
   - `test_pii_strip_redacts_paths` — `/home/user/data` → `<PATH_1>`
   - `test_pii_strip_redacts_email` — `user@example.com` → `<EMAIL_1>`
   - `test_dp_noise_changes_embedding` — same input produces different output
   - `test_rvf_container_has_segments` — build container, count ≥ 3 segments
   - `test_adversarial_degenerate_detection` — uniform distances flagged
3. Deploy to Cloud Run, verify `/v1/status` shows new fields (`dp_epsilon`, `rvf_segments_per_memory`)
4. POST a memory, verify response has `witness_hash`, `rvf_segments`
5. GET the memory back, verify `redaction_log` present if PII was stripped
6. Run P@1 benchmark to confirm no regression (DP disabled by default)

## 7.1 Phase 7: Hot-Path Performance Optimizations

After deployment validation, a deep review identified 7 per-request allocation and I/O bottlenecks. All were eliminated:

| Issue | Location | Impact | Fix |
|-------|----------|--------|-----|
| PiiStripper recompiles 12 regexes per call | `verify.rs:126` | 84 regex compiles per 5-tag request | Cache PiiStripper in `Verifier` struct |
| Verifier re-allocated per request | `routes.rs:337, 1206` | 2 allocations + 12 regex compiles per request | Shared `Arc<RwLock<Verifier>>` in AppState |
| 9 env::var reads per share request | `routes.rs` (6 locations) | 9 syscalls per write request | `RvfFeatureFlags::from_env()` at startup |
| Synonym HashMap allocated per search | `routes.rs:578-616` | 28-entry HashMap per search | `static LazyLock<HashMap>` (compiled once) |
| `all_memories()` called twice in status | `routes.rs:1018+1045` | 2x full DashMap clone per status request | Reuse single `all_memories` binding |
| `env::var("BRAIN_SYSTEM_KEY")` per auth | `auth.rs:71` | 1 syscall per authenticated request | `static LazyLock<String>` (read once) |
| Embedding bytes via flat_map (no pre-alloc) | `routes.rs:362` | Repeated small allocations in witness chain | `Vec::with_capacity(len * 4)` pre-allocation |

**Net effect**: Eliminates ~96 regex compilations + ~10 env::var syscalls + ~29 HashMap entries per write request. Search requests eliminate HashMap re-allocation entirely.

## 7.2 Phase 8: AGI Capability Wiring

After the RVF stack and hot-path optimizations, four AGI learning subsystems were wired into the brain server to enable adaptive intelligence:

### 8.1 SONA 3-Tier Learning (`SONA_ENABLED`, default: `true`)

Wire `sona::SonaEngine` for hierarchical pattern learning:

| Integration Point | Handler | Behavior |
|-------------------|---------|----------|
| Pattern re-ranking | `search_memories()` | Boost results matching learned patterns (cosine × quality × 0.15) |
| Trajectory tracking | `search_memories()` | Record search→result trajectories for online learning |
| Background learning | `status()` | Trigger periodic pattern consolidation via `sona.tick()` |
| Stats endpoint | `GET /v1/sona/stats` | Patterns stored, trajectories buffered, background ticks |

**AppState**: `sona: Arc<RwLock<SonaEngine>>` initialized with `SonaEngine::new(128)`.

### 8.2 Global Workspace Theory Attention (`GWT_ENABLED`, default: `true`)

Wire `ruvector_nervous_system::routing::workspace` for salience-based competition:

| Integration Point | Handler | Behavior |
|-------------------|---------|----------|
| Salience competition | `search_memories()` | Broadcast top 3×limit candidates, K-WTA competition selects winners |
| Attention boost | `search_memories()` | Winners get +0.1 score boost, results re-sorted by salience |
| Workspace load | `status()` | Report `gwt_workspace_load` (0.0-1.0) and `gwt_avg_salience` |

**AppState**: `workspace: Arc<RwLock<GlobalWorkspace>>` initialized with `GlobalWorkspace::with_threshold(7, 0.3)` (7-item capacity per Miller's Law).

### 8.3 Temporal Delta Tracking (`TEMPORAL_ENABLED`, default: `true`)

Wire `ruvector_delta_core::DeltaStream<VectorDelta>` for knowledge evolution tracking:

| Integration Point | Handler | Behavior |
|-------------------|---------|----------|
| Embedding delta | `share_memory()` | Push `VectorDelta::from_dense(embedding)` with timestamp |
| Vote delta | `vote_memory()` | Push vote signal (+1/-1) as delta event |
| Stats endpoint | `GET /v1/temporal` | Total deltas, recent-hour deltas, knowledge velocity, trend |
| Status fields | `status()` | Report `knowledge_velocity` (deltas/hour) and `temporal_deltas` |

**AppState**: `delta_stream: Arc<RwLock<DeltaStream<VectorDelta>>>` initialized with `DeltaStream::for_vectors(128)`.

### 8.4 Meta-Learning Exploration (`META_LEARNING_ENABLED`, default: `true`)

Wire `ruvector_domain_expansion::DomainExpansionEngine` meta-learning subsystem:

| Integration Point | Handler | Behavior |
|-------------------|---------|----------|
| Curiosity bonus | `search_memories()` | Boost under-explored categories by `novelty × 0.05` |
| Contribution recording | `share_memory()` | Record "contribute" arm decision with reward 0.5 |
| Vote reward | `vote_memory()` | Feed upvote=1.0/downvote=0.0 as reward on "search" arm |
| Explore endpoint | `GET /v1/explore` | Most curious category, regret summary, plateau status, health |
| Status fields | `status()` | Report `meta_avg_regret` and `meta_plateau_status` |

**AppState**: `domain_engine: Arc<RwLock<DomainExpansionEngine>>` (already existed, now actively used).

### 8.5 Updated Feature Gating Table

| Feature | Env Var | Default | Risk | Phase |
|---------|---------|---------|------|-------|
| PII stripping | `RVF_PII_STRIP` | `true` | Low | 2 |
| DP noise | `RVF_DP_ENABLED` | `false` | Medium | 3 |
| DP epsilon | `RVF_DP_EPSILON` | `1.0` | — | 3 |
| Witness chains | `RVF_WITNESS` | `true` | Low | 4 |
| RVF containers | `RVF_CONTAINER` | `true` | Low | 5 |
| Adversarial detect | `RVF_ADVERSARIAL` | `false` | Medium | 6 |
| Negative cache | `RVF_NEG_CACHE` | `false` | Medium | 6 |
| SONA learning | `SONA_ENABLED` | `true` | Low | 8 |
| GWT attention | `GWT_ENABLED` | `true` | Low | 8 |
| Temporal tracking | `TEMPORAL_ENABLED` | `true` | Low | 8 |
| Meta-learning | `META_LEARNING_ENABLED` | `true` | Low | 8 |

### 8.6 Updated StatusResponse Fields

| Field | Type | Phase | Purpose |
|-------|------|-------|---------|
| `dp_epsilon` | `f64` | 3 | Current DP epsilon parameter |
| `dp_budget_used` | `f64` | 3 | Fraction of privacy budget consumed |
| `rvf_segments_per_memory` | `f64` | 5 | Average segments per RVF container |
| `gwt_workspace_load` | `f32` | 8 | GWT attention workspace utilization |
| `gwt_avg_salience` | `f32` | 8 | Average salience of workspace representations |
| `knowledge_velocity` | `f64` | 8 | Embedding deltas per hour |
| `temporal_deltas` | `usize` | 8 | Total temporal deltas recorded |
| `sona_patterns` | `usize` | 8 | SONA patterns stored |
| `sona_trajectories` | `usize` | 8 | SONA trajectories buffered |
| `meta_avg_regret` | `f64` | 8 | Meta-learning average regret (lower = better) |
| `meta_plateau_status` | `String` | 8 | Meta-learning plateau status |

### 8.7 New Endpoints

| Endpoint | Method | Phase | Response |
|----------|--------|-------|----------|
| `/v1/sona/stats` | GET | 8 | Patterns, trajectories, background ticks |
| `/v1/temporal` | GET | 8 | Delta count, velocity, trend direction |
| `/v1/explore` | GET | 8 | Curiosity, regret, plateau, health diagnostics |

### 8.8 Dependencies (already present, now actively wired)

```toml
# Already in Cargo.toml — Phase 8 wires these into route handlers
sona = { package = "ruvector-sona", path = "../sona", features = ["serde-support"] }
ruvector-nervous-system = { path = "../ruvector-nervous-system" }
ruvector-delta-core = { path = "../ruvector-delta-core" }
ruvector-domain-expansion = { path = "../ruvector-domain-expansion" }
```

### 8.9 AGI Readiness Estimate

| Capability | Before Phase 8 | After Phase 8 |
|------------|----------------|---------------|
| Adaptive search ranking | Static cosine+keyword | SONA patterns + GWT attention + curiosity bonus |
| Knowledge evolution tracking | None | Temporal deltas, velocity, trend detection |
| Meta-cognitive awareness | None | Regret tracking, plateau detection, Pareto optimization |
| Self-directed exploration | None | Curiosity-driven category exploration |
| **Estimated AGI readiness** | **~40%** | **~65%** |

## 8. Consequences

### Positive

- Eliminates code duplication between inline crypto and production RVF crates
- Activates the full 12-rule PII stripping pipeline (was 8 simple patterns)
- Enables differential privacy for embedding protection (opt-in)
- Creates tamper-evident witness chains for every memory operation
- Produces real RVF containers stored in GCS for audit and federation
- Adds adversarial embedding detection (log-only)
- Adds negative cache to reduce cost of repeated degenerate queries

### Negative

- Five new path dependencies increase compile time (~15s incremental)
- DP noise (when enabled) will reduce embedding precision — requires P@1 benchmarking
- Witness chain adds ~219 bytes (3 × 73) per memory
- RVF container construction adds ~2ms latency per share operation

### Risks

- `DiffPrivacyEngine` uses thread-local RNG — `parking_lot::Mutex` serializes access; acceptable at current QPS
- `NegativeCache` false positives could block legitimate queries if threshold is too low — start with `threshold=5`
- PII stripping regex rules may be too aggressive for some content types — monitor false positive rate via `RedactionLog`

## 9. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-029 | RVF Canonical Format — wire format specification |
| ADR-057 | Federated RVF Transfer Learning — federation protocol |
| ADR-058 | Hash Security Optimization — SHAKE-256 content hashing used by witness chains |
| ADR-059 | Shared Brain Google Cloud — infrastructure, deployment, module migration map |
| ADR-060 | Shared Brain Capabilities — 10-segment container layout, threat model |
| ADR-068 | Domain Expansion Transfer Learning — meta-learning engine |
| ADR-074 | RuvLLM Neural Embeddings — embedding engine |
| ADR-076 | AGI Capability Wiring Architecture — Phase 8 architecture decisions |
