# PRD: Prime-Indexed Acceleration Layer (PIAL)

> Creative Miller-Rabin–driven optimizations for ruvector's hashing,
> sharding, sketching, and witness-chain layers.

| Field              | Value                                                |
|--------------------|------------------------------------------------------|
| **Status**         | Draft                                                |
| **Date**           | 2026-04-16                                           |
| **Owner**          | RuVector Core / Architecture                         |
| **Related ADR**    | ADR-151 (this PRD's binding decision record)         |
| **Cross-refs**     | ADR-027 (HNSW), ADR-038 (witness), ADR-058 (hash),   |
|                    | ADR-148/149 (brain perf), ADR-150 (π-brain)          |
| **Tier (ADR-026)** | T1 (Agent Booster eligible) for the core utility;    |
|                    | T2 (Haiku) for the integration patches.              |

---

## 1. Background

Three years of incremental work have left ruvector with several places where
**arithmetic on indices, hashes, and shard keys defaults to power-of-two
moduli** — convenient on hardware (`x & (N - 1)`), pathological on real data:

| Site                                              | Current modulus    | Failure mode                                               |
|---------------------------------------------------|--------------------|------------------------------------------------------------|
| `ruvector-graph` shard router (ADR-058 #6)        | `xxh3_64() mod 2^k`| ~50% collision @ 2³² nodes; biased on Zipfian keys         |
| `micro-hnsw-wasm` adjacency map                   | open-addressed 2^k | clustering on near-duplicate vectors (e.g. timestamps)     |
| `ruvector-sparsifier` stride sampler              | power-of-2 stride  | aliasing on lattice / image-grid graphs                    |
| `ruvector-attn-mincut` LSH sketch                 | ad-hoc constant    | breaks 2-independence of universal hash family             |
| pi-brain witness fingerprint (ADR-038)            | XXH3 only          | single-hash tamper risk; no per-share entropy              |

The fix in every one of these is **the same primitive**: a fast, deterministic
primality test that lets us mint a prime *near a target size* on demand.

We choose **Miller-Rabin** because it is:

- **Deterministic** for all `u64` inputs with the Sinclair witness set
  `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` — no probabilistic guarantees
  needed for our hot paths.
- **O(k · log³ n)** — a `next_prime(2^32)` call costs ~2 µs in benchmarks;
  amortized to zero against shard-rebalance cycles.
- **WASM-friendly** — pure integer arithmetic, no FFI, fits in <1 KB compiled.
- **Tier-1 eligible** under ADR-026 — pure transform work, no LLM in the loop.

This PRD frames a single, surgically scoped utility (`primality.rs`) that
*unblocks* a portfolio of creative optimizations across the workspace. The
binding architectural commitments live in ADR-151.

---

## 2. Goals

| # | Goal                                                                 | Metric / Acceptance                                  |
|---|----------------------------------------------------------------------|------------------------------------------------------|
| G1| Provide `is_prime`, `next_prime`, `prev_prime` over `u32`/`u64`     | Deterministic, ≥ 200 M ops/s on M-series             |
| G2| Re-shard ruvector-graph by prime modulus                             | ≥ 30% reduction in shard-load std-dev on Zipfian load|
| G3| Convert HNSW adjacency tables to prime-bucket open addressing        | ≥ 15% drop in p99 insert latency at 1 M vectors      |
| G4| Replace LSH stride/modulus constants with certified primes           | Restore 2-independence; pass property tests          |
| G5| Add per-share ephemeral prime fingerprint to π-brain witness chain   | +8 bytes/share; published in `brain_share` payload   |
| G6| Cross-target: the utility compiles for native, WASM, and `no_std`    | Single crate, no feature-flag explosion              |

## 3. Non-Goals

- **No cryptographic key generation.** Miller-Rabin alone is *not* a substitute
  for proven-prime generation in RSA/ECC; we only use it for hashing/sharding.
- **No new heap allocations** in the inner loop — the utility must be
  allocation-free past the (constant-size) witness array.
- **No replacement** of `prime-radiant` (which is a coherence-gate crate and
  unrelated despite the name collision).
- **No big-integer support.** 64-bit (and an opt-in `u128` mode) is enough for
  every ruvector use case identified above.
- **No SHAKE/HMAC redesign.** ADR-058's other findings stand independently.

---

## 4. Creative Use-Cases (the "why this is interesting")

### 4.1 Prime-Modulus Shard Routing — *direct fix for ADR-058 #6*

Today's shard router is `xxh3_64(node_id) & (shards - 1)`. The mask discards
all but `log₂(shards)` bits of entropy, which is exactly when adversarial /
Zipfian inputs cluster. Replacing it with `xxh3_64(node_id) % p`, where
`p = prev_prime(shards)`, recovers full entropy and gives provably balanced
buckets under universal hashing.

> **Creative twist:** because `prev_prime(k)` is cheap, we can *adapt* the
> modulus during a rolling re-shard (every N minutes) — the cluster never
> sees a power-of-two pathology because the modulus literally never *is* a
> power of two for two consecutive epochs.

### 4.2 Prime-Bucket HNSW Adjacency

`micro-hnsw-wasm` and `ruvector-hyperbolic-hnsw` store edges in open-addressed
tables sized to the next power of two. Probe-sequence collisions on
near-duplicate vectors (e.g. real-time sensor or timestamp embeddings) blow up
p99 insert latency. Switching to `prev_prime(2^k)` capacity with linear or
quadratic probing keeps the table size cache-friendly while breaking the
power-of-two clustering.

### 4.3 Certified Modulus for Universal LSH

Several sketch modules (`ruvector-attn-mincut`, sparsifier samplers) build
hash families of the form `((a · x + b) mod p) mod m`. The 2-independence
guarantee *requires* `p` to be prime and `> universe_size`. Today these are
hand-picked Mersenne-shaped constants (`2^61 − 1`, `2^31 − 1`); when the
universe grows past those bounds the family silently degrades. Miller-Rabin
lets us call `next_prime(universe_size)` on dataset load and store the chosen
modulus alongside the index.

### 4.4 Witness-Chain Ephemeral Primes (π-brain)

The pi-brain witness chain (ADR-038, CLAUDE.md "Witness Chain Rules")
currently fingerprints each shared memory with XXH3 only. We propose:

```text
share = { payload, fingerprint_xxh3, ephemeral_prime q, fingerprint_modq }
        where q = next_prime( seed = SHA256(payload)[0..8] )
```

A tampering peer attempting to substitute payloads must collide *both*
fingerprints — including a hash modulo a prime `q` they cannot precompute,
because `q` is derived per-share. Cost: 8 bytes on the wire, ~2 µs at the
sender, ~50 ns at every verifier. The asymmetry is the point.

### 4.5 Anti-Aliasing Stride for Sparsifier Sampling

Spectral sparsifiers in `ruvector-sparsifier` use stride-based subsampling
when the importance sketch is too expensive. Power-of-two strides alias
brutally on grid-structured graphs (image, mesh, lattice). A prime stride
breaks the alignment for the same reason linear-congruential generators
demand prime moduli — borrowed wisdom, decades old, free to reuse.

### 4.6 Bonus: Prime-Sized Quantization Codebooks

Product-quantization codebooks (used by ruvector-cnn-wasm and ruQu) sized to
prime cardinalities show measurably better recall@k on standard benchmarks
than power-of-two codebooks because they break the implicit "code-of-codes"
correlation across sub-spaces. This is an opt-in mode, not a default.

---

## 5. Proposed Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  crates/ruvector-collections/src/primality.rs  (new, ~250 LoC) │
│                                                                │
│   pub fn is_prime_u32(n: u32) -> bool         // {2,7,61}     │
│   pub fn is_prime_u64(n: u64) -> bool         // Sinclair-12  │
│   pub fn is_prime_u128(n: u128, k: u8) -> bool // probabilistic│
│   pub fn next_prime_u64(n: u64) -> u64                        │
│   pub fn prev_prime_u64(n: u64) -> u64                        │
│   pub fn ephemeral_prime(seed: u64) -> u64    // for §4.4     │
│                                                                │
│   #[cfg(target_arch = "wasm32")] // shares same impl          │
└──────────────────┬───────────────────────────┬────────────────┘
                   │                           │
        ┌──────────┴──────────┐      ┌─────────┴───────────┐
        ▼                     ▼      ▼                     ▼
  shard router          HNSW buckets   LSH families     witness chain
  (ruvector-graph)      (micro-hnsw)   (sparsifier,     (mcp-brain-server,
                                        attn-mincut)     pi-brain)
```

### Why `ruvector-collections`?

- It already houses cross-cutting data-structure utilities.
- All five consumers depend on it transitively, so no new edges in the
  dependency graph.
- Keeps the workspace top-level crate count flat (we have 60+ already).

### Public API (sketch)

```rust
//! crates/ruvector-collections/src/primality.rs
//!
//! Deterministic Miller-Rabin primality for u32/u64 and probabilistic
//! Miller-Rabin for u128. Allocation-free, no_std-friendly.
//!
//! Hot-path strategy: tabled primes for the common power-of-two-aligned
//! sizes (zero runtime cost), Miller-Rabin descent as the general fallback.

#[inline]
pub const fn is_prime_u32(n: u32) -> bool { /* witnesses: 2, 7, 61 */ }

#[inline]
pub const fn is_prime_u64(n: u64) -> bool {
    // Sinclair (2011): deterministic for all u64
    // witnesses: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
}

pub fn is_prime_u128(n: u128, rounds: u8) -> bool { /* probabilistic */ }

// ── Generation: dual-path ────────────────────────────────────────────
//
// Fast path: lookup table for "largest prime < 2^k", k ∈ [8, 64].
// CI validates every entry against the Miller-Rabin descent at build
// time, so the table is never the source of truth — MR is.
const PRIMES_BELOW_2K: [u64; 57] = [
    251,                          // < 2^8
    509,                          // < 2^9
    1021,                         // < 2^10
    // ... entries for k = 11..=31 ...
    4_294_967_291,                // < 2^32  (shard-router common case)
    // ... entries for k = 33..=63 ...
    18_446_744_073_709_551_557,   // < 2^64
];

#[inline]
pub const fn prev_prime_below_pow2(k: u32) -> u64 {
    debug_assert!(k >= 8 && k <= 64);
    PRIMES_BELOW_2K[(k - 8) as usize]
}

#[inline]
pub fn prev_prime_u64(n: u64) -> u64 {
    // Fast path: power-of-two-aligned inputs (HNSW buckets, shard sizes)
    if n.is_power_of_two() && n.trailing_zeros() >= 8 {
        return prev_prime_below_pow2(n.trailing_zeros());
    }
    // General path: 6k±1 wheel + Miller-Rabin descent
    miller_rabin_descent(n, Direction::Down)
}

#[inline]
pub fn next_prime_u64(n: u64) -> u64 {
    if n.is_power_of_two() && n.trailing_zeros() >= 8 {
        // Symmetric optional fast path: PRIMES_ABOVE_2K table
        return next_prime_above_pow2(n.trailing_zeros());
    }
    miller_rabin_descent(n, Direction::Up)
}

pub fn ephemeral_prime(seed: u64) -> u64 {
    // seed → next_prime((seed | 1) % 2^61) — used by witness chain (§4.4)
    // No table — input is unpredictable by design.
}
```

### Why the dual-path matters

Three of PIAL's five generation sites (shard router, HNSW bucket sizing,
sparsifier strides) ask for primes near *fixed* sizes that never change
between releases. The table converts those calls into a single L1-cached
load — no Miller-Rabin work at runtime at all.

The two unpredictable sites (LSH universe, witness-chain ephemeral primes)
fall through to the general MR path. They're cold paths anyway —
microsecond-scale generation cost is invisible against the surrounding work.

**Crucially, MR is still the source of truth.** A `build.rs` script
regenerates `PRIMES_BELOW_2K` and `PRIMES_ABOVE_2K` from the MR
implementation on every build, and a `#[test]` cross-checks every entry
under `cargo test`. The table is an *amortization*, not a substitute.

| Generation site             | Path taken         | Runtime cost |
|-----------------------------|--------------------|--------------|
| Shard router (`prev_prime(2^k)`)  | Fast (table)       | ~1 ns        |
| HNSW bucket (`prev_prime(2^k)`)   | Fast (table)       | ~1 ns        |
| Sparsifier stride (table-friendly)| Fast (table)       | ~1 ns        |
| LSH modulus (`next_prime(N)`)     | General (MR)       | ~250 ns      |
| Witness ephemeral (`next_prime(seed)`)| General (MR)   | ~250 ns      |

---

## 6. Performance Targets

> **Revised 2026-04-16 (Phase 0).** The original `is_prime_u64` worst-case
> target of 50 ns was found to be unachievable in pure safe Rust;
> `num-prime` itself measures ~880 ns on the same hardware. Target relaxed
> to track the empirical safe-Rust ceiling. See §6.1 and the Phase 0
> Findings section of ADR-151 for the full justification.

| Operation                                      | Target (M-series)   | Target (WASM)      |
|------------------------------------------------|---------------------|--------------------|
| `is_prime_u64(p)` (worst-case)                 | **≤ 1 µs** *(was 50 ns)* | **≤ 4 µs** *(was 200 ns)* |
| `prev_prime_below_pow2(k)` (table fast path)   | **≤ 1 ns**          | **≤ 2 ns**         |
| `next_prime_u64(2^32)` (table fast path)       | **≤ 1 ns**          | **≤ 2 ns**         |
| `next_prime_u64(arbitrary N)` (general MR path)| ≤ 2 µs              | ≤ 8 µs             |
| `next_prime_u64(2^61)` (general MR path)       | ≤ 12 µs             | ≤ 40 µs            |
| Shard re-route on 1 M nodes                    | ≤ 30 ms (one-shot)  | n/a                |
| HNSW p99 insert @ 1 M vectors                  | -15% vs baseline    | -10% vs baseline   |
| WASM bundle growth from `PRIMES_BELOW_2K`+`_ABOVE_2K` | n/a          | ≤ 1 KB total       |

Benchmarks live in `crates/ruvector-collections/benches/primality.rs` and run
under existing `npm run bench` infrastructure.

### 6.1 Empirical findings (Phase 0)

Phase 0 measurements on M-series, criterion release profile:

| Bench                                      | Measured  | Revised target | Status |
|--------------------------------------------|-----------|----------------|--------|
| `prev_prime_below_pow2(32)`                | 552 ps    | ≤ 1 ns         | met    |
| `next_prime_u64(2^61 − 1)`                 | 10.97 µs  | ≤ 12 µs        | met    |
| `next_prime_u64(arbitrary ≈ 1e9)`          | 2.23 µs   | ≤ 2 µs         | +11%   |
| `is_prime_u64(u64::MAX − 58)` worst-case   | 15.24 µs  | ≤ 1 µs         | does not meet revised target — Phase 0.1 |

A throwaway scratch crate compiling a verbatim copy of our kernel
alongside `num-prime` 0.4.4 in the same binary on the same input
measured **ours = 15.63 µs, num-prime = 884 ns** (criterion sanity no-op
= 467 ps confirms harness honesty). The 17.7× gap is recoverable in pure
safe Rust by porting Montgomery-form modular multiplication into
`mr_mulmod_u64` / `mr_powmod_u64` (~80 LoC). That is Phase 0.1 scope and
ships in a separate PR; see ADR-151 "Phase 0 Findings" for the full plan
and the explicit rejection of the empirical 7-witness "Sinclair" set as
a correctness regression dressed as a perf win.

---

## 7. Rollout Plan

| Phase | Scope                                                                   | Gate                                       |
|-------|-------------------------------------------------------------------------|--------------------------------------------|
| **0** | Land `primality.rs` + tests + benches in `ruvector-collections`         | `npm test && npm run lint` green           |
| **1** | Wire `next_prime` into ruvector-graph shard router behind feature flag  | A/B Zipfian load; ≥ 30% std-dev reduction  |
| **2** | Convert HNSW adjacency to prime buckets (micro-hnsw-wasm first)         | recall@k unchanged; p99 insert -15%        |
| **3** | Switch sparsifier + attn-mincut LSH families to certified primes        | property tests pass; no regression in cuts |
| **4** | Ship ephemeral-prime fingerprint in pi-brain witness payload (opt-in)   | `brain_share` accepts new field; verifiers |
|       |                                                                         | tolerant of absence (backward compatible)  |
| **5** | Optional: prime-sized PQ codebooks in ruvector-cnn-wasm                 | recall@10 ≥ baseline on SIFT-1M            |

Each phase is a separate PR; no big-bang merge.

---

## 8. Risks & Mitigations

| Risk                                                            | Mitigation                                                     |
|-----------------------------------------------------------------|----------------------------------------------------------------|
| Modulo-by-prime is a *division*, slower than mask               | Use Lemire's `fastmod` (one mul + one shift) — already in tree |
| Sinclair witness set has subtle bugs in edge cases (n < 9)      | Hard-code small-prime fast path + 100% branch coverage tests   |
| WASM `u128` codegen is ~5× slower than native                   | u128 mode is opt-in; default paths are u64                     |
| Cluster mid-flight reshard exposes intermediate state           | Phase 1 ships behind `--feature prime-shard`; rollout is gated |
| Witness-chain change breaks older pi-brain peers                | New field is `Option<…>`; verifiers ignore-on-absent           |
| "Yet another collections crate" sprawl                          | All work lives in *existing* `ruvector-collections`            |

---

## 9. Open Questions

1. Should `next_prime_u64` accept a *budget* (max-distance) and return
   `Option<u64>` instead of looping unbounded? (Probably yes.)
2. Do we want a `PrimeModHash<H>` newtype wrapper that auto-applies fastmod,
   or expose `prev_prime` and let callers compose? (Lean: wrapper.)
3. Does the witness-chain ephemeral prime need to be authenticated under the
   sender's key, or is per-share derivation from `SHA256(payload)` enough?
   (Defer to security review during Phase 4.)

---

## 10. Out of Scope (deliberately)

- Big-integer / arbitrary-precision Miller-Rabin (use `num-bigint` if ever
  needed — not on the roadmap).
- Replacing XXH3 as ruvector's primary hash (ADR-058's job).
- Strong-pseudoprime-based Lucas certificates (yagni for hashing).
- Distributed prime-generation protocols (we mint locally, deterministically).

---

## 11. Approval Checklist

- [ ] Architecture review (links ADR-151)
- [ ] Security review (esp. §4.4 witness chain)
- [ ] Performance baseline captured for shard-router and HNSW p99
- [ ] WASM size budget verified (`micro-hnsw-wasm` < +2 KB)
- [ ] Documentation: README in `ruvector-collections` references new module
