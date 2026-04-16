# ADR-151: Miller-Rabin–Driven Prime Optimizations (PIAL)

## Status

Proposed

## Date

2026-04-16

## Authors

ruv.io · RuVector Architecture

## Relates To

- **PRD**: `docs/research/miller-rabin-optimizations/PRD.md`
- ADR-027 — HNSW parameterized query fix
- ADR-038 — npx-ruvector / RVLite witness integration
- ADR-058 — RVF hash security & optimization (finding #6)
- ADR-148 — Brain hypothesis engine
- ADR-149 — Brain performance optimizations
- ADR-150 — π-brain + RuvLtra via Tailscale

## Tier (per ADR-026)

- **Core utility**: Tier-1 (Agent Booster eligible — pure WASM transform)
- **Integration patches**: Tier-2 (Haiku-cost simple edits)

---

## Context

Five independent subsystems in ruvector default to **power-of-two moduli** for
hashing, sharding, sketching, and adjacency storage. Each has a documented or
empirically observed pathology:

1. **ruvector-graph shard router** (ADR-058 finding #6, P3): `xxh3_64() mod
   2^k` produces ~50% birthday collisions at 2³² nodes and biases under
   Zipfian keys.
2. **micro-hnsw-wasm / hyperbolic-hnsw adjacency**: open-addressed tables
   sized to `2^k` cluster on near-duplicate vectors (timestamps, sensor
   streams), inflating p99 insert latency.
3. **ruvector-sparsifier stride sampler**: power-of-two strides alias on
   grid-structured graphs (images, meshes, lattices) — well-known LCG-era
   problem with a well-known fix.
4. **ruvector-attn-mincut LSH families**: `((a·x+b) mod p) mod m` requires
   `p` to be prime and `> universe`; today's hand-picked Mersenne constants
   silently degrade past their bounds.
5. **pi-brain witness chain** (ADR-038): single-hash (XXH3) tamper-evidence
   with no per-share entropy.

A grep across all crates confirms **zero existing primality-testing code** in
ruvector. The `prime-radiant` crate's name is metaphorical (coherence-gate)
and unrelated. There is no infrastructure to build on, but the surface area
is small enough that a single utility module unlocks all five consumers.

We need a primality test that is:

- **Deterministic** for `u64` (the size used by every consumer above).
- **Allocation-free** (hot paths in `no_std` and WASM contexts).
- **Constant-time-ish** for cryptographic-flavored use (witness chain).
- **Cheap enough** to call mid-resharding without operator coordination.

**Miller-Rabin** with the Sinclair (2011) witness set
`{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` satisfies all of these for
`u64`. For `u32`, the Pomerance/Selfridge/Wagstaff set `{2, 7, 61}` is
sufficient. For `u128` (an opt-in mode for future BFV-flavored work),
probabilistic Miller-Rabin with `k = 40` rounds gives a soundness error of
`< 2^-80` — adequate for hashing and far below cryptographic thresholds.

## Decision

We will introduce a single new module — `crates/ruvector-collections/src/primality.rs` —
exposing a deterministic Miller-Rabin primality test plus `next_prime` /
`prev_prime` helpers, and we will wire it into five consumer subsystems
**incrementally, behind feature flags**, in the order described in the PRD's
Rollout Plan.

We deliberately reject every alternative that fragments the workspace
further (new crate, external dependency on `glass_pumpkin` / `num-prime`,
or duplicating logic across `micro-hnsw-wasm` and `ruvector-graph`).

### Architecture Summary

```
┌──────────────────────────────────────────────────────────────┐
│  ruvector-collections::primality   (NEW, ~250 LoC, no_std)    │
│                                                                │
│   is_prime_u32 / is_prime_u64 / is_prime_u128                │
│   next_prime_u64 / prev_prime_u64                            │
│   ephemeral_prime(seed)            ← π-brain witness only    │
└────────┬──────────────┬──────────────┬──────────────┬─────────┘
         ▼              ▼              ▼              ▼
   shard router    HNSW buckets    LSH families   witness chain
   (P1)            (P2)            (P3, P4, P5)   (P6, opt-in)
```

### What We Already Have

| Component                           | Location                                    | Status        |
|-------------------------------------|---------------------------------------------|---------------|
| Workspace utility crate             | `crates/ruvector-collections`               | Established   |
| Lemire `fastmod`                    | already vendored in tree                    | Reusable      |
| HNSW adjacency abstraction          | `crates/micro-hnsw-wasm`                    | Existing      |
| Shard router using XXH3-64          | `crates/ruvector-graph/src/distributed/`    | ADR-058 #6    |
| Pi-brain witness payload            | `crates/mcp-brain-server`                   | XXH3 only     |
| Sparsifier samplers                 | `crates/ruvector-sparsifier/src/sampler.rs` | Power-of-2    |
| LSH sketch (mincut attention)       | `crates/ruvector-attn-mincut`               | Hand-picked p |

### What We Will Build

| Item                                                    | Owner        | Phase |
|---------------------------------------------------------|--------------|-------|
| `primality.rs` + benches + property tests               | core         | 0     |
| `PRIMES_BELOW_2K` / `PRIMES_ABOVE_2K` tables + `build.rs` regen + CI cross-check vs MR | core | 0 |
| Shard-router `--feature prime-shard` switch (uses table fast path) | distributed | 1 |
| HNSW prime-bucket capacity strategy (uses table fast path) | hnsw       | 2     |
| Certified-prime LSH modulus (`p = next_prime(universe)`, general MR path) | sketches | 3 |
| Witness-chain `Option<EphemeralPrimeFingerprint>` field (general MR path) | brain | 4 |
| Optional: prime-cardinality PQ codebooks                | cnn / quant  | 5     |

### Generation Strategy: Table Fast Path + Miller-Rabin Fallback

Three of the five integration sites (shard router, HNSW buckets,
sparsifier strides) request primes near **fixed power-of-two sizes**
that never change between releases. For these we ship a static table
of "largest prime < 2^k" for k ∈ [8, 64] (~456 bytes, ~1 KB combined
with the symmetric `_ABOVE_` table) and route those calls to a single
L1-cached load — **zero Miller-Rabin work at runtime**.

The two unpredictable sites (LSH universe, witness ephemeral primes)
fall through to the general Miller-Rabin descent path at ~250 ns per
call. Both are cold paths (index-build time and per-share, respectively).

Crucially, **Miller-Rabin remains the source of truth.** The tables are
generated by a `build.rs` script that calls the MR implementation, and
a `#[test]` re-validates every entry under `cargo test`. The table is
an *amortization* of MR to compile time, not a replacement for it.

This refinement keeps the proposal's runtime cost honest: PIAL adds
≤ 1 ns to the hottest paths (shard routing, HNSW probe sequences) and
~250 ns to the coldest paths (one-shot index build, per-share fingerprint).

### Determinism Guarantees

| Range        | Witnesses                                         | Result          |
|--------------|---------------------------------------------------|-----------------|
| `n < 2^32`   | 2, 7, 61                                          | Deterministic   |
| `n < 2^64`   | 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37        | Deterministic   |
| `n < 2^128`  | 40 random rounds                                  | Pr[err] < 2⁻⁸⁰  |

Tests will pin every documented "hard" pseudoprime (e.g. 3215031751,
2152302898747) so the deterministic guarantee is regression-protected.

### Hot-Path Avoidance

Modulo-by-prime is a hardware *division* and would dominate any inner loop
that runs it per-element. To avoid this we will:

1. Compute the prime **once** per shard-rebalance / index-build.
2. Wrap it in **Lemire fastmod** (`u64 → u32` reduction with one multiply
   and one shift) so the per-element cost matches `& mask` to within ~1 ns.
3. Cache the fastmod constants alongside the modulus in the shard / HNSW /
   LSH structures.

This is what makes prime moduli cheap enough to use *everywhere*; without
fastmod the proposal would not pencil out.

## Consequences

### Positive

- **Closes ADR-058 finding #6** without the cost of switching the primary
  hash function.
- Restores the **2-independence guarantee** of the LSH families used by
  sparsifier and mincut attention — these were silently degraded.
- Gives the pi-brain witness chain a **second, cheap-to-add line of defense**
  with per-share entropy, addressing a long-standing gap.
- Adds a small, broadly useful **building block** to
  `ruvector-collections` that has zero new external dependencies.
- All work is **tier-1 / tier-2** under ADR-026 — no Opus tokens needed for
  the bulk of the implementation.

### Negative

- Five integration sites must each be reviewed and benchmarked. The PRD's
  staged rollout is mandatory — a big-bang merge would be hard to reason
  about.
- Modulo-by-prime is slower than mask if `fastmod` is forgotten. We mitigate
  by *requiring* fastmod in the integration patches and gating CI on a
  micro-benchmark that catches the regression.
- WASM `u128` is ~5× slower than native; the `u128` mode is therefore
  opt-in and will be cfg-gated out of WASM bundles by default.
- The witness-chain change is wire-format-adjacent. We make it a backward
  compatible `Option<…>` field; verifiers must accept payloads that lack it.

### Neutral / Followups

- Future work could explore Lucas–Lehmer for explicitly Mersenne-shaped
  moduli (e.g. `2^61 − 1`) — a separate ADR if benchmarks warrant.
- A `PrimeModHash<H>` newtype wrapper is the most likely next abstraction;
  we'll prototype it in Phase 1 and decide.

## Alternatives Considered

| Option                                              | Why rejected                                                       |
|-----------------------------------------------------|--------------------------------------------------------------------|
| Use `num-prime` or `glass_pumpkin` crate            | New external dep, allocates, > 100 KB WASM cost                    |
| Hard-code a static table of "good" primes           | Doesn't adapt to runtime resharding; exhausted at 2³²              |
| Switch shard hash to BLAKE3 (cryptographic)         | 8–10× slower than XXH3; ADR-058 already declined this              |
| Probabilistic-only Miller-Rabin everywhere          | Unnecessary uncertainty in the hot path; deterministic is free     |
| Build a new `ruvector-primes` crate                 | Adds a 61st workspace crate for ~250 lines of code; not worth it   |
| Do nothing                                          | Leaves five known-bad subsystems on the floor                      |

## Security Considerations

- Miller-Rabin alone is **not** a cryptographic prime generator; we never
  claim it as one. The witness-chain use (§4.4 of the PRD) layers it
  *alongside* an existing XXH3 fingerprint and a future TEE-backed
  signature (ADR-042) — defense in depth, not standalone integrity.
- Per-share ephemeral primes are derived from `SHA256(payload)[0..8]` so
  they cannot be precomputed by an attacker who has not seen the payload.
  An attacker who *has* seen the payload still needs to forge the original
  XXH3 fingerprint as well, which is the existing security baseline.
- The `u128` probabilistic mode is **never** exposed to externally-supplied
  numbers in default builds; it is gated behind `--feature unstable-u128`.

## Acceptance Criteria

A reviewer should be able to verify ADR-151 is "Done" when:

1. `cargo test -p ruvector-collections primality` is green and includes
   pinned-pseudoprime regressions (e.g. 3215031751, 2152302898747).
2. `cargo test -p ruvector-collections primality::table_cross_check`
   re-validates **every entry** of `PRIMES_BELOW_2K` and
   `PRIMES_ABOVE_2K` against the Miller-Rabin descent, confirming the
   table is consistent with the source-of-truth implementation.
3. `cargo bench -p ruvector-collections primality` reports
   `is_prime_u64 ≤ 50 ns`, `prev_prime_below_pow2 ≤ 1 ns` (table fast
   path), and `next_prime_u64(arbitrary N) ≤ 2 µs` (general MR path) on
   M-series.
4. ruvector-graph shard router under `--feature prime-shard` shows
   ≥ 30% reduction in shard-load std-dev on the Zipfian micro-bench.
5. micro-hnsw-wasm p99 insert latency at 1 M vectors drops by ≥ 15%.
6. The pi-brain `brain_share` payload tolerates *both* presence and
   absence of the new ephemeral-prime field across two release versions.
7. WASM bundle size growth: `micro-hnsw-wasm` ≤ +2 KB, `mcp-brain-server`
   ≤ +1.5 KB, prime tables ≤ +1 KB total.

---

## Notes for Reviewers

This ADR's *creative* contribution is not Miller-Rabin itself (textbook,
1976) — it is the observation that **one tiny utility unlocks five
independently identified pathologies** across hashing, sharding, sketching,
adjacency, and witnessing in a workspace that today has no primality
infrastructure at all. The PRD goes deeper on each use-case; this ADR
binds the architectural choices.
