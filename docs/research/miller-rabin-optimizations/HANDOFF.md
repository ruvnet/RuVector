# Handoff — Phase 0 Kickoff (PIAL)

You are starting **Phase 0** of PIAL (Prime-Indexed Acceleration Layer):
land the Miller-Rabin primality utility in `ruvector-collections` and
nothing else. Five integration phases follow in separate PRs.

## Read first (in order)

1. **`docs/adr/ADR-151-miller-rabin-prime-optimizations.md`** — the binding
   decision (status, scope, acceptance criteria, alternatives rejected).
2. **`docs/research/miller-rabin-optimizations/PRD.md`** — full design,
   five creative use-cases, performance targets, six-phase rollout, risks.
3. **This file** — Phase 0 specifics. Do not skip.

## Branch

`feat/miller-rabin-prime-optimizations` (off `main`). Already created.

## Target crate

`crates/ruvector-collections/` already exists in the workspace. Today it
contains `collection.rs`, `error.rs`, `lib.rs`, `manager.rs`. No
`benches/` directory and no `build.rs` yet — both are Phase 0 work.

## Phase 0 Deliverables (four files, one PR)

| File | Purpose | Source of truth |
|---|---|---|
| `src/primality.rs` | Deterministic Miller-Rabin for u32/u64; probabilistic for u128; tabled `prev_prime_below_pow2` / `next_prime_above_pow2` fast paths; general `prev_prime_u64` / `next_prime_u64` MR-descent paths; `ephemeral_prime(seed)` for the witness chain | PRD §5 |
| `build.rs` | Generate `PRIMES_BELOW_2K[57]` and `PRIMES_ABOVE_2K[57]` (k ∈ [8, 64]) from the MR implementation at compile time; emit as `${OUT_DIR}/prime_tables.rs` for `include!`-inclusion in `primality.rs` | ADR-151 "Generation Strategy" |
| `benches/primality.rs` | Criterion benches: `is_prime_u64`, `prev_prime_below_pow2`, `next_prime_u64(arbitrary)`, `next_prime_u64(2^61)`. Targets in PRD §6 | PRD §6 |
| `tests/table_cross_check.rs` | For every k ∈ [8, 64], assert `is_prime_u64(PRIMES_BELOW_2K[k-8])` is true and that no prime exists in `(PRIMES_BELOW_2K[k-8], 2^k)`. Same for `_ABOVE_`. This is the gate that makes MR the source of truth | ADR-151 acceptance #2 |

## Library wiring

Add `pub mod primality;` to `crates/ruvector-collections/src/lib.rs` and
re-export the public API at the crate root. Update the crate-level
doc-comment to mention the new module.

## Dependencies — explicitly do not add

The PRD rejects `num-prime`, `glass_pumpkin`, and any other external
prime/big-integer crates. Use **only** `core` integer arithmetic.
Add `criterion` under `[dev-dependencies]` for benches if it is not
already inherited via the workspace.

## Witnesses (the whole correctness story in three lines)

- `u32`: `{ 2, 7, 61 }` — Pomerance/Selfridge/Wagstaff. Deterministic.
- `u64`: `{ 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 }` — Sinclair (2011). Deterministic.
- `u128`: 40 random rounds, **only** behind `--feature unstable-u128`. Probabilistic, error < 2⁻⁸⁰.

## Pinned pseudoprime regressions

Include these in `tests/primality_pseudoprimes.rs` so future witness-set
"optimizations" cannot silently regress correctness:

- `3_215_031_751` — strong pseudoprime to bases {2, 3, 5, 7} (must be detected by Sinclair-12).
- `2_152_302_898_747` — strong pseudoprime to {2, 3, 5, 7, 11}.
- `3_825_123_056_546_413_051` — large 64-bit known-hard composite.

Add small-prime sanity (1, 2, 3, 4, 5, 7, 9, ..., 100) and edge cases
(0, 1, `u64::MAX`, `u64::MAX - 58` which is the largest u64 prime).

## Performance targets (from PRD §6)

| Operation | M-series | WASM |
|---|---|---|
| `is_prime_u64` worst-case | ≤ 50 ns | ≤ 200 ns |
| `prev_prime_below_pow2(k)` (table) | ≤ 1 ns | ≤ 2 ns |
| `next_prime_u64(2^32)` (table) | ≤ 1 ns | ≤ 2 ns |
| `next_prime_u64(arbitrary N)` (general MR) | ≤ 2 µs | ≤ 8 µs |
| `next_prime_u64(2^61)` (general MR) | ≤ 12 µs | ≤ 40 µs |

## Phase 0 is "Done" when

ADR-151 acceptance criteria #1, #2, #3 are all green:

1. `cargo test -p ruvector-collections primality` passes (includes pinned pseudoprimes).
2. `cargo test -p ruvector-collections primality::table_cross_check` validates all 114 table entries against MR.
3. `cargo bench -p ruvector-collections primality` meets the targets above on M-series.

**Do not start Phase 1 in this PR.** Phases ship as separate PRs
(PRD §7). Keep this one tightly scoped to the utility itself.

## First commands in the new session

```bash
# Confirm you are on the right branch
git status   # should show "On branch feat/miller-rabin-prime-optimizations" with no changes

# Baseline — confirm the crate compiles before you touch it
cargo check -p ruvector-collections

# Re-read the binding documents
cat docs/adr/ADR-151-miller-rabin-prime-optimizations.md | head -80
cat docs/research/miller-rabin-optimizations/PRD.md | sed -n '150,260p'   # §5 API + §6 perf
```

Then start with `crates/ruvector-collections/src/primality.rs`. The
deterministic u64 Miller-Rabin is ~80 lines including comments;
everything else (tables via `build.rs`, benches, cross-check test)
follows mechanically from it.

## What is explicitly **not** Phase 0

- Editing `crates/ruvector-graph/` (that's Phase 1).
- Editing any HNSW crate (Phase 2).
- Editing sparsifier or attn-mincut (Phase 3).
- Editing `crates/mcp-brain-server/` or pi-brain payloads (Phase 4).
- Editing CNN / quantization codebooks (Phase 5).

If you find yourself touching any of those, stop and split the PR.
