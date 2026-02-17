# ADR-039: RVF Solver WASM — Self-Learning AGI Engine Integration

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-02-16 |
| **Deciders** | RuVector core team |
| **Supersedes** | -- |
| **Related** | ADR-032 (RVF WASM integration), ADR-037 (Publishable RVF acceptance test), ADR-038 (npx/rvlite witness verification) |

## Context

ADR-037 established the publishable RVF acceptance test with a SHAKE-256 witness chain, and ADR-038 planned npm integration for **verifying** those artifacts. However, neither the existing `rvf-wasm` microkernel nor the npm packages expose the actual self-learning engine that produces the AGI benchmarks.

The core AGI capabilities live exclusively in the Rust benchmarks crate (`examples/benchmarks/src/`):
- **PolicyKernel**: Thompson Sampling two-signal model (safety Beta + cost EMA)
- **KnowledgeCompiler**: Signature-based pattern cache with compiled skip-mode configs
- **AdaptiveSolver**: Three-loop architecture (fast: solve, medium: policy, slow: compiler)
- **ReasoningBank**: Trajectory tracking with checkpoint/rollback and non-regression gating
- **Acceptance test**: Multi-cycle training/holdout evaluation with three ablation modes

These components have no FFI dependencies, no filesystem access during solve, and no system clock requirements — making them ideal candidates for WASM compilation.

## Decision

### Create `rvf-solver-wasm` as a standalone no_std WASM module

A new crate at `crates/rvf/rvf-solver-wasm/` compiles the complete self-learning solver to `wasm32-unknown-unknown`. It is a `no_std + alloc` crate (same architecture as `rvf-wasm`) with a C ABI export surface.

**Key design choices:**

| Choice | Rationale |
|--------|-----------|
| **no_std + alloc** | Matches rvf-wasm pattern, runs in any WASM runtime (browser, Node.js, edge) |
| **Self-contained types** | Pure-integer `Date` type replaces `chrono` dependency; `BTreeMap` replaces `HashMap` |
| **libm for float math** | `sqrt`, `log`, `cos`, `pow` via `libm` crate (pure Rust, no_std compatible) |
| **xorshift64 RNG** | Deterministic, no `rand` crate dependency, identical to benchmarks RNG |
| **C ABI exports** | Maximum compatibility — works with any WASM host (no wasm-bindgen required) |
| **Handle-based API** | Up to 8 concurrent solver instances, same pattern as `rvf_store_*` exports |

### WASM Export Surface

```
┌─────────────────────────────────────────────────────┐
│              rvf-solver-wasm exports                │
├─────────────────────────────────────────────────────┤
│ Memory:                                             │
│   rvf_solver_alloc(size) -> ptr                     │
│   rvf_solver_free(ptr, size)                        │
│                                                     │
│ Lifecycle:                                          │
│   rvf_solver_create() -> handle                     │
│   rvf_solver_destroy(handle)                        │
│                                                     │
│ Training (three-loop learning):                     │
│   rvf_solver_train(handle, count,                   │
│     min_diff, max_diff, seed_lo, seed_hi) -> i32    │
│                                                     │
│ Acceptance test (full ablation):                    │
│   rvf_solver_acceptance(handle, holdout,            │
│     training, cycles, budget,                       │
│     seed_lo, seed_hi) -> i32                        │
│                                                     │
│ Result / Policy / Witness reads:                    │
│   rvf_solver_result_len(handle) -> i32              │
│   rvf_solver_result_read(handle, out_ptr) -> i32    │
│   rvf_solver_policy_len(handle) -> i32              │
│   rvf_solver_policy_read(handle, out_ptr) -> i32    │
│   rvf_solver_witness_len(handle) -> i32             │
│   rvf_solver_witness_read(handle, out_ptr) -> i32   │
└─────────────────────────────────────────────────────┘
```

### Architecture Preserved in WASM

The WASM module preserves all five AGI capabilities:

1. **Thompson Sampling two-signal model** — Beta posterior for safety (correct & no early-commit) + EMA for cost. Gamma sampling via Marsaglia's method using `libm`.

2. **18 context buckets** — 3 range (small/medium/large) x 3 distractor (clean/some/heavy) x 2 noise = 18 buckets. Each bucket maintains per-arm stats for `None`, `Weekday`, `Hybrid` skip modes.

3. **Speculative dual-path** — When top-2 arms are within delta 0.15 and variance > 0.02, the solver speculatively executes the secondary arm. This is preserved identically in WASM.

4. **KnowledgeCompiler** — Constraint signature cache (`v1:{difficulty}:{sorted_constraint_types}`). Compiles successful trajectories into optimized configs with compiled skip-mode, step budget, and confidence scores.

5. **Three-loop solver** — Fast (constraint propagation + solve), Medium (PolicyKernel selection), Slow (ReasoningBank → KnowledgeCompiler). Checkpoint/rollback on accuracy regression.

### Integration with RVF Ecosystem

```
┌──────────────────────┐         ┌──────────────────────┐
│   rvf-solver-wasm    │         │     rvf-wasm         │
│   (self-learning     │ ──────▶ │   (verification)     │
│    AGI engine)       │ witness │                      │
│                      │ chain   │ rvf_witness_verify   │
│ rvf_solver_train     │         │ rvf_witness_count    │
│ rvf_solver_acceptance│         │                      │
│ rvf_solver_witness_* │         │ rvf_store_*          │
└──────────┬───────────┘         └──────────────────────┘
           │ uses
    ┌──────▼──────┐
    │  rvf-crypto  │
    │  SHAKE-256   │
    │  witness     │
    │  chain       │
    └─────────────┘
```

The solver produces a SHAKE-256 witness chain (via `rvf_crypto::create_witness_chain`) for every acceptance test run. This chain is in the native 73-byte-per-entry format, directly verifiable by `rvf_witness_verify` in the rvf-wasm microkernel.

### npm Integration Path

#### High-Level SDK (`@ruvector/rvf-solver`)

The `@ruvector/rvf-solver` npm package provides a typed TypeScript wrapper around the raw WASM C-ABI exports, with automatic WASM loading, memory management, and JSON deserialization.

```typescript
import { RvfSolver } from '@ruvector/rvf-solver';

// Create solver (lazy-loads WASM on first call)
const solver = await RvfSolver.create();

// Train on 1000 puzzles (three-loop learning)
const result = solver.train({ count: 1000, minDifficulty: 1, maxDifficulty: 10, seed: 42n });
console.log(`Accuracy: ${(result.accuracy * 100).toFixed(1)}%`);

// Run full acceptance test (A/B/C ablation)
const manifest = solver.acceptance({ holdoutSize: 100, trainingPerCycle: 100, cycles: 5, seed: 42n });
console.log(`Mode C passed: ${manifest.allPassed}`);

// Inspect policy state (Thompson Sampling parameters, context buckets)
const policy = solver.policy();
console.log(`Context buckets: ${Object.keys(policy?.contextStats ?? {}).length}`);

// Get tamper-evident witness chain (73 bytes per entry, SHAKE-256)
const chain = solver.witnessChain();
console.log(`Witness chain: ${chain?.length ?? 0} bytes`);

solver.destroy();
```

The SDK also re-exports through the unified `@ruvector/rvf` package:

```typescript
// Unified import — solver + database in one package
import { RvfDatabase, RvfSolver } from '@ruvector/rvf';
```

#### npm Package Structure

```
npm/packages/rvf-solver/
├── package.json          # @ruvector/rvf-solver, CJS/ESM dual exports
├── tsconfig.json         # ES2020 target, strict mode, declarations
├── pkg/
│   ├── rvf_solver.js     # WASM loader (singleton, Node CJS/ESM + browser)
│   ├── rvf_solver.d.ts   # Low-level WASM C-ABI type declarations
│   └── rvf_solver_bg.wasm  # Built from rvf-solver-wasm crate
└── src/
    ├── index.ts          # Barrel exports: RvfSolver + all types
    ├── solver.ts         # RvfSolver class (create/train/acceptance/policy/witnessChain/destroy)
    └── types.ts          # TrainOptions, AcceptanceManifest, PolicyState, etc.
```

| Type | Fields | Purpose |
|------|--------|---------|
| `TrainOptions` | `count`, `minDifficulty?`, `maxDifficulty?`, `seed?` | Configure training run |
| `TrainResult` | `trained`, `correct`, `accuracy`, `patternsLearned` | Training outcome |
| `AcceptanceOptions` | `holdoutSize?`, `trainingPerCycle?`, `cycles?`, `stepBudget?`, `seed?` | Configure acceptance test |
| `AcceptanceManifest` | `modeA`, `modeB`, `modeC`, `allPassed`, `witnessEntries`, `witnessChainBytes` | Full ablation results |
| `PolicyState` | `contextStats`, `earlyCommitPenalties`, `prepass`, `speculativeAttempts` | Thompson Sampling state |
| `SkipModeStats` | `attempts`, `successes`, `alphaSafety`, `betaSafety`, `costEma` | Per-arm bandit stats |

#### Low-Level WASM Usage (advanced)

```javascript
// Direct WASM C-ABI usage (without the SDK wrapper)
const wasm = await WebAssembly.instantiate(solverModule);

const handle = wasm.exports.rvf_solver_create();
const correct = wasm.exports.rvf_solver_train(handle, 1000, 1, 10, 42, 0);

const len = wasm.exports.rvf_solver_result_len(handle);
const ptr = wasm.exports.rvf_solver_alloc(len);
wasm.exports.rvf_solver_result_read(handle, ptr);
const json = new TextDecoder().decode(new Uint8Array(wasm.memory.buffer, ptr, len));

// Witness chain verifiable by rvf-wasm
const wLen = wasm.exports.rvf_solver_witness_len(handle);
const wPtr = wasm.exports.rvf_solver_alloc(wLen);
wasm.exports.rvf_solver_witness_read(handle, wPtr);
const chain = new Uint8Array(wasm.memory.buffer, wPtr, wLen);
const verified = rvfWasm.exports.rvf_witness_verify(chainPtr, wLen);

wasm.exports.rvf_solver_destroy(handle);
```

## Module Structure

```
crates/rvf/rvf-solver-wasm/
├── Cargo.toml          # no_std + alloc, dlmalloc, libm, serde_json
├── src/
│   ├── lib.rs          # WASM exports, instance registry, panic handler
│   ├── alloc_setup.rs  # dlmalloc global allocator, rvf_solver_alloc/free
│   ├── types.rs        # Date arithmetic, Constraint, Puzzle, Rng64
│   ├── policy.rs       # PolicyKernel, Thompson Sampling, KnowledgeCompiler
│   └── engine.rs       # AdaptiveSolver, ReasoningBank, PuzzleGenerator, acceptance test
```

| File | Lines | Purpose |
|------|-------|---------|
| `types.rs` | 239 | Pure-integer date math (Howard Hinnant algorithm), constraints, puzzle type |
| `policy.rs` | ~480 | Full Thompson Sampling with Marsaglia gamma sampling, 18-bucket context |
| `engine.rs` | ~490 | Three-loop solver, acceptance test runner, puzzle generator |
| `lib.rs` | ~280 | 12 WASM exports, handle registry (8 slots), witness chain integration |

## Binary Size

| Build | Size |
|-------|------|
| Release (wasm32-unknown-unknown) | ~160 KB |
| After wasm-opt -Oz (estimated) | ~80-100 KB |

## npm Package Ecosystem

The AGI solver is exposed through a layered npm package architecture:

| Package | Version | Role | Install |
|---------|---------|------|---------|
| `@ruvector/rvf-solver` | 0.1.0 | Typed TypeScript SDK for the self-learning solver | `npm i @ruvector/rvf-solver` |
| `@ruvector/rvf` | 0.1.8 | Unified SDK re-exporting solver + database | `npm i @ruvector/rvf` |
| `@ruvector/rvf-node` | 0.1.6 | Native NAPI bindings with AGI methods (`indexStats`, `verifyWitness`, `freeze`, `metric`) | `npm i @ruvector/rvf-node` |
| `@ruvector/rvf-wasm` | 0.1.5 | WASM microkernel with witness verification | `npm i @ruvector/rvf-wasm` |

### Dependency Graph

```
@ruvector/rvf (unified SDK)
├── @ruvector/rvf-node (required, native NAPI)
├── @ruvector/rvf-wasm (optional, browser fallback)
└── @ruvector/rvf-solver (optional, AGI solver)
    └── rvf-solver-wasm WASM binary (loaded at runtime)
```

### AGI NAPI Methods (rvf-node)

The native NAPI bindings expose AGI-relevant methods beyond basic vector CRUD:

| Method | Returns | Purpose |
|--------|---------|---------|
| `indexStats()` | `RvfIndexStats` | HNSW index statistics (layers, M, ef_construction, indexed count) |
| `verifyWitness()` | `RvfWitnessResult` | Verify tamper-evident SHAKE-256 witness chain integrity |
| `freeze()` | `void` | Snapshot-freeze current state for deterministic replay |
| `metric()` | `string` | Get distance metric name (`l2`, `cosine`, `dotproduct`) |

## Consequences

### Positive

- The actual self-learning AGI engine runs in the browser, Node.js, and edge runtimes via WASM
- No Rust toolchain required for end users — `npm install` + WASM load is sufficient
- Deterministic: same seed → same puzzles → same learning → same witness chain
- Witness chains produced in WASM are verifiable by the existing `rvf_witness_verify` export
- PolicyKernel state is inspectable via `rvf_solver_policy_read` (JSON serializable)
- Handle-based API supports up to 8 concurrent solver instances
- 160 KB binary includes the complete solver, Thompson Sampling, and serde_json
- TypeScript SDK (`@ruvector/rvf-solver`) provides ergonomic async API with automatic WASM memory management
- Unified SDK (`@ruvector/rvf`) re-exports solver alongside database for single-import usage
- Native NAPI bindings expose AGI methods (index stats, witness verification, freeze) for server-side usage

### Negative

- Date arithmetic is reimplemented (pure-integer) rather than using `chrono`, requiring validation against the original
- `HashMap` → `BTreeMap` changes iteration order (sorted vs hash-order), which may produce different witness chain hashes than the native benchmarks
- Float math via `libm` may have minor precision differences vs std `f64` methods, affecting Thompson Sampling distributions
- The puzzle generator is simplified compared to the full benchmarks generator (no cross-cultural constraints)

### Neutral

- The native benchmarks crate remains the reference implementation for full-fidelity acceptance tests
- The WASM module is a faithful port, not a binding — both implementations should converge on the same acceptance test outcomes given identical seeds
- `rvf-solver-wasm` is a member of the `crates/rvf` workspace alongside `rvf-wasm`
