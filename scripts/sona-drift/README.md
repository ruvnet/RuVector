# SONA cross-implementation behavioral-parity harness

## What drift this guards

The repo has **three independent SONA learn-from-feedback implementations** that
have drifted before:

| Impl | Seam | Past regression |
|---|---|---|
| `rust-sona` — `crates/sona` (via `@ruvector/sona` N-API, `npm/packages/sona`) | `learn_from_feedback` / `end_trajectory` (wasm.rs, napi_simple.rs) | #519: single-step / constant-reward trajectories produced exact-zero gradients — feedback never adapted anything |
| `ruvllm-ts` — `SonaCoordinator` in `npm/packages/ruvllm/src/sona.ts` | `processInstantLearning` | #553: instant loop was a no-op stub — no micro-LoRA weight ever changed |
| `ruvector-cli` — `IntelligenceEngine` in `npm/packages/ruvector/src/core/intelligence-engine.ts` | `recordRouteOutcome` | #517: outcomes stored under state keys `route()` never queried |

The same stub shipped twice because nothing enforced cross-implementation
behavioral parity. These scripts make "learn-from-feedback actually learns"
an executable contract.

## The three guards

```bash
node scripts/sona-drift/harness.mjs --json        # behavioral contract matrix
node scripts/sona-drift/rvf-fingerprint.mjs        # fingerprint drift vs reference.rvf
node scripts/sona-drift/rvf-fingerprint.mjs --update   # regenerate reference (intentional change)
node scripts/sona-drift/stub-tripwire.mjs          # static no-op seam detector
```

### 1. `harness.mjs` — behavioral contracts

Drives every implementation through one deterministic scenario (fixed probe
vectors, fixed feedback embedding, fixed qualities 0.9 / 0.1 / 0.5) and checks:

- **C1** fresh engine has zero adaptation
- **C2** ONE positive feedback adapts — the #519/#553 regression tripwire
- **C3** negative feedback also adapts (unlearning)
- **C4** neutral (0.5) is a no-op — only enforced for impls that define it so
  (`ruvllm-ts`, `ruvector-cli`; the Rust engine intentionally applies a small
  quality-weighted update at 0.5, recorded but not failed)
- **C5** inference output (apply/forward/route confidence) actually changes

It also extracts a 6-component **behavioral fingerprint**
`[m1-fresh, m2-1pos, m3-6pos, m4-1neg, m5-neutral, m6-inference-change]`.
Every component is a deterministic metric — nothing timing-based is included
(durations, timestamps, throughput are inherently jittery and are excluded by
construction). The harness runs each scenario **twice** and errors out on any
bit-level difference, so jitter can never silently enter a fingerprint.
The `ruvllm-ts` micro-LoRA is re-seeded with a fixed xorshift sequence
(its production init uses `Math.random()`).

If `npm/packages/ruvector/dist` is absent or `recordRouteOutcome` does not
exist (the #517 fix is not on `main` yet), `ruvector-cli` is reported as
`"skipped: ..."` and does not fail the run.

Exit non-zero if any non-skipped implementation fails any contract.

### 2. `rvf-fingerprint.mjs` — RVF reference artifact

Fingerprints are stored in a real **RVF store** (`reference.rvf`, dimensions=6,
L2 metric) via `@ruvector/rvf`'s native NodeBackend — one vector per
implementation, id = `sha256("sona-fingerprint:<impl>")[0..16]`. The SDK also
writes `reference.rvf.idmap.json` (string-id ↔ native-label sidecar); **commit
both files together**.

Default mode re-derives current fingerprints, queries each against the stored
reference vector and fails if L2 distance > `max(1e-9, 1e-3 · ‖fingerprint‖)`.
That tolerance is tight on purpose: fingerprints are deterministic, so the only
legitimate distance sources are float32 quantization in the store (~1e-7
relative) and cross-platform FP reassociation in SIMD code (<1e-6 relative);
real regressions move the fingerprint by >0.5 relative. If CI runs on a
different platform than the one that generated the reference and this fires
with a tiny distance, regenerate the reference on that platform.

### 3. `stub-tripwire.mjs` — static no-op detector

Extracts each seam function body, strips comments and logging-only statements
(`console.*`, `web_sys::console`, `log::*!`, `println!`, …) and fails unless at
least one state-mutating statement remains (assignment, compound assignment,
increment, or a non-logging call). Deliberately lenient — it catches "body is
empty or only logs", not subtle bugs; the harness catches those.

## When a guard fires

1. **You changed SONA learning behavior intentionally** → verify the harness
   contract matrix still passes, then `node scripts/sona-drift/rvf-fingerprint.mjs --update`
   and commit `reference.rvf` + `reference.rvf.idmap.json` in the same PR as
   the behavior change. The diff of the reference is the reviewable record of
   the behavioral change.
2. **C2/C3/C5 failed** → a learn-from-feedback path stopped updating state.
   This is exactly the #519/#553 failure mode. Find which seam regressed
   (the failing impl's `detail` field names the API) and fix it — do NOT
   update the reference.
3. **Tripwire fired** → a seam body was reduced to comments/logging. Restore
   the real implementation.
4. **`no reference vector for id=...`** → a new implementation was added;
   run `--update` intentionally to enroll it.

## Verifying the guards themselves

- `rvf-fingerprint.mjs --perturb` (dev only) corrupts `m2` in memory before
  validating — must exit 1.
- Harness/tripwire failure modes can be demoed against a scratch copy: copy
  `npm/packages/ruvllm/dist/cjs` to a temp dir, replace the body of
  `processInstantLearning` with a comment, then run the harness with
  `SONA_DRIFT_RUVLLM_SONA_JS=<temp>/sona.js` (C2 fails) and
  `scanFile(<gutted .ts>, [{name:'processInstantLearning', required:true}])`
  from `stub-tripwire.mjs` (tripwire fires). Never patch the real tree.

## Build notes (CI / clean checkout)

- `ruvllm-ts`: harness auto-runs `npm run build:cjs` in `npm/packages/ruvllm`
  if `dist/cjs/sona.js` is missing.
- `rust-sona`: `npm/packages/sona/index.js` is committed but the platform
  `.node` binary is not; harness auto-runs `npm run build` there (napi build
  against `crates/sona`, needs cargo, ~40s).
- `@ruvector/rvf` is consumed via `npm/packages/rvf/dist/database.js` directly
  (committed) on top of the `@ruvector/rvf-node` platform binary resolved from
  the `npm/` workspace `node_modules`. No registry fetch needed.
- Plain `npm install` at the workspace root fails on win32 (darwin-only
  optional dep); use `npm install --workspaces=false` if you must install.
