# ADR-145: WASM/NAPI Training Pipeline Fixes

**Status**: Accepted
**Date**: 2026-04-06
**Authors**: Claude Code (Opus 4.6)
**Supersedes**: None
**Related**: ADR-144 (Monorepo Quality Analysis Strategy)

---

## Context

The WASM and NAPI training pipeline spans two crate pairs:
- `ruvector-learning-wasm` — MicroLoRA adaptation (WASM)
- `ruvector-attention-wasm` — Contrastive loss + optimizers (WASM)
- `ruvector-attention-node` — Contrastive loss + optimizers (NAPI/Node.js)

Three issues were reported that prevent the training pipeline from producing meaningful adaptation:

1. LoRA weights initialize to zero, producing identity transforms
2. `computeContrastiveLoss` has a type mismatch in the WASM binding
3. `optimizerStep` has a Buffer reference issue in the NAPI bridge

---

## Decision

### Issue 1: LoRA Zero Initialization — NOT A BUG

**File**: `crates/ruvector-learning-wasm/src/lora.rs:62-93`

The B matrix is initialized to zeros (line 83) while A is initialized with Kaiming-like scaling (lines 66-80). This produces an identity transform on the first forward pass: `output = input + alpha * (input @ A @ 0) = input`.

**This is correct LoRA design** per Hu et al. (2021). The LoRA paper specifies:
- A is initialized with random Gaussian
- B is initialized to zero
- The initial delta is zero, so the pre-trained model is preserved at the start of fine-tuning

The `adapt()` method (lines 148-179) updates B via outer-product gradient updates. After one or more `adapt()` calls, the forward pass produces non-trivial outputs. The existing test at line 523 explicitly verifies this: output differs from input after adaptation.

**Action**: No code change. Document in the npm package README that `adapt()` or `adapt_with_reward()` must be called before the LoRA produces non-identity transforms.

### Issue 2: WASM Contrastive Loss Type Mismatch — REAL BUG

**File**: `crates/ruvector-attention-wasm/src/training.rs:29-39`

```rust
// CURRENT (broken): negatives param is untyped JsValue
pub fn compute(
    &self,
    anchor: &[f32],
    positive: &[f32],
    negatives: JsValue,  // ← Problem: JS Float32Array[] doesn't deserialize to Vec<Vec<f32>>
) -> Result<f32, JsError> {
    let negatives_vec: Vec<Vec<f32>> = serde_wasm_bindgen::from_value(negatives)?;
    // ...
}
```

When JS passes `Float32Array[]`, `serde_wasm_bindgen::from_value` fails because `Float32Array` is a TypedArray with an `ArrayBuffer` backing, not a regular JS Array of numbers. The deserializer sees a TypedArray and cannot convert it to `Vec<f32>`.

The NAPI binding (`ruvector-attention-node/src/training.rs:53-66`) handles this correctly using native `Vec<Float32Array>` type.

**Fix**: Convert each `Float32Array` element explicitly via `js_sys::Float32Array` before collecting into `Vec<Vec<f32>>`.

### Issue 3: NAPI Optimizer Step Buffer Reference — DESIGN BUG

**Files**: `crates/ruvector-attention-node/src/training.rs:269,347,419`

```rust
// CURRENT: consumes params, returns new allocation
pub fn step(&mut self, params: Float32Array, gradients: Float32Array) -> Float32Array {
    let mut params_vec = params.to_vec();  // ← copies data from Buffer
    let gradients_slice = gradients.as_ref();
    self.inner.step(&mut params_vec, gradients_slice);
    Float32Array::new(params_vec)  // ← allocates new Buffer, original is dropped
}
```

The `step()` method takes `Float32Array` by value, copies to a Vec, mutates the copy, and returns a new `Float32Array` backed by a new Buffer allocation. This means:
- The caller's original Buffer reference is invalidated (consumed by the NAPI bridge)
- Each step allocates and deallocates a Buffer (GC pressure)
- Callers expecting in-place mutation of their typed array see no change

The Rust `Optimizer::step()` trait method operates on `&mut [f32]` (in-place), but the NAPI binding doesn't expose this correctly.

**Fix**: Use `Buffer` or `&mut [f32]` semantics to mutate in-place, or clearly document the copy-return pattern so callers assign the return value.

---

## Affected Files

### Crate: `ruvector-attention-wasm`

| File | Change | Priority |
|------|--------|----------|
| `src/training.rs:29-39` | Replace `JsValue` negatives param with explicit `Float32Array` array handling via `js_sys` | Critical |

### Crate: `ruvector-attention-node`

| File | Change | Priority |
|------|--------|----------|
| `src/training.rs:269` | `SGDOptimizer::step` — document copy-return or switch to in-place mutation | High |
| `src/training.rs:347` | `AdamOptimizer::step` — same fix | High |
| `src/training.rs:419` | `AdamWOptimizer::step` — same fix | High |

### Crate: `ruvector-learning-wasm`

| File | Change | Priority |
|------|--------|----------|
| `src/lora.rs` | No code change — add documentation clarifying B=0 is by design | Low |

---

## Consequences

### Positive

- **Contrastive loss becomes usable from JS**: Float32Array[] inputs will correctly deserialize
- **Optimizer step semantics become clear**: Either in-place mutation or documented copy-return
- **LoRA misconception resolved**: Documented that identity-on-init is correct LoRA behavior

### Negative

- **WASM API signature change**: `compute()` parameter type changes from `JsValue` to explicit typed array handling — breaking change for any existing callers
- **NAPI optimizer API may change**: If switching to in-place mutation, callers that rely on the return value need updating

### Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| WASM API break affects downstream | Low | Medium | This API was broken anyway (always errored on Float32Array[]) |
| In-place mutation causes NAPI safety issues | Medium | Low | Use `Buffer::from_mut` or `Ref<Float32Array>` |

---

## References

- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (2021) — B=0 initialization
- wasm-bindgen TypedArray handling: https://docs.rs/js-sys/latest/js_sys/struct.Float32Array.html
- NAPI-RS Buffer semantics: https://napi.rs/docs/concepts/external
