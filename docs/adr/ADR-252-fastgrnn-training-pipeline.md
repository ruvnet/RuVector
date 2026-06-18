# ADR-252: FastGRNN Training Pipeline for Tiny Dancer Routing

- **Status**: accepted
- **Date**: 2026-06-15
- **Deciders**: ruvnet
- **Tags**: tiny-dancer, routing, training, fastgrnn, safetensors

## Context

`@ruvector/tiny-dancer` ships a native FastGRNN router that runs inference on
eight platforms (ADR — platform matrix, 0.1.20). It is inference-only: the
public API exposes `Router` / `version` / `hello`, and `RouterConfig.modelPath`
requires a pre-trained `.safetensors` file the stack could not produce.

Inspection of `ruvector-tiny-dancer-core` found three gaps blocking an
end-to-end routing integration:

1. **No real gradient.** `training.rs::train_batch` ran forward passes and
   computed loss, but the backward pass was a placeholder comment
   (`// placeholder for gradient computation`) and `apply_gradients` was a no-op.
   The model never learned.
2. **No persistence.** `model.rs::save`/`load` were `TODO` stubs returning
   `Ok(())` / a default model; `safetensors` was not even a dependency. A
   trained model could not be written or reloaded.
3. **No input adapter / export.** Nothing mapped the DRACO matrix (query
   embedding → per-model quality) into a trainable dataset, and training was not
   reachable from JS.

The FastGRNN cell here is applied as a **single step** with `h₀ = 0` (input is a
fixed-length engineered feature vector, not a sequence). This makes the required
backward pass single-step analytic backprop rather than full BPTT, and the
BCE-with-sigmoid output gradient reduces to `pred − target`.

## Decision

Implement a complete, in-stack training pipeline in `ruvector-tiny-dancer-core`,
keeping the existing scaffolding (Adam state, dataset, batch iterator, early
stopping, distillation hooks) and filling the three gaps:

1. **Analytic single-step backprop.** Add `FastGRNN::forward_cached` (returns the
   prediction plus cached pre/post activations) and `FastGRNN::backward` (returns
   `FastGRNNGradients` for all five weight matrices and four bias vectors). The
   gradient is validated against central finite differences.
2. **Real Adam step.** `train_batch` accumulates per-sample gradients (mean over
   the batch, distillation-blended target where enabled), then a single Adam
   update applies L2 regularization, global-norm gradient clipping, bias
   correction, and the parameter step — using the optimizer moment state that
   already existed.
3. **safetensors persistence.** Add the `safetensors` dependency; `save`/`load`
   serialize every weight/bias tensor (f32, little-endian) with the model
   config stored in the safetensors `__metadata__` map. A save→load round trip
   reproduces inference bit-for-bit.
4. **DRACO adapter + export.** `TrainingDataset::from_draco` derives binary
   labels (cheap model within tolerance of the best) and soft targets from a
   quality matrix; a napi `Trainer` binding and a runnable example expose
   train→save from outside the crate.

The output artifact is a standard `.safetensors` consumable by
`RouterConfig.modelPath`, closing the loop: DRACO matrix → trained model → native
routing inference on all platforms.

## Consequences

### Positive
- tiny-dancer can produce its own model; the "you supply the file" dependency in
  the routing integration is removed.
- Gradients are finite-difference-verified; save/load is round-trip-exact.
- No change to the inference path or the published ABI — training is additive.

### Negative
- `safetensors` is a new core dependency (small, pure-Rust).
- Single-step cell with `h₀ = 0` means the reset gate and recurrent matrix carry
  zero gradient by construction; they remain in the format for forward-compat but
  are inert under the current training regime. Documented, not hidden.

### Neutral
- The napi `Trainer` export requires a tiny-dancer node rebuild/republish to
  reach JS consumers; the core pipeline and CLI/example work without it.

## Links
- Relates to: `@ruvector/tiny-dancer` platform matrix (0.1.20)
- `crates/ruvector-tiny-dancer-core/src/{model.rs,training.rs}`
