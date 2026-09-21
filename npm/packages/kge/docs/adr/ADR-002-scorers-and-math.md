# ADR-002: Scorers and math — HolE via FFT, one `Scorer` trait

## Status
Implemented and measured (2026-09-21); optimize campaign pending

## Date
2026-09-21

## Context

The engine scores a triple `(s, r, o)`. Three facts from the research brief
govern the math:

- **HolE score** `= r · (e_s ⋆ e_o)` where `⋆` is circular correlation,
  computable in `O(d log d)` via FFT versus `O(d²)` directly. This is standard
  DSP complexity (high confidence), but **no benchmark of a Rust FFT crate
  against KGE circular correlation was found** — throughput is unmeasured.
- **HolE ≡ ComplEx** (Hayashi & Shimbo, arXiv:1702.05563): the two are the same
  model class, so HolE's expressiveness is ComplEx's, and — critically — HolE's
  ComplEx-equivalent numbers can be borrowed as gates (LibKGE never implemented
  HolE, folding it into ComplEx).
- **RotatE** (arXiv:1902.10197) is the one family member that provably
  represents relation **composition** `r1∘r2`, the concrete gap for
  compositional queries.

No FFT crate is a workspace dependency today (`grep rustfft|realfft Cargo.toml`
is empty), so the FFT backend is a **new dependency**, not a reuse.

## Decision

1. **One `Scorer` trait, three implementations.**

   ```rust
   pub trait Scorer {
       fn score(&self, s: &[f32], r: &[f32], o: &[f32]) -> f32;
       fn query_vector(&self, r: &[f32], anchor: &[f32], head_side: bool) -> Vec<f32>; // for ANN
       fn id(&self) -> &str;
   }
   ```

   - `HolE` (default): real-valued circular correlation via FFT.
   - `RotatE` (opt-in): `-‖e_s ∘ r − e_o‖` with `r` constrained to the unit
     circle in complex space — the composition-capable scorer.
   - `ComplEx` (internal, test-only): `Re(⟨r, e_s, conj(e_o)⟩)`, kept to assert
     the equivalence, not exposed as a product scorer.

2. **The HolE inner-product transform is spelled out and unit-tested.** With
   `ŝ=F(e_s)`, `ô=F(e_o)`, `r̂=F(r)`, Parseval gives

   ```
   score(s,r,o) = r·(e_s ⋆ e_o) = (1/d)·Re⟨ r̂ ⊙ ŝ , ô ⟩
                = (1/d)·( Re(r̂⊙ŝ)·Re(ô) + Im(r̂⊙ŝ)·Im(ô) )
   ```

   a plain real dot product. This is exactly ComplEx's `Re⟨r,s,conj(o)⟩` with
   complex parameters equal to the DFT of HolE's real parameters — so
   **HolE≡ComplEx is a unit test, not just a citation**: score both forms on the
   same random parameters and assert equality to floating-point tolerance. It
   also gives ADR-001's ANN query vector `q = [Re(r̂⊙ŝ); Im(r̂⊙ŝ)]` for free.

3. **FFT crate choice is a native+wasm spike with a hard pass/fail**, the same
   discipline typesafe ADR-002 §3 used for tract INT8. The spike matrix:

   | axis | values |
   |---|---|
   | dimension `d` | 128, 256, 512 |
   | target | native (x86-64/aarch64), wasm32 |
   | path | FFT (`realfft`/`rustfft`) vs direct `O(d²)` correlation |

   Correctness: every FFT result asserted equal to the direct correlation on
   random vectors. **Pass = FFT beats direct at d=512 native *and* the chosen
   crate compiles and runs on wasm32.** At d=256 the constant-factor overhead
   (twiddle setup, complex arithmetic) may leave direct competitive — the spike
   decides the crossover per target, it is not assumed. `rustfft` has had
   historical wasm32 feature-detection friction; that is exactly what the spike
   exists to catch. Plan caching amortises FFT setup across batch scoring.

4. **Batch scoring is matrix-shaped, not triple-by-triple.** For a fixed
   relation, scoring against the whole entity table is a matmul of the query
   vector against the Fourier-feature matrix; `ruvector-gnn::tensor`'s
   `hadamard_product` (tensor.rs:390) and SIMD dot products
   (`ruvector-router-core::distance`) are the elementwise primitives. This is the
   trick production KGE servers (DGL-KE, GraphVite) use, which HNSW/MIPS then
   approximates sublinearly (Budgeted MIPS, arXiv:1610.03317).

5. **Quantization is v2.** RaBitQ/INT8 compression of the entity table
   (`ruvector-rabitq`, reusable as-is) is deferred: it changes recall and must
   be gated on its own recall@k measurement, not shipped as a v1 default.

## Consequences

- The scorer is the seam that keeps HolE and RotatE swappable and keeps the
  ComplEx equivalence checkable; adding a future scorer is a trait impl plus a
  gate, not an engine change.
- A new FFT dependency enters the tree; its licence is checked by `cargo-deny`
  at add time (ADR-005). `rustfft` and `realfft` are expected to be
  MIT OR Apache-2.0 (both in `deny.toml:178-186`) — to be confirmed, not
  asserted, when the crate lands.
- Until the ADR-002 spike runs, **no FFT-vs-direct throughput number is
  claimed**; ADR-006's latency/throughput gates are explicitly "set after the
  spike".

## Alternatives considered

- **Direct `O(d²)` correlation only, skip FFT.** Simplest, and possibly fine at
  d≤256 — but it forecloses large-`d` graphs and the tidy Fourier-domain ANN
  index. Kept as the correctness oracle and the small-`d` fallback the spike may
  select, not as the sole path.
- **Ship ComplEx directly instead of HolE.** Same expressiveness, but doubles
  the canonical parameter count and loses the literal "holographic" story and
  the single real-valued table. Kept internal for the equivalence test.
- **Make RotatE the default.** It adds composition but its complex modulus score
  is not an inner product, so it loses the ANN-retrieval trick; wrong default
  for a vector-DB product. Opt-in for composition-heavy graphs.

## Evidence

- Circular correlation / FFT complexity: Nickel et al., AAAI 2016; standard DSP.
- HolE≡ComplEx: arXiv:1702.05563. RotatE: arXiv:1902.10197. MIPS/ANN for
  trilinear scoring: Budgeted MIPS arXiv:1610.03317; Flashlight arXiv:2209.10100.
- Repo primitives: `crates/ruvector-gnn/src/tensor.rs:390` (`hadamard_product`);
  `crates/ruvector-router-core/src/distance.rs` (SIMD dot/cosine/L2);
  `crates/ruvector-rabitq` (v2 quantization).
- No FFT crate in workspace: `grep -niE 'rustfft|realfft' Cargo.toml` empty on
  2026-09-21. Licence allow-list: `deny.toml:178-186`.
- Unmeasured (must be gated, not caveated): Rust FFT-vs-direct throughput (brief
  Q7); wasm32 viability of the FFT crate.
