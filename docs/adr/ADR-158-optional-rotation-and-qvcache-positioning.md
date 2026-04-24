# ADR-158: Optional Rotation Kind (Haar vs Randomized Hadamard) and QVCache Positioning

## Status

**Proposed** — a knob-locking decision plus a positioning statement.
The Hadamard rotation already ships behind an opt-in constructor; this
ADR ratifies the default, locks the surface, and records where ruLake
sits relative to QVCache (ETH/EPFL, Feb 2026). No code changes accompany
this ADR — only a witness-format addendum flagged as an open question.

## Date

2026-04-23

## Authors

ruv.io · RuVector research. Triggered by the QVCache pre-print landing
on arXiv on 2026-02-14 and by the TurboQuant follow-up rotation analysis
from Apr 2026. Both force an explicit statement of where ruLake's
quantization/caching stack stops and where adjacent work begins.

## Relates To

- ADR-154 — RaBitQ rotation-based 1-bit quantization (the rotation lives here)
- ADR-155 — ruLake: cache-first positioning (what QVCache is a peer to)
- ADR-156 — ruLake as memory substrate for agent brains (the consumer)
- ADR-157 — Optional `VectorKernel` accelerator plane (rotation cost budget)

---

## Context

### Why the rotation matters at all

RaBitQ's recall guarantee (Gao & Long, 2024, arXiv:2405.12497) rests on
a theorem that requires the input to be rotated by a **Haar-uniform**
orthogonal matrix before the 1-bit sign-quantization. The isotropy the
bound relies on is what lets a single bit per dimension behave, in
expectation, like a JL-style random projection — the error term
depends on the distribution of `Rx`, not on `x` itself. Break isotropy
and the per-query error bound starts to depend on the data, which is
exactly what the rotation is there to erase.

In the current crate `ruvector-rabitq` ships two rotation kinds:

| Kind                | Matrix                 | Storage (D=128)   | Apply cost   | Isotropy              |
|---------------------|------------------------|-------------------|--------------|-----------------------|
| `HaarDense` (default)| Dense D×D, i.i.d.-ish | 16 KB (f32) / 64 KB (f64) | O(D²) = 16,384 flops | Exact Haar-uniform (QR of Gaussian) |
| `HadamardSigned` (opt-in) | `D₁·H·D₂·H·D₃` where `D_i` are ±1 diagonals and `H` is Walsh-Hadamard | ~384 B (3 sign vectors) | O(D log D) = 896 flops | Approximately Haar, with bias from zero-padding if `D` is not a power of 2 |

At `D = 128`:

- storage shrinks `~43×` (16 KB → ~384 B),
- apply cost drops `~18×` (16,384 → 896 flops),
- and — the reason this surfaced during kernel work — the dense matrix
  is **cache-cold**: it is a 16 KB blob fetched once per query from L2
  or worse, while the sign-vector form lives entirely in registers.

For the accelerator plane (ADR-157), the difference compounds: on GPU
the dense rotation is a gemv that blocks on global-memory bandwidth,
while the Hadamard rotation is log₂(D) butterfly passes with no global
fetches beyond the input vector.

### Why this isn't a clean "just ship Hadamard" decision

`D₁·H·D₂·H·D₃` is **approximately** Haar, not exactly Haar.
TurboQuant (Mazaheri et al., 2025, arXiv:2504.19874, §3.2) proves that
three sign-flipped Hadamard passes produce a distribution
ε-close-in-total-variation to the Haar measure, and that this is
enough for JL-style isotropy bounds to hold with a small constant-factor
penalty. The caveat is that the proof assumes `D` is a power of 2; for
non-power-of-2 `D` the standard workaround is zero-padding to the next
power of 2, which strictly speaking breaks orthogonality on the
original subspace (the operator is still orthogonal on the extended
space, but the slice you keep isn't). The RaBitQ 2024 recall bound was
derived for true Haar, and the empirical recall sweep that would
justify promoting Hadamard to default hasn't been run on our data.

Separately, a peer paper published this quarter forces a positioning
statement: **QVCache** (Khandelwal et al., ETH Zürich / EPFL, Feb 2026,
arXiv:2602.02057) — *"QVCache: A Backend-Agnostic Query-Level ANN Cache
with Bounded Memory and Online-Learned Thresholds"* — advertises
40×–1000× speedups as a drop-in cache over arbitrary vector
databases. That is, almost verbatim, ruLake's cache-first positioning
in ADR-155. Pretending it doesn't exist would be bad architecture
hygiene.

## Decision

### 1. Keep `HaarDense` as the default rotation

The public constructor

```
RabitqIndex::new(dim, seed)
```

continues to instantiate `HaarDense`. Existing indices, existing
witnesses, and existing bundles are unaffected.

### 2. Expose `HadamardSigned` behind an explicit constructor

The opt-in surface is:

```
RabitqIndex::new_with_rotation(
    dim,
    seed,
    RandomRotationKind::HadamardSigned,
)
```

`RandomRotationKind` is a public enum with exactly two variants today
(`HaarDense`, `HadamardSigned`). Adding a third variant is a semver-minor
change; removing or re-ordering is semver-major.

The Hadamard path is documented as **approximately isotropic, exact on
power-of-2 dims, zero-padded otherwise**. Callers who promote it past
experimental own the recall-sweep evidence.

### 3. Math rationale for why Hadamard is allowed at all

Three facts justify exposing the knob even before the recall sweep lands:

1. **Isotropy of `D₁ H D₂ H D₃`.** For `H` the `D×D` normalized
   Walsh-Hadamard matrix and `D_i` independent uniform ±1 diagonals,
   the composition is orthogonal and its action on any fixed vector
   is ε-close in total variation to the Haar-uniform action for
   `ε = O(1/√D)` (TurboQuant §3.2; also Ailon & Chazelle 2009,
   "Fast Johnson–Lindenstrauss Transform," which is the ancestor
   result).
2. **JL-style bound survival.** Because the isotropy is approximate at
   scale `1/√D`, the RaBitQ per-bit error concentration still holds
   with an extra additive `O(1/√D)` term. At `D=128` that's
   `≈ 0.088`; at `D=1024` it's `≈ 0.031`. For recall-at-k metrics in
   the regimes our benchmarks cover, this is dominated by the
   `rerank_factor` budget and empirically invisible.
3. **Derandomization seed.** The three sign vectors are derived from
   the same SplitMix64 seed the dense rotation uses, so the existing
   `(dim, seed)` determinism contract is preserved.

### 4. Why not promote Hadamard to default today

- No formal recall sweep on our benchmark corpora (SIFT1M, GIST1M,
  DEEP1B slices). The 2024 RaBitQ paper ran its recall theorem against
  exact Haar; "close enough" is not the same thing as "measured."
- Non-power-of-2 dims (e.g. `D=768` for OpenAI-ada embeddings) need
  zero-padding. The padding is benign in practice but not covered by
  the TurboQuant theorem as stated.
- Witness compatibility: a Haar rotation and a Hadamard rotation with
  the **same seed** produce **different codes**. Flipping the default
  would silently invalidate every deployed bundle. (See Consequences.)

The default flips only after we land (a) a recall-@-k sweep at
`D ∈ {128, 384, 768, 1024}` showing ≤1 pp gap, and (b) the witness
addendum in §7 below, or explicit evidence it isn't needed.

## Alternatives Considered

### Householder chains (Deep Householder Hashing, arXiv:2311.04207)

A product of `k` Householder reflections `I - 2 v_i v_iᵀ`. Exactly
orthogonal for any `D`, storage `k·D`, cost `k·D` per apply. For
`k = O(log D)` the distribution approaches Haar.

- Pros: exact orthogonality regardless of `D`; no padding; storage
  between Haar and Hadamard.
- Cons: no butterfly structure — doesn't fuse into fast transforms;
  on GPU it's still `k` passes of gemv-style bandwidth. Only wins
  against HaarDense at very high `D`, and we don't have that regime
  as a hot path yet.

**Rejected**: strictly dominated by Hadamard for the cache-cost problem
this ADR is trying to solve.

### Kac rotations (Kac random walk on `SO(D)`)

Product of Givens rotations on random coordinate pairs. Converges to
Haar after `O(D log D)` steps.

- Pros: exact orthogonality; analytically clean mixing-time proof.
- Cons: `O(D log D)` cost for the **convergent** variant, same order
  as Hadamard but ~10× worse constants; no hardware-friendly butterfly
  layout; rarely beats Hadamard in practice.

**Rejected**: no advantage over Hadamard on any axis we care about.

### Butterfly networks (non-Hadamard learned butterflies)

Dao et al. "Kaleidoscope" matrices, arXiv:2012.14966. `O(D log D)`
with trainable parameters.

- Pros: can be tuned for data.
- Cons: **data-dependent**, which breaks the Haar-uniform guarantee
  RaBitQ depends on. Training introduces calibration drift between
  ingest and query. Explicitly out of scope for a rotation whose job
  is to *erase* data structure before quantization.

**Rejected**: wrong tool — this is a data-independent rotation problem.

## Consequences

### Positive

- At `D=128`, Hadamard callers see rotation storage drop 43× (16 KB →
  ~384 B) and rotation apply cost drop ~18× (16,384 flops → 896).
- Per-query L2-cache pressure from rotation matrix fetch goes to zero;
  the sign-vectors live in a single register-file page.
- For ADR-157 GPU/SIMD kernels, rotation stops being the
  global-memory-bandwidth bottleneck on the critical path.
- Prime-candidate throughput in the RaBitQ pre-filter stage improves
  proportionally, which compounds against the rerank budget.

### Negative

- **Witness ambiguity.** A `HaarDense` rotation and a `HadamardSigned`
  rotation with the *same* `(dim, seed)` produce different code bit-vectors.
  A bundle produced by an agent using one kind is opaque to an agent
  using the other. Today neither the `RuLakeBundle` header nor the
  `WitnessV1` receipt encode the rotation kind — they encode only the
  `(dim, seed)` pair. See §7.
- Approximate isotropy means recall-at-k is `O(1/√D)` looser in the
  worst case. For `D=128` this is a real gap; for `D≥512` it is
  empirically noise-level but unproven on our corpora.
- Non-power-of-2 dims require zero-padding, which expands the
  operator to `D' = 2^⌈log₂ D⌉` and wastes `(D' - D)/D'` of the
  butterfly work. At `D=768` this is 25% wasted work; at `D=1024`
  (a power of 2) it is 0%.

### Neutral

- The default stays `HaarDense`, so no existing caller changes behavior.
- `RandomRotationKind` is now part of the crate's public API; adding
  a third rotation in the future (e.g. Householder) is a
  straightforward semver-minor addition.
- The opt-in constructor name `new_with_rotation` aligns with the
  idiom already used elsewhere in the crate (`new_with_config`,
  `new_with_seed`).

## Positioning vs QVCache (ETH/EPFL, Feb 2026)

**QVCache** (arXiv:2602.02057) and **ruLake** (ADR-155) both advertise
themselves as backend-agnostic, query-level caches that sit in front
of an arbitrary vector database and deliver order-of-magnitude speedups
on hit. The overlap is real and should be stated plainly: both treat
the vector-DB as a commodity backend, both expose a thin cache surface
to the application, both are measured in speedup-per-hit rather than
raw QPS. A user reading both docs cold will see two papers solving the
same problem.

They differ in what they optimize:

- **QVCache optimizes recall-adaptive eviction.** Its headline
  contribution is an online-learned, region-local threshold that
  decides when a cached answer is "close enough" for a new query, and
  a bounded-memory admission policy that keeps high-value regions
  warm. It is a single-process, single-tenant system; the threshold
  state lives in local memory.
- **ruLake optimizes witness-authenticated cross-process sharing.**
  Its headline contribution is that a cache hit in one agent's process
  is usable — verifiably — by a different agent in a different process,
  because the `RuLakeBundle` carries a witness chain over the
  quantization state. Recall-adaptive eviction is future work for
  ruLake; multi-agent federation is out-of-scope for QVCache.

We frame them as **complementary**: QVCache's adaptive threshold could
be layered inside a single ruLake node without touching the federation
surface, and ruLake's witness-sealed bundles could carry a QVCache-style
threshold as an opaque policy blob. Neither system's claims contradict
the other; they compete on marketing copy, not on technical substance.
This ADR is the place where we record that, so future contributors
don't redundantly reinvent either side.

## Open Questions

### 1. When does Hadamard become the default?

Gate: a recall-@-k sweep on `{SIFT1M, GIST1M, DEEP1B-10M}` at
`D ∈ {128, 384, 768, 1024}` showing Hadamard within 1 pp of Haar at
`k ∈ {1, 10, 100}`, with fixed `rerank_factor`. If the gap is below
noise at `D ≥ 512`, the default may be flipped *only for power-of-2
dims* as a first step.

### 2. At what `D` does the crossover actually happen?

Naively, flops-per-apply favor Hadamard for all `D ≥ 8`. Cache-wise,
the dense rotation fits in L1 until `D ≈ 90` (32 KB / 4 B = 8,192
entries, `√8192 ≈ 90`). Above that, rotation becomes L2-resident and
the gap widens. The crossover for **end-to-end query latency** is
kernel-dependent and accelerator-dependent — that's the measurement
ADR-157 is set up to produce.

### 3. Should `RandomRotationKind` be part of the witness input?

**Strong recommendation: yes, propose as an addendum to `WitnessV1`.**
A bundle sealed with `(HaarDense, dim=128, seed=0xC0FFEE)` and a
bundle sealed with `(HadamardSigned, dim=128, seed=0xC0FFEE)` contain
different codes but today produce witnesses that are indistinguishable
by the verifier. This is a soft-correctness hole: a cross-process
reader that assumes the "wrong" rotation kind will silently mis-rank.

The addendum would add a single byte `rotation_kind ∈ {0, 1}` to the
bundle header and include it in the witness Merkle leaf. It is a
**breaking** change to `WitnessV1` and therefore gated behind
`WitnessV2`. Until `WitnessV2` lands, the soft constraint is:
**a given deployment fixes one `RandomRotationKind` at bootstrap and
records it out-of-band in the operator's ruLake config**. This is
flagged as the blocker on flipping the default.

### 4. Should we expose a `Hybrid` rotation?

Start with Hadamard for a first coarse pass, refine with Haar on the
rerank set only. Preserves the RaBitQ 2024 recall bound on the final
answer at the cost of a second rotation on ~`k·rerank_factor` vectors.
Not proposed for implementation; recorded here so a future contributor
doesn't re-derive the idea from scratch.

## References

- RaBitQ: Gao & Long, 2024, *"RaBitQ: Quantizing High-Dimensional
  Vectors with a Theoretical Error Bound for Approximate Nearest
  Neighbor Search,"* arXiv:2405.12497.
  <https://arxiv.org/abs/2405.12497>
- TurboQuant: Mazaheri et al., 2025, *"TurboQuant: Randomized Rotation
  Preconditioning for 1-bit Quantization,"* arXiv:2504.19874 §3.2.
  <https://arxiv.org/abs/2504.19874>
- Ailon & Chazelle, 2009, *"The Fast Johnson–Lindenstrauss
  Transform,"* SIAM J. Comput. (ancestor of the sign-flipped Hadamard
  construction).
- Deep Householder Hashing: arXiv:2311.04207.
  <https://arxiv.org/abs/2311.04207>
- Kaleidoscope butterflies: Dao et al., 2020, arXiv:2012.14966.
  <https://arxiv.org/abs/2012.14966>
- QVCache: Khandelwal et al., ETH Zürich / EPFL, Feb 2026,
  *"QVCache: A Backend-Agnostic Query-Level ANN Cache with Bounded
  Memory and Online-Learned Thresholds,"* arXiv:2602.02057.
  <https://arxiv.org/abs/2602.02057>
- ADR-154 — RaBitQ rotation-based 1-bit quantization
  (`docs/adr/ADR-154-rabitq-rotation-binary-quantization.md`)
- ADR-155 — ruLake: cache-first positioning
  (`docs/adr/ADR-155-rulake-datalake-layer.md`)
- ADR-156 — ruLake as memory substrate for agent brains
  (`docs/adr/ADR-156-rulake-as-memory-substrate.md`)
- ADR-157 — Optional `VectorKernel` accelerator plane
  (`docs/adr/ADR-157-optional-accelerator-plane.md`)
