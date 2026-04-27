# 02 — Integration Opportunities

For each candidate consumer crate this section answers: what it
stores, where similarity matters, what 32× compression buys, the
friction, the effort, and the strategic value. Candidates are
clustered by the (value × effort) quadrant they fall into so a
roadmap can pick from the top of the list without re-deriving the
trade-offs.

The numbers behind "32× compression" are from
`crates/ruvector-rabitq/BENCHMARK.md`: at D=128 n=100k, RabitqPlus
with rerank×20 holds **100% recall@10** at **957 QPS** vs
FlatF32's 306 QPS, with 53.5 MB vs 50.4 MB total memory **including
the originals reranked from**. Strip rerank (RabitqIndex, no rerank)
and the codes alone are 2.4 MB vs 50.4 MB — that's the **17.5×–32×
compression** number cited in ADR-154.

---

## Tier A — High-value, low-effort (do first)

### A1. `ruvector-diskann` — replace or sit alongside the PQ quantizer

**What it stores.** `crates/ruvector-diskann/src/index.rs:57`
`DiskAnnIndex` keeps a Vamana graph plus per-vector PQ codes via
`crates/ruvector-diskann/src/pq.rs:14` (`ProductQuantizer` with
k-means codebooks). Insert (`index.rs:98`), batch insert
(`index.rs:118`), and search (`index.rs:169`) are all vector-keyed.

**Hot-path similarity.** Beam search inside Vamana scores candidates
via `pq::distance_with_table` (`pq.rs:220`). The PQ table is built
per query (`pq.rs:194`).

**What 32× buys.** ADR-154 §"Integration path" already calls this
shot: *use BinaryCode for the in-memory candidate list during beam
search; full vectors stay on SSD; binary codes in DRAM for filtering*.
RaBitQ's popcount kernel is faster than the table-lookup PQ inner
loop (`O(D/64)` vs `O(M)` with cache-bound LUT) and ships
deterministic codes — k-means PQ is non-deterministic across runs.

**Friction.** PQ has a `train` step (`pq.rs:46`) RaBitQ doesn't —
RaBitQ's "training" is a single rotation matrix from a seed, so the
DiskAnnIndex API can shed `train(...)` entirely on the rabitq path.
The on-disk format (`save`/`load` at `index.rs:219,297`) currently
serialises PQ codebooks; would need a parallel `.rbpx` slot or a tag
discriminating the two encodings.

**Effort.** Small — one new module, one feature flag (or a constructor
variant `DiskAnnIndex::new_rabitq(seed, rerank)`), and a code path in
beam search. ≤500 LoC.

**Strategic value.** **High.** DiskANN is the SSD-friendly cousin of
HNSW; pairing it with RaBitQ closes the "billion-scale on commodity
disk + DRAM" pitch in ADR-154. Also breaks the PQ training-data
bootstrap problem at index build.

---

### A2. `ruvector-graph` — vector-property index for nodes

**What it stores.** `crates/ruvector-graph/src/index.rs:15-79`
ships `LabelIndex`, `PropertyIndex`, `EdgeTypeIndex`, `AdjacencyIndex`.
There is no vector-property index today.

**Hot-path similarity.** The `PropertyIndex::get_nodes_by_property`
path (`index.rs:118`) does exact matching on `PropertyValue`. The
moment a property is a `Vec<f32>` (an embedding stored on a node),
this collapses to "scan every node, compute distance, return top-k" —
which the crate cannot do today without a sister index.

**What 32× buys.** A graph database with millions of nodes that each
carry a 768-dim embedding (LLM context, agent memory, code symbol)
needs vector-near-neighbor lookup as a property-search primitive.
RaBitQ codes turn that lookup from "scan everything" into "scan 1-bit
codes, rerank top candidates", and the codes themselves cost ~32× less
RAM than the originals.

**Friction.** Graph database semantics: insert/update/delete on a
single node should not rebuild the rotation. `RabitqPlusIndex::add`
(`crates/ruvector-rabitq/src/index.rs`) already supports incremental
insertion under the existing rotation. Witness chain doesn't apply
here — graph nodes have their own ID semantics, so the rabitq index is
a sub-index keyed by `NodeId`.

**Effort.** Medium-low — a new `VectorPropertyIndex` next to the
existing four (`crates/ruvector-graph/src/index.rs`), with the same
lifecycle hooks (`add_node`, `remove_node`). ~600 LoC.

**Strategic value.** **High.** Unlocks "graph-structured RAG" inside
the same crate, which is what `crates/ruvector-graph-transformer/` and
the GNN consumers actually want.

---

### A3. `ruvector-gnn` — KNN for `differentiable_search`

**What it stores.** `crates/ruvector-gnn/src/search.rs:4` exposes
`cosine_similarity(a, b)` and `differentiable_search(query, candidates,
top_k, temperature)` (`search.rs:56`). The candidates list is held by
the caller — typically as a `Vec<Vec<f32>>`.

**Hot-path similarity.** `differentiable_search` sorts every candidate
by cosine, takes top-k, and reweights the survivors with softmax.
`hierarchical_forward` at `search.rs:105` does this **once per
hierarchy layer per forward pass** during inference and training.

**What 32× buys.** GNN inference at scale (10⁵+ nodes, 768-dim
features) hits a hard memory ceiling on the candidate set; replacing
the f32 candidate fan-out with RaBitQ codes lets a 100× larger
candidate pool fit in DRAM. Symmetric estimator
(`crates/ruvector-rabitq/src/lib.rs:11-14`) is `O(D/64)` vs cosine's
`O(D)` — the same algorithmic win the rabitq-demo measures (3.1×
QPS).

**Friction.** `differentiable_search` returns *weights* via softmax,
not just ids. The 1-bit angular estimator `cos(π·(1 − B/D))` is a
proxy — top-k selection is fine, but the softmax weights would need to
come from RabitqPlus's exact-rerank f32 scores so gradients stay
meaningful. Practical: rerank top-k×10 with the f32 estimator, softmax
those.

**Effort.** Small — replace the candidate-scan loop in
`search.rs:56` with `RabitqPlusIndex::search_with_rerank`. ≤300 LoC
plus a test that shows recall@k matches the brute-force cosine
within tolerance.

**Strategic value.** **High.** Unlocks attention-over-large-graphs
patterns inside the GNN trainer. Pair with A2 for graph + GNN sharing
one rabitq sub-index.

---

## Tier B — High-value, high-effort (medium-term)

### B1. `ruvector-attention` — KV-cache compression behind 1-bit

**What it stores.** `crates/ruvector-attention/src/attention/kv_cache.rs:253`
`CacheManager` owns key/value tensors per layer, with `append`
(`kv_cache.rs:284`), `get` (`:309`), `evict` (`:325`), and
`pyramid_budget` (`:398`). It already has its own asymmetric/symmetric
quantize (`:130, :182`) producing `QuantizedTensor` (`:90`) at 4–8
bits.

**Hot-path similarity.** Attention is `softmax(QK^T / sqrt(d)) V`.
The K-cache is the database, the Q is the query — exactly RaBitQ's
*asymmetric* setting (`crates/ruvector-rabitq/src/lib.rs:16-18`,
`RabitqAsymIndex` in `src/index.rs`).

**What 32× buys.** A 32k-token cache at D=4096 is **524 MB per layer**
in f16; RaBitQ-Asym takes 16 MB for the codes. The asymmetric
estimator `‖q‖·‖x‖·(1/√D)·Σ sign(x_rot)·q_rot` keeps the query in
f32 — exactly what attention needs.

**Friction.** Attention's existing 4–8-bit quantize is bf16/f16 native
across the rest of the LLM stack; introducing a third datatype path is
real work. Also, RabitqAsym's QPS at D=128 was only **26 QPS**
(`BENCHMARK.md` headline) — that path needs the SIMD/GPU kernel from
ADR-157 before it's competitive with the existing 4-bit path.
Determinism on rerank (float reduction order) is a problem on GPU.

**Effort.** Large — touches an existing performance-sensitive cache,
needs SIMD kernel development, needs fallback to existing
`QuantizedTensor` when D is too small for the rotation cost to pay
off. ~2000+ LoC across kv_cache + a feature flag.

**Strategic value.** **Medium-high but speculative.** ruvllm's KV
cache is the bigger target (B2); attention is the upstream library.
If B2 lands first, B1 follows.

---

### B2. `ruvllm` — LLM serving KV cache + retrieval-augmented prompt cache

**What it stores.** `crates/ruvllm/src/kv_cache.rs:203` `KvMemoryPool`
holds aligned f32 buffers (`AlignedBuffer` at `kv_cache.rs:45`). The
ruvllm hot path is the same shape as B1 but at the serving layer:
multi-tenant, eviction-pressured, latency-sensitive.

**Hot-path similarity.** Same as B1 (attention K-cache). Plus, ruvllm's
RAG path is whatever the embedding model + a separate ANN index look
like — and that's a free win for ruLake (since ruvllm could just
embed and query a `RuLake` instance instead of holding its own).

**What 32× buys.** Multi-tenant serving is RAM-bound; 32× compression
of long-context K-caches lets one box serve 32× more concurrent
sessions before eviction.

**Friction.** ruvllm has its own backend abstraction
(`crates/ruvllm/src/backends/`), GGUF loaders
(`crates/ruvllm/src/gguf/`), Metal kernels (`/metal/`), and bitnet
support (`/bitnet/`). Adding RaBitQ as another quantization path
needs to live behind that backend trait, not in the cache directly.

**Effort.** Large — needs ADR-class decision on whether ruvllm
adopts ruLake as its retrieval substrate (which solves both the K-cache
and the RAG question with one integration). Otherwise: a dedicated
RaBitQ K-cache implementation. 1500–3000 LoC depending on path.

**Strategic value.** **High.** ruvllm is the LLM serving frontend;
RaBitQ-as-K-cache-compression is a marketing-grade moat ("32× more
concurrent contexts on the same hardware").

---

### B3. `ruvector-temporal-tensor` — time-windowed compressed segments

`crates/ruvector-temporal-tensor/src/{lib,tiering,quantizer,compressor,f16,segment,bitpack}.rs`
ships a temperature-tiered compression stack already (hot/warm/cold via
`tier_policy.rs`, with its own quantizer in `core_trait.rs`). Cold-tier
reads currently pay an unpack cost; if the segment payload is 1-bit
RaBitQ codes the read can stay in compressed form for proximity-of-
time-window search. 32× compression pushes billion-sample D=128/768
working sets onto one machine at the tier boundary.

**Friction.** RaBitQ is a new tier alongside scalar/PQ; determinism
still matters because of the cross-tier coherence story
(`coherence.rs`). **Effort:** medium — codec for `.rbpx`, hook into
`tier_policy`. 800–1200 LoC. **Value:** medium-high. Pairs naturally
with ruLake (different problem, same compression substrate).

### B4. `ruvector-domain-expansion` — embedding-based domain shift

`crates/ruvector-domain-expansion/src/lib.rs:90` `DomainExpansionEngine`
exposes `embed(...)` (`lib.rs:199`) and `initiate_transfer(...)`
(`:205`). The kNN-over-domains lookup at transfer time would benefit
from RaBitQ, but domain counts are 10²–10⁴ — the compression win is
modest. The real win is **consistency**: embeddings would gain witness
+ cross-process sharing for free if stored in ruLake.

**Friction.** The embedding type is `DomainEmbedding`, not raw
`Vec<f32>`; light refactor. **Effort:** small (300 LoC) — Tier B not
because of effort but because of value. **Value:** medium-low; this
is consistency hygiene, not load-bearing.

---

## Tier C — Speculative (defer or kill)

### C1. `ruvector-mincut` — graph cut over vector similarity

`crates/ruvector-mincut/src/core/`, `sparsify/`, `localkcut/`,
`cluster/` — graph-cut algorithms over edge-weighted graphs. MinCut
operates on edges, not raw vectors. The vector → kNN-graph build step
could feed RaBitQ, but that's an instance of A2/A3, not a separate
consumer. **Verdict:** defer (downstream of A2).

### C2. `ruvector-cnn` — embedding producer, not indexer

`crates/ruvector-cnn/src/embedding.rs:122` `MobileNetEmbedder`
produces `Vec<f32>` via `extract`. The crate ends at producing the
embedding; consumers do their own indexing. The "integration" is a
one-liner in user code, not a crate change. **Verdict:** kill as a
crate-level integration; add a README example showing the
producer→`RabitqPlusIndex::add` plug.

### C3. `ruvector-fpga-transformer` — RaBitQ popcount on FPGA

`crates/ruvector-fpga-transformer/src/lib.rs:86` `Engine` for
transformer inference. RaBitQ's popcount kernel is **the** kernel a
small FPGA can do well — 64-bit XOR + popcount is two LUT levels deep.
A `ruvector-rabitq-fpga` kernel under ADR-157 is a research project,
not a near-term integration. **Verdict:** defer to ADR-157 follow-on.

### C4. `ruvector-sparsifier` — spectral sparsification

Same logic as C1: vectors only enter via the kNN-graph build step.
**Verdict:** defer (downstream of A2).

### C5. `rvagent-a2a` — already integrated by reference

`crates/rvAgent/rvagent-a2a/src/artifact_types.rs:64` defines
`ArtifactKind::RuLakeWitness { witness, data_ref, capabilities }` —
a by-reference vector handle that travels between agents without
moving bytes (ADR-159 §"Typed artifact semantics"). A2A doesn't carry
RaBitQ codes directly; it carries the witness that resolves to a
ruLake bundle. The integration **is** the witness type. **Verdict:**
no-op; the "witness-as-handle" pattern is already paying off here.

---

## Summary table

| Tier | Crate | Effort | Value | Notes |
|------|-------|--------|-------|-------|
| A1 | `ruvector-diskann` | small | high | Replace/augment PQ; ADR-154 already named this |
| A2 | `ruvector-graph` | medium-low | high | New `VectorPropertyIndex` |
| A3 | `ruvector-gnn` | small | high | `differentiable_search` rewrite |
| B1 | `ruvector-attention` | large | medium-high | KV cache, asymmetric path |
| B2 | `ruvllm` | large | high | K-cache + RAG via ruLake |
| B3 | `ruvector-temporal-tensor` | medium | medium-high | New temperature tier |
| B4 | `ruvector-domain-expansion` | small | medium-low | Hygiene rather than load-bearing |
| C1 | `ruvector-mincut` | — | low | Downstream of A2 |
| C2 | `ruvector-cnn` | — | none | Pure user-code example |
| C3 | `ruvector-fpga-transformer` | research | speculative | ADR-157 kernel |
| C4 | `ruvector-sparsifier` | — | low | Downstream of A2 |
| C5 | `rvagent-a2a` | — | done | Witness-by-reference shipped |

12 candidates surveyed. Phase 1 picks A1 + A2 + A3. Phase 2 picks B1
*or* B2 (one of them, not both — they answer the same question). Phase
3 is the workspace-canonical-compression ADR question (§05).
