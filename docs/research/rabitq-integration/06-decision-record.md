# 06 — Decision Record

## The sharpest insight from the research

**The `VectorKernel` trait is shipped, the `CpuKernel` exists, and the
dispatch policy from ADR-157 is already specified — but no caller in
the workspace wires it up.** The only reference is a doc comment at
`crates/ruvector-rulake/src/lake.rs:595`. This means every consumer
that thinks it's getting "free pluggable acceleration" by adopting the
trait would actually be the **first non-test caller**, and would have
to implement the dispatch policy itself.

The implication is non-obvious: Pattern 2 (§03) is currently more
expensive than Pattern 1 because there is no working dispatch
implementation to copy. The right fix is to wire dispatch into ruLake
*first* (Phase 2.A in §05), making it the canonical reference, then
let other Pattern-2 consumers inherit the pattern. Otherwise we'll
end up with two consumers each writing their own divergent dispatch
policies and quietly breaking the determinism gate from ADR-157
§"Determinism as a hard gate".

This finding shifts the recommendation: don't start a Pattern-2
integration in any new crate until ruLake's `register_kernel` is real.
The §05 phase ordering is built around that.

---

## Top 3 integrations to start now

1. **`ruvector-diskann` RaBitQ backend** (§02 A1; §05 P1.A). ADR-154
   already named this as the next step; the consumer code is shaped
   right; PQ replacement is a controlled scope; ≤500 LoC. Strategic
   value: closes the "billion-scale on disk + DRAM" pitch.

2. **`ruvector-graph` `VectorPropertyIndex`** (§02 A2; §05 P1.B).
   Unblocks vector-keyed property lookup that the graph-transformer
   and GNN consumers want; pairs naturally with #3; ≤600 LoC.

3. **`ruvector-gnn` `differentiable_search`** (§02 A3; §05 P1.C). The
   smallest of the three by LoC, the highest QPS multiplier of the
   three by §02's analysis, and complements #2 directly. ≤300 LoC.

All three use Pattern 1 (direct embed); all three pin
`ruvector-rabitq = "^2.2"`; all three avoid the §03 anti-patterns.

---

## One thing we should refuse

**Don't build per-consumer 1-bit compression.** A PR that adds an
`ad-hoc binary code` module to any workspace crate — most likely
under the rationale "we just need a quick binary path before ruLake
is ready" — re-creates the original `BinaryQuantized` failure mode
that ADR-154 was specifically written to retire (15–20% recall vs
RaBitQ's 40.8% no-rerank / 98.9% rerank×5 on the same dataset, per
the §"Measured gap" comparison).

The cost of refusing is real (some consumers will wait one quarter
for ADR-160 / Phase 3 before getting their dep wired). The cost of
allowing is permanent: a fragmented compression substrate where the
witness chain (ADR-155) and the kernel-dispatch determinism contract
(ADR-157) both stop holding across crate boundaries.

If a consumer genuinely cannot wait, they get Pattern 1 (direct embed
of `ruvector-rabitq`) — not their own fork.

---

## Open questions for stakeholders

1. **Do we commit to Phase 2 (`VectorKernel` real in ruLake) before
   Phase 1 (three new direct-embed consumers) finishes?** Phase 1
   produces no Pattern-2 consumers; Phase 2 has one (ruLake) plus
   one other. Sequencing them concurrently is fine if there are two
   engineers; sequentially Phase 1 first is the safer single-engineer
   path because the §02 candidates with the highest §"Strategic
   value" (A1, A2, A3) all happen to be Pattern 1.

2. **Does ADR-160 (Phase 3.A) need to land before B2 (`ruvllm` →
   ruLake)?** The ruvllm KV cache + RAG integration is the largest
   single ROI in §02 but also the one most disrupted by getting
   the substrate question wrong. If ADR-160 says "ruLake is the
   canonical retrieval cache", B2 is straightforward; if it says "no
   canonical cache, choose per consumer", B2 becomes a multi-week
   design conversation.

3. **Should `ruvector-rabitq` ship a portable SIMD kernel as part of
   the default build, or behind a feature flag?** §05 P2.B sets it
   behind `simd`. Default-on simplifies dispatch (every CPU caller
   just gets the SIMD path) at the cost of the WASM/embedded
   footprint (§04 §5). The WASM consumers don't yet exist, so
   default-on is plausible — but reversing it later is a SemVer
   minor bump.

4. **Does Phase 2's second consumer choice between B1
   (`ruvector-attention`) and C3 (`ruvector-fpga-transformer`) matter
   strategically?** B1 is the realistic near-term win; C3 is the
   research-mode showcase. Recommendation: B1 in Phase 2; C3 lands
   only if a customer asks for FPGA inference.

5. **Is there a customer pressure to ship a Node.js / WASM binding
   parallel to the Python SDK (M1)?** None of §02 surveys this
   directly. ruvector-py shipping in PR #381 is a precedent that
   establishes the binding pattern; replicating it for Node and
   WASM is mostly mechanical *if* §04's cross-language contract is
   followed. Estimate: ~2 engineer-weeks per binding once §05 Phase
   1 has landed.
