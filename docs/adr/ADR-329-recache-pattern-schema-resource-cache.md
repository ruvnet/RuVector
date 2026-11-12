# ADR-329: Content-Addressed Schema-Resource Cache (ReCache-Pattern)

- **Status**: Proposed
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: ruvector ADR-301 (semantic query cache — prior art for caching mechanics, different cached object); ADR-320 (PIR — content-addressed identity discipline this ADR mirrors); ADR-011 (prefix caching, Proposed, never implemented as a named module); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, recache, schema-cache, context-assembly, mcp, ttft, security

## Context

Wave-4 evidence review grades this paper **A** —
[arXiv:2608.19662](https://arxiv.org/abs/2608.19662), "ReCache: Efficient
KV Cache Reuse and Compression for Tool-Augmented LLM Agents," submitted
2026-08-20.

Confirmed mechanism, verbatim against the abstract: resource-wise
attention "removes cross-resource interactions and assigns resource-local
positions, producing composition-invariant KV blocks," with visibility
restricted to "contribution-selected layer–KV-head-group routes" plus
structural and semantic pruning. Numbers, verbatim: "Resource-wise
attention matches dense invocation performance (82.3% versus 82.4%
Inv-F1) while providing a 3.655× time-to-first-token speedup," and "The
complete framework reduces allocated KV-tensor memory by 92.43% and
accelerates attention by 1.423×." **The 82.3/82.4 metric is Inv-F1
(tool-invocation F1), not "accuracy"** — the program brief's phrasing is
corrected here. **Scale caveat**: results are at Qwen3-1.7B/4B only.

**Artifact availability — checked, not assumed: code exists but is
unlicensed.** [`EIT-NLP/ReCache`](https://github.com/EIT-NLP/ReCache) is
linked from the abstract itself and confirmed the authors' repo with real
content — but it has **no LICENSE file** at check time. **No code may be
copied or vendored from it**; this ADR is a pattern reimplementation from
the paper text only, re-checkable if a license appears.

**Related paper, carried with its collision warning**: CacheRoute
([arXiv:2608.19677](https://arxiv.org/abs/2608.19677), planned
prefix-affinity routing — 176±11 QPS on 60 H100s, KV hit rate
64.1±1.3% → 93.2±0.5%) demonstrates what cache-affinity placement adds at
serving scale. Two binding notes: (a) **the 259-star
`AstraNetLab/CacheRoute` repo is NOT that paper's code** — same domain,
different mechanism, no arXiv reference; always cite the arXiv ID and
never adopt the name; (b) the paper's own caveat is load-bearing — two
32B counterexample workloads reduce or erase the improvement, and the
authors "recommend gating any deployment with a shadow replay rather than
enabling affinity from workload statistics alone." Any future
cache-affinity placement work inherits that shadow-replay gate as a
requirement.

**What exists in-repo today — checked, not assumed**: no tool-schema or
prefix cache exists anywhere in the repo. `crates/ruvector-query-cache`
(ADR-301) caches ANN *query results*; `crates/ruvllm` has KV-cache
machinery at the inference layer; `docs/adr/ADR-011-prefix-caching.md` is
Proposed and unimplemented. MCP tool schemas are defined and served from
`crates/mcp-gate` (the schema-serving gate), `crates/mcp-brain`,
`crates/ruvector-cli`'s MCP server, and `crates/rvAgent/rvagent-mcp`.
The cache surface this ADR creates is genuinely net-new.

**Why this matters to this program**: every agent context assembled today
re-serializes thousands of identical schema tokens per invocation, and
re-prefills them whenever ordering changes. ReCache's core insight —
composition-invariant, independently reusable per-resource representations
— applies at the context-assembly layer even before any KV-level
integration: give every resource a stable identity, compile its
representation once, and assemble contexts from cached blocks.

**Preprint-reproduction rule** (standing): ReCache's 3.655×/92.43%
figures are the paper's own, at small-model scale, for full KV-level
reuse. This ADR's acceptance bar is the program's own measured ≥2× TTFT
reduction (ruv's Wave-4 acceptance test) on our own assembly path against
our own uncached baseline — never the paper's numbers.

## Decision

Build a content-addressed schema-resource cache in `crates/mcp-gate`:

1. **Stable resource identity.** Every tool schema, skill, agent role,
   policy, and MCP schema gets a content-addressed `ResourceId` — SHA-256
   over a canonical encoding of the **resolved** content (mirroring
   ADR-320's `ObservationId` discipline). Identity binds to what the
   resource actually is, never to a self-declared name, version string,
   or claimed hash — the Wave-3 #887 lesson (hashing self-declared
   strings is a false safety claim) is a design constraint here, not an
   afterthought.
2. **Compile once, reuse everywhere.** A resource's reusable
   representation (canonical serialized block, token-count metadata,
   optional compression) is compiled once per `ResourceId` and stored in
   the cache; identical content shares one entry regardless of which
   agent, ordering, or context requests it — the context-assembly
   analogue of ReCache's composition-invariant KV blocks.
3. **Assembly from blocks.** Agent context assembly requests resources by
   `ResourceId` and concatenates cached position-independent blocks
   rather than re-serializing schemas per invocation. Ordering changes
   re-order block references; they do not invalidate entries.
4. **Downgrade-only hit accounting.** Hit/miss/partial-hit accounting is
   engineered so misclassification can only *understate* cache benefit:
   a miss or partial hit never scores as a hit, and any accounting
   malfunction fails loud rather than scoring optimistically — the same
   metric-integrity posture HarnessRisk established. The Wave-4
   acceptance measurement (≥2× lower TTFT) is taken against a real
   uncached baseline on the identical assembly path.
5. **Honest scope: context-assembly now, KV-level later.** This ADR
   delivers assembly-level caching and its TTFT-proxy benchmark. True
   KV-block reuse inside `crates/ruvllm`'s inference path (the paper's
   full mechanism) is explicitly follow-up work, gated on its own
   research-gate delta.
6. **Pattern reimplementation only.** No code from `EIT-NLP/ReCache` is
   copied, ported, or vendored while its repo carries no license.

## Consequences

### Positive

- Directly targets the Wave-4 acceptance test's second clause with a
  measurable, honestly-accounted TTFT reduction on a surface this repo
  fully controls.
- Establishes the resource-identity primitive that every later layer
  (KV-block reuse, cache-affinity placement, cross-agent sharing) needs
  anyway; combined with CacheRoute-pattern placement it is the path
  toward a content-addressable cognitive cache for agents.
- Content-addressing gives schema distribution an integrity property for
  free: a block's identity proves its content.

### Negative

- Assembly-level caching cannot reach the paper's 92% KV-memory figure —
  that requires the ruvllm follow-up; expectations are bounded
  accordingly and the ADR says so.
- A new cache is a new trust surface (see gates) and a new invalidation
  problem; stale schema blocks after a legitimate tool update must be
  handled by identity change, and anything that caches derived state
  keyed on mutable inputs must re-derive on input change.
- The unlicensed reference repo means no code reuse — implementation
  effort is higher than the "code released" headline suggests.

## Security / Validation Gates

- **Bind-to-resolved-content rule (blocking)**: no cache entry may be
  created or served whose identity was not computed from the resolved
  bytes being served; a self-declared hash is never trusted.
- **Poisoned-cache defense**: cache writes go through signature/source
  validation at the choke point; a block failing re-verification on read
  is evicted and the event fails loud, never served silently.
- **Downgrade-only accounting** (Decision §4), blocking not advisory —
  the 2× claim must be unfakeable by construction.
- **Tenant isolation**: cached blocks are namespaced; a resource compiled
  under one tenant is never served to another (mirrors ADR-320's tenant
  boundary).
- **Standard repo gate**: `npx @claude-flow/cli@latest security scan`
  after the cache write path lands (this is Wave 4's `security`-labeled
  work package).

## Affected Repos

- `ruvnet/ruvector` only this wave — `crates/mcp-gate` (new module +
  tests + TTFT benchmark). `crates/ruvllm` KV-block integration and any
  ruflo-side adoption of cached schema blocks are follow-up items
  coordinated through the epic, not this ADR.

## Dependencies

None hard among Wave-4 ADRs. Mirrors ADR-320's content-addressing
discipline; uses ADR-301's cache-variant benchmarking pattern (NoCache
baseline vs. cached) for its own measurement design.

## Alternatives Considered

- **Cache inside `ruvector-query-cache`**: rejected — that crate caches
  ANN query results with similarity semantics; schema blocks need exact
  content-addressed identity, and conflating the two surfaces would blur
  ADR-301's measured claims.
- **Go straight to KV-level reuse in ruvllm**: rejected for this wave —
  the paper's KV mechanism is validated only at 1.7B/4B scale, the
  acceptance test is satisfiable at the assembly layer this repo fully
  controls, and the assembly-layer identity primitive is prerequisite
  work for the KV layer anyway.
- **Adopt the upstream repo as a dependency**: rejected — no LICENSE
  file; legally unusable until that changes.
- **Name the module "ReCache"**: rejected — name-collision discipline;
  the module is the schema-resource cache, cited to arXiv:2608.19662.
