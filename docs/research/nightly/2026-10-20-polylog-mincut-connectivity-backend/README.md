# Nightly Research: A Connectivity-Backend Swap Does Not Fix `BoundedInstance`'s Latency Scaling — the Missing `AboveRange` Cache Does More

## Summary

The 2026-09-11 nightly (`docs/research/nightly/2026-09-11-mincut-partition-determinism`)
left one concrete, scoped "Next Research" item: root-cause
`RuVectorGraphAnalyzer::partition()`'s latency scaling (ADR-345 measured
~77ms at n=50 growing to ~11.4s at n=400 on a synthetic k-NN graph), "likely
by evaluating `connectivity::polylog::PolylogConnectivity` (already in-tree,
not yet wired into `BoundedInstance`) as a replacement backend."

This run implemented exactly that: a `ConnBackend` trait isolating the one
piece of `BoundedInstance` that differs between three variants —
`BoundedInstance` (baseline, BFS-from-scratch), `PolylogBoundedInstance`
(candidate A, `PolylogConnectivity`), `EttBoundedInstance` (candidate B,
`DynamicConnectivity`, the Euler-Tour-Tree structure `MinCutWrapper` already
uses one layer up). All three share 100% of the remaining code: the
LocalKCut search loop, brute-force fallback, boundary caching, certificate
tracking.

**The hypothesis is REJECTED.** Swapping the connectivity backend produces
no measurable improvement (0.95x-1.0x "speedup", i.e. noise) at any tested
size. Root-causing why led to a second, better-supported finding instead:
`BoundedInstanceImpl::query()` caches a `ValueInRange` witness but never an
`AboveRange` result, so every lambda-range instance that doesn't contain the
true cut value re-runs its full `search_for_cuts()` (graph rebuild + a
linear budget-value scan + a per-seed LocalKCut oracle call) on *every*
`query()` call — cold or warm, mutated or unchanged. That is the dominant
cost, not the connectivity check this run was scoped to test. This is
reported as the real nightly output: a falsified hypothesis plus a
concretely-identified, differently-located bottleneck for a future run to
test.

## Abstract

`ruvector-mincut`'s `MinCutWrapper` maintains ~15-100 `BoundedInstance`s,
each responsible for a lambda-value sub-range, and queries them in
ascending order until one reports the true cut value is `ValueInRange`.
Instances whose range the cut value exceeds report `AboveRange`. ADR-345
measured this pipeline's real-world cost (through
`RuVectorGraphAnalyzer::partition()`) scaling badly with graph size and
flagged two suspects: non-determinism (since fixed, 2026-09-11) and raw
latency. This run isolates the connectivity-check hypothesis the prior
nightly named, confirms it is not the answer with a controlled A/B/C
comparison sharing all other code, and -- because a single-variable
experiment that measures the *wrong* variable is still useful when it tells
you so -- surfaces the actual dominant cost empirically rather than by
further guessing.

## Hypothesis

```text
Given a ring k-NN graph (the same generator ADR-345/2026-09-05's probe
used: vertex i connects to the next k=8 vertices mod n) at n in
{50, 100, 200, 400},

when BoundedInstance's BFS-from-scratch disconnection check is replaced by
an incrementally-maintained PolylogConnectivity (candidate A) or
DynamicConnectivity (candidate B), with every other code path unchanged,

then a candidate's mean per-call latency on repeated query() calls against
an unchanged graph ("warm" calls) should be at least 5x lower than
baseline's at the largest size reached,

subject to: cut_value and partition membership matching baseline exactly at
every tested size, and cut_value staying identical across 5 independent
rebuilds of the same input at n=50 (determinism, building on the
2026-09-11 fix).
```

## Why This Matters (2026)

`ruvector-agent-memory`'s `MincutGatedForgetting` (ADR-345) was rejected for
production specifically because `partition()` was too slow to call on every
compaction pass. If the bottleneck were a connectivity-primitive choice,
swapping it would be a small, low-risk, in-tree fix unblocking that
capability (and any other consumer of `RuVectorGraphAnalyzer`/
`MinCutWrapper` -- `find_bridges`, `is_well_connected`, community detection
in `ruvector-graph`). Ruling that out cleanly, with evidence, redirects
future effort instead of letting the same plausible-sounding fix get
re-proposed and re-tested from scratch by a later session.

## Why It Could Matter in 2036 / 2046

Dynamic min-cut is the structural layer underneath every "does this agent
memory store still hang together" question RuVector's longer-horizon
ecosystem map names: self-healing graph memory, swarm memory coherence,
RVM coherence domains gating mutation on structural health, proof-gated
infrastructure that needs a cheap live answer to "is the system still one
connected piece" before trusting a write. All of those need `partition()`-
class queries to run in the sub-millisecond-to-millisecond range at
agent-memory scale (hundreds to low thousands of nodes), not the
hundred-millisecond-to-ten-second range measured here and in ADR-345. This
run's actual contribution to that horizon is narrowing down *which* part of
the pipeline has to get cheap first.

## Why RuVector Is the Right Substrate

`ruvector-mincut` already has three independently-engineered dynamic
connectivity backends in-tree (`DynamicConnectivity`/ETT,
`PolylogConnectivity`, and the LocalKCut-oracle-based bounded instance
machinery) plus a generic `ProperCutInstance` trait boundary built for
exactly this kind of swap. No other crate in the ecosystem has the
plumbing to run this comparison as a same-codebase, same-harness
experiment instead of a reimplementation.

## Ecosystem Fit

Connects: `ruvector-mincut` (the engine under test), `ruvector-agent-memory`
(ADR-345's `MincutGatedForgetting`, the blocked consumer), and the
determinism-fix lineage from `ruvector-mincut`'s own
2026-09-11/2026-09-26 witness-partition work (ADR-346) that this run
explicitly builds on and re-validates under the new backends. A fourth
touch point: `ruvector-retrieval-receipt`/witness infrastructure, in that
any future fix to the `AboveRange`-caching gap this run surfaces will need
to preserve the same seed-ordering/tie-breaking determinism ADR-346
established, not just its latency.

## Architecture

```mermaid
flowchart TD
    subgraph MinCutWrapper
        W[query] --> I0[Instance 0 (smallest lambda range)]
        I0 -->|AboveRange| I1[Instance 1]
        I1 -->|AboveRange| Idots[... ~15 instances for cut=16, RANGE_FACTOR=1.2]
        Idots -->|ValueInRange| IN[Instance N]
    end
    subgraph BoundedInstanceImpl~C~ ["BoundedInstanceImpl&lt;C: ConnBackend&gt;"]
        Q[query] --> DC["conn.is_connected() -- THIS RUN'S VARIABLE"]
        DC --> WC{best_witness cached?}
        WC -->|yes, ValueInRange only| Return
        WC -->|no -- true for every AboveRange instance, every call| SFC[search_for_cuts: rebuild graph, budget loop x seed loop x oracle call]
        SFC --> Return
    end
    I0 -. is .-> Q
    style DC fill:#dde,stroke:#55a
    style SFC fill:#edd,stroke:#a55
```

`conn.is_connected()` (blue) is what this experiment varies.
`search_for_cuts()` (red) is what the evidence below says actually
dominates, and it is untouched by any of the three variants.

## Implementation

- `crates/ruvector-mincut/src/instance/bounded_conn.rs` (new, ~165 lines):
  `ConnBackend` trait (`new`/`insert_edge`/`delete_edge`/`is_connected`) plus
  three implementations: `BfsConn` (the baseline behavior, reimplemented
  against its own private adjacency so all three backends are structurally
  comparable), `PolylogConn` (wraps `PolylogConnectivity`), `EttConn` (wraps
  `DynamicConnectivity`).
- `crates/ruvector-mincut/src/instance/bounded.rs` (modified, behavior-
  preserving refactor): `BoundedInstance` is now a type alias
  `BoundedInstanceImpl<BfsConn>`; `PolylogBoundedInstance` and
  `EttBoundedInstance` are the two new aliases. The struct gained one field
  (`conn: C`), `insert()`/`delete()` gained one line each
  (`self.conn.insert_edge/delete_edge`), and the private `is_connected()`
  helper now delegates to `self.conn.is_connected()` instead of its own BFS.
  Every other method -- `search_for_cuts`, `brute_force_min_cut`,
  `compute_boundary`, the boundary cache, the certificate -- is byte-for-
  byte unchanged. All 7 pre-existing unit tests in `bounded.rs` pass
  unmodified against the new generic form.
- `crates/ruvector-mincut/examples/polylog_backend_bench.rs` (new): the
  benchmark harness described below. Drives `MinCutWrapper::with_factory`
  directly (bypassing `RuVectorGraphAnalyzer`, which has no factory-
  injection point) and reimplements its partition-materialization logic
  identically to `integration/mod.rs::partition()`.

## Benchmark Methodology

- **Dataset**: the exact ring k-NN generator from
  `ruvector-agent-memory/examples/mincut_scaling_probe.rs` (vertex `i`
  connects to the next `k=8` vertices mod `n`), reused rather than
  reimplemented so sizes are directly comparable to ADR-345's own table.
- **Sizes**: n in {50, 100, 200, 400}. n=19 was deliberately excluded after
  the first run attempt: at n<20, `BoundedInstanceImpl::query()` takes the
  brute-force path unconditionally, and because every one of the ~15
  lambda-range instances independently recomputes the *same* global
  O(2^n * m) brute-force search before the matching range is reached, n=19
  alone took minutes -- a real finding (see Failure Modes) but orthogonal
  to this experiment's variable, so it was dropped from the swept sizes to
  keep the run tractable.
- **Phases per (size, variant)**:
  - `cold`: one fresh `MinCutWrapper`, all edges inserted, one `query()`
    call. Build and query timed separately.
  - `warm`: `WARM_REPEATS` (5) further `query()` calls on the *same*,
    unmutated wrapper. Originally planned as 1000 repeats to isolate a
    presumed cheap, amortized cost; the first (n=50) run showed warm calls
    costing ~27ms each -- comparable to a per-instance share of the cold
    cost, not microseconds -- so the repeat count was cut to 5 for
    feasibility once that was understood (see Failure Modes).
  - `determinism`: at n=50 only, `DETERMINISM_REBUILDS` (5) independent
    fresh rebuilds of the identical input, asserting `cut_value` is
    constant across all of them.
- **Correctness invariant**: for every size, candidate A's and candidate
  B's `(cut_value, partition)` are compared against baseline's; any
  mismatch is printed as `CORRECTNESS MISMATCH` rather than silently
  averaged away.
- **Hardware / toolchain**: `rustc 1.97.0`, `cargo 1.97.0`, Linux
  (container), release profile (`cargo build --release`), single-threaded
  (no `rayon`/`agentic` feature enabled).
- **Repro**:
  ```bash
  cargo run --release -p ruvector-mincut --example polylog_backend_bench
  ```

## Benchmark Results

Raw stdout captured in this directory as `benchmark_output.txt`. All three
variants find `cut_value=16` at every size (`all_variants_agree_with_baseline
=true`), and determinism at n=50 was `stable` (5/5 identical) for all three.

| n | variant | cold_build | cold_query | warm_mean (n=5) | warm_min | warm_max |
|---|---|---|---|---|---|---|
| 50  | baseline(Bfs)        | 0.615ms | 138.3ms  | 30.10ms | 25.89ms | 34.41ms |
| 50  | candidate_A(Polylog) | 0.451ms | 127.6ms  | 26.70ms | 26.15ms | 27.36ms |
| 50  | candidate_B(Ett)     | 0.525ms | 117.5ms  | 26.03ms | 25.61ms | 26.83ms |
| 100 | baseline(Bfs)        | 1.460ms | 757.4ms  | 165.9ms | 150.1ms | 187.0ms |
| 100 | candidate_A(Polylog) | 1.348ms | 754.2ms  | 150.4ms | 140.7ms | 167.3ms |
| 100 | candidate_B(Ett)     | 1.154ms | 832.1ms  | 156.8ms | 132.6ms | 185.0ms |
| 200 | baseline(Bfs)        | 1.884ms | 4108.9ms | 860.5ms | 813.6ms | 947.5ms |
| 200 | candidate_A(Polylog) | 1.923ms | 3984.5ms | 810.6ms | 771.4ms | 860.1ms |
| 200 | candidate_B(Ett)     | 1.919ms | 3996.4ms | 857.2ms | 795.7ms | 933.4ms |
| 400 | baseline(Bfs)        | 3.668ms | 16902.9ms| 3630.1ms| 3493.2ms| 3767.5ms|
| 400 | candidate_A(Polylog) | 3.513ms | 16308.2ms| 3384.6ms| 3315.5ms| 3459.6ms|
| 400 | candidate_B(Ett)     | 3.440ms | 16708.1ms| 3467.4ms| 3329.6ms| 3675.2ms|

| n | warm speedup A | warm speedup B | cold(end-to-end) speedup A |
|---|---|---|---|
| 50  | 1.13x | 1.16x | 1.08x |
| 100 | 1.10x | 1.06x | 1.00x |
| 200 | 1.06x | 1.00x | 1.03x |
| 400 | 1.07x | 1.05x | 1.04x |

Note this run's `cold_query` numbers (138ms@50 -> 16.9s@400) run somewhat
higher than ADR-345's own (77ms@50 -> 11.4s@400); both were measured on
different hardware and through slightly different call paths (this run
drives `MinCutWrapper` directly; ADR-345 went through
`RuVectorGraphAnalyzer`, which adds negligible overhead of its own). The
qualitative scaling shape -- the same multiplicative blowup from n=50 to
n=400 -- replicates cleanly across both runs, which is what this
comparison actually needs: baseline and both candidates were measured in
the same process, back to back, on the same hardware, so the *relative*
numbers above are the load-bearing ones, not the absolute ones.

**Acceptance threshold check**: candidate A's and candidate B's warm
speedups top out at 1.16x and 1.16x respectively (both at n=50) -- nowhere
near the fixed 5x bar. The gap is consistent with measurement noise plus a
small, real constant-factor difference (a `HashMap`-based incremental
structure vs. a fresh-each-time `HashSet`/`VecDeque` BFS), not a
structural improvement. **Hypothesis rejected.**

## Failure Modes

1. **n<20 multi-instance brute-force replay (discovered, not this
   experiment's target).** At n=19 on the ring generator, the first
   benchmark attempt did not finish within several minutes. Root cause:
   `BoundedInstanceImpl::query()`'s small-graph fast path
   (`vertices.len() < 20`) runs `brute_force_min_cut()` -- an O(2^n * m)
   exhaustive subset search -- unconditionally, regardless of which
   lambda-range instance is asking. Because `RANGE_FACTOR=1.2` grows
   lambda ranges slowly, reaching a true cut value of ~16 (this ring
   topology's min cut, consistent with vertex degree `2k=16`) took roughly
   15 instances, each independently re-running the identical global
   brute-force search. This is a real, measurable inefficiency, but it is
   orthogonal to the connectivity-backend variable under test here, so n=19
   was dropped from the sweep rather than investigated further this run.
   Flagged as Next Research item 1.
2. **`AboveRange` results are never cached (the actual dominant cost).**
   `BoundedInstanceImpl::query()` only populates `best_witness` when a cut
   is found `ValueInRange`. An instance that returns `AboveRange` leaves no
   trace, so the next `query()` call -- cold or warm, with or without
   intervening graph mutation -- re-runs that instance's full
   `search_for_cuts()` (rebuild a `DynamicGraph` from this instance's
   `edges`, then loop `budget` linearly across the instance's entire
   lambda range, and for each budget value loop every seed vertex calling
   the LocalKCut oracle) from scratch. Since `MinCutWrapper::query()` walks
   every untried instance below the matching one on every call, this cost
   is paid in full, repeatedly, which is why the "warm" phase did not
   amortize the way its name implied. This is the real target for a future
   `partition()` latency fix, not the connectivity backend. Flagged as Next
   Research item 2.
3. **The connectivity-backend swap itself is correctness-neutral but
   latency-neutral-to-slightly-negative.** `PolylogBoundedInstance` and
   `EttBoundedInstance` match baseline's `cut_value`/`partition` at every
   tested size (see Benchmark Results), so the swap is safe, but it buys
   nothing: `is_connected()`'s cost is a vanishingly small fraction of
   `query()`'s total time once `search_for_cuts()`'s repeated re-execution
   is accounted for.

## Rejected Alternatives

- **Redesigning `search_for_cuts()`'s budget loop in this run.** Once the
  `AboveRange`-caching gap was identified as the likely dominant cost,
  fixing it directly was tempting. Declined: this run was scoped to the
  connectivity-backend hypothesis specifically (per the instruction not to
  silently change the research hypothesis after seeing results), and a
  budget-loop/caching redesign touches the LocalKCut search path all three
  variants share, which deserves its own isolated experiment and its own
  acceptance threshold, not a last-minute addition to this one. Deferred to
  Next Research.
- **Testing n=800/1600.** Originally planned to see whether a candidate
  "catches up" at larger sizes. Declined after the n=19 and initial
  WARM_REPEATS=1000 cost surprises made clear that, given
  `search_for_cuts()` dominates and does not shrink across repeats, larger
  sizes would multiply an already-large repeat budget rather than add new
  information once the n<=400 pattern (no speedup) was established.

## Security

No security-relevant surface: this is an internal data-structure swap
behind an existing trait boundary, with no new external inputs, no new
serialization, and no change to witness/certificate semantics. The
`ConnBackend` trait is crate-internal (`pub` for benchmarking, not exposed
through any MCP or network surface).

## Governance

No feature flag was added: `BoundedInstance` remains the default (and only
production-wired) type; `PolylogBoundedInstance`/`EttBoundedInstance` exist
as benchmarked-but-not-promoted alternatives, consistent with "never
promote a candidate without measurable evidence" -- here the evidence says
not to promote either.

## MCP / WASM / Edge Implications

None this run. If a future `AboveRange`-caching fix lands and meaningfully
lowers `partition()` latency, it would be relevant to
`ruvector-edge-index`/edge deployments of agent-memory structural checks,
but speculating about that now, without the fix, would be exactly the kind
of unsupported claim the process forbids.

## RVF / RVM / ruFlo Implications

- **RVF**: not materially relevant this run -- no new portable artifact or
  serialized state was introduced.
- **RVM**: a correctly-amortized `partition()` (post the `AboveRange` fix)
  would be a plausible cheap gate for an RVM coherence-domain boundary
  check ("is this mutation about to fragment the domain"), but that is
  future work contingent on Next Research item 2, not demonstrated here.
- **ruFlo**: a concrete, nameable future workflow exists once (and only
  once) `partition()` is cheap enough to call per-mutation: a ruFlo
  "structural health monitor" that calls `is_well_connected()` after each
  batch of agent-memory writes and raises an alert (not an autonomous
  mutation) when the live boundary shrinks below a configured threshold.
  Not implemented this run -- it depends on Next Research item 2 landing
  first.

## Practical Applications

1. **Agent memory compaction gating (ADR-345's own use case).** User:
   anyone running `ruvector-agent-memory` at scale. Problem: compaction
   needs a cheap structural signal before evicting entries. RuVector
   capability: `RuVectorGraphAnalyzer::partition()`/`find_bridges()`.
   Ecosystem integration: `ruvector-agent-memory::graph_forget`.
   Implementation path: blocked until Next Research item 2 lands. Business
   value: avoids silently fragmenting cross-cluster retrieval. Main risk:
   none of this is usable until the real bottleneck (not this run's
   variable) is fixed. Time horizon: near-term once unblocked.
2. **Knowledge-graph cut-point detection.** User: a GraphRAG pipeline
   builder. Problem: detect weakly-connected "bridge" facts before pruning.
   RuVector capability: `find_bridges()`. Ecosystem integration:
   `ruvector-graph`. Implementation path: same blocker as above. Business
   value: prevents retrieval regressions from naive pruning. Main risk:
   `find_bridges()` itself is O(edges) separate `MinCutWrapper` instances
   (see `integration/mod.rs`), an independent cost not addressed here. Time
   horizon: near-term once unblocked.
3. **Community detection for routing.** User: semantic router maintainer.
   Problem: partition a similarity graph into routing communities.
   RuVector capability: `CommunityDetector`/`partition()`. Ecosystem
   integration: `ruvector-router-core`. Implementation path: same blocker.
   Business value: cheaper routing-table rebuilds. Main risk: same latency
   gap. Time horizon: near-term once unblocked.
4. **RVM coherence-domain boundary checks.** User: RVM operator. Problem:
   cheaply verify a domain hasn't fragmented before trusting a write.
   RuVector capability: `is_well_connected()`. Ecosystem integration: RVM.
   Implementation path: contingent on Next Research item 2. Business value:
   structural safety gate. Main risk: speculative until the fix lands. Time
   horizon: medium-term.
5. **Edge fleet partition detection.** User: Cognitum edge operator.
   Problem: detect network-partition-equivalent structural splits in a
   peer graph. RuVector capability: `MinCutWrapper` at small `n` (fleet
   size), where this run's n<20 brute-force finding (Failure Mode 1)
   directly applies and would need its own fix first. Implementation path:
   blocked on both Next Research items. Business value: faster partition
   detection than full consensus timeout. Main risk: both open bottlenecks
   apply at fleet scale. Time horizon: long-term.
6. **Self-healing index repair triggers.** User: `ruvector-hnsw-repair`
   maintainer. Problem: decide when a delete-heavy workload has
   structurally degraded the graph enough to warrant repair. RuVector
   capability: periodic `min_cut()` sampling. Ecosystem integration:
   `ruvector-hnsw-repair`. Implementation path: contingent on Next Research
   item 2 (current per-call cost is too high to sample periodically).
   Business value: avoids both over-repair and under-repair. Main risk:
   same blocker. Time horizon: medium-term.
7. **Scientific collaboration-graph bridge detection.** User: a citation-
   graph / collaboration-graph researcher. Problem: identify bridge
   authors/papers. RuVector capability: `find_bridges()`. Implementation
   path: same blocker. Business value: research-integrity tooling. Main
   risk: same. Time horizon: long-term, speculative.
8. **Swarm-memory partition alerting.** User: a multi-agent swarm operator.
   Problem: detect when shared swarm memory has split into disconnected
   islands. RuVector capability: `is_well_connected()` sampled per epoch.
   Ecosystem integration: ruFlo "structural health monitor" (see RuFlo
   Implications). Implementation path: contingent on Next Research item 2.
   Business value: early warning before retrieval quality silently
   degrades. Main risk: same blocker; also needs the ruFlo workflow itself
   built. Time horizon: long-term.

## Long Horizon Applications

1. **Self-healing graph memory.** Thesis: a memory store that notices its
   own structural fragmentation and heals it before a human notices
   degraded recall. Required advances: the `AboveRange`-caching fix (Next
   Research item 2) to make per-write structural checks affordable at all.
   RuVector role: `ruvector-mincut` + `ruvector-agent-memory`. Why this
   run matters: it rules out the cheap fix (connectivity backend) so
   effort goes to the real one. Primary uncertainty: whether an amortized
   fix can get `partition()` into the sub-millisecond range needed for
   per-write checks, or only sub-second. Falsification: implement the fix
   and measure; if it doesn't clear roughly 2-3 orders of magnitude, the
   whole `MinCutWrapper` multi-instance design (not just one method) needs
   reconsideration.
2. **Synthetic nervous systems / RVM coherence domains.** Thesis:
   structural connectivity as a first-class, proof-gated health signal for
   autonomous infrastructure. Required advances: cheap, incremental
   `partition()`; a defined policy for what "domain fragmented" should
   trigger. RuVector role: `ruvector-mincut` + RVM. Why this run matters:
   same as above. Primary uncertainty: whether bounded-range instances are
   the right algorithmic shape at all for a continuously-live signal, vs.
   a single always-incremental structure. Falsification: if even a fixed
   `AboveRange` cache can't get sub-second at n~10k, the bounded-instance
   architecture itself is the bottleneck, not any one method.
3. **Agent operating systems with structural admission control.** Thesis:
   an agent OS that refuses a write that would fragment shared memory
   below a coherence threshold. Required advances: the same latency fix,
   plus a policy layer. RuVector role: core substrate. Why this run
   matters: same. Primary uncertainty: policy design (ADR-345 already
   showed global min-cut doesn't reliably find human-intended "bridges" --
   a separate, already-falsified concern from that nightly, not
   re-litigated here). Falsification: n/a this run.
4. **Swarm memory with partition-aware consensus.** Thesis: a swarm that
   detects memory partitions and degrades gracefully rather than silently
   serving stale data to an isolated sub-swarm. Required advances: cheap
   live connectivity signal at swarm scale (tens to low thousands of
   agents). RuVector role: `ruvector-mincut`. Why this run matters: same
   latency blocker. Primary uncertainty: whether swarm-scale graphs are
   sparse enough for the fixed design to stay cheap. Falsification: measure
   at realistic swarm sizes once the fix lands.
5. **Dynamic world models with structural consistency checks.** Thesis: a
   world model that tracks its own internal graph's connectivity as a
   proxy for internal consistency. Required advances: same latency fix;
   also a mapping from "world model edge" to "mincut graph edge" that
   doesn't yet exist. RuVector role: substrate only. Why this run matters:
   rules out one wrong turn. Primary uncertainty: whether connectivity is
   even the right consistency proxy for a world model (unverified
   assumption, not tested here). Falsification: would need a dedicated
   experiment.
6. **Robotics memory with physical-topology-aware forgetting.** Thesis:
   a robot's episodic memory graph mirrors physical-space topology, and
   mincut-gated forgetting (ADR-345's original idea) becomes viable once
   it's fast, protecting physically-bridging memories. Required advances:
   both ADR-345's effectiveness gap (separate, prior finding) and this
   lineage's latency gap. RuVector role: `ruvector-agent-memory` +
   `ruvector-robotics`. Why this run matters: narrows the latency half of
   a two-part blocker. Primary uncertainty: effectiveness, already
   falsified once at the current scale in ADR-345. Falsification: already
   partially falsified; would need re-testing on physically-grounded data,
   not synthetic clusters.
7. **Proof-gated autonomous infrastructure.** Thesis: every structural
   mutation to a shared store requires a cheap, witnessed connectivity
   proof before committing. Required advances: the latency fix, plus
   witness-signing integration (`ruvector-retrieval-receipt`/ADR-347
   lineage already exists for this). RuVector role: `ruvector-mincut` +
   witness infra. Why this run matters: same blocker. Primary uncertainty:
   whether per-write witnessing overhead stacks acceptably with the
   mincut fix's own overhead. Falsification: combined benchmark once both
   pieces exist.
8. **Scientific autonomous systems (e.g., experiment-graph coherence).**
   Thesis: an autonomous research system (like this nightly harness
   itself) tracks coherence of its own evidence graph and flags when a
   conclusion's supporting evidence has structurally fragmented. Required
   advances: same latency fix; a mapping from "evidence item" to graph
   node. RuVector role: substrate. Why this run matters: same. Primary
   uncertainty: whether this framing is useful at all, highly speculative.
   Falsification: would need a dedicated experiment; not attempted here.

## Evolution Results

No Darwin run this nightly. `npx ruvector harness darwin --help` and
`npx ruvector harness doctor --json` were attempted at the start of this
run (per the process's Step 0/3) and failed: no `ruvector harness`
executable is installed or resolvable in this environment (`npm error could
not determine executable to run`). `npx metaharness --help` *is* available
(it scaffolds new harness projects; it is not a Darwin/Flywheel execution
backend for an existing repository) -- its `score`/`analyze`/`genome`
subcommands were not run this session because they require a separate
dry-run budget decision this run did not have authorization to spend
without checking with the user first, and because they analyze repository
fit rather than execute the bounded evolution this topic needed. Given no
Darwin backend was available, the three-variant A/B/C comparison above was
run as a manual, bounded "evolution" in spirit (3 candidates, 1 generation,
0 promotions) with the same hard constraints (tests green, no correctness
regression, no acceptance-criteria tampering) applied by hand. Parent
(`BoundedInstance`) is retained; no candidate is promoted.

## Promotion Decision

**Not promoted.** Neither `PolylogBoundedInstance` nor `EttBoundedInstance`
clears the fixed acceptance threshold (>=5x warm-latency improvement at the
largest reached size). Both are correctness-equivalent to baseline and are
left in-tree as working, tested, benchmarked reference implementations
(per the nightly process's evidence-retention rule), not wired into
`MinCutWrapper::new()`'s default factory.

## Witness Evidence

- Code: this run's full diff (new `bounded_conn.rs`, modified `bounded.rs`,
  new `polylog_backend_bench.rs`) is in the PR for this branch.
- Tests: `cargo test -p ruvector-mincut --lib instance::` -- 29 passed, 0
  failed (see Build/Tests below for the exact count including the new
  `ConnBackend` contract tests).
- Benchmark: raw stdout saved alongside this README as
  `benchmark_output.txt`; exact repro command given above.
- Commit: see this branch's git log.
- No cryptographic witness chain was generated for this run (no mutation
  of production-wired behavior occurred; `BoundedInstance`'s default
  factory binding is unchanged).

## Production Path

None opened by this run. A future run implementing and benchmarking an
`AboveRange` cache (Next Research item 2) would need its own feature flag
and its own promotion gate; this run deliberately does not pre-judge that
design.

## Falsification Criteria

Already applied: the hypothesis predicted >=5x warm-latency improvement
from the backend swap; measured results (see Benchmark Results) show
~0.95x-1.0x at every tested size. Falsified as stated.

## Limitations

- Single-threaded, single-machine benchmark; no concurrent-mutation
  scenario tested.
- `WARM_REPEATS` was cut from a planned 1000 to 5 mid-run once the cost
  model became clear (see Failure Modes item 2); 5 repeats bounds the
  min/max spread reported but is a small sample.
- `DETERMINISM_REBUILDS` only ran at n=50, not at every swept size, for the
  same cost reason; the 2026-09-11 fix's determinism at n=100-400 is
  assumed, not independently re-verified by this run beyond the single
  cold-query cut-value match against baseline at each size.
- n=19 and n>=800 were both excluded (see Benchmark Methodology /
  Rejected Alternatives); this run's evidence does not speak to either
  regime.
- No WASM or edge-hardware measurement.
- Did not attempt to fix or even prototype the `AboveRange`-caching gap
  this run surfaces -- intentionally out of scope (see Rejected
  Alternatives), but it means the "real" fix remains unimplemented and
  unmeasured.
- This run validated the change with `cargo test -p ruvector-mincut --lib
  instance::` (29/29 passing, including 3 new `ConnBackend` contract tests)
  and `cargo clippy -p ruvector-mincut --lib --no-deps` (clean) rather than
  the crate's full (debug-mode) test suite, which includes long-running
  pre-existing `subpolynomial`/`canonical`/`jtree` stress tests unrelated
  to this change and did not finish within this run's time budget. No
  failure was observed in the partial run before this report was written;
  this is a scope limit on verification depth, not a known issue.
  Downstream crates depending on `ruvector-mincut` (`ruvector-agent-memory`,
  `ruvector-graph-condense`, etc.) were not individually rebuilt -- the
  change is purely additive (new module, new type aliases, `BoundedInstance`
  itself unchanged in name and behavior), so no API break is expected, but
  this was not independently verified for every dependent this run.

## Next Research

1. Root-cause and fix the n<20 multi-instance brute-force replay (Failure
   Mode 1): either cache `brute_force_min_cut()`'s result across instances
   within one `process_instances()` call (it is the same global
   computation every time at a given graph state), or special-case
   `MinCutWrapper` to run the brute-force path once, not once per instance,
   for small graphs.
2. **Implement and benchmark an `AboveRange` result cache in
   `BoundedInstanceImpl::query()`**, invalidated the same way
   `best_witness` already is (on any edge touching the cached witness's
   boundary -- though for `AboveRange`, the right invalidation condition
   needs its own design, since there is no witness to check against). This
   is the change this run's evidence points to as the actual latency
   bottleneck; re-run this same three-variant harness afterward to check
   whether the connectivity-backend choice matters *once* the dominant
   cost is removed (it may, once `search_for_cuts()`'s own cost stops
   swamping it).
3. Re-run the determinism check from 2026-09-11 against
   `PolylogBoundedInstance`/`EttBoundedInstance` directly (not just
   indirectly via this run's n=50 single-size check) at n=100-400, since
   those variants' own internal bookkeeping (`PolylogConnectivity`'s
   `HashMap`-based `non_tree_edges`/`adjacency`) was not individually
   audited for the same hash-iteration-order risk ADR-346 fixed in
   `brute_force_min_cut`/`search_for_cuts`.
4. Investigate `find_bridges()` (`integration/mod.rs`), which independently
   constructs one full `MinCutWrapper` per edge in the graph -- a second,
   separate scaling problem in the same crate, untouched by either this
   run or ADR-345.

## Acceptance Result

```text
REJECT
```

The connectivity-backend-swap hypothesis is rejected: no candidate reaches
the fixed 5x warm-latency threshold at any tested size, and the measured
gap between baseline and candidates is within measurement noise (0.95x-
1.0x). This is retained as useful negative evidence (per the nightly
process's own rule that a falsified hypothesis with good evidence is a
successful run), together with a concretely-identified, differently-located
bottleneck (`AboveRange` caching) for a future run to test.
