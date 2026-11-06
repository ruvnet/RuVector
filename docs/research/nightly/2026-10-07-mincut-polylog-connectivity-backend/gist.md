# Root-causing a dynamic min-cut engine's latency: why the "obvious" fix was never the bottleneck

## The setup

`ruvector-mincut` is an in-tree, from-scratch implementation of a December
2024 subpolynomial dynamic minimum-cut paper. `ruvector-agent-memory`
tried to reuse its high-level convenience API,
`RuVectorGraphAnalyzer::partition()`, to give agent-memory compaction a
structural "don't evict the bridge" signal (`MincutGatedForgetting`). That
attempt was rejected on two measured grounds: `partition()` scaled from
~77ms to ~11.4s across 50-400 vertex graphs, and — even setting speed
aside — it showed zero measurable bridge-protection benefit at the one
corpus size that was fast enough to test (84 memories).

A follow-up run fixed a real, separate correctness bug (non-deterministic
tie-breaking from unstable `HashMap`/`DashMap` iteration order) and closed
with a specific, concrete lead for the latency problem: the crate already
contains `PolylogConnectivity`, an implementation of a 2025 paper
guaranteeing O(log³n) expected worst-case update time, sitting unused next
to the slower `DynamicConnectivity` the wrapper actually uses. Swap it in,
the theory went, and the latency problem should improve.

It is a reasonable theory. It is also wrong, and figuring out *why* it is
wrong turned out to be more valuable than the swap itself would have been.

## Reading the code before trusting the theory

`MinCutWrapper` — the thing that sits underneath `partition()` — holds a
`conn_ds: DynamicConnectivity` field. Grep for where it's used, and there
are exactly two call sites: `insert_edge`/`delete_edge` (buffering updates)
and one `is_connected()` call, at the very top of `query()`, as a fast
path for "is the whole graph even connected?" That's it. It never touches
the actual cut search.

The actual cut search lives in `BoundedInstance`, and it works like this:
`MinCutWrapper` maintains roughly 100 `BoundedInstance`s, each responsible
for a geometric slice of possible cut values — instance *i* covers
`[⌊1.2^i⌋, ⌊1.2^(i+1)⌋]`. On a query, the wrapper walks instances starting
from 0, lazily building each one (feeding it every edge in the graph) and
asking it to find a cut in its range. The first instance that says "yes, I
found one in my range" wins; everything before it said "above my range,
try the next one."

Here's the part that matters: for small graphs (under 20 vertices),
`BoundedInstance::brute_force_min_cut` finds a cut by enumerating every
non-trivial subset of vertices (`2^n - 2` of them), computing each
subset's boundary, and keeping the global minimum. It does this
*completely independently of its own lambda range* — the range is only
consulted afterward, to decide whether the (already fully computed) global
answer happens to fall inside it. Walking the instance ladder doesn't
narrow the search in any way. It just repeats the exact same exhaustive
computation, from scratch, once per instance, until the precomputed answer
happens to land in whichever instance's range is currently being checked.

Instrumenting the wrapper confirmed it directly: for a 400-vertex graph,
*every one* of 16 geometric-range instances got built and queried before
the loop found its answer, and the per-instance costs (0.5-1.8 seconds
each) summed to the full measured 14-second latency. None of those 16
calls ever touch `conn_ds`. The thing that was supposed to be the
bottleneck was never in the loop.

## Measuring it anyway

Code-reading evidence is strong, but this process's own rule is "measure,
don't infer" wherever a measurement is affordable. So: build the same
graph once, run `partition()` under each backend in turn against that
*exact* graph (so the backend choice is the only thing that differs), and
repeat across five random seeds at five graph sizes (19, 50, 84, 100, 200
vertices — 84 matches the agent-memory benchmark's own corpus size).

The numbers land exactly where the code reading predicted: at every size,
the two backends' means differ by less than either one's own standard
deviation. At n=100, `Polylog` was actually marginally *faster* on
average than the baseline — a result with no systematic direction is
itself evidence that whatever's producing the ~15-20ms of run-to-run
variance isn't the connectivity backend. Running the real downstream
benchmark (`MincutGatedForgetting`'s acceptance test) under both backends
produces the same qualitative rejection either way: same 0% bridge-survival
gap, same ~1,700-2,000x slowdown, same REJECT.

## A bug found along the way

Writing a cross-backend equivalence test (insert a random sequence of
edges into both structures, assert they agree on every connectivity query)
turned up two real divergences — not in anything this run wrote, but in
code that was already sitting in the crate.

The first is narrow and harmless: querying whether a never-inserted vertex
is "connected to itself" returns `false` in one backend and `true` in the
other, because of how each one's internal `find()` handles an unknown key.
Neither implementation documents behavior for this case, and nothing in
the crate ever hits it in practice.

The second is a real bug. On a sequence that mixes edge insertions and
deletions, `PolylogConnectivity`'s delete path — which tries to find a
replacement edge to keep two halves of a cut tree connected after removing
a tree edge — can fail to find a replacement that actually exists, and
incorrectly concludes the graph split into two components. The baseline
backend, which does a full from-scratch rebuild on every deletion, gets it
right every time (it has to; a full rebuild has no shortcuts to get
wrong). This divergence is real, reproducible, and was sitting latent in
an already-shipped module. It doesn't affect anything currently merged in
this codebase — nothing deletes edges through this particular interface
today — but it's now a documented, tested, named constraint instead of an
undiscovered landmine.

## What actually happened here

A specific, named hypothesis ("swap the connectivity backend, latency
improves") was tested honestly and failed. That's not a wasted night: it
closed a concrete, previously-open research question with a definite
answer instead of leaving it to be re-investigated by a future session
that hadn't yet read the code closely enough to find the real cost center.
The actual bottleneck — a geometric-range instance ladder that redundantly
recomputes the exact same exhaustive calculation from scratch, once per
instance, with zero sharing of work between instances — is now identified
with direct evidence, not guessed at. That's the thread worth pulling
next.

## The honest scorecard

- **Claim tested**: swapping `PolylogConnectivity` in for
  `DynamicConnectivity` improves `partition()` latency/scaling. **Result:
  rejected**, with both code-path and measured evidence agreeing on why.
- **What got built anyway**: a real, selectable, tested
  `ConnectivityBackend` abstraction, off by default, because the backend
  itself is a legitimate alternative even though it doesn't solve this
  problem.
- **What got found that wasn't being looked for**: a reproducible
  correctness bug in `PolylogConnectivity`'s delete path, now documented
  and pinned by a regression test instead of being an undiscovered risk
  for whoever picks that backend next.
- **What's still broken**: `MincutGatedForgetting` is exactly as far from
  production-viable as it was before this run. The real fix — stop
  redundantly recomputing the same exhaustive search across ~16 instances
  — is unattempted, and is the obvious next experiment.

No fabricated numbers, no forced win, no quiet scope-narrowing to make the
result look better than it is. A falsified hypothesis, measured honestly,
with the real bottleneck identified along the way, is itself a complete
and useful night's work.
