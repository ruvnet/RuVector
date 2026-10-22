# Verifying a Partition Without Recomputing It: Signed Cut Certificates and Their Adversarial Limit

*A standalone technical write-up of a 2026-10-22 RuVector nightly research
run. No external gist was published — this environment has no `gh` CLI, so
this file is the artifact, kept in-repo at
`docs/research/nightly/2026-10-22-verified-partition-import/gist.md`
alongside the full research report.*

## The problem in one paragraph

Agent A computes a global minimum-cut partition over a shared graph —
maybe a memory-admission graph, maybe a namespace-merge decision. Agent B
needs that same partition, but B doesn't trust A's wire blindly, and B
doesn't want to pay the full cost of recomputing the cut from scratch
just to get the same answer A already has. Can B get a cheap guarantee
that what A sent is both *authentic* (actually from A, not tampered in
transit) and *correct* (actually the partition A claims it is)? This
write-up builds the obvious first answer — sign it — measures exactly how
much latency that buys, and then builds and measures the adversarial case
that the obvious answer cannot fix.

## Why "just sign it" isn't the whole story

A signature over a message proves one thing: whoever holds the signing
key produced that exact message. It says nothing about whether the
message's *content* is true. If agent A signs "the minimum cut is X,"
and B checks the signature and finds it valid, B has learned "A really
did send this," not "X is really the minimum cut." If A is buggy, or A's
key is held by someone who doesn't care about correctness, or A is simply
wrong, a valid signature does not catch it. This is not a subtle
cryptographic point — it is definitional — but it is easy to elide in a
design discussion that starts from "let's add a signature" and stops
there. This research run refuses to stop there: it builds the signature
scheme, measures its real speedup, and then deliberately constructs the
case where the signature scheme fails, and measures exactly how often and
why.

## The design

Three pieces, in `crates/ruvector-partition-import`:

**A deterministic test graph.** Real numbers need a real, reproducible
input. `graph::generate_graph(n, seed, avg_degree)` builds a connected
sparse weighted graph: a backbone of 2 edges per node to earlier nodes
(guaranteeing connectivity, so the minimum cut is never trivially zero)
plus random fill edges to a target average degree. Every edge weight is
drawn from a seeded PRNG. The first version of this generator had a real
bug — it iterated a `HashSet` whose per-instance hash seed varies even
within the same process, silently breaking reproducibility between two
calls with the same seed. Fixed by sorting before iterating; caught by a
test that calls the generator twice and compares digests.

**A reference minimum-cut algorithm.** To have ground truth to compare
against, the crate implements Stoer-Wagner — a well-known, simple,
`O(VE log V)`-ish algorithm for the global minimum cut of a weighted
graph — as a self-contained module (`baseline.rs`), deliberately *not*
integrated with this workspace's existing `ruvector-mincut` crate. That
crate is real and mature, but its API is built around dynamic
min-cut maintenance under edge updates and canonical tie-breaking among
multiple optimal cuts — a different, larger problem than "compute one
static cut, once, as a timing baseline." Reading its `certificate` and
`canonical` modules (1,767 lines combined) made clear that integrating
properly would take longer than the time available for this pass; a
self-contained reference algorithm isolates the actual hypothesis (import
latency) from that separate, larger integration question. This is a
statement about scope, not about `ruvector-mincut`'s quality.

**A signed certificate.** `certificate::CutCertificate` binds
`(SHA-256(graph), SHA-256(partition), cut_value)` under one Ed25519
signature. The signed message is always exactly 106 bytes — a fixed-size
digest triple, never the raw graph or partition — so signing and
verification cost never grows with graph size, which matters: if you
signed the raw bytes, you'd pay `O(N)` or worse on *every* signature
operation, eating into the exact latency advantage the design is trying
to buy.

**Two ways for B to use what A sent.** `candidate_a` checks the signature
and, if valid, accepts the partition outright — no recomputation at all.
`candidate_b` checks the signature and then samples a fraction `k` of the
graph's vertices, checking for each one whether moving it to the other
side of the partition would strictly reduce the cut. If it would, that's
proof the partition isn't actually a global minimum cut — a real,
necessary (not sufficient) mathematical condition, checkable in
`O(degree(v))` per sampled vertex. If any sampled vertex fails, `candidate_b`
rejects and falls back to full recomputation, same as an invalid signature.

## What "necessary, not sufficient" cost in practice

Building the spot-check surfaced a second real bug, and it's instructive.
A test asserted that `candidate_b` must accept the *actual* output of the
reference mincut algorithm — obviously true, you'd think, since it's a
genuine global minimum cut. The test failed. The reference algorithm had
found a global minimum cut that isolated exactly one vertex (a completely
legitimate outcome on a sparse random graph — sometimes the cheapest cut
really is "pull out the one vertex with the fewest, lightest edges"), and
the spot-check flagged that lone vertex as suspicious: moving it to the
other side would reduce the cut to zero. True, arithmetically. But "moving
the only vertex on a side to the other side" doesn't produce a competing
two-sided cut — it produces one side with nothing in it, which isn't a
cut at all. The fix excludes a vertex from the check when it's the sole
member of its side. This is exactly the kind of edge case that "necessary,
not sufficient" conditions hide until you actually try to use them against
real output, not just against hand-picked examples.

## The numbers

Everything below comes from one real `cargo run --release` execution of
the crate's benchmark binary, on a 4-vCPU container, `rustc 1.97.0`. Full
raw output is in the companion research report.

| N | baseline mean latency | candidate_a mean latency | speedup | candidate_b(k=0.5) mean latency |
|---|---|---|---|---|
| 500 | 54.5 ms | 125.4 us | 434x | 188.1 us |
| 2,000 | 1,003.9 ms | 473.2 us | 2,121x | 720.0 us |
| 8,000 | 21,923.5 ms | 1,302.4 us | 16,833x | 3,174.1 us |
| 20,000 | 156,929.3 ms | 4,800.8 us | 32,688x | 12,663.2 us |

Tamper-in-transit detection was 100% for both candidates at every size,
every trial. The accepted cut value matched the baseline's recomputed
cut value in every single honest-signer trial across all four sizes —
the benchmark process asserts this in-process and would have aborted on
any mismatch; it printed `benchmark complete.` instead.

The signed-but-suboptimal adversarial test (2% of vertices perturbed to a
genuinely worse partition, then honestly signed):

| N | candidate_a wrongly accepted | candidate_b caught, k=0.10 | k=0.25 | k=0.50 |
|---|---|---|---|---|
| 500 | 100% | 69.5% | 97.5% | 100% |
| 2,000 | 100% | 99.0% | 100% | 100% |
| 8,000 | 100% | 100% | 100% | 100% |
| 20,000 | 100% | 100% | 100% | 100% |

`candidate_a` wrongly accepted the adversarial partition in literally
every trial, at every size — not a near-miss, exactly the structural
100% the design argument predicts. `candidate_b`'s detection rate rises
with both the sample fraction and the graph size, consistent with
"catching at least one of a fixed-percentage set of bad vertices gets
easier as the sample (and the bad set, in absolute terms) grows" — and
flattens to 100% once the sample is large enough relative to the
perturbation that missing every bad vertex becomes implausible. At
`N=500` and the smallest sample (`k=0.10`, 50 vertices sampled against 10
perturbed), the detection rate is visibly below 100% (69.5%) — the one
place in this run's data where the "necessary, not sufficient" caveat is
not just a theoretical disclaimer but a directly visible number.

**Verdict: ACCEPT**, against the threshold fixed before any number was
seen — candidate_a's speedup at the largest size (32,688x) clears the
required "at most 25% of baseline" bar by roughly three orders of
magnitude, tamper detection was 100% everywhere, and the accepted cut
value never once diverged from the baseline's in an honest-signer trial.

The scaling is the headline: Stoer-Wagner's own cost grows fast enough
(roughly `O(VE log V)`) that recomputing from scratch becomes genuinely
expensive at larger graphs, while signature verification and the
spot-check stay in the microsecond range regardless of graph size (both
are bounded by, respectively, `O(1)` and `O(k * N * avg_degree)`, neither
of which involves running a global-cut algorithm). That gap is exactly
what a signed certificate is for.

## The adversarial test, and what it actually shows

Here's the part that matters more than the speedup. Take the real, correct
partition. Perturb 2% of its vertices to the wrong side — deterministically,
checking that this actually makes the cut worse (it always did, on the
first try, across every size tested). Sign this *worse* partition exactly
the way an honest signer would. Hand it to both candidates.

`candidate_a` accepted it. Every single time, at every graph size —
100%. This is not noise or a tuning problem; it is the design working
exactly as specified. `candidate_a` has no mechanism that could ever
reject a validly-signed message, because validity of the signature is the
*only* thing it checks.

`candidate_b` caught it some of the time, and — this is the point — the
rate it caught it at depended directly on how many of the perturbed
vertices happened to land in the sample. Perturb 10 vertices out of 500
(2%), sample 10% of the graph (50 vertices), and you'll typically catch
several of the 10 bad ones just by chance, but not always all of them.
Sample half the graph, and you'll almost always catch at least one.
That's a probabilistic argument, not a guarantee, and the measured numbers
bear it out — see the detection-rate figures in the results above and in
the research report's full discussion. An adversary who perturbed fewer
vertices, or who could predict which vertices get sampled, would evade
`candidate_b` more often than this run's fixed 2% perturbation shows.

## What this is actually good for

Not "replacing trust with cryptography" in general — that's not what
happened here. What happened is narrower and more honest: a cheap,
real, measured way to skip redundant computation when you already trust
the signer for correctness (not just authenticity), plus a cheap, real,
measured, partial defense if you don't fully trust the signer but can't
afford full recomputation every time either. Which of those two
situations you're in is a judgment call the system cannot make for you —
and this run's whole point is making sure that judgment call is made with
real numbers in front of it, not an assumption.

Concrete near-term fits in this codebase: an agent-memory namespace-merge
or graph-condense operation where a replica needs the same partition a
peer already computed (an immediate use for `candidate_b`); a federated or
edge agent-memory deployment where an edge node can't afford to recompute
a global mincut locally but can afford a cheap signature check and spot
sample. Longer-term, and not built here: embedding a `CutCertificate` in
an RVF portable-memory export's provenance section, or wiring
`candidate_b` into a ruFlo cross-agent workflow step.

## What this doesn't claim

Not that `candidate_a` is secure in any general sense — the opposite,
demonstrated with a real number. Not that `candidate_b`'s spot-check
proves optimality — it's a sampled necessary condition, evidence, not
proof. Not that the Stoer-Wagner implementation here says anything about
`ruvector-mincut`'s own performance — it's an independent reference,
chosen to keep this run's actual question (import latency) separate from
a much larger integration question. Not that the synthetic graph models
any real workload's topology — it's a simple, reproducible, connected
sparse graph, nothing claimed beyond that.

## Where the ground truth came from

Five targeted searches for prior art turned up real, relevant, but not
exactly matching work: Chiesa and Tromer's Proof-Carrying Data (2010),
the general paradigm this design is a much smaller, non-recursive special
case of; ALITHEIA (CCS 2014), the closest prior system for verifiable
computation specifically over graph algorithms, though it predates this
run's search window; a 2026 survey on federated agentic AI that proposes
federated graph learning for shared agent memory without the
cryptographic verification angle this run adds; and Hager and Krylyuk's
local-optimality conditions for graph partitioning, the closest treatment
found of the per-vertex necessary condition `candidate_b`'s spot-check
uses, though without the sampling idea itself. No prior work combining
"signed cut certificate" with "measured signed-but-wrong detection via
spot-check sampling" was found in these searches — as far as they show,
that specific combination is this run's own contribution, not a
replication.

## Try it yourself

```bash
cargo test -p ruvector-partition-import
cargo run --release -p ruvector-partition-import --bin benchmark
```

Both are deterministic (fixed seeds throughout), so the numbers above
should reproduce exactly on the same hardware and toolchain, and should
be directionally identical on any other.
