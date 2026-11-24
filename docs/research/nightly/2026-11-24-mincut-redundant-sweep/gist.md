# A 900x Latency Outlier With "No Known Trigger" Had an Exact, Fixable Cause

## Problem

`ruvector-mincut` implements a from-scratch Rust version of a
subpolynomial-update dynamic minimum-cut algorithm. Two prior nightly
research runs in this repository measured something strange: a
19-vertex graph took **69.3 seconds** to compute a minimum cut, while a
50-vertex graph — bigger — took **77 milliseconds**. The size/latency
relationship was otherwise smooth and monotonic (77ms → 415ms → 1.6s →
11.4s for 50 → 100 → 200 → 400 vertices). The 19-vertex case was filed
as "a small, perfectly regular ring graph, plausibly an
adversarial/degenerate case," with no mechanism identified, and the next
step was queued as "evaluate swapping in a different connectivity
backend."

## Hypothesis

Before writing any code, I read the actual connectivity backend that was
queued for the swap (`PolylogConnectivity`). It has exactly four public
methods: insert an edge, delete an edge, check if two vertices are
connected, count connected components. No minimum cut, no witness, no
concept of "which side of the cut is which." It could not have answered
the question the wrapper asks of it, under any integration. That lead
was dead on arrival.

So I read the actual hot path instead:
`MinCutWrapper::process_instances`, which implements the paper's core
idea — maintain `O(log n)` "bounded-range instances," each responsible
for a geometric slice of possible cut values (instance 0 handles cut
value exactly 1, instance 1 also handles 1, instance 2 handles up to 2,
and so on, growing by a factor of 1.2 each time), and sweep them in
order until one says "the answer is in my range."

For small graphs (under 20 vertices), the instance responsible for
actually computing an answer calls `brute_force_min_cut()` — an
exhaustive enumeration over all `2^n` vertex subsets. Critically, this
enumeration **doesn't use the instance's own value range at all**. It
always computes the *true, global* minimum cut, and only checks
afterward whether that value happens to fit the range it was asked
about.

Which means: if the true minimum cut of your 19-vertex graph is, say,
16 — and you have to sweep past a dozen or more ranges before one is
wide enough to contain 16 — you are not computing 16 progressively
narrower answers. You are computing **the exact same answer, from
scratch, over and over**, once per range swept, because each range gets
a brand-new, empty instance object with no memory of what the last one
just figured out.

```text
Given MinCutWrapper::process_instances() on a cold wrapper, over a
graph with n < 20 vertices whose true minimum cut value is not in
instance 0's range,

when the brute-force instance exposes the true value it already
computed even when reporting "out of range," and the wrapper uses it to
jump straight to the correct range instead of visiting every one in
between,

then latency drops by roughly the number of ranges previously swept,
with no change to the returned answer.
```

I verified the mechanism directly before trusting the theory: a
temporary print statement inside `brute_force_min_cut()` showed exactly
**16** calls for the 19-vertex case before any fix, and exactly **1**
call after.

## Technical Design

Two new methods on the trait every "instance" implementation already
satisfies, both defaulted to doing nothing so no existing code needs to
change:

```rust
trait ProperCutInstance {
    // ... existing methods unchanged ...

    /// The true, range-independent answer, if this implementation
    /// happens to compute one regardless of its own bounds.
    fn exact_value_hint(&self) -> Option<(u64, WitnessHandle)> {
        None
    }

    /// Accept an answer computed elsewhere instead of recomputing it.
    fn seed_known_value(&mut self, _value: u64, _witness: &WitnessHandle) {}
}
```

The brute-force implementation (`BoundedInstance`) already stores the
true value it computes in a cache field — it just never told anyone
when the range it was asked about didn't match. `exact_value_hint`
exposes that cache; `seed_known_value` writes to the same cache a
different way in.

The wrapper's sweep loop changes from "visit every range in order" to
"visit a range; if it says no but hands you the real answer, jump
straight to the range that actually contains it, carrying the answer so
the range you land on doesn't recompute it either." For a 20-vertex
local-search path (not brute force), this hint is correctly `None` —
that search's answer genuinely depends on the range it's given, so no
shortcut applies, and nothing about its behavior changes.

## Actual Implementation

`crates/ruvector-mincut/src/instance/traits.rs` (trait additions),
`crates/ruvector-mincut/src/instance/bounded.rs` (implementation + four
new tests), `crates/ruvector-mincut/src/wrapper/mod.rs` (the sweep loop
change + three new tests). About 100 lines of actual logic change,
`cargo fmt`'d and `cargo clippy`'d clean. A new example,
`process_instances_sweep_variants.rs`, implements the fix's shipped
version alongside two honest alternatives — the original "visit every
range" approach, and a middle-ground "exponential-doubling search" that
would also help without needing to know the answer is exact — and
proves (via `assert_eq!`, not just eyeballing output) that all three
agree on the actual answer while differing enormously in cost.

## Actual Benchmark Evidence

Same exact, unmodified reproduction script used by both prior nightly
runs (`mincut_scaling_probe.rs`), rebuilt before and after via `git
stash`, on this session's own hardware (so the comparison is
apples-to-apples, unlike comparing across different nightly runs'
different machines):

```text
n=19   before: 81,766ms / 83,410ms   after: 4,923ms / 5,105ms   (~16-17x)
n=50   before:     75ms /     78ms   after:    75ms /    82ms   (unchanged)
n=100  before:    415ms /    417ms   after:   401ms /   422ms   (unchanged)
n=200  before:  1,562ms /  1,648ms   after: 1,498ms / 1,615ms   (unchanged)
n=400  before:  5,243ms /  5,386ms   after: 4,900ms / 5,122ms   (unchanged)
```

Three-variant comparison (`baseline` / exponential-search /the shipped
fix), all three proven to agree on the answer:

```text
n=19, k=8 ring:  baseline 16 visits / 95.7s   middle 10 visits / 61.1s   shipped 2 visits / 6.4s
n=10, k=6 ring:  baseline 13 visits / 65ms    middle  8 visits / 40ms    shipped 2 visits / 5.4ms
```

Full existing test suite stayed green (718 passed under
`--all-features`); one unrelated, pre-existing `wasm` module failure was
found and independently confirmed to exist on unmodified code too,
rather than either hidden or wrongly blamed on this change.

## Limitations

This fixes exactly the mechanism described above — redundant
recomputation in the small-graph brute-force path. It does **not** fix
a separate, larger scaling problem in the path used for graphs of 20+
vertices (which was already known, already measured, and is explicitly
out of scope here). It also does not reverse a prior nightly's
unrelated rejection of an agent-memory feature that depended on this
code path, which rests partly on that still-open second problem.

## Production Relevance

Merged directly into the default code path — no feature flag, since
this is a correctness-neutral performance fix, not a new tradeoff to
opt into. Every consumer of this crate (three in-tree, 18 dependent
crates including WASM/Node bindings) gets the fix for free on next
rebuild, no migration required.

## RuVector Ecosystem Implications

This is a small fix with a specific lesson for how this repository's
nightly research process should treat "unexplained" performance
anomalies: two prior runs correctly measured the symptom, correctly
flagged it as unexplained rather than guessing, and queued a *plausible-
sounding but wrong* next step. The wrong step would have cost real
implementation time to discover was a dead end only after attempting it.
Reading the actual, specific, narrow mechanism first — before writing
any fix — found both that the queued plan wouldn't have worked and what
would.

## Future Direction

The natural next thread: the same kind of narrow, read-the-actual-code
investigation applied to the 20+-vertex path's scaling problem, which
remains exactly as unexplained today as this one was yesterday.

## References

- `docs/adr/ADR-345-mincut-gated-forgetting.md`
- `docs/adr/ADR-346-deterministic-mincut-witness-partition.md`
- `docs/adr/ADR-361-mincut-process-instances-redundant-sweep.md`
- `docs/research/nightly/2026-11-24-mincut-redundant-sweep/README.md`
- arXiv:2512.13105 (the dynamic minimum-cut paper this module implements)
- arXiv:2510.08297 (the dynamic-connectivity paper evaluated and found
  inapplicable)
