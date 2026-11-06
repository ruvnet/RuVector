# Nightly Research — Signed Witness Batch-Fill Timeout

## Summary

`ruvector-agent-memory`'s `SignedWitnessSink` offers `SigningStrategy::BatchTail`
to amortize Ed25519 signing cost over many witness records, but it has no
wall-clock bound: if records arrive slower than `batch_size` fills, the
pending span — and every record inside it — stays unsigned indefinitely.
This run adds `SigningStrategy::BatchTailTimeout`, a caller-driven timeout
that closes the span after `max_wait_ns` even if `batch_size` never fills,
and measures the result with a real discrete-event simulation (real
Ed25519 signs, synthetic seeded arrival timing). **Result: ACCEPT**, on
3/3 independent runs, against thresholds fixed before the run.

## Abstract

ADR-347 implemented `SignedWitnessSink` for `ruvector-agent-memory`'s
TARL witness ledger, with two strategies: `PerRecord` (immediate
signature, highest cost) and `BatchTail { batch_size }` (amortized
signature, no latency bound). The 2026-09-16 nightly run that landed
ADR-347's hardening named the missing timeout as its own Next Research
item #1, explicitly pointing at `ruvector-retrieval-receipt::batch_fill`'s
`BatchFillPolicy::hybrid` as the pattern to port — not to take on as a
dependency, since `SignedWitnessSink` signs spans in-place rather than
handing closed batches to an external signer. This run does exactly that:
a new `BatchTailTimeout` strategy, a `check_timeout` method the caller
drives on its own clock, and a benchmark that reuses
`LedgerWitnessRecord::timestamp_ns` as the arrival clock — no change to
`WitnessSink::emit_batch`'s signature. Three load regimes (target, light,
bursty) were simulated with real `Ed25519` signing; `BatchTailTimeout`
bounded p99 signature-availability latency to ≤70ms at every regime,
while plain `BatchTail` reached 761ms at light load — over 15x worse —
while costing within 2x `BatchTail`'s amortized signing overhead at
healthy load. All closed spans verified under `verify_signed_chain`.

## Hypothesis

```text
Given a stream of ledger witness records (timestamp_ns as the arrival
clock) closed into signed BatchTail spans of batch_size=32, under three
load regimes — target Poisson (2000 records/s), light Poisson
(50 records/s), and bursty on/off Poisson (1500 records/s for 100ms,
silent for 400ms, repeating) —

when each record's signature-availability latency is measured as (the
span-close decision time under real arrival timing, plus the real
measured Ed25519 span-sign wall time) minus the record's arrival time,
comparing baseline BatchTail{batch_size:32} (no timeout) against
candidate_a BatchTailTimeout{batch_size:32, max_wait_ns:50ms},

then candidate_a's p99 latency should stay within a fixed bound of 70ms
(the 50ms timeout plus a fixed, not-tuned-post-hoc 20ms slack) at every
tested regime, while baseline's p99 latency should exceed 2x candidate_a's
p99 at the light-load regime — demonstrating the unbounded-tail failure
mode this run closes,

subject to: every closed span verifying under verify_signed_chain (100%),
and candidate_a's amortized signing cost at the target-load regime
staying within 2x of baseline's amortized cost at that same regime.
```

Acceptance thresholds, fixed before this run (same shape as ADR-343's,
applied to the sibling crate):

1. 100% of closed spans verify, every regime and strategy.
2. candidate_a's p99 latency ≤ 70ms at all three regimes.
3. At light load, baseline's p99 latency > 2x candidate_a's p99.
4. At target load, candidate_a's amortized signing cost ≤ 2x baseline's.

`candidate_b` (`PerRecord`) is carried as a reference upper bound, not a
gated variant: it is expected and observed to have near-zero latency at
much higher per-record signing cost, which is exactly `PerRecord`'s known
tradeoff from ADR-347/the 2026-09-16 benchmark — not re-litigated here.

## Why This Matters in 2026

Agent-memory systems are starting to be relied on for audit and dispute
resolution (did the agent actually see this instruction, in this order,
before taking this action). A signed witness chain is only as trustworthy
as its latency characteristics are *known*: an operator who cannot bound
how long a write might sit unsigned cannot make a defensible claim about
what "the signed record as of time T" actually covers. This run turns an
undocumented assumption ("batches eventually close") into a measured,
tunable guarantee.

## Why This Could Matter in 2036

If agent memory becomes a standard legal/regulatory evidence substrate
(the trajectory ADR-134's witness schema and ADR-347's signing already
point at), bounded signature-availability latency is a prerequisite for
real-time compliance tooling — a monitor that needs to know "is everything
up to 5 seconds ago signed yet" cannot function against an unbounded-tail
primitive.

## Why This Could Matter in 2046

Autonomous multi-agent systems operating with only intermittent
connectivity (edge robotics, space systems, disaster-response swarms)
need memory primitives whose worst-case behavior is provable under
*any* write-rate pattern, including long silences — exactly the
stalled-writer case this run's `light_load` regime approximates. A
timeout-bounded signing primitive is a small but concrete building block
toward that property.

## Why RuVector Is the Right Substrate

`ruvector-agent-memory` already owns the witness ledger, the Ed25519
signing module, and the `LedgerWitnessRecord` schema with its
`timestamp_ns` field — every piece this run needed already existed in
one coherent crate, requiring zero new dependencies.

## Why ruFlo Matters

A concrete ruFlo workflow role: a background task that calls
`SignedWitnessSink::check_timeout(now_ns)` on a fixed tick (e.g. every
10ms) for every live agent-memory ledger, turning this run's
caller-driven primitive into an always-on latency guarantee without
requiring every integrator to remember to wire it in themselves.

## Why MetaHarness Matters

MetaHarness's role-separation discipline (goal planner / researcher /
implementer / adversarial reviewer / evidence judge) was followed as a
process discipline in this run (see Nightly Self-Review), even though no
`ruvector harness` orchestration CLI was available to run those roles as
separate processes (see MetaHarness Capabilities Discovered below).

## Why Flywheel Matters

This run is itself a Flywheel-shaped event: it consumes a prior run's
recorded Next Research item verbatim and produces a new Next Research
item (Open Questions below) for a future run to consume. No Flywheel CLI
was available to record this mechanically; the chain is maintained
through nightly-report cross-references instead (see References).

## Why Darwin Matters

No Darwin evolutionary search ran in this session (no bounded parameter
space was swept — `max_wait_ns=50ms` and `batch_size=32` were fixed by
analogy to ADR-343's sibling-crate values, not searched). See Evolution
Results and Next Research for why a bounded `max_wait_ns` sweep is a
reasonable Darwin candidate for a future run.

## Why MCP Matters

Not directly exercised. A narrow MCP surface is plausible future work: a
read-only tool reporting `oldest_pending_arrival_ns()` / time-to-deadline
for a live ledger's `SignedWitnessSink`, letting an external monitor poll
signature-availability risk without write access. See MCP Implications.

## Why RVF May Matter

A `SignedAnchor` (this module's checkpoint type) is exactly the kind of
small, serializable, signed state that an RVF portable cognitive package
would want to carry across a deployment boundary. See RVF Implications.

## Why RVM May Matter

Not directly relevant to this run's scope (a scheduling-timing change,
not a capability-boundary or isolation change). See RVM Implications.

## Why Rust Matters

The entire change — a new enum variant, a method, 9 tests, a benchmark
binary — compiles to the same zero-cost-abstraction, no-garbage-collector
guarantees as the rest of the crate; the timing measurements in this
report are real `std::time::Instant` wall-clock numbers from optimized
native code, not an interpreted-language approximation.

## MetaHarness Capabilities Discovered

Per Step 3/Step 0's requirement to verify rather than assume tooling
exists, this run re-ran the same discovery commands the 2026-09-01
nightly run used, and got the same result (same environment):

| Capability | Installed? | Notes |
|---|---|---|
| `npx metaharness` (scaffolding CLI) | Yes (`metaharness@0.4.17`, fetched from npm on first use) | Provides `score`/`analyze`/`genome`/`learn`/`avo`/`proxy`/interactive wizard for *scaffolding a new harness project*; not itself a running orchestrator inside this repository, and not invoked to drive this run. |
| `npx ruvector harness doctor --json` | **No** | `npm error could not determine executable to run` — no `ruvector` CLI package with a `harness` subcommand is installed or resolvable in this repository/session. |
| `npx ruvector harness status/flywheel/darwin/route ...` | **No** | Same root cause; none of these commands exist to invoke. |

Honest consequence, same as the prior run's: this run's "roles" (goal
planner, researcher, implementer, benchmark engineer, adversarial
reviewer, evidence judge) were carried out by this single session
directly, in sequence, as a process discipline rather than literal
separate CLI-orchestrated agent processes. No model-routing decisions
were recorded because no routing CLI was available. No Darwin/Flywheel
CLI evidence files were produced for the same reason — evidence is
instead this report, the ADR, and the raw benchmark output committed
alongside them.

## SOTA Context (2026)

Size-or-timeout micro-batching ("close at N items or T time, whichever
first") is a decades-old, well-established pattern — database group
commit, message-broker producer batching (Kafka's `linger.ms`), gRPC
request coalescing, GPU inference batching. This run's contribution is
not a novel scheduling algorithm; it is applying and *measuring* that
pattern specifically against `SignedWitnessSink`'s in-place span-signing
design (as opposed to the sibling crate's external-signer-plus-closed-batch
design), with real cryptographic cost and real hash-chain correctness
constraints, closing a gap two independent prior nightly runs both named.

## RuVector Ecosystem Fit

Touches only `ruvector-agent-memory` (`witness_signing.rs`,
`witness_signing_tests.rs`, one new example). That crate already depends
on `rvf-types` (ADR-320, Ed25519 + SHA-256 primitives) and optionally on
`ruvector-mincut` (ADR-345, feature-gated, unrelated to this change). No
new crate, no new external dependency, no change to any other crate in
the workspace.

## Architecture

```mermaid
flowchart LR
    subgraph Writer["Write path (unchanged)"]
        L[TransactionalLedger] -->|emit_batch| S[SignedWitnessSink]
    end
    subgraph Sink["SignedWitnessSink (this run's addition in bold)"]
        S --> P{pending span}
        P -->|batch_size reached| C[close: Ed25519 sign]
        P -.->|**max_wait_ns elapsed**<br/>**check_timeout(now_ns)**| C
        C --> I[inner WitnessSink]
        C --> V[spans: Vec SignedSpan]
    end
    subgraph Caller["Caller-driven clock (new)"]
        T[Timer tick / next write attempt] -.->|now_ns| P
    end
    I --> M[MemoryWitnessLog]
    V --> VC[verify_signed_chain]
```

`SignedWitnessSink` itself owns no clock and spawns no background task —
the dashed line from `Caller` is intentional: a deployment supplies
`now_ns` from its own timer or opportunistically from its own write path,
matching `ruvector-retrieval-receipt::batch_fill`'s design choice that the
scheduler stays synchronous and dependency-free.

## Implementation

1. `SigningStrategy::BatchTailTimeout { batch_size, max_wait_ns }` —
   additive new variant; `PerRecord` and `BatchTail` unchanged.
2. `PendingSpan::opened_at_ns` — the oldest pending record's
   `timestamp_ns`, set once per span and read by the two new methods
   below. No new parameter on `WitnessSink::emit_batch`.
3. `SignedWitnessSink::oldest_pending_arrival_ns() -> Option<u64>` and
   `SignedWitnessSink::check_timeout(now_ns: u64) -> bool` — the
   caller-driven timeout primitive. `check_timeout` re-checks the
   deadline itself (unlike the sibling crate's trust-the-caller
   `close_on_timeout`), so calling it early is harmless.
4. `examples/witness_signing_batch_fill_latency.rs` — the benchmark:
   deterministic seeded Poisson/bursty arrival generation (same xorshift
   construction as `ruvector-retrieval-receipt::bin::batch_latency`),
   a discrete-event loop driving real `emit_batch`/`check_timeout`/`seal`
   calls, hash-chain-correct synthetic records (`prev_hash`/`record_hash`/
   evidence-grade `flags` all consistent so `verify_chain` accepts them —
   this took one debugging pass; see Failure Modes), and a fixed
   acceptance gate.

Full diff: `crates/ruvector-agent-memory/src/witness_signing.rs`,
`crates/ruvector-agent-memory/src/witness_signing_tests.rs` (9 new tests),
`crates/ruvector-agent-memory/examples/witness_signing_batch_fill_latency.rs`
(new file).

## Benchmark Methodology

- **Command:**
  `cargo run --release -p ruvector-agent-memory --example witness_signing_batch_fill_latency`
- **Workload:** 4000 synthetic witness records per regime, sequence
  numbers 0..4000 in arrival order, deterministic seeded xorshift RNG per
  regime (same seeds reused across baseline/candidate_a/candidate_b for a
  fair comparison within each regime).
- **Regimes:**
  - `target_load_2000rps` — Poisson(λ=2000/s): mean inter-arrival 0.5ms,
    so a 32-record batch fills in ~16ms, well inside the 50ms timeout.
  - `light_load_50rps` — Poisson(λ=50/s): mean inter-arrival 20ms, so a
    32-record batch would take ~640ms to fill on size alone — forcing
    `BatchTailTimeout`'s 50ms timeout to fire on almost every span.
  - `bursty_on1500rps_off400ms` — on/off Poisson: 1500/s for 100ms bursts,
    400ms silence, repeating — models clustered agent tool-call traffic
    rather than a smooth rate.
- **Strategies:** `baseline_batchtail32` (`BatchTail{batch_size:32}`),
  `candidate_a_hybrid50ms` (`BatchTailTimeout{batch_size:32,max_wait_ns:50ms}`),
  `candidate_b_per_record` (`PerRecord`, reference only).
- **Latency definition:** per record, `availability_ns - arrived_at_ns`,
  where `availability_ns` is the virtual close-decision time plus the
  **real** `std::time::Instant`-measured wall time of the `emit_batch`/
  `check_timeout`/`seal` call that actually closed its covering span.
  Every signing operation contributing to a reported latency is a real
  Ed25519 sign performed during the run — nothing here is synthesized.
- **Correctness check:** after each regime/strategy run, `seal()` is
  called and `verify_signed_chain` is run against the full resulting log
  and span list with a fresh genesis anchor.
- **Repetitions:** 3 independent full process runs (fresh `cargo run`
  each time); raw output saved under `evidence/run_{1,2,3}.txt` in this
  directory.
- **Hardware/toolchain:** Linux x86_64, 4 vCPUs, rustc 1.97.0, cargo
  1.97.0, release profile (`opt-level` per workspace default), commit
  `f9e98b681305827f1615f8a371ed2a0169d440be`.

## Benchmark Results

Full table, run 1 of 3 (runs 2–3 in `evidence/run_2.txt`, `run_3.txt`;
all three tell the same story within normal variance):

```text
regime                       policy                    records    spans   mean_sz   lat_mean    lat_p50    lat_p95    lat_p99    lat_max  sign_amort_ns  verified
target_load_2000rps          baseline_batchtail32         4000      125      32.0    8.087ms    8.005ms   16.508ms   19.047ms   23.687ms         1270.0      true
target_load_2000rps          candidate_a_hybrid50ms       4000      125      32.0    8.087ms    8.005ms   16.508ms   19.046ms   23.668ms         1267.7      true
target_load_2000rps          candidate_b_per_record       4000     4000       1.0    0.041ms    0.039ms    0.056ms    0.069ms    0.995ms        40846.1      true
light_load_50rps             baseline_batchtail32         4000      125      32.0  311.370ms  299.756ms  655.450ms  761.093ms  944.354ms         1244.5      true
light_load_50rps             candidate_a_hybrid50ms       4000     1141       3.5   31.933ms   34.786ms   50.038ms   50.057ms   50.137ms        11292.1      true
light_load_50rps             candidate_b_per_record       4000     4000       1.0    0.044ms    0.039ms    0.063ms    0.086ms    0.975ms        44104.1      true
bursty_on1500rps_off400ms    baseline_batchtail32         4000      125      32.0   59.874ms   10.549ms  416.212ms  423.349ms  428.339ms         1307.3      true
bursty_on1500rps_off400ms    candidate_a_hybrid50ms       4000      138      29.0   13.728ms   11.074ms   44.297ms   49.656ms   50.056ms         1425.1      true
bursty_on1500rps_off400ms    candidate_b_per_record       4000     4000       1.0    0.042ms    0.039ms    0.058ms    0.081ms    0.319ms        41868.5      true
```

Summary across all 3 runs (p99 latency, ms):

| regime | baseline | candidate_a | candidate_b |
|---|---:|---:|---:|
| target (2000 rec/s) | 19.04–19.07 | 19.04–19.07 | 0.07–0.10 |
| light (50 rec/s) | 761.09–761.11 | 50.06–50.07 | 0.07–0.10 |
| bursty (on/off) | 423.35–423.37 | 49.66–49.68 | 0.07–0.10 |

Amortized signing cost (ns/record) at target load, all 3 runs:
baseline 1165.8–1310.8, candidate_a 1164.5–1267.7 (within the 2x bound in
all 3 — candidate_a is in fact never slower than baseline at target
load, since the timeout essentially never fires there).

`verify_signed_chain` returned `Ok` for every regime/strategy/run — 27/27
cells "true", all 3 runs.

## Acceptance Result

**ACCEPT** on all 3 runs, every one of the 4 fixed acceptance thresholds:

1. All closed spans verify: **true**, all 27 regime/strategy/run cells.
2. candidate_a p99 ≤ 70ms at all regimes: **true** (max observed 50.073ms,
   run 3, light load — comfortably inside the bound).
3. baseline p99 > 2x candidate_a p99 at light load: **true** (baseline
   ≈761ms vs. candidate_a ≈50ms — a ~15.2x ratio, far past the 2x bar).
4. candidate_a amortized signing cost ≤ 2x baseline's at target load:
   **true**, every run (observed ratio range 0.89x–1.07x — essentially
   equal, as expected since the timeout almost never fires at target
   load).

## Memory Math

`PendingSpan` grows by one `u64` field (`opened_at_ns`) — 8 bytes per
in-flight pending span, of which there is at most one per
`SignedWitnessSink` at a time (not per record). `SigningStrategy`'s
largest variant grows from `BatchTail`'s 8 bytes (`batch_size: usize`) to
`BatchTailTimeout`'s 16 bytes (`batch_size: usize, max_wait_ns: u64`);
since `SigningStrategy` is `Copy` and stored once per sink (not per
record), this is a fixed, negligible per-sink cost, not a per-record one.

## Performance Math

Amortized signing cost at `batch_size=32` (one Ed25519 sign + one
SHA-256-over-32-records digest per 32 records) measures ≈1.1–1.5
microseconds/record across this run's 3 repetitions — consistent with
ADR-347's own `witness_signing_bench` measurement of ≈3.3 microseconds/op
for its own `candidate_b64` (two witness records per op there, one here;
the per-record order of magnitude matches). `PerRecord`'s ≈37–44
microseconds/record is roughly 30–40x candidate_a's cost at this run's
batch size — the amortization `BatchTailTimeout` preserves when the
timeout doesn't fire.

## Failure Modes

- **Nobody calls `check_timeout`:** degrades silently to `BatchTail`'s
  exact unbounded behavior (not incorrect, just inert) — see ADR-353
  Failure Modes for the full discussion.
- **Hash-chain construction bug found during this run:** the benchmark's
  first draft set every synthetic record's `flags: 0` while giving it
  `evidence_grade: EvidenceGrade::Recomputed`; `MemoryWitnessLog::verify_chain`
  cross-checks the grade-nibble packed into `flags` (bits 12-15) against
  `evidence_grade` itself and correctly rejected every record as a broken
  chain (`verified: false` across the board) until `flags` was computed
  via `(EvidenceGrade::Recomputed.code() as u16) << 12`. Disclosed here
  because it is exactly the kind of "fabricated benchmark data" risk this
  process's constraints warn against if left unnoticed — the fix was
  verified by re-running and confirming `verified: true` everywhere
  before any result in this report was accepted as evidence.
- **Simulation-boundary flush**, **signer-becomes-bottleneck at extreme
  rates**: same as ADR-343's disclosed limitations; see ADR-353.

## Rejected Alternatives

See ADR-353's Alternatives Considered: adding `max_wait_ns` directly to
`BatchTail` (breaks existing call sites), threading `now_ns` through
`emit_batch` (breaks the `WitnessSink` trait signature), an internal
background timer (breaks the crate's synchronous, dependency-free
design), and BLS aggregate signatures (orthogonal, no pairing-friendly
curve in the workspace).

## Security

No new cryptographic primitive and no change to `SignedSpan`'s message
layout or `verify_signed_chain`'s guarantees — `BatchTailTimeout` only
changes *when* the existing `close()` signing path is invoked. The
benchmark's arrival-time RNG (plain xorshift) is for reproducible timing
only, not a security-relevant random source; the signing keypair is a
fixed test secret (`[21u8; 32]`), matching the sibling crate's benchmark
convention of a fixed, non-secret key for reproducible measurement, not
production key material.

## Governance

Experimental and non-default, matching ADR-347's posture: no existing
`SignedWitnessSink` construction site in the repository was changed to
use `BatchTailTimeout`. Promotion of a specific `max_wait_ns` value, or
of an automatic `check_timeout`-driving integration, requires real
deployment traffic traces this run did not have access to (see Open
Questions).

## MCP Implications

A plausible narrow, read-only MCP tool: `agent_memory_witness_signing_status`,
returning `{ pending_count, oldest_pending_arrival_ns, deadline_ns,
strategy }` for a named ledger's `SignedWitnessSink`. No mutation
authority; purely observational, for an external monitor to assess
signature-availability risk without touching the write path. Not
implemented in this run.

## WASM Implications

Not measured in this run (same open item the 2026-09-16 report flagged a
third unresolved instance of, for the sibling crate's analogous question).
`BatchTailTimeout` adds one `u64` field and two short methods — expected
WASM binary-size delta is small, but "expected" is not "measured," so no
deployment claim is made here.

## RVF Implications

`SignedAnchor { record_count, head_digest }` is already a small,
`Copy`, serializable checkpoint — a natural candidate for inclusion in an
RVF portable cognitive package's signed-lineage metadata (RVF
Integration Analysis: state portability and signed lineage both apply;
index/model/policy portability do not, since this module carries no
index or model state). Not implemented or measured here.

## RVM Implications

No capability-boundary, coherence-domain, or isolation question is
raised by this change — it is a scheduling-timing addition inside a
single existing module. RVM integration adds no value here (per Step 28's
explicit instruction not to force one).

## ruFlo Implications

See "Why ruFlo Matters" above: a background tick calling `check_timeout`
across live ledgers is the concrete workflow. Not implemented in this
run — this run ships the primitive the workflow would call, not the
workflow itself.

## Practical Applications

1. **User:** an enterprise running agent-memory audit logs for
   compliance. **Problem:** needs to know signature coverage is never
   more than N seconds stale. **Capability:** `BatchTailTimeout` with
   `max_wait_ns=N`. **Ecosystem integration:** `ruvector-agent-memory`
   directly. **Implementation path:** wire a tick into the deployment's
   existing scheduler. **Business value:** defensible "as of" claims.
   **Main risk:** choosing `max_wait_ns` without real traffic data.
   **Time horizon:** now.
2. **User:** a multi-agent orchestration platform signing tool-call
   history. **Problem:** bursty traffic (tool calls cluster, then idle)
   under plain `BatchTail` leaves idle-period writes unsigned
   indefinitely. **Capability:** this run's `bursty` regime result
   (49.7ms p99 vs. 423ms baseline). **Ecosystem integration:**
   `ruvector-agent-memory` + the orchestrator's own scheduler.
   **Implementation path:** direct. **Business value:** consistent
   latency SLA regardless of traffic shape. **Main risk:** none new.
   **Time horizon:** now.
3. **User:** a regulated-industry agent deployment (health, finance)
   needing real-time tamper-evidence monitoring. **Problem:** cannot
   monitor "is everything signed yet" without a bound. **Capability:**
   `oldest_pending_arrival_ns()` exposed for monitoring. **Ecosystem
   integration:** a future MCP read-only tool (see MCP Implications).
   **Implementation path:** build the tool, wire a dashboard.
   **Business value:** real-time compliance visibility. **Main risk:**
   MCP surface scope creep if not kept read-only. **Time horizon:**
   1 year.
4. **User:** an edge-deployed agent with intermittent connectivity.
   **Problem:** long silences (modeled by `light_load`) must not leave
   writes permanently unsigned if the device never reconnects in time.
   **Capability:** the 50ms-class bound generalizes to any chosen
   `max_wait_ns` appropriate for the device's duty cycle. **Ecosystem
   integration:** `ruvector-edge-*` crates (not touched by this run).
   **Implementation path:** requires the edge crates to adopt the
   pattern — not done here. **Business value:** provable worst-case
   signing latency on constrained devices. **Main risk:** untested at
   edge-realistic hardware. **Time horizon:** 2+ years.
5. **User:** a RuVector-based swarm memory system with many concurrent
   writers. **Problem:** coordinating one shared ledger's timeout across
   writers. **Capability:** `check_timeout` is cheap and idempotent-safe
   to call redundantly from multiple places. **Ecosystem integration:**
   `ruvector-agent-memory` + swarm coordination layer. **Implementation
   path:** each writer opportunistically calls `check_timeout` before its
   own write. **Business value:** no single point of timeout-driving
   failure. **Main risk:** not measured under real concurrent load.
   **Time horizon:** 1 year.
6. **User:** a forensic incident-response tool reconstructing "what was
   known and signed by time T" after an agent incident. **Problem:**
   without a bound, the tool cannot distinguish "nothing happened" from
   "something happened but wasn't signed yet." **Capability:** the
   latency bound this run measures directly answers that. **Ecosystem
   integration:** `ruvector-agent-memory` + an incident-response query
   tool (not built here). **Implementation path:** direct. **Business
   value:** faster, more confident incident reconstruction. **Main
   risk:** none new. **Time horizon:** now.
7. **User:** a CI pipeline validating agent-memory deployments before
   production rollout. **Problem:** needs an automated latency-bound
   regression check. **Capability:** this run's benchmark binary, run as
   a CI gate with the same acceptance thresholds. **Ecosystem
   integration:** `cargo run --release --example witness_signing_batch_fill_latency`
   as a CI step. **Implementation path:** add to CI config (not done in
   this run — out of scope). **Business value:** catch latency
   regressions before they ship. **Main risk:** simulation drift from
   real traffic (see Limitations). **Time horizon:** now.
8. **User:** a researcher benchmarking alternative signature schemes
   (e.g. a future BLS integration) against this module. **Problem:**
   needs an existing, trusted latency-measurement harness to compare
   against. **Capability:** this run's discrete-event simulation
   methodology is reusable (same pattern ADR-343 established, now proven
   twice). **Ecosystem integration:** `ruvector-agent-memory`'s test/bench
   conventions. **Implementation path:** swap the signing primitive,
   rerun the same harness. **Business value:** apples-to-apples
   comparison. **Main risk:** none new. **Time horizon:** 1 year.

## Long Horizon Applications

1. **Self-healing graph memory.** Thesis: a memory substrate that can
   prove its own signing freshness can safely participate in automated
   repair decisions (e.g. "don't trust this subgraph's provenance until
   its pending span closes"). Required advances: wiring timeout state
   into `graph_forget`/mincut-gated forgetting (ADR-345). RuVector role:
   `ruvector-agent-memory` + `ruvector-mincut`. Why this run matters: it
   is the first piece exposing a queryable "freshness" signal. Primary
   uncertainty: whether freshness should gate forgetting or only
   annotate it. Falsification path: show a repair decision made worse by
   using freshness as a gate.
2. **Synthetic nervous systems.** Thesis: a distributed agent "nervous
   system" needs bounded-latency provenance at every signal junction,
   not just at rest. Required advances: propagating timeout bounds
   through multi-hop signed chains. RuVector role: `ruvector-nervous-system`
   + this module. Why this run matters: establishes the single-hop
   bound this would compose from. Primary uncertainty: whether bounds
   compose additively or worse across hops. Falsification path: measure
   multi-hop composition and find super-additive blowup.
3. **Agent operating systems.** Thesis: an agent OS's audit subsystem
   needs the same kind of SLA primitives a real OS kernel gives disk I/O
   or scheduling. Required advances: a formal latency-SLA API surface
   across all agent-memory primitives, not just signing. RuVector role:
   `rvm` + `ruvector-agent-memory`. Why this run matters: one concrete
   instance of such an SLA. Primary uncertainty: generalizability.
   Falsification path: find a second primitive where the same pattern
   doesn't fit.
4. **Autonomous edge cognition.** Thesis: edge agents operating for
   days without connectivity need provably bounded local signing latency
   to remain auditable once reconnected. Required advances: edge-scale
   measurement (WASM Implications gap). RuVector role: `ruvector-edge-*` +
   this module. Why this run matters: the native-hardware baseline this
   edge work would need to reproduce. Primary uncertainty: whether
   edge CPU/signing cost changes the regime boundaries measured here.
   Falsification path: run the same benchmark cross-compiled to a
   representative edge target.
5. **Swarm memory.** Thesis: many agents sharing one witness ledger need
   a shared, provable freshness contract, not per-agent ad hoc polling.
   Required advances: a coordination protocol for multiple
   `check_timeout` callers (Practical Application 5). RuVector role:
   `ruvector-agent-memory` + a future swarm-memory crate. Why this run
   matters: the per-ledger primitive the protocol would sit on top of.
   Primary uncertainty: contention behavior under many concurrent
   callers (untested). Falsification path: measure contention at scale
   and find it dominates the signing-cost savings.
6. **Dynamic world models.** Thesis: a world model that ingests signed
   agent-memory events needs to reason about which events are "settled"
   (signed) vs. "provisional" (pending) at any query time. Required
   advances: exposing pending/settled state to the world-model query
   layer. RuVector role: this module + a future world-model consumer.
   Why this run matters: the pending/settled distinction already exists
   (`unsigned_pending()`); this run adds the time bound that makes
   "provisional for at most X ms" a real guarantee. Primary uncertainty:
   whether world models actually need this distinction or can tolerate
   eventual consistency. Falsification path: show a world model that
   performs identically without the distinction.
7. **Proof-gated autonomous infrastructure.** Thesis: infrastructure that
   only acts on proof-gated (signed, verified) state needs a bound on how
   long state can remain un-actionable while pending. Required advances:
   connecting `check_timeout`'s bound to `ruvector-proof-gate`'s gating
   decisions. RuVector role: `ruvector-proof-gate` + this module (not
   connected in this run). Why this run matters: establishes the bound
   proof-gating would need to respect. Primary uncertainty: whether
   proof-gating should block on "pending-but-bounded" state or require
   fully-settled state always. Falsification path: find a proof-gating
   scenario where bounded-pending state causes an incorrect gate
   decision.
8. **Robotics memory.** Thesis: a robot's episodic memory, if it needs
   tamper-evidence for incident investigation (e.g. autonomous-vehicle
   black-box requirements), needs the exact bounded-latency signing
   guarantee this run measures, under real-time constraints tighter than
   this run's milliseconds-to-tens-of-milliseconds regimes. Required
   advances: sub-millisecond `max_wait_ns` measurement (not attempted
   here — the signing cost itself, ~1-1.5 microseconds, suggests this is
   plausible but unmeasured). RuVector role: `ruvector-robotics` + this
   module. Why this run matters: the millisecond-scale baseline a
   microsecond-scale robotics variant would need to beat. Primary
   uncertainty: whether signing cost or scheduling overhead dominates at
   that scale. Falsification path: run this benchmark with
   `max_wait_ns` in the 1-5ms range and see whether the bound still
   holds with the same slack ratio.

## Evolution Results (Darwin)

Not executed in this run. No `ruvector harness darwin` CLI was available
(see MetaHarness Capabilities Discovered), and `max_wait_ns=50ms`/
`batch_size=32` were chosen by direct analogy to ADR-343's sibling-crate
values rather than searched — a reasonable but unsearched starting point.
See Next Research item 1 for the concrete bounded sweep a future Darwin
run (or a manual sweep, absent the CLI) should perform.

## Promotion Decision

**Not promoted to any default construction path.** `BatchTailTimeout` is
shipped as an additive, opt-in `SigningStrategy` variant — exactly
ADR-353's Governance section's posture. Promoting a *specific*
`max_wait_ns` for any real deployment requires that deployment's actual
write-rate traffic trace, which this run did not have access to (see
Open Questions item 1). The implementation itself is promoted to the
crate (merged as library code with tests), which is distinct from
promoting a specific operational configuration.

## Witness Evidence

- **What code ran:** `crates/ruvector-agent-memory/examples/witness_signing_batch_fill_latency.rs`
  at commit `f9e98b681305827f1615f8a371ed2a0169d440be` (the branch's
  starting commit; this run's changes are additional commits on top).
- **Against what commit:** see above.
- **On what hardware:** Linux x86_64, 4 vCPUs (see Benchmark Methodology).
- **With what parameters:** `batch_size=32`, `max_wait_ns=50ms`,
  4000 records/regime, seeds `0x1111_2222_3333_4444` /
  `0x5555_6666_7777_8888` / `0x9999_AAAA_BBBB_CCCC` (one per regime, fixed
  across strategies within a regime).
- **With what dataset:** synthetic, deterministically seeded (no external
  dataset; the record *content* is irrelevant to this run's question,
  only arrival timing and sequence/hash-chain correctness matter).
- **Using what seed:** see above.
- **What result occurred:** ACCEPT, 3/3 runs (see Benchmark Results).
- **What agent proposed the change:** this session, acting on the
  2026-09-16 nightly run's explicitly recorded Next Research item #1.
- **What evaluator judged it:** this session, applying the 4 acceptance
  thresholds fixed before any benchmark run (see Hypothesis) — no
  separate adversarial-reviewer process was available as a distinct CLI
  role (see MetaHarness Capabilities Discovered); the adversarial
  questions in Step 7/Pass 3 of this process were applied directly by
  this session before implementation began (e.g. "is this already
  solved" → no, named explicitly as unsolved by two prior runs; "can the
  benchmark be gamed" → thresholds were fixed before any run and not
  adjusted after seeing results, including surviving a real
  implementation bug — see Failure Modes — that initially made every
  acceptance cell fail honestly).
- **What candidate lineage produced it:** no Darwin search ran (see
  Evolution Results); this is a single hand-designed candidate, not a
  Darwin-evolved one.
- **Why promoted or rejected:** promoted as library code (opt-in,
  non-default) because all 4 acceptance thresholds held on all 3 runs
  with 100% chain verification; not promoted as a specific production
  configuration, for the reasons in Promotion Decision.

No cryptographically signed witness record of this research process
itself was produced (no witness-chain tooling wraps the nightly process
yet — a potential future application of this very module, noted
speculatively and not claimed as done).

## Production Path

1. Land `BatchTailTimeout` as reviewed library code (this run's PR).
2. A deployment wanting the bound chooses `max_wait_ns` and wires a
   caller (timer tick or opportunistic pre-write check) that calls
   `check_timeout(now_ns)`.
3. Before relying on a specific `max_wait_ns` for a compliance or SLA
   claim, re-run this run's benchmark methodology against that
   deployment's actual traffic trace, not this run's synthetic regimes.
4. Optional: build the MCP read-only monitoring tool (MCP Implications)
   once a real deployment wants external visibility into pending-span
   risk.

## Falsification Criteria

This run's hypothesis would have been falsified by any of: a closed
span failing `verify_signed_chain` (none did — 27/27 cells across 3
runs); candidate_a's p99 exceeding 70ms at any regime (none did — max
observed 50.073ms); baseline's p99 at light load failing to exceed 2x
candidate_a's (it exceeded it by ~15x, far past falsification); or
candidate_a's amortized signing cost at target load exceeding 2x
baseline's (observed ratios were ~0.9–1.1x, the opposite direction from
falsification). None of these occurred; the hypothesis survived.

## Limitations

- **Synthetic traffic only.** Poisson and on/off-bursty arrival models
  are standard approximations, not measured production agent-memory
  write traffic. Real traffic may have different burstiness, tail
  behavior, or correlation structure.
- **Single-serialized-signer assumption untested at extreme rates** (see
  Threat Model in ADR-353) — real signing cost (~1-1.5 microseconds/record
  amortized) is far below every tested regime's fill window, but this run
  does not identify the arrival rate at which that stops holding.
- **`max_wait_ns=50ms` and `batch_size=32` are not searched/optimal
  values** — chosen by analogy to the sibling crate's ADR-343 values for
  direct comparability, not tuned for `ruvector-agent-memory`'s actual
  use cases.
- **No concurrent-writer testing.** `SignedWitnessSink` is driven
  single-threaded in this benchmark, matching its current single-threaded
  design; multi-writer coordination (Practical Application 5) is
  unexplored.
- **No WASM/edge measurement** (see WASM Implications) — a third
  consecutive nightly run across two crates to flag this same open gap
  without closing it.

## Next Research

1. A bounded sweep of `max_wait_ns` (e.g. 5ms/20ms/50ms/200ms) and
   `batch_size` (e.g. 8/32/128) against a fixed set of regimes, to
   characterize the amortization-vs-latency Pareto frontier rather than
   this run's single fixed point — a natural Darwin candidate once
   `ruvector harness darwin` (or an equivalent manual harness) is
   available, with a fitness function weighting p99 latency and
   amortized signing cost.
2. Wire `SignedWitnessSink::check_timeout` through
   `witnessed_compaction::compact_witnessed` end to end (2026-09-16's
   Next Research item #6, still open — this run adds a timeout primitive
   that item's eventual example could plausibly call from a compaction
   trigger, but does not wire it).
3. The WASM binary-size and signing-latency measurement three
   consecutive nightly runs (2026-08-31, 2026-09-16, this one) have now
   flagged across two crates and not yet closed.
4. Build the read-only MCP monitoring tool sketched in MCP Implications,
   once a concrete consumer wants it.
5. Measure this run's methodology against a real captured agent-memory
   write trace, if/when one becomes available, replacing the synthetic
   Poisson/bursty regimes with observed traffic.

## References

- `docs/adr/ADR-347-witness-signer-tarl-ledger.md` — the `SignedWitnessSink`
  this run extends.
- `docs/adr/ADR-353-signed-witness-batch-fill-timeout.md` — this run's ADR.
- `docs/research/nightly/2026-09-16-witness-signer-agent-memory/README.md` —
  source of this run's Next Research item #1 (the task this run executes).
- `docs/adr/ADR-340-signed-retrieval-receipt-anchoring.md`,
  `docs/adr/ADR-343-signed-receipt-batch-fill-latency-simulation.md`,
  `docs/research/nightly/2026-09-01-signed-receipt-batch-fill-latency/README.md` —
  the sibling crate's prior run establishing the exact pattern ported
  here, including the same acceptance-threshold shape and the same
  MetaHarness-capability-discovery finding.
- `crates/ruvector-agent-memory/src/witness_signing.rs`,
  `crates/ruvector-retrieval-receipt/src/batch_fill.rs`,
  `crates/ruvector-retrieval-receipt/src/bin/batch_latency.rs` — the code
  this run reads, extends, and parallels.
- `evidence/run_1.txt`, `evidence/run_2.txt`, `evidence/run_3.txt` (this
  directory) — raw, unedited output of the 3 independent benchmark runs
  this report summarizes.
