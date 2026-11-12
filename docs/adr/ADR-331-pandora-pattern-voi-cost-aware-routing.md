# ADR-331: Value-of-Information Cost-Aware Routing (Pandora-Pattern)

- **Status**: Proposed
- **Date**: 2026-08-22
- **Deciders**: RuV Perpetual Intelligence Runtime (PIR) Program
- **Related**: `crates/ruvector-tiny-dancer-core` (existing FastGRNN routing + conformal uncertainty, no cost model today — checked); harness `capabilities.ts` `routeModel` (price-aware routing under a quality floor — adjacent prior art); see `docs/research/perpetual-intelligence-runtime/10-wave4-evidence-review.md` and `11-wave4-program-plan.md`
- **Tags**: pir, wave-4, pandora, voi, routing, cost, tiny-dancer

## Context

Wave-4 evidence review grades this paper **A on mechanism** —
[arXiv:2608.20316](https://arxiv.org/abs/2608.20316), "Pandora's AI Model
Routing Box: Efficient Allocation with Costly Value Estimation,"
submitted 2026-08-20 (Fisch, Trivedi, Huot, Cohen, Kaisers, Lapata,
Larson, Eisenstein — a largely DeepMind-affiliated author list). Note the
naming: **"Pandora's Router" is the centralized policy inside the paper,
not the paper's title.**

Confirmed mechanism, verbatim against the abstract: "value estimation has
a cost. Cheap estimators (e.g., embedding-based predictors) are fast but
noisy, while accurate estimators ... are expensive. We formalize this
tradeoff as an instance of Pandora's Box ... Under a Gaussian signal
model, the resulting policies have closed-form value-of-information
expressions." Headline empirical claim is qualitative only ("matches the
routing quality of exhaustive estimation, while querying the expensive
estimator far less often") — no figures in the abstract; none are
invented here.

**A stated negative result the brief omitted, carried as binding scope**:
the paper's decentralized variant, "Pandora's Bidder" (specialists
self-assess before accepting an offered price), "can increase the
strategic specialist's utility at the expense of others" when competing
estimates are noisy. **The decentralized variant is explicitly out of
scope for this program.**

**Corroborating related work** (different, unrelated author group):
[arXiv:2608.19802](https://arxiv.org/abs/2608.19802), "Stopping and
Routing LLM Judge Panels" (accepted at WISE 2026) independently reaches
the same economics for evaluator panels — call complementary judges,
route specialists to relevant cases, stop when incremental validation
gain falls below cost.

**Artifact availability — checked, not assumed: no code for either
paper.** From-scratch implementation; the closed-form structure
(Weitzman-index / Pandora's Box machinery) is standard and derivable from
the formalization.

**What exists in-repo today — checked at source**:
`crates/ruvector-tiny-dancer-core` is the real router ("Production-grade
AI agent routing system with FastGRNN neural inference ... for optimizing
LLM inference costs") with `router.rs`, `uncertainty.rs` (conformal
prediction), `circuit_breaker.rs`, and full observability — but a
case-insensitive grep finds **exactly one occurrence of "cost" in its
entire source: the doc-comment phrase**. No cost model, no cascade, no
escalation logic exists. `ruvector-router-core`, despite its name, is a
vector DB, not a request router. The nearest cost-aware machinery is the
harness's `routeModel` (price-aware under a quality bar) and the
flywheel's `costPerWin` — neither is a VoI computation.

**Why one primitive**: the same buy-information decision governs model
selection, retrieval depth, verifier calls, agent spawning, and
escalation. Landing it once in the router core gives every surface the
same audited rule instead of five ad-hoc heuristics.

## Decision

Add a value-of-information module (`voi.rs`) to
`crates/ruvector-tiny-dancer-core`:

1. **The decision rule.** Before invoking a costlier estimator (a larger
   model, a deeper retrieval, a verifier pass, an extra judge), compute
   whether purchasing the better estimate has positive expected value:
   buy iff
   `expected_quality_gain × value_of_success > estimator_cost + latency_cost`.
   Under the paper's Gaussian signal model this has closed form; the
   module implements the closed-form index and exposes it as a pure,
   deterministic function of (prior mean, prior variance, estimator noise,
   costs).
2. **Integration, not replacement.** The existing FastGRNN scorer and
   conformal `uncertainty.rs` supply exactly the prior mean/variance the
   VoI computation consumes; `RoutingDecision` gains VoI fields
   (index value, buy/skip decision, costs used) so every decision is
   auditable in the existing metrics/tracing surface. The current
   routing path's behavior is unchanged when VoI is disabled
   (config-gated, default off until research-gate evaluation).
3. **Centralized policy only.** Per the paper's own negative result, no
   decentralized self-assessment/bidding variant is implemented.
4. **One primitive, staged adoption.** This ADR lands the primitive and
   its use for model-selection routing in tiny-dancer. Adoption by other
   surfaces (retrieval depth, verifier calls, judge panels per
   arXiv:2608.19802, agent spawning, escalation) is follow-up, each
   consuming the same module rather than reimplementing the rule.
5. **Numeric hygiene.** All inputs pass an `is_finite()` gate before the
   index computation (Wave-3 #888 lesson); degenerate variances clamp to
   documented bounds; the function is total — it returns a decision or a
   typed error, never NaN.
6. **Escalate-only invariant (blocking, mirrors ADR-330's downgrade-only
   rule).** **The VoI gate may only convert a would-be-lightweight route
   into an escalation; it must never rescue a below-threshold or
   over-uncertain candidate into the cheap path.** The primitive's
   "decide on the current belief" arm (`VoiDecision::Route` — no purchase
   has positive expected value) is *not* a routing verdict and must never
   be read as "use the lightweight model": it falls back to the legacy
   rule, so `confidence_threshold` and `max_uncertainty` are consulted on
   every path, gated or not. This is the routing analog of ADR-330's
   downgrade-only confidence bound — the failure direction is
   conservative by construction, and the gate can only ever spend *more*
   than the legacy rule would, never less. Corollary for numeric hygiene:
   any non-finite value surviving the primitive's `max(0.0)` clamp must
   resolve toward **escalation**, never toward the lightweight path — a
   NaN must not be able to buy a cheap route.

   *Provenance*: the WP28 audit (PR #893, HIGH) found exactly this
   fail-open — the router wiring read the `Route` arm as "use the
   lightweight model," so a 0.10 score against a 0.85
   `confidence_threshold` routed to the cheap model and `max_uncertainty`
   was void whenever escalation happened to be priced above the VoI. The
   invariant is stated here so the contract survives independently of the
   wiring that violated it.

## Consequences

### Positive

- Replaces "always run the expensive path" / "never run it" heuristics
  with a principled, closed-form, auditable rule — the difficulty-3,
  value-10 item in ruv's briefing — and does so in the crate that
  already owns routing types, uncertainty, and observability.
- The conformal-prediction machinery already present is precisely the
  calibrated-uncertainty input VoI needs; the integration is additive.
- One shared primitive prevents five divergent cost heuristics from
  accreting across the stack.

### Negative

- The Gaussian signal model is an assumption; miscalibrated priors yield
  bad buy decisions. Mitigated by conformal calibration and by the
  config gate + research-gate evaluation before default-on.
- `value_of_success` and cost parameters are caller-supplied policy;
  garbage in, garbage out. The module documents units and provides
  conservative defaults but cannot validate a caller's economics. See
  the calibration hazard below — this is not a theoretical concern.

### Operational note — the `value_of_success` calibration hazard

**A misconfigured `value_of_success` degenerates the gate into a
permanent never-escalate switch that still looks like it is working.**
The closed-form VoI of a Gaussian signal is bounded by roughly `0.4σ`.
In this router σ is a **conformal uncertainty on a [0, 1] score**, so a
typical σ ≈ 0.05 caps VoI at ≈ 0.02 *in score units*. With
`value_of_success = 1.0`, the expected gain is therefore ≈ $0.02, and
**no escalation costing more than about two cents is ever purchasable at
any score** — the gate silently answers "don't buy" every time while
metrics, tracing, and decision records all look healthy.

Operators must express `value_of_success` as **the currency value of a
correct route** — the business value of getting this request right —
which is typically orders of magnitude above 1, not as a normalized
score weight. This "looks healthy while inert" failure mode is recorded
here, not only in rustdoc, because a reader of the config has no local
signal that the gate has stopped doing anything.
- No reference implementation or paper figures exist to calibrate
  against; validation is entirely this program's own benchmarks
  (routing-inference bench gains a VoI suite).

## Security / Validation Gates

- **Escalate-only invariant** (blocking, Decision §6) — the gate may
  only turn a lightweight route into an escalation; `confidence_threshold`
  and `max_uncertainty` are consulted on every path, and the "decide on
  current belief" arm falls back to the legacy rule rather than
  short-circuiting to the cheap model. Regression-tested with a
  below-threshold candidate and an over-uncertain candidate on the gated
  path.
- **Non-finite rejection at the choke point** (blocking) — inputs and
  the computed index; a non-finite anywhere is a typed error, never a
  routing decision, and any non-finite that survives a clamp resolves
  toward escalation, never toward the lightweight path.
- **Fail-conservative**: on any VoI-computation error the router falls
  back to the pre-VoI path (existing behavior), and the fallback is
  counted in metrics — an error can never silently become a "skip the
  verifier" decision.
- **Config-gated rollout**: default off; enabling requires the standard
  research-gate paired evaluation showing quality-per-cost improvement
  on this repo's own routing benchmarks.
- **Decentralized variant prohibited** (Decision §3) — revisit only with
  a new ADR if future work addresses the paper's negative result.
- **Standard repo gate**: `cargo nextest run -p ruvector-tiny-dancer-core`
  plus the existing routing benches.

## Affected Repos

- `ruvnet/ruvector` only — `crates/ruvector-tiny-dancer-core/src/`
  (new `voi.rs`; additive wiring in `router.rs`, `uncertainty.rs`,
  `metrics.rs`). Single-repo scope.

## Dependencies

None hard among Wave-4 ADRs. Consumes existing tiny-dancer uncertainty
machinery. The judge-panels adoption path (arXiv:2608.19802) is follow-up
under this same primitive.

## Alternatives Considered

- **Implement in the TS harness next to `routeModel`**: rejected —
  `routeModel` is bench-harness tooling; production routing, its types,
  and its observability live in tiny-dancer, and Rust-side landing serves
  every downstream consumer (node/wasm bindings included).
- **A learned buy/skip classifier instead of the closed form**: rejected
  for this wave — the closed form is auditable, parameter-light, and the
  paper's own contribution; a learned policy would be an untrusted
  proposer under Invariant 7 and needs its own gate later.
- **Implement Pandora's Bidder too**: rejected — the paper's own stated
  negative result under noisy competing estimates; out of scope.
- **Name the module "pandora"**: rejected — name-collision discipline
  ("Pandora" is maximally overloaded); the module is `voi`, cited to
  arXiv:2608.20316.
