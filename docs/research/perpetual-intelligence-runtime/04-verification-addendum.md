# Verification Addendum — Direct Source Checks Against LatentMesh and Autogenous

Status: addendum to `02-asset-map.md` and `03-program-plan.md`, compiled
2026-08-19 by a third research pass (asset-scout) that cloned both external
repos directly rather than relying on `gh` CLI metadata or org-search
results. **This document does not edit `02-asset-map.md` or
`03-program-plan.md` in place** — it records what those two documents still
get wrong or omit as of this program's ADR-authoring pass, so the correction
is traceable rather than silently folded in. Where a claim below contradicts
`02-asset-map.md`, the section header says so explicitly.

Repos cloned: `ruvnet/LatentMesh` (HEAD `922ea196`, dated 2026-08-18) and
`ruvnet/autogenous` (HEAD `b5c6e838`, dated 2026-08-19).

## 1. LatentMesh — accurate but understated in scale

`02-asset-map.md` §8 correctly describes LatentMesh's crates, ADR sequence,
and "not implemented" markers, and its headline finding's loop diagram and
"statistical primitive and admission gate implemented; closed loop not
wired" quote both match source verbatim, including the per-stage markers on
ADR-004 through ADR-008 (004 not wired to live MidStream; 005 not wired to a
live RuVector instance; 006 no live Darwin loop; 007 no live Radio/RuView;
008's admission gate implemented in `crates/latentmesh-gate`, RVF packaging
and RVM enforcement not wired).

**What §8 does not convey: LatentMesh is a small research prototype.** The
entire workspace is 1,407 LOC across four crates:

| crate | LOC | tests |
|---|---|---|
| `latentmesh-core` | 272 | 6 |
| `latentmesh-align` | 454 | 6 |
| `latentmesh-gate` (incl. `causal.rs`) | 534 | 11 |
| `latentmesh-bench` | 147 | 0 |

That is real, tested code, but roughly 1/23rd the size of `ruvector-core`
alone (32,380 LOC / 503 tests). **There is no network transport crate
anywhere in the workspace** — no QUIC/TCP/transport module exists. This is
consistent with, and reinforces, `02-asset-map.md`'s own §8 correction that
`ruvector`-side LatentMesh work is greenfield, not integration — but it also
means "adopt LatentMesh ADR-009 as the spine" (this program's ADR-305) is
adopting a *design contract and ~1.4K LOC of primitives*, not inheriting a
mature system. Scope every ADR and work package that references LatentMesh
accordingly.

**Minor date correction**: `02-asset-map.md` §8 states LatentMesh was "last
updated 2026-08-19." The correct date, per the cloned HEAD commit, is
**2026-08-18**.

## 2. Autogenous — `02-asset-map.md` is materially incomplete

### 2a. The ADR catalog is undercounted, and the missing ADRs are the program itself

`02-asset-map.md` §9 cites autogenous's ADRs as **"391/392/393/397."** The
actual catalog, confirmed by directory listing, runs **391 through 403 — 13
ADRs, not 4**:

```
ADR-391-autogenous-governed-self-evolving-architecture.md
ADR-392-autogenous-genome-language-antibody-protocol.md
ADR-393-autogenous-product-thesis-adaptive-agent-firewall.md
ADR-394-cryptographic-closure-of-the-promotion-path.md
ADR-395-radio-realtime-streaming-peer-expert-mesh.md
ADR-396-peer-expert-protocol-security-and-governed-evolution.md
ADR-397-autogenous-streaming-mixture-of-agents.md
ADR-398-applications-and-development-loop-integration.md
ADR-399-provider-backed-mesh-run-rvm-rvf.md
ADR-400-self-evolving-mesh-perpetual-loop.md
ADR-401-perpetual-intelligence-machine.md
ADR-402-ruview-cognitum-spaces-spatial-intelligence.md
ADR-403-verifiable-execution-loop.md
```

The eight missing from the asset map's list matter directly to this program:

- **ADR-401 — "The Perpetual Intelligence Machine."** Status: **Accepted**
  (framing + capability map) · **Partial** (implementation). Dated
  2026-08-16. Defines a 10-capability map with an honest Built/Partial/
  Gap/Narrative status legend, a stage-by-stage operating loop, and a V1 +
  30-day acceptance test. Carries an explicit honesty anchor: *"'Perpetual'
  does not mean conscious, immortal, or continuously improving… The system
  targets operational continuity, not infallibility."* This is, on its face,
  substantially the same product this program's brief describes. See
  ADR-305 for the required reconciliation.
- **ADR-400 — "Self-Evolving Mesh Perpetual Loop."** Status: **Accepted —
  Implemented (first flywheel turn measured).**
- ADR-394 (cryptographic closure of the promotion path), ADR-395/396 (radio
  realtime streaming peer-expert mesh + its security/governed-evolution
  counterpart), ADR-398 (applications/dev-loop integration), ADR-399
  (provider-backed mesh runs on RVM/RVF).
- **ADR-402 — "RuView/Cognitum Spaces Spatial Intelligence."** Status:
  Accepted; read-side adapters, fail-closed seam, production HTTPS API
  complete; write-side sync and a 30-day acceptance test outstanding.
  Directly relevant to this program's RF Sensing Substrate context (WP14).
- **ADR-403 — "Verifiable Execution Loop."** Status: Accepted;
  `VerifiedPromotion` implemented, concurrent-promotion fencing and durable
  replay partially implemented, enforced isolation accepted-design-pending.

ADR-401 also cross-references **"metaharness ADR-322 (flywheel receipts/
promotion)"** — a third ADR series appearing in none of this program's
research documents. Its content is unlocated as of this addendum; see
ADR-305's Decision §3 for the resulting action item.

### 2b. The governance implementation is TypeScript, not the Rust crates `02-asset-map.md` implies

`02-asset-map.md` §9 lists autogenous's crates as a Rust workspace (`witness,
antibody, agl-types, generator, evaluator, verifier, lineage, ledger,
promotion, deployment, constitution, runtime, midstream-adapter, service`).
That crate list is correct as far as it goes, but ADR-401's capability map
points almost entirely at **`packages/radio-moe/src/*.ts`**, not the Rust
workspace:

- `packages/radio-moe/src` — 6,171 LOC; `packages/radio-moe/test` — 3,161 LOC
- All 11 modules ADR-401 names exist: `mesh.ts` (252), `mesh-evolve.ts`
  (277), `mixture.ts` (685), `action-gate.ts` (440), `failover.ts` (455),
  `disclosure.ts` (125), `reputation.ts` (154), `rvf-trajectory.ts` (67),
  `lineage-independence.ts` (243), `relevance.ts` (99), `capability.ts` (31)
- The Rust crates total 8,096 LOC / 91 test functions — real, but a
  different layer (crypto/protocol primitives) than the governance loop
  ADR-401 describes as Built/Partial.

Any PIR ADR describing autogenous's governance loop (ADR-315, this program)
should point readers at `radio-moe`, not only at the Rust crate names.

### 2c. Crate list omission

`02-asset-map.md` §9 lists 14 autogenous crates and gets 14 right, but
**misses `envelope`** (1,009 LOC, 7 tests) — the second-largest crate in the
workspace after `runtime` (1,471 LOC). Everything else in the asset map's
list is confirmed accurate.

### 2d. Cloud Run deployment — confirmed, not just inferred

`02-asset-map.md` documents Cloud Run deployment concretely for
`mcp-brain-server`/`ruvbrain`, but does not confirm it for autogenous
specifically. The root `Dockerfile` in `ruvnet/autogenous` is headed
*"autogenous-service — Cloud Run image (the Autogenous AGL control-plane as
a service)"*, multi-stage `rust:1-bookworm` → `debian:bookworm-slim`, `ENV
PORT=8080`, with a comment noting Cloud Run sets `$PORT`. Backed by
`crates/service` (980 LOC).

### 2e. The prototype-badge / Accepted-ADR tension

`02-asset-map.md` §9 correctly cites autogenous's README "research prototype"
status badge and its "Honest status" note that it is "not wired to live
MidStream/MetaHarness/RVF/RVM yet" and that "performance and economics claims
in the ADRs are hypotheses until benchmarked." What is worth flagging
explicitly: several of autogenous's own ADRs (400, 401, 402, 403) carry
**Accepted** status despite the repo-level prototype badge. When citing
autogenous maturity in any PIR ADR, cite the README badge for overall repo
maturity and the individual ADR's status line for that specific capability —
the two are not always in agreement, and neither alone tells the full story.

## 3. Witness-crate comparison (grounds ADR-312's reframe)

`02-asset-map.md` §9's "duplication note" (repeated in `03-program-plan.md`'s
original ADR #8 draft) described `rvm-witness` and autogenous's `witness`
crate as having "near-identical framing," recommending autogenous depend on
or converge with `rvm-witness`. A direct comparison of both crates does not
support a merge/dependency framing:

| | `autogenous/crates/witness` | `ruvector`'s `crates/rvm/crates/rvm-witness` |
|---|---|---|
| Size | 302 LOC, 4 tests, one file | 4,405 LOC, multi-file (`v2.rs` 880, `replay.rs` 253, `hash.rs` 204, `record.rs`) |
| Hash | SHA-256 over canonical JSON | u64 chain/record hashes (`compute_chain_hash`, `compute_record_hash`) |
| Auth | Ed25519 (`ed25519_dalek`), per-role `SigningAuthority` | Keyed chain MAC with derivation + ratcheting (`derive_chain_key`, `ratchet_chain_key`, `erase_key`, `CHAIN_KEY_CONTEXT`) |
| Record shape | `WitnessRecord` over JSON artifacts (genome, mutation, antibody, incident) | 64-byte cache-line-aligned privileged-action records in a fixed-capacity ring (`WitnessLog<const N>`, `WitnessLogV2<const N, const SEG>`, `DEFAULT_RING_CAPACITY = 262_144`) |
| Extras | `WitnessSeal`, `verify_seal`, `verify_chain` | `CoveragePolicy`/`CoverageError`, `verify_chain_v2_ratcheted`, `v1_head_to_genesis`, `verify_log_bytes`, queries by partition/action-kind/time-range |
| Runtime | std, service-side, offline+deterministic | `no_std`, hypervisor-side |

The shared element is the abstract idea of an append-only, hash-chained,
tamper-evident log — cryptographic primitives, data model, storage strategy,
and target runtime all differ, and each choice is correct for its own layer:
SHA-256-over-JSON does not fit in a `no_std` 64-byte-aligned ring buffer, and
`rvm-witness`'s u64 keyed-MAC chain is deliberately not a signature scheme.
**ADR-312 in this program set is written against a shared witness record
schema plus a cross-layer verification/anchoring contract, not a crate
merge or a dependency edge — merging the two crates as originally proposed
would be a design error.**

## 4. Summary of corrections to carry into this program's ADRs

| Claim in `02-asset-map.md` | Correction here | PIR ADR affected |
|---|---|---|
| LatentMesh "last updated 2026-08-19" | 2026-08-18 | ADR-305 |
| LatentMesh described without scale context | 1,407 LOC / 4 crates / no transport crate | ADR-305, ADR-309 |
| Autogenous ADRs "391/392/393/397" | 391–403 (13 ADRs); 400/401/402/403 previously uncited | ADR-305, ADR-315 |
| Autogenous governance described via Rust crate list only | Governance loop lives in `packages/radio-moe/src/*.ts` (6,171 LOC) | ADR-315 |
| Autogenous crate list (14 named) | Missing `envelope` (1,009 LOC) | ADR-312 |
| Autogenous Cloud Run presence unconfirmed for this repo specifically | Confirmed via root `Dockerfile` (`autogenous-service`, `crates/service`) | ADR-315 |
| `rvm-witness`/autogenous `witness` "near-identical framing" → merge/depend | Different primitives, data models, std/no_std, layers; reframe as shared schema + anchoring contract | ADR-312 |

## 5. `metaharness ADR-322` located — it is `ruflo` ADR-322, misattributed by autogenous

A fourth research pass located the ADR autogenous ADR-400/401 cite as
"metaharness ADR-322": it lives in **`ruvnet/ruflo`**, at
`v3/docs/adr/ADR-322-metaharness-flywheel-integration.md` — **"Adopt
`@metaharness/{flywheel,darwin}` as pluggable engines behind ruflo's ADR-176
self-improvement flywheel,"** status Accepted (phases 0–2 implemented),
dated 2026-07-28. It is ruflo's own ADR *about consuming* MetaHarness, not an
ADR published by the MetaHarness/`agent-harness-generator` project (which was
itself renamed to `ruvnet/metaharness`, 230 ADRs topping out at ADR-250 —
no ADR-322 exists there). **Cite this dependency as "ruflo ADR-322," never
"metaharness ADR-322."**

ADR-322 has three Accepted, implemented children directly relevant to this
program's promotion-gating ADRs (ADR-306, ADR-312):

- **322A** — evaluation↔promotion transaction model (`RUFLO_FLYWHEEL_TRANSACTION_V1`).
- **322B** — Darwin proposer adapter, enforcing a separation-of-powers
  invariant this program's own ADR-401 reconciliation (ADR-305) is reaching
  for independently: *"A proposer produces untrusted candidates only. It
  cannot issue promotion decisions or mutate active policy."*
  `promoteFlywheelCandidate` is the sole promotion authority.
- **322C** — receipt/ledger/verification protocol: RFC 8785 JCS canonical
  JSON, SHA-256 digests, **Ed25519 with domain separation**
  (`Ed25519(domainPrefix || 0x00 || canonicalBytes)`), UUIDv7 run IDs,
  independent statistical recomputation via deterministic paired bootstrap.

Separately, **ruflo PR #2956** implements the sequential-promotion-evidence
mechanism (anytime-valid e-process, `α_k = α_total · 6/(π²k²)`; 0.6%
measured family-wise false-promotion rate over 1,000 simulated nulls, the
bound holding **per epoch**), and **ruflo ADR-381** (**Proposed**,
2026-08-10) governs stream identity and budget-exhaustion recovery over it.
Together they are the concrete prior art for any PIR ADR claiming
statistical promotion gating. See §8d for the full correction — an earlier
version of this paragraph attributed the mechanism to ADR-381 directly,
omitted its Proposed status, and dropped the per-epoch qualifier; that
version is superseded by this one and by §8d, not by both disagreeing.

ruflo has already suffered the exact ADR-number collision this program is
trying to avoid: a dream-cycle research PR independently proposed
`ADR-322-dream-cycle-memory-typed-provenance.md` and was closed as superseded
when it collided with the already-merged flywheel-integration ADR-322 — a
live precedent for verifying numbers against the actual filename list
immediately before committing, not against a max+1 assumption.

## 6. `ruvnet/dream-machine` is a real, Accepted, shipped repo — corrects §"Dream Machine" reasoning above and in `02-asset-map.md`/`03-program-plan.md`

An earlier pass of this research reported "Dream Machine" as a name with no
matching asset, later corrected in `02-asset-map.md`'s revision to point at
`ruvector`'s own `scripts/research-gate/` + `ruvector-sota-bench/harness`
(ADR-282) as the functional equivalent. **Both were incomplete**:
`ruvnet/dream-machine` is itself a real, public, published-to-npm repo
(pushed 2026-08-19), and its own **ADR-0001** — *"The Dream Machine engine —
a config-driven, evidence-gated nightly evolution loop composed from the
ruvnet stack"* — is **Accepted, engine v0.1.0 shipped** (compile / ledger /
witness / schedule / memory modules + CLI/TUI, 85 tests, ~2,558 LOC across
six packages), dated 2026-08-13.

ADR-0001's own thesis is the one this program's brief and ADR-306 are
circling: *"Freeze the model. Evolve the harness. Evaluation is not
promotion — the machine never merges; a human does."* It **composes**
`@metaharness/flywheel` 0.1.10, `@metaharness/darwin` 0.9.1,
`@metaharness/redblue` 0.1.4, `metaharness` CLI 0.4.5, `ruvector` 0.2.41, and
`agentdb` 3.0.0-alpha (all optional/peer dependencies), and its own text
names `ruvnet/ruflo`'s and `ruvnet/metaharness`'s nightly dream-cycle
routines as prior instances it **subsumes** — specifically metaharness
ADR-251 ("MetaHarness Nightly Dream Cycle") and the Ruflo Nightly Dream Cycle
v3 **(the "metaharness ADR-251" half of that claim does not resolve — see
§8b; reported here as dream-machine ADR-0001's own assertion, not as
independently verified)**.

**Net effect**: the "Dream Machine" mechanism now exists in **four** places
— `ruvnet/dream-machine` (the generalized, already-shipped engine),
`ruvnet/ruflo`'s nightly dream cycle, `ruvnet/metaharness`'s `docs/dream-cycle/`
(not an ADR — see §8b for why the "ADR-251" identifier is wrong), and
`ruvector`'s own `scripts/research-gate/` + ADR-282. ADR-306 in this program
is revised to cite `ruvnet/dream-machine` ADR-0001 as the consolidating
design ruvector's `research-gate` should align with, not as a name this
program invents or as work fully covered by `research-gate` alone.

**Version-drift note**: `ruvnet/dream-machine` composes `@metaharness/darwin`
0.9.1 / `@metaharness/flywheel` 0.1.10; `ruvector` currently pins `darwin`
0.8.0 / `flywheel` 0.1.7 — three of the nine `@metaharness/*` packages
`ruvector` depends on (per `02-asset-map.md` §1) are behind the versions
`dream-machine` itself composes against. Worth a line in WP0b's dependency
remediation scope, not a blocker for this ADR set.

## 7. Open items not resolved by this addendum

None remaining from the original metaharness-ADR-322 question (resolved in
§5). Still open: no repo checked so far publishes a formal reconciliation
between `ruvnet/dream-machine` ADR-0001, `ruvector`'s ADR-282
(`research-gate`), and autogenous ADR-400/401's flywheel — WP1/WP2 own
producing that reconciliation, per ADR-305 and ADR-306.

## 8. PR #847 review corrections — a fourth research pass (adr-reviewer + asset-scout, post-merge-review)

An adversarial review of the initial PIR ADR set (PR #847) verified every
cross-repo claim against direct clones of `ruvnet/LatentMesh`, `ruvnet/autogenous`,
`ruvnet/dream-machine`, and `ruvnet/metaharness`, and separately against a
clone of `ruvnet/ruflo` (HEAD `fa13ee4`, 2026-08-15; 177 ADRs in
`v3/docs/adr/`) that §1–§7 above did not have access to. Three findings were
blocking; this addendum records the corrected facts so future PIR work
doesn't regress to the earlier drafts' claims.

### 8a. Autogenous ADR-401's promotion predicate is DONE, not open

§2a above (and the earlier draft of ADR-305/ADR-315) cited ADR-401's
capability-table row 5 ("not yet one checked predicate") as the program's
residual scope. **That row is stale relative to ADR-401's own Decision
section.** ADR-401's Update 1 §3, "Converge the promotion invariant to one
predicate," is marked **DONE**: `mesh-evolve.ts` exports
`promoteAuthorized(candidate, champion, { authorized, reversible }) →
PromotionDecision`, the single gate `Promote = Better ∧ Safe ∧ Authorized ∧
Reversible` with each conjunct independently blocking, `evolveMesh` routing
every promotion through it, proven by `test/promote-authorized.test.ts`
(all-four promotes; any three-of-four does not). ADR-315 (this program) is
rescoped accordingly: it adopts `promoteAuthorized`, it does not build or
close it.

### 8b. "metaharness ADR-251" does not exist — corrected provenance

ADR-306 (and this addendum's own §6, before this correction) restated
dream-machine ADR-0001's "Prior instances" citation of "metaharness ADR-251
(MetaHarness Nightly Dream Cycle)" as fact. **The dangling citation
originates in `dream-machine` ADR-0001 line 9 itself** — *"Prior instances:
`ruvnet/metaharness` ADR-251 (MetaHarness Nightly Dream Cycle)…"* — and was
relayed verbatim from that source by the research pass that first surfaced
it, without checking that it resolved. It is not a fabrication introduced by
this program on either the research or ADR-authoring side; it is a citation
inside a real, Accepted upstream ADR that itself needed resolving before
being repeated. Direct inspection of `ruvnet/metaharness` (HEAD `5453c8c`)
finds **230 ADR files topping out at `ADR-250-sota-proof-ladder.md`** — no
ADR-251. The Nightly Dream Cycle material instead lives in
`docs/dream-cycle/` (`2026-08-13-gist.md`, `2026-08-14-gist.md`,
`LEDGER.md`), not as an ADR. ADR-306 now cites `docs/dream-cycle/` and
states this provenance rather than repeating the citation as independently
verified. **Future readers**: the correct fix is ADR-306's current text —
do not "fix" it back to "metaharness ADR-251" on the assumption that an ADR
citation inside another Accepted ADR must already be resolved.

### 8c. "ruvector ADR-150 (optionalDependencies policy)" is a misattribution — resolved to `ruflo` ADR-150

ADR-313 (and ADR-306, and `03-program-plan.md`) cited "ruvector ADR-150" as
the source of the `optionalDependencies` policy `METAHARNESS-README.md`
claims compliance with. Checked against all three clones:

- `ruvector`'s own `ADR-150` is `ADR-150-pi-brain-ruvltra-tailscale.md` — "π
  Brain + RuvLtra via Tailscale — Semantic Embedding Upgrade." Unrelated.
- `metaharness`'s own `ADR-150` is
  `ADR-150-tailscale-local-frontier-concurrent-benchmarks.md`. Also
  unrelated.
- `METAHARNESS-README.md` itself attributes the policy to "**ADR-150**:
  MetaHarness Integration Surfaces (**upstream**)."

**A first pass through this section marked that attribution unverified. It
resolves cleanly**: the upstream document is **`ruflo`'s own ADR-150**,
`v3/docs/adr/ADR-150-metaharness-integration-surfaces.md` — *"ADR-150 —
MetaHarness Integration Surfaces in `npx ruflo`,"* **Status: Implemented**
(2026-06-16, rev. 06-17). The evidence chain: (1) the title matches
`METAHARNESS-README.md`'s citation on its distinctive phrase; (2) this ADR
set already established the README's *other* "upstream" citation
(ADR-103, same README, same word "upstream") is ruflo's — corroborated
independently by ruflo ADR-322C line 105 (see §8d) — same pattern, same
repo; (3) `METAHARNESS-README.md`'s own section headings mirror ruflo
ADR-150's four rules one-for-one (e.g. its "MetaHarness Removable" heading
against ADR-150's rule 1); (4) ruflo ADR-322 line 17 shows the policy in
force upstream, at the exact package versions ADR-306's version-drift note
already flags `ruvector` as pinning (`@metaharness/darwin@^0.8.0`,
`@metaharness/flywheel@^0.1.7`, both declared `optionalDependencies`).
ADR-150's normative rule 2 is verbatim the policy in question: *"Optional in
`package.json`: `@metaharness/*` packages MUST appear in
`optionalDependencies` or `peerDependencies` (optional), never in
`dependencies`."* Rule 4 — *"at least one CI job must run `--ignore-optional`
… the only structural defense against accidentally promoting an optional dep
to required"* — is a stronger, CI-testable acceptance criterion than a
plain `npm install` check, and is now the one this program's ADRs and plan
cite.

This is the exact failure mode ADR-305 §4 requires PIR documents to avoid
(repo-qualify every cross-repo ADR reference) reproduced inside the set
meant to enforce it — and it is now the **fourth** instance of the same
"wrong repo owns this ADR number" pattern this program has caught (ADR-103,
"metaharness ADR-322" → ruflo ADR-322, "metaharness ADR-251" → does not
exist, and this one). Every reference in the ADR set and the plan now cites
`ruflo` ADR-150 directly instead of an unverified attribution.

### 8d. The ruflo ADR-322 family verified verbatim; ADR-381 needed two corrections

A full clone of `ruvnet/ruflo` confirms the ADR-322 family exactly as this
program's ADR-306/310/312/313 cite it:

| ADR | Title | Status |
|---|---|---|
| 322 | Adopt `@metaharness/{flywheel,darwin}` as pluggable engines behind ruflo's ADR-176 self-improvement flywheel | Accepted — phases 0–2 implemented (2026-07-28) |
| 322A | Evaluation and promotion transaction model | Accepted — implemented (`RUFLO_FLYWHEEL_TRANSACTION_V1`) |
| 322B | Darwin proposer adapter | Accepted — implemented for bounded retrieval-policy candidates (`RUFLO_FLYWHEEL_DARWIN_V1`) |
| 322C | Receipt, ledger, and verification protocol | Accepted — implemented (`RUFLO_FLYWHEEL_RECEIPT_V1`) |

322B's separation-of-powers sentence is verbatim (`ADR-322B-darwin-proposer-adapter.md`
line 10): *"A proposer produces untrusted candidates only. It cannot issue
promotion decisions or mutate active policy."* ADR-322 line 15 reinforces it:
*"Darwin adapters remain candidate generators and never gain promotion
authority."* 322C's stack is confirmed with more precision than this
program's earlier drafts stated: canonical JSON is RFC 8785 JCS, digest is
SHA-256, signature is `Ed25519(domainPrefix || 0x00 || canonicalBytes)`
across **two distinct Ed25519 signing domains** — `ruflo/flywheel-receipt/v1`
(L96) and `ruflo/flywheel-ledger-head/v1` (L102) — plus a third
domain-separated prefix, `ruflo/bootstrap/v1` (L82), that seeds the
deterministic paired bootstrap's statistics rather than signing anything (an
earlier draft of this addendum and of ADR-312 called this "three signing
domains," overcounting by one), with `candidateId = SHA-256(JCS(candidate
policy))` and `receiptId
= SHA-256(JCS(unsigned receipt payload))`. Every authorizing term in a 322C
record is graded `recomputed`, `signature-verified`, or `trusted-assertion`.
322C line 105 also settles the ADR-103 cross-repo ambiguity this program
flagged in ADR-305: keys "use ADR-103's provider mechanism but a distinct
purpose/domain," confirming the witness-manifest ADR-103 is `ruflo`'s.

**ADR-381 needed three corrections.** (a) **Status is Proposed, not
Accepted** (line 3, dated 2026-08-10; no supersede or later Accepted line —
earlier drafts of ADR-306/310/312 implied it was settled governance
alongside its Accepted siblings). (b) **The 0.6% figure and the `α_k =
α_total · 6/(π²k²)` allocation belong to `ruflo` PR #2956's mechanism**,
which ADR-381 *governs*, not to ADR-381's own decision. ADR-381's actual
contribution is narrower: scoping the α ledger to one stream per project
root within the ADR-322 transaction state, and `resetSequentialEvidence` — an
explicit, `confirm: true`-gated, human-reasoned evidence-epoch reset for
budget exhaustion, expiring all outstanding receipts so stale evidence
can't be replayed against a fresh budget. (c) The false-promotion bound is
**per-epoch**, not global: line 33 states the guarantee as "family-wise
false-promotion probability is ≤ α_total **per epoch**" after a reset. Any
PIR document citing this bound now states it the same way.

### 8e. The `ruvllm` HTTP-307 bug is already fixed; the systemic lesson

ADR-313 (and `03-program-plan.md`) cited an HTTP-307 redirect bug in
`ruvllm`'s model-download path as an open blocker. It was already fixed on
`main` before this ADR set was written (commit `946275a61`, PR #590,
2026-06-18). The actual remaining download blocker is a GGUF glob/alias bug
in `ruvllm-cli`'s `get_files_to_download()` (`download.rs:193`'s glob
pattern disagreeing with `models.rs:65`'s alias resolution).

All three blocking findings (8a–8c) and this one share a root cause: a claim
was carried forward from a source document — an upstream ADR's stale table
row, a citation inherited from another repo's ADR, a documented-but-unlocated
upstream attribution, or an earlier bug report — without checking that
source's own current state via fix history (`git log` on the named path,
merged PRs, release notes). ADR-305 now states this as a standing
verification rule for the remainder of this program (see its Decision §6).
