# Evidence Review — RuV Perpetual Intelligence Runtime

Status: draft for ADR seeding. Compiled 2026-08-19 by parallel web-verification research agents against the primary sources (arXiv abstracts, fetched directly). Every citation below was independently located and read; nothing is fabricated. Two of the eight claims in the original brief could not be substantiated and are marked UNVERIFIED — the program plan must not depend on those two as load-bearing evidence.

## Summary table

| # | Claim (brief's shorthand) | Found | Source | Grade | Numbers verified |
|---|---|---|---|---|---|
| 1 | SHAPER — frozen-weight embodied skill/harness evolution | Yes | arXiv:2608.11350 | A | qualitative claim confirmed |
| 2 | WorldCycle — reversible-action verification signals | Yes | arXiv:2608.04964 | A | 44% drift reduction, ~4x accuracy — exact match |
| 3 | LiveMem — fixed-capacity persistent memory state | Yes | arXiv:2608.02515 | A | qualitative claim confirmed |
| 4 | TARL — 5-op executable memory ledger | Yes | arXiv:2608.03699 | A | qualitative claim confirmed |
| 5 | Causal audit of latent multi-agent comms | Yes | arXiv:2607.26773 | A | date discrepancy (see below) |
| 6 | LATTE — anomaly quarantine for latent comms | **No** | — | **UNVERIFIED** | not found under this name |
| 7 | Cross-model KV-cache mapping (2.7–25x) | Yes | arXiv:2608.03893 | A | 2.7–25x and 2/6-pairs-degrade — exact match |
| 8 | Universal WiFi CSI "RF latent vocabulary" | **No** | — | **UNVERIFIED** | no matching paper found |

Six of eight claims are grade A with primary sources directly confirming the mechanism and, where numeric, the exact figures. Two claims (LATTE, universal CSI vocabulary) are unverified after a genuine search effort and should be treated in the program plan as **internal design targets we would be first to build**, not as citations of prior work.

---

## 1. SHAPER

- **Found**: Yes — [arXiv:2608.11350](https://arxiv.org/abs/2608.11350), "Self-Evolving Embodied Agents via Skill-Harness Evolution," submitted 2026-08-11 (matches the brief's date exactly).
- **Grade**: A.
- **Claim vs. source**: Matches closely. The abstract confirms a train-free framework that keeps model parameters frozen while evolving reusable skills and a context/code harness through target-environment rollouts; the same frozen model serves as both planner and optimizer. Evaluated on VLABench and ESI-Bench (embodied agents with varied low-level action interfaces).
- **Notes**: No name collision found; "SHAPER" appears specific to this paper.

## 2. WorldCycle

- **Found**: Yes — [arXiv:2608.04964](https://arxiv.org/abs/2608.04964), "WorldCycle: Self-Verifiable Reinforcement Learning for Long-Horizon Video World Models," submitted 2026-08-05.
- **Grade**: A.
- **Claim vs. source**: Exact match on both numbers. Abstract states verbatim: "WorldCycle reduces state returning drift by up to 44% and lifts composite-action accuracy nearly 4x over the base model." A diagnostic benchmark, CycleBench, ships alongside it. Mechanism as claimed: verification signals derived from reversible/closed action cycles (an action sequence plus its inverse should return to the initial state) used for annotation-free supervision.
- **Notes**: No collision with unrelated prior work under this name.

## 3. LiveMem

- **Found**: Yes — [arXiv:2608.02515](https://arxiv.org/abs/2608.02515), "LiveMem: Maintaining Memory State Continuity in Long-Running LLM Inference," submitted 2026-08-03 (v1, matches exactly), revised 2026-08-07 (v2).
- **Grade**: A.
- **Claim vs. source**: Matches. Confirms a fixed-capacity memory state whose lifetime is independent of the active context, maintained via a parallel memory branch (Gated DeltaNet-2 backbone) alongside a bounded KV attention window — reframing long-running inference as "computational state preservation" rather than context growth. The system can answer questions from memory state after supporting evidence has left the context window.
- **Notes — name collision flagged**: "LiveMem" also names a pre-existing, unrelated consumer iOS app ("LiveMem – Live Photo Editor"). Different domain entirely, low confusion risk, but worth a footnote disambiguating "LiveMem (arXiv:2608.02515)" from the app.

## 4. TARL

- **Found**: Yes — [arXiv:2608.03699](https://arxiv.org/abs/2608.03699), "TARL: Transaction-Aware Reliable Ledgers for Executable Memory Management in Long-Term Agents," submitted 2026-08-04 (v1, matches exactly), revised 2026-08-11 (v2).
- **Grade**: A.
- **Claim vs. source**: Matches. Each incoming statement maps to one of five executable actions (add / ignore / revise-outdated-belief / reject-unreliable / defer-for-verification), maintaining accepted, pending, and rejected ledgers — explicitly framed as reducing "memory pollution" and limiting "cumulative corruption" (i.e., addressing memory poisoning).
- **Notes — name collision flagged, cite carefully**: "TARL" is heavily overloaded in prior ML/RL/robotics literature: "Taint Analysis and Reinforcement Learning" (robot software repair, arXiv:2005.03813), "Target-Aligned Reinforcement Learning" (arXiv:2603.29501), a separate "Test-time Adapted RL" concept, and an unrelated GitHub repo `xushoukai/TARL`. None relate to agent memory ledgers. **Always spell out "TARL (Transaction-Aware Reliable Ledgers)" on first use in any ADR or doc to avoid ambiguity.**

## 5. Causal audit of latent multi-agent communication

- **Found**: Yes — [arXiv:2607.26773](https://arxiv.org/abs/2607.26773), "Do Latent Channels Actually Communicate? A Causal Audit of Latent Multi-Agent LLM Communication."
- **Grade**: A.
- **Claim vs. source**: Content matches closely; **date discrepancy** — arXiv's own submission timestamp is **2026-07-29**, not Aug 5 as stated in the brief (no Aug 5 revision was found). Correct the date to late July 2026 in any citation. Substance confirmed: end-task performance gains alone don't establish that a receiver used task-relevant information passed through the latent (KV-cache) channel; the paper introduces a controlled-replacement causal audit, tested on Qwen3-4B/8B over GSM8K, ARC-C, and MATH-500.
- **Notes**: This paper is the evidentiary basis for invariant #3 in the brief ("every agent communication is attributable") — cite it as the motivating critique, dated correctly.

## 6. LATTE — anomaly quarantine (UNVERIFIED)

- **Found**: No.
- **Grade**: UNVERIFIED.
- **Search effort**: Extensive — direct phrase search, all major combinations. Found 8+ unrelated papers using "LATTE" as an acronym (latent diffusion transformer for video, atomic environment descriptors, hyperbolic Lorentz attention for EEG, robotics trajectory transformer, quantum error-correction decoding, federated test-time adaptation, bank-transaction embeddings, linear-time attention) — none relate to multi-agent latent-communication anomaly quarantine.
- **Closest genuine match on topic**: "When Latent Agents Lie: KV-Cache Integrity in Multi-Agent LLM Collaboration" ([arXiv:2606.28958](https://arxiv.org/abs/2606.28958)), fetched and confirmed to use an HMAC-SHA256 manifest-based integrity/tamper-detection scheme (774 honest payloads accepted, 295 tampered rejected) — related in spirit but a **different mechanism** (cryptographic verification, not statistical anomaly quarantine) and does **not** use the name "LATTE" anywhere.
- **Implication for the program**: Do not cite "LATTE" as prior art. Either (a) cite arXiv:2606.28958 for the integrity-checking half of the mechanism and design the statistical-anomaly-quarantine half as net-new work, or (b) treat anomaly quarantine for latent channels as an open research contribution the program can genuinely claim as novel, tied instead to LATTE's constitutional/witness-based verification concepts already present in `ruvnet/autogenous` and `ruvnet/rvm` (see 02-asset-map.md).

## 7. Cross-model KV-cache mapping

- **Found**: Yes — [arXiv:2608.03893](https://arxiv.org/abs/2608.03893), "Cross-Model KV Cache Transfer in LLM Families: A Closed-Form Linear Mapping for Prefill Reuse," submitted 2026-08-04 16:26 UTC (matches the brief's date exactly).
- **Grade**: A — the strongest-evidence item in this review.
- **Claim vs. source**: Exact match on every figure. Abstract states the mapper runs "2.7-25x faster than re-prefill" (matches). Tests six pairs across three model families; the linear mapper "retains 73-98% of the receiver's standalone-prefill accuracy on four pairs, while two degrade sharply" (matches "2 of 6 pairs degrade badly" exactly). A nonlinear MLP variant recovers up to +37pp HellaSwag accuracy on the failing pairs — directly supports the brief's implication that migration quality must be predicted/handled before blind use.
- **Notes**: This is the fast-follow item (brief's #4, KV migration in ruvLLM) — build directly against this paper's closed-form linear mapper plus the MLP fallback for degrading pairs, and implement the "predict transfer quality before migrating" routing gate the paper itself motivates.

## 8. Universal WiFi CSI "RF latent vocabulary" (UNVERIFIED)

- **Found**: No.
- **Grade**: UNVERIFIED (best adjacent work grades C — related but not matching).
- **Search effort**: Multiple phrasings tried (direct phrase, "chipset-agnostic," "shared vocabulary"/"common backbone," "tokenize" + CSI, explicit vendor-name combinations). No paper frames CSI from Realtek/Qualcomm/Nexmon/MediaTek chipsets specifically as a shared "RF latent vocabulary" feeding one common backbone.
- **Adjacent (not matching) work found**: "A Comprehensive Survey of Wireless Foundation Models for AI-Native 6G Networks" (arXiv:2608.14694, Aug 2026 — general survey of shared representations across heterogeneous wireless data, not chipset-vocabulary-specific); "WiFo-MiSAC" (arXiv:2604.18255, Apr 2026 — tokenizes heterogeneous signals via MoE backbone, but for multimodal sensing/comms, not cross-chipset CSI specifically); "UniFi" (arXiv:2512.22143 — irregular CSI sampling across packets/bands); "CSI-JEPA" (arXiv:2605.14171 — masked-prediction CSI foundation model).
- **Implication for the program**: This item (brief's #5, universal RF intelligence) is genuinely unclaimed territory — good news for novelty, bad news for evidence backing. Treat as an internal, first-party research bet with `ruvnet/RuView` as the implementation target, cite the four adjacent surveys/papers above as prior art to build on and differentiate from, and do not present it as validated by existing literature.

## Bottom line for ADR-seeding

- Build items **1 (SHAPER-style frozen-weight harness evolution)**, **2 (WorldCycle-style reversible-action verification)**, **3 (LiveMem-style persistent state)**, **4 (TARL-style transactional memory ledger)**, **5 (causal-audit-motivated attribution)**, and **7 (KV-cache cross-model migration)** all rest on grade-A, independently verifiable primary sources. These can be cited directly in ADRs.
- Item **6 (LATTE quarantine)** has no real citation — reframe the corresponding ADR as a novel contribution informed by arXiv:2606.28958's integrity-manifest approach plus `rvm`/`autogenous`'s existing witness-chain primitives, not as "implementing LATTE."
- Item **8 (universal CSI vocabulary)** has no real citation — reframe as a first-party research bet for `RuView`, citing the adjacent wireless-foundation-model literature as motivation, not as prior art being reproduced.
