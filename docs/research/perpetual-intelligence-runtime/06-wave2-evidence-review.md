# Wave-2 Evidence Review — Six Aug 18–19, 2026 Papers

Status: draft for ADR seeding. Compiled 2026-08-20 by direct web-verification
against primary sources (arXiv abstracts fetched directly, cross-checked
against the papers' HTML full text where the abstract page omitted a
claimed figure, and against each paper's linked GitHub repository via the
GitHub API where a link existed). Matches the grading discipline of
`01-evidence-review.md`: grade A/B/C, mark UNVERIFIED plainly, never
fabricate a citation, flag name collisions, and — per ADR-305 §6's
fix-history rule — treat every "code is available" claim as unverified
until the repository itself is checked, not just the paper's prose.

**One item was initially incomplete and is now resolved**: the brief for
this wave asked for "ruv's acceptance test verbatim." That text was not
present in the Wave-2 brief as relayed to this research pass, and a search
of the tracking issue (`ruvnet/ruvector#837`, all 14 comments as of
2026-08-20T12:41:41Z — the issue has since accumulated further comments)
found only the priority order, not an acceptance-test statement.
`07-wave2-program-plan.md` originally flagged this as an open item rather
than inventing test criteria; ruv subsequently supplied the exact wording
directly (2026-08-20), which is now carried verbatim in
`07-wave2-program-plan.md`'s acceptance-test section and in ADR-317. Issue
#862 is the test's canonical written record.

## Summary table

| # | Paper (brief's shorthand) | Found | Source | Grade | Numbers verified | Code/artifact available now |
|---|---|---|---|---|---|---|
| 1 | HarnessRisk | Yes | arXiv:2608.17597 | A | all figures exact match | **Yes** — `Baiyajing/HarnessRisk` (GitHub, real content) + HF dataset |
| 2 | TRUSS | Yes | arXiv:2608.17588 | A | all figures exact match | **No** — no code/data statement anywhere in the paper |
| 3 | MemFuse | Yes | arXiv:2608.18704 | A | qualitative claim confirmed | **Yes** — `Darwin-Agent/Mi-Memory/tree/master/MemFuse` (real content) |
| 4 | SkillForge | Yes | arXiv:2608.18933 | B+ | mechanism confirmed; headline "beats baselines" claim is qualitative only, no percentage in abstract | **Yes** — `cslsolow/SkillForge` (real content) |
| 5 | StagedWorkspace | Yes | arXiv:2608.18050 | A | all figures exact match | **No** — arXiv comments say "Under Review," no repo found |
| 6 | Pipeline shards | Yes | arXiv:2608.19147 | A | all figures exact match | **Yes** — `labscommunity/pipeline-sharded-inference-paper`, strongest artifact package of the six (full `reproduction/` directory with claims/hardware/results docs + scripts) |

All six papers exist, exactly as named, dated Aug 18–19 2026 as claimed. This
is a better hit rate than Wave 1 (six of eight found there; here, six of six).
The two load-bearing artifact questions the brief specifically flagged —
HarnessRisk's 128 cases and the pipeline-shards reproduction package — **both
check out as real, non-empty, publicly pushed repositories**, verified via
the GitHub API (file listings, sizes, push timestamps), not just by trusting
the papers' prose. TRUSS and StagedWorkspace, by contrast, have **no
available code or benchmark artifact today** despite TRUSS's benchmark names
(SkillInject, SkillSafetyBench, SkillGenBench) sounding like something to
build against — those three are *other people's* pre-existing benchmarks
TRUSS evaluates on, not artifacts TRUSS itself ships.

**Most important finding of this wave**: MemFuse has a serious, load-bearing
name collision — see §3 below — more serious than any collision flagged in
Wave 1, because it collides with an actively-used competing open-source
project in the *same problem space* (LLM memory), not an unrelated app.

---

## 1. HarnessRisk

- **Found**: Yes — [arXiv:2608.17597](https://arxiv.org/abs/2608.17597), "HarnessRisk: A Lifecycle-Oriented Benchmark for Agent Harness Safety," submitted 2026-08-18 (matches). Authors: Yajing Bai, Jinhao Duan, Jie Peng, Xianfeng Wu, Sijia Liu, Song Wang, Tianlong Chen. Subjects: cs.CR, cs.AI.
- **Grade**: A.
- **Claims vs. source** (all confirmed verbatim against the abstract and, for the one figure absent from the abstract, the paper's HTML full text):
  - 128 sandboxed cases — abstract: "HarnessRisk contains 128 sandboxed cases."
  - 14 model×harness configurations — abstract: "Across three harnesses, six language models, and 14 model and harness configurations."
  - Attack success range 12.6%–80.9% — abstract, verbatim: "attack success ranges from 12.6% to 80.9%."
  - Harness Configuration = most vulnerable phase — abstract, verbatim: "Harness Configuration is the most vulnerable phase across all three harnesses."
  - Same model, 4.3× difference across harnesses — **not in the abstract**, but confirmed in the paper body (Results, under the heading "The same model can be over four times less safe under a different harness"): Table 2 reports **"GLM-5.2 records a 54.7% ASR on OpenClaw but only 12.6% on Nanobot, a 4.3× difference."** Exact match to the brief's claim.
- **Artifact availability — checked, not assumed**: the paper states in Appendix H that it releases "the benchmark cases, harness adapters, mock-service implementations, evaluator prompts, and analysis scripts." A project page (`baiyajing.github.io/harness-risk/`) links to `github.com/Baiyajing/HarnessRisk` and a Hugging Face dataset (`huggingface.co/datasets/YajingB/HarnessRisk`). Verified directly via the GitHub API: the repo is **public, 574 KB, last pushed 2026-08-19T08:32:47Z**, and contains real content — `harness_adapter/`, `data/`, `services/`, `requirements.txt` — not a placeholder. **The 128 cases and adapters are genuinely available today.**
- **Name collision — flag for citation care**: a *different*, earlier paper, "Harness-Bench: Measuring Harness Effects across Models in Realistic Agent Workflows" (arXiv:2605.27922, `github.com/Qihoo360/harness-bench`, 106 sandboxed tasks), is close enough in name and topic (both are harness-configuration benchmarks) to cause citation confusion. They are unrelated projects by unrelated authors measuring different things (Harness-Bench: general harness-effect measurement; HarnessRisk: lifecycle-phase safety/attack-success). **Always cite as "HarnessRisk (arXiv:2608.17597)" in full, never abbreviate to a bare comparison with "Harness-Bench."**

## 2. TRUSS

- **Found**: Yes — [arXiv:2608.17588](https://arxiv.org/abs/2608.17588), "TRUSS: Towards Task-Reliable and User-Safe Automated Agent Skill Generation," submitted 2026-08-18 (matches). Authors: Zhibo Zhang, Zhen Ouyang, Ling Shi, Kailong Wang. Subjects: cs.AI, cs.SE.
- **Grade**: A.
- **Claims vs. source** (all confirmed verbatim against the abstract):
  - Static evaluation + shadow-agent execution with brokered tools and provenance traces — abstract, verbatim: "Candidates admitted by this static gate are loaded by a shadow agent inside a Controllable Execution Environment, where brokered tools expose requested actions to policy enforcement and record their results as provenance preserving execution traces."
  - Effectiveness 17.11% → 52.94% — abstract, verbatim: "TRUSS raises task effectiveness from 17.11% without Skills to 52.94%."
  - Security rate 50.80% → 100% — abstract, verbatim: "increasing the benchmark Security rate from 50.80% to 100.00%."
  - Bonus figures beyond the brief, also verbatim in the abstract: 100.00% precision/recall in vulnerability detection; repair reduces attack success from 38.71%→19.35% (GPT-5.5) and 46.45%→29.68% (GPT-5.4), "with zero attack regression"; evaluated on 168 SkillInject artifacts, 155 SkillSafetyBench cases, and all 187 SkillGenBench tasks.
- **Artifact availability — checked, not assumed**: no "Comments:" field on the arXiv abstract page, and a direct fetch of the paper's full HTML text found **no code, GitHub, dataset, "Availability," or "Reproducibility" section anywhere in the document**. A targeted web search for a standalone TRUSS repository returned nothing. **TRUSS's own implementation is not available today** — only the three benchmarks it evaluates *against* (SkillInject, SkillSafetyBench arXiv:2605.12015, SkillGenBench arXiv:2605.18693) are pre-existing, independently published artifacts, which is a point in TRUSS's favor as *evidence rigor* (it isn't grading itself on a benchmark it also wrote) but does **not** mean TRUSS's own shadow-execution framework/code is downloadable.
- **Name collision — flag, cite carefully**: "TRUSS"/"Truss" is moderately overloaded in exactly this program's domain — an active commercial security platform at `truss-security.com` (threat-intelligence agent network), a distinct AI coding-agent product at `truss-agent.com` ("bring your own model, review what the agent can do before it acts, choose permission policies") that is uncomfortably close in subject matter to this paper's brokered-tool/policy-enforcement mechanism, and a software consultancy at `truss.works`. None of these are the arXiv paper. Given the domain overlap with `truss-agent.com` specifically, **spell out "TRUSS (Task-Reliable and User-Safe Skill generation, arXiv:2608.17588)" on first use in any ADR**, the same discipline Wave-1 applied to TARL.

## 3. MemFuse

- **Found**: Yes — [arXiv:2608.18704](https://arxiv.org/abs/2608.18704), "MemFuse: Multi-Source Memory Fusion from Fragmented Observations," submitted 2026-08-19 (matches). 30 pages, 4 figures, 4 tables per the arXiv comments field.
- **Grade**: A.
- **Claim vs. source**: the brief's claim is qualitative (atomic source-tagged events fused into a causal graph with provenance to original evidence) and matches the abstract closely, verbatim: *"MemFuse, a structured memory system that preserves source-level evidence in event-layer atomic memory and organizes related atomic events into cluster-layer fused memory within a causal fusion graph."* The abstract additionally confirms retrieval-time traceability: *"During retrieval, MemFuse retrieves and organizes related evidence fragments while maintaining traceability to original source events."*
- **Artifact availability — checked, not assumed**: code is referenced at `github.com/Darwin-Agent/Mi-Memory/tree/master/MemFuse`. Verified directly via the GitHub API: `Darwin-Agent/Mi-Memory` is public, **14.5 MB, last pushed 2026-08-19T03:23:36Z**, and the `MemFuse` subdirectory contains real content (`MemFuse.pdf`, `MemFuseBench/`, `README.md`, `figure/`) — not a placeholder.
- **Name collision — SEVERE, the most important finding of this review**: `github.com/memfuse/memfuse` is an **established, actively used, unrelated open-source project** — "the lightning-fast open-source memory layer that gives LLMs persistent, queryable memory across conversations and sessions," with its own layered L0 (episodic)/L1 (semantic)/L2 (knowledge-graph) memory architecture. This predates the arXiv paper, is maintained by a different team, and — unlike Wave 1's LiveMem collision (an unrelated consumer photo-editing app, low confusion risk) — **this collision is a direct conceptual competitor**: both are literally "memory fusion for LLM agents" under the same name. There is also a third, likely-fork repository `xuyongfu/memfuse-0630` mirroring the `memfuse/memfuse` project's description, reinforcing that the pre-existing project has real usage/forking activity, not just a single abandoned repo.
  - **Action required before this program cites or ships anything under this name**: never use bare "MemFuse" as an npm package, crate, or module name in this program's own deliverables. Always disambiguate the paper as **"MemFuse (arXiv:2608.18704, `Darwin-Agent/Mi-Memory`)"**, explicitly distinct from the pre-existing `memfuse/memfuse` open-source memory layer, in every ADR and code comment that references it.

## 4. SkillForge

- **Found**: Yes — [arXiv:2608.18933](https://arxiv.org/abs/2608.18933), "SkillForge: Self-Distilling Agents for Project-Specific Issue Resolution," submitted 2026-08-19 (matches).
- **Grade**: B+ (mechanism confirmed verbatim; the brief's "beats issue-resolution baselines" claim is real but under-specified in the abstract — no percentage figure is given, so the *magnitude* of improvement is not independently verifiable from the abstract alone).
- **Claims vs. source**:
  - Synthesizes project-specific issues from test-covered functionality — abstract, verbatim: "SkillForge synthesizes project-specific issues by re-implementing test-covered core functionalities of the repository."
  - Distills entity-grounded skills — abstract, verbatim: "SkillForge distills reusable project-specific knowledge into entity-grounded skills and associates them with relevant repository entities for future issue resolution."
  - Beats issue-resolution baselines — abstract states this qualitatively only: "consistently improves issue resolution performance over strong baselines," with no percentage-point figure in the abstract text obtained. This is confirmed as a real claim, not fabricated, but the concrete magnitude requires reading the paper's results tables directly (not done in this pass) before citing a specific number in any ADR — **do not invent a percentage for this one.**
- **Artifact availability — checked, not assumed**: arXiv comments field states plainly, *"Our code and data are available at `github.com/cslsolow/SkillForge`."* Verified directly via the GitHub API: public, **2.5 MB, last pushed 2026-08-20T01:28:03Z** (same day as this review), containing real content (`distilling/`, `synthesis/`, `src/`, `evaluation_result/`, `pyproject.toml`) — not a placeholder.
- **Name collision**: no major pre-existing project found in a targeted search. "SkillForge" follows a common naming pattern ("X-Forge") shared by many unrelated tools/games, so it is not a *unique* name, but no specific conflicting project surfaced. Lower risk than TRUSS or MemFuse; still worth a final targeted check before this program uses "SkillForge" as its own package/crate name.

## 5. StagedWorkspace

- **Found**: Yes — [arXiv:2608.18050](https://arxiv.org/abs/2608.18050), "StagedWorkspace: A Versioned Workspace for Knowledge-Work Agents," submitted 2026-08-18 (matches).
- **Grade**: A.
- **Claims vs. source** (both confirmed verbatim against the abstract):
  - Binds views to content hashes of artifacts — abstract, verbatim: "The workspace binds parsed records and review diffs to content hashes of the native files as they change."
  - OfficeQA Pass@1 improvement of +8.3–12.1pp vs. single-view — abstract, verbatim: "dual parsed/native access has the highest point estimate for every tested model; relative to the more limiting single view, it improves OfficeQA Pass@1 by 8.3-12.1 points."
- **Benchmark provenance — checked**: "OfficeQA" is **not** a benchmark StagedWorkspace introduces; it is a pre-existing, independently published Databricks benchmark (`databricks/officeqa`, also documented in a separate arXiv paper, "OfficeQA Pro: An Enterprise Benchmark for End-to-End Grounded Reasoning," arXiv:2603.08655) for enterprise document-grounded reasoning over U.S. Treasury Bulletin data. This is a point in StagedWorkspace's favor on the same rigor axis as TRUSS above — it reports results on someone else's independent benchmark rather than a self-authored one.
- **Artifact availability — checked, not assumed**: arXiv comments field reads only "Under Review" — no code/data link. A targeted web search for a standalone "StagedWorkspace" repository found nothing. **Not available today.**
- **Name collision**: no matching existing tool or project found under this exact name in a targeted search (results returned only generic "staged"/"workspace hash" tooling unrelated to content-addressed agent workspaces). Low collision risk.

## 6. Pipeline shards

- **Found**: Yes — [arXiv:2608.19147](https://arxiv.org/abs/2608.19147), "Pre-Compiled Pipeline Shards for Distributed LLM Inference on Intel AI PC Fleets," submitted 2026-08-19 (matches).
- **Grade**: A — the strongest-evidence item in this wave, matching Wave 1's treatment of its own strongest item (cross-model KV-cache mapping).
- **Claims vs. source** (all confirmed verbatim against the abstract):
  - OpenVINO precompiled per-stage shards, with a specific optimization detail beyond the brief's summary: a naive per-stage export "misses an OpenVINO GPU optimization," and injecting a `beam_idx` Gather into each shard triggers the `IndirectKVCache` fusion, bringing shards to parity with the unsplit model.
  - Speculative decoding "on stateful OpenVINO models" — confirmed.
  - 1.79× concurrent throughput, two-node Llama 3.1 8B INT4 — abstract, verbatim: "a two-node Llama 3.1 8B INT4 pipeline serves two concurrent users at 1.79x the single-user throughput of the unsplit model on the same hardware."
  - Four-node Lunar Lake, 70B model — abstract, verbatim: "a four-node deployment of Lunar Lake AI PCs on Intel Tiber Cloud serves a single user at interactive speed, with output token-for-token identical to the same four-node pipeline decoding without speculation."
  - **Precision flag for downstream citation**: the brief's phrasing ("1.79× concurrent throughput 2-node Llama 3.1 8B, 4-node Lunar Lake running 70B") reads as one combined result but is two separate ones in the source — the 1.79× figure is reported only for the **2-node/8B** configuration; the **4-node/70B** configuration's reported result is single-user interactive-speed serving via speculative decoding (5.72 tok/s single-stream at 72.2% accept rate; 6.43 tok/s aggregate two-stream, per the paper's full text), with no 1.79× figure attached. `07-wave2-program-plan.md` WP20 must cite these as two distinct results, not one.
- **Artifact availability — checked, not assumed, and the strongest of the six**: abstract states, "Code, raw benchmark logs, and reproduction scripts ship as a self-contained package at `github.com/labscommunity/pipeline-sharded-inference-paper` (in the top-level `reproduction/` directory)." Verified directly via the GitHub API: public, **653 KB, last pushed 2026-08-19T16:59:13Z**. The `reproduction/` directory itself was checked and contains `CLAIMS.md`, `HARDWARE.md`, `MODELS.md`, `RESULTS.md`, `configs/`, and `scripts/` — a genuine, structured reproduction package, not just a code dump. This is the single best-evidenced artifact-availability case across both Wave 1 and Wave 2.
- **Name collision**: no significant collision found for "pipeline-sharded inference" or the paper's likely short names in a targeted search.

---

## Novelty claim — verdict: UNCERTAIN, leaning TRUE, with two close partial matches worth naming

**The claim checked**: ruv asserts no open-source system today combines, end-to-end in one pipeline: (1) harness-generation, (2) capability-controlled shadow execution with brokered tools, (3) verified/provenance-grounded learning, (4) cryptographic state binding, and (5) lifecycle security benchmarking.

**Search performed**: targeted web/GitHub search for systems combining these properties, plus a direct re-fetch of the most likely internal candidate — `ruvnet/autogenous` ADR-393 — rather than trusting a prior research pass's summary of it, per ADR-305 §6's fix-history rule and the explicit Wave-1 lesson that "Dream Machine doesn't exist" and "Autogenous is unfindable" were both wrong when a shallow search was trusted.

**No exact match for all five properties in one shipped, open-source system was found.** The closest candidates, each covering a subset:

- **`ruvnet/autogenous` ADR-393** ("Autogenous product thesis: the evolutionary control plane + adaptive agent firewall," Proposed, 2026-08-15, fetched directly) chains MidStream (signed evidence) → MetaHarness (candidate defense generation) → Darwin (competitive testing) → verifier (rejects capability expansion/regressions) → RVF (provenance-packaged winning defense) → RVM (constrained-authority deployment). This covers properties 1, 3, and 4 well and gestures at 2, but is **Proposed, not shipped** (the four-component MVP is explicitly "the next build"), has **no dedicated lifecycle security benchmark** (property 5) comparable to HarnessRisk's phase-by-phase ASR measurement, and is a **sibling repo this program already extends via ADR-315** — a full match here wouldn't falsify novelty relative to *external* prior art, only mean PIR shouldn't re-derive it (which ADR-305/315 already guard against).
- **`agentnotary`** (surfaced fresh via web search, not previously in this program's asset map) — a Python CLI notarizing/auditing AI agents with cryptographic seal, runtime guard, an adversarial fuzzer, and EU AI Act compliance docs. Covers properties 3 and 4 well and partially 5, but has no harness-*generation* (property 1) and no shadow-execution-with-brokered-tools design (property 2) — it audits an existing agent rather than generating and vetting new candidates.
- **TRUSS itself** (this wave's #2) is the closest single-paper match on properties 2 and 3 (shadow-agent CEE + brokered tools + provenance traces) but has no harness-generation, no crypto state binding, no lifecycle security benchmark, and — per §2 above — no confirmed code, so it isn't a deployed open-source *system* either way.

**Caveat, stated explicitly per the Wave-1 lesson**: this is a negative-existence claim over a fast-moving, adjacent-terminology space ("agent firewall," "agent governance," "AI-BOM"), and a shallow search missed real systems twice already in this program. This pass went one step further than Wave 1's failure mode by re-fetching the most likely internal candidate directly rather than trusting a summary, but it did not individually fetch and read every "agent security tooling" entry surfaced only by name (e.g. various `agentguard` variants). **Treat the novelty claim as directionally supported, not proven** — keep watching for a full 5-property match rather than treat this verdict as closed.

---

## Bottom line for ADR-seeding

- All six papers are real, dated as claimed, and every specific numeric claim in the brief checks out verbatim against the primary source — a materially better verification rate than Wave 1 (where two of eight claims were UNVERIFIED). None of the six should be treated as fabricated or as a name collision with an unrelated paper of the same title.
- **Two papers (HarnessRisk, pipeline shards) have genuinely available, verified-non-empty code/benchmark artifacts today** — both were independently confirmed via the GitHub API, not assumed from the papers' own claims. These are safe to build directly against.
- **Two papers (TRUSS, StagedWorkspace) have no available artifact today** — cite their mechanisms and numbers, but any PIR work package that assumes "clone their repo" for either one is currently wrong; build against the paper's described mechanism as a from-scratch implementation, the same posture Wave 1 recommended for the LATTE quarantine item.
- **SkillForge's code is available, but its "beats baselines" figure is not independently quantified from the abstract** — cite the mechanism with confidence, do not put an invented percentage into any ADR.
- **MemFuse carries this wave's most serious name-collision risk** — a live, unrelated, actively-forked open-source project shares the exact name in the exact same problem domain. Every ADR, package name, and doc referencing this paper must disambiguate explicitly (see §3).
- Consistent with `ADR-305 §6`'s fix-history rule: every "code is available" claim above was checked against the actual repository (file listing, size, push timestamp), not taken on the paper's word — this is the standard `07-wave2-program-plan.md`'s work packages should hold to when they, in turn, claim to build on these six papers.
