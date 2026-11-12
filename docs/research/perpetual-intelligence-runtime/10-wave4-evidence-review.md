# Wave-4 Evidence Review — Six Aug 20, 2026 Papers + NVIDIA Security Stack Items

Status: draft for ADR seeding. Compiled 2026-08-22 by direct web-verification
against primary sources (arXiv abstracts fetched directly, one full-text HTML
body fetched where the brief's numbers were not in the abstract, and every
candidate GitHub/GitLab artifact checked via the GitHub API or a live HTTP
fetch to distinguish "code released" from "code claimed") — the same
discipline `08-wave3-evidence-review.md` applied.

**Repo-state check, done before anything else, per the discipline Waves 1–3
required**: Waves 1, 2, and 3 are **confirmed merged on `origin/main`** —
ADR-305 through ADR-321, ADR-323 through ADR-327 (322 deliberately skipped;
it refers to *ruflo's* ADR-322 family, an external dependency — do not
allocate it), verified via `git ls-tree origin/main -- docs/adr/`, and all
24 PIR PRs (#847–#889 range) verified merged via `gh pr list`. Unlike Wave
3's kickoff, `docs/adr/INDEX.md` **is current this time** — its generated
header reads "Next available ADR number: 328," matching a direct recount of
the live filename list. **This wave's ADRs are ADR-328 through ADR-333,
confirmed free against the live tree at kickoff.** The local working copies
of this program's own docs 01–09 found on the stale
`ci/nextest-heavy-timeout` checkout are older than `origin/main`'s (main
carries corrections: Invariant 7, the four-stage memory-pipeline count, the
verbatim Wave-2 acceptance test) — `origin/main`'s copies are authoritative,
and this review was compiled against them.

**First correction to the brief before anything else**: the brief dates two
papers (AI4AI Bench, Pandora Router) to **Aug 21 2026**; both were actually
**submitted Aug 20 2026** (all six papers in this wave are Aug 20
submissions). One-day metadata error, no substantive impact, but ADRs should
carry the correct date.

## Summary table

| # | Paper (brief's shorthand) | Found | Source | Grade | Numbers verified | Code/artifact available now |
|---|---|---|---|---|---|---|
| 1 | AI4AI Bench | Yes | arXiv:2608.20318 | A | all figures exact match | **Yes** — `Einsia/AI4AI-Bench` (GitHub, Apache-2.0, real content, README links the arXiv ID) |
| 2 | ReCache | Yes | arXiv:2608.19662 | A | all figures exact match; metric is Inv-F1, not "accuracy" | **Yes** — `EIT-NLP/ReCache` (GitHub, linked from the abstract itself, real content) |
| 3 | CacheRoute | Yes | arXiv:2608.19677 | A | all figures exact match (with ± bounds the brief dropped) | **No** — and a **severe same-domain name collision** with `AstraNetLab/CacheRoute`, which is NOT this paper's repo |
| 4 | CAMA | Yes | arXiv:2608.19701 | A | mechanism claims verbatim; abstract reports **no quantitative figures at all** | **No** — no code link anywhere found |
| 5 | Pandora Router (+ judge panels) | Yes | arXiv:2608.20316 (+2608.19802) | A | mechanism claims verbatim; no numeric claims made in brief | **No** for both papers |
| 6 | RF sensing comparison | Yes | arXiv:2608.20322 | B+ | every number real, **but the brief's technology-to-number mapping is wrong** (swaps FMCW↔IR-UWB on both F1 and cost) | **Dataset yes** — live GitLab (imec); no model code found |

All six papers exist at exactly the arXiv IDs the brief gave — a clean 6/6
ID hit rate, better than Wave 1 and matching Wave 2/3 on existence. Two of
six ship artifacts today (AI4AI-Bench, ReCache) plus one dataset (RF
sensing); CacheRoute, CAMA, and Pandora Router are
reproduction-from-description efforts.

**Most important findings of this wave**:
1. **CacheRoute name collision** — the highest-starred GitHub repo of that
   exact name (`AstraNetLab/CacheRoute`, 259 stars) is in the *same domain*
   (KV-cache reuse for LLM serving) but is **not this paper's code**. Anyone
   grabbing "the CacheRoute repo" gets the wrong system. Severity comparable
   to Wave 3's SPADE collision. See §3.
2. **The brief's RF-sensing numbers are attributed to the wrong sensors** —
   the figures are real but the FMCW/IR-UWB columns are swapped. See §6.

---

## 1. AI4AI Bench

- **Found**: Yes — [arXiv:2608.20318](https://arxiv.org/abs/2608.20318),
  "AI4AI-Bench: Benchmarking LLM Agents in Algorithmic Design for Recursive
  Self-Improvement," submitted **2026-08-20** (brief said Aug 21 — wrong by
  one day). Authors: Yizhe Chi, Wenyi Li, Deyao Hong, Xiaoqiu Wang, Mingju
  Gao, Kaisen Yang, Bingxiang He, Youjie Zheng, Calvin Xiao, Qinhuai Na.
  Subjects: cs.AI, cs.CL, cs.LG.
- **Grade**: A.
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **10 frozen research repos, agent rewrites the learning algorithm** —
    verbatim: "We present AI4AI-Bench, 10 frozen research repositories
    spanning 10 training algorithm families. In each task, an agent has 4
    hours on one B300 to rewrite the training algorithm; its code is then
    rerun from scratch for up to 12 hours and scored by a fixed evaluator
    hidden from the agent." Matches the brief exactly.
  - **Best 0.250, avg 0.166** — verbatim: "Across 29 configurations of 6
    systems on all 10 tasks the mean score is 0.166, and the best system
    reaches 0.250." Exact match.
  - **Reasoning takes genuine-algorithm-change submissions 8%→64%** —
    verbatim: "More reasoning effort mostly buys the willingness to go
    there, taking that minority from 8% of submissions to 64% and the mean
    score from 0.094 to 0.196." Exact match, plus a bonus figure pair
    (0.094→0.196 mean score) the brief did not carry.
  - **Bonus context the brief omitted**: the score scale is anchored — 0 =
    uninformative model, **0.1 = the algorithm the repo already ships**,
    1.0 = task optimum. So "best 0.250" means the best system closed under
    a fifth of the ship-to-optimum distance. Also: "most [submissions]
    never change how the model learns at all, and the minority that do
    average 0.226 against 0.126 for the rest."
- **Artifact availability — checked, not assumed**: the abstract states "We
  release the task suite, the evaluators and every scored submission." The
  arXiv page has no Comments field with a link; a GitHub search found
  `Einsia/AI4AI-Bench` — created 2026-08-20 (submission day), Apache-2.0,
  696 KB, 14 stars, pushed 2026-08-21. Its README's first badge links
  **arXiv:2608.20318 directly** (confirming it is the authors' own repo, per
  the ADR-305 §6 check-at-source rule), plus a project homepage
  (lab.einsia.ai/ai4ai/), 290 released trajectories, and a Docker image
  (`chiyizhe/ai4ai`). Contents are real (orchestrator/, tasks/, tools/,
  pyproject) — not a placeholder. **The benchmark is genuinely available
  today.** Note the stated hardware bar: official runs target one NVIDIA
  B300; requires Docker + NVIDIA Container Toolkit and a native Codex or
  Claude CLI.
- **Name collision**: "AI4AI" is a generic-ish phrase used loosely in
  AI-for-AI-research discourse, but no established same-name benchmark or
  framework was found. Low risk; still cite with the full arXiv ID.

## 2. ReCache

- **Found**: Yes — [arXiv:2608.19662](https://arxiv.org/abs/2608.19662),
  "ReCache: Efficient KV Cache Reuse and Compression for Tool-Augmented LLM
  Agents," submitted 2026-08-20 (matches brief). Authors: Yichu Fang, Sitong
  Wei, Haozhe Hu, Xiaoyu Shen. Subjects: cs.CL. Comments: "17 pages, 4
  figures."
- **Grade**: A.
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **82.3% vs 82.4%** — verbatim: "Resource-wise attention matches dense
    invocation performance (82.3% versus 82.4% Inv-F1) while providing a
    3.655× time-to-first-token speedup." Exact figures — **but the metric
    is Inv-F1 (tool-invocation F1), not "accuracy" as the brief phrased
    it**. Cite the metric correctly in any ADR.
  - **3.655x TTFT** — verbatim, exact match (see quote above).
  - **92.43% KV memory reduction, 1.423x attention speedup** — verbatim:
    "The complete framework reduces allocated KV-tensor memory by 92.43%
    and accelerates attention by 1.423×." Exact match.
  - **Mechanism** — verbatim: resource-wise attention "removes
    cross-resource interactions and assigns resource-local positions,
    producing composition-invariant KV blocks," then restricts visibility
    to "contribution-selected layer–KV-head-group routes" plus structural
    and semantic pruning. Matches the brief's reuse+compression framing.
  - **Scale caveat the brief omitted**: experiments are on **Qwen3-1.7B and
    Qwen3-4B** (per the repo README) — small models. The 92% memory figure
    is unproven at 70B scale.
- **Artifact availability — checked, not assumed**: the abstract itself
  links https://github.com/EIT-NLP/ReCache. Verified via the GitHub API:
  public, 1,884 KB, 1 star, no license file, pushed 2026-08-21. README opens
  with the paper title and links arXiv:2608.19662 back — confirmed authors'
  repo. Contents are real (capsule/, evaluation/, utils/, assets/,
  requirements + requirements-training) — not a placeholder. **"Released
  with code" is confirmed true.** Minor hygiene note: no LICENSE file at
  check time — flag before any code adaptation.
- **Name collision**: "ReCache" collides with assorted small caching
  libraries across ecosystems, none in the LLM-agent domain. Low risk;
  disambiguate with the arXiv ID as usual.

## 3. CacheRoute

- **Found**: Yes — [arXiv:2608.19677](https://arxiv.org/abs/2608.19677),
  "CacheRoute: Planned Prefix-Affinity Routing for Large-Scale LLM
  Serving," submitted 2026-08-20. **Single author: Huang Cheng.** Subjects:
  cs.DC, cs.LG.
- **Grade**: A (all brief figures exact; the brief dropped the paper's own
  error bars and its deployment caveat — carry both into any ADR).
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **176 QPS on 60 H100s** — verbatim: "On Llama-3.3-70B in fp8 across 60
    H100 GPUs, CacheRoute sustains 176+/-11 QPS at a 3.5-s p99 SLO, 2.3x
    the strongest of five baselines." The brief's "176 QPS" is correct but
    dropped the ±11 bound, the fp8/70B context, the 3.5-s p99 SLO
    condition, and the 2.3x-over-baseline framing.
  - **KV hit rate 64.1%→93.2%** — verbatim: "Served KV-cache hit rate rises
    from 64.1+/-1.3% under cache-blind balancing to 93.2+/-0.5%." Exact
    match (± bounds again dropped by the brief).
  - **Mechanism** — verbatim: "a periodic routing plan. It admits high-rate
    keys to a stable warm set and places their assignments by expected
    load." Matches "prefix-affinity routing."
  - **Load-bearing caveats the brief omitted entirely**: (a) the primary
    workload is **semi-synthetic**; (b) verbatim — "Two 32B workloads
    provide the counterexamples: when affinity recovers too little KV work,
    its residual load skew reduces or erases the improvement. We therefore
    recommend gating any deployment with a shadow replay rather than
    enabling affinity from workload statistics alone." Any ADR adopting
    this must carry the shadow-replay gate as the paper's own
    recommendation, not an optional hardening step.
- **Artifact availability — checked, not assumed**: no code link in the
  abstract, no Comments field. A GitHub search for "CacheRoute" surfaces
  **`AstraNetLab/CacheRoute` (259 stars) — which is NOT this paper's
  repo**: it was created 2025-11-17 (nine months before submission), last
  pushed 2026-08-10 (ten days *before* submission), is a vLLM+LMCache-based
  *knowledge-injection* KV-cache system (chooses text-based vs
  KV-based knowledge injection per request) — a different mechanism from
  the paper's periodic prefix-affinity routing plan — and its README
  contains **no reference to arXiv 2608.19677 and an unchecked TODO
  reading "Paper and citation release."** No repo tied to the actual paper
  was found. **No code for arXiv:2608.19677 is available today; treat as a
  from-scratch reproduction.**
- **Name collision — the most severe of this wave**: `AstraNetLab/
  CacheRoute` is a live, actively developed, 259-star project **in the
  identical domain (KV-cache reuse for LLM serving)** sharing the exact
  name. This is the SPADE/MemFuse severity class: same field, high
  confusion risk, and in this case the collision is *more* dangerous than
  SPADE's because the colliding repo is what any naive "find the CacheRoute
  code" search returns first. **Always cite as "CacheRoute
  (arXiv:2608.19677, planned prefix-affinity routing)" and explicitly note
  that `AstraNetLab/CacheRoute` is an unrelated system; never adopt
  "CacheRoute" as a package/crate/module name.**

## 4. CAMA

- **Found**: Yes — [arXiv:2608.19701](https://arxiv.org/abs/2608.19701),
  "Beyond Memory Majority: Latent-Source Reasoning for Multi-Agent Memory
  Arbitration," submitted 2026-08-20. Authors: Chenchen Lin, Wenhao Yuan,
  Xuehe Wang, Edith Cheuk Han Ngai. Subjects: cs.AI. Title matches the
  brief exactly, including the CAMA expansion (Correlation-Aware Memory
  Arbitration — bolded as such in the abstract itself).
- **Grade**: A on the brief's claims (every mechanism claim confirmed
  verbatim) — with the explicit caveat that **the abstract contains no
  quantitative results whatsoever**, so this paper's evidence posture is
  the weakest of the wave (same class as Wave 3's DeAR "consistently
  outperforms" and Wave 2's SkillForge).
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **Memory Correlation Bias** — verbatim: "memories written by different
    agents may inherit the same upstream source or shared bias, causing
    correlated evidence to be repeatedly counted and creating a false
    majority. We term this failure mode Memory Correlation Bias." Exact
    match to the brief's framing.
  - **Neural dependency inference + symbolic provenance for effective
    independent source counting** — verbatim: "combine neural dependency
    inference with provenance-based symbolic priors to estimate the
    effective number of independent evidence sources." Exact match.
  - **Bonus mechanism the brief omitted**: CAMA also "learns a sequential
    recovery policy that actively retrieves alternative evidence or traces
    upstream sources before making the final decision" — active evidence
    recovery, not just discounting. Include if citing the mechanism.
  - **Results are qualitative only**: "Experiments on multiple benchmarks
    demonstrate the superiority of our method over the state-of-the-art
    baseline methods." No numbers, no named benchmarks in the abstract. Do
    not invent figures in any ADR.
- **Artifact availability — checked, not assumed**: no Comments field, no
  code link on the arXiv page, and a GitHub repository search for "CAMA
  memory arbitration" returned zero results. **No code available today;
  from-scratch reproduction posture.**
- **Name collision**: "CAMA" is a short acronym reused by unrelated
  projects/organizations (none found in the agent-memory domain).
  Moderate-generic risk — always cite with the arXiv ID and full expansion.

## 5. Pandora Router (+ related judge-panels paper)

- **Found**: Yes — [arXiv:2608.20316](https://arxiv.org/abs/2608.20316),
  "Pandora's AI Model Routing Box: Efficient Allocation with Costly Value
  Estimation," submitted **2026-08-20** (brief said Aug 21 — wrong by one
  day). Authors: Adam Fisch, Shubhendu Trivedi, Fantine Huot, William W.
  Cohen, Michael Kaisers, Mirella Lapata, Kate Larson, Jacob Eisenstein
  (a heavyweight, largely Google DeepMind-affiliated author list).
  Subjects: cs.AI. Note the actual title — "Pandora Router" is the name of
  the *centralized policy inside* the paper ("We call the centralized
  policy Pandora's Router"), not the paper's title.
- **Grade**: A (the brief made only mechanism claims; all confirmed
  verbatim).
- **Claims vs. source** (confirmed verbatim against the abstract):
  - **Value-of-information routing with costly estimators** — verbatim:
    "value estimation has a cost. Cheap estimators (e.g., embedding-based
    predictors) are fast but noisy, while accurate estimators ... are
    expensive. We formalize this tradeoff as an instance of Pandora's Box
    ... Under a Gaussian signal model, the resulting policies have
    closed-form value-of-information expressions." Exact match.
  - **Bonus the brief omitted**: a decentralized variant, "Pandora's
    Bidder," where specialists self-assess before accepting an offered
    price — with a stated negative result: "when competing estimates are
    noisy, however, it can increase the strategic specialist's utility at
    the expense of others." Carry the negative result if citing the
    decentralized variant.
  - Headline empirical claim (qualitative): "Pandora's Router matches the
    routing quality of exhaustive estimation, while querying the expensive
    estimator far less often" across three domains. No numeric figures in
    the abstract; do not invent any.
- **Related paper** — Yes, [arXiv:2608.19802](https://arxiv.org/abs/2608.19802),
  "Stopping and Routing LLM Judge Panels," submitted 2026-08-20, authors
  Bin Zhu, Yi Xie, Yanghui Rao (a *different, unrelated* author group —
  the brief's "Related" is topical, not a companion paper). Comments: "21
  pages, 2 figures, 20 tables. Accepted at WISE 2026." Classifies judges
  as copies/complements/specialists and derives stop/route policies across
  seven evaluation domains. Exists as described.
- **Artifact availability — checked, not assumed**: no code link on either
  arXiv page; GitHub searches for "Pandora Router LLM" returned zero
  relevant results. **No code available for either paper today.**
- **Name collision**: "Pandora" is maximally overloaded (music service,
  FMS tooling, countless projects). Never use a bare "Pandora" name; cite
  by full title + arXiv ID.

## 6. RF sensing comparison

- **Found**: Yes — [arXiv:2608.20322](https://arxiv.org/abs/2608.20322),
  "A comparison between ceiling-mounted FMCW, IR-UWB and Wi-Fi radar for
  in-bedroom human activity monitoring and sleep interruption detection,"
  submitted 2026-08-20. Authors: Anton Lambrecht, Reda El Hail, Xianjun
  Jiao, Pieter Crombez, Dominique Schreurs, Peter Karsmakers, Adnan Shahid,
  Eli De Poorter (Ghent/imec + KU Leuven ecosystem). Subjects: cs.LG.
  Comments: "submitted to IEEE Access Journal and is currently undergoing
  review" — i.e., **not yet peer-reviewed**.
- **Grade**: B+ — every individual number the brief quotes is real, but
  (a) most come from the paper *body/tables*, not the abstract (verified
  via the arXiv HTML full text, so confirmable — this is not the DeAR
  situation), and (b) **the brief's technology-to-number mapping is
  wrong**, which is a genuine discrepancy, not a gloss.
- **Claims vs. source**:
  - **Brief said**: "FMCW vs IR-UWB vs WiFi, macro F1 89.0/83.4/79.0."
    **Actual attribution** (abstract + body): **IR-UWB 89.0%** ("IR-UWB
    achieves the highest cross-subject activity recognition performance
    (89.0% macro F1)"), **FMCW 83.4%** (body: "Under LOBPO, FMCW changes
    from 83.4% to 83.8%"), **Wi-Fi 79.0%** (derived exactly from the body:
    Wi-Fi drops 10.2 pp to 68.8% under unseen-room evaluation ⇒ 79.0%
    cross-subject). The three numbers are correct; **the brief's stated
    order assigns 89.0 to FMCW — it belongs to IR-UWB**.
  - **Unseen-room 83.8 vs 68.8** — confirmed: FMCW 83.8% macro F1 on unseen
    room layouts (abstract, verbatim: "FMCW generalizes best to unseen room
    layouts (83.8% macro F1)"); Wi-Fi 68.8% (body: 79.0% − 10.2 pp).
    IR-UWB drops 10.5 pp (⇒ 78.5%). Match.
  - **Sleep 92.6%** — confirmed as the *floor*: body, verbatim — "under
    LOBPO, all three remain within 1.6 pp: IR-UWB reaches 94.2%, FMCW
    93.4% and Wi-Fi 92.6%." The brief's 92.6% is specifically Wi-Fi's
    (worst) sleep score; abstract says "all technologies exceed 92%."
  - **Costs €14/€20/€320** — confirmed from the body, verbatim: IR-UWB
    "totals approximately EUR 14," FMCW "roughly EUR 20," Wi-Fi
    "approximately EUR 320." Again, **against the brief's FMCW/IR-UWB/WiFi
    ordering the first two are swapped**: €14 is IR-UWB, €20 is FMCW.
  - The consistent pattern: the brief lists numbers in
    best-to-worst/cheapest-to-priciest order while naming the technologies
    in a fixed FMCW/IR-UWB/WiFi order. Any ADR must attribute per
    technology explicitly.
- **Artifact availability — checked, not assumed**: the paper releases an
  open synchronized dataset at
  https://gitlab.ilabt.imec.be/datasets/Activity-recognition-datasets —
  fetched directly, **HTTP 200, live GitLab project** ("Datasets / Activity
  Recognition Datasets"). No model-training code release was found — the
  CNN is described but not shipped. Dataset-available, code-absent posture.
- **Name collision**: not applicable (descriptive title, no system name).

---

## NVIDIA items — both exist

### "Where Security Fits in an AI Agent Stack" (developer.nvidia.com)

- **Found**: Yes —
  https://developer.nvidia.com/blog/where-security-fits-in-an-ai-agent-stack,
  published **2026-08-21**, authors Johnny Greco, Kirit Thadaka, Ali
  Golshan, Alex Watson (NVIDIA AI safety/security teams).
- **Content**: describes a five-layer agent stack — (1)
  distribution/product, (2) orchestration/meta-harness, (3) agent harness,
  (4) secure runtime, (5) inference infrastructure — and argues security
  must live **below the agent boundary**, in runtime and infrastructure,
  not in harness logic. Two load-bearing quotes: "The harness guides what
  an agent tries. The infrastructure controls what an agent can do," and
  "A layer designed to be modified cannot reliably enforce controls
  against its own modification." Controls are "enforced outside the agent
  process," with every high-impact effect crossing an enforcement point in
  the system that performs the action. Motivating context: summer-2026
  incidents reported by OpenAI, Anthropic, and the UK AI Security
  Institute of frontier agents operating beyond intended boundaries.
  OpenShell is the article's worked example of the secure-runtime layer.

### NVIDIA OpenShell

- **Found**: Yes — docs at https://docs.nvidia.com/openshell/ and repo at
  https://github.com/NVIDIA/OpenShell — verified via the GitHub API:
  **Apache-2.0, created 2026-02-24, 8,322 stars, 1,228 forks, 52 MB,
  pushed 2026-08-22 (today)** — a large, very active, genuinely open
  project, not vaporware.
- **What it actually provides**: an open-source **sandboxed runtime for
  autonomous AI agents** with kernel-level isolation — Landlock for
  filesystem restriction and seccomp for process/syscall restriction — 
  governed by a **declarative YAML policy** across four protection layers:
  filesystem (path-based access control), network (outbound connection
  filtering), process (privilege-escalation prevention), and **inference
  (API routing)**. Network and inference policies are hot-reloadable at
  runtime; filesystem and process policies lock at sandbox creation.
  Agents run **unmodified** — the docs name Claude Code, OpenCode, Codex,
  GitHub Copilot CLI (and OpenClaw per the repo description) — while
  OpenShell enforces controls externally with "a full audit trail of every
  allow and deny decision." Policy YAML is intended to be
  version-controlled and audited as security controls. Docs are
  agent-friendly (llms.txt index, .md page variants). Deployment is
  modular: community sandbox images or bring-your-own containerized
  runtime.
- **Relevance note for PIR**: OpenShell's placement thesis (enforce below
  the modifiable harness layer) is the same invariant this program's
  frozen-model/evolving-harness posture depends on — the blog is a
  credible third-party articulation of it, and OpenShell is a concrete,
  Apache-licensed enforcement surface worth evaluating against RVM's
  scope rather than duplicating.

---

## Bottom line for ADR-seeding

- **AI4AI-Bench and ReCache are the strongest-evidenced items**: grade A,
  verbatim figure matches, and real, live, authors'-own reference
  implementations confirmed via the GitHub API (`Einsia/AI4AI-Bench`,
  `EIT-NLP/ReCache`). Adapt-from-repo posture. Caveats: AI4AI-Bench's
  official runs assume a B300-class GPU; ReCache is validated only at
  Qwen3-1.7B/4B scale and its repo has no LICENSE file yet; ReCache's
  "82.3% vs 82.4%" is Inv-F1, not accuracy.
- **CacheRoute, CAMA, and Pandora Router have no available code** —
  from-scratch reproduction posture, sized like Wave 2's TRUSS/
  StagedWorkspace and Wave 3's D²ACCI/DeAR.
- **CacheRoute carries this wave's most serious name-collision risk** — a
  same-domain, higher-profile, unrelated repo (`AstraNetLab/CacheRoute`,
  259 stars) owns the name on GitHub. Disambiguate in every citation and
  never adopt the name internally.
- **The RF-sensing brief entry must be corrected before citation**: swap
  the FMCW/IR-UWB attributions (IR-UWB 89.0 F1 / €14; FMCW 83.4 F1 (83.8
  unseen-room) / €20; Wi-Fi 79.0 F1 (68.8 unseen-room) / €320), note the
  paper is under review at IEEE Access (not yet peer-reviewed), and note
  the release is a dataset, not code.
- **Both NVIDIA items are real**: the blog (Aug 21 2026) argues security
  belongs below the agent boundary; OpenShell is a live, Apache-2.0,
  8k-star kernel-level sandboxed runtime (Landlock + seccomp + YAML
  policy + audit trail) that runs Claude Code/Codex-class agents
  unmodified.
- **Brief metadata errata**: AI4AI-Bench and Pandora Router are Aug 20
  submissions, not Aug 21; "Pandora Router" is the policy name inside
  "Pandora's AI Model Routing Box," not the paper title; the judge-panels
  paper (2608.19802) is topically related but by a different author group.

Sources: arXiv abstract pages 2608.20318, 2608.19662, 2608.19677,
2608.19701, 2608.20316, 2608.19802, 2608.20322 (+ HTML full text of
2608.20322v1); GitHub API: Einsia/AI4AI-Bench, EIT-NLP/ReCache,
AstraNetLab/CacheRoute, NVIDIA/OpenShell; live fetches of
gitlab.ilabt.imec.be dataset page, developer.nvidia.com blog post,
docs.nvidia.com/openshell overview.
