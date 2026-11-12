# Wave-5 Evidence Review — NVIDIA Dynamo Skills, Cordis, OpenAI Monitoring, EMWM, Covenant

Status: draft for ADR seeding. Compiled 2026-08-23 by direct verification
against primary sources — every GitHub artifact fetched via the GitHub API
(not inferred from a search-result description), every arXiv claim checked
against arXiv API metadata and the abstract, the NVIDIA claim checked
against the **source `.mdx` in the repo** rather than only the rendered
docs page, and the merge date established from the actual merge commit and
PR record. Same discipline as `08-wave3-evidence-review.md` and
`10-wave4-evidence-review.md`, and per ADR-305 §6's check-at-source rule.

**Repo-state check, done before anything else, per the discipline every
prior wave required**: Wave 4 is **confirmed merged on `origin/main`** —
ADR-328 through ADR-334 all exist as files and are indexed, verified via a
direct listing of `docs/adr/`. `docs/adr/INDEX.md` is current and
self-consistent (generated header reads "Next available ADR number: 335",
"Highest allocated number: ADR-334"). **This wave's ADRs are ADR-335
through ADR-338, confirmed free against the live tree at kickoff.**
ADR-322 remains permanently skipped — it names *ruflo's* ADR-322 family, an
external dependency; do not allocate it. The `adr-numbering` CI job added
by PR #905 (`.github/workflows/ci.yml`, job `adr-numbering`, running
`node scripts/adr-index.mjs --check`) is present and passing on main, so
duplicate-number collisions are now caught mechanically rather than by
review attention.

**A working-copy warning that cost prior waves real time**: the local
checkout on `ci/nextest-heavy-timeout` is **stale** — its
`crates/ruvector-sota-bench/harness/src/` has 9 files against 22 on
`origin/main`, and its untracked copy of this program's own docs has 7
files against 11. Anyone reading the local tree will conclude Wave-4
components are absent. This review was compiled against `origin/main`
throughout; so should every Wave-5 work package.

**Method note on one source**: `openai.com` returns HTTP 403 to automated
fetches. The OpenAI claim was verified through three independent outlets
that quote the post verbatim and agree word-for-word on both the mechanism
description and the 20% figure. Graded accordingly (see §3) — high
confidence, but one step removed from the primary document. **A human
should open the URL before this citation ships in anything external.**

## Summary table

| # | Claim | Exists | Correct ID / URL | Grade | Numbers verified | Artifacts live now |
|---|---|---|---|---|---|---|
| 1 | NVIDIA Dynamo agent optimization skills | Yes | Page URL OK; canonical source `docs/fern/pages/blog/2026/agent-optimization-skills.mdx`; **PR #13557** | **A** (fidelity) — but see the n=2 flag | "15% to 77%" verbatim exact; merge date exact to the second | **Yes** — `ai-dynamo/dynamo` `.agents/skills/`, Apache-2.0 |
| 2 | Cordis (`cordiverse/cordis`) / spatiotemporal composability | Yes | `github.com/cordiverse/paper` — **briefing URL correct** | **A** | Attribution **confirmed** (PKU + DeepSeek-AI); date exact | **Yes** — paper PDF + implementation + `deepseek-ai/deepseek-harness` |
| 2b | deepseek-harness mount/unmount + "not a security boundary" | Yes | Path correct but on branch **`master`**, not `main` | **A** | Warning text quoted verbatim in §2b | **Yes** — MIT, live |
| 3 | OpenAI runtime monitoring architecture | Yes | URL correct | **A−** (primary blocked; three concurring verbatim mirrors) | 20% figure exact **and correctly attributed** | N/A (policy post) |
| 4 | Electromagnetic World Model (EMWM) | Yes | arXiv:2608.17769 — **exact hit** | **A** | 0.9699 SGCS and 28 GHz exact | **No** — no code, **and no dataset** |
| 5 | Covenant (`open-covenant/covenant`) | Yes | URL correct | **A** on existence; **strategic inference half-right** | Six named features all really implemented | **Yes** — live, pushed same day, but 8 stars |

**Headline result: zero misattributions this wave.** This is a first for
the program. Wave 1 carried two wrong IDs; Wave 3 carried one framing
overreach; Wave 4 carried two misattributed number sets (the CacheRoute
repo collision and the swapped FMCW/IR-UWB columns). This briefing's dates
are exact on all four dated items (Aug 21, Aug 13, Aug 18, Aug 18), and its
single high-risk affiliation claim — "DeepSeek and Peking University" for
Cordis — **is correct**, confirmed from the paper's own title page.

**The failure mode this wave is omission, not error.** Both items needing
correction are cases where every stated fact is true but the omitted
context changes the decision:

1. **The NVIDIA 15–77% range rests on exactly two paired comparisons.** The
   briefing says "internal paired tests" (plural, accurate) but not how
   few. The source says "one Claude Code pair and one Codex pair." Since
   this is priority 1 for implementation, see §1 — the *skills* are
   excellent evidence; the *number* is anecdotal.
2. **Covenant is real, but has ~zero adoption and is largely
   agent-authored.** The briefing's factual list is accurate; its strategic
   conclusion overstates the case in one direction and understates it in
   another. See §5.

A process suggestion follows from both: add **"what would change the
reader's decision if omitted?"** to the next briefing's checklist. Wave 4's
checklist question was "is this attributed to the right thing?", which this
briefing passes cleanly and which would not have caught either item above.

---

## 1. NVIDIA Dynamo agent optimization skills

- **Found**: Yes. The page at the briefing's URL
  (`https://docs.nvidia.com/dynamo/dev/digest/agent-optimization-skills`)
  resolves, titled **"Dynamo Agent Optimization Skills"**, subtitle "A
  skillpack that turns coding agents into disciplined performance
  engineers — August 2026". Author: Ben Hamm. Category "Agentic AI", dated
  **August 21, 2026**. Canonical in-repo source:
  `docs/fern/pages/blog/2026/agent-optimization-skills.mdx`, SPDX header
  `Apache-2.0`.
- **Grade**: **A** for source fidelity — every element of the briefing is
  verbatim correct. The evidence-strength caveat below is the substantive
  finding, and it is about sample size, not accuracy.
- **Merge date — confirmed exactly, not approximately**:
  [PR #13557](https://github.com/ai-dynamo/dynamo/pull/13557),
  "feat(agent-skills): performance optimization skill pack, agent roles,
  and guides", opened 2026-08-19T19:55:45Z, **merged
  2026-08-21T04:54:39Z** by `BenHamm`. 4,397 additions, 1,008 deletions,
  89 files changed. **The briefing's "merged Aug 21 2026" is exact.**
- **Mechanism claims — all four confirmed verbatim from the post**:
  - *Objective function + benchmark*: "your coding agent will work with you
    to clarify optimization objectives and turn those into a disciplined
    benchmarking script (default is AIPerf, our open-source benchmarking
    tool)."
  - *Experimental discipline*: "these skills ensure that your agent tracks
    experiments systematically, **isolates variables to test, and passes
    adversarial review before consuming valuable GPU time**. Absent skills,
    even frontier agents tend to struggle with running a clean lab."
    Confirms both "isolate experimental variables" and "adversarial review
    before spending GPU time" — exact.
  - *Domain knowledge*: "What optimization levers are there in Dynamo and
    which should be tried first? How do we layer cluster-level
    optimizations atop engine-level tuning? Agents don't know this
    natively."
- **The throughput numbers — verbatim, and this is the sentence that
  matters**:

  > "Compared to 'unskilled' coding agents with the Dynamo repo, our
  > skill-infused agents reached 15% to 77% better throughput in our
  > internal A/B tests (same model, same GPUs, same goal per pair; **one
  > Claude Code pair and one Codex pair**, measured with AIPerf)."

  It **is** a genuine paired A/B with the skillpack as the declared
  variable — "same model, same GPUs, same goal per pair" is exactly the
  control the briefing described, and NVIDIA states it plainly. **But
  n = 2 pairs.** "15% to 77%" is not a distribution or a confidence
  interval; it is the min and max of two observations. The 77% is a single
  run. No variance is reported, no repetition, no workload is disclosed for
  the A/B pairs, and there is no statement that the two pairs used the same
  workload as each other.
- **Discrepancy — by omission, not by error**: the briefing's phrasing is
  *literally true and correctly attributed*, but a reader would reasonably
  infer a test suite rather than two runs. **Any ADR citing this must carry
  the "one Claude Code pair and one Codex pair" parenthetical.** The
  defensible posture, given this is priority 1: adopt the skills for their
  *design* — which is independently sound and fully inspectable in the
  repo — and treat 15–77% as a directional anecdote, **not a target to plan
  against**. It is emphatically not a substitute for this program's own
  research-gate delta.
- **Stronger evidence the briefing did not carry** — the PR body describes
  a far more rigorous validation than the blog's A/B line:

  > "Testing: exercised end-to-end in isolated agent runs on real clusters
  > (SGLang on GB200, vLLM on H100); every recipe the workflow produced was
  > independently re-deployed from its shipped files alone and
  > re-benchmarked on the full workload, **reproducing or beating the
  > agents' claimed numbers**."

  That is a reproduction check on the *outputs* rather than a throughput
  delta, but it is better evidence that the skillpack works than the 15–77%
  range is. Cite it alongside, and prefer it.
- **Open source — yes, and the location is concrete**: the skills live at
  `.agents/skills/` in `ai-dynamo/dynamo` (public, 7,834 stars, pushed
  2026-08-23, actively developed). Confirmed by direct git-tree
  enumeration: **139 paths** under `.agents/skills/`. The eight
  optimization skills from PR #13557 are identifiable:
  `synthesize-user-workload`, `configure-aiperf-benchmark`,
  `run-aiperf-benchmark`, `analyze-aiperf-results`,
  `create-optimization-hypothesis`, **`perform-adversarial-review`**,
  `consult-perf-knowledge`, `author-baseline-dgd`, plus
  `deploy-dynamo-recipe`. Supporting material: five sub-agent roles under
  `agents/` (interviewer, deployer, perf analyzer, hypothesis generator,
  challenger) and guides under `agent-docs/` including
  `guides/optimization/optimize-loop.md` (the workflow spine) and
  benchmark-validity rules.
  **Vendoring note**: `.agents/skills` was demoted to a symlink in favour
  of `skills/` by PR #10017 (2026-05-27) — check both paths. Skills are
  `SKILL.md` instruction files plus Python/shell scripts, and are
  harness-agnostic by design ("Make our agent skills work with all agents,
  not just Claude Code", PR #8840). Licence is Apache-2.0 per SPDX headers;
  the repo's GitHub licence field reads `NOASSERTION` only because
  `LICENSE` is prefixed with a DeepSeek-V3.2 test-data notice.
- **Name collision**: none found in the agent-skills space. "Dynamo"
  collides broadly (Amazon DynamoDB, Autodesk Dynamo, Netflix Dynomite),
  so cite as "NVIDIA Dynamo (`ai-dynamo/dynamo`)" and the risk is low.

## 2. Cordis (`cordiverse/cordis`) — spatiotemporal composability

- **Found**: Yes —
  [`github.com/cordiverse/paper`](https://github.com/cordiverse/paper),
  **the briefing's URL is correct**. Repo created 2026-08-13T09:53:33Z,
  2,711 stars, 126 forks, last pushed 2026-08-22. Contains `README.md` and
  `paper.pdf` (88 pages, 2.1 MB). README states "Draft of August 13, 2026"
  — **the briefing's date is exact**, corroborated by the repo creation
  timestamp.
- **Grade**: **A**.
- **Attribution — the claim Wave 4's lessons said to check hardest, and it
  holds.** From the paper's own title page:

  > A Programming Paradigm for Spatiotemporal Composability
  > **Yifan Shi**^1,2, **Wei Zhang**^1, **Tianyi Cui**^2
  > ^1 **Peking University**  ^2 **DeepSeek-AI**

  **The briefing's "DeepSeek and Peking University" is correct** — both
  affiliations, no invented third party, no missing lead institution.
  Corresponding author `shigma@cordis.io` (Yifan Shi, known in OSS as
  "shigma").
- **The two composability definitions — verbatim from the abstract**:

  > "We identify two orthogonal dimensions of the problem: *temporal
  > composability*, the ability to completely revert a component's side
  > effects upon removal, and *spatial composability*, the ability to
  > declare and reactively manage inter-component dependencies."

  The briefing's "every side effect has an inverse, fully revertible" maps
  precisely; the abstract goes further — "we formalize *revertible
  effects*, in which **every context transformation carries an inverse that
  the runtime tracks**." The spatial gloss maps to "*reactive coeffects*,
  in which each change of the context notifies a component against its
  coeffect specification."
- **Is there a formal calculus, as claimed? Yes — verbatim**:

  > "we combine these mechanisms into the notion of a *component* and
  > **give a calculus of dynamic composition, whose metatheory carries
  > spatiotemporal composability from a single component to a whole system
  > of interleaved components**."

  A calculus with a stated metatheory and a compositionality result. The
  briefing's "formalizes" is justified. Theoretical framing: "lifting
  classical effect and coeffect concepts to runtime mechanisms," unifying
  effect and coeffect context into a single *context type*.
- **Three caveats the briefing did not carry**, all minor, all worth an ADR
  footnote:
  1. **Not peer-reviewed and not on arXiv.** Self-published PDF in a GitHub
     repo — no arXiv ID, no DOI, no venue. The README says so plainly:
     "This is a preprint under active revision. The content may change
     substantially; please cite the latest version and check back before
     relying on specific results." **Cite it with the commit SHA** or it
     will drift underneath us.
  2. **Cordis is not a DeepSeek invention and predates the paper by
     years.** The implementation repo `cordiverse/cordis` was **created
     2022-05-17** (7,191 stars, TypeScript) and is Shi's independent OSS
     project — the plugin kernel behind the Koishi chatbot framework, with
     a large community plugin ecosystem. The paper formalizes an existing,
     field-proven system; DeepSeek adopted it for its harness. Framing it
     as "DeepSeek built Cordis" would be wrong.
  3. **Same-day release coupling**: `deepseek-ai/deepseek-harness` was
     created 2026-08-13T11:56:32Z — roughly two hours after the paper repo,
     same day. Paper and harness are a coordinated release.
- **Name collision — real, and large.** "Cordis" collides with **EU
  CORDIS**, the European Commission's official research-and-development
  information service — the canonical public database of EU-funded
  projects. It is long-established, high-traffic, and generates real repos
  (`marzeelabs/cordis-serverless`, `KTH-Library/cordis`). There is also
  `cordis-lib/cordis`, a Discord API wrapper. **Never write a bare "Cordis"
  in an ADR, crate, or module name** — write "Cordis (`cordiverse/cordis`,
  the spatiotemporal-composability meta-framework)". Severity note: unlike
  SPADE (Wave 3) and CacheRoute (Wave 4), the colliding entity is in a
  *different* domain, so misretrieval risk is lower — but search-result
  noise is much higher, because EU CORDIS dominates the query.

### 2b. deepseek-harness — fully confirmed, with the warning text

- **File location — the briefing's path is right, but the branch is not
  `main`.** The file is at
  `.agents/notes/implemented/feature/2026-07-08-self-referential-cordis-toolset.md`
  on branch **`master`** (`deepseek-ai/deepseek-harness` has
  `default_branch: master`). Fetching from `main` returns 404. Trivial, but
  it would have cost someone time. Repo: MIT, **186,936 stars**, created
  2026-08-13, pushed 2026-08-21.
- **Mount/unmount mid-session — confirmed.** The note describes
  `@deepseek-ai/dsh-tool-cordis`, which "gives the model three tools over
  the live Cordis runtime in the current DSH process: inspect it, **mount
  an in-memory temporary Plugin, and unmount that Plugin to quiescence**."
  - `cordis_mount` — "Evaluates `code` now as an async JavaScript-function
    body in a `node:vm` sandbox and saves it nowhere."
  - `cordis_unmount` — "Unmounts one `cordis_mount` temporary Plugin by id
    and **returns only after every owned tool, listener, service, timer,
    and effect reaches quiescence**." This is §2's temporal-composability
    guarantee actually enforced at runtime — paper claim and harness
    implementation line up.
  - Cross-mount composition works through ordinary `provide`/`inject`:
    "unmounting A sends B back to pending (its registrations unwound)" —
    spatial composability, live.
  - Lifetime: "Temporary Plugins exist only in process memory... do not
    survive restart."
- **The security warning — the briefing's characterization is accurate.
  Verbatim:**

  > "The vm isolates accidental global pollution, and the context façade
  > hides framework internals. **Neither restricts the authority of exposed
  > services: a temporary Plugin can call `ctx.shell` with the host
  > executor's privileges and reach the real filesystem and web
  > services.** It runs in the shared DSH runtime and may affect other
  > sessions in that process. **This is an opt-in development tool with
  > bash-equivalent trust, not a security boundary or product default.**"

  And again, more emphatically, under rejected alternatives ("A hardened /
  capability-restricted sandbox"):

  > "Trapping Node built-ins and handing mount code a whitelist façade
  > rather than the raw context might suggest an intent to sandbox for
  > safety. **It is explicitly not that** ... the capabilities the façade
  > exposes (`ctx.shell`, `ctx.fs`, `ctx.web`) reach the real runtime,
  > **so it is not a security boundary**."

  **Both halves of the briefing's secondary claim are confirmed.** If
  anything the briefing *understates* it: the note also warns that a
  mounted waterfall listener "can stop the agent's own tool dispatch" —
  mounted code can disable the agent's own tooling. Also: "Host-realm
  helpers still make Node escape possible, consistent with the trusted
  posture," and "`vmTimeoutMs` bounds only synchronous evaluation."
- **Ecosystem note relevant to this repo**: three independent
  reimplementations exist and are compact references — `dshbox/cordis-rs`
  (Rust port, zero dependencies), `SheltonLiu-N/nano-cordis`,
  `adpanru/cordis-mini` (~600-line Python). **The Rust port is directly
  relevant given this repo's crates.**

## 3. OpenAI runtime monitoring architecture

- **Found**: Yes — "Pacing model development in an era of cyber-critical
  capabilities," at the briefing's exact URL, published **August 18,
  2026**. **The briefing's date is exact.**
- **Grade**: **A−**. Every claim confirmed with verbatim text, and three
  independent outlets quote the post identically — but `openai.com` serves
  HTTP 403 to automated fetches, so this is verified one step from the
  primary source. The consistency of verbatim quotes across unrelated
  outlets makes fabrication implausible; **a human should still open the
  URL before this is cited in a shipped ADR.**
- **The escalation hierarchy — verbatim, matching the briefing precisely**:

  > "**activation classifiers** — detectors that inspect a model's internal
  > activity — that **run at every sampled token** and **escalate potential
  > concerns to increasingly sophisticated and high-compute automated
  > investigators**"

  and what those investigators examine:

  > "**tool actions, available reasoning, and the full sequence of
  > activity** for unauthorized access, data theft, destructive behavior,
  > and attempts to defeat safeguards"

- **The 20% figure — verbatim, and correctly attributed by the briefing**:

  > "**OpenAI's current estimates put monitoring overhead at roughly 20% of
  > the inference compute being monitored**"

  with the immediate qualifier that cost "varies substantially across
  training and evaluation workloads."

  It **is** OpenAI's own estimate, explicitly self-attributed, and the
  denominator is **the inference compute being monitored** — *not* total
  inference compute, and not training compute.
- **Scope — the context the briefing omitted, and it materially changes the
  cost picture.** Monitoring is not applied to everything. It is "required
  for all RL training and evaluations involving tools for models of Sol
  capability or higher," and after OpenAI determined on **August 7** that
  its Astra model may have critical cyber capabilities, it added "an
  additional monitoring requirement for all inference of Astra with tools
  (not just RL training and evaluations)." So the 20% applies to a
  deliberately narrow, risk-selected slice. **An ADR that reads "monitoring
  costs 20%" without this scoping would overstate the budget by a large and
  unknown factor** — the fleet-wide cost depends entirely on what fraction
  of inference is placed under monitoring, which is a design choice, not a
  constant.
- **Additional confirmed detail, arguably the more actionable one**: a
  **30-minute alerting SLO**. If the system flags a likely violation of a
  critical security boundary it raises a highest-priority alert, pages the
  safety/security/research teams, "and if they cannot conclusively
  determine within 30 minutes that the flag is a false positive, those
  teams are expected to **pause the activity**." That is a fail-closed
  human-in-the-loop gate on a wall clock — directly relevant to anything
  this program builds around runtime governance.
- **Name collision**: none.

## 4. Electromagnetic World Model (EMWM)

- **Found**: Yes — [**arXiv:2608.17769**](https://arxiv.org/abs/2608.17769)
  resolves to exactly this paper. **The briefing's arXiv ID is exact.**
  Title: "Electromagnetic World Model for 6G: A Unified Framework for Joint
  Environment Reconstruction and Channel Prediction." Authors: Yizhu Zhao,
  Li Yu, Jianhua Zhang, Yuxiang Zhang, Zhen Zhang, Guangyi Liu. Submitted
  **2026-08-18T13:32:17Z** — **exact**. Primary category `eess.SP`
  (signal processing) **only**; note this is not a cs.AI/cs.LG paper, so it
  will not surface in ML-venue sweeps.
- **Grade**: **A**. Every figure and mechanism claim matches the abstract
  verbatim.
- **Claims vs. source** (all confirmed against the abstract):
  - **Joint learning from partial CSI and multiview RGB** — verbatim:
    "partial channel state information (CSI) and multi-view
    red-green-blue (RGB) images are encoded into CSI and visual tokens and
    jointly processed by a hierarchical world-model backbone with local and
    global aggregation." Matches, and adds architectural detail the
    briefing omitted.
  - **Two heads** — verbatim: "a **mixture-of-experts (MoE)-based CSI
    prediction head** reconstructs the complete CSI, while a **depth
    prediction head** estimates multi-view depth maps that are further
    converted into three-dimensional (3D) point clouds." Matches exactly.
    Bonus detail the briefing dropped: the CSI head is **MoE-based**, and
    point clouds are *derived from* predicted depth maps rather than
    predicted directly.
  - **0.9699** — verbatim: "achieving a **squared generalized cosine
    similarity (SGCS) of 0.9699** for CSI prediction." Exact match, and the
    briefing's metric name is also exactly right — a place where briefings
    often drift.
  - **28 GHz zero-shot** — verbatim: "while demonstrating robustness across
    different signal-to-noise ratio (SNR) conditions and **zero-shot
    generalization at 28 GHz**." Exact match; SNR-robustness is claimed
    alongside.
  - **Framing**: claims to be "the first unified framework for joint
    environment reconstruction and channel prediction," motivated by "the
    shared dependence of optical and radio-frequency signals on the
    surrounding environment." Baselines beaten are "conventional neural
    network and large language model (LLM) baselines."
- **Artifact availability — checked, not assumed: No, and worse than
  usual.** The arXiv API returns an **empty Comments field** (no project
  page, no code link), the abstract contains no repository URL, and GitHub
  searches for "electromagnetic world model", "EMWM 6G", and "EMWM CSI"
  return nothing attributable to these authors. **Neither code nor the
  dataset is available** — and the dataset matters more than usual here:
  the paper states "a large-scale multi-modal dataset is constructed based
  on a campus digital twin," which is the authors' own construction and is
  not released. **Reproduction therefore requires rebuilding a digital twin
  from scratch. Size any corresponding work package as
  reproduction-from-description with a heavy data-generation component**,
  not as an integration effort.
- **Affiliations**: arXiv's affiliation field is empty. Author names are
  consistent with a BUPT / China Mobile 6G-channel-modelling group, but
  this was **not** confirmed from the PDF and **must not be asserted in an
  ADR** without opening the paper. The briefing made no affiliation claim
  here, so there is nothing to correct.
- **Name collision**: "world model" is heavily overloaded (JEPA, Genie,
  Dreamer); "EMWM" is an unregistered acronym. Cite as "EMWM
  (arXiv:2608.17769)" in full.

## 5. Covenant (`open-covenant/covenant`) — strategic correction

- **Found**: Yes — live and actively developed. Rust, **Apache-2.0**,
  created **2026-04-24**, last pushed **2026-08-23**. Homepage
  `opencovenant.org`, docs at `docs.opencovenant.org`, a public live
  sandbox, a Zenodo DOI (10.5281/zenodo.20134416), and a published npm SDK
  (`@covenant-org/sdk`). Self-reported scale: "42 Rust crates, ~272k lines,
  3709 source-discovered Rust tests including 483 live boundary tests."
- **Grade**: **A** on "it exists and implements the listed features." The
  briefing's *factual* list is accurate. Its *strategic conclusion* is
  where this review diverges.
- **Feature-by-feature check of the briefing's six claims** — all six
  genuinely implemented, verified from README, `BUILT.md`,
  `docs/capabilities.md`, and `docs/runtime-sandbox-security.md`:

  | Briefing claim | Verdict | Evidence |
  |---|---|---|
  | Signed grants | **Yes** | `covenant-permissions`: "Signed capabilities with known-scope validation, dispatch-time enforcement, expiry, and revocation tombstones." |
  | Revocation | **Yes** | Revocation tombstones; peer revocation and operator-token rotation in `covenant-identity`. |
  | Audit chains | **Yes** | `covenant-audit`: "append-only JSONL events, local hash-chain integrity reports, retention controls, signed actions, and audit-root attestations." |
  | Memory | **Yes** | `covenant-memory`: SQLite-backed working / episodic / long-term tiers with embedding hooks, drift reports, repair, bounded compaction. |
  | Provenance | **Yes** | "Commit-scoped provenance envelopes that bind task records, changed Git blobs, transition events, and validation evidence," with `provenance.mjs verify-all` in CI. |
  | Fail-closed dispatch | **Yes, with a caveat** | Verbs "matched exactly," "deny-by-default grants"; unsupported sandbox policies "fail closed instead." |

- **Where the briefing is right, and it matters**: this is not vaporware.
  There is a real Rust daemon (`covenantd`), a CLI, a TUI, a Next.js
  operator console, MCP and A2A adapters, and a live public sandbox. The
  capability model is more detailed than a headline suggests — verb-exact
  matching, so `a2a.repair.requeue` does not authorize
  `a2a.repair.force_error`. **A claim of the form "we are the first to do
  signed capabilities with audit chains for agents" would be false, and the
  briefing is right to flag it.** Any ADR making that claim would repeat
  the Wave-1 "component absent" mistake and fail ADR-305 §6.
- **Where the briefing overstates — three things, and the third matters
  most**:
  1. **"Fail-closed dispatch" is real; sandbox isolation is not.** These
     are easy to conflate, and the project's own one-line description
     ("fail-closed sandbox dispatch") invites it.
     `docs/runtime-sandbox-security.md` is explicit: the default runner is
     `trusted-local` and "**is not a security boundary against hostile
     agent code**." A gVisor runner exists but is opt-in, Linux-only,
     supports only `filesystem=read-only-package` / `network=off`, and its
     live CI check "is not yet a required check." macOS is trusted-local
     only. `BUILT.md` disclaims "production sandbox-grade isolation for
     arbitrary untrusted agents" outright. **Covenant is prior art for
     capability-gated dispatch, not for sandboxed execution.**
  2. **Substantial parts are deployed-but-unexercised.** `BUILT.md` is
     unusually honest — it has a section literally titled "Honesty
     Boundaries" — and the pattern recurs: the Solana settlement program is
     on mainnet but "the daemon-driven economic lifecycle is not yet
     production"; the EAS reputation schema is registered "but no score is
     attested on-chain yet"; the bond verifier "is deployed but unfunded."
     Reading the feature list without `BUILT.md` overstates maturity —
     though the project actively guards against this, which counts in its
     favour.
  3. **Adoption is effectively zero, and the code is largely
     agent-authored.** **8 stars, 3 forks**, four months old. Contributor
     list: an account named `covenant` with **2,852 commits** against
     258 / 105 / 12 / 9 / 9 for humans — and `BUILT.md` states plainly that
     "Covenant is developed with an autonomous engineering loop." So ~272k
     lines of Rust largely written by agents, with essentially no external
     users. The project also has a memecoin (`$CVNT` on pump.fun) attached,
     **which shapes how an ADR should cite it** — as a technical reference,
     with the association noted, not as an ecosystem standard.
- **Assessment of the strategic conclusion.** The briefing's inference —
  "signed capabilities alone are no longer a unique claim" — is
  **half-right, and the half it gets wrong points the differentiation
  decision slightly astray.**
  - **Right**: as a *novelty* claim it is dead. Drop that framing entirely.
  - **Overstated**: prior art in *code* is not prior art in the
    *ecosystem*. An 8-star, four-month-old, agent-authored project with no
    external adoption does not foreclose a design space the way a widely
    deployed standard would. Covenant establishes that the *idea* is not
    novel; it does not establish that the *problem is solved*.
  - **The recommendation this review makes instead**: differentiate on the
    axes Covenant explicitly disclaims rather than abandoning capabilities.
    Its own Honesty Boundaries name the gaps — production sandbox-grade
    isolation for untrusted agents, production multi-peer operation across
    untrusted hosts, and benchmarked self-improvement are all listed as
    **not claimed**. Combine that with §2b (DeepSeek ships a live
    mount/unmount runtime it **explicitly refuses to call a security
    boundary**) and §3 (OpenAI pays ~20% of monitored inference compute for
    runtime monitoring **because static gating is insufficient**), and
    **three of this wave's five items converge on the same gap**: the
    unsolved problem is **runtime enforcement and isolation under a live,
    mutating capability set** — not the grant/revoke/audit ledger, which is
    now table stakes. That is a sharper thesis than "capabilities are
    taken," and it is supported by three independent sources rather than
    one.
- **Name collision — real, and worse than Cordis's for our purposes.**
  "Covenant" is generic, and the top GitHub hit by a wide margin is
  **`cobbr/Covenant` (4,729 stars): "a collaborative .NET C2 framework for
  red teamers"** — an offensive-security tool, *adjacent to the security
  domain we are writing in*. That makes it a **Wave-3-SPADE-class
  collision**: someone searching "Covenant agent security" lands on a
  command-and-control framework. There is also
  **`csehammad/covenant-layer` (52 stars): "an open model for
  outcome-based coordination between users, agents, brokers"** — *the same
  domain as ours* — plus `EthicalSource/contributor_covenant` (2,247),
  `CovenantSQL/CovenantSQL` (1,528), and `patriksvensson/covenant` (an
  SBOM tool). **Never write a bare "Covenant" in an ADR.** Always
  "Covenant (`open-covenant/covenant`, opencovenant.org)".

---

## Bottom line for ADR-seeding

- **NVIDIA Dynamo skills are the strongest-evidenced item and the right
  priority-1 choice** — Apache-2.0, live, inspectable, harness-agnostic,
  with a genuinely rigorous output-reproduction check in the PR body.
  **But cite the throughput range with its n=2 parenthetical, and never as
  a target.** Our bar remains our own research-gate delta.
- **Cordis's formalization is real and its runtime counterpart is live** —
  adopt the *concepts* (revertible effects with runtime-tracked inverses;
  reactive coeffects), cite by commit SHA, and do not claim DeepSeek
  invented Cordis.
- **deepseek-harness is the clearest statement of the gap this program
  should target**: a live mount/unmount capability runtime whose authors
  state outright that it "is not a security boundary."
- **OpenAI's monitoring architecture is sound to build on, but its 20%
  figure must always carry its denominator** (the monitored subset) and its
  narrow scoping. The 30-minute fail-closed alerting SLO is arguably the
  more actionable design element.
- **EMWM is the expensive kind of reproduction-from-description** — no
  code *and* no dataset, with a self-built digital twin as the missing
  input. Do not size it as an integration.
- **Covenant forecloses the novelty framing, not the design space.** The
  differentiation thesis is signed **plus reversible plus evolvable**,
  targeting runtime enforcement under a mutating capability set — backed by
  three of this wave's five sources.
- **Two name collisions to enforce with full-qualification discipline**:
  "Cordis" (EU CORDIS — different domain, very high search noise) and
  "Covenant" (`cobbr/Covenant` C2 framework — adjacent domain;
  `covenant-layer` — same domain). Same treatment as SPADE and CacheRoute.
- **One branch-path correction**: the deepseek-harness note is on `master`,
  not `main`.

Sources: `docs.nvidia.com/dynamo` blog page + `ai-dynamo/dynamo` PR #13557
and git-tree enumeration of `.agents/skills/`; `cordiverse/paper`
(README + `paper.pdf` title page and abstract) and `cordiverse/cordis`
metadata; `deepseek-ai/deepseek-harness` `.agents/notes/implemented/
feature/2026-07-08-self-referential-cordis-toolset.md` (branch `master`);
three independent verbatim mirrors of the OpenAI post of 2026-08-18;
arXiv API metadata and abstract for 2608.17769; `open-covenant/covenant`
README, `BUILT.md`, `docs/capabilities.md`, `docs/runtime-sandbox-security.md`,
and GitHub API contributor/star metadata.
