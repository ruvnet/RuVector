# ADR-062: Brainpedia — Structured Knowledge Encyclopedia with Delta-Based Editing

**Status**: Accepted
**Date**: 2026-02-27
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities), ADR-061 (Reasoning Kernel Architecture)

## 1. Context

The Shared Brain (ADR-059, ADR-060) accumulates knowledge as individual memories — embeddings, quality scores, witness chains. But memories are granular artifacts, not structured knowledge. A developer looking for "how does RVF serialization work" may find 30 loosely related memories with no coherent narrative.

Wikipedia solves this for human text. The Brainpedia applies the same concept — canonical, community-maintained knowledge pages — but the unit of contribution is not raw text. It is a structured, versioned, verifiable knowledge artifact with outcomes.

Every entry is an RVF memory plus a delta stream. Contributions are deltas, not overwrites. Evidence links tie claims to outcome proofs. Transfer assets make knowledge portable across domains. The result is a knowledge encyclopedia that is executable, learning-native, transferable, and reversible.

## 2. Design Principles

### 2.1 The Unit of Contribution

A Brainpedia contribution is one of:

**RVF Memory**: A canonical knowledge artifact with the standard 10-segment layout (ADR-060 Section 2.5). This is the stable snapshot — the "current page version."

**Delta Entry**: A structured modification to an existing page. Deltas carry their own witness chain and are individually verifiable. A delta contains:
- `parent_id`: The memory ID being modified
- `delta_type`: One of `Correction`, `Extension`, `Evidence`, `Deprecation`
- `content_diff`: The structured change (not raw text diff — semantic field updates)
- `evidence_links`: Outcome proofs that support the change
- `contributor_id`: Pseudonym of the contributor
- `witness_hash`: SHAKE-256 chain entry for this delta

**Evidence Link**: A reference to a verifiable outcome that supports a claim. Evidence types:
- `TestPass { test_name, repo, commit_hash }`: A test that passed after applying the knowledge
- `BuildSuccess { pipeline_url, commit_hash }`: A CI build that succeeded
- `MetricImproval { metric_name, before, after, measurement_url }`: A measured improvement
- `PeerReview { reviewer_pseudonym, vote_direction, quality_score }`: A peer attestation

### 2.2 Properties of Knowledge Entries

Every Brainpedia entry is:

**Executable**: Contains embeddings, transfer priors, and policy kernels that can be directly consumed by `brain_search`, `brain_sync`, and `brain_transfer`. Knowledge is not just readable — it is machine-consumable.

**Learning-native**: Quality scores update via Bayesian voting (BetaParams). High-quality entries strengthen the MicroLoRA consensus. Low-quality entries are auto-archived. The encyclopedia improves through use.

**Transferable**: Transfer assets (TransferPrior, PolicyKernel, CostCurve) enable cross-domain knowledge portability. A debugging pattern from Rust can transfer to TypeScript with measured acceleration.

**Reversible**: No direct mutation. Every change is a delta. The full history is reconstructable from the delta stream. Any delta can be reverted by applying a compensating delta. Witness chains prove the order and integrity of all changes.

## 3. Page Structure

A Brainpedia page is a canonical memory plus its delta log.

### 3.1 Canonical Page

The canonical page is a `BrainMemory` with `page_status: Canonical`. It represents the current community-accepted version of the knowledge. Canonical pages have:

- `category`: The knowledge domain (Architecture, Pattern, Solution, Debug, etc.)
- `title`: Human-readable page title (unique within category)
- `content`: The structured knowledge content
- `tags`: Searchable labels
- `embedding`: Structured hash features + MicroLoRA transform (standard pipeline)
- `quality_score`: Bayesian BetaParams accumulated from votes
- `evidence_count`: Number of verified evidence links
- `delta_count`: Number of applied deltas
- `transfer_assets`: Optional TransferPrior + PolicyKernel + CostCurve

### 3.2 Delta Log

Each page maintains an ordered delta log. Deltas are immutable once accepted. The delta log enables:

- **Version reconstruction**: Any historical version can be reconstructed by replaying deltas
- **Attribution**: Every change is attributed to a specific contributor
- **Audit**: Witness chains prove the integrity of the delta sequence
- **Revert**: A `Deprecation` delta marks a previous delta as superseded

### 3.3 Evidence Links

Evidence links are the mechanism by which the encyclopedia distinguishes claims from knowledge. A claim without evidence is a candidate. A claim with evidence is knowledge.

**Evidence lifecycle**:
1. Contributor submits a delta with one or more evidence links
2. Server verifies evidence link format and contributor authentication
3. Evidence is recorded in the delta's witness chain
4. Other contributors can add corroborating evidence via additional deltas
5. Quality score reflects the evidence density: more verified evidence = higher quality

## 4. Governance Gates

Three gates control what enters the encyclopedia and how.

### 4.1 Gate 1: Identity

All contributions require API key authentication with contributor pseudonym derivation (same as ADR-060 Section 6). Anonymous users can read but cannot contribute. This prevents Sybil attacks on the knowledge base.

### 4.2 Gate 2: Evidence

**For deltas**: Every delta must include at least one evidence link. A correction without evidence is rejected. An extension without evidence is held in a `Proposed` state until evidence is added.

**For new pages**: A new canonical page requires at least 3 evidence links from independent contributors (not the page creator) before promotion from `Draft` to `Canonical`. This ensures community validation before knowledge becomes authoritative.

**Evidence verification**: The server does not execute tests or check CI — it verifies that evidence links are well-formed, that the contributor is authenticated, and that the linked artifacts exist (where checkable). Outcome verification is a community responsibility: voters assess whether evidence is credible.

### 4.3 Gate 3: Consensus

**Page promotion**: A Draft page becomes Canonical when:
- `quality_score.mean() >= 0.7` (community approval)
- `quality_score.observations() >= 5` (sufficient review)
- `evidence_count >= 3` from `>= 2` distinct contributors

**Delta acceptance**: A delta is accepted when:
- Contributor is authenticated (Gate 1)
- At least one evidence link is provided (Gate 2)
- Contributor has not been flagged for poisoning (reputation > 0.1)

**Delta promotion to canonical**: When a delta accumulates sufficient quality votes, the canonical page is updated to incorporate the delta. The old canonical version is preserved in the delta log.

## 5. Submission Model

### 5.1 New Users (Reputation < 0.5, Contribution Count < 10)

New users can:
- Read all pages and delta logs
- Submit deltas to existing pages (with evidence)
- Vote on pages and deltas
- Cannot create new canonical pages

This prevents low-reputation users from flooding the encyclopedia with low-quality pages while still allowing them to contribute improvements to existing knowledge.

### 5.2 Established Users (Reputation >= 0.5, Contribution Count >= 10)

Established users can:
- All new user capabilities
- Create new pages (initially as `Draft`)
- Promote deltas with sufficient evidence

### 5.3 System Contributors

System contributors (`is_system: true`, e.g., `ruvector-seed`) can:
- All established user capabilities
- Create pages directly as `Canonical` (for seed data)
- Bypass evidence count requirements (seed data is pre-validated)

## 6. Delta-Based Editing Protocol

### 6.1 No Direct Mutation

The Brainpedia has no "edit page" operation. All modifications are deltas. This is enforced at the API level — the `PUT /v1/pages/{id}` endpoint does not exist.

### 6.2 Delta Types

**Correction**: Fixes an error in the canonical page. Requires evidence showing the current content is wrong and the correction is right.

**Extension**: Adds new information to the canonical page. Requires evidence showing the extension is valid and useful.

**Evidence**: Adds a new evidence link to an existing claim. No content change — purely strengthens the evidentiary basis.

**Deprecation**: Marks the canonical page or a specific delta as superseded. Requires evidence showing why the content is no longer valid (e.g., a library version change, a security vulnerability discovered).

### 6.3 Conflict Resolution

When multiple deltas target the same section of a canonical page, conflict resolution uses:

1. **Quality score**: Higher-quality deltas take precedence
2. **Evidence density**: More evidence wins ties
3. **Recency**: Among equal quality and evidence, newer deltas win
4. **Manual resolution**: If automated resolution fails, the page enters `Contested` status and requires a contributor with reputation >= 0.7 to resolve

## 7. First Domain: Code Debugging

The first domain for the Brainpedia is **code debugging** (category: `Debug`). Rationale (same as ADR-061 Section 6.3):

- Highest frequency task in daily development
- Binary success signal (test passes or does not)
- Bounded domain — compiler errors are structured and classifiable
- Rich trajectory data from existing sessions
- Direct impact on developer velocity

### 7.1 Initial Page Structure for Debug Domain

Each debug page covers a specific failure class:

**Title**: Descriptive name for the failure pattern (e.g., "tokio::spawn deadlock with parking_lot::RwLock")

**Content**: Structured fields:
- `error_pattern`: The error message or symptom signature
- `root_cause`: Why this failure occurs
- `fix_pattern`: The canonical fix approach
- `code_before`: Example code that triggers the failure
- `code_after`: Example code after the fix
- `applies_to`: Language, framework, and version constraints

**Evidence**: TestPass links showing the fix resolves the error in specific repositories.

**Transfer assets**: TransferPrior for similar concurrency patterns in other async runtimes.

### 7.2 Seed Pages

The seed pipeline (`ruvector-seed` contributor) will create initial debug pages from:
- Known Rust compiler error patterns
- Common `tokio` / `async-std` pitfalls
- RVF serialization edge cases
- MinCut graph construction errors
- SONA learning convergence failures

## 8. Data Model Extensions

### 8.1 New Types

```rust
/// Page status in the Brainpedia lifecycle
pub enum PageStatus {
    Draft,       // Newly created, awaiting evidence and votes
    Canonical,   // Community-accepted, authoritative
    Contested,   // Conflicting deltas, needs resolution
    Archived,    // Superseded or low-quality, read-only
}

/// A delta entry modifying a canonical page
pub struct PageDelta {
    pub id: Uuid,
    pub page_id: Uuid,                  // The canonical page being modified
    pub delta_type: DeltaType,
    pub content_diff: serde_json::Value, // Structured field updates
    pub evidence_links: Vec<EvidenceLink>,
    pub contributor_id: String,
    pub quality_score: BetaParams,
    pub witness_hash: String,
    pub created_at: DateTime<Utc>,
}

pub enum DeltaType {
    Correction,
    Extension,
    Evidence,
    Deprecation,
}

/// Evidence linking a claim to a verifiable outcome
pub struct EvidenceLink {
    pub evidence_type: EvidenceType,
    pub description: String,
    pub contributor_id: String,
    pub verified: bool,                  // Server-side format check passed
    pub created_at: DateTime<Utc>,
}

pub enum EvidenceType {
    TestPass { test_name: String, repo: String, commit_hash: String },
    BuildSuccess { pipeline_url: String, commit_hash: String },
    MetricImproval { metric_name: String, before: f64, after: f64 },
    PeerReview { reviewer: String, direction: VoteDirection, score: f64 },
}
```

### 8.2 Extended BrainMemory

The existing `BrainMemory` gains optional page metadata:

```rust
// Added to BrainMemory
pub page_status: Option<PageStatus>,
pub evidence_count: u32,
pub delta_count: u32,
```

### 8.3 New API Endpoints

| Method | Path | Purpose | Gate |
|--------|------|---------|------|
| POST | `/v1/pages` | Create a new Draft page | Identity + Reputation >= 0.5 |
| GET | `/v1/pages/{id}` | Get page with delta log | Public read |
| POST | `/v1/pages/{id}/deltas` | Submit a delta | Identity + Evidence |
| GET | `/v1/pages/{id}/deltas` | List deltas for a page | Public read |
| POST | `/v1/pages/{id}/evidence` | Add evidence to a page | Identity |
| POST | `/v1/pages/{id}/promote` | Promote Draft to Canonical | Consensus (auto-checked) |

### 8.4 New MCP Tools

| Tool | Purpose |
|------|---------|
| `brain_page_create` | Create a new knowledge page (Draft) |
| `brain_page_get` | Get a page with its delta log and evidence |
| `brain_page_delta` | Submit a delta to an existing page |
| `brain_page_evidence` | Add evidence to a page or delta |
| `brain_page_search` | Search pages by title, content, or category |
| `brain_page_list` | List pages by category, status, or quality |

## 9. Implementation Status

### Deferred (Future Work)

All components described in this ADR are deferred for future implementation. They build on the existing Shared Brain infrastructure (ADR-059, ADR-060) which is fully implemented.

| Component | Priority | Dependency |
|-----------|----------|------------|
| `PageStatus` enum + `PageDelta` types | High | None (types only) |
| Delta submission endpoint | High | Types |
| Evidence link validation | High | Delta endpoint |
| Page creation with reputation gate | Medium | Reputation system (shipped) |
| Page promotion logic | Medium | Evidence + quality thresholds |
| Conflict resolution | Low | Multiple concurrent deltas |
| Seed debug pages | High | Page creation endpoint |
| MCP tools (6 new) | Medium | All endpoints |

## 10. Acceptance Criteria

### Evidence-Gated Quality

- 50 debug pages seeded from known Rust error patterns
- Each page requires >= 3 evidence links before Canonical promotion
- Pages without evidence remain Draft and are excluded from `brain_search` top-k results
- Quality score distribution: Canonical pages have mean quality >= 0.7

### Delta Integrity

- Full delta log reconstructs any historical page version
- Every delta has a witness chain entry verifiable by any participant
- Deprecation deltas correctly mark superseded content
- No direct mutation path exists (no PUT endpoint on page content)

### Reputation Gating

- New users (reputation < 0.5) cannot create pages (403 response)
- New users can submit deltas with evidence (accepted)
- System contributors can create Canonical pages directly
- Poisoned contributor (reputation < 0.1) cannot submit deltas (403 response)

### First Domain Validation

Run on the Debug category:
- 50 seeded debug pages covering common Rust error patterns
- 10 community-contributed deltas with TestPass evidence
- 5 pages promoted from Draft to Canonical via consensus
- Mean search recall@10 on debug queries >= 40% improvement vs unstructured memories

## 11. Answers to Design Questions

**Q: What is the first domain you want the community Brainpedia to cover?**

A: Code debugging. Same rationale as ADR-061: highest frequency task, binary success signal, bounded domain, richest existing trajectory data. Debug pages have natural evidence links (TestPass, BuildSuccess) that make the evidence gate practical from day one.

**Q: Do you want anyone to be able to submit pages, or only submit deltas to existing pages until they earn reputation?**

A: Deltas only until reputation is earned. New users (composite reputation < 0.5, contribution count < 10) can submit deltas to existing pages but cannot create new pages. This prevents low-reputation flooding while encouraging incremental improvement. Page creation requires demonstrated quality through delta contributions.

## 12. Related ADRs

| ADR | Relationship |
|-----|-------------|
| ADR-057 | Federated RVF Transfer Learning — protocol foundation |
| ADR-058 | Hash Security Optimization — SHAKE-256 for content integrity |
| ADR-059 | Shared Brain Google Cloud — infrastructure and security |
| ADR-060 | Shared Brain Capabilities — sub-capabilities and business outcomes |
| ADR-061 | Reasoning Kernel Architecture — training pipeline that consumes Brainpedia data |
