# ADR-122: rvAgent Autonomous Gemini Grounding Agents

**Status**: Approved with Revisions
**Date**: 2026-03-23
**Author**: Claude (ruvnet)
**Related**: ADR-121 (Gemini Grounding), ADR-112 (rvAgent MCP), ADR-110 (Neural-Symbolic), ADR-115 (Common Crawl)

## Context

The pi.ruv.io brain has accumulated 1,809 memories with 612K graph edges and a working Gemini optimizer with Google Search grounding (ADR-121). However, the symbolic reasoning layer is underutilized:

- All 11 propositions are `is_type_of` -- no relational predicates
- 4 Horn clause rules exist for transitivity of `causes`, `solves`, `relates_to`, `similar_to` but cannot fire
- SONA has 0 learned patterns (insufficient trajectory data)
- No mechanism to verify whether stored knowledge is factually current
- No cross-domain connection discovery

The existing Gemini optimizer (ADR-121) generates suggestions but does not act on them. The rvagent MCP server has brain integration tools but no autonomous execution loop.

## Decision

Implement four autonomous agents that run as Cloud Run Jobs on Cloud Scheduler, using Gemini 2.5 Flash with Google Search grounding to verify, enrich, and extend brain knowledge:

| Agent | Function | Schedule |
|-------|----------|----------|
| **Fact Verifier** (Phase 1) | Verify memory claims against live web sources | Every 6 hours |
| **Relation Generator** (Phase 2) | Create relational propositions between verified memories | Daily 02:00 UTC |
| **Cross-Domain Explorer** (Phase 3) | Find unexpected connections across domains | Daily 06:00 UTC |
| **Research Director** (Phase 4) | Research high-drift topics, inject findings, generate SONA trajectories | Every 12 hours |

### Architecture

```
Cloud Scheduler --> Cloud Run Job (node agent-runner.js --phase=N)
                      |
                      +-- Gemini 2.5 Flash (with grounding)
                      +-- pi.ruv.io Brain API (memories, propositions, inference)
```

Each agent:
1. Reads from the brain via its REST API
2. Sends sanitized content to Gemini with `tools: [{"google_search": {}}]`
3. Parses grounding metadata (source URLs, support scores)
4. Writes to a **candidate layer** (NOT the canonical graph directly)
5. Candidates promoted only after passing promotion gates
6. Logs structured provenance for every mutation

### Write Safety: Candidate vs Canonical Graph

All machine-generated relations and discoveries are first written to a candidate graph with full provenance, grounding metadata, and replayable mutation logs. Promotion into the canonical graph requires policy evaluation:

| Promotion Gate | Threshold |
|---------------|-----------|
| Source count minimum | ≥ 2 grounding sources |
| Source diversity | ≥ 2 distinct domains |
| Recency window | Sources from last 12 months |
| Contradiction scan | No conflicts with existing canonical propositions |
| Predicate allowlist | Only `causes`, `treats`, `depends_on`, `part_of`, `measured_by` |
| Confidence threshold | ≥ 0.7 for `causes`/`depends_on`, ≥ 0.8 for `treats` |

### Truth Maintenance State Machine

Every proposition follows this lifecycle:

```
unverified → grounded_supported → promoted (canonical)
          → grounded_conflicted → deprecated
          → stale (source aged out)
          → candidate_only (never promoted)
```

Drift trigger formula:
```
drift_score = recency_decay + contradiction_weight + source_disagreement + novelty_gap
```
Phase 4 triggers only when `drift_score > threshold`, not on a timer.

### Key Design Choices

**1. Standalone scripts, not library code**
Agents live in `scripts/rvagent-grounding/` as plain Node.js, not integrated into `npm/packages/ruvector/`. This keeps the rvagent library clean and makes Cloud Run Job deployment straightforward.

**2. Allowlist-based PHI sanitization (not regex-only)**
Instead of stripping bad text from raw content, agents send a normalized record:
```json
{
  "memory_id": "m123",
  "claim": "X is associated with Y under condition Z",
  "domain": "biomedical",
  "created_at_bucket": "2026-Q1",
  "evidence_type": "article_claim"
}
```
Pipeline: entity detector → pattern detector → allowlist serializer → audit log with redaction hash → periodic canary tests with seeded PHI strings.

**3. Verification-first pipeline**
Phase 2 only generates relations for Phase 1-verified memories. This prevents the Horn clause engine from chaining inferences based on potentially incorrect facts.

**4. Token budget enforcement**
Each agent cycle has a configurable token budget (default: 50K tokens/cycle). Agents stop processing when the budget is exhausted, preventing cost overruns.

**5. GOAP methodology**
Each agent defines preconditions, effects, and costs following Goal-Oriented Action Planning:

| Action | Precondition | Effect | Cost (tokens) |
|--------|-------------|--------|---------------|
| verify_memory | memory.unverified | memory.grounding_status set | ~500 |
| generate_relation | both memories verified, predicate in allowlist | new candidate proposition | ~800 |
| discover_bridge | relations exist, 2+ domains, bridge concept identified | cross-domain candidate | ~1,200 |
| research_drift | drift velocity > 2.0 | new findings, SONA trajectory | ~1,500 |

## Consequences

### Positive

- **Horn clause engine activates**: Relational propositions enable inference chains (`A causes B, B causes C` -> `A causes C`)
- **Self-correcting knowledge**: Contradictions detected by Phase 1 are researched and corrected by Phase 4
- **Cross-domain insight**: Automated discovery of connections humans would not typically seek
- **SONA bootstrapping**: Phase 4 generates real trajectories, enabling pattern learning
- **Evidence-based knowledge**: Every generated proposition has grounding source URLs
- **Low cost**: Estimated $4-5/month at current pricing

### Negative

- **Latency**: Grounded Gemini calls take 2-5 seconds each (acceptable for batch processing)
- **Internet dependency**: Agents require internet access for Gemini API and Google Search
- **Semantic drift risk**: Automated proposition injection could accumulate errors without human review
- **PHI detection imperfect**: Regex-based PHI detection may miss edge cases

### Mitigations

- **Confidence thresholds**: Relations below 0.5 confidence are not injected
- **Contradiction loop**: Phase 1 detects contradictions; Phase 4 investigates them
- **Monthly human review**: Flag propositions with confidence < 0.6 for manual review
- **Budget cap**: Hard token limit prevents runaway costs
- **Dry-run mode**: `DRY_RUN=true` logs actions without mutating brain state

## Implementation

Detailed implementation plan: [docs/research/rvagent-gemini-grounding/implementation-plan.md](../research/rvagent-gemini-grounding/implementation-plan.md)

### Files to Create

```
scripts/rvagent-grounding/
  agent-runner.js           -- Entry point
  lib/gemini-client.js      -- Gemini API with grounding
  lib/brain-client.js       -- pi.ruv.io REST client
  lib/phi-detector.js       -- PHI sanitization
  phases/verify.js          -- Phase 1
  phases/relate.js          -- Phase 2
  phases/explore.js         -- Phase 3
  phases/research.js        -- Phase 4
  package.json
  Dockerfile
```

### Files to Modify

- `crates/mcp-brain-server/src/routes.rs`: Add batch proposition injection endpoint
- `npm/packages/ruvector/bin/mcp-server.js`: Add `brain_ground` and `brain_reason` MCP tools

### Cloud Resources

- 4 Cloud Run Jobs (512Mi, 1 vCPU each)
- 4 Cloud Scheduler jobs
- 2 secrets (GEMINI_API_KEY, PI token)

## Cost Estimate

`monthly_cost = input_tokens × input_rate + output_tokens × output_rate + grounded_requests × grounding_rate + Cloud Run`

| Scenario | Tokens | Grounded Requests | Compute | Total |
|----------|--------|-------------------|---------|-------|
| **Base** (10 memories/cycle) | $2.00 | $1.50 | $0.14 | **$3.64** |
| **Expected** (20 memories/cycle) | $4.00 | $3.00 | $0.28 | **$7.28** |
| **Worst credible** (50 memories/cycle) | $10.00 | $7.50 | $0.70 | **$18.20** |

Key metric: **cost per promoted proposition** — target ≤ $0.02.

Budget cap: $50/month. Track: tokens per successful mutation, grounded requests per accepted proposition.

## Acceptance Criteria

### Volume
1. Phase 1: 80%+ of quality ≥ 3 memories have grounding status after 7 days
2. Phase 2: ≥ 50 candidate relational propositions after 2 weeks
3. Phase 3: ≥ 10 cross-domain discoveries after 3 weeks
4. Phase 4: SONA patterns > 0 after 4 weeks

### Precision
5. **Grounding precision**: 100-item human audit — ≥ 90% of `grounded_supported` correctly labeled
6. **Relation precision**: 50-item audit — ≥ 80% of promoted relations judged useful and correct
7. **Inference utility**: Horn clause produces ≥ 20 non-trivial inferences with < 10% human-judged error

### Safety
8. **Write safety**: Zero direct writes to canonical graph without promotion gate
9. **Privacy**: Zero PHI findings in outbound audit sample
10. **Economics**: Cost per promoted proposition ≤ $0.02

### Provenance
11. Every promoted proposition replayable from: raw source → prompt version → grounding evidence → policy decision → mutation log

## Provenance Schema

Every grounded mutation stores:

```typescript
interface GroundedMutation {
  mutation_id: string;          // UUID
  source_memory_ids: string[];  // Input memories
  source_urls: string[];        // Grounding chunk URLs
  retrieval_timestamp: string;  // When grounding sources were fetched
  claim_text: string;           // Sanitized claim sent to Gemini
  support_snippets: string[];   // Grounding support text segments
  model_id: string;             // e.g., "gemini-2.5-flash"
  prompt_template_version: string;
  confidence: number;
  mutation_decision: 'promote' | 'candidate_only' | 'reject';
  prior_state: string | null;   // Previous proposition if updating
  new_state: string;            // New proposition
  promotion_gates_passed: string[]; // Which gates cleared
  promotion_gates_failed: string[]; // Which gates blocked (if candidate_only)
}
```

**Acceptance test**: if you can replay any promoted proposition from raw source, prompt version, grounding evidence, policy decision, and final mutation log, the architecture is strong enough to trust.

## References

- [Gemini Grounding docs](https://ai.google.dev/gemini-api/docs/grounding)
- [ADR-121: Gemini Search Grounding](./ADR-121-gemini-grounding-integration.md)
- [ADR-110: Neural-Symbolic Internal Voice](./ADR-110-neural-symbolic-internal-voice.md)
- [ADR-112: rvAgent MCP Server](./ADR-112-rvagent-mcp-server.md)
- [Research: architecture.md](../research/rvagent-gemini-grounding/architecture.md)
