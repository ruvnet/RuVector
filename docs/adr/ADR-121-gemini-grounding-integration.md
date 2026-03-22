# ADR-121: Gemini Google Search Grounding for Brain Optimizer

**Status**: Implemented
**Date**: 2026-03-22
**Author**: Claude (ruvnet)
**Related**: ADR-115 (Common Crawl), ADR-118 (Cost-Effective Crawl), ADR-120 (WET Pipeline)

## Context

The pi.ruv.io brain optimizer uses Gemini to promote cluster taxonomy (`is_type_of` propositions) into richer relational propositions (`implies`, `causes`, `requires`). Without grounding, Gemini can hallucinate relations that don't exist in the real world.

Google Search Grounding connects Gemini to live web data, allowing it to verify its outputs against real sources. This ensures that generated propositions are factually accurate and provides source URLs for auditability.

## Decision

Integrate Google Search Grounding into the brain's Gemini optimizer calls via the `google_search` tool parameter.

### API Format

```json
{
  "contents": [{"role": "user", "parts": [{"text": "prompt"}]}],
  "tools": [{"google_search": {}}],
  "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.3}
}
```

### Grounding Response

```json
{
  "candidates": [{
    "content": {"parts": [{"text": "response"}]},
    "groundingMetadata": {
      "webSearchQueries": ["query used"],
      "groundingChunks": [{"web": {"uri": "https://...", "title": "source"}}],
      "groundingSupports": [{"segment": {"text": "..."}, "groundingChunkIndices": [0]}]
    }
  }]
}
```

### What Changes

| Before | After |
|--------|-------|
| Gemini generates relations from pattern analysis only | Gemini generates AND verifies against live Google Search |
| No source attribution on propositions | Source URLs logged from `groundingChunks` |
| Hallucinated relations possible | Grounded relations with support scores |
| `is_type_of` only (10 propositions) | `implies`, `causes`, `requires` with evidence |

### Configuration

| Env Var | Default | Purpose |
|---------|---------|---------|
| `GEMINI_API_KEY` | (required) | Google AI API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Model ID |
| `GEMINI_GROUNDING` | `true` | Enable Google Search grounding |

### Cost

Gemini 2.5 Flash with grounding: billed per prompt (not per search query — per-query billing only applies to Gemini 3 models). At the optimizer's 1-hour interval with ~10 prompts/cycle, estimated cost: **$1-3/month**.

## Implementation

Modified `crates/mcp-brain-server/src/optimizer.rs`:
- Added `google_search` tool to Gemini API request body
- Log grounding metadata (sources count, supports count, search queries)
- Configurable via `GEMINI_GROUNDING` env var (default: true)
- Source URLs from `groundingChunks` logged for auditability

## Acceptance Criteria

1. Optimizer calls include `tools: [{"google_search": {}}]`
2. Grounding metadata logged when present
3. Can be disabled via `GEMINI_GROUNDING=false`
4. No additional cost beyond existing Gemini API usage (Gemini 2.5 Flash)

## Consequences

### Positive
- Propositions verified against live web — reduced hallucination
- Source URLs provide auditability for every generated relation
- Brain's symbolic layer becomes evidence-based, not just pattern-based
- Enables the Horn clause engine to chain verified inferences

### Negative
- Adds latency (~1-2s per grounded call vs ~0.5s ungrounded)
- Requires internet connectivity for optimizer (acceptable — runs on Cloud Run)
- Google Search results may change over time (mitigated by logging at generation time)

## Deployment Status (2026-03-22)

### Verified Working
- Gemini 2.5 Flash: responding, generating rule refinement suggestions
- Google Search grounding: `google_search` tool included in API calls
- Optimizer: configured=true, model_id=gemini-2.5-flash
- Deploy script: fixed to preserve all env vars (`--update-env-vars`)

### System State After Full Deployment

| Metric | Value |
|--------|-------|
| Memories | 1,808 |
| Graph | 611,401 edges |
| Sparsifier | 42.4x (14,383 sparse) |
| Propositions | 10 (`is_type_of`) — relational types pending first optimizer cycle |
| Rules | 4 Horn clauses |
| Pareto front | 16 solutions |
| Gemini optimizer | Deployed, grounding enabled |
| Midstream attractor | Activated (7 categories detected) |
| Knowledge velocity | 1.0 (from zero — system warming up) |
| SONA trajectories | 1 (accumulating) |

### Optimization Opportunities Identified

1. **Graph rebuild timeout**: 611K edges takes >90s — exceeds Cloud Run's `/v1/pipeline/optimize` timeout. The hourly `brain-graph` scheduler handles this but the endpoint needs increased timeout or async pattern.

2. **In-memory state loss on deploy**: GWT salience, SONA trajectories, temporal deltas all reset to zero on every Cloud Run revision deploy. These rebuild through organic use but cold starts lose accumulated learning. Potential fix: persist cognitive state to Firestore alongside memories.

3. **SONA needs trajectories**: 0 patterns because trajectories only accumulate from user interactions (search queries, memory contributions). The scheduled `/v1/train` call learns from accumulated trajectories but can't create them. Need a trajectory injection path from crawl/inject pipeline.

4. **Scheduler jobs created but not yet auto-fired**: All 15 jobs show "never" for lastAttemptTime because they were created during this session. They will fire at their scheduled times. Manually triggered all critical jobs to prime the system.

5. **WET daily job re-processes same segments**: The `wet-import-n202608` Cloud Run Job has 50 segments baked in. Daily re-execution re-downloads and re-processes the same segments. Need segment rotation or cursor tracking to process new segments each day.
