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
