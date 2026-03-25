# ADR-127: Gist Deep Research Loop — Brain-Guided Discovery Publishing

**Status:** Implemented
**Date:** 2026-03-25
**PR:** #300

## Context

The π Brain's autonomous gist publisher was generating repetitive, low-quality content:
- Every gist was "X shows weak co-occurrence with Y (confidence: 50%)"
- Same 8 generic categories recycled (debug, architecture, geopolitics, pattern, solution, tooling, convention, discovery)
- 45 identical-structure gists published in 48 hours
- Gemini inflated weak signals into long articles with no substance

Root causes:
1. Cross-domain similarity threshold too low (0.3 = nearly every pair matches)
2. Novelty gates trivially easy to pass (2 inferences, 100 evidence)
3. No quality filter on inference content
4. No content-based dedup (only title dedup)
5. Single-pass Gemini rewrite with no external validation

## Decision

### 1. Strict Novelty Gates

Raise all thresholds to require genuinely novel, high-confidence findings:

| Gate | Before | After |
|------|--------|-------|
| MIN_NEW_INFERENCES | 2 | 5 |
| MIN_EVIDENCE | 100 | 500 |
| MIN_STRANGE_LOOP_SCORE | 0.008 | 0.05 |
| MIN_PROPOSITIONS | 5 | 10 |
| MIN_PARETO_GROWTH | 1 | 2 |
| MIN_INFERENCE_CONFIDENCE | — | 0.60 |
| Rate limit | 4 hours | 24 hours |

### 2. Quality Filters

Add `strong_inferences()` and `strong_propositions()` that reject:
- "Weak co-occurrence" language
- Generic cluster IDs as subjects
- Confidence < 60%
- `co_occurs_with` at confidence < 55%

### 3. Source Signal Quality (symbolic.rs)

Raise thresholds at the proposition extraction level:
- Cross-domain similarity: 0.3 → 0.45
- `may_influence`: 0.7 → 0.75
- `associated_with`: 0.5 → 0.55

### 4. Three-Pass Brain-Guided Research Loop

Replace single-pass Gemini rewrite with iterative research:

```
Pass 1: Gemini + Google Search Grounding
  → Research domain topics, find real papers/data (2024-2026)
  → Return structured findings

Pass 2: Brain Memory Search
  → Query pi.ruv.io/v1/memories/search for each topic
  → Get internal context the brain has accumulated

Pass 3: Gemini Synthesis
  → Combine: brain's autonomous findings + grounded research + brain memories
  → Produce article that neither source could create alone
```

The brain guides the research by providing the initial discovery signal (which domains to investigate), and the synthesis loop grounds it in real-world evidence.

### 5. Content Dedup

Replace title-only dedup with `category:dominant_inference` key matching. This prevents publishing "geopolitics associated_with architecture" followed by "architecture associated_with geopolitics".

## Consequences

**Positive:**
- Gists will only publish ~1/day at most, and only when substantive
- Content grounded in real papers and data via Google Search
- Brain memories provide unique internal context
- No more "weak co-occurrence" noise

**Negative:**
- May publish nothing for days if no novel signals emerge (acceptable)
- Three Gemini API calls per publish (cost ~$0.01/gist, negligible)
- Brain memory search adds ~500ms latency (non-blocking, background task)

**Risks:**
- If Gemini grounding returns irrelevant results, the fallback raw format is still used
- Brain memory search requires BRAIN_SYSTEM_KEY env var
