# rvAgent + Gemini Grounding: Autonomous Knowledge Enhancement for pi.ruv.io

## Overview

This research document describes a system where rvagent autonomously uses Google Gemini with Search Grounding to verify, enrich, and extend the pi.ruv.io brain's knowledge graph. The system operates in four phases, each building on the previous, transforming the brain from a passive store of `is_type_of` propositions into an active, self-correcting knowledge engine with relational reasoning.

## Problem Statement

The pi.ruv.io brain currently has:
- 1,809 memories across medical, CS, and physics domains
- 612K graph edges with 42.2x sparsifier compression
- 11 propositions, **all `is_type_of`** -- no relational predicates
- 4 Horn clause rules that cannot fire without relational input
- 0 SONA patterns (no trajectory data to learn from)
- A Gemini optimizer that suggests improvements but does not act on them

The Horn clause engine has rules for transitivity of `causes`, `solves`, `relates_to`, and `similar_to`, but no propositions of those types exist. The inference engine is structurally complete but starved of data.

## Solution: Four-Phase Autonomous Agent System

| Phase | Agent | Purpose | Prerequisite |
|-------|-------|---------|-------------|
| 1 | Fact Verifier | Ground-truth check existing memories via Google Search | None |
| 2 | Relation Generator | Create `implies`, `causes`, `requires` propositions | Phase 1 tagging |
| 3 | Cross-Domain Explorer | Find unexpected bridges between domains | Phase 2 relations |
| 4 | Research Director | Autonomously research drifting topics | Phase 3 bridges |

## Key Design Decisions

1. **All agents operate through existing rvagent MCP tools** -- no new tool types needed
2. **All brain mutations go through the pi.ruv.io REST API** -- no direct database access
3. **Gemini calls use Search Grounding** (ADR-121) -- every generated fact has source URLs
4. **HIPAA-safe**: Memory content is summarized before sending to Gemini; no raw PHI
5. **Budget**: < $50/month at Gemini 2.5 Flash pricing ($0.15/1M input tokens)
6. **Execution**: Cloud Scheduler triggers Cloud Run Jobs, each running one agent cycle

## Documents

| File | Contents |
|------|----------|
| [architecture.md](./architecture.md) | Detailed system design, data flows, API interactions |
| [implementation-plan.md](./implementation-plan.md) | Step-by-step build plan with code outlines and file locations |
| [ADR-122](../../adr/ADR-122-rvagent-gemini-grounding-agents.md) | Architecture Decision Record |

## Cost Estimate

| Component | Monthly Volume | Unit Cost | Monthly Cost |
|-----------|---------------|-----------|-------------|
| Gemini 2.5 Flash (input) | ~15M tokens | $0.15/1M | $2.25 |
| Gemini 2.5 Flash (output) | ~3M tokens | $0.60/1M | $1.80 |
| Google Search Grounding | ~500 grounded calls | included* | $0.00 |
| Cloud Run Jobs | ~120 job-minutes | $0.00002/vCPU-s | $0.14 |
| Cloud Scheduler | 4 jobs | free tier | $0.00 |
| **Total** | | | **~$4.19** |

*Grounding is included in Gemini 2.5 Flash pricing; per-query billing only applies to Gemini 3+ models.

## Current System Integration Points

```
rvagent MCP server (npm/packages/ruvector/bin/mcp-server.js)
  |
  +-- brain_search  --> GET  pi.ruv.io/v1/memories/search
  +-- brain_share   --> POST pi.ruv.io/v1/memories
  +-- brain_list    --> GET  pi.ruv.io/v1/memories/list
  +-- brain_status  --> GET  pi.ruv.io/v1/status
  +-- brain_drift   --> GET  pi.ruv.io/v1/drift
  |
  +-- hooks_learn / hooks_recall / hooks_remember (local intelligence)
  +-- hooks_trajectory_begin / step / end (SONA)

pi.ruv.io Brain API (crates/mcp-brain-server)
  |
  +-- POST /v1/ground        --> inject new propositions
  +-- GET  /v1/propositions  --> list existing propositions
  +-- POST /v1/reason        --> run Horn clause inference
  +-- POST /v1/train         --> trigger SONA + domain training
  +-- POST /v1/optimize      --> trigger Gemini optimizer
  +-- GET  /v1/drift         --> knowledge drift detection

Gemini API (generativelanguage.googleapis.com)
  |
  +-- generateContent with tools: [{"google_search": {}}]
  +-- Returns groundingMetadata: chunks, supports, searchQueries
```
