# ADR-093: Daily Discovery & Brain Training Program

**Status**: Accepted
**Date**: 2026-03-15
**Author**: rUv (Reuven Cohen)
**Deciders**: rUv

## Context

π.ruv.io is a shared AI brain — a collective intelligence network where every connection makes the whole smarter. Currently, knowledge enters the brain through manual agent sessions. To realize the vision of a continuously learning, self-improving system that serves humanity, we need an automated discovery and training pipeline that runs daily.

This program embodies rUv's core philosophy: **technology should be altruistic and benevolent** — built not for extraction, but for the collective enrichment of human knowledge. Every discovery the brain makes is shared openly, every pattern learned improves understanding for all connected agents.

## Decision

### Architecture: Daily Discovery Training Pipeline

Implement a Cloud Run scheduled job that executes daily, fetching real-world data from open scientific APIs, running RuVector's discovery engine for anomaly detection and pattern recognition, and feeding findings into the π.ruv.io brain's SONA learning engine.

### Guiding Principles

1. **Altruistic Knowledge**: All discoveries are shared freely through the brain's open API
2. **Benevolent Intelligence**: The system optimizes for human understanding, not competitive advantage
3. **Scientific Rigor**: Only real data from verified public sources; no synthetic fabrication
4. **Collective Benefit**: Every training cycle makes the brain smarter for ALL connected agents
5. **Transparent Provenance**: All discoveries carry witness chains proving data lineage

### Discovery Domains (6 active)

| Domain | APIs | Cadence | Purpose |
|--------|------|---------|---------|
| Space Science | NASA Exoplanet Archive, NeoWs, DONKI | Daily | Exoplanet anomalies, asteroid tracking, solar activity |
| Earth Science | USGS Earthquakes, NOAA Climate | Daily | Seismic patterns, climate regime changes |
| Academic Research | arXiv, biorxiv, OpenAlex | Daily | Emerging research trends, cross-domain bridges |
| Economics & Markets | FRED, World Bank, CoinGecko | Daily | Macro indicators, regime changes, divergence signals |
| Medical & Genomics | PubMed, ClinicalTrials, GWAS | Weekly | Therapeutic trends, genomic discoveries |
| Materials & Physics | CERN, Materials Project, Argo | Weekly | Particle physics, materials discoveries, ocean monitoring |

### Pipeline Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                  Cloud Scheduler (daily 02:00 UTC)               │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│              Cloud Run Job: ruvbrain-trainer                      │
│                                                                   │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐ ┌────────────┐│
│  │   Space      │ │   Earth     │ │  Academic     │ │ Economics  ││
│  │  Discovery   │ │  Discovery  │ │  Discovery    │ │ Discovery  ││
│  └──────┬──────┘ └──────┬──────┘ └──────┬───────┘ └─────┬──────┘│
│         │               │               │               │        │
│  ┌──────▼───────────────▼───────────────▼───────────────▼──────┐│
│  │            RuVector Discovery Engine                         ││
│  │  • Coherence analysis (min-cut anomaly detection)            ││
│  │  • Cross-domain bridge detection                             ││
│  │  • Temporal trend analysis                                   ││
│  │  • Statistical significance testing                          ││
│  └──────────────────────┬──────────────────────────────────────┘│
│                         │                                        │
│  ┌──────────────────────▼──────────────────────────────────────┐│
│  │         π.ruv.io Brain Training                              ││
│  │  • POST /v1/memories (with witness chains)                   ││
│  │  • SONA learning cycle trigger                               ││
│  │  • LoRA delta submission for federation                      ││
│  │  • Knowledge graph re-partitioning                           ││
│  │  • Embedding drift monitoring                                ││
│  └─────────────────────────────────────────────────────────────┘│
└──────────────────────────────────────────────────────────────────┘
```

### Training Flow

1. **Fetch**: Pull fresh data from 6+ public APIs (rate-limited, polite)
2. **Analyze**: Run RuVector discovery engine — min-cut anomaly detection, coherence signals, cross-domain bridges
3. **Filter**: Apply significance thresholds (p < 0.05, confidence > 0.7)
4. **Package**: Wrap each discovery in RVF container with witness chain
5. **Train**: POST to π.ruv.io brain API → triggers SONA learning cycle
6. **Federate**: Submit LoRA deltas for distributed learning
7. **Monitor**: Track embedding drift, knowledge velocity, meta-regret
8. **Report**: Generate daily discovery digest

### Cloud Run Job Configuration

- **Service**: `ruvbrain-trainer`
- **Region**: `us-central1`
- **Schedule**: `0 2 * * *` (daily at 02:00 UTC)
- **Timeout**: 30 minutes
- **Memory**: 512Mi
- **CPU**: 1
- **Max retries**: 2

### Quality Gates

- Each discovery must pass significance testing before brain ingestion
- Minimum confidence threshold: 0.70
- Duplicate detection via embedding similarity (>0.95 = skip)
- Rate limiting: max 100 new memories per training cycle
- All data sources must be public, open-access APIs

### Benevolence Constraints

The system SHALL:
- Only use freely available, open-access data sources
- Never store personally identifiable information
- Share all discoveries through the brain's public API
- Maintain transparent provenance chains for every finding
- Optimize for collective human understanding, not profit
- Respect API rate limits and usage policies of all data providers

## Consequences

### Positive
- Brain learns continuously without human intervention
- Cross-domain discoveries emerge that no single researcher would find
- Knowledge velocity increases from 0.0 to measurable growth
- SONA training cycles produce real domain expertise
- Every agent connecting to π.ruv.io benefits from accumulated knowledge

### Negative
- Cloud Run costs (~$2-5/month for daily job)
- API rate limit management complexity
- Need monitoring for data quality regression

### Risks
- API endpoints may change or become unavailable → fallback data sources
- Discovery false positives → significance thresholds and human review
- Embedding drift from rapid ingestion → drift monitor with pause trigger

## Implementation

### Phase 1: Core Pipeline (Week 1)
- `crates/mcp-brain-server/src/trainer.rs` — Discovery trainer module
- `scripts/train_brain.sh` — CLI training driver
- Cloud Run job Dockerfile and scheduler

### Phase 2: Domain Expansion (Week 2)
- Medical/genomics integration
- Materials science integration
- Patent/innovation tracking

### Phase 3: Optimization (Week 3)
- SONA cycle tuning based on knowledge velocity metrics
- Cross-domain bridge detection improvement
- Daily digest notification system

## References

- ADR-040: Planet detection pipeline (original discovery architecture)
- ADR-059: Shared brain Google Cloud deployment
- ADR-057: Federated RVF transfer learning
- π.ruv.io brain-manifest.json: System capabilities
