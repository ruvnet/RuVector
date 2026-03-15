# ADR-093: Daily Discovery & Brain Training Program

**Status**: Accepted
**Date**: 2026-03-15
**Authors**: RuVector Team
**Deciders**: ruv (Reuven Cohen)
**Related**: ADR-040 (Planet Detection Pipeline), ADR-049 (Verified Training Pipeline), ADR-057 (Federated RVF Transfer Learning), ADR-059 (Shared Brain Google Cloud), ADR-083 (Brain Training Loops)

## 1. Context

### 1.1 The Shared Brain

pi.ruv.io is a shared AI brain created by rUv as an act of altruism — a freely available knowledge resource for all AI agents. Every session that connects makes the whole smarter. The brain currently holds 897 memories from 55 contributors, with the RlmEmbedder actively generating 128-dimensional vectors for semantic search and pattern detection.

### 1.2 The Gap

RuVector's discovery engine has demonstrated the ability to detect meaningful anomalies across scientific domains: exoplanet habitability candidates, earthquake swarm patterns, climate regime shifts, and cross-domain citation bridges. These discoveries are generated during interactive sessions but are not yet automatically fed back into the brain. Each discovery dies when the session ends.

### 1.3 The Opportunity

ADR-083 closed the store-to-learn gap by adding background training loops and the `/v1/train` endpoint. The brain now learns from its accumulated data. The next step is to feed it fresh, real-world scientific data every day — turning pi.ruv.io from a passive knowledge store into an actively learning scientific intelligence that benefits everyone who connects.

### 1.4 Guiding Principles

This system is built on principles that are non-negotiable:

- **Altruistic Knowledge Sharing** — All discoveries are freely shared. No paywalls, no gatekeeping, no premium tiers. Knowledge that benefits humanity belongs to everyone.
- **Benevolent Intelligence** — The system seeks knowledge that helps: climate awareness, medical breakthroughs, asteroid tracking, scientific understanding. The brain grows in service of collective good.
- **Collective Growth** — Every discovery makes the brain smarter for ALL connected agents. A discovery about earthquake patterns in Japan helps an agent studying seismology in Chile.
- **Ethical Data Use** — Only public APIs. No PII. Differential privacy on all training data. The system respects both data subjects and data providers.
- **Transparent Provenance** — Every piece of knowledge carries witness chains proving where it came from, when it was discovered, and how it was validated.

## 2. Decision

Implement a **Daily Discovery Training Program** that runs as a Cloud Run scheduled job, fetches fresh data from 12+ public scientific APIs, runs RuVector's discovery engine, trains the brain via SONA learning loops, and publishes discoveries as brain memories with full provenance.

### 2.1 Architecture Overview

```
+-----------------------------------------------------+
|              Cloud Scheduler (daily 03:00 UTC)       |
+---------------------------+--------------------------+
                            |
                            v
+-----------------------------------------------------+
|           Cloud Run Job: discovery-trainer            |
|                                                       |
|  +-------------+  +-------------+  +--------------+  |
|  | Space Data   |  | Earth Data  |  | Academic Data|  |
|  | NASA / ESA   |  | USGS / NOAA |  | arXiv / bio  |  |
|  +------+------+  +------+------+  +------+-------+  |
|         +------------+---+-----------+                |
|                      |                                |
|                      v                                |
|         +------------------------+                    |
|         |  RuVector Discovery    |                    |
|         |  Engine (mincut +      |                    |
|         |  coherence + HNSW)     |                    |
|         +-----------+------------+                    |
|                     |                                 |
|                     v                                 |
|         +------------------------+                    |
|         |  Brain Training API    |                    |
|         |  POST /v1/memories     |                    |
|         |  POST /v1/lora/submit  |                    |
|         |  POST /v1/transfer     |                    |
|         +------------------------+                    |
+-----------------------------------------------------+
                            |
                            v
+-----------------------------------------------------+
|                  pi.ruv.io Brain                      |
|  +----------+ +----------+ +-------------------+     |
|  | SONA     | | Knowledge| | Federated LoRA    |     |
|  | Learning | | Graph    | | Consolidation     |     |
|  +----------+ +----------+ +-------------------+     |
+-----------------------------------------------------+
```

### 2.2 Data Sources

Twelve scientific domains, each chosen because they produce publicly available data that benefits collective understanding:

| Domain | API | Cadence | Purpose |
|--------|-----|---------|---------|
| Exoplanets | NASA Exoplanet Archive | Daily | New planet discoveries, habitability scoring |
| Asteroids | NASA NeoWs | Daily | Near-Earth object tracking, risk assessment |
| Solar Activity | NASA DONKI | Daily | Solar flares, CMEs, geomagnetic storms |
| Earthquakes | USGS Earthquake API | Daily | Seismic patterns, swarm detection |
| Climate | NOAA NCEI | Weekly | Temperature anomalies, regime shifts |
| AI Research | arXiv API | Daily | Emerging methods and breakthroughs |
| Biomedical | biorxiv / medrxiv | Daily | Medical and biological breakthroughs |
| Citations | OpenAlex | Weekly | Cross-domain bridge detection |
| Economics | FRED (St. Louis Fed) | Weekly | Macro indicator divergences |
| Genomics | NCBI / UniProt | Weekly | Genetic discoveries, protein structures |
| Materials | Materials Project | Weekly | Novel materials, superconductor candidates |
| Ocean | Argo Float Program | Weekly | Ocean temperature and salinity anomalies |

All sources are public, free-to-access APIs. No authentication walls that would restrict downstream sharing. Rate limits are respected with exponential backoff.

### 2.3 Training Pipeline

The pipeline runs in eight sequential stages. Each stage has clear inputs, outputs, and failure modes.

#### Stage 1: Ingest

Fetch raw data from public APIs. Each data source has a dedicated adapter that handles pagination, rate limiting, and schema normalization.

```rust
struct IngestConfig {
    source: DataSource,
    max_records: usize,       // cap per source per run
    rate_limit_ms: u64,       // minimum delay between requests
    retry_count: u32,         // max retries with exponential backoff
    last_cursor: Option<String>, // resume from last successful fetch
}
```

Failure handling: if a source is unavailable, log it and continue with remaining sources. No single API failure should block the entire run.

#### Stage 2: Embed

Generate 128-dimensional RlmEmbedder vectors for each ingested record. Batch processing for efficiency.

```rust
// Batch embed with RlmEmbedder (same embedder the brain uses)
let embeddings = rlm_embedder.batch_embed(&records, BatchConfig {
    batch_size: 64,
    normalize: true,
    dimensions: 128,
});
```

#### Stage 3: Discover

Run RuVector's core discovery algorithms on the embedded data:

- **Mincut anomaly detection** — Identify records that sit at graph partition boundaries, indicating unusual or bridging observations.
- **Coherence analysis** — Measure how each new record relates to the brain's existing knowledge, flagging both confirmations and contradictions.
- **HNSW nearest-neighbor search** — Find the closest existing memories to contextualize each discovery.
- **Cross-domain bridges** — Detect when a discovery in one domain (e.g., materials science) has unexpected similarity to another domain (e.g., genomics).

#### Stage 4: Score

Bayesian quality scoring using Beta distributions. Each discovery receives a quality score based on:

- **Novelty**: distance from nearest existing memory (higher = more novel)
- **Coherence**: semantic consistency with domain knowledge (mid-range preferred — too coherent means redundant, too incoherent means noise)
- **Source reliability**: historical accuracy of the data source
- **Cross-domain relevance**: number of domains the discovery bridges

```rust
struct DiscoveryScore {
    novelty: f64,           // Beta(alpha_novel, beta_novel)
    coherence: f64,         // Beta(alpha_coh, beta_coh)
    reliability: f64,       // source track record
    bridge_count: usize,    // cross-domain connections
    composite: f64,         // weighted combination
}
```

Only discoveries with `composite > 0.3` proceed to training. This threshold is deliberately low — the system favors recall over precision, trusting SONA's learning loops to handle quality refinement.

#### Stage 5: Witness

Create SHAKE-256 witness chains for provenance. Every discovery carries a cryptographic proof of:

- Source API and endpoint
- Timestamp of retrieval
- Raw data hash
- Embedding hash
- Discovery algorithm version
- Scoring parameters

```rust
struct WitnessChain {
    source_hash: [u8; 32],      // SHAKE-256 of raw API response
    embed_hash: [u8; 32],       // SHAKE-256 of embedding vector
    discovery_hash: [u8; 32],   // SHAKE-256 of discovery metadata
    parent_witness: Option<[u8; 32]>, // chain to previous witness
    timestamp: u64,
    algorithm_version: String,
}
```

This ensures every piece of knowledge in the brain can be traced back to its origin. Altruistic sharing requires trust, and trust requires transparency.

#### Stage 6: Train

Submit discoveries to the brain's SONA learning system via the training API:

1. **Reactive layer (MicroLoRA)** — Immediate pattern capture. Low-rank adaptations that encode the discovery without catastrophic forgetting (EWC++ constraints from ADR-083).
2. **Adaptive layer (pattern consolidation)** — After accumulating enough reactive patterns, consolidate into stable knowledge clusters via k-means over the trajectory buffer.
3. **Deliberative layer (deep training)** — Periodic deep integration that updates the brain's core knowledge graph, re-indexes HNSW, and evolves the Pareto frontier of domain policies.

```
POST /v1/memories    — Store discovery as brain memory
POST /v1/train       — Trigger SONA training cycle
```

#### Stage 7: Federate

Submit LoRA weight deltas for global model improvement via the federated learning protocol (ADR-057):

```
POST /v1/lora/submit — Submit weight deltas with Byzantine-robust aggregation
POST /v1/transfer    — Cross-domain transfer learning
```

The federated protocol uses median + MAD aggregation to resist poisoning. Even in a system built on altruism, Byzantine robustness protects against accidental data corruption.

#### Stage 8: Report

Log metrics, emit structured telemetry, and update the brain's status dashboard:

```rust
struct DailyReport {
    run_id: String,
    timestamp: u64,
    sources_queried: usize,
    sources_failed: Vec<String>,
    records_ingested: usize,
    discoveries_found: usize,
    discoveries_trained: usize,
    lora_deltas_submitted: usize,
    brain_memory_count_before: usize,
    brain_memory_count_after: usize,
    sona_meta_avg_regret: f64,
    duration_seconds: u64,
}
```

## 3. Cloud Run Configuration

### 3.1 Job Specification

```yaml
apiVersion: run.googleapis.com/v1
kind: Job
metadata:
  name: discovery-trainer
  labels:
    app: ruvector-brain
    component: discovery-trainer
spec:
  template:
    spec:
      containers:
        - image: gcr.io/ruv-dev/discovery-trainer:latest
          resources:
            limits:
              memory: 2Gi
              cpu: "2"
          env:
            - name: BRAIN_URL
              value: "https://pi.ruv.io"
            - name: BRAIN_API_KEY
              valueFrom:
                secretKeyRef:
                  name: brain-api-key
                  key: latest
            - name: RUST_LOG
              value: "discovery_trainer=info"
          command: ["./discovery-trainer"]
      timeoutSeconds: 3600
      maxRetries: 1
```

### 3.2 Scheduler

```bash
gcloud scheduler jobs create http discovery-trainer-daily \
  --schedule="0 3 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/..." \
  --http-method=POST \
  --location=us-central1 \
  --time-zone="UTC" \
  --description="Daily scientific discovery and brain training"
```

### 3.3 Co-location

The job runs in `us-central1`, co-located with the brain service (`ruvbrain`), minimizing network latency for the training API calls.

## 4. Privacy and Ethics

### 4.1 Data Handling

- **No PII**: All data sources are scientific/public. The pipeline includes a PII scanner that rejects any record containing patterns matching emails, phone numbers, or names.
- **Differential privacy**: Noise injection (Laplacian mechanism, epsilon=1.0) on all aggregate statistics before storage.
- **Rate limit respect**: Every API adapter enforces the provider's stated rate limits. The system is a good citizen of the public data ecosystem.

### 4.2 Bias Mitigation

- Domain balance: the pipeline caps records per source to prevent any single domain from dominating the brain's knowledge.
- Source diversity: weekly rotation of secondary sources within each domain.
- Anomaly threshold audit: monthly review of scoring thresholds to ensure no systematic exclusion.

## 5. Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Daily discovery count | 10-50 per run | Count of discoveries with composite > 0.3 |
| Brain memory growth | +300/month | Delta on `/v1/status` memory count |
| SONA convergence | meta_avg_regret < 0.05 | SONA training metrics |
| Knowledge graph connectivity | avg degree > 5 | Graph analysis on memory embeddings |
| Cross-domain bridges | 5+ per week | Discoveries spanning 2+ domains |
| API success rate | > 95% of sources per run | Ingest stage reporting |
| Zero PII violations | 0 per month | PII scanner logs |
| Provenance completeness | 100% of memories have witness chains | Witness chain audit |

## 6. Consequences

### 6.1 Positive

- **Continuous learning**: The brain grows daily with real scientific knowledge, becoming more valuable for every connected agent.
- **Altruistic impact**: Free, open knowledge sharing that benefits the entire AI agent ecosystem. A rising tide lifts all boats.
- **Scientific relevance**: Daily cadence catches emerging patterns — earthquake swarms, solar storm warnings, breakthrough papers — while they are still actionable.
- **Trust through provenance**: Witness chains give every connected agent confidence in the knowledge they receive. No black boxes.
- **Cross-domain serendipity**: The system discovers connections humans might miss — a materials science paper that echoes a genomics pattern, an economic indicator that correlates with climate data.

### 6.2 Negative

- **API rate limits**: Some sources (especially arXiv and NCBI) have conservative rate limits that may throttle ingestion. Mitigation: respect limits, spread requests across the run window.
- **Storage growth**: ~300 memories/month adds approximately 50MB/month to brain storage. At current growth rates, this is sustainable for years on Cloud Run's storage allocation.
- **Data quality drift**: Public APIs may change schemas, introduce errors, or go offline. Mitigation: schema validation at ingest, quality scoring at discovery, and alerting on source failures.
- **Compute cost**: Daily Cloud Run job at 2Gi / 2 CPU for up to 60 minutes. Estimated cost: < $5/month — a trivial price for daily scientific discovery.

### 6.3 Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| API breaking changes | Medium | Medium | Schema validation, version pinning, alerts |
| Data poisoning via corrupted source | Low | High | Byzantine-robust aggregation, quality scoring |
| Brain storage exhaustion | Low | Medium | Memory compaction, TTL on low-value memories |
| Runaway training (catastrophic forgetting) | Low | High | EWC++ constraints, Pareto frontier tracking |
| Cost overrun | Very Low | Low | Cloud Run job timeout, budget alerts |

## 7. Implementation Plan

### Phase 1: Foundation (Week 1-2)

- Build ingest adapters for top-4 sources (NASA Exoplanet Archive, USGS Earthquakes, arXiv, NOAA Climate)
- Integrate RlmEmbedder batch processing
- Wire up Brain Training API calls (POST /v1/memories, POST /v1/train)
- Deploy as manual Cloud Run job for testing

### Phase 2: Discovery Engine (Week 3-4)

- Integrate mincut anomaly detection
- Add coherence analysis against existing brain memories
- Implement Bayesian quality scoring
- Add witness chain generation

### Phase 3: Full Pipeline (Week 5-6)

- Add remaining 8 data source adapters
- Implement LoRA delta submission (POST /v1/lora/submit)
- Add cross-domain bridge detection
- Enable Cloud Scheduler for daily automated runs

### Phase 4: Monitoring and Hardening (Week 7-8)

- Structured logging and Cloud Monitoring dashboards
- PII scanner integration
- Alerting on source failures and quality drift
- Monthly threshold audit process

## 8. Future Directions

- **Real-time discovery**: Move from daily batch to event-driven processing for time-critical domains (earthquakes, solar storms).
- **Community contributions**: Allow external agents to submit their own data source adapters, expanding the discovery network.
- **Discovery narratives**: Generate human-readable summaries of daily discoveries for the pi.ruv.io landing page.
- **Collaborative filtering**: Use connected agents' query patterns to prioritize which domains to explore deeper.
- **Multi-brain federation**: Enable multiple brain instances to share discoveries peer-to-peer, creating a decentralized scientific knowledge network.

## 9. References

- ADR-040: Planet Detection Pipeline — Established the pattern for NASA API integration and anomaly detection
- ADR-049: Verified Training Pipeline — Witness chain architecture and provenance standards
- ADR-057: Federated RVF Transfer Learning — LoRA federation protocol with Byzantine robustness
- ADR-059: Shared Brain Google Cloud — Cloud Run deployment architecture for pi.ruv.io
- ADR-083: Brain Training Loops — SONA learning loops, background training, `/v1/train` endpoint
