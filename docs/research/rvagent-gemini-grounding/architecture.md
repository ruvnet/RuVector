# Architecture: rvAgent Gemini Grounding Agents

## System Architecture

```
                    Cloud Scheduler (4 cron jobs)
                           |
              +------------+------------+------------+
              |            |            |            |
        [Phase 1]    [Phase 2]    [Phase 3]    [Phase 4]
        Verifier     Relator      Explorer     Director
              |            |            |            |
              +------------+------------+------------+
                           |
                    Cloud Run Job
                    (node agent-runner.js --phase N)
                           |
              +------------+------------+
              |                         |
        rvagent MCP tools         Gemini API
        (brain_*, hooks_*)        (with grounding)
              |                         |
        pi.ruv.io Brain          Google Search
        (REST API)               (live web data)
```

## Data Flow Per Agent Cycle

### Phase 1: Grounded Fact Verification

**Goal state**: Every memory with quality >= 3 has a `grounding_status` tag (`verified`, `unverified`, `contradicted`).

**Preconditions**: Brain has memories; Gemini API key configured.

**Action sequence**:

```
1. brain_list(limit=20, sort=quality, offset=cursor)
   --> Get batch of high-quality, untagged memories

2. For each memory:
   a. Extract key claims from memory content
      - Strip any PHI indicators (names, dates, MRNs)
      - Summarize to 2-3 factual claims

   b. Call Gemini with grounding:
      Prompt: "Verify these claims using current sources:
               Claim 1: {claim}
               Claim 2: {claim}
               For each claim, state VERIFIED, UNVERIFIED, or CONTRADICTED
               with the source URL."
      Tools: [{"google_search": {}}]

   c. Parse groundingMetadata:
      - Extract groundingChunks[].web.uri (source URLs)
      - Extract groundingSupports[].segment.text (supporting text)
      - Map each claim to its verification status

   d. Tag memory via brain_share (create a linked verification record):
      brain_share({
        title: "Verification: {memory_title}",
        content: "Status: VERIFIED\nSources: [url1, url2]\nClaims checked: ...",
        category: "verification",
        tags: ["grounded", "phase-1", memory_id, status]
      })

   e. Record trajectory step:
      hooks_trajectory_step({
        action: "verify",
        observation: status,
        reward: status === "verified" ? 1.0 : status === "contradicted" ? -1.0 : 0.0
      })

3. Update cursor for next batch
```

**Cost per cycle**: ~20 memories x ~500 input tokens = 10K tokens = $0.0015

**Effects on world state**:
- Memories gain grounding provenance
- SONA receives trajectory data for learning
- Contradictions surface for human review

### Phase 2: Relational Proposition Generator

**Goal state**: Brain has `implies`, `causes`, `requires`, `contradicts` propositions linking related memories. Horn clause engine can chain inferences.

**Preconditions**: Phase 1 has verified memories (grounding_status = verified).

**Action sequence**:

```
1. brain_search(query="*", limit=50)
   --> Get memory embeddings (returned in search results)

2. Compute pairwise cosine similarity (top-k pairs per memory)
   - Use the embedding vectors from search results
   - Filter to pairs with similarity in [0.4, 0.85]
     (too high = near-duplicate; too low = unrelated)
   - Prioritize cross-category pairs

3. For each pair (memory_A, memory_B):
   a. Construct Gemini prompt:
      "Given these two knowledge items:
       A: {memory_A.title} - {memory_A.content_summary}
       B: {memory_B.title} - {memory_B.content_summary}

       What is the relationship between A and B?
       Choose one or more:
       - A implies B (if A is true, B is likely true)
       - A causes B (A is a mechanism/cause of B)
       - A requires B (A depends on B)
       - A contradicts B (A and B cannot both be true)
       - A is_similar_to B (A and B describe the same concept differently)
       - A solves B (A is a solution to problem B)
       - no_relationship (no meaningful connection)

       Verify your answer using current sources.
       Output JSON: {predicate, confidence, explanation, source_urls}"
      Tools: [{"google_search": {}}]

   b. Parse response for predicate(s)

   c. Inject proposition via brain API:
      POST /v1/ground {
        predicate: "causes",
        arguments: [memory_A.id, memory_B.id],
        embedding: average(memory_A.embedding, memory_B.embedding),
        evidence_ids: [memory_A.id, memory_B.id]
      }

   d. Share discovery:
      brain_share({
        title: "Relation: {A.title} {predicate} {B.title}",
        content: "Grounded relation with {confidence}...",
        category: "pattern",
        tags: ["relation", predicate, "phase-2", "grounded"]
      })

4. Trigger inference:
   POST /v1/reason {query: "transitive inferences", limit: 20}
   --> Horn clauses can now chain: if A causes B and B causes C, then A causes C
```

**Cost per cycle**: ~25 pairs x ~800 input tokens = 20K tokens = $0.003

**Effects on world state**:
- New relational propositions feed the Horn clause engine
- Inference chains become possible
- Knowledge graph gains semantic edges (not just vector similarity)

### Phase 3: Cross-Domain Discovery

**Goal state**: Brain has `cross-domain-discovery` memories linking concepts across medicine, CS, and physics that humans would not typically connect.

**Preconditions**: Phase 2 has generated relational propositions; brain has memories in 2+ domains.

**Action sequence**:

```
1. Identify domain boundaries:
   brain_list(category="architecture")  --> CS memories
   brain_list(category="solution")      --> cross-domain memories
   brain_search(query="dermatology")    --> medical memories
   brain_search(query="quantum")        --> physics memories

2. Find cross-domain pairs:
   - Search each domain's memories against other domains
   - brain_search(query=medical_memory.title) within CS results
   - Filter to pairs with cosine similarity in [0.25, 0.60]
     (unexpected similarity -- too high means obvious connection)

3. For each cross-domain pair:
   a. Construct Gemini prompt:
      "These two concepts come from different domains:
       Domain 1 ({domain_A}): {memory_A.content_summary}
       Domain 2 ({domain_B}): {memory_B.content_summary}

       Is there a meaningful but non-obvious connection?
       Examples of valid connections:
       - Shared mathematical structure (e.g., diffusion equations in physics and epidemiology)
       - Analogous mechanisms (e.g., neural attention and immunological memory)
       - Transferable methods (e.g., graph algorithms for protein folding)

       If yes, explain the connection and provide evidence from current sources.
       Output JSON: {connection_type, explanation, confidence, evidence_urls, bridge_predicate}"
      Tools: [{"google_search": {}}]

   b. If connection found (confidence > 0.6):
      brain_share({
        title: "Cross-Domain: {domain_A} <-> {domain_B}: {connection_type}",
        content: "{explanation}\n\nEvidence: {evidence_urls}",
        category: "pattern",
        tags: ["cross-domain-discovery", domain_A, domain_B, "phase-3"]
      })

      POST /v1/ground {
        predicate: "relates_to",  // or "similar_to" for structural analogy
        arguments: [memory_A.id, memory_B.id],
        ...
      }
```

**Cost per cycle**: ~15 pairs x ~1200 input tokens = 18K tokens = $0.0027

### Phase 4: Autonomous Goal-Directed Research

**Goal state**: Brain stays current on fast-changing topics; SONA accumulates meaningful trajectories.

**Preconditions**: Drift detection shows changing areas; Phases 1-3 operational.

**Action sequence**:

```
1. Check drift:
   brain_drift() --> {domains: [{name, velocity, trend}]}

2. For each high-drift domain (velocity > 2.0):
   a. Formulate research questions:
      - "What changed in {domain} in the last 30 days?"
      - "Are there new papers/findings that affect our knowledge of {topic}?"

   b. Begin trajectory:
      hooks_trajectory_begin({
        name: "research_{domain}_{date}",
        metadata: {domain, drift_velocity}
      })

   c. Send to Gemini with grounding:
      Prompt: "Research question: {question}
               Current brain knowledge: {summary_of_domain_memories}
               What is the latest information? Cite sources."
      Tools: [{"google_search": {}}]

   d. For each finding:
      hooks_trajectory_step({
        action: "discover",
        observation: finding_summary,
        reward: finding.relevance_score
      })

      brain_share({
        title: "Research: {finding_title}",
        content: "{finding_content}\nSources: {urls}",
        category: "solution",
        tags: ["research", domain, "phase-4", "grounded"]
      })

   e. End trajectory:
      hooks_trajectory_end({reward: overall_relevance})

3. Trigger training:
   POST /v1/train --> SONA learns from accumulated trajectories
```

**Cost per cycle**: ~10 questions x ~1500 input tokens = 15K tokens = $0.0023

## Privacy and Security Architecture

### PHI Protection Pipeline

```
Memory content
  |
  +-- PHI detector (regex + heuristics)
  |     - Names: [A-Z][a-z]+ [A-Z][a-z]+ pattern
  |     - Dates: ISO 8601, MM/DD/YYYY
  |     - MRNs: 6+ digit sequences
  |     - Emails, SSNs, phone numbers
  |
  +-- Summarizer (extracts factual claims only)
  |     - "Patient X responded to treatment Y" -> "Treatment Y effective for condition Z"
  |     - Strip all proper nouns except domain terms
  |
  +-- Gemini prompt (sanitized content only)
```

### Data Flow Security

1. **Brain API auth**: All calls use `Authorization: Bearer ${PI}` token
2. **Gemini API auth**: Uses `GEMINI_API_KEY` from Cloud Secret Manager
3. **No raw memory content to Gemini**: Only summarized claims
4. **Grounding sources logged**: Every Gemini grounding URL stored with the verification record
5. **Audit trail**: Every agent action creates a brain_share record with phase tag

## State Space (GOAP Model)

### World State Properties

```typescript
interface WorldState {
  // Phase 1
  memories_total: number;            // ~1,809
  memories_verified: number;         // 0 -> grows
  memories_contradicted: number;     // 0 -> grows
  memories_unverified: number;       // 0 -> grows
  verification_cursor: number;       // pagination offset

  // Phase 2
  propositions_total: number;        // 11 -> grows
  propositions_relational: number;   // 0 -> grows (non-is_type_of)
  inference_chains: number;          // 0 -> grows
  pairs_evaluated: number;           // 0 -> grows

  // Phase 3
  cross_domain_discoveries: number;  // 0 -> grows
  domains_connected: Set<string>;    // {} -> grows

  // Phase 4
  sona_trajectories: number;         // 1 -> grows
  sona_patterns: number;             // 0 -> grows
  drift_domains_researched: number;  // 0 -> grows
  research_findings: number;         // 0 -> grows

  // System
  gemini_tokens_used: number;        // budget tracking
  last_run_timestamp: Date;
  errors_this_cycle: number;
}
```

### GOAP Actions

| Action | Preconditions | Effects | Cost |
|--------|--------------|---------|------|
| `verify_memory` | memory.grounding_status == null | memory.grounding_status = verified/contradicted | 500 tokens |
| `generate_relation` | mem_A.verified && mem_B.verified | propositions_relational++ | 800 tokens |
| `discover_bridge` | propositions_relational > 10, domains >= 2 | cross_domain_discoveries++ | 1200 tokens |
| `research_drift` | drift_velocity > 2.0 | sona_trajectories++, research_findings++ | 1500 tokens |
| `trigger_inference` | propositions_relational > 5 | inference_chains++ | 0 (local) |
| `trigger_training` | sona_trajectories > 5 | sona_patterns++ | 0 (local) |

### Goal States

```
Phase 1 Goal: memories_verified > memories_total * 0.8
Phase 2 Goal: propositions_relational > 50 AND inference_chains > 10
Phase 3 Goal: cross_domain_discoveries > 20 AND domains_connected.size >= 3
Phase 4 Goal: sona_patterns > 10 AND drift_domains_researched == high_drift_count
```

## Error Handling and Self-Correction

### Retry Strategy

```
Gemini API error:
  429 (rate limit) --> exponential backoff: 1s, 2s, 4s, max 3 retries
  500 (server)     --> retry once after 5s
  Other            --> log and skip this memory/pair

Brain API error:
  401 (auth)       --> abort cycle, alert
  429 (rate limit) --> backoff 2s
  500              --> retry once
  Other            --> log and continue
```

### Self-Correction Loop

When Phase 1 finds a contradiction:
1. The contradicted memory gets tagged `grounding_status: contradicted`
2. Phase 2 skips contradicted memories when generating relations
3. Phase 4 adds the contradicted topic to its research queue
4. When Phase 4 finds updated information, it creates a new memory
5. Phase 1 (next cycle) verifies the new memory
6. Phase 2 (next cycle) generates relations for the verified replacement

This creates a closed loop where the system self-corrects over time.

## Scheduling

| Job | Schedule | Phase | Max Duration |
|-----|----------|-------|-------------|
| `rvagent-verify` | Every 6 hours | 1 | 10 minutes |
| `rvagent-relate` | Daily at 02:00 UTC | 2 | 15 minutes |
| `rvagent-explore` | Daily at 06:00 UTC | 3 | 10 minutes |
| `rvagent-research` | Every 12 hours | 4 | 15 minutes |

All jobs run as Cloud Run Jobs (not services) to avoid cold-start costs. Each job is a single Node.js process running `agent-runner.js --phase N`.

## Metrics and Observability

### Structured Logging

Every agent action emits a structured log line:

```json
{
  "severity": "INFO",
  "agent": "phase-1-verifier",
  "action": "verify_memory",
  "memory_id": "uuid",
  "result": "verified",
  "sources": 3,
  "latency_ms": 2400,
  "tokens_input": 487,
  "tokens_output": 312
}
```

### Dashboard Metrics (exported to Cloud Monitoring)

- `rvagent/memories_verified` (counter)
- `rvagent/propositions_generated` (counter)
- `rvagent/cross_domain_discoveries` (counter)
- `rvagent/gemini_tokens_used` (counter, labeled by phase)
- `rvagent/cycle_duration_seconds` (histogram)
- `rvagent/errors` (counter, labeled by phase and error_type)
