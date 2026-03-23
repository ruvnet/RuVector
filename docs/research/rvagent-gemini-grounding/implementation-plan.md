# Implementation Plan: rvAgent Gemini Grounding Agents

## Prerequisites

Before implementation, verify:
- [ ] Gemini API key accessible via `gcloud secrets versions access latest --secret=GOOGLE_AI_API_KEY`
- [ ] pi.ruv.io Brain API accessible with auth token (PI env var)
- [ ] Cloud Run Jobs enabled in the project
- [ ] Cloud Scheduler API enabled

## Phase 0: Agent Runner Infrastructure

### Step 1: Create the Agent Runner Entry Point

**File**: `scripts/rvagent-grounding/agent-runner.js`

This is the main entry point that Cloud Run Jobs execute. It handles phase selection, configuration, and the execution loop.

```javascript
// Outline -- not final implementation
const PHASES = {
  1: { name: 'verifier',  handler: require('./phases/verify'),   batchSize: 20 },
  2: { name: 'relator',   handler: require('./phases/relate'),   batchSize: 25 },
  3: { name: 'explorer',  handler: require('./phases/explore'),  batchSize: 15 },
  4: { name: 'director',  handler: require('./phases/research'), batchSize: 10 },
};

async function main() {
  const phase = parseInt(process.argv.find(a => a.startsWith('--phase'))?.split('=')[1] || '1');
  const config = {
    brainUrl: process.env.BRAIN_URL || 'https://pi.ruv.io',
    brainKey: process.env.PI,
    geminiKey: process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY,
    geminiModel: process.env.GEMINI_MODEL || 'gemini-2.5-flash',
    maxTokensBudget: parseInt(process.env.MAX_TOKENS_BUDGET || '50000'),
    dryRun: process.env.DRY_RUN === 'true',
  };

  // Validate preconditions
  // Execute phase handler
  // Log metrics
  // Exit
}
```

### Step 2: Create the Gemini Client with Grounding

**File**: `scripts/rvagent-grounding/lib/gemini-client.js`

Wraps the Gemini API with grounding support, PHI detection, and token tracking.

```javascript
// Key methods:
class GeminiGroundedClient {
  constructor(apiKey, model, options = {}) { /* ... */ }

  // Send a prompt with Google Search grounding enabled
  async groundedQuery(prompt, options = {}) {
    // Returns: { text, groundingChunks, groundingSupports, searchQueries, tokensUsed }
  }

  // Sanitize content before sending to Gemini (PHI removal)
  sanitize(content) {
    // Strip: names, dates, MRNs, emails, SSNs, phone numbers
    // Return: factual claims only
  }

  // Token budget tracking
  get tokensUsed() { /* ... */ }
  get budgetRemaining() { /* ... */ }
}
```

### Step 3: Create the Brain Client

**File**: `scripts/rvagent-grounding/lib/brain-client.js`

Wraps the pi.ruv.io REST API with retry logic. Reuses the same patterns as `mcp-server.js` brain tool handlers.

```javascript
class BrainClient {
  constructor(baseUrl, authToken) { /* ... */ }

  async search(query, options = {}) { /* GET /v1/memories/search */ }
  async list(options = {}) { /* GET /v1/memories/list */ }
  async share(memory) { /* POST /v1/memories */ }
  async getStatus() { /* GET /v1/status */ }
  async getDrift(domain) { /* GET /v1/drift */ }
  async listPropositions(options = {}) { /* GET /v1/propositions */ }
  async groundProposition(proposition) { /* POST /v1/ground */ }
  async reason(query, limit) { /* POST /v1/reason */ }
  async train() { /* POST /v1/train */ }
}
```

## Phase 1: Fact Verification Agent

### Step 4: Implement the Verifier

**File**: `scripts/rvagent-grounding/phases/verify.js`

```javascript
// Outline:
async function verify(config, brain, gemini) {
  // 1. Get cursor from brain (stored as a memory with tag "cursor-phase-1")
  const cursor = await getCursor(brain, 'phase-1-cursor');

  // 2. Fetch batch of memories
  const memories = await brain.list({
    limit: config.batchSize,
    offset: cursor,
    sort: 'quality',
  });

  // 3. Filter out already-verified (check for existing verification records)
  const unverified = await filterUnverified(brain, memories);

  // 4. For each unverified memory:
  for (const memory of unverified) {
    if (gemini.budgetRemaining <= 0) break;

    // a. Extract and sanitize claims
    const claims = extractClaims(memory.content);
    const sanitized = gemini.sanitize(claims);

    // b. Query Gemini with grounding
    const result = await gemini.groundedQuery(
      buildVerificationPrompt(sanitized),
      { maxTokens: 1024 }
    );

    // c. Parse verification status
    const status = parseVerificationResult(result.text);

    // d. Store verification record
    await brain.share({
      title: `Verification: ${memory.title}`,
      content: formatVerificationRecord(status, result.groundingChunks),
      category: 'verification',
      tags: ['grounded', 'phase-1', memory.id, status.overall],
    });

    // e. Log structured metrics
    log({ action: 'verify', memory_id: memory.id, status: status.overall,
          sources: result.groundingChunks?.length || 0 });
  }

  // 5. Update cursor
  await saveCursor(brain, 'phase-1-cursor', cursor + memories.length);
}
```

**Verification Prompt Template**:

```
You are a fact-checker. Verify each claim below using current, authoritative sources.

Claims to verify:
{{#each claims}}
{{@index}}. {{this}}
{{/each}}

For each claim, respond with:
- VERIFIED: the claim is supported by current sources
- UNVERIFIED: cannot find supporting evidence
- CONTRADICTED: current evidence contradicts this claim

Output JSON array:
[{"claim_index": 0, "status": "VERIFIED", "explanation": "...", "source_url": "..."}]
```

## Phase 2: Relational Proposition Generator

### Step 5: Implement the Relator

**File**: `scripts/rvagent-grounding/phases/relate.js`

```javascript
async function relate(config, brain, gemini) {
  // 1. Get verified memories (search for phase-1 verification records)
  const verifiedIds = await getVerifiedMemoryIds(brain);

  // 2. Fetch memory pairs with moderate similarity
  //    Use brain_search with each memory's title to find related ones
  const pairs = await findCandidatePairs(brain, verifiedIds, {
    minSimilarity: 0.4,
    maxSimilarity: 0.85,
    maxPairs: config.batchSize,
  });

  // 3. For each pair, ask Gemini to determine relationship
  for (const [memA, memB] of pairs) {
    if (gemini.budgetRemaining <= 0) break;

    const result = await gemini.groundedQuery(
      buildRelationPrompt(memA, memB),
      { maxTokens: 1024 }
    );

    const relations = parseRelationResult(result.text);

    for (const rel of relations) {
      if (rel.predicate === 'no_relationship') continue;
      if (rel.confidence < 0.5) continue;

      // Inject into symbolic engine
      await brain.groundProposition({
        predicate: rel.predicate,
        arguments: [memA.id, memB.id],
        embedding: averageEmbeddings(memA.embedding, memB.embedding),
        evidence_ids: [memA.id, memB.id],
      });

      // Share as discoverable memory
      await brain.share({
        title: `Relation: ${memA.title} ${rel.predicate} ${memB.title}`,
        content: `${rel.explanation}\nConfidence: ${rel.confidence}\nSources: ${rel.source_urls?.join(', ')}`,
        category: 'pattern',
        tags: ['relation', rel.predicate, 'phase-2', 'grounded'],
      });
    }
  }

  // 4. Trigger inference engine
  const inferences = await brain.reason('transitive inferences from new relations', 20);
  log({ action: 'inference', chains: inferences?.inferences?.length || 0 });
}
```

**Relation Prompt Template**:

```
Analyze the relationship between these two knowledge items:

A: {{memA.title}}
   {{memA.content_summary}}

B: {{memB.title}}
   {{memB.content_summary}}

Determine if any of these relationships exist (choose all that apply):
- implies: if A is true, B is likely true
- causes: A is a mechanism or cause of B
- requires: A depends on or requires B
- contradicts: A and B cannot both be true
- similar_to: A and B describe the same concept differently
- solves: A provides a solution to problem B
- no_relationship: no meaningful connection

Verify your assessment against current sources.

Output JSON:
[{"predicate": "causes", "confidence": 0.85, "explanation": "...", "source_urls": ["..."]}]
```

## Phase 3: Cross-Domain Discovery

### Step 6: Implement the Explorer

**File**: `scripts/rvagent-grounding/phases/explore.js`

```javascript
async function explore(config, brain, gemini) {
  // 1. Identify domain boundaries
  const domains = await identifyDomains(brain);
  // Expected: ["medicine", "computer_science", "physics", ...]

  // 2. For each domain pair, find unexpected similarity
  for (let i = 0; i < domains.length; i++) {
    for (let j = i + 1; j < domains.length; j++) {
      if (gemini.budgetRemaining <= 0) break;

      const domainA = domains[i];
      const domainB = domains[j];

      // Search domain A memories against domain B
      const crossPairs = await findCrossDomainPairs(brain, domainA, domainB, {
        minSimilarity: 0.25,
        maxSimilarity: 0.60,
        maxPairs: 5,
      });

      for (const [memA, memB] of crossPairs) {
        const result = await gemini.groundedQuery(
          buildCrossDomainPrompt(memA, memB, domainA.name, domainB.name),
          { maxTokens: 1536 }
        );

        const connection = parseCrossDomainResult(result.text);
        if (!connection || connection.confidence < 0.6) continue;

        await brain.share({
          title: `Cross-Domain: ${domainA.name} <-> ${domainB.name}: ${connection.type}`,
          content: `${connection.explanation}\n\nEvidence: ${connection.evidence_urls?.join('\n')}`,
          category: 'pattern',
          tags: ['cross-domain-discovery', domainA.name, domainB.name, 'phase-3'],
        });

        await brain.groundProposition({
          predicate: connection.bridge_predicate || 'relates_to',
          arguments: [memA.id, memB.id],
          embedding: averageEmbeddings(memA.embedding, memB.embedding),
          evidence_ids: [memA.id, memB.id],
        });
      }
    }
  }
}
```

## Phase 4: Autonomous Research Director

### Step 7: Implement the Research Director

**File**: `scripts/rvagent-grounding/phases/research.js`

```javascript
async function research(config, brain, gemini) {
  // 1. Check drift
  const drift = await brain.getDrift();
  const highDrift = drift.domains?.filter(d => d.velocity > 2.0) || [];

  if (highDrift.length === 0) {
    log({ action: 'research_skip', reason: 'no_high_drift_domains' });
    return;
  }

  for (const domain of highDrift) {
    if (gemini.budgetRemaining <= 0) break;

    // 2. Formulate research questions
    const questions = [
      `What are the latest developments in ${domain.name} in the past 30 days?`,
      `Are there new findings that change our understanding of ${domain.name}?`,
    ];

    // 3. Get current brain knowledge for context
    const existing = await brain.search(domain.name, { limit: 10 });
    const context = existing.map(m => m.title).join('; ');

    for (const question of questions) {
      const result = await gemini.groundedQuery(
        buildResearchPrompt(question, context),
        { maxTokens: 2048 }
      );

      const findings = parseResearchResult(result.text);

      for (const finding of findings) {
        // 4. Store finding
        await brain.share({
          title: `Research: ${finding.title}`,
          content: `${finding.content}\n\nSources: ${finding.sources?.join('\n')}`,
          category: 'solution',
          tags: ['research', domain.name, 'phase-4', 'grounded'],
        });
      }
    }
  }

  // 5. Trigger training to learn from new data
  await brain.train();
}
```

## Deployment

### Step 8: Dockerfile for Cloud Run Job

**File**: `scripts/rvagent-grounding/Dockerfile`

```dockerfile
FROM node:20-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --production
COPY . .
ENTRYPOINT ["node", "agent-runner.js"]
```

### Step 9: Cloud Run Job Definitions

```bash
# Build and push
gcloud builds submit --tag gcr.io/$PROJECT/rvagent-grounding:latest \
  scripts/rvagent-grounding/

# Create jobs (one per phase)
for PHASE in 1 2 3 4; do
  gcloud run jobs create rvagent-phase-${PHASE} \
    --image gcr.io/$PROJECT/rvagent-grounding:latest \
    --args="--phase=${PHASE}" \
    --set-secrets="GEMINI_API_KEY=GOOGLE_AI_API_KEY:latest,PI=PI_BRAIN_TOKEN:latest" \
    --set-env-vars="BRAIN_URL=https://pi.ruv.io,GEMINI_MODEL=gemini-2.5-flash,GEMINI_GROUNDING=true" \
    --memory=512Mi \
    --cpu=1 \
    --max-retries=1 \
    --task-timeout=900s \
    --region=us-central1
done
```

### Step 10: Cloud Scheduler Jobs

```bash
# Phase 1: Verify every 6 hours
gcloud scheduler jobs create http rvagent-verify \
  --schedule="0 */6 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/rvagent-phase-1:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA_EMAIL \
  --location=us-central1

# Phase 2: Relate daily at 02:00 UTC
gcloud scheduler jobs create http rvagent-relate \
  --schedule="0 2 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/rvagent-phase-2:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA_EMAIL \
  --location=us-central1

# Phase 3: Explore daily at 06:00 UTC
gcloud scheduler jobs create http rvagent-explore \
  --schedule="0 6 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/rvagent-phase-3:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA_EMAIL \
  --location=us-central1

# Phase 4: Research every 12 hours
gcloud scheduler jobs create http rvagent-research \
  --schedule="0 */12 * * *" \
  --uri="https://us-central1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT/jobs/rvagent-phase-4:run" \
  --http-method=POST \
  --oauth-service-account-email=$SA_EMAIL \
  --location=us-central1
```

## File Summary

### New Files to Create

| File | Purpose | Lines (est.) |
|------|---------|-------------|
| `scripts/rvagent-grounding/agent-runner.js` | Entry point, phase dispatch | ~120 |
| `scripts/rvagent-grounding/lib/gemini-client.js` | Gemini API with grounding + PHI sanitizer | ~200 |
| `scripts/rvagent-grounding/lib/brain-client.js` | pi.ruv.io REST client with retry | ~180 |
| `scripts/rvagent-grounding/lib/phi-detector.js` | PHI detection and removal | ~80 |
| `scripts/rvagent-grounding/phases/verify.js` | Phase 1: Fact verification | ~150 |
| `scripts/rvagent-grounding/phases/relate.js` | Phase 2: Relation generation | ~180 |
| `scripts/rvagent-grounding/phases/explore.js` | Phase 3: Cross-domain discovery | ~160 |
| `scripts/rvagent-grounding/phases/research.js` | Phase 4: Autonomous research | ~140 |
| `scripts/rvagent-grounding/package.json` | Dependencies | ~15 |
| `scripts/rvagent-grounding/Dockerfile` | Cloud Run Job image | ~8 |
| `docs/adr/ADR-122-rvagent-gemini-grounding-agents.md` | ADR | ~150 |

### Existing Files to Modify

| File | Change | Reason |
|------|--------|--------|
| `crates/mcp-brain-server/src/routes.rs` | Add `POST /v1/ground` handler for batch propositions | Phase 2 needs to inject multiple propositions per cycle |
| `npm/packages/ruvector/bin/mcp-server.js` | Add `brain_ground` and `brain_reason` tool definitions | Enable rvagent MCP tools to access proposition injection |

### No Changes Required

| Component | Reason |
|-----------|--------|
| `crates/mcp-brain-server/src/symbolic.rs` | `GroundedProposition`, `HornClause`, `NeuralSymbolicBridge` already support all needed predicate types |
| `crates/mcp-brain-server/src/optimizer.rs` | Gemini client with grounding already implemented; agents use their own client |
| `npm/packages/ruvector/src/core/` | Agents run as standalone scripts, not as part of the rvagent library |

## Testing Strategy

### Unit Tests

```
scripts/rvagent-grounding/tests/
  phi-detector.test.js     -- PHI patterns detection
  gemini-client.test.js    -- Response parsing, sanitization (mocked HTTP)
  brain-client.test.js     -- API mapping, retry logic (mocked HTTP)
  verify.test.js           -- Verification prompt construction, result parsing
  relate.test.js           -- Pair selection, relation parsing
  explore.test.js          -- Domain identification, cross-domain filtering
  research.test.js         -- Drift handling, question formulation
```

### Integration Tests

Run with `DRY_RUN=true` to test against real brain API without Gemini calls:

```bash
DRY_RUN=true node agent-runner.js --phase=1  # Fetches memories, logs what it would verify
```

### Acceptance Criteria

| Phase | Criterion | How to Verify |
|-------|-----------|---------------|
| 1 | 80%+ of high-quality memories have grounding status | `brain_search("grounded phase-1")` |
| 2 | >= 50 relational propositions exist | `GET /v1/propositions?predicate=causes` |
| 2 | Horn clause engine produces inferences | `POST /v1/reason` returns non-empty |
| 3 | >= 10 cross-domain discoveries | `brain_search("cross-domain-discovery")` |
| 4 | SONA patterns > 0 | `GET /v1/sona/stats` |
| All | Monthly Gemini cost < $50 | Token counter in agent-runner.js |

## Rollout Plan

### Week 1: Infrastructure + Phase 1
- Implement `agent-runner.js`, `gemini-client.js`, `brain-client.js`, `phi-detector.js`
- Implement Phase 1 verifier
- Deploy Cloud Run Job for Phase 1
- Manual execution to verify 50 memories
- Create Cloud Scheduler job

### Week 2: Phase 2
- Implement Phase 2 relator
- Run locally against verified memories
- Verify propositions appear in `/v1/propositions`
- Verify Horn clause inferences via `/v1/reason`
- Deploy and schedule

### Week 3: Phases 3 + 4
- Implement Phases 3 and 4
- Run cross-domain explorer on medical + CS domains
- Run research director on one high-drift domain
- Deploy and schedule

### Week 4: Monitoring + Tuning
- Set up Cloud Monitoring dashboard
- Review cost after first full week of scheduled execution
- Tune batch sizes and similarity thresholds based on results
- Document findings in ADR-122 appendix
