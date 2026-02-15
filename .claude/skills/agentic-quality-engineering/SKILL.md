---
name: agentic-quality-engineering
description: "AI agents as force multipliers for quality work. Core skill for all 19 QE agents using PACT principles."
category: qe-core
priority: critical
tokenEstimate: 1400
agents: [qe-test-generator, qe-test-executor, qe-coverage-analyzer, qe-quality-gate, qe-quality-analyzer, qe-performance-tester, qe-security-scanner, qe-requirements-validator, qe-production-intelligence, qe-fleet-commander, qe-deployment-readiness, qe-regression-risk-analyzer, qe-test-data-architect, qe-api-contract-validator, qe-flaky-test-hunter, qe-visual-tester, qe-chaos-engineer, qe-code-complexity, qx-partner]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [pact, agents, fleet, coordination, autonomous, foundational]
trust_tier: 1
validation:
  schema_path: schemas/output.json

---

# Agentic Quality Engineering

<default_to_action>
When implementing agentic QE or coordinating agents:
1. SPAWN appropriate agent(s) for the task using `Task` tool with agent type
2. CONFIGURE agent coordination (hierarchical/mesh/sequential)
3. EXECUTE with PACT principles: Proactive analysis, Autonomous operation, Collaborative feedback, Targeted risk focus
4. VALIDATE results through quality gates before deployment
5. LEARN from outcomes - store patterns in `aqe/learning/*` namespace

**Quick Agent Selection:**
- Test generation needed → `qe-test-generator`
- Coverage gaps → `qe-coverage-analyzer`
- Quality decision → `qe-quality-gate`
- Security scan → `qe-security-scanner`
- Performance test → `qe-performance-tester`
- Full pipeline → `qe-fleet-commander`

**Critical Success Factors:**
- Agents amplify human expertise, not replace it
- Human-in-the-loop for critical decisions
- Measure: bugs caught, time saved, coverage improved
</default_to_action>

## Quick Reference Card

### When to Use
- Designing autonomous testing systems
- Scaling QE with intelligent agents
- Implementing multi-agent coordination
- Building CI/CD quality pipelines

### PACT Principles
| Principle | Agent Behavior | Human Role |
|-----------|---------------|------------|
| **P**roactive | Analyze pre-merge, predict risk | Set guardrails |
| **A**utonomous | Execute tests, fix flaky tests | Review critical |
| **C**ollaborative | Multi-agent coordination | Provide context |
| **T**argeted | Risk-based prioritization | Define risk areas |

### 19-Agent Fleet
| Category | Agents | Primary Use |
|----------|--------|-------------|
| Core Testing (5) | test-generator, test-executor, coverage-analyzer, quality-gate, quality-analyzer | Daily testing |
| Performance/Security (2) | performance-tester, security-scanner | Non-functional |
| Strategic (3) | requirements-validator, production-intelligence, fleet-commander | Planning |
| Advanced (4) | regression-risk-analyzer, test-data-architect, api-contract-validator, flaky-test-hunter | Specialized |
| Visual/Chaos (2) | visual-tester, chaos-engineer | Edge cases |
| Deployment (1) | deployment-readiness | Release |
| Analysis (1) | code-complexity | Maintainability |

### Coordination Patterns
```
Hierarchical: fleet-commander → [generators] → [executors] → quality-gate
Mesh: test-gen ↔ coverage ↔ quality (peer decisions)
Sequential: risk-analyzer → test-gen → executor → coverage → gate
```

### Success Criteria
✅ 10x deployment frequency with same/better quality
✅ Coverage gaps detected in real-time
✅ Bugs caught pre-production
❌ Agents acting without human oversight on critical decisions
❌ Deploying all 19 agents at once (start with 1-2)

---

## Core Concepts

### QE Evolution
| Stage | Approach | Limitation |
|-------|----------|------------|
| Traditional | Manual everything | Human bottleneck |
| Automation | Scripts + fixed scenarios | Needs orchestration |
| **Agentic** | AI agents + human judgment | Requires trust-building |

**Core Premise:** Agents amplify human expertise for 10x scale.

### Key Capabilities

**1. Intelligent Test Generation**
```typescript
// Agent analyzes code change, generates targeted tests
const tests = await qeTestGenerator.generate(prDiff);
// → Happy path, edge cases, error handling tests
```

**2. Pattern Detection** - Scan logs, find anomalies, correlate errors

**3. Adaptive Strategy** - Adjust test focus based on risk signals

**4. Root Cause Analysis** - Link failures to code changes, suggest fixes

---

## Agent Coordination

### Memory Namespaces
```
aqe/test-plan/*     - Test planning decisions
aqe/coverage/*      - Coverage analysis results
aqe/quality/*       - Quality metrics and gates
aqe/learning/*      - Patterns and Q-values
aqe/coordination/*  - Cross-agent state
```

### Memory Operations (MCP Tools)

**CRITICAL**: Always use `mcp__agentic-qe__memory_store` with `persist: true` for learnings.

**1. Store data to persistent memory:**
```javascript
// Store test plan decisions (persisted to .agentic-qe/memory.db)
mcp__agentic-qe__memory_store({
  key: "aqe/test-plan/pr-123",
  namespace: "aqe/test-plan",
  value: {
    prNumber: 123,
    riskLevel: "medium",
    requiredCoverage: 85,
    testTypes: ["unit", "integration"],
    estimatedTime: 1800
  },
  persist: true,  // ⚠️ REQUIRED for cross-session persistence
  ttl: 604800     // 7 days (0 = permanent)
})
```

**2. Retrieve prior learnings before task:**
```javascript
// Query patterns before starting test generation
const priorData = await mcp__agentic-qe__memory_retrieve({
  key: "aqe/learning/patterns/test-generation/*",
  namespace: "aqe/learning",
  includeMetadata: true
})

// Use patterns to guide current task
if (priorData.success) {
  console.log(`Loaded ${priorData.patterns.length} prior patterns`);
}
```

**3. Store coverage analysis results:**
```javascript
mcp__agentic-qe__memory_store({
  key: "aqe/coverage/auth-module",
  namespace: "aqe/coverage",
  value: {
    moduleId: "auth-module",
    currentCoverage: 78,
    gaps: ["error-handling", "edge-cases"],
    suggestedTests: 12,
    priority: "high"
  },
  persist: true,
  ttl: 1209600  // 14 days
})
```

### Three-Phase Memory Protocol

For coordinated multi-agent tasks, use the STATUS → PROGRESS → COMPLETE pattern:

```javascript
// PHASE 1: STATUS - Task starting
mcp__agentic-qe__memory_store({
  key: "aqe/coordination/task-123/status",
  namespace: "aqe/coordination",
  value: { status: "running", agent: "qe-test-generator", startTime: Date.now() },
  persist: true
})

// PHASE 2: PROGRESS - Intermediate updates
mcp__agentic-qe__memory_store({
  key: "aqe/coordination/task-123/progress",
  namespace: "aqe/coordination",
  value: { progress: 50, action: "generating-unit-tests", testsGenerated: 25 },
  persist: true
})

// PHASE 3: COMPLETE - Task finished
mcp__agentic-qe__memory_store({
  key: "aqe/coordination/task-123/complete",
  namespace: "aqe/coordination",
  value: {
    status: "complete",
    result: "success",
    testsGenerated: 47,
    coverageAchieved: 92.3,
    duration: 15000
  },
  persist: true
})
```

### Blackboard Events
| Event | Trigger | Subscribers |
|-------|---------|-------------|
| `test:generated` | New tests created | executor, coverage |
| `coverage:gap` | Gap detected | test-generator |
| `quality:decision` | Gate evaluated | fleet-commander |
| `security:finding` | Vulnerability found | quality-gate |

### Example: PR Quality Pipeline
```typescript
// 1. Risk analysis
const risks = await Task("Analyze PR", prDiff, "qe-regression-risk-analyzer");

// 2. Generate tests for risks
const tests = await Task("Generate tests", risks, "qe-test-generator");

// 3. Execute + analyze
const results = await Task("Run tests", tests, "qe-test-executor");
const coverage = await Task("Check coverage", results, "qe-coverage-analyzer");

// 4. Quality decision
const decision = await Task("Evaluate", {results, coverage}, "qe-quality-gate");
// → GO/NO-GO with rationale
```

---

## Implementation Phases

| Phase | Duration | Goal | Agent(s) |
|-------|----------|------|----------|
| Experiment | Weeks 1-4 | Validate one use case | 1 agent |
| Integrate | Months 2-3 | CI/CD pipeline | 3-4 agents |
| Scale | Months 4-6 | Multiple use cases | 8+ agents |
| Evolve | Ongoing | Continuous learning | Full fleet |

### Phase 1 Example
```bash
# Week 1: Deploy single agent
aqe agent spawn qe-test-generator

# Weeks 2-3: Generate tests for 10 PRs
# Track: bugs found, test quality, review time

# Week 4: Measure impact
aqe agent metrics qe-test-generator
# → Tests: 150, Bugs: 12, Time saved: 8h
```

---

## Limitations & Strengths

### Agents Excel At
- **Volume**: Scan thousands of logs in seconds
- **Patterns**: Find correlations humans miss
- **Tireless**: 24/7 testing and monitoring
- **Speed**: Instant code change analysis

### Agents Need Humans For
- Business context and priorities
- Ethical judgment and trade-offs
- Creative exploration ("what if" scenarios)
- Domain expertise (healthcare, finance, legal)

---

## Best Practices

| Do | Don't |
|----|-------|
| Start with one agent, one use case | Deploy all 18 at once |
| Build feedback loops early | Deploy and forget |
| Human reviews agent output | Auto-merge without review |
| Measure bugs caught, time saved | Track vanity metrics (test count) |
| Build trust gradually | Give full autonomy immediately |

### Trust Progression
```
Month 1: Agent suggests → Human decides
Month 2: Agent acts → Human reviews after
Month 3: Agent autonomous on low-risk
Month 4: Agent handles critical with oversight
```

---

## Agent Coordination Hints

```yaml
coordination:
  topology: hierarchical
  commander: qe-fleet-commander
  memory_namespace: aqe/coordination
  blackboard_topic: qe-fleet

preload_skills:
  - agentic-quality-engineering  # Always (this skill)
  - risk-based-testing           # For prioritization
  - quality-metrics              # For measurement

agent_assignments:
  qe-test-generator: [api-testing-patterns, tdd-london-chicago]
  qe-coverage-analyzer: [quality-metrics, risk-based-testing]
  qe-security-scanner: [security-testing, risk-based-testing]
  qe-performance-tester: [performance-testing]
```

---

## Related Skills
- `holistic-testing-pact` - PACT principles deep dive
- `risk-based-testing` - Prioritize agent focus
- `quality-metrics` - Measure agent effectiveness
- `api-testing-patterns`, `security-testing`, `performance-testing` - Specialized testing

## Resources
- Agent definitions: `.claude/agents/`
- CLI: `aqe agent --help`
- Fleet status: `aqe fleet status`

---

**Success Metric:** Deploy 10x more frequently with same or better quality through intelligent agent collaboration.
