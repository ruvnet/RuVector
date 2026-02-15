# Code Intelligence Domain Shard

**Domain**: code-intelligence
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Knowledge Graph Integrity**: All code indexing operations MUST maintain Knowledge Graph consistency; orphan nodes and broken edges are prohibited.

2. **Semantic Search Accuracy**: Semantic code search MUST return results with relevance scores; unranked results are invalid for impact analysis.

3. **Impact Analysis Completeness**: Impact analysis MUST traverse all dependency paths (direct and transitive); shallow analysis is prohibited for critical changes.

4. **Halstead Metrics Validation**: Complexity metrics MUST use validated Halstead formulas; custom metrics require explicit documentation.

5. **Index Freshness**: Code indices MUST be refreshed on file changes; stale indices (>24 hours without sync) MUST be flagged.

6. **C4 Model Accuracy**: Architecture diagrams generated from C4ModelService MUST reflect actual code structure, not aspirational design.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Index Completeness | 0.9 | 0.99 | < 0.7 |
| Search Relevance | 0.7 | 0.85 | < 0.5 |
| Impact Accuracy | 0.8 | 0.95 | < 0.6 |
| Graph Consistency | 0.95 | 1.0 | < 0.9 |
| Confidence | 0.7 | 0.85 | < 0.5 |

---

## Invariants

```
INVARIANT knowledge_graph_consistency:
  FOR ALL node IN knowledge_graph.nodes:
    NOT EXISTS orphan_edge WHERE
      orphan_edge.source = node.id AND
      NOT EXISTS target_node WHERE target_node.id = orphan_edge.target
```

```
INVARIANT semantic_search_ranking:
  FOR ALL search_result IN search_results:
    search_result.relevance_score IS NOT NULL AND
    search_result.relevance_score >= 0.0 AND
    search_result.relevance_score <= 1.0
```

```
INVARIANT impact_analysis_depth:
  FOR ALL impact_analysis IN analyses:
    IF impact_analysis.change_criticality = 'high' THEN
      impact_analysis.traversal_depth = 'full' AND
      impact_analysis.includes_transitive = true
```

```
INVARIANT index_freshness:
  FOR ALL index IN code_indices:
    (NOW() - index.last_sync) < 24_HOURS OR
    index.stale_flag = true
```

```
INVARIANT c4_model_accuracy:
  FOR ALL diagram IN c4_diagrams:
    diagram.source = 'code_analysis' AND
    diagram.validated_against_code = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/code-intelligence/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Knowledge Graph Service | `services/knowledge-graph.ts` | Code dependency graph |
| Semantic Analyzer Service | `services/semantic-analyzer.ts` | Code semantics and complexity |
| Impact Analyzer Service | `services/impact-analyzer.ts` | Change impact assessment |
| C4 Model Service | `services/c4-model/` | Architecture visualization |
| Metric Collector | `services/metric-collector/` | Code metrics aggregation |
| Code Intelligence Coordinator | `coordinator.ts` | Workflow orchestration |

**Key Interfaces**: `interfaces/index.ts` defines `KGNode`, `KGEdge`, `ImpactAnalysis`, `DependencyMap`, and related types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-code-analyst` | Full indexing, graph operations |
| **Secondary** | `qe-impact-assessor` | Impact analysis, change assessment |
| **Support** | `qe-architecture-visualizer` | C4 diagram generation |
| **Readonly** | `qe-test-architect` | Query dependencies for test planning |
| **Readonly** | `qe-coverage-specialist` | Query structure for gap analysis |

**Forbidden Actions**: No agent may modify Knowledge Graph without consistency validation.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Graph consistency < 0.9 | CRITICAL | Halt indexing, escalate for repair |
| Index completeness < 0.7 | CRITICAL | Force re-index, escalate to Queen Coordinator |
| Stale index > 48 hours | HIGH | Force sync, escalate |
| Impact accuracy < 0.6 | HIGH | Escalate, recalibrate analyzer |
| Orphan nodes detected | HIGH | Queue for repair, log pattern |
| Search relevance < 0.5 | MEDIUM | Recalibrate embedding model |
| C4 diagram-code mismatch | MEDIUM | Regenerate diagram, notify |

---

## Memory Namespace

- **Namespace**: `qe-patterns/code-intelligence`
- **Retention**: 30 days
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `coverage-analysis` | Output | Provide code structure for gap detection |
| `defect-intelligence` | Output | Provide structure for defect prediction |
| `test-generation` | Output | Provide dependencies for test targeting |
| `security-compliance` | Output | Provide code paths for security analysis |
| `quality-assessment` | Output | Provide complexity metrics |
| `learning-optimization` | Bidirectional | Share code patterns |

---

## Complexity Metrics (Halstead)

```typescript
interface HalsteadMetrics {
  vocabulary: number;      // n = n1 + n2 (distinct operators + operands)
  length: number;          // N = N1 + N2 (total operators + operands)
  calculatedLength: number; // N^ = n1*log2(n1) + n2*log2(n2)
  volume: number;          // V = N * log2(n)
  difficulty: number;      // D = (n1/2) * (N2/n2)
  effort: number;          // E = D * V
  time: number;            // T = E / 18 seconds
  bugs: number;            // B = V / 3000
}
```

---

## Risk Weights for Impact Analysis

```typescript
interface RiskWeights {
  codeChurn: 0.25;         // Recent change frequency
  complexity: 0.20;        // Cyclomatic and Halstead
  fanOut: 0.20;            // Number of dependencies
  defectHistory: 0.15;     // Historical defect density
  testCoverage: 0.10;      // Inverse of coverage
  age: 0.10;               // Time since last major refactor
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
