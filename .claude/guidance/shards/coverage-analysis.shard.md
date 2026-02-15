# Coverage Analysis Domain Shard

**Domain**: coverage-analysis
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **Sublinear Algorithm Mandate**: Coverage gap detection MUST use the O(log n) HNSW-based SublinearCoverageAnalyzer for codebases > 1000 files. Linear scan is prohibited at scale.

2. **Real Data Only**: Coverage reports MUST be derived from actual test execution results, not estimated or synthesized data.

3. **Risk-Weighted Prioritization**: Coverage gaps MUST be prioritized by risk score (complexity, change frequency, defect history), not just coverage percentage.

4. **Trend Accuracy**: Coverage trend analysis MUST use at least 3 historical data points; single-point trends are invalid.

5. **Embedding Freshness**: Coverage embeddings MUST be regenerated when source code changes exceed the staleness threshold (default: 7 days or 100+ file changes).

6. **Gap Verification**: Identified coverage gaps MUST be verified against the actual codebase before generating test suggestions.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Coverage Score | 0.6 | 0.8 | < 0.4 |
| Confidence | 0.7 | 0.9 | < 0.5 |
| Gap Detection Accuracy | 0.8 | 0.95 | < 0.6 |
| Risk Scoring Accuracy | 0.7 | 0.85 | < 0.5 |
| HNSW Query Efficiency | O(log n) | O(log n) | O(n) |

---

## Invariants

```
INVARIANT sublinear_at_scale:
  FOR ALL analysis IN coverage_analyses:
    IF analysis.codebase_size > 1000 THEN
      analysis.algorithm = 'HNSW_SUBLINEAR' AND
      analysis.complexity <= O(log(analysis.codebase_size))
```

```
INVARIANT real_coverage_data:
  FOR ALL report IN coverage_reports:
    EXISTS execution_source WHERE
      execution_source.report_id = report.id AND
      execution_source.type IN ['jest', 'vitest', 'nyc', 'c8'] AND
      execution_source.timestamp IS NOT NULL
```

```
INVARIANT trend_minimum_points:
  FOR ALL trend IN coverage_trends:
    trend.data_points.length >= 3
```

```
INVARIANT embedding_freshness:
  FOR ALL embedding IN coverage_embeddings:
    (NOW() - embedding.created_at) < STALENESS_THRESHOLD OR
    embedding.regeneration_scheduled = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/coverage-analysis/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Coverage Analyzer Service | `services/coverage-analyzer.ts` | Core coverage analysis |
| Gap Detector Service | `services/gap-detector.ts` | Coverage gap identification |
| Risk Scorer Service | `services/risk-scorer.ts` | Multi-factor risk scoring |
| HNSW Index | `services/hnsw-index.ts` | O(log n) vector indexing |
| Coverage Embedder | `services/coverage-embedder.ts` | Code-to-vector embedding |
| Sublinear Analyzer | `services/sublinear-analyzer.ts` | ADR-003 sublinear implementation |

**ADR Reference**: ADR-003 defines sublinear algorithm requirements.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-coverage-specialist` | Full analysis, gap detection, risk scoring |
| **Secondary** | `qe-gap-detector` | Gap detection, test suggestions |
| **Support** | `qe-test-architect` | Receive gaps, generate tests |
| **Readonly** | `qe-quality-gate` | Coverage validation |

**Forbidden Agents**: Agents without HNSW index access MUST NOT perform gap detection on large codebases.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Coverage < 0.4 | CRITICAL | Escalate to Queen Coordinator, prioritize test generation |
| O(n) algorithm used at scale | CRITICAL | Escalate, require algorithm fix |
| Gap detection accuracy < 0.6 | HIGH | Escalate, recalibrate model |
| Stale embeddings > 14 days | HIGH | Escalate, force regeneration |
| Trend shows declining coverage | MEDIUM | Notify coordinator, generate report |
| Risk scoring confidence < 0.5 | MEDIUM | Request additional data points |
| Synthetic data detected | CRITICAL | Block report, escalate for investigation |

---

## Memory Namespace

- **Namespace**: `qe-patterns/coverage-analysis`
- **Retention**: 30 days with minimum 3 uses
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-execution` | Input | Receive execution coverage data |
| `test-generation` | Output | Provide gaps for test targeting |
| `quality-assessment` | Output | Report coverage metrics |
| `code-intelligence` | Input | Receive code structure for risk scoring |
| `learning-optimization` | Bidirectional | Share coverage patterns |

---

## Performance Benchmarks (ADR-003)

| Codebase Size | Traditional O(n) | v3 O(log n) | Improvement |
|---------------|-----------------|-------------|-------------|
| 1,000 files | 1,000 ops | 10 ops | 100x |
| 10,000 files | 10,000 ops | 13 ops | 770x |
| 100,000 files | 100,000 ops | 17 ops | 5,900x |

---

*This shard is enforced by @claude-flow/guidance governance system.*
