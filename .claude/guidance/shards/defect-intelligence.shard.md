# Defect Intelligence Domain Shard

**Domain**: defect-intelligence
**Version**: 1.0.0
**Last Updated**: 2026-02-03
**Parent Constitution**: `.claude/guidance/constitution.md`

---

## Domain Rules

1. **ML Model Transparency**: Defect predictions MUST include confidence scores and feature importance explanations; black-box predictions without explainability are prohibited.

2. **Historical Data Requirement**: Defect prediction models MUST be trained on at least 30 days of historical defect data before being used for production predictions.

3. **Root Cause Verification**: Root cause analyses MUST be validated against actual fix data when available, with feedback loops to improve accuracy.

4. **Pattern Learning Discipline**: Defect patterns MUST only be learned from confirmed defects (not suspected or predicted), with FlashAttention optimization where available.

5. **Regression Risk Honesty**: Regression risk assessments MUST clearly distinguish between measured risk (from data) and estimated risk (from heuristics).

6. **Cluster Quality**: Defect clustering MUST achieve minimum silhouette score for valid cluster formation; invalid clusters MUST NOT be reported as patterns.

---

## Quality Thresholds

| Metric | Minimum | Target | Critical |
|--------|---------|--------|----------|
| Prediction Accuracy | 0.7 | 0.85 | < 0.5 |
| Confidence | 0.6 | 0.8 | < 0.4 |
| Root Cause Accuracy | 0.6 | 0.8 | < 0.4 |
| Pattern Match Rate | 0.5 | 0.7 | < 0.3 |
| Cluster Silhouette Score | 0.3 | 0.5 | < 0.2 |
| Model Freshness | < 7 days | < 3 days | > 14 days |

---

## Invariants

```
INVARIANT prediction_explainability:
  FOR ALL prediction IN defect_predictions:
    prediction.confidence IS NOT NULL AND
    prediction.feature_importance IS NOT NULL AND
    prediction.feature_importance.length >= 3
```

```
INVARIANT historical_data_minimum:
  FOR ALL model IN active_models:
    model.training_data_days >= 30 AND
    model.training_samples >= 100
```

```
INVARIANT root_cause_feedback:
  FOR ALL rca IN root_cause_analyses:
    IF EXISTS actual_fix WHERE actual_fix.defect_id = rca.defect_id THEN
      rca.validated_against_fix = true
```

```
INVARIANT confirmed_defect_learning:
  FOR ALL pattern IN learned_patterns:
    pattern.source_defect.status = 'confirmed'
```

```
INVARIANT cluster_quality:
  FOR ALL cluster IN defect_clusters:
    cluster.silhouette_score >= 0.2 OR
    cluster.marked_as_invalid = true
```

---

## Patterns

**Domain Source**: `v3/src/domains/defect-intelligence/`

| Pattern | Location | Description |
|---------|----------|-------------|
| Defect Predictor Service | `services/defect-predictor.ts` | ML-based prediction |
| Pattern Learner Service | `services/pattern-learner.ts` | Defect pattern extraction |
| Root Cause Analyzer Service | `services/root-cause-analyzer.ts` | RCA algorithms |
| Defect Intelligence Coordinator | `coordinator.ts` | Workflow orchestration |

**Key Interfaces**: `interfaces/index.ts` defines `PredictionResult`, `RootCauseAnalysis`, `DefectCluster`, and related types.

---

## Agent Constraints

| Role | Agent ID | Permissions |
|------|----------|-------------|
| **Primary** | `qe-defect-predictor` | Full prediction, model training |
| **Secondary** | `qe-root-cause-analyzer` | RCA execution, pattern extraction |
| **Support** | `qe-pattern-learner` | Pattern learning from confirmed defects |
| **Readonly** | `qe-quality-gate` | Query predictions |

**Forbidden Agents**: Agents MUST NOT modify prediction models without retraining validation.

---

## Escalation Triggers

| Trigger | Severity | Action |
|---------|----------|--------|
| Prediction accuracy < 0.5 | CRITICAL | Escalate to Queen Coordinator, retrain model |
| Model stale > 14 days | CRITICAL | Force model refresh, escalate |
| Root cause accuracy < 0.4 | HIGH | Escalate, request manual validation |
| Cluster silhouette < 0.2 | HIGH | Invalidate clusters, recalibrate |
| FlashAttention unavailable | MEDIUM | Fall back to standard attention, log |
| Unconfirmed defect used for learning | HIGH | Revert pattern, escalate |
| Feature importance missing | MEDIUM | Block prediction, request explanation |

---

## Memory Namespace

- **Namespace**: `qe-patterns/defect-intelligence`
- **Retention**: 90 days (longer for validated patterns)
- **Contradiction Check**: Enabled

---

## Integration Points

| Domain | Integration Type | Purpose |
|--------|-----------------|---------|
| `test-execution` | Input | Receive failure data |
| `code-intelligence` | Input | Receive code structure for prediction |
| `coverage-analysis` | Input | Receive coverage data for risk |
| `quality-assessment` | Output | Report defect density |
| `learning-optimization` | Bidirectional | Share validated patterns |

---

## Model Metrics Schema

```typescript
interface ModelMetrics {
  modelId: string;
  version: string;
  trainedAt: Date;
  trainingDataDays: number;
  trainingSamples: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  flashAttentionEnabled: boolean;
  lastValidation: Date;
  validationAccuracy: number;
}
```

---

## Prediction Output Schema

```typescript
interface DefectPrediction {
  fileId: string;
  probability: number;
  confidence: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  featureImportance: {
    feature: string;
    importance: number;
    direction: 'increases_risk' | 'decreases_risk';
  }[];
  similarDefects: string[];
  recommendedActions: string[];
  dataSource: 'measured' | 'heuristic';
}
```

---

*This shard is enforced by @claude-flow/guidance governance system.*
