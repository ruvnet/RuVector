---
name: shift-right-testing
description: "Testing in production with feature flags, canary deployments, synthetic monitoring, and chaos engineering. Use when implementing production observability or progressive delivery."
category: testing-methodologies
priority: high
tokenEstimate: 1000
agents: [qe-production-intelligence, qe-chaos-engineer, qe-performance-tester, qe-quality-analyzer]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [shift-right, production-testing, canary, feature-flags, chaos-engineering, monitoring]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/shift-right-testing.yaml

---

# Shift-Right Testing

<default_to_action>
When testing in production or implementing progressive delivery:
1. IMPLEMENT feature flags for progressive rollout (1% → 10% → 50% → 100%)
2. DEPLOY with canary releases (compare metrics before full rollout)
3. MONITOR with synthetic tests (proactive) + RUM (reactive)
4. INJECT failures with chaos engineering (build resilience)
5. ANALYZE production data to improve pre-production testing

**Quick Shift-Right Techniques:**
- Feature flags → Control who sees what, instant rollback
- Canary deployment → 5% traffic, compare error rates
- Synthetic monitoring → Simulate users 24/7, catch issues before users
- Chaos engineering → Netflix-style failure injection
- RUM (Real User Monitoring) → Actual user experience data

**Critical Success Factors:**
- Production is the ultimate test environment
- Ship fast with safety nets, not slow with certainty
- Use production data to improve shift-left testing
</default_to_action>

## Quick Reference Card

### When to Use
- Progressive feature rollouts
- Production reliability validation
- Performance monitoring at scale
- Learning from real user behavior

### Shift-Right Techniques
| Technique | Purpose | When |
|-----------|---------|------|
| Feature Flags | Controlled rollout | Every feature |
| Canary | Compare new vs old | Every deployment |
| Synthetic Monitoring | Proactive detection | 24/7 |
| RUM | Real user metrics | Always on |
| Chaos Engineering | Resilience validation | Regularly |
| A/B Testing | User behavior validation | Feature decisions |

### Progressive Rollout Pattern
```
1% → 10% → 25% → 50% → 100%
↓      ↓      ↓      ↓
Check  Check  Check  Monitor
```

### Key Metrics to Monitor
| Metric | SLO Target | Alert Threshold |
|--------|------------|-----------------|
| Error rate | < 0.1% | > 1% |
| p95 latency | < 200ms | > 500ms |
| Availability | 99.9% | < 99.5% |
| Apdex | > 0.95 | < 0.8 |

---

## Feature Flags

```javascript
// Progressive rollout with LaunchDarkly/Unleash pattern
const newCheckout = featureFlags.isEnabled('new-checkout', {
  userId: user.id,
  percentage: 10,  // 10% of users
  allowlist: ['beta-testers']
});

if (newCheckout) {
  return <NewCheckoutFlow />;
} else {
  return <LegacyCheckoutFlow />;
}

// Instant rollback on issues
await featureFlags.disable('new-checkout');
```

---

## Canary Deployment

```yaml
# Flagger canary config
apiVersion: flagger.app/v1beta1
kind: Canary
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: checkout-service
  progressDeadlineSeconds: 60
  analysis:
    interval: 1m
    threshold: 5      # Max failed checks
    maxWeight: 50     # Max traffic to canary
    stepWeight: 10    # Increment per interval
    metrics:
      - name: request-success-rate
        threshold: 99
      - name: request-duration
        threshold: 500
```

---

## Synthetic Monitoring

```javascript
// Continuous production validation
await Task("Synthetic Tests", {
  endpoints: [
    { path: '/health', expected: 200, interval: '30s' },
    { path: '/api/products', expected: 200, interval: '1m' },
    { path: '/checkout', flow: 'full-purchase', interval: '5m' }
  ],
  locations: ['us-east', 'eu-west', 'ap-south'],
  alertOn: {
    statusCode: '!= 200',
    latency: '> 500ms',
    contentMismatch: true
  }
}, "qe-production-intelligence");
```

---

## Chaos Engineering

```typescript
// Controlled failure injection
await Task("Chaos Experiment", {
  hypothesis: 'System handles database latency gracefully',
  steadyState: {
    metric: 'error_rate',
    expected: '< 0.1%'
  },
  experiment: {
    type: 'network-latency',
    target: 'database',
    delay: '500ms',
    duration: '5m'
  },
  rollback: {
    automatic: true,
    trigger: 'error_rate > 5%'
  }
}, "qe-chaos-engineer");
```

---

## Production → Pre-Production Feedback Loop

```typescript
// Convert production incidents to regression tests
await Task("Incident Replay", {
  incident: {
    id: 'INC-2024-001',
    type: 'performance-degradation',
    conditions: { concurrent_users: 500, cart_items: 10 }
  },
  generateTests: true,
  addToRegression: true
}, "qe-production-intelligence");

// Output: New test added to prevent recurrence
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/shift-right/
├── canary-results/*      - Canary deployment metrics
├── synthetic-tests/*     - Monitoring configurations
├── chaos-experiments/*   - Experiment results
├── production-insights/* - Issues → test conversions
└── rum-analysis/*        - Real user data patterns
```

### Fleet Coordination
```typescript
const shiftRightFleet = await FleetManager.coordinate({
  strategy: 'shift-right-testing',
  agents: [
    'qe-production-intelligence',  // RUM, incident replay
    'qe-chaos-engineer',           // Resilience testing
    'qe-performance-tester',       // Synthetic monitoring
    'qe-quality-analyzer'          // Metrics analysis
  ],
  topology: 'mesh'
});
```

---

## Related Skills
- [shift-left-testing](../shift-left-testing/) - Pre-production testing
- [chaos-engineering-resilience](../chaos-engineering-resilience/) - Failure injection deep dive
- [performance-testing](../performance-testing/) - Load testing
- [agentic-quality-engineering](../agentic-quality-engineering/) - Agent coordination

---

## Remember

**Production is the ultimate test environment.** Feature flags enable instant rollback. Canary catches issues before 100% rollout. Synthetic monitoring detects problems before users. Chaos engineering builds resilience. RUM shows real user experience.

**With Agents:** Agents monitor production, replay incidents as tests, run chaos experiments, and convert production insights to pre-production tests. Use agents to maintain continuous production quality.
