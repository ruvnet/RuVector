---
name: performance-testing
description: "Test application performance, scalability, and resilience. Use when planning load testing, stress testing, or optimizing system performance."
category: specialized-testing
priority: high
tokenEstimate: 1100
agents: [qe-performance-tester, qe-quality-analyzer, qe-production-intelligence]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2025-12-02
dependencies: []
quick_reference_card: true
tags: [performance, load-testing, stress-testing, scalability, k6, bottlenecks]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/performance-testing.yaml
---

# Performance Testing

<default_to_action>
When testing performance or planning load tests:
1. DEFINE SLOs: p95 response time, throughput, error rate targets
2. IDENTIFY critical paths: revenue flows, high-traffic pages, key APIs
3. CREATE realistic scenarios: user journeys, think time, varied data
4. EXECUTE with monitoring: CPU, memory, DB queries, network
5. ANALYZE bottlenecks and fix before production

**Quick Test Type Selection:**
- Expected load validation → Load testing
- Find breaking point → Stress testing
- Sudden traffic spike → Spike testing
- Memory leaks, resource exhaustion → Endurance/soak testing
- Horizontal/vertical scaling → Scalability testing

**Critical Success Factors:**
- Performance is a feature, not an afterthought
- Test early and often, not just before release
- Focus on user-impacting bottlenecks
</default_to_action>

## Quick Reference Card

### When to Use
- Before major releases
- After infrastructure changes
- Before scaling events (Black Friday)
- When setting SLAs/SLOs

### Test Types
| Type | Purpose | When |
|------|---------|------|
| **Load** | Expected traffic | Every release |
| **Stress** | Beyond capacity | Quarterly |
| **Spike** | Sudden surge | Before events |
| **Endurance** | Memory leaks | After code changes |
| **Scalability** | Scaling validation | Infrastructure changes |

### Key Metrics
| Metric | Target | Why |
|--------|--------|-----|
| p95 response | < 200ms | User experience |
| Throughput | 10k req/min | Capacity |
| Error rate | < 0.1% | Reliability |
| CPU | < 70% | Headroom |
| Memory | < 80% | Stability |

### Tools
- **k6**: Modern, JS-based, CI/CD friendly
- **JMeter**: Enterprise, feature-rich
- **Artillery**: Simple YAML configs
- **Gatling**: Scala, great reporting

### Agent Coordination
- `qe-performance-tester`: Load test orchestration
- `qe-quality-analyzer`: Results analysis
- `qe-production-intelligence`: Production comparison

---

## Defining SLOs

**Bad:** "The system should be fast"
**Good:** "p95 response time < 200ms under 1,000 concurrent users"

```javascript
export const options = {
  thresholds: {
    http_req_duration: ['p(95)<200'],  // 95% < 200ms
    http_req_failed: ['rate<0.01'],     // < 1% failures
  },
};
```

---

## Realistic Scenarios

**Bad:** Every user hits homepage repeatedly
**Good:** Model actual user behavior

```javascript
// Realistic distribution
// 40% browse, 30% search, 20% details, 10% checkout
export default function () {
  const action = Math.random();
  if (action < 0.4) browse();
  else if (action < 0.7) search();
  else if (action < 0.9) viewProduct();
  else checkout();

  sleep(randomInt(1, 5)); // Think time
}
```

---

## Common Bottlenecks

### Database
**Symptoms:** Slow queries under load, connection pool exhaustion
**Fixes:** Add indexes, optimize N+1 queries, increase pool size, read replicas

### N+1 Queries
```javascript
// BAD: 100 orders = 101 queries
const orders = await Order.findAll();
for (const order of orders) {
  const customer = await Customer.findById(order.customerId);
}

// GOOD: 1 query
const orders = await Order.findAll({ include: [Customer] });
```

### Synchronous Processing
**Problem:** Blocking operations in request path (sending email during checkout)
**Fix:** Use message queues, process async, return immediately

### Memory Leaks
**Detection:** Endurance testing, memory profiling
**Common causes:** Event listeners not cleaned, caches without eviction

### External Dependencies
**Solutions:** Aggressive timeouts, circuit breakers, caching, graceful degradation

---

## k6 CI/CD Example

```javascript
// performance-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '1m', target: 50 },   // Ramp up
    { duration: '3m', target: 50 },   // Steady
    { duration: '1m', target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ['p(95)<200'],
    http_req_failed: ['rate<0.01'],
  },
};

export default function () {
  const res = http.get('https://api.example.com/products');
  check(res, {
    'status is 200': (r) => r.status === 200,
    'response time < 200ms': (r) => r.timings.duration < 200,
  });
  sleep(1);
}
```

```yaml
# GitHub Actions
- name: Run k6 test
  uses: grafana/k6-action@v0.3.0
  with:
    filename: performance-test.js
```

---

## Analyzing Results

### Good Results
```
Load: 1,000 users | p95: 180ms | Throughput: 5,000 req/s
Error rate: 0.05% | CPU: 65% | Memory: 70%
```

### Problems
```
Load: 1,000 users | p95: 3,500ms ❌ | Throughput: 500 req/s ❌
Error rate: 5% ❌ | CPU: 95% ❌ | Memory: 90% ❌
```

### Root Cause Analysis
1. Correlate metrics: When response time spikes, what changes?
2. Check logs: Errors, warnings, slow queries
3. Profile code: Where is time spent?
4. Monitor resources: CPU, memory, disk
5. Trace requests: End-to-end flow

---

## Anti-Patterns

| ❌ Anti-Pattern | ✅ Better |
|----------------|-----------|
| Testing too late | Test early and often |
| Unrealistic scenarios | Model real user behavior |
| 0 to 1000 users instantly | Ramp up gradually |
| No monitoring during tests | Monitor everything |
| No baseline | Establish and track trends |
| One-time testing | Continuous performance testing |

---

## Agent-Assisted Performance Testing

```typescript
// Comprehensive load test
await Task("Load Test", {
  target: 'https://api.example.com',
  scenarios: {
    checkout: { vus: 100, duration: '5m' },
    search: { vus: 200, duration: '5m' },
    browse: { vus: 500, duration: '5m' }
  },
  thresholds: {
    'http_req_duration': ['p(95)<200'],
    'http_req_failed': ['rate<0.01']
  }
}, "qe-performance-tester");

// Bottleneck analysis
await Task("Analyze Bottlenecks", {
  testResults: perfTest,
  metrics: ['cpu', 'memory', 'db_queries', 'network']
}, "qe-performance-tester");

// CI integration
await Task("CI Performance Gate", {
  mode: 'smoke',
  duration: '1m',
  vus: 10,
  failOn: { 'p95_response_time': 300, 'error_rate': 0.01 }
}, "qe-performance-tester");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/performance/
├── results/*       - Test execution results
├── baselines/*     - Performance baselines
├── bottlenecks/*   - Identified bottlenecks
└── trends/*        - Historical trends
```

### Fleet Coordination
```typescript
const perfFleet = await FleetManager.coordinate({
  strategy: 'performance-testing',
  agents: [
    'qe-performance-tester',
    'qe-quality-analyzer',
    'qe-production-intelligence',
    'qe-deployment-readiness'
  ],
  topology: 'sequential'
});
```

---

## Pre-Production Checklist

- [ ] Load test passed (expected traffic)
- [ ] Stress test passed (2-3x expected)
- [ ] Spike test passed (sudden surge)
- [ ] Endurance test passed (24+ hours)
- [ ] Database indexes in place
- [ ] Caching configured
- [ ] Monitoring and alerting set up
- [ ] Performance baseline established

---

## Related Skills
- [agentic-quality-engineering](../agentic-quality-engineering/) - Agent coordination
- [api-testing-patterns](../api-testing-patterns/) - API performance
- [chaos-engineering-resilience](../chaos-engineering-resilience/) - Resilience testing

---

## Remember

**Performance is a feature:** Test it like functionality
**Test continuously:** Not just before launch
**Monitor production:** Synthetic + real user monitoring
**Fix what matters:** Focus on user-impacting bottlenecks
**Trend over time:** Catch degradation early

**With Agents:** Agents automate load testing, analyze bottlenecks, and compare with production. Use agents to maintain performance at scale.
