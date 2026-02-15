---
name: observability-testing-patterns
description: "Observability and monitoring validation patterns for dashboards, alerting, log aggregation, APM traces, and SLA/SLO verification. Use when testing monitoring infrastructure, dashboard accuracy, alert rules, or metric pipelines."
category: specialized-testing
priority: high
tokenEstimate: 1600
agents: [qe-integration-tester, qe-performance-tester, qe-visual-tester]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2026-02-04
dependencies: [api-testing-patterns, shift-right-testing]
quick_reference_card: true
tags: [observability, monitoring, kibana, elasticsearch, dashboards, alerting, metrics, logging]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/observability-testing-patterns.yaml
---

# Observability Testing Patterns

<default_to_action>
When testing observability infrastructure, dashboards, or monitoring:
1. VALIDATE data accuracy (source data matches what the dashboard displays)
2. TEST alert rules fire correctly at defined thresholds
3. VERIFY log aggregation completeness (no missing logs across services)
4. TRACE distributed requests end-to-end through APM
5. MEASURE dashboard performance (render time, query latency)
6. CONFIRM SLA/SLO compliance through synthetic monitoring
7. TEST metric pipeline integrity from collection to display

**Quick Pattern Selection:**
- Dashboard shows wrong numbers -> Data accuracy validation
- Alerts not firing -> Alert rule threshold testing
- Missing logs in Kibana -> Log aggregation completeness
- Slow dashboard -> Dashboard performance testing
- Broken traces -> APM trace validation
- SLA disputes -> SLO compliance validation

**Critical Success Factors:**
- Observability is only as good as the data it shows
- A dashboard that lies is worse than no dashboard
- Alert fatigue kills response times; test thresholds carefully
</default_to_action>

## Quick Reference Card

### When to Use
- Validating dashboard data accuracy (Kibana, Grafana, Datadog)
- Testing alert rule thresholds and notification delivery
- Verifying log aggregation completeness across microservices
- Validating distributed tracing (APM) correctness
- Measuring SLA/SLO compliance
- Testing metric pipeline integrity (collection -> aggregation -> display)

### Testing Levels
| Level | Purpose | Dependencies | Speed |
|-------|---------|--------------|-------|
| Query Validation | Elasticsearch/PromQL query accuracy | Data source | Fast |
| Dashboard Accuracy | Visual matches source data | Full stack | Medium |
| Alert Threshold | Trigger and notification testing | Alerting stack | Medium |
| Pipeline Integrity | End-to-end metric flow | Full pipeline | Slower |
| Performance | Dashboard render time, query latency | Full stack | Slower |

### Critical Test Scenarios
| Scenario | Must Test | Example |
|----------|----------|---------|
| Data Accuracy | Dashboard = source truth | Order count on dashboard = DB count |
| Alert Firing | Threshold triggers alert | Error rate > 5% fires PagerDuty |
| Alert Recovery | Auto-resolve when recovered | Error rate drops below 5% clears alert |
| Log Completeness | All services emit logs | 10 microservices, all logs in Kibana |
| Trace Integrity | Full request path visible | Auth -> API -> DB -> Cache spans |
| SLO Compliance | Error budget tracking | 99.9% availability over 30 days |
| Time Accuracy | Timestamps aligned | Log timestamp matches event time |

### Tools
- **Dashboards**: Kibana, Grafana, Datadog, New Relic
- **Search**: Elasticsearch, OpenSearch, Loki
- **Metrics**: Prometheus, InfluxDB, CloudWatch
- **Tracing**: Jaeger, Zipkin, Datadog APM, OpenTelemetry
- **Alerting**: PagerDuty, OpsGenie, Alertmanager
- **Synthetic**: Datadog Synthetics, Checkly, Playwright

### Agent Coordination
- `qe-integration-tester`: Validate data pipelines, query accuracy, log completeness
- `qe-performance-tester`: Dashboard render performance, query latency
- `qe-visual-tester`: Dashboard visual regression, layout accuracy

---

## Dashboard Data Accuracy Validation

### Compare Source Data to Dashboard
```javascript
describe('Dashboard Data Accuracy', () => {
  it('order count on dashboard matches database', async () => {
    // Step 1: Get ground truth from source database
    const dbResult = await db.query(
      "SELECT COUNT(*) as count FROM orders WHERE created_at >= NOW() - INTERVAL '24 HOURS'"
    );
    const dbCount = parseInt(dbResult.rows[0].count);

    // Step 2: Query Elasticsearch (same data source as dashboard)
    const esResult = await esClient.search({
      index: 'orders-*',
      body: {
        query: {
          range: { created_at: { gte: 'now-24h' } }
        },
        size: 0,
        track_total_hits: true
      }
    });
    const esCount = esResult.hits.total.value;

    // Step 3: Compare
    expect(esCount).toBe(dbCount);
  });

  it('revenue metric on dashboard matches transaction totals', async () => {
    const dbRevenue = await db.query(
      "SELECT SUM(total) as revenue FROM orders WHERE status = 'COMPLETED' AND created_at >= NOW() - INTERVAL '24 HOURS'"
    );
    const expectedRevenue = parseFloat(dbRevenue.rows[0].revenue);

    const esResult = await esClient.search({
      index: 'orders-*',
      body: {
        query: {
          bool: {
            must: [
              { term: { status: 'COMPLETED' } },
              { range: { created_at: { gte: 'now-24h' } } }
            ]
          }
        },
        aggs: {
          total_revenue: { sum: { field: 'total' } }
        },
        size: 0
      }
    });
    const dashboardRevenue = esResult.aggregations.total_revenue.value;

    // Allow small floating point tolerance
    expect(Math.abs(dashboardRevenue - expectedRevenue)).toBeLessThan(0.01);
  });

  it('error rate percentage is calculated correctly', async () => {
    const esResult = await esClient.search({
      index: 'logs-*',
      body: {
        query: { range: { '@timestamp': { gte: 'now-1h' } } },
        aggs: {
          total: { value_count: { field: 'status_code' } },
          errors: {
            filter: { range: { status_code: { gte: 500 } } },
            aggs: { count: { value_count: { field: 'status_code' } } }
          }
        },
        size: 0
      }
    });

    const total = esResult.aggregations.total.value;
    const errors = esResult.aggregations.errors.count.value;
    const expectedErrorRate = (errors / total) * 100;

    // Fetch what the dashboard shows via Kibana API
    const dashboardPanel = await kibanaApi.get('/api/saved_objects/visualization/error-rate-gauge');
    const displayedErrorRate = await evaluateKibanaVisualization(dashboardPanel);

    expect(Math.abs(displayedErrorRate - expectedErrorRate)).toBeLessThan(0.1);
  });
});
```

---

## Elasticsearch Query Result Validation

```javascript
describe('Elasticsearch Query Validation', () => {
  it('validates date histogram aggregation returns correct buckets', async () => {
    // Insert known test data
    const testDocs = [];
    for (let hour = 0; hour < 24; hour++) {
      const timestamp = new Date();
      timestamp.setHours(hour, 0, 0, 0);
      testDocs.push({
        '@timestamp': timestamp.toISOString(),
        service: 'order-api',
        status_code: hour % 5 === 0 ? 500 : 200,
        response_time: 100 + (hour * 10)
      });
    }

    await esClient.bulk({
      index: 'test-logs',
      body: testDocs.flatMap(doc => [{ index: {} }, doc])
    });
    await esClient.indices.refresh({ index: 'test-logs' });

    // Run the same query the dashboard uses
    const result = await esClient.search({
      index: 'test-logs',
      body: {
        query: { match_all: {} },
        aggs: {
          requests_over_time: {
            date_histogram: { field: '@timestamp', fixed_interval: '1h' },
            aggs: {
              avg_response: { avg: { field: 'response_time' } },
              error_count: {
                filter: { range: { status_code: { gte: 500 } } }
              }
            }
          }
        },
        size: 0
      }
    });

    const buckets = result.aggregations.requests_over_time.buckets;
    expect(buckets.length).toBe(24);

    // Verify specific bucket values
    const errorBuckets = buckets.filter(b => b.error_count.doc_count > 0);
    expect(errorBuckets.length).toBe(5); // Hours 0, 5, 10, 15, 20
  });

  it('validates term aggregation for top services', async () => {
    const result = await esClient.search({
      index: 'logs-*',
      body: {
        query: { range: { '@timestamp': { gte: 'now-1h' } } },
        aggs: {
          top_services: {
            terms: { field: 'service.keyword', size: 10 }
          }
        },
        size: 0
      }
    });

    const services = result.aggregations.top_services.buckets;
    expect(services.length).toBeGreaterThan(0);

    // Each bucket should have reasonable doc counts
    for (const bucket of services) {
      expect(bucket.key).toBeDefined();
      expect(bucket.doc_count).toBeGreaterThan(0);
    }
  });
});
```

---

## Kibana Dashboard Element Assertions

```javascript
describe('Kibana Dashboard Visual Validation', () => {
  it('validates dashboard panels render without errors', async () => {
    await page.goto(`${kibanaUrl}/app/dashboards#/view/operations-overview`);

    // Wait for all panels to finish loading
    await page.waitForSelector('.embPanel__content', { state: 'visible' });
    await page.waitForFunction(() => {
      const loaders = document.querySelectorAll('.euiLoadingSpinner');
      return loaders.length === 0;
    }, { timeout: 30000 });

    // Check no error icons on any panel
    const errorPanels = await page.locator('.embPanel--error').count();
    expect(errorPanels).toBe(0);

    // Check no "No results found" where data is expected
    const noResultPanels = await page.locator('text="No results found"').count();
    expect(noResultPanels).toBe(0);
  });

  it('validates metric visualization shows correct value', async () => {
    await page.goto(`${kibanaUrl}/app/dashboards#/view/operations-overview`);
    await page.waitForLoadState('networkidle');

    // Get the displayed metric value
    const metricValue = await page.locator('[data-test-subj="metricVis-total-orders"] .mtrVis__value').textContent();
    const displayedCount = parseInt(metricValue.replace(/,/g, ''));

    // Compare with direct ES query
    const esResult = await esClient.count({ index: 'orders-*' });

    expect(displayedCount).toBe(esResult.count);
  });

  it('validates table visualization columns and sorting', async () => {
    await page.goto(`${kibanaUrl}/app/dashboards#/view/operations-overview`);
    await page.waitForLoadState('networkidle');

    // Verify expected columns exist
    const headers = await page.locator('.euiTable th').allTextContents();
    expect(headers).toContain('Service');
    expect(headers).toContain('Error Rate');
    expect(headers).toContain('P95 Latency');

    // Verify sorting works
    await page.click('th:has-text("Error Rate")');
    const firstRow = await page.locator('.euiTable tbody tr:first-child td').allTextContents();
    const secondRow = await page.locator('.euiTable tbody tr:nth-child(2) td').allTextContents();

    const firstErrorRate = parseFloat(firstRow[1]);
    const secondErrorRate = parseFloat(secondRow[1]);
    expect(firstErrorRate).toBeGreaterThanOrEqual(secondErrorRate);
  });
});
```

---

## Alert Rule Testing

```javascript
describe('Alert Rule Validation', () => {
  it('fires alert when error rate exceeds threshold', async () => {
    // Generate errors to exceed the 5% threshold
    const requests = [];
    for (let i = 0; i < 100; i++) {
      requests.push({
        '@timestamp': new Date().toISOString(),
        service: 'payment-api',
        status_code: i < 10 ? 500 : 200, // 10% error rate > 5% threshold
        response_time: 200
      });
    }

    await esClient.bulk({
      index: 'logs-payment',
      body: requests.flatMap(doc => [{ index: {} }, doc])
    });
    await esClient.indices.refresh({ index: 'logs-payment' });

    // Wait for alert evaluation cycle (typically 1 minute)
    await sleep(90000);

    // Check alert was fired
    const alerts = await alertManager.getActiveAlerts({
      filter: 'alertname="HighErrorRate" AND service="payment-api"'
    });
    expect(alerts.length).toBeGreaterThan(0);
    expect(alerts[0].labels.severity).toBe('critical');
  });

  it('alert auto-resolves when condition clears', async () => {
    // First trigger the alert
    await injectErrors('payment-api', { count: 50, total: 100 });
    await sleep(90000);

    let alerts = await alertManager.getActiveAlerts({ filter: 'alertname="HighErrorRate"' });
    expect(alerts.length).toBeGreaterThan(0);

    // Now inject healthy traffic to bring error rate below threshold
    await injectSuccessRequests('payment-api', { count: 1000 });
    await sleep(90000);

    // Alert should auto-resolve
    alerts = await alertManager.getActiveAlerts({ filter: 'alertname="HighErrorRate"' });
    expect(alerts.length).toBe(0);
  });

  it('alert notification reaches correct channel', async () => {
    // Subscribe to notification channel
    const notifications = [];
    const subscription = pagerDutyMock.onIncident((incident) => {
      notifications.push(incident);
    });

    // Trigger alert condition
    await injectErrors('critical-service', { count: 50, total: 100 });
    await sleep(120000);

    expect(notifications.length).toBeGreaterThan(0);
    expect(notifications[0].service.name).toBe('critical-service');
    expect(notifications[0].urgency).toBe('high');

    subscription.unsubscribe();
  });

  it('alert does not fire for brief transient spikes', async () => {
    // Inject a brief 30-second spike (alert requires 5 minutes sustained)
    await injectErrors('api-service', { count: 20, total: 50, duration: 30000 });
    await sleep(120000);

    const alerts = await alertManager.getActiveAlerts({ filter: 'alertname="HighErrorRate"' });
    expect(alerts.length).toBe(0); // Should NOT fire for transient spike
  });
});
```

---

## Log Aggregation Completeness

```javascript
describe('Log Aggregation Completeness', () => {
  it('all microservice logs appear in centralized index', async () => {
    const traceId = uuid();
    const services = ['api-gateway', 'auth-service', 'order-service', 'payment-service', 'notification-service'];

    // Generate a log entry with known traceId in each service
    for (const service of services) {
      await serviceLogEmitter.emit(service, {
        level: 'INFO',
        message: `Completeness test - ${traceId}`,
        traceId,
        timestamp: new Date().toISOString()
      });
    }

    // Wait for log pipeline to process (Filebeat -> Logstash -> Elasticsearch)
    await sleep(15000);

    // Query Elasticsearch for the trace ID
    const result = await esClient.search({
      index: 'logs-*',
      body: {
        query: { term: { 'traceId.keyword': traceId } },
        size: 100
      }
    });

    const foundServices = result.hits.hits.map(h => h._source.service);

    // All services should have their log entry in Elasticsearch
    for (const service of services) {
      expect(foundServices).toContain(service);
    }
    expect(foundServices.length).toBe(services.length);
  });

  it('logs retain correct structure after pipeline processing', async () => {
    const testLog = {
      level: 'ERROR',
      message: 'Payment declined',
      traceId: uuid(),
      userId: 'user-123',
      orderId: 'order-456',
      errorCode: 'INSUFFICIENT_FUNDS',
      timestamp: new Date().toISOString()
    };

    await serviceLogEmitter.emit('payment-service', testLog);
    await sleep(10000);

    const result = await esClient.search({
      index: 'logs-*',
      body: { query: { term: { 'traceId.keyword': testLog.traceId } } }
    });

    expect(result.hits.hits.length).toBe(1);
    const indexed = result.hits.hits[0]._source;

    // Verify all fields survived the pipeline
    expect(indexed.level).toBe('ERROR');
    expect(indexed.message).toBe('Payment declined');
    expect(indexed.userId).toBe('user-123');
    expect(indexed.orderId).toBe('order-456');
    expect(indexed.errorCode).toBe('INSUFFICIENT_FUNDS');
  });

  it('detects log volume drops indicating pipeline issues', async () => {
    // Get baseline log volume for the past hour
    const baseline = await esClient.count({
      index: 'logs-*',
      body: { query: { range: { '@timestamp': { gte: 'now-2h', lt: 'now-1h' } } } }
    });

    const current = await esClient.count({
      index: 'logs-*',
      body: { query: { range: { '@timestamp': { gte: 'now-1h' } } } }
    });

    // Current volume should be at least 50% of baseline (not a sudden drop)
    const ratio = current.count / baseline.count;
    expect(ratio).toBeGreaterThan(0.5);
  });
});
```

---

## APM Trace Validation

```javascript
describe('Distributed Trace Validation', () => {
  it('captures complete trace across all services', async () => {
    // Make a request that traverses multiple services
    const response = await httpClient.post('/api/orders', {
      customerId: 'CUST-TRACE',
      items: [{ sku: 'ITEM-1', qty: 1 }]
    });

    const traceId = response.headers['x-trace-id'];
    expect(traceId).toBeDefined();

    // Wait for trace to be indexed
    await sleep(10000);

    // Query Jaeger/APM for the trace
    const trace = await jaegerClient.getTrace(traceId);

    // Verify all expected spans exist
    const spanNames = trace.spans.map(s => s.operationName);
    expect(spanNames).toContain('POST /api/orders');
    expect(spanNames).toContain('auth.validateToken');
    expect(spanNames).toContain('order.create');
    expect(spanNames).toContain('payment.authorize');
    expect(spanNames).toContain('inventory.reserve');
    expect(spanNames).toContain('db.insert orders');

    // Verify parent-child relationships
    const apiSpan = trace.spans.find(s => s.operationName === 'POST /api/orders');
    const authSpan = trace.spans.find(s => s.operationName === 'auth.validateToken');
    expect(authSpan.references[0].refType).toBe('CHILD_OF');
    expect(authSpan.references[0].spanID).toBe(apiSpan.spanID);
  });

  it('traces capture error spans correctly', async () => {
    // Trigger a known error
    const response = await httpClient.post('/api/orders', {
      customerId: 'INVALID-CUSTOMER',
      items: [{ sku: 'ITEM-1', qty: 1 }]
    });

    const traceId = response.headers['x-trace-id'];
    await sleep(10000);

    const trace = await jaegerClient.getTrace(traceId);

    // Find error span
    const errorSpan = trace.spans.find(s => s.tags.some(t => t.key === 'error' && t.value === true));
    expect(errorSpan).toBeDefined();
    expect(errorSpan.logs).toContainEqual(
      expect.objectContaining({
        fields: expect.arrayContaining([
          expect.objectContaining({ key: 'error.message' })
        ])
      })
    );
  });

  it('validates trace sampling rate', async () => {
    const requestCount = 100;
    const traceIds = [];

    for (let i = 0; i < requestCount; i++) {
      const resp = await httpClient.get('/api/health');
      if (resp.headers['x-trace-id']) {
        traceIds.push(resp.headers['x-trace-id']);
      }
    }

    await sleep(15000);

    let tracesFound = 0;
    for (const traceId of traceIds) {
      try {
        await jaegerClient.getTrace(traceId);
        tracesFound++;
      } catch (e) {
        // Trace not sampled
      }
    }

    // With 10% sampling rate, expect roughly 10 traces (allow variance)
    const samplingRate = tracesFound / requestCount;
    expect(samplingRate).toBeGreaterThan(0.05);
    expect(samplingRate).toBeLessThan(0.20);
  });
});
```

---

## SLA/SLO Validation

```javascript
describe('SLA/SLO Compliance Validation', () => {
  it('validates 99.9% availability SLO over 30 days', async () => {
    const result = await prometheusClient.query(
      'avg_over_time(up{job="api-service"}[30d])'
    );
    const availability = parseFloat(result.data.result[0].value[1]) * 100;

    expect(availability).toBeGreaterThanOrEqual(99.9);

    // Calculate error budget remaining
    const totalMinutes = 30 * 24 * 60;
    const allowedDowntime = totalMinutes * 0.001; // 43.2 minutes
    const actualDowntime = totalMinutes * (1 - availability / 100);
    const errorBudgetRemaining = ((allowedDowntime - actualDowntime) / allowedDowntime) * 100;

    expect(errorBudgetRemaining).toBeGreaterThan(0);
    console.log(`Error budget remaining: ${errorBudgetRemaining.toFixed(1)}%`);
  });

  it('validates P95 latency SLO', async () => {
    const result = await prometheusClient.query(
      'histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket{service="api-service"}[24h])) by (le))'
    );
    const p95Latency = parseFloat(result.data.result[0].value[1]) * 1000; // Convert to ms

    expect(p95Latency).toBeLessThan(500); // SLO: P95 < 500ms
  });

  it('runs synthetic monitoring check for uptime', async () => {
    const endpoints = [
      { url: '/api/health', expectedStatus: 200, maxLatency: 200 },
      { url: '/api/orders', expectedStatus: 401, maxLatency: 300 }, // Auth required
      { url: '/api/products', expectedStatus: 200, maxLatency: 500 }
    ];

    const results = [];
    for (const endpoint of endpoints) {
      const start = Date.now();
      const response = await httpClient.get(endpoint.url);
      const latency = Date.now() - start;

      results.push({
        url: endpoint.url,
        status: response.status,
        latency,
        statusMatch: response.status === endpoint.expectedStatus,
        latencyOk: latency <= endpoint.maxLatency
      });
    }

    // All checks should pass
    for (const result of results) {
      expect(result.statusMatch).toBe(true);
      expect(result.latencyOk).toBe(true);
    }
  });
});
```

---

## Metric Pipeline Integrity

```javascript
describe('Metric Pipeline - Collection to Display', () => {
  it('validates custom metric flows from app to Prometheus to Grafana', async () => {
    // Step 1: Emit a known custom metric from application
    const metricName = 'test_orders_processed_total';
    const expectedValue = 42;
    await appMetrics.set(metricName, expectedValue, { service: 'test' });

    // Step 2: Wait for scrape interval (15s default)
    await sleep(20000);

    // Step 3: Query Prometheus directly
    const promResult = await prometheusClient.query(`${metricName}{service="test"}`);
    const promValue = parseFloat(promResult.data.result[0].value[1]);
    expect(promValue).toBe(expectedValue);

    // Step 4: Query Grafana datasource API (same as dashboard would)
    const grafanaResult = await grafanaApi.post('/api/ds/query', {
      queries: [{
        datasource: { type: 'prometheus' },
        expr: `${metricName}{service="test"}`,
        refId: 'A'
      }]
    });

    const grafanaValue = parseFloat(grafanaResult.body.results.A.frames[0].data.values[1][0]);
    expect(grafanaValue).toBe(expectedValue);
  });

  it('validates histogram metric percentile accuracy', async () => {
    // Generate known latency distribution
    const latencies = [10, 20, 30, 50, 100, 200, 300, 500, 1000, 2000]; // ms
    for (const latency of latencies) {
      await appMetrics.observe('http_request_duration_ms', latency, { endpoint: '/test' });
    }

    await sleep(20000);

    // Verify P50 and P99
    const p50 = await prometheusClient.query(
      'histogram_quantile(0.5, rate(http_request_duration_ms_bucket{endpoint="/test"}[5m]))'
    );
    const p99 = await prometheusClient.query(
      'histogram_quantile(0.99, rate(http_request_duration_ms_bucket{endpoint="/test"}[5m]))'
    );

    const p50Value = parseFloat(p50.data.result[0].value[1]);
    const p99Value = parseFloat(p99.data.result[0].value[1]);

    // P50 should be around 100ms (median of our distribution)
    expect(p50Value).toBeGreaterThan(50);
    expect(p50Value).toBeLessThan(300);

    // P99 should be around 2000ms
    expect(p99Value).toBeGreaterThan(1000);
  });
});
```

---

## Dashboard Performance Testing

```javascript
describe('Dashboard Performance', () => {
  it('dashboard loads within acceptable time', async () => {
    const start = Date.now();

    await page.goto(`${kibanaUrl}/app/dashboards#/view/operations-overview`);

    // Wait for all panels to finish loading
    await page.waitForFunction(() => {
      const spinners = document.querySelectorAll('.euiLoadingSpinner');
      return spinners.length === 0;
    }, { timeout: 30000 });

    const loadTime = Date.now() - start;

    expect(loadTime).toBeLessThan(10000); // Dashboard should load in under 10s
  });

  it('dashboard handles large time range without timeout', async () => {
    // Set time range to 30 days
    await page.goto(`${kibanaUrl}/app/dashboards#/view/operations-overview?_g=(time:(from:now-30d,to:now))`);

    // Should complete without timeout error
    await page.waitForFunction(() => {
      const errors = document.querySelectorAll('.embPanel--error');
      const spinners = document.querySelectorAll('.euiLoadingSpinner');
      return errors.length === 0 && spinners.length === 0;
    }, { timeout: 60000 });

    const errorPanels = await page.locator('.embPanel--error').count();
    expect(errorPanels).toBe(0);
  });

  it('Elasticsearch query performance is within bounds', async () => {
    const queries = [
      { name: 'date_histogram', body: { aggs: { over_time: { date_histogram: { field: '@timestamp', fixed_interval: '1h' } } }, size: 0 } },
      { name: 'terms_agg', body: { aggs: { top_services: { terms: { field: 'service.keyword', size: 20 } } }, size: 0 } },
      { name: 'percentiles', body: { aggs: { latency: { percentiles: { field: 'response_time', percents: [50, 90, 95, 99] } } }, size: 0 } }
    ];

    for (const query of queries) {
      const start = Date.now();
      await esClient.search({ index: 'logs-*', body: { query: { range: { '@timestamp': { gte: 'now-24h' } } }, ...query.body } });
      const elapsed = Date.now() - start;

      expect(elapsed).toBeLessThan(5000); // Each query under 5s
    }
  });
});
```

---

## Time-Series Data Accuracy

```javascript
describe('Time-Series Data Accuracy', () => {
  it('validates no data gaps in time-series metrics', async () => {
    const result = await prometheusClient.queryRange(
      'up{job="api-service"}',
      { start: 'now-24h', end: 'now', step: '5m' }
    );

    const values = result.data.result[0].values;
    const expectedPoints = (24 * 60) / 5; // 288 data points for 24h at 5m intervals

    // Allow up to 5% missing data points
    expect(values.length).toBeGreaterThan(expectedPoints * 0.95);

    // Check for gaps longer than 15 minutes (3 consecutive missing points)
    for (let i = 1; i < values.length; i++) {
      const gap = values[i][0] - values[i - 1][0]; // timestamp difference
      expect(gap).toBeLessThanOrEqual(900); // No gap > 15 minutes
    }
  });

  it('validates timestamp alignment across sources', async () => {
    // Generate event with precise timestamp
    const eventTime = new Date();
    const traceId = uuid();

    await httpClient.post('/api/test-event', { traceId });
    await sleep(15000);

    // Check timestamp in logs
    const logResult = await esClient.search({
      index: 'logs-*',
      body: { query: { term: { 'traceId.keyword': traceId } } }
    });
    const logTimestamp = new Date(logResult.hits.hits[0]._source['@timestamp']);

    // Check timestamp in metrics (approximate)
    // Timestamps should be within 5 seconds of each other
    const diff = Math.abs(logTimestamp.getTime() - eventTime.getTime());
    expect(diff).toBeLessThan(5000);
  });
});
```

---

## Best Practices

### Do This
- Compare dashboard values against source-of-truth databases
- Test alert thresholds with known data to verify exact firing conditions
- Validate log completeness by injecting traceable test events
- Test alert recovery (auto-resolve) not just alert firing
- Monitor log volume as a proxy for pipeline health
- Validate sampling rates for APM traces
- Test dashboards at realistic time ranges (not just last 15 minutes)

### Avoid This
- Trusting dashboard numbers without source validation
- Testing alerts only with manual threshold checks in the UI
- Ignoring log pipeline latency (logs may take seconds to minutes to appear)
- Skipping alert fatigue testing (too many false positives)
- Assuming metrics are accurate without end-to-end pipeline validation
- Testing only with small data volumes (performance issues appear at scale)
- Forgetting to test alert notification delivery (PagerDuty, Slack, email)

---

## Agent-Assisted Observability Testing

```typescript
// Data accuracy validation
await Task("Dashboard Data Accuracy Validation", {
  dashboard: 'operations-overview',
  panels: ['order-count', 'revenue-total', 'error-rate'],
  sourceDatabase: 'orders-db',
  compareFields: ['count', 'sum(total)', 'error_percentage'],
  tolerance: 0.01
}, "qe-integration-tester");

// Dashboard performance testing
await Task("Dashboard Performance Benchmark", {
  dashboardUrl: 'http://kibana:5601/app/dashboards#/view/operations-overview',
  timeRanges: ['15m', '1h', '24h', '7d', '30d'],
  maxLoadTime: 10000,
  maxQueryTime: 5000,
  captureScreenshots: true
}, "qe-performance-tester");

// Dashboard visual regression
await Task("Dashboard Visual Regression", {
  dashboardUrl: 'http://kibana:5601/app/dashboards#/view/operations-overview',
  baselineScreenshots: 'baseline/dashboards/',
  threshold: 0.05,
  ignoreRegions: ['timestamp-header', 'dynamic-counters']
}, "qe-visual-tester");

// Alert rule validation
await Task("Alert Rule Comprehensive Test", {
  alertRules: ['HighErrorRate', 'HighLatency', 'ServiceDown'],
  testFiring: true,
  testRecovery: true,
  testNotificationChannel: true,
  validateSilencing: true
}, "qe-integration-tester");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/observability-testing/
  dashboards/          - Dashboard test results and screenshots
  alerts/              - Alert rule test outcomes
  logs/                - Log completeness validation results
  traces/              - APM trace validation results
  slo/                 - SLA/SLO compliance metrics
  pipelines/           - Metric pipeline integrity checks
  performance/         - Dashboard and query performance benchmarks
```

### Fleet Coordination
```typescript
const observabilityFleet = await FleetManager.coordinate({
  strategy: 'observability-testing',
  agents: [
    'qe-integration-tester',  // Data accuracy, log completeness, alert rules
    'qe-performance-tester',  // Dashboard load time, query latency
    'qe-visual-tester'        // Dashboard visual regression
  ],
  topology: 'mesh'
});

await observabilityFleet.execute({
  targets: [
    { type: 'dashboard', id: 'operations-overview', checks: ['accuracy', 'performance', 'visual'] },
    { type: 'alerts', rules: ['HighErrorRate', 'HighLatency'], checks: ['fire', 'resolve', 'notify'] },
    { type: 'logs', services: ['api', 'auth', 'payment'], checks: ['completeness', 'structure'] },
    { type: 'traces', endpoints: ['/api/orders'], checks: ['spans', 'errors', 'sampling'] }
  ]
});
```

---

## Related Skills
- [api-testing-patterns](../api-testing-patterns/) - API testing for services feeding observability
- [shift-right-testing](../shift-right-testing/) - Production monitoring and observability
- [performance-testing](../performance-testing/) - Performance baselines and SLO definition
- [chaos-engineering-resilience](../chaos-engineering-resilience/) - Fault injection to validate alerting
- [visual-testing-advanced](../visual-testing-advanced/) - Dashboard visual regression

---

## Remember

Observability testing is about proving that your monitoring tells the truth. A dashboard that shows green when the system is on fire is worse than no dashboard at all. Validate data accuracy by comparing against source databases, test alert thresholds with controlled data injection, and verify log completeness by tracing known events through the entire pipeline.

**With Agents:** Agents automate the tedious comparison of dashboard values against source databases, systematically test alert thresholds with synthetic load, and validate log pipeline completeness across all services. Use agents to continuously verify that your observability stack is trustworthy.
