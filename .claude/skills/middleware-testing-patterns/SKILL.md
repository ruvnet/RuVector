---
name: middleware-testing-patterns
description: "Enterprise middleware testing patterns for message routing, transformation, DLQ, protocol mediation, ESB error handling, and EIP patterns. Use when testing middleware layers, message brokers, ESBs, or integration buses."
category: enterprise-integration
priority: high
tokenEstimate: 1800
agents: [qe-middleware-validator, qe-message-broker-tester, qe-soap-tester]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2026-02-04
dependencies: [api-testing-patterns, contract-testing]
quick_reference_card: true
tags: [middleware, esb, soap, messaging, iib, mq, transformation, routing]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/middleware-testing-patterns.yaml
---

# Middleware Testing Patterns

<default_to_action>
When testing enterprise middleware, ESBs, or message-driven systems:
1. VALIDATE message routing logic (content-based, header-based, recipient list)
2. TEST message transformations end-to-end (input format -> output format)
3. VERIFY dead letter queue handling (poison messages, retry exhaustion)
4. ASSERT message ordering and sequencing with correlation IDs
5. EXERCISE error handling and compensation patterns
6. TEST protocol mediation (SOAP-to-REST, sync-to-async)
7. VALIDATE EIP patterns (splitter, aggregator, content enricher, normalizer)

**Quick Pattern Selection:**
- Message routing issues -> Content-based router tests
- Transformation failures -> Schema-in/schema-out validation
- Lost messages -> DLQ and retry pattern tests
- Protocol bridging -> Mediation round-trip tests
- Complex flows -> Correlation ID tracing tests

**Critical Success Factors:**
- Middleware is invisible when it works; test the invisible
- Always validate both the happy path AND the error channel (DLQ)
- Correlation IDs are your lifeline for tracing multi-hop messages
</default_to_action>

## Quick Reference Card

### When to Use
- Testing ESB message flows (IBM IIB/ACE, MuleSoft, WSO2)
- Validating message transformations (XSLT, JSON-to-XML, flat-file parsing)
- Testing message broker routing (MQ, Kafka, RabbitMQ)
- Verifying dead letter queue behavior
- Testing protocol mediation (SOAP/REST bridging)
- Validating Enterprise Integration Patterns (EIP)

### Testing Levels
| Level | Purpose | Dependencies | Speed |
|-------|---------|--------------|-------|
| Unit Transform | Single mapping correctness | None | Fast |
| Route Logic | Routing decision accuracy | Mocked endpoints | Fast |
| Integration | End-to-end message flow | Broker + endpoints | Medium |
| DLQ/Error | Error handling and recovery | Full middleware stack | Slower |

### Critical Test Scenarios
| Scenario | Must Test | Example |
|----------|----------|---------|
| Routing | Correct destination selection | Order type A -> Queue A, type B -> Queue B |
| Transformation | Schema compliance after mapping | XML -> JSON field mapping accuracy |
| DLQ | Poison message handling | Malformed XML lands in DLQ, not lost |
| Ordering | Sequence preservation | Messages 1-2-3 arrive in order |
| Correlation | Multi-hop tracing | Request-reply matched by correlation ID |
| Retry | Transient failure recovery | 3 retries with backoff, then DLQ |
| Mediation | Protocol bridging fidelity | SOAP request produces correct REST call |

### Tools
- **Message Brokers**: IBM MQ, RabbitMQ, Apache Kafka, ActiveMQ
- **ESBs**: IBM IIB/ACE, MuleSoft, WSO2, Apache Camel
- **Testing**: SoapUI, Postman, custom harnesses
- **Virtualization**: Mountebank, WireMock, HoverFly
- **Monitoring**: Splunk, ELK, Datadog

### Agent Coordination
- `qe-middleware-validator`: Validate routing rules, transformation accuracy, EIP patterns
- `qe-message-broker-tester`: Test broker connectivity, DLQ behavior, message ordering
- `qe-soap-tester`: SOAP/WSDL validation, WS-Security, protocol mediation

---

## Message Routing Pattern Testing

### Content-Based Router
```javascript
describe('Content-Based Router - Order Type', () => {
  it('routes standard orders to fulfillment queue', async () => {
    const message = {
      correlationId: uuid(),
      body: { orderType: 'STANDARD', orderId: 'ORD-001', items: [{ sku: 'A1', qty: 2 }] }
    };

    await broker.publish('orders.inbound', message);

    const routed = await broker.consume('orders.fulfillment', { timeout: 5000 });
    expect(routed.body.orderId).toBe('ORD-001');
    expect(routed.correlationId).toBe(message.correlationId);

    const dlq = await broker.tryConsume('orders.dlq', { timeout: 1000 });
    expect(dlq).toBeNull(); // Nothing in DLQ
  });

  it('routes express orders to priority queue', async () => {
    const message = {
      correlationId: uuid(),
      body: { orderType: 'EXPRESS', orderId: 'ORD-002', items: [{ sku: 'B1', qty: 1 }] }
    };

    await broker.publish('orders.inbound', message);

    const routed = await broker.consume('orders.priority', { timeout: 5000 });
    expect(routed.body.orderId).toBe('ORD-002');
  });

  it('sends unrecognized order types to DLQ', async () => {
    const message = {
      correlationId: uuid(),
      body: { orderType: 'UNKNOWN', orderId: 'ORD-003' }
    };

    await broker.publish('orders.inbound', message);

    const dlqMessage = await broker.consume('orders.dlq', { timeout: 5000 });
    expect(dlqMessage.body.orderId).toBe('ORD-003');
    expect(dlqMessage.headers['x-error-reason']).toContain('Unrecognized orderType');
  });
});
```

### Header-Based Router
```javascript
describe('Header-Based Router - Region', () => {
  it('routes by x-region header to regional queues', async () => {
    const regions = ['US', 'EU', 'APAC'];

    for (const region of regions) {
      const message = {
        headers: { 'x-region': region },
        body: { orderId: `ORD-${region}` }
      };

      await broker.publish('orders.global', message);

      const routed = await broker.consume(`orders.${region.toLowerCase()}`, { timeout: 5000 });
      expect(routed.body.orderId).toBe(`ORD-${region}`);
    }
  });

  it('routes messages without region header to default queue', async () => {
    await broker.publish('orders.global', { body: { orderId: 'ORD-NO-REGION' } });

    const routed = await broker.consume('orders.default', { timeout: 5000 });
    expect(routed.body.orderId).toBe('ORD-NO-REGION');
  });
});
```

### Recipient List
```javascript
describe('Recipient List - Order Notifications', () => {
  it('fans out order to all registered recipients', async () => {
    const message = {
      body: { orderId: 'ORD-100', total: 250.00 }
    };

    await broker.publish('orders.confirmed', message);

    // All three systems should receive the message
    const [warehouse, billing, crm] = await Promise.all([
      broker.consume('notify.warehouse', { timeout: 5000 }),
      broker.consume('notify.billing', { timeout: 5000 }),
      broker.consume('notify.crm', { timeout: 5000 })
    ]);

    expect(warehouse.body.orderId).toBe('ORD-100');
    expect(billing.body.orderId).toBe('ORD-100');
    expect(crm.body.orderId).toBe('ORD-100');
  });
});
```

---

## Message Transformation Testing

### JSON-to-XML Transformation
```javascript
describe('JSON to XML Transformation', () => {
  it('transforms order JSON to SAP XML format', async () => {
    const jsonInput = {
      orderId: 'ORD-500',
      customer: { id: 'CUST-01', name: 'Acme Corp' },
      items: [
        { sku: 'MAT-100', quantity: 10, unitPrice: 25.00 }
      ]
    };

    await broker.publish('transform.json-in', { body: jsonInput });

    const xmlOutput = await broker.consume('transform.xml-out', { timeout: 5000 });
    const parsed = parseXml(xmlOutput.body);

    expect(parsed.SalesOrder.OrderNumber).toBe('ORD-500');
    expect(parsed.SalesOrder.SoldToParty.CustomerID).toBe('CUST-01');
    expect(parsed.SalesOrder.Items.Item[0].MaterialNumber).toBe('MAT-100');
    expect(parsed.SalesOrder.Items.Item[0].Quantity).toBe('10');
  });

  it('handles missing optional fields gracefully', async () => {
    const jsonInput = {
      orderId: 'ORD-501',
      customer: { id: 'CUST-02' }, // name is missing
      items: [{ sku: 'MAT-200', quantity: 5 }] // unitPrice missing
    };

    await broker.publish('transform.json-in', { body: jsonInput });

    const xmlOutput = await broker.consume('transform.xml-out', { timeout: 5000 });
    const parsed = parseXml(xmlOutput.body);

    expect(parsed.SalesOrder.SoldToParty.Name).toBe(''); // Default empty
    expect(parsed.SalesOrder.Items.Item[0].UnitPrice).toBe('0.00');
  });

  it('rejects invalid input and sends to DLQ with error details', async () => {
    const invalidInput = { items: [] }; // Missing required orderId and customer

    await broker.publish('transform.json-in', { body: invalidInput });

    const dlq = await broker.consume('transform.dlq', { timeout: 5000 });
    expect(dlq.headers['x-error-reason']).toContain('orderId is required');
  });
});
```

### Flat-File to XML Transformation
```javascript
describe('Flat-File to XML Transformation', () => {
  it('parses fixed-width flat file into structured XML', async () => {
    // Fixed-width: positions 1-10 = orderId, 11-20 = sku, 21-25 = qty
    const flatFile = 'ORD-001   MAT-10000010\nORD-001   MAT-20000005\n';

    await broker.publish('transform.flatfile-in', { body: flatFile });

    const xmlOutput = await broker.consume('transform.flatfile-xml-out', { timeout: 5000 });
    const parsed = parseXml(xmlOutput.body);

    expect(parsed.Orders.Order).toHaveLength(1);
    expect(parsed.Orders.Order[0].Items.Item).toHaveLength(2);
    expect(parsed.Orders.Order[0].Items.Item[0].Quantity).toBe('10');
  });

  it('handles lines with trailing whitespace correctly', async () => {
    const flatFile = 'ORD-002   MAT-30000001   \n';

    await broker.publish('transform.flatfile-in', { body: flatFile });

    const xmlOutput = await broker.consume('transform.flatfile-xml-out', { timeout: 5000 });
    const parsed = parseXml(xmlOutput.body);

    expect(parsed.Orders.Order[0].OrderId).toBe('ORD-002');
    expect(parsed.Orders.Order[0].Items.Item[0].Quantity).toBe('1');
  });
});
```

---

## Dead Letter Queue (DLQ) Testing

```javascript
describe('Dead Letter Queue Strategy', () => {
  it('moves message to DLQ after max retries', async () => {
    // Configure endpoint to always fail
    mockEndpoint.respondWith(500, 'Service Unavailable');

    await broker.publish('orders.process', {
      body: { orderId: 'ORD-FAIL' },
      headers: { 'x-correlation-id': 'corr-123' }
    });

    // Wait for retries to exhaust (3 retries with backoff)
    const dlqMessage = await broker.consume('orders.dlq', { timeout: 30000 });

    expect(dlqMessage.body.orderId).toBe('ORD-FAIL');
    expect(dlqMessage.headers['x-retry-count']).toBe('3');
    expect(dlqMessage.headers['x-original-queue']).toBe('orders.process');
    expect(dlqMessage.headers['x-correlation-id']).toBe('corr-123');
    expect(dlqMessage.headers['x-error-reason']).toContain('Service Unavailable');
    expect(dlqMessage.headers['x-first-failure-timestamp']).toBeDefined();
  });

  it('preserves original message body in DLQ', async () => {
    const originalBody = {
      orderId: 'ORD-PRESERVE',
      items: [{ sku: 'A', qty: 1 }, { sku: 'B', qty: 2 }]
    };

    mockEndpoint.respondWith(500, 'Timeout');
    await broker.publish('orders.process', { body: originalBody });

    const dlqMessage = await broker.consume('orders.dlq', { timeout: 30000 });
    expect(dlqMessage.body).toEqual(originalBody); // Body unchanged
  });

  it('DLQ messages can be replayed after fix', async () => {
    // Step 1: Message fails, lands in DLQ
    mockEndpoint.respondWith(500, 'Down');
    await broker.publish('orders.process', { body: { orderId: 'ORD-REPLAY' } });
    const dlqMessage = await broker.consume('orders.dlq', { timeout: 30000 });

    // Step 2: Fix the endpoint
    mockEndpoint.respondWith(200, 'OK');

    // Step 3: Replay from DLQ
    await broker.publish('orders.process', dlqMessage);
    const success = await broker.consume('orders.completed', { timeout: 5000 });
    expect(success.body.orderId).toBe('ORD-REPLAY');
  });
});
```

---

## Message Ordering and Sequencing

```javascript
describe('Message Ordering Validation', () => {
  it('preserves FIFO order within same partition key', async () => {
    const messages = Array.from({ length: 10 }, (_, i) => ({
      headers: { 'x-partition-key': 'ORDER-GROUP-1', 'x-sequence': String(i + 1) },
      body: { orderId: `ORD-${i + 1}`, sequence: i + 1 }
    }));

    for (const msg of messages) {
      await broker.publish('orders.sequenced', msg);
    }

    const received = [];
    for (let i = 0; i < 10; i++) {
      const msg = await broker.consume('orders.sequenced.out', { timeout: 5000 });
      received.push(msg.body.sequence);
    }

    expect(received).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
  });

  it('detects out-of-order messages via sequence gap', async () => {
    // Send messages 1, 2, 4 (skip 3)
    for (const seq of [1, 2, 4]) {
      await broker.publish('orders.sequenced', {
        headers: { 'x-partition-key': 'GROUP-2', 'x-sequence': String(seq) },
        body: { sequence: seq }
      });
    }

    // The middleware should detect the gap and raise an alert
    const alert = await broker.consume('orders.sequence-gap', { timeout: 10000 });
    expect(alert.body.missingSequence).toBe(3);
    expect(alert.body.partitionKey).toBe('GROUP-2');
  });
});
```

---

## ESB Error Handling and Compensation

```javascript
describe('Compensation Pattern - Saga', () => {
  it('compensates completed steps when a later step fails', async () => {
    // Saga: Reserve Inventory -> Charge Payment -> Ship
    // If Ship fails, Payment and Inventory must be compensated

    mockServices.inventory.respondWith(200); // Step 1 succeeds
    mockServices.payment.respondWith(200);   // Step 2 succeeds
    mockServices.shipping.respondWith(500);  // Step 3 fails

    await broker.publish('saga.order', {
      body: { orderId: 'ORD-SAGA', amount: 100.00 }
    });

    // Verify compensation messages were sent
    const paymentRefund = await broker.consume('saga.compensate.payment', { timeout: 10000 });
    expect(paymentRefund.body.action).toBe('REFUND');
    expect(paymentRefund.body.orderId).toBe('ORD-SAGA');

    const inventoryRelease = await broker.consume('saga.compensate.inventory', { timeout: 10000 });
    expect(inventoryRelease.body.action).toBe('RELEASE');
    expect(inventoryRelease.body.orderId).toBe('ORD-SAGA');
  });
});
```

---

## Protocol Mediation Testing

### SOAP-to-REST Mediation
```javascript
describe('SOAP to REST Protocol Mediation', () => {
  it('converts SOAP request to REST call and back', async () => {
    const soapRequest = `
      <soapenv:Envelope xmlns:soapenv="http://schemas.xmlsoap.org/soap/envelope/"
                        xmlns:ord="http://example.com/orders">
        <soapenv:Body>
          <ord:GetOrder>
            <ord:OrderId>ORD-700</ord:OrderId>
          </ord:GetOrder>
        </soapenv:Body>
      </soapenv:Envelope>
    `;

    // ESB should convert SOAP -> REST GET /orders/ORD-700
    mockRestBackend.onGet('/orders/ORD-700').reply(200, {
      orderId: 'ORD-700',
      status: 'SHIPPED',
      total: 150.00
    });

    const response = await soapClient.send(soapRequest, { endpoint: esbSoapEndpoint });

    // Verify SOAP response wraps REST data correctly
    const parsed = parseSoapResponse(response);
    expect(parsed.OrderId).toBe('ORD-700');
    expect(parsed.Status).toBe('SHIPPED');
    expect(parsed.Total).toBe('150.00');
  });

  it('maps REST error to SOAP fault', async () => {
    const soapRequest = buildSoapGetOrder('ORD-NOTFOUND');

    mockRestBackend.onGet('/orders/ORD-NOTFOUND').reply(404, {
      error: 'Order not found'
    });

    const response = await soapClient.send(soapRequest, { endpoint: esbSoapEndpoint });
    const fault = parseSoapFault(response);

    expect(fault.faultCode).toBe('Client');
    expect(fault.faultString).toContain('Order not found');
  });
});
```

### Sync-to-Async Mediation
```javascript
describe('Sync to Async Mediation', () => {
  it('accepts sync request, returns correlation ID, delivers result async', async () => {
    // Client sends synchronous HTTP POST
    const response = await httpClient.post(`${esbEndpoint}/async-orders`, {
      orderId: 'ORD-ASYNC',
      items: [{ sku: 'X1', qty: 5 }]
    });

    // ESB returns 202 Accepted with correlation ID
    expect(response.status).toBe(202);
    expect(response.body.correlationId).toBeDefined();
    const correlationId = response.body.correlationId;

    // Result arrives asynchronously on callback queue
    const result = await broker.consume('orders.async.results', {
      timeout: 15000,
      filter: (msg) => msg.headers['x-correlation-id'] === correlationId
    });

    expect(result.body.orderId).toBe('ORD-ASYNC');
    expect(result.body.status).toBe('ACCEPTED');
  });
});
```

---

## Enterprise Integration Patterns (EIP)

### Splitter Pattern
```javascript
describe('Splitter - Batch Order to Individual Items', () => {
  it('splits batch order into individual item messages', async () => {
    const batchOrder = {
      orderId: 'BATCH-001',
      items: [
        { sku: 'A1', qty: 10 },
        { sku: 'B2', qty: 5 },
        { sku: 'C3', qty: 20 }
      ]
    };

    await broker.publish('orders.batch', { body: batchOrder });

    const splitMessages = [];
    for (let i = 0; i < 3; i++) {
      splitMessages.push(await broker.consume('orders.items', { timeout: 5000 }));
    }

    expect(splitMessages).toHaveLength(3);
    expect(splitMessages.map(m => m.body.sku).sort()).toEqual(['A1', 'B2', 'C3']);

    // Each split message retains parent correlation
    for (const msg of splitMessages) {
      expect(msg.headers['x-batch-id']).toBe('BATCH-001');
      expect(msg.headers['x-split-total']).toBe('3');
    }
  });
});
```

### Aggregator Pattern
```javascript
describe('Aggregator - Collect Item Results', () => {
  it('aggregates individual item results into batch response', async () => {
    const batchId = 'BATCH-002';
    const itemResults = [
      { sku: 'A1', status: 'PICKED', batchId },
      { sku: 'B2', status: 'PICKED', batchId },
      { sku: 'C3', status: 'PICKED', batchId }
    ];

    for (const result of itemResults) {
      await broker.publish('items.completed', {
        headers: { 'x-batch-id': batchId, 'x-split-total': '3' },
        body: result
      });
    }

    const aggregated = await broker.consume('orders.batch.completed', { timeout: 10000 });

    expect(aggregated.body.batchId).toBe('BATCH-002');
    expect(aggregated.body.items).toHaveLength(3);
    expect(aggregated.body.items.every(i => i.status === 'PICKED')).toBe(true);
  });

  it('handles aggregation timeout for incomplete batches', async () => {
    const batchId = 'BATCH-TIMEOUT';

    // Only send 2 of 3 expected items
    for (let i = 0; i < 2; i++) {
      await broker.publish('items.completed', {
        headers: { 'x-batch-id': batchId, 'x-split-total': '3' },
        body: { sku: `ITEM-${i}`, status: 'PICKED', batchId }
      });
    }

    // After timeout, aggregator should emit partial result with error
    const partial = await broker.consume('orders.batch.incomplete', { timeout: 60000 });
    expect(partial.body.batchId).toBe('BATCH-TIMEOUT');
    expect(partial.body.receivedCount).toBe(2);
    expect(partial.body.expectedCount).toBe(3);
  });
});
```

### Content Enricher Pattern
```javascript
describe('Content Enricher - Customer Data Lookup', () => {
  it('enriches order with customer details from CRM', async () => {
    mockCrm.onGet('/customers/CUST-50').reply(200, {
      name: 'Acme Corp',
      tier: 'GOLD',
      creditLimit: 50000
    });

    await broker.publish('orders.enrich', {
      body: { orderId: 'ORD-ENRICH', customerId: 'CUST-50', total: 1200 }
    });

    const enriched = await broker.consume('orders.enriched', { timeout: 5000 });

    expect(enriched.body.orderId).toBe('ORD-ENRICH');
    expect(enriched.body.customer.name).toBe('Acme Corp');
    expect(enriched.body.customer.tier).toBe('GOLD');
    expect(enriched.body.creditCheckPassed).toBe(true); // 1200 < 50000
  });
});
```

---

## Retry and Circuit Breaker Patterns

```javascript
describe('Retry with Exponential Backoff', () => {
  it('retries 3 times with increasing delays before DLQ', async () => {
    const timestamps = [];
    mockEndpoint.onPost('/process').reply(() => {
      timestamps.push(Date.now());
      return [500, 'Temporary failure'];
    });

    await broker.publish('orders.process', { body: { orderId: 'ORD-RETRY' } });

    await broker.consume('orders.dlq', { timeout: 60000 });

    // Verify exponential backoff timing (1s, 2s, 4s approximately)
    expect(timestamps).toHaveLength(4); // 1 initial + 3 retries
    const delay1 = timestamps[1] - timestamps[0];
    const delay2 = timestamps[2] - timestamps[1];
    expect(delay1).toBeGreaterThan(800);  // ~1s
    expect(delay2).toBeGreaterThan(1800); // ~2s
  });
});

describe('Circuit Breaker', () => {
  it('opens circuit after threshold failures', async () => {
    // Fail 5 messages to trip circuit breaker (threshold = 5)
    mockEndpoint.onPost('/process').reply(500, 'Down');

    for (let i = 0; i < 5; i++) {
      await broker.publish('orders.circuit', { body: { orderId: `ORD-CB-${i}` } });
    }

    // Wait for circuit to open
    await sleep(5000);

    // Next message should be immediately rejected (circuit open)
    const startTime = Date.now();
    await broker.publish('orders.circuit', { body: { orderId: 'ORD-CB-FAST-FAIL' } });
    const dlq = await broker.consume('orders.circuit.dlq', { timeout: 3000 });
    const elapsed = Date.now() - startTime;

    expect(dlq.headers['x-circuit-state']).toBe('OPEN');
    expect(elapsed).toBeLessThan(2000); // Fast fail, no retry
  });
});
```

---

## Service Virtualization for Isolated Testing

```javascript
describe('Service Virtualization - Middleware Isolation', () => {
  let virtualService;

  beforeAll(async () => {
    virtualService = await mountebank.createImposter({
      port: 4545,
      protocol: 'http',
      stubs: [
        {
          predicates: [{ equals: { method: 'POST', path: '/sap/orders' } }],
          responses: [{
            is: {
              statusCode: 200,
              headers: { 'Content-Type': 'application/xml' },
              body: '<OrderResponse><Status>CREATED</Status><SapOrderId>SAP-001</SapOrderId></OrderResponse>'
            }
          }]
        },
        {
          predicates: [{ equals: { method: 'POST', path: '/sap/orders' },
                         contains: { body: '<InvalidField>' } }],
          responses: [{
            is: { statusCode: 400, body: '<Error>Invalid request</Error>' }
          }]
        }
      ]
    });
  });

  afterAll(async () => {
    await mountebank.deleteImposter(4545);
  });

  it('middleware correctly handles SAP success response', async () => {
    await broker.publish('orders.to-sap', {
      body: { orderId: 'ORD-VIRT', customer: 'CUST-01' }
    });

    const result = await broker.consume('orders.sap-confirmed', { timeout: 5000 });
    expect(result.body.sapOrderId).toBe('SAP-001');
  });
});
```

---

## Best Practices

### Do This
- Test both the happy path AND the error channel (DLQ) for every flow
- Use correlation IDs to trace messages across multi-hop middleware
- Validate schema compliance at every transformation boundary
- Test message ordering with partitioned/sequenced queues
- Use service virtualization to isolate middleware from backend systems
- Test retry exhaustion to confirm DLQ behavior
- Validate compensation patterns (saga rollback) for multi-step flows

### Avoid This
- Testing only the happy path and ignoring DLQ/error queues
- Hardcoding queue names; use configuration-driven tests
- Ignoring message ordering requirements
- Skipping transformation edge cases (empty fields, special characters, encoding)
- Testing middleware without correlation ID validation
- Assuming retries work correctly without verifying backoff timing
- Deploying without testing the circuit breaker threshold

---

## Agent-Assisted Middleware Testing

```typescript
// Validate middleware routing rules
await Task("Middleware Routing Validation", {
  routingConfig: 'esb/routing-rules.xml',
  messageTypes: ['STANDARD_ORDER', 'EXPRESS_ORDER', 'RETURN'],
  validateDLQ: true,
  checkCorrelationIds: true
}, "qe-middleware-validator");

// Test message broker behavior
await Task("Message Broker DLQ Test", {
  broker: 'rabbitmq',
  queues: ['orders.inbound', 'orders.process', 'orders.dlq'],
  retryPolicy: { maxRetries: 3, backoff: 'exponential' },
  poisonMessages: ['malformed-xml', 'missing-required-fields', 'oversized-payload']
}, "qe-message-broker-tester");

// SOAP/WSDL contract validation
await Task("SOAP Service Contract Validation", {
  wsdl: 'services/OrderService.wsdl',
  endpoints: ['CreateOrder', 'GetOrderStatus', 'CancelOrder'],
  validateWsSecurity: true,
  testFaultHandling: true
}, "qe-soap-tester");

// Transformation accuracy testing
await Task("Transformation Accuracy Suite", {
  transformations: [
    { name: 'JSON-to-SAP-XML', input: 'testdata/order.json', expected: 'testdata/order-sap.xml' },
    { name: 'Flat-to-XML', input: 'testdata/inventory.dat', expected: 'testdata/inventory.xml' }
  ],
  validateSchemas: true,
  edgeCases: ['empty-fields', 'special-chars', 'max-length']
}, "qe-middleware-validator");
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/middleware-testing/
  routing/           - Routing rule configurations and test results
  transformations/   - Mapping definitions and validation results
  dlq/               - DLQ test results and replay outcomes
  eip-patterns/      - EIP pattern test coverage
  circuit-breakers/  - Circuit breaker state and threshold configs
  correlation/       - Correlation ID tracing results
```

### Fleet Coordination
```typescript
const middlewareFleet = await FleetManager.coordinate({
  strategy: 'middleware-testing',
  agents: [
    'qe-middleware-validator',   // Routing, transformation, EIP patterns
    'qe-message-broker-tester',  // DLQ, ordering, retry, circuit breaker
    'qe-soap-tester'             // SOAP contracts, WS-Security, mediation
  ],
  topology: 'mesh'
});

await middlewareFleet.execute({
  flows: [
    { name: 'order-inbound', stages: ['route', 'transform', 'enrich', 'deliver'] },
    { name: 'payment-processing', stages: ['validate', 'authorize', 'settle'] }
  ],
  testDLQ: true,
  testCompensation: true
});
```

---

## Related Skills
- [api-testing-patterns](../api-testing-patterns/) - REST/GraphQL API testing fundamentals
- [contract-testing](../contract-testing/) - Consumer-driven contract testing with Pact
- [enterprise-integration-testing](../enterprise-integration-testing/) - Orchestrating all enterprise agents
- [chaos-engineering-resilience](../chaos-engineering-resilience/) - Fault injection and resilience testing
- [wms-testing-patterns](../wms-testing-patterns/) - Warehouse system integration testing

---

## Remember

Middleware is the nervous system of enterprise architecture. When it works, nobody notices. When it fails, everything breaks. Test the routing decisions, the transformations, the error channels, and the compensation patterns. Always validate with correlation IDs across multi-hop flows, and always confirm that poison messages end up in the DLQ with enough metadata to diagnose and replay.

**With Agents:** Agents validate routing rules against message types, test transformation accuracy with schema compliance, and exercise DLQ/retry/circuit-breaker behavior systematically. Use agents to cover the combinatorial explosion of routing paths and transformation edge cases that manual testing cannot reach.
