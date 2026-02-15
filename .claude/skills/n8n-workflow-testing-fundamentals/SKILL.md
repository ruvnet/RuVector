---
name: n8n-workflow-testing-fundamentals
description: "Comprehensive n8n workflow testing including execution lifecycle, node connection patterns, data flow validation, and error handling strategies. Use when testing n8n workflow automation applications."
category: n8n-testing
priority: high
tokenEstimate: 1200
agents: [n8n-workflow-executor, n8n-node-validator, n8n-trigger-test]
implementation_status: production
optimization_version: 1.0
last_optimized: 2025-12-15
dependencies: []
quick_reference_card: true
tags: [n8n, workflow, automation, testing, data-flow, nodes, triggers]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/n8n-workflow-testing-fundamentals.yaml

---

# n8n Workflow Testing Fundamentals

<default_to_action>
When testing n8n workflows:
1. VALIDATE workflow structure before execution
2. TEST with realistic test data
3. VERIFY node-to-node data flow
4. CHECK error handling paths
5. MEASURE execution performance

**Quick n8n Testing Checklist:**
- All nodes properly connected (no orphans)
- Trigger node correctly configured
- Data mappings between nodes valid
- Error workflows defined
- Credentials properly referenced

**Critical Success Factors:**
- Test each execution path separately
- Validate data transformations at each node
- Check retry and error handling behavior
- Verify integrations with external services
</default_to_action>

## Quick Reference Card

### When to Use
- Testing new n8n workflows
- Validating workflow changes
- Debugging failed executions
- Performance optimization
- Pre-deployment validation

### n8n Workflow Components

| Component | Purpose | Testing Focus |
|-----------|---------|---------------|
| **Trigger** | Starts workflow | Reliable activation, payload handling |
| **Action Nodes** | Process data | Configuration, data mapping |
| **Logic Nodes** | Control flow | Conditional routing, branches |
| **Integration Nodes** | External APIs | Auth, rate limits, errors |
| **Error Workflow** | Handle failures | Recovery, notifications |

### Workflow Execution States

| State | Meaning | Test Action |
|-------|---------|-------------|
| `running` | Currently executing | Monitor progress |
| `success` | Completed successfully | Validate outputs |
| `failed` | Execution failed | Analyze error |
| `waiting` | Waiting for trigger | Test trigger mechanism |

---

## Workflow Structure Validation

```typescript
// Validate workflow structure before execution
async function validateWorkflowStructure(workflowId: string) {
  const workflow = await getWorkflow(workflowId);

  // Check for trigger node
  const triggerNode = workflow.nodes.find(n =>
    n.type.includes('trigger') || n.type.includes('webhook')
  );
  if (!triggerNode) {
    throw new Error('Workflow must have a trigger node');
  }

  // Check for orphan nodes (no connections)
  const connectedNodes = new Set();
  for (const [source, targets] of Object.entries(workflow.connections)) {
    connectedNodes.add(source);
    for (const outputs of Object.values(targets)) {
      for (const connections of outputs) {
        for (const conn of connections) {
          connectedNodes.add(conn.node);
        }
      }
    }
  }

  const orphans = workflow.nodes.filter(n => !connectedNodes.has(n.name));
  if (orphans.length > 0) {
    console.warn('Orphan nodes detected:', orphans.map(n => n.name));
  }

  // Validate credentials
  for (const node of workflow.nodes) {
    if (node.credentials) {
      for (const [type, ref] of Object.entries(node.credentials)) {
        if (!await credentialExists(ref.id)) {
          throw new Error(`Missing credential: ${type} for node ${node.name}`);
        }
      }
    }
  }

  return { valid: true, orphans, triggerNode };
}
```

---

## Execution Testing

```typescript
// Test workflow execution with various inputs
async function testWorkflowExecution(workflowId: string, testCases: TestCase[]) {
  const results: TestResult[] = [];

  for (const testCase of testCases) {
    const startTime = Date.now();

    // Execute workflow
    const execution = await executeWorkflow(workflowId, testCase.input);

    // Wait for completion
    const result = await waitForCompletion(execution.id, testCase.timeout || 30000);

    // Validate output
    const outputValid = validateOutput(result.data, testCase.expected);

    results.push({
      testCase: testCase.name,
      success: result.status === 'success' && outputValid,
      duration: Date.now() - startTime,
      actualOutput: result.data,
      expectedOutput: testCase.expected
    });
  }

  return results;
}

// Example test cases
const testCases = [
  {
    name: 'Valid customer data',
    input: { name: 'John Doe', email: 'john@example.com' },
    expected: { processed: true, customerId: /^cust_/ },
    timeout: 10000
  },
  {
    name: 'Missing email',
    input: { name: 'Jane Doe' },
    expected: { error: 'Email required' },
    timeout: 5000
  },
  {
    name: 'Invalid email format',
    input: { name: 'Bob', email: 'not-an-email' },
    expected: { error: 'Invalid email' },
    timeout: 5000
  }
];
```

---

## Data Flow Validation

```typescript
// Trace data through workflow nodes
async function validateDataFlow(executionId: string) {
  const execution = await getExecution(executionId);
  const nodeResults = execution.data.resultData.runData;

  const dataFlow: DataFlowStep[] = [];

  for (const [nodeName, runs] of Object.entries(nodeResults)) {
    for (const run of runs) {
      dataFlow.push({
        node: nodeName,
        input: run.data?.main?.[0]?.[0]?.json || {},
        output: run.data?.main?.[0]?.[0]?.json || {},
        executionTime: run.executionTime,
        status: run.executionStatus
      });
    }
  }

  // Validate data transformations
  for (let i = 1; i < dataFlow.length; i++) {
    const prev = dataFlow[i - 1];
    const curr = dataFlow[i];

    // Check if expected data passed through
    validateDataMapping(prev.output, curr.input);
  }

  return dataFlow;
}

// Validate data mapping between nodes
function validateDataMapping(sourceOutput: any, targetInput: any) {
  // Check all required fields are present
  const missingFields: string[] = [];

  for (const [key, value] of Object.entries(targetInput)) {
    if (value === undefined && sourceOutput[key] === undefined) {
      missingFields.push(key);
    }
  }

  if (missingFields.length > 0) {
    console.warn('Missing fields in data mapping:', missingFields);
  }

  return missingFields.length === 0;
}
```

---

## Error Handling Testing

```typescript
// Test error handling paths
async function testErrorHandling(workflowId: string) {
  const errorScenarios = [
    {
      name: 'API timeout',
      inject: { delay: 35000 }, // Trigger timeout
      expectedError: 'timeout'
    },
    {
      name: 'Invalid data',
      inject: { invalidField: true },
      expectedError: 'validation'
    },
    {
      name: 'Missing credentials',
      inject: { removeCredentials: true },
      expectedError: 'authentication'
    }
  ];

  const results: ErrorTestResult[] = [];

  for (const scenario of errorScenarios) {
    // Execute with error injection
    const execution = await executeWithErrorInjection(workflowId, scenario.inject);

    // Check error was caught
    const result = await waitForCompletion(execution.id);

    // Validate error handling
    results.push({
      scenario: scenario.name,
      errorCaught: result.status === 'failed',
      errorType: result.data?.resultData?.error?.type,
      expectedError: scenario.expectedError,
      errorWorkflowTriggered: await checkErrorWorkflowTriggered(execution.id),
      alertSent: await checkAlertSent(execution.id)
    });
  }

  return results;
}

// Verify error workflow was triggered
async function checkErrorWorkflowTriggered(executionId: string): Promise<boolean> {
  const errorExecutions = await getExecutions({
    filter: {
      metadata: { errorTriggeredBy: executionId }
    }
  });

  return errorExecutions.length > 0;
}
```

---

## Node Connection Patterns

### Linear Flow
```
Trigger → Process → Transform → Output
```
**Testing:** Execute once, validate each node output

### Branching Flow
```
Trigger → IF → [Branch A] → Merge → Output
              → [Branch B] →
```
**Testing:** Test both branches separately, verify merge behavior

### Parallel Flow
```
Trigger → Split → [Process A] → Merge → Output
                → [Process B] →
```
**Testing:** Validate parallel execution, check merge timing

### Loop Flow
```
Trigger → SplitInBatches → Process → [Loop back until done] → Output
```
**Testing:** Test with varying batch sizes, verify all items processed

---

## Common Testing Patterns

### Test Data Generation

```typescript
// Generate test data for common n8n patterns
const testDataGenerators = {
  webhook: () => ({
    body: { event: 'test', timestamp: new Date().toISOString() },
    headers: { 'Content-Type': 'application/json' },
    query: { source: 'test' }
  }),

  slack: () => ({
    type: 'message',
    channel: 'C123456',
    user: 'U789012',
    text: 'Test message'
  }),

  github: () => ({
    action: 'opened',
    issue: {
      number: 1,
      title: 'Test Issue',
      body: 'Test body'
    },
    repository: {
      full_name: 'test/repo'
    }
  }),

  stripe: () => ({
    type: 'payment_intent.succeeded',
    data: {
      object: {
        id: 'pi_test123',
        amount: 1000,
        currency: 'usd'
      }
    }
  })
};
```

### Execution Assertions

```typescript
// Common assertions for workflow execution
const workflowAssertions = {
  // Assert workflow completed
  assertCompleted: (execution) => {
    expect(execution.finished).toBe(true);
    expect(execution.status).toBe('success');
  },

  // Assert specific node executed
  assertNodeExecuted: (execution, nodeName) => {
    const nodeData = execution.data.resultData.runData[nodeName];
    expect(nodeData).toBeDefined();
    expect(nodeData[0].executionStatus).toBe('success');
  },

  // Assert data transformation
  assertDataTransformed: (execution, nodeName, expectedData) => {
    const nodeOutput = execution.data.resultData.runData[nodeName][0].data.main[0][0].json;
    expect(nodeOutput).toMatchObject(expectedData);
  },

  // Assert execution time
  assertExecutionTime: (execution, maxMs) => {
    const duration = new Date(execution.stoppedAt) - new Date(execution.startedAt);
    expect(duration).toBeLessThan(maxMs);
  }
};
```

---

## Agent Coordination Hints

### Memory Namespace
```
aqe/n8n/
├── workflows/*          - Cached workflow definitions
├── test-results/*       - Test execution results
├── validations/*        - Validation reports
├── patterns/*           - Discovered testing patterns
└── executions/*         - Execution tracking
```

### Fleet Coordination
```typescript
// Comprehensive n8n testing with fleet
const n8nFleet = await FleetManager.coordinate({
  strategy: 'n8n-testing',
  agents: [
    'n8n-workflow-executor',  // Execute and validate
    'n8n-node-validator',     // Validate configurations
    'n8n-trigger-test',       // Test triggers
    'n8n-expression-validator', // Validate expressions
    'n8n-integration-test'    // Test integrations
  ],
  topology: 'parallel'
});
```

---

## Related Skills
- [n8n-expression-testing](../n8n-expression-testing/) - Expression validation
- [n8n-trigger-testing-strategies](../n8n-trigger-testing-strategies/) - Trigger testing
- [n8n-integration-testing-patterns](../n8n-integration-testing-patterns/) - Integration testing
- [n8n-security-testing](../n8n-security-testing/) - Security validation

---

## Remember

**n8n workflows are JSON-based execution flows** that connect 400+ services. Testing requires validating:
- Workflow structure (nodes, connections)
- Trigger reliability (webhooks, schedules)
- Data flow (transformations between nodes)
- Error handling (retry, fallback, notifications)
- Performance (execution time, resource usage)

**With Agents:** Use n8n-workflow-executor for execution testing, n8n-node-validator for configuration validation, and coordinate multiple agents for comprehensive workflow testing.
