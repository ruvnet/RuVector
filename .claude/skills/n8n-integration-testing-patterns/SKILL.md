---
name: n8n-integration-testing-patterns
description: "API contract testing, authentication flows, rate limit handling, and error scenario coverage for n8n integrations with external services. Use when testing n8n node integrations."
category: n8n-testing
priority: high
tokenEstimate: 1100
agents: [n8n-integration-test]
implementation_status: production
optimization_version: 1.0
last_optimized: 2025-12-15
dependencies: []
quick_reference_card: true
tags: [n8n, integration, api, authentication, oauth, rate-limiting, testing]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/n8n-integration-testing-patterns.yaml

---

# n8n Integration Testing Patterns

<default_to_action>
When testing n8n integrations:
1. VERIFY connectivity and authentication
2. TEST all configured operations
3. VALIDATE API response handling
4. CHECK rate limit behavior
5. CONFIRM error handling works

**Quick Integration Checklist:**
- Credentials valid and not expired
- API permissions sufficient for operations
- Rate limits understood and respected
- Error responses properly handled
- Data formats match API expectations

**Critical Success Factors:**
- Test in isolation before workflow integration
- Verify OAuth token refresh works
- Check API version compatibility
- Monitor rate limit headers
</default_to_action>

## Quick Reference Card

### Common n8n Integrations

| Category | Services | Auth Type |
|----------|----------|-----------|
| **Communication** | Slack, Teams, Discord | OAuth2, Webhook |
| **Data Storage** | Google Sheets, Airtable | OAuth2, API Key |
| **CRM** | Salesforce, HubSpot | OAuth2 |
| **Dev Tools** | GitHub, Jira, Linear | OAuth2, API Key |
| **Marketing** | Mailchimp, SendGrid | API Key |

### Authentication Types

| Type | Setup | Refresh |
|------|-------|---------|
| **OAuth2** | User authorization flow | Automatic token refresh |
| **API Key** | Manual key entry | Manual rotation |
| **Basic Auth** | Username/password | No refresh needed |
| **Header Auth** | Custom header | Manual rotation |

---

## Connectivity Testing

```typescript
// Test integration connectivity
async function testIntegrationConnectivity(nodeName: string): Promise<ConnectivityResult> {
  const node = await getNodeConfig(nodeName);

  // Check credential exists
  if (!node.credentials) {
    return { connected: false, error: 'No credentials configured' };
  }

  // Test based on integration type
  switch (getIntegrationType(node.type)) {
    case 'slack':
      return await testSlackConnectivity(node.credentials);
    case 'google-sheets':
      return await testGoogleSheetsConnectivity(node.credentials);
    case 'jira':
      return await testJiraConnectivity(node.credentials);
    case 'github':
      return await testGitHubConnectivity(node.credentials);
    default:
      return await testGenericAPIConnectivity(node);
  }
}

// Slack connectivity test
async function testSlackConnectivity(credentials: any): Promise<ConnectivityResult> {
  try {
    const response = await fetch('https://slack.com/api/auth.test', {
      headers: { 'Authorization': `Bearer ${credentials.accessToken}` }
    });
    const data = await response.json();

    return {
      connected: data.ok,
      workspace: data.team,
      user: data.user,
      scopes: data.response_metadata?.scopes || []
    };
  } catch (error) {
    return { connected: false, error: error.message };
  }
}

// Google Sheets connectivity test
async function testGoogleSheetsConnectivity(credentials: any): Promise<ConnectivityResult> {
  try {
    const response = await fetch('https://www.googleapis.com/drive/v3/about?fields=user', {
      headers: { 'Authorization': `Bearer ${credentials.accessToken}` }
    });

    if (response.status === 401) {
      // Try refresh
      const refreshed = await refreshOAuthToken(credentials);
      if (refreshed) {
        return testGoogleSheetsConnectivity({ ...credentials, accessToken: refreshed });
      }
      return { connected: false, error: 'Token expired, refresh failed' };
    }

    const data = await response.json();
    return { connected: true, user: data.user };
  } catch (error) {
    return { connected: false, error: error.message };
  }
}
```

---

## API Operation Testing

```typescript
// Test integration operations
async function testIntegrationOperations(nodeName: string): Promise<OperationResult[]> {
  const node = await getNodeConfig(nodeName);
  const operations = getNodeOperations(node.type);
  const results: OperationResult[] = [];

  for (const operation of operations) {
    const testData = generateTestData(node.type, operation);

    try {
      const startTime = Date.now();
      const response = await executeOperation(node, operation, testData);

      results.push({
        operation,
        success: true,
        responseTime: Date.now() - startTime,
        responseStatus: response.status,
        dataValid: validateResponseData(response.data, operation)
      });
    } catch (error) {
      results.push({
        operation,
        success: false,
        error: error.message,
        errorType: classifyError(error)
      });
    }
  }

  return results;
}

// Generate test data for operations
function generateTestData(nodeType: string, operation: string): any {
  const testDataMap = {
    'slack': {
      'postMessage': {
        channel: 'C123456',
        text: 'Test message from n8n integration test'
      },
      'uploadFile': {
        channels: 'C123456',
        content: 'Test file content',
        filename: 'test.txt'
      }
    },
    'google-sheets': {
      'appendData': {
        spreadsheetId: 'test-spreadsheet-id',
        range: 'Sheet1!A:Z',
        values: [['Test', 'Data', new Date().toISOString()]]
      },
      'readRows': {
        spreadsheetId: 'test-spreadsheet-id',
        range: 'Sheet1!A1:Z10'
      }
    },
    'jira': {
      'createIssue': {
        project: 'TEST',
        issueType: 'Task',
        summary: 'Test issue from n8n',
        description: 'Created by integration test'
      },
      'updateIssue': {
        issueKey: 'TEST-1',
        fields: { summary: 'Updated by n8n test' }
      }
    }
  };

  return testDataMap[nodeType]?.[operation] || {};
}
```

---

## Authentication Testing

### OAuth2 Flow Testing

```typescript
// Test OAuth2 authentication
async function testOAuth2Authentication(credentials: any): Promise<OAuth2Result> {
  const result: OAuth2Result = {
    tokenValid: false,
    refreshWorking: false,
    scopes: [],
    expiresIn: 0
  };

  // Test current token
  const tokenTest = await testAccessToken(credentials.accessToken);
  result.tokenValid = tokenTest.valid;
  result.scopes = tokenTest.scopes;

  // Check expiration
  if (credentials.expiresAt) {
    result.expiresIn = new Date(credentials.expiresAt).getTime() - Date.now();
    result.expiresSoon = result.expiresIn < 3600000; // Less than 1 hour
  }

  // Test refresh token
  if (credentials.refreshToken) {
    try {
      const newToken = await refreshOAuthToken(credentials);
      result.refreshWorking = !!newToken;
    } catch (error) {
      result.refreshError = error.message;
    }
  }

  return result;
}

// Test required scopes
async function testRequiredScopes(credentials: any, requiredScopes: string[]): Promise<ScopeResult> {
  const currentScopes = await getTokenScopes(credentials.accessToken);
  const missingScopes = requiredScopes.filter(s => !currentScopes.includes(s));

  return {
    hasAllScopes: missingScopes.length === 0,
    currentScopes,
    missingScopes,
    recommendation: missingScopes.length > 0
      ? `Re-authorize with scopes: ${missingScopes.join(', ')}`
      : null
  };
}
```

### API Key Testing

```typescript
// Test API key validity
async function testAPIKey(integration: string, apiKey: string): Promise<APIKeyResult> {
  const endpoints = {
    'sendgrid': 'https://api.sendgrid.com/v3/user/profile',
    'mailchimp': 'https://us1.api.mailchimp.com/3.0/ping',
    'airtable': 'https://api.airtable.com/v0/meta/whoami'
  };

  const endpoint = endpoints[integration];
  if (!endpoint) {
    return { valid: false, error: 'Unknown integration' };
  }

  try {
    const response = await fetch(endpoint, {
      headers: { 'Authorization': `Bearer ${apiKey}` }
    });

    return {
      valid: response.status === 200,
      status: response.status,
      rateLimit: extractRateLimitInfo(response.headers)
    };
  } catch (error) {
    return { valid: false, error: error.message };
  }
}
```

---

## Rate Limit Testing

```typescript
// Test rate limit handling
async function testRateLimits(nodeName: string, requestCount: number): Promise<RateLimitResult> {
  const results: RequestResult[] = [];
  let rateLimitHit = false;
  let retryAfter = 0;

  for (let i = 0; i < requestCount; i++) {
    const startTime = Date.now();
    const response = await makeRequest(nodeName);

    results.push({
      requestNumber: i + 1,
      status: response.status,
      responseTime: Date.now() - startTime,
      rateLimitRemaining: response.headers['x-ratelimit-remaining'],
      rateLimitLimit: response.headers['x-ratelimit-limit']
    });

    if (response.status === 429) {
      rateLimitHit = true;
      retryAfter = parseInt(response.headers['retry-after'] || '60');
      break;
    }

    // Small delay between requests
    await sleep(100);
  }

  return {
    requestsMade: results.length,
    rateLimitHit,
    retryAfter,
    results,
    recommendation: rateLimitHit
      ? `Implement exponential backoff, retry after ${retryAfter}s`
      : 'Rate limit not reached, consider increasing request count for thorough testing'
  };
}

// Extract rate limit info from headers
function extractRateLimitInfo(headers: Headers): RateLimitInfo {
  return {
    limit: headers.get('x-ratelimit-limit'),
    remaining: headers.get('x-ratelimit-remaining'),
    reset: headers.get('x-ratelimit-reset'),
    retryAfter: headers.get('retry-after')
  };
}
```

---

## Error Handling Testing

```typescript
// Test error scenarios
async function testErrorScenarios(nodeName: string): Promise<ErrorTestResult[]> {
  const scenarios = [
    { name: 'Invalid credentials', modify: { credentials: null } },
    { name: 'Invalid endpoint', modify: { url: 'https://invalid.example.com' } },
    { name: 'Timeout', modify: { timeout: 1 } },
    { name: 'Invalid data', modify: { data: { invalid: true } } },
    { name: 'Not found', modify: { resourceId: 'nonexistent-123' } },
    { name: 'Permission denied', modify: { scope: 'read-only' } }
  ];

  const results: ErrorTestResult[] = [];

  for (const scenario of scenarios) {
    try {
      const response = await executeWithModification(nodeName, scenario.modify);

      results.push({
        scenario: scenario.name,
        errorHandled: response.status >= 400,
        errorCode: response.status,
        errorMessage: response.data?.error?.message,
        retried: response.metadata?.retryCount > 0
      });
    } catch (error) {
      results.push({
        scenario: scenario.name,
        errorHandled: true,
        exceptionThrown: true,
        errorType: error.constructor.name,
        errorMessage: error.message
      });
    }
  }

  return results;
}

// Classify error types
function classifyError(error: any): string {
  if (error.status === 401 || error.status === 403) return 'authentication';
  if (error.status === 404) return 'not-found';
  if (error.status === 429) return 'rate-limit';
  if (error.status >= 500) return 'server-error';
  if (error.code === 'ETIMEDOUT') return 'timeout';
  if (error.code === 'ECONNREFUSED') return 'connection';
  return 'unknown';
}
```

---

## Integration-Specific Patterns

### Slack Integration

```typescript
const slackTestPatterns = {
  // Test message posting
  testPostMessage: async (credentials) => {
    return await fetch('https://slack.com/api/chat.postMessage', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${credentials.accessToken}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        channel: 'C123456',
        text: 'Integration test message'
      })
    });
  },

  // Test file upload
  testFileUpload: async (credentials) => {
    const formData = new FormData();
    formData.append('channels', 'C123456');
    formData.append('content', 'Test file content');
    formData.append('filename', 'test.txt');

    return await fetch('https://slack.com/api/files.upload', {
      method: 'POST',
      headers: { 'Authorization': `Bearer ${credentials.accessToken}` },
      body: formData
    });
  },

  // Validate required scopes
  requiredScopes: ['chat:write', 'files:write', 'channels:read']
};
```

### Google Sheets Integration

```typescript
const googleSheetsTestPatterns = {
  // Test read operation
  testReadRows: async (credentials, spreadsheetId) => {
    return await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${spreadsheetId}/values/Sheet1!A1:Z10`,
      { headers: { 'Authorization': `Bearer ${credentials.accessToken}` } }
    );
  },

  // Test append operation
  testAppendRow: async (credentials, spreadsheetId, values) => {
    return await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${spreadsheetId}/values/Sheet1!A:Z:append?valueInputOption=USER_ENTERED`,
      {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${credentials.accessToken}`,
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ values: [values] })
      }
    );
  },

  // Required scopes
  requiredScopes: ['https://www.googleapis.com/auth/spreadsheets']
};
```

---

## Test Report Template

```markdown
# Integration Test Report

## Summary
| Integration | Status | Auth | Operations | Errors |
|-------------|--------|------|------------|--------|
| Slack | PASS | OK | 4/4 | 0 |
| Google Sheets | WARN | Expiring | 3/3 | 0 |
| Jira | FAIL | OK | 2/4 | 2 |

## Authentication Status
- Slack: OAuth2 valid, expires in 28 days
- Google Sheets: OAuth2 expires in 2 hours - REFRESH RECOMMENDED
- Jira: API Key valid

## Rate Limit Status
| Integration | Limit | Used | Remaining |
|-------------|-------|------|-----------|
| Slack | 50/min | 12 | 38 |
| Google Sheets | 100/min | 45 | 55 |
| Jira | 100/min | 8 | 92 |

## Failed Operations
### Jira: Transition Issue
- Error: Invalid transition for current state
- Fix: Check workflow transitions in Jira

## Recommendations
1. Refresh Google Sheets OAuth token before expiration
2. Fix Jira workflow transition logic
```

---

## Related Skills
- [n8n-workflow-testing-fundamentals](../n8n-workflow-testing-fundamentals/)
- [n8n-security-testing](../n8n-security-testing/)
- [api-testing-patterns](../api-testing-patterns/)

---

## Remember

**n8n integrates with 400+ services**, each with unique authentication, rate limits, and API quirks. Testing requires:
- Connectivity verification
- Authentication validation (OAuth refresh, API key expiry)
- Operation testing with realistic data
- Rate limit awareness
- Error handling verification

**With Agents:** Use n8n-integration-test for comprehensive integration testing. Coordinate with n8n-workflow-executor to test integrations in context.
