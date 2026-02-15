---
name: n8n-security-testing
description: "Credential exposure detection, OAuth flow validation, API key management testing, and data sanitization verification for n8n workflows. Use when validating n8n workflow security."
category: n8n-testing
priority: critical
tokenEstimate: 1100
agents: [n8n-integration-test]
implementation_status: production
optimization_version: 1.0
last_optimized: 2025-12-15
dependencies: []
quick_reference_card: true
tags: [n8n, security, credentials, oauth, api-keys, encryption, testing]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/n8n-security-testing.yaml

---

# n8n Security Testing

<default_to_action>
When testing n8n security:
1. SCAN for credential exposure in workflows
2. VERIFY encryption of sensitive data
3. TEST OAuth token handling
4. CHECK for insecure data transmission
5. VALIDATE input sanitization

**Quick Security Checklist:**
- No credentials in workflow JSON
- No credentials in execution logs
- OAuth tokens properly encrypted
- API keys not in version control
- Webhook authentication enabled
- Input data sanitized

**Critical Success Factors:**
- Scan all workflow exports
- Test credential rotation
- Verify encryption at rest
- Check audit logging
</default_to_action>

## Quick Reference Card

### Security Risk Areas

| Area | Risk Level | Testing Focus |
|------|------------|---------------|
| **Credential Storage** | Critical | Encryption, exposure |
| **Webhook Security** | High | Authentication, validation |
| **Expression Injection** | High | Input sanitization |
| **Data Leakage** | Medium | Logging, error messages |
| **OAuth Flows** | Medium | Token handling, refresh |

### Credential Types

| Type | Exposure Risk | Rotation |
|------|---------------|----------|
| **API Keys** | High if exposed | Manual |
| **OAuth Tokens** | Medium (short-lived) | Automatic |
| **Passwords** | Critical | Manual |
| **Webhooks** | Medium | Generate new |

---

## Credential Security Testing

### Scan for Exposed Credentials

```typescript
// Scan workflow JSON for credential exposure
async function scanForExposedCredentials(workflowId: string): Promise<CredentialScanResult> {
  const workflow = await getWorkflow(workflowId);
  const workflowJson = JSON.stringify(workflow, null, 2);

  const sensitivePatterns = [
    // API Keys
    { name: 'Generic API Key', pattern: /api[_-]?key["\s:=]+["']?([a-zA-Z0-9_-]{20,})["']?/gi },
    { name: 'AWS Access Key', pattern: /AKIA[0-9A-Z]{16}/g },
    { name: 'AWS Secret Key', pattern: /[a-zA-Z0-9/+=]{40}/g },
    // Tokens
    { name: 'Bearer Token', pattern: /bearer\s+[a-zA-Z0-9_-]{20,}/gi },
    { name: 'JWT Token', pattern: /eyJ[a-zA-Z0-9_-]*\.eyJ[a-zA-Z0-9_-]*\.[a-zA-Z0-9_-]*/g },
    { name: 'Slack Token', pattern: /xox[baprs]-[0-9]{10,13}-[0-9]{10,13}-[a-zA-Z0-9]{24}/g },
    // Passwords
    { name: 'Password Field', pattern: /"password":\s*"[^"]+"/gi },
    { name: 'Secret Field', pattern: /"secret":\s*"[^"]+"/gi },
    // OAuth
    { name: 'Client Secret', pattern: /client[_-]?secret["\s:=]+["']?([a-zA-Z0-9_-]{20,})["']?/gi },
    { name: 'Refresh Token', pattern: /refresh[_-]?token["\s:=]+["']?([a-zA-Z0-9_-]{20,})["']?/gi }
  ];

  const findings: CredentialFinding[] = [];

  for (const pattern of sensitivePatterns) {
    const matches = workflowJson.match(pattern.pattern);
    if (matches) {
      for (const match of matches) {
        findings.push({
          type: pattern.name,
          location: findLocationInWorkflow(workflow, match),
          severity: 'CRITICAL',
          recommendation: `Remove ${pattern.name} from workflow. Use n8n credentials instead.`
        });
      }
    }
  }

  return {
    workflowId,
    scanned: true,
    findingsCount: findings.length,
    findings,
    secure: findings.length === 0
  };
}
```

### Verify Credential Encryption

```typescript
// Verify credentials are encrypted at rest
async function verifyCredentialEncryption(credentialId: string): Promise<EncryptionResult> {
  // Get credential metadata (not the actual credential)
  const credential = await getCredentialMetadata(credentialId);

  // Check if credential data is encrypted
  const encryptionChecks = {
    // Check if stored data looks encrypted (not plain text)
    isEncrypted: !isPlainText(credential.data),
    // Check encryption algorithm
    algorithm: credential.encryptionAlgorithm || 'unknown',
    // Check key derivation
    keyDerivation: credential.keyDerivation || 'unknown',
    // Check if using instance encryption key
    instanceEncryption: credential.useInstanceKey || false
  };

  return {
    credentialId,
    credentialName: credential.name,
    credentialType: credential.type,
    encryption: encryptionChecks,
    secure: encryptionChecks.isEncrypted && encryptionChecks.algorithm !== 'unknown',
    recommendations: generateEncryptionRecommendations(encryptionChecks)
  };
}

// Check if data appears to be plain text
function isPlainText(data: string): boolean {
  // Plain text credentials often have recognizable patterns
  const plainTextPatterns = [
    /^[a-zA-Z0-9_-]+$/, // Simple alphanumeric
    /^sk-[a-zA-Z0-9]+$/, // API key format
    /^Bearer\s/, // Bearer token
  ];

  return plainTextPatterns.some(p => p.test(data));
}
```

### Test Credential Rotation

```typescript
// Test credential rotation process
async function testCredentialRotation(credentialId: string): Promise<RotationTestResult> {
  const credential = await getCredentialMetadata(credentialId);

  const rotationTests = {
    // Check if credential has rotation metadata
    hasRotationSchedule: !!credential.rotationSchedule,
    lastRotated: credential.lastRotatedAt,
    rotationDue: isRotationDue(credential),

    // Test OAuth token refresh
    oauthRefresh: credential.type.includes('oauth')
      ? await testOAuthRefresh(credentialId)
      : null,

    // Check credential age
    credentialAge: calculateAge(credential.createdAt),
    isStale: calculateAge(credential.createdAt) > 90 // 90 days
  };

  return {
    credentialId,
    rotationTests,
    recommendations: generateRotationRecommendations(rotationTests)
  };
}

// Test OAuth token refresh
async function testOAuthRefresh(credentialId: string): Promise<OAuthRefreshResult> {
  try {
    // Trigger refresh
    const refreshed = await refreshCredential(credentialId);

    return {
      success: true,
      newExpiry: refreshed.expiresAt,
      refreshedAt: new Date()
    };
  } catch (error) {
    return {
      success: false,
      error: error.message,
      recommendation: 'Re-authorize OAuth connection'
    };
  }
}
```

---

## Webhook Security Testing

### Authentication Testing

```typescript
// Test webhook authentication enforcement
async function testWebhookAuthentication(webhookUrl: string): Promise<WebhookAuthResult> {
  const authTests = [
    // No authentication
    {
      name: 'No Auth',
      headers: {},
      expectedStatus: 401
    },
    // Invalid Basic Auth
    {
      name: 'Invalid Basic Auth',
      headers: { 'Authorization': 'Basic aW52YWxpZDppbnZhbGlk' },
      expectedStatus: 401
    },
    // Invalid Bearer Token
    {
      name: 'Invalid Bearer',
      headers: { 'Authorization': 'Bearer invalid-token-12345' },
      expectedStatus: 401
    },
    // Invalid Header Auth
    {
      name: 'Invalid Header Auth',
      headers: { 'X-API-Key': 'invalid-key' },
      expectedStatus: 401
    }
  ];

  const results: AuthTestResult[] = [];

  for (const test of authTests) {
    const response = await fetch(webhookUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...test.headers
      },
      body: '{}'
    });

    results.push({
      test: test.name,
      status: response.status,
      passed: response.status === test.expectedStatus,
      actualStatus: response.status,
      expectedStatus: test.expectedStatus
    });
  }

  // Check if webhook has ANY auth
  const noAuthResponse = results.find(r => r.test === 'No Auth');
  const webhookHasAuth = noAuthResponse?.status === 401;

  return {
    webhookUrl,
    hasAuthentication: webhookHasAuth,
    testResults: results,
    allTestsPassed: results.every(r => r.passed),
    recommendation: !webhookHasAuth
      ? 'CRITICAL: Enable authentication on webhook'
      : null
  };
}
```

### Input Validation Testing

```typescript
// Test webhook input validation
async function testWebhookInputValidation(webhookUrl: string): Promise<InputValidationResult> {
  const maliciousPayloads = [
    // XSS attempts
    {
      name: 'XSS Script Tag',
      payload: { text: '<script>alert("xss")</script>' },
      check: 'sanitized'
    },
    {
      name: 'XSS Event Handler',
      payload: { text: '<img onerror="alert(1)" src="x">' },
      check: 'sanitized'
    },
    // SQL Injection
    {
      name: 'SQL Injection',
      payload: { id: "1; DROP TABLE users; --" },
      check: 'escaped'
    },
    // Command Injection
    {
      name: 'Command Injection',
      payload: { filename: '; rm -rf /' },
      check: 'rejected'
    },
    // Path Traversal
    {
      name: 'Path Traversal',
      payload: { path: '../../../etc/passwd' },
      check: 'rejected'
    },
    // JSON Injection
    {
      name: 'JSON Injection',
      payload: { data: '{"admin": true}' },
      check: 'escaped'
    },
    // Oversized payload
    {
      name: 'Oversized Payload',
      payload: { data: 'x'.repeat(10000000) }, // 10MB
      check: 'rejected'
    }
  ];

  const results: ValidationTestResult[] = [];

  for (const test of maliciousPayloads) {
    try {
      const response = await fetch(webhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(test.payload)
      });

      const responseBody = await response.text();

      results.push({
        test: test.name,
        status: response.status,
        handled: response.status !== 500, // Not a server error
        sanitized: !responseBody.includes(test.payload.text || test.payload.data),
        recommendation: response.status === 500
          ? `Input not handled safely: ${test.name}`
          : null
      });
    } catch (error) {
      results.push({
        test: test.name,
        handled: false,
        error: error.message
      });
    }
  }

  return {
    webhookUrl,
    testsRun: maliciousPayloads.length,
    passed: results.filter(r => r.handled).length,
    failed: results.filter(r => !r.handled).length,
    results,
    secure: results.every(r => r.handled)
  };
}
```

---

## Expression Security Testing

### Detect Dangerous Expressions

```typescript
// Scan expressions for security vulnerabilities
async function scanExpressionsForSecurity(workflowId: string): Promise<ExpressionSecurityResult> {
  const workflow = await getWorkflow(workflowId);
  const expressions = extractExpressions(workflow);

  const dangerousPatterns = [
    // Code execution
    { name: 'eval()', pattern: /eval\s*\(/g, severity: 'CRITICAL' },
    { name: 'Function()', pattern: /new\s+Function\s*\(/g, severity: 'CRITICAL' },
    { name: 'setTimeout string', pattern: /setTimeout\s*\(\s*["'`]/g, severity: 'HIGH' },
    { name: 'setInterval string', pattern: /setInterval\s*\(\s*["'`]/g, severity: 'HIGH' },

    // File system access
    { name: 'require()', pattern: /require\s*\(/g, severity: 'HIGH' },
    { name: 'import()', pattern: /import\s*\(/g, severity: 'HIGH' },
    { name: 'fs access', pattern: /\bfs\./g, severity: 'HIGH' },

    // Process/child execution
    { name: 'child_process', pattern: /child_process/g, severity: 'CRITICAL' },
    { name: 'process.', pattern: /process\./g, severity: 'MEDIUM' },
    { name: 'exec()', pattern: /exec\s*\(/g, severity: 'CRITICAL' },
    { name: 'spawn()', pattern: /spawn\s*\(/g, severity: 'CRITICAL' },

    // Network access
    { name: 'fetch()', pattern: /fetch\s*\(/g, severity: 'MEDIUM' },
    { name: 'XMLHttpRequest', pattern: /XMLHttpRequest/g, severity: 'MEDIUM' },

    // Prototype pollution
    { name: '__proto__', pattern: /__proto__/g, severity: 'HIGH' },
    { name: 'constructor.prototype', pattern: /constructor\.prototype/g, severity: 'HIGH' }
  ];

  const findings: SecurityFinding[] = [];

  for (const expr of expressions) {
    for (const pattern of dangerousPatterns) {
      if (pattern.pattern.test(expr.expression)) {
        findings.push({
          node: expr.nodeName,
          parameter: expr.parameter,
          expression: expr.expression,
          pattern: pattern.name,
          severity: pattern.severity,
          recommendation: `Remove ${pattern.name} from expression. Use safer alternatives.`
        });
      }
    }
  }

  return {
    workflowId,
    expressionsScanned: expressions.length,
    findings,
    secure: findings.length === 0,
    criticalIssues: findings.filter(f => f.severity === 'CRITICAL').length,
    highIssues: findings.filter(f => f.severity === 'HIGH').length
  };
}
```

---

## Data Leakage Testing

### Scan Execution Logs

```typescript
// Scan execution logs for credential leakage
async function scanExecutionLogs(workflowId: string, executionCount: number = 10): Promise<LogScanResult> {
  const executions = await getRecentExecutions(workflowId, executionCount);
  const findings: LogFinding[] = [];

  const sensitivePatterns = [
    { name: 'Password', pattern: /password["\s:=]+["']?[^"'\s]+["']?/gi },
    { name: 'API Key', pattern: /api[_-]?key["\s:=]+["']?[^"'\s]{20,}["']?/gi },
    { name: 'Token', pattern: /token["\s:=]+["']?[a-zA-Z0-9_-]{20,}["']?/gi },
    { name: 'Secret', pattern: /secret["\s:=]+["']?[^"'\s]+["']?/gi },
    { name: 'Authorization Header', pattern: /authorization["\s:]+["']?(bearer|basic)\s+[^"'\s]+["']?/gi }
  ];

  for (const execution of executions) {
    const logString = JSON.stringify(execution.data, null, 2);

    for (const pattern of sensitivePatterns) {
      const matches = logString.match(pattern.pattern);
      if (matches) {
        findings.push({
          executionId: execution.id,
          type: pattern.name,
          matchCount: matches.length,
          severity: 'HIGH',
          recommendation: `Mask ${pattern.name} in logs`
        });
      }
    }
  }

  return {
    workflowId,
    executionsScanned: executions.length,
    findings,
    secure: findings.length === 0,
    recommendation: findings.length > 0
      ? 'Enable credential masking in n8n settings'
      : null
  };
}
```

### Check Error Message Exposure

```typescript
// Check if error messages expose sensitive information
async function checkErrorMessageSecurity(workflowId: string): Promise<ErrorMessageResult> {
  // Trigger intentional errors
  const errorScenarios = [
    { name: 'Invalid credentials', inject: { credentials: null } },
    { name: 'Invalid endpoint', inject: { url: 'https://invalid' } },
    { name: 'Database error', inject: { query: 'INVALID SQL' } }
  ];

  const findings: ErrorFinding[] = [];

  for (const scenario of errorScenarios) {
    try {
      await executeWithError(workflowId, scenario.inject);
    } catch (error) {
      const errorMessage = error.message;

      // Check for sensitive data in error
      const sensitiveData = [
        { name: 'Connection string', pattern: /mongodb:\/\/[^@]+@/i },
        { name: 'Password in URL', pattern: /:\/\/[^:]+:[^@]+@/i },
        { name: 'Full file path', pattern: /\/(?:home|Users|var)\/[^\s]+/i },
        { name: 'Stack trace', pattern: /at\s+\w+\s+\([^)]+\)/i },
        { name: 'Internal IP', pattern: /\b(?:10|172\.(?:1[6-9]|2[0-9]|3[01])|192\.168)\.\d+\.\d+\b/i }
      ];

      for (const check of sensitiveData) {
        if (check.pattern.test(errorMessage)) {
          findings.push({
            scenario: scenario.name,
            exposedData: check.name,
            severity: 'MEDIUM',
            recommendation: `Sanitize ${check.name} from error messages`
          });
        }
      }
    }
  }

  return {
    workflowId,
    scenariosTested: errorScenarios.length,
    findings,
    secure: findings.length === 0
  };
}
```

---

## Security Report Template

```markdown
# n8n Security Audit Report

## Summary
| Category | Status | Findings |
|----------|--------|----------|
| Credential Security | PASS/FAIL | X issues |
| Webhook Security | PASS/FAIL | X issues |
| Expression Security | PASS/FAIL | X issues |
| Data Leakage | PASS/FAIL | X issues |

## Critical Findings

### CRIT-001: API Key Exposed in Workflow
- **Location:** HTTP Request node, URL parameter
- **Impact:** Credential theft, unauthorized access
- **Fix:** Move to n8n credentials store

### CRIT-002: eval() in Expression
- **Location:** Set node, custom field
- **Impact:** Remote code execution
- **Fix:** Remove eval, use explicit logic

## Recommendations

1. **Enable webhook authentication** - All public webhooks
2. **Rotate exposed credentials** - Immediately
3. **Enable log masking** - For all credentials
4. **Regular security scans** - Weekly automated scans

## Compliance Status
- OWASP Top 10: X/10 addressed
- SOC 2: Partially compliant
- GDPR: Review data handling
```

---

## Related Skills
- [n8n-workflow-testing-fundamentals](../n8n-workflow-testing-fundamentals/)
- [n8n-integration-testing-patterns](../n8n-integration-testing-patterns/)
- [compliance-testing](../compliance-testing/)

---

## Remember

**n8n handles sensitive credentials** for 400+ integrations. Security testing requires:
- Credential exposure scanning
- Encryption verification
- Webhook authentication testing
- Expression security analysis
- Data leakage detection

**Critical practices:** Never expose credentials in workflow JSON. Enable webhook authentication. Mask sensitive data in logs. Rotate credentials regularly. Scan expressions for dangerous functions.
