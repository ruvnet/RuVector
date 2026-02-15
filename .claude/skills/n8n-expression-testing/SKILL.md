---
name: n8n-expression-testing
description: "n8n expression syntax validation, context-aware testing, common pitfalls detection, and performance optimization. Use when validating n8n expressions and data transformations."
category: n8n-testing
priority: high
tokenEstimate: 1000
agents: [n8n-expression-validator]
implementation_status: production
optimization_version: 1.0
last_optimized: 2025-12-15
dependencies: []
quick_reference_card: true
tags: [n8n, expressions, javascript, data-transformation, validation]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/n8n-expression-testing.yaml

---

# n8n Expression Testing

<default_to_action>
When testing n8n expressions:
1. VALIDATE syntax before execution
2. TEST with multiple context scenarios
3. CHECK for null/undefined handling
4. VERIFY type safety
5. SCAN for security vulnerabilities

**Quick Expression Checklist:**
- Valid JavaScript syntax
- Context variables properly referenced ($json, $node)
- Null-safe access patterns (?., ??)
- No dangerous functions (eval, Function)
- Efficient for large data sets

**Common Pitfalls:**
- Accessing nested properties without null checks
- Type coercion issues
- Missing fallback values
- Inefficient array operations
</default_to_action>

## Quick Reference Card

### n8n Expression Syntax

| Pattern | Example | Description |
|---------|---------|-------------|
| Basic access | `{{ $json.field }}` | Access JSON field |
| Nested access | `{{ $json.user.email }}` | Access nested property |
| Array access | `{{ $json.items[0] }}` | Access array element |
| Node reference | `{{ $node["Name"].json.id }}` | Access other node's data |
| Method call | `{{ $json.name.toLowerCase() }}` | Call string method |
| Conditional | `{{ $json.x ? "yes" : "no" }}` | Ternary expression |

### Context Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `$json` | Current item data | `{{ $json.email }}` |
| `$node["Name"]` | Other node's data | `{{ $node["HTTP"].json.body }}` |
| `$items()` | Multiple items | `{{ $items("Node", 0, 0).json }}` |
| `$now` | Current timestamp | `{{ $now.toISO() }}` |
| `$today` | Today's date | `{{ $today }}` |
| `$runIndex` | Run iteration | `{{ $runIndex }}` |
| `$workflow` | Workflow info | `{{ $workflow.name }}` |

---

## Expression Syntax Patterns

### Safe Data Access

```javascript
// BAD: Can fail if nested objects are null
{{ $json.user.profile.email }}

// GOOD: Optional chaining with fallback
{{ $json.user?.profile?.email ?? '' }}

// BAD: Array access without bounds check
{{ $json.items[0].name }}

// GOOD: Safe array access
{{ $json.items?.[0]?.name ?? 'No items' }}
```

### Type Conversions

```javascript
// String to Number
{{ parseInt($json.quantity, 10) }}
{{ parseFloat($json.price) }}
{{ Number($json.value) }}

// Number to String
{{ String($json.id) }}
{{ $json.amount.toString() }}
{{ $json.count.toFixed(2) }}

// Date handling
{{ new Date($json.timestamp).toISOString() }}
{{ DateTime.fromISO($json.date).toFormat('yyyy-MM-dd') }}

// Boolean conversion
{{ Boolean($json.active) }}
{{ $json.enabled === 'true' }}
```

### String Operations

```javascript
// Case conversion
{{ $json.name.toLowerCase() }}
{{ $json.name.toUpperCase() }}
{{ $json.name.charAt(0).toUpperCase() + $json.name.slice(1) }}

// String manipulation
{{ $json.text.trim() }}
{{ $json.text.replace(/\s+/g, ' ') }}
{{ $json.text.substring(0, 100) }}

// Template strings
{{ `Hello, ${$json.firstName} ${$json.lastName}!` }}
{{ `Order #${$json.orderId} - ${$json.status}` }}
```

### Array Operations

```javascript
// Mapping
{{ $json.items.map(item => item.name) }}
{{ $json.items.map(item => ({ id: item.id, total: item.price * item.qty })) }}

// Filtering
{{ $json.items.filter(item => item.active) }}
{{ $json.items.filter(item => item.price > 100) }}

// Reducing
{{ $json.items.reduce((sum, item) => sum + item.price, 0) }}
{{ $json.items.reduce((acc, item) => ({ ...acc, [item.id]: item }), {}) }}

// Finding
{{ $json.items.find(item => item.id === $json.targetId) }}
{{ $json.items.findIndex(item => item.name === 'target') }}

// Joining
{{ $json.tags.join(', ') }}
{{ $json.items.map(i => i.name).join(' | ') }}
```

---

## Validation Patterns

```typescript
// Validate expression syntax
function validateExpressionSyntax(expression: string): ValidationResult {
  // Remove n8n template markers
  const code = expression.replace(/\{\{|\}\}/g, '').trim();

  try {
    // Check if valid JavaScript
    new Function(`return (${code})`);
    return { valid: true };
  } catch (error) {
    return {
      valid: false,
      error: error.message,
      suggestion: suggestFix(error.message, code)
    };
  }
}

// Validate context variables
function validateContextVariables(expression: string): string[] {
  const contextVars = ['$json', '$node', '$items', '$now', '$today', '$runIndex', '$workflow'];
  const usedVars = [];
  const invalidVars = [];

  // Find all $ prefixed variables
  const varPattern = /\$\w+/g;
  let match;

  while ((match = varPattern.exec(expression)) !== null) {
    const varName = match[0];
    if (contextVars.some(cv => varName.startsWith(cv))) {
      usedVars.push(varName);
    } else {
      invalidVars.push(varName);
    }
  }

  return { usedVars, invalidVars };
}

// Test expression with sample data
function testExpression(expression: string, context: any): TestResult {
  const code = expression.replace(/\{\{|\}\}/g, '').trim();

  try {
    // Create function with context
    const fn = new Function('$json', '$node', '$items', '$now', '$today',
      `return (${code})`);

    const result = fn(
      context.$json || {},
      context.$node || {},
      context.$items || (() => ({})),
      context.$now || new Date(),
      context.$today || new Date()
    );

    return { success: true, result };
  } catch (error) {
    return { success: false, error: error.message };
  }
}
```

---

## Common Errors and Fixes

### Undefined Property Access

```javascript
// ERROR: Cannot read property 'email' of undefined
{{ $json.user.email }}

// FIX 1: Optional chaining
{{ $json.user?.email }}

// FIX 2: With fallback
{{ $json.user?.email ?? 'no-email@example.com' }}

// FIX 3: Conditional
{{ $json.user ? $json.user.email : '' }}
```

### Type Errors

```javascript
// ERROR: toLowerCase is not a function (when null)
{{ $json.name.toLowerCase() }}

// FIX: Null check first
{{ $json.name?.toLowerCase() ?? '' }}

// ERROR: toFixed is not a function (string instead of number)
{{ $json.price.toFixed(2) }}

// FIX: Parse as number first
{{ parseFloat($json.price).toFixed(2) }}

// ERROR: map is not a function (not an array)
{{ $json.items.map(i => i.name) }}

// FIX: Ensure array
{{ (Array.isArray($json.items) ? $json.items : []).map(i => i.name) }}
```

### Node Reference Errors

```javascript
// ERROR: Node "Previous Node" not found
{{ $node["Previous Node"].json.data }}

// FIX: Use exact node name (case-sensitive)
{{ $node["Previous Node1"].json.data }}

// FIX: Add fallback for safety
{{ $node["Previous Node"]?.json?.data ?? {} }}
```

---

## Security Patterns

### Dangerous Functions to Avoid

```javascript
// DANGEROUS: Never use eval
{{ eval($json.code) }}

// DANGEROUS: Dynamic function creation
{{ new Function($json.code)() }}

// DANGEROUS: setTimeout with string
{{ setTimeout($json.code, 1000) }}

// SAFE: Use explicit operations instead
{{ $json.value * 2 }}
{{ JSON.parse($json.jsonString) }}
```

### Input Validation

```javascript
// Validate email format
{{ /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test($json.email) ? $json.email : '' }}

// Sanitize for HTML (basic)
{{ $json.text.replace(/[<>&"']/g, c => ({
  '<': '&lt;', '>': '&gt;', '&': '&amp;', '"': '&quot;', "'": '&#39;'
}[c])) }}

// Limit string length
{{ $json.input.substring(0, 1000) }}

// Validate number range
{{ Math.min(Math.max(parseInt($json.value), 0), 100) }}
```

---

## Performance Optimization

### Efficient Array Operations

```javascript
// SLOW: Multiple iterations
{{ $json.items.filter(i => i.active).map(i => i.name).join(', ') }}

// FASTER: Single reduce
{{ $json.items.reduce((acc, i) => i.active ? (acc ? `${acc}, ${i.name}` : i.name) : acc, '') }}

// SLOW: Nested loops
{{ $json.items.map(i => $json.categories.find(c => c.id === i.categoryId)) }}

// FASTER: Create lookup map first (in Code node)
const categoryMap = Object.fromEntries($json.categories.map(c => [c.id, c]));
return $json.items.map(i => categoryMap[i.categoryId]);
```

### Avoid in Expressions

```javascript
// AVOID: Complex logic in expressions
{{ $json.items.reduce((acc, item) => {
  const category = $json.categories.find(c => c.id === item.catId);
  if (category && category.active) {
    acc.push({ ...item, categoryName: category.name });
  }
  return acc;
}, []) }}

// BETTER: Move to Code node for complex transformations
```

---

## Testing Patterns

```typescript
// Expression test suite
const expressionTests = [
  {
    name: 'Basic property access',
    expression: '{{ $json.name }}',
    context: { $json: { name: 'John' } },
    expected: 'John'
  },
  {
    name: 'Nested with optional chaining',
    expression: '{{ $json.user?.email ?? "default" }}',
    context: { $json: { user: null } },
    expected: 'default'
  },
  {
    name: 'Array mapping',
    expression: '{{ $json.items.map(i => i.id).join(",") }}',
    context: { $json: { items: [{ id: 1 }, { id: 2 }] } },
    expected: '1,2'
  },
  {
    name: 'Conditional expression',
    expression: '{{ $json.score >= 70 ? "Pass" : "Fail" }}',
    context: { $json: { score: 85 } },
    expected: 'Pass'
  },
  {
    name: 'Node reference',
    expression: '{{ $node["Previous"].json.result }}',
    context: { $node: { Previous: { json: { result: 'success' } } } },
    expected: 'success'
  }
];

// Run tests
for (const test of expressionTests) {
  const result = testExpression(test.expression, test.context);
  console.log(`${test.name}: ${result.result === test.expected ? 'PASS' : 'FAIL'}`);
}
```

---

## Agent Coordination

### Memory Namespace
```
aqe/n8n/expressions/
├── validations/*    - Expression validation results
├── patterns/*       - Discovered expression patterns
├── errors/*         - Common error catalog
└── optimizations/*  - Performance suggestions
```

### Fleet Coordination
```typescript
// Coordinate expression validation with workflow testing
await Task("Validate expressions", {
  workflowId: "wf-123",
  validateAll: true,
  testWithSampleData: true
}, "n8n-expression-validator");
```

---

## Related Skills
- [n8n-workflow-testing-fundamentals](../n8n-workflow-testing-fundamentals/) - Workflow testing
- [n8n-security-testing](../n8n-security-testing/) - Security validation

---

## Remember

**n8n expressions are JavaScript-like** with special context variables ($json, $node, etc.). Testing requires:
- Syntax validation
- Context variable verification
- Null safety checks
- Type compatibility
- Security scanning

**Key patterns:** Use optional chaining (`?.`) and nullish coalescing (`??`) for safety. Move complex logic to Code nodes. Always test with edge cases (null, undefined, empty arrays).
