# Evidence Classification Guide

Guidelines for classifying evidence in Quality Criteria recommendations.

## Evidence Types

### Direct Evidence
**Definition:** Actual code quote, explicit documentation statement, or measurable fact from source.

**Requirements:**
- Must include `file_path:line_range` reference (e.g., `src/auth/login.ts:45-52`)
- Line ranges should be narrow (max 10-15 lines)
- Must quote or directly reference the source

**Examples:**
```
Source: src/payment/processor.ts:123-128
Type: Direct
Finding: No input validation before API call
Reasoning: Unvalidated input could enable injection attacks
```

### Inferred Evidence
**Definition:** Logical deduction from observed patterns, architectural implications, or domain knowledge.

**Requirements:**
- Must show reasoning chain
- Can use architectural implications
- Should reference what was observed

**Examples:**
```
Source: Architecture review of src/api/
Type: Inferred
Finding: No rate limiting middleware detected
Reasoning: API endpoints could be vulnerable to DoS; need to verify with load testing
```

### Claimed Evidence
**Definition:** Statement that requires verification - based on assumptions or incomplete data.

**Requirements:**
- Must state "requires verification" or "needs inspection to confirm"
- Must NOT speculate about what "could" or "might" happen
- Used when source is unavailable or claim needs validation

**Examples:**
```
WRONG: "Could range from efficient to aggressive implementation"
RIGHT: "Poll interval not specified - requires code inspection to verify"
```

## Evidence Table Format

```html
<table class="evidence-table">
  <thead>
    <tr>
      <th>Source Reference</th>
      <th>Type</th>
      <th>Quality Implication</th>
      <th>Reasoning</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>src/auth/session.ts:89-94</code></td>
      <td><span class="evidence-type direct">Direct</span></td>
      <td>Session tokens stored without encryption</td>
      <td class="evidence-reasoning">Credential exposure risk if storage is compromised</td>
    </tr>
  </tbody>
</table>
```

## Source Reference Format

### For Specific Code
```
file_path:start_line-end_line
Example: src/agents/FleetCommanderAgent.ts:847-852
```

### For File-Level Metrics
```
file_path (metric)
Example: src/agents/N8nBaseAgent.ts (683 LOC)
```

### For Search Results (No Matches)
```
N/A (verified via Glob/Grep search)
- NOT: tests/**/n8n/**/*.test.ts (glob pattern)
```

## Reasoning Column Guidelines

The Reasoning column must explain **WHY** something matters, not **WHAT** the code does.

| WRONG (describes WHAT) | CORRECT (explains WHY) |
|------------------------|------------------------|
| "Retry logic with exponential backoff" | "Retry pattern handles transient failures; needs edge case testing for timeout exhaustion" |
| "Session cookie stored in memory" | "Credential in memory could leak if agent state is serialized to logs" |
| "getWorkflow supports forceRefresh flag" | "Cache bypass prevents stale data; but increases load on source system" |

**Formula:**
```
{What the code does} → {Why that matters for quality} → {What could go wrong}
```

## Prohibited Patterns

- **No confidence percentages**: Use evidence types instead of "85% confident"
- **No vague blast radius**: Use "affects 19 agents" not "affects many"
- **No speculation in Claimed**: Use "requires verification" not "could be X or Y"
- **No keyword matching claims**: Show semantic reasoning, not keyword counts
