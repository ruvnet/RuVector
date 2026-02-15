---
name: sfdipot-product-factors
description: "James Bach's HTSM Product Factors (SFDIPOT) analysis for comprehensive test strategy generation. Use when analyzing requirements, epics, or user stories to generate prioritized test ideas across Structure, Function, Data, Interfaces, Platform, Operations, and Time dimensions."
category: requirements-analysis
priority: high
tokenEstimate: 1500
agents: [qe-product-factors-assessor, qe-test-idea-rewriter]
implementation_status: optimized
optimization_version: 1.0
last_optimized: 2026-01-17
dependencies: [brutal-honesty-review]
quick_reference_card: true
tags: [sfdipot, htsm, product-factors, test-strategy, requirements, james-bach]
---

# SFDIPOT Product Factors Analysis

<default_to_action>
When analyzing requirements:
1. IDENTIFY document type (epic, user story, spec, architecture doc)
2. DETECT domain context (ecommerce, healthcare, finance, etc.)
3. ANALYZE each SFDIPOT category systematically
4. GENERATE test ideas with action verbs (no "Verify" patterns)
5. ASSIGN priorities (P0-P3) and automation fitness
6. SURFACE clarifying questions for coverage gaps
7. OUTPUT in requested format (HTML, JSON, MD, Gherkin)

**Quality Gates:**
- Human exploration >= 10%
- No "Verify X" test patterns
- Priority distribution: P0 (8-12%), P1 (20-30%), P2 (35-45%), P3 (20-30%)
</default_to_action>

## Quick Reference Card

### The 7 Product Factors

| Factor | Question | Example Focus Areas |
|--------|----------|---------------------|
| **S**tructure | What IS it? | Components, code, dependencies |
| **F**unction | What does it DO? | Features, calculations, errors |
| **D**ata | What does it PROCESS? | Input, output, persistence |
| **I**nterfaces | How does it CONNECT? | UI, API, integrations |
| **P**latform | What does it DEPEND ON? | OS, browser, hardware |
| **O**perations | How is it USED? | Workflows, admin, recovery |
| **T**ime | WHEN do things happen? | Concurrency, scheduling |

### When to Use This Skill

| Scenario | Priority | Output |
|----------|----------|--------|
| New epic planning | High | Full SFDIPOT assessment + test ideas |
| Sprint requirement review | Medium | Targeted analysis + clarifying questions |
| Test strategy definition | High | Comprehensive test coverage plan |
| Risk assessment | High | Domain-weighted priority assignment |
| Coverage gap analysis | Medium | Penetrating questions + recommendations |

---

## SFDIPOT Analysis Framework

### Structure (What the product IS)

**Subcategories:**
- Code Integrity: Code quality, complexity, maintainability
- Hardware: Physical components, devices, peripherals
- Non-Executable Files: Config, data files, documents
- Executable Files: Binaries, scripts, libraries
- Dependencies: Third-party libraries, external services

**Sample Test Ideas:**
- Analyze cyclomatic complexity of critical modules; flag functions > 10
- Inject corrupted config file; confirm graceful degradation
- Upgrade dependency version; validate no breaking changes

### Function (What the product DOES)

**Subcategories:**
- Application: Core business features
- Calculation: Mathematical operations, transformations
- Error Handling: Exception management, recovery
- Security: Authentication, authorization, encryption
- State Transitions: Workflow states, data lifecycle
- Messaging: Notifications, events, queues
- Data Transformation: Format conversion, mapping

**Sample Test Ideas:**
- Submit order with 1000 line items; measure processing time
- Inject SQL in search field; confirm sanitization prevents execution
- Trigger state transition timeout; observe retry behavior

### Data (What the product PROCESSES)

**Subcategories:**
- Input: User input, file uploads, API payloads
- Output: Responses, reports, exports
- Persistence: Database storage, caching
- Storage: File systems, cloud storage
- Boundaries: Min/max values, edge cases
- Validation: Format rules, business rules
- Formats: JSON, XML, CSV handling

**Sample Test Ideas:**
- Upload 100MB file; confirm chunked processing completes
- Insert emoji in text field; validate storage and retrieval
- Query boundary date (2038-01-19); check timestamp handling

### Interfaces (How the product CONNECTS)

**Subcategories:**
- User Interface: Web, mobile, desktop UI
- APIs: REST, GraphQL, gRPC endpoints
- Integrations: Third-party service connections
- Protocols: HTTP, WebSocket, MQTT
- CLI: Command-line interfaces

**Sample Test Ideas:**
- Resize browser to 320px width; confirm responsive breakpoints
- Send malformed JSON to API; assert 400 response with details
- Disconnect WebSocket mid-transaction; observe reconnection

### Platform (What the product DEPENDS ON)

**Subcategories:**
- Browser: Chrome, Firefox, Safari, Edge
- OS: Windows, macOS, Linux, iOS, Android
- Hardware: CPU, memory, storage constraints
- External Software: Databases, message queues
- Network: Latency, bandwidth, offline

**Sample Test Ideas:**
- Execute on 1GB RAM device; profile memory usage
- Run with 500ms network latency; measure user experience
- Test on iOS 15 Safari; validate CSS grid support

### Operations (How the product is USED)

**Subcategories:**
- Common Use: Happy path workflows
- Extreme Use: Edge cases, stress conditions
- User Management: Roles, permissions, profiles
- Admin Operations: Configuration, monitoring
- Recovery: Backup, restore, disaster recovery

**Sample Test Ideas:**
- Execute 10,000 concurrent user sessions; monitor resource usage
- Revoke admin access mid-session; confirm immediate effect
- Restore from 24-hour-old backup; validate data integrity

### Time (WHEN things happen)

**Subcategories:**
- Concurrency: Parallel operations, race conditions
- Scheduling: Cron jobs, timed events
- Sequences: Order of operations, dependencies

**Sample Test Ideas:**
- Two users update same record simultaneously; confirm conflict resolution
- Execute scheduled job at DST transition; validate correct timing
- Cancel operation mid-sequence; observe partial state handling

---

## Quality Rules

### Test Idea Quality

**NEVER use "Verify X" patterns. Always use action verbs.**

| Bad Pattern | Good Pattern |
|-------------|--------------|
| Verify login works | Submit credentials; confirm session created |
| Verify API returns 200 | Send request; assert 200 response |
| Verify error displays | Trigger error; observe message |

### Priority Distribution

| Priority | Percentage | Criteria |
|----------|------------|----------|
| P0 (Critical) | 8-12% | Security, data loss, regulatory |
| P1 (High) | 20-30% | Core functionality, user impact |
| P2 (Medium) | 35-45% | Standard features, edge cases |
| P3 (Low) | 20-30% | Polish, minor improvements |

### Automation Fitness

| Type | Percentage | Use For |
|------|------------|---------|
| Unit | 30-40% | Isolated logic, calculations |
| Integration | 15-25% | API contracts, data flow |
| E2E | <=50% | User journeys, workflows |
| Human Exploration | >=10% | Usability, edge cases |

---

## Agent Integration

```typescript
// Full SFDIPOT assessment
await Task("SFDIPOT Assessment", {
  input: epicDocument,
  outputFormat: "html",
  domains: ["ecommerce"],
  brutalHonesty: true
}, "qe-product-factors-assessor");

// Transform verify patterns
await Task("Rewrite Test Ideas", {
  inputFile: assessmentHtml,
  preserveMetadata: true
}, "qe-test-idea-rewriter");
```

---

## Memory Namespace

```
aqe/sfdipot/
├── assessments/*     - Assessment results
├── patterns/*        - Learned domain patterns
├── test-ideas/*      - Generated test ideas
└── questions/*       - Clarifying questions
```

---

## Related Skills

- [brutal-honesty-review](../brutal-honesty-review/) - Quality validation
- [risk-based-testing](../risk-based-testing/) - Priority assignment
- [context-driven-testing](../context-driven-testing/) - Domain awareness
- [exploratory-testing-advanced](../exploratory-testing-advanced/) - Session design

---

## Remember

**SFDIPOT ensures comprehensive test coverage by examining products from 7 distinct perspectives.** Each category reveals risks that other perspectives might miss. Use domain detection to weight priorities appropriately.
