# AQE V3 Agents Index

This directory contains V3 QE agents installed by `aqe init`.

> **Note**: This directory only contains AQE-specific agents (v3-qe-*).
> Claude-flow core agents (adr-architect, memory-specialist, etc.) are part of
> the claude-flow system and are available separately.

## Summary

- **Total Agents**: 60
- **V3 QE Domain Agents**: 53
- **V3 Subagents**: 7

## Usage

Spawn agents using Claude Code's Task tool:

```javascript
Task("Generate tests for UserService", "...", "v3-qe-test-architect")
Task("Analyze coverage gaps", "...", "v3-qe-coverage-specialist")
Task("Run security scan", "...", "v3-qe-security-scanner")
```

## V3 QE Domain Agents (53)

Quality Engineering agents mapped to the 12 DDD bounded contexts.


### Visual Accessibility

- **qe-accessibility-auditor**: WCAG accessibility auditing with automated testing, screen reader validation, and remediation guidance
- **qe-responsive-tester**: Responsive design testing across viewports, devices, and breakpoints with layout regression detection
- **qe-visual-tester**: Visual regression testing with AI-powered screenshot comparison and multi-viewport support


### Test Generation

- **qe-bdd-generator**: BDD scenario generation with Gherkin syntax, example discovery, and step definition mapping
- **qe-mutation-tester**: Mutation testing specialist for test suite effectiveness evaluation with mutation score analysis
- **qe-property-tester**: Property-based testing with fast-check for edge case discovery through randomized input generation
- **qe-test-architect**: AI-powered test generation with sublinear optimization, multi-framework support, and self-learning capabilities
- **qe-test-idea-rewriter**: Transform passive test descriptions into active, observable test actions by eliminating


### Chaos Resilience

- **qe-chaos-engineer**: Chaos engineering specialist for controlled fault injection, resilience testing, and system weakness discovery
- **qe-load-tester**: Load and performance testing with traffic simulation, stress testing, and baseline management
- **qe-performance-tester**: Performance testing with load, stress, endurance testing and regression detection


### Quality Assessment

- **qe-code-complexity**: Code complexity analysis with cyclomatic/cognitive metrics, hotspot detection, and refactoring recommendations
- **qe-deployment-advisor**: Deployment readiness assessment with go/no-go decisions, risk aggregation, and rollback planning
- **qe-quality-gate**: Quality gate enforcement with configurable thresholds, policy validation, and AI-powered deployment decisions
- **qe-risk-assessor**: Quality risk assessment with multi-factor scoring, impact analysis, and mitigation recommendations


### Code Intelligence

- **qe-code-intelligence**: Knowledge graph builder with semantic code search, impact analysis, and HNSW-indexed vector retrieval
- **qe-dependency-mapper**: Dependency graph analysis with coupling metrics, circular detection, and security advisories
- **qe-kg-builder**: Knowledge graph construction with entity extraction, relationship inference, and HNSW-indexed queries


### Contract Testing

- **qe-contract-validator**: API contract validation with consumer-driven testing, provider verification, and breaking change detection
- **qe-graphql-tester**: GraphQL API testing with schema validation, query/mutation testing, and security analysis


### Coverage Analysis

- **qe-coverage-specialist**: O(log n) sublinear coverage analysis with risk-weighted gap detection and HNSW vector indexing
- **qe-gap-detector**: Coverage gap detection with risk scoring, semantic analysis, and targeted test recommendations


### Defect Intelligence

- **qe-defect-predictor**: ML-powered defect prediction using historical data, code metrics, and change patterns
- **qe-impact-analyzer**: Change impact analysis with blast radius calculation, test selection, and risk assessment
- **qe-regression-analyzer**: Regression risk analysis with intelligent test selection, historical analysis, and change impact scoring
- **qe-root-cause-analyzer**: Systematic root cause analysis for test failures and incidents with prevention recommendations


### General

- **qe-devils-advocate**: Meta-agent that challenges other agents
- **qe-fleet-commander**: Fleet management with agent lifecycle, workload distribution, and cross-domain coordination at scale
- **qe-integration-architect**: V3 deep agentic-flow@alpha integration specialist implementing ADR-001 for eliminating duplicate code and building claude-flow as a specialized extension
- **qe-integration-tester**: Integration test specialist for component interactions, API contracts, and system boundaries
- **qe-queen-coordinator**: V3 QE Queen Coordinator - MCP-powered swarm orchestration with real fleet coordination
- **qe-tdd-specialist**: TDD Red-Green-Refactor specialist for test-driven development with London and Chicago school support


### Test Execution

- **qe-flaky-hunter**: Flaky test detection and remediation with pattern recognition and auto-stabilization
- **qe-parallel-executor**: Parallel test execution with intelligent sharding, worker pool management, and result aggregation
- **qe-retry-handler**: Intelligent test retry with adaptive backoff, circuit breakers, and failure classification


### Learning Optimization

- **qe-learning-coordinator**: Fleet-wide learning coordination with pattern recognition, knowledge synthesis, and cross-project transfer
- **qe-metrics-optimizer**: Learning metrics optimization with hyperparameter tuning, A/B testing, and feedback loop implementation
- **qe-pattern-learner**: Pattern discovery and learning from QE activities for test generation and defect prediction
- **qe-transfer-specialist**: Knowledge transfer learning with domain adaptation, cross-framework learning, and knowledge distillation


### Enterprise Integration

- **qe-message-broker-tester**: Message broker and queue testing specialist for JMS, AMQP, MQTT, Kafka, and IBM MQ with transactional and reliability validation
- **qe-middleware-validator**: ESB and middleware validation specialist for routing rules, message transformations, protocol mediation, and integration pattern testing
- **qe-odata-contract-tester**: OData v2/v4 service contract testing with metadata validation, CRUD operations, batch processing, SAP-specific extensions, and concurrency control
- **qe-sap-idoc-tester**: SAP IDoc testing with type/segment validation, ALE configuration verification, async processing assertions, and cross-system flow validation
- **qe-sap-rfc-tester**: SAP RFC/BAPI testing specialist for remote function call validation, parameter testing, and system landscape verification
- **qe-soap-tester**: SOAP/WSDL testing specialist for enterprise web services with WS-Security, schema validation, and protocol compliance
- **qe-sod-analyzer**: SAP Segregation of Duties analysis with conflict detection, role-to-permission mapping, GRC integration, and compliance audit trail generation


### Security Compliance

- **qe-pentest-validator**: Graduated exploit validation with parallel vulnerability pipelines, browser-based attack execution, and
- **qe-security-auditor**: Security audit specialist with OWASP coverage, compliance validation, and remediation workflows
- **qe-security-scanner**: Comprehensive security scanning with SAST, DAST, dependency scanning, and secrets detection


### Requirements Validation

- **qe-product-factors-assessor**: SFDIPOT product factors analysis using James Bach
- **qe-quality-criteria-recommender**: HTSM v6.3 Quality Criteria analysis for shift-left quality engineering during PI/Sprint Planning
- **qe-qx-partner**: Quality Experience partnership bridging QA and UX with user journey analysis and experience impact assessment
- **qe-requirements-validator**: Requirements validation with testability analysis, BDD scenario generation, and acceptance criteria validation


## V3 Subagents (7)

Specialized sub-task agents for TDD and code review.

- **qe-code-reviewer**: Code review specialist for quality, maintainability, and standards compliance with actionable feedback
- **qe-integration-reviewer**: Integration review specialist for API compatibility, cross-service interactions, and breaking change detection
- **qe-performance-reviewer**: Performance review specialist for algorithmic complexity, resource usage, and bottleneck detection in code changes
- **qe-security-reviewer**: Security review specialist for vulnerability detection, authentication/authorization review, and secure coding practices
- **qe-tdd-green**: TDD GREEN phase specialist for implementing minimal code to make failing tests pass
- **qe-tdd-red**: TDD RED phase specialist for writing failing tests that define expected behavior before implementation
- **qe-tdd-refactor**: TDD REFACTOR phase specialist for improving code design while maintaining all passing tests

---

*Generated by AQE v3 init on 2026-02-15T15:40:59.301Z*
