# Six Hats Example: REST API Testing Strategy

**Context**: New REST API with 25 endpoints, 3-week development sprint, need to define comprehensive test strategy.

**Team**: 2 backend developers, 1 QE engineer
**Timeline**: Week 1 (planning), Weeks 2-3 (implementation & testing)
**Stack**: Node.js, Express, PostgreSQL, JWT auth

---

## 🤍 White Hat - Facts (5 minutes)

### Current State
- **API Endpoints**: 25 total (8 GET, 7 POST, 5 PUT, 3 DELETE, 2 PATCH)
- **Authentication**: JWT with refresh tokens
- **Database**: PostgreSQL with 12 tables
- **Documentation**: OpenAPI 3.0 spec (80% complete)
- **Current Coverage**: 0% (greenfield project)
- **Team Experience**: Strong backend, moderate API testing experience

### Technical Details
- **Expected Load**: 100 requests/second peak
- **Response Time Target**: <200ms for 95th percentile
- **Uptime SLA**: 99.9%
- **Data Volume**: ~50k records in production

### Available Resources
- **CI/CD**: GitHub Actions configured
- **Test Frameworks**: Jest, Supertest, Newman (Postman)
- **Timeline**: 2 weeks for test development
- **Budget**: Standard (no new tools needed)

### Known Risks (from product)
- Authentication complexity (OAuth2 + JWT)
- Multi-tenant data isolation
- Real-time updates via WebSockets

---

## ❤️ Red Hat - Feelings (3 minutes)

### Confidence Levels
- ✅ **Confident**: CRUD operations, database schema design
- ⚠️ **Somewhat Anxious**: Authentication edge cases, rate limiting
- ❌ **Very Concerned**: WebSocket testing, multi-tenant isolation

### Gut Instincts
- "The authentication flow feels overly complex - bugs will hide there"
- "I have a bad feeling about the data isolation between tenants"
- "The WebSocket real-time updates worry me - we've never tested those before"
- "The happy path will work, but edge cases will bite us"

### Team Sentiment
- Backend devs are excited about the API design
- QE engineer feels time pressure (2 weeks is tight)
- Product manager seems unaware of testing complexity
- Overall: 6/10 confidence we'll ship on time with quality

---

## 🖤 Black Hat - Risks & Problems (7 minutes)

### High-Risk Areas

**1. Authentication & Authorization**
- JWT token expiration not tested
- Refresh token rotation edge cases
- Password reset flow vulnerable
- Multi-tenant authorization bypass possible
- No rate limiting on auth endpoints

**2. Data Isolation (Multi-Tenancy)**
- Tenant A could potentially access Tenant B data
- No tests verify data isolation
- Database queries lack tenant ID filters
- Admin endpoints could leak data

**3. WebSocket Real-Time Updates**
- Zero experience testing WebSockets
- Connection stability untested
- Concurrent user scenarios unknown
- Message ordering not validated

**4. Performance & Scalability**
- No load testing planned
- N+1 query problems likely
- Database connection pooling untested
- No caching strategy

### Coverage Gaps
- Error handling: Only 30% of endpoints have error tests
- Edge cases: Boundary values, null handling untested
- Integration: Database + API + auth not tested together
- Regression: No test suite exists yet

### Assumptions to Challenge
- ❌ "JWT handles security" → Need to test token validation, expiry, revocation
- ❌ "ORM prevents SQL injection" → Need to validate input sanitization
- ❌ "Database constraints ensure data integrity" → Need application-level validation tests

### What Could Go Wrong in Production
- **Authentication bypass**: Attacker gains unauthorized access
- **Data leak**: Tenant A sees Tenant B's sensitive data
- **Performance degradation**: 100 req/sec causes database crash
- **Race conditions**: Concurrent updates corrupt data
- **API breaking changes**: Clients break on deployment

---

## 💛 Yellow Hat - Benefits & Opportunities (5 minutes)

### Current Strengths
- ✅ Well-documented OpenAPI spec (can generate tests from it!)
- ✅ Team has strong Jest experience
- ✅ CI/CD already configured (easy to add test stage)
- ✅ Database schema well-designed (migrations tested)
- ✅ Clean architecture (easy to mock/stub)

### Reusable Assets
- Existing authentication test utilities from previous project
- Database seeding scripts already written
- Postman collection with 15 example requests
- Swagger UI for manual testing

### Opportunities
- **Test automation framework**: Build once, reuse for future APIs
- **Contract testing**: Share API contracts with frontend team
- **Performance baseline**: Establish benchmarks early
- **Documentation**: Tests serve as living documentation

### Quick Wins
- Generate basic tests from OpenAPI spec (1 day)
- Automate Postman collection in CI (4 hours)
- Add authentication test suite from previous project (2 hours)

### Strategic Value
- First API with comprehensive testing → template for future projects
- Build QE reputation with strong quality delivery
- Reduce production incidents (costly to fix)

---

## 💚 Green Hat - Creative Ideas (7 minutes)

### Alternative Testing Approaches

**1. Contract Testing (Pact)**
- Define API contracts between frontend and backend
- Generate tests from contracts automatically
- Prevent breaking changes
- Enable independent team deployments

**2. Property-Based Testing (fast-check)**
- Define properties: "All GET requests return 200 or 404"
- Generate hundreds of random inputs
- Uncover edge cases humans miss
- Example: Test all possible JWT token formats

**3. Chaos Testing**
- Randomly kill database connections
- Inject network latency
- Corrupt request payloads
- Test system resilience

**4. Visual API Testing**
- Use Postman Mock Server
- Validate response schemas visually
- Enable non-technical stakeholders to review
- Generate documentation from tests

**5. AI-Generated Test Data**
- Use LLMs to generate realistic test users
- Create synthetic PII (privacy-safe)
- Generate edge case scenarios
- Cover cultural/language variations

### Crazy Ideas (That Might Work)

- **Idea**: Test API by deploying to production with feature flags (0% traffic)
  - **Why it might work**: Real production environment, no synthetic load, early validation

- **Idea**: Crowdsource testing with bug bounty program
  - **Why it might work**: Security experts find vulnerabilities we'd miss

- **Idea**: Record production traffic and replay in test
  - **Why it might work**: Test with real usage patterns, not synthetic scenarios

- **Idea**: Use GitHub Copilot to generate test cases from API documentation
  - **Why it might work**: Fast initial coverage, human review for quality

### Emerging Techniques to Explore

- **Fuzzing**: AFL, libFuzzer for input validation
- **GraphQL Testing**: If we migrate from REST
- **Service Virtualization**: For external API dependencies
- **Shift-Left Security**: OWASP ZAP in CI/CD

---

## 🔵 Blue Hat - Process & Action Plan (5 minutes)

### Test Strategy Summary

**Goal**: 80% automated coverage, <5% production defects, ship on time

**Approach**: Risk-based testing with focus on authentication and data isolation

**Frameworks**:
- Unit: Jest
- Integration: Supertest
- Contract: Pact (frontend collaboration)
- Performance: k6 (load testing)
- Security: OWASP ZAP (automated scan)

### Prioritized Test Development

**Week 1: Foundation (Red Hat priorities)**
- ✅ Day 1-2: Authentication test suite (JWT, refresh, expiry)
- ✅ Day 3: Multi-tenant data isolation tests
- ✅ Day 4: Error handling and edge cases
- ✅ Day 5: Integration tests (DB + API + Auth)

**Week 2: Expansion (Yellow Hat opportunities)**
- Day 1-2: Property-based testing for input validation
- Day 3: Contract testing setup with frontend
- Day 4: Performance baseline (load testing)
- Day 5: Security scan (OWASP ZAP)

**Week 3: Polish (Green Hat innovations)**
- Day 1-2: WebSocket testing framework
- Day 3: Chaos testing experiments
- Day 4: Documentation and knowledge sharing
- Day 5: Buffer for fixes

### Action Plan

| Priority | Action | Owner | Deadline | Success Criteria |
|----------|--------|-------|----------|------------------|
| P0 | Build auth test suite with JWT validation | QE | Week 1 Day 2 | 30 tests covering auth flows |
| P0 | Create tenant isolation tests | QE | Week 1 Day 3 | Verified no data leaks |
| P0 | Integration test framework | Dev 1 | Week 1 Day 4 | CI/CD passing |
| P1 | Property-based testing POC | QE | Week 2 Day 2 | 100+ generated test cases |
| P1 | Contract testing with frontend | Dev 2 | Week 2 Day 3 | Pact broker deployed |
| P1 | Load testing baseline | QE | Week 2 Day 4 | 100 req/sec validated |
| P2 | WebSocket test framework | Dev 1 | Week 3 Day 2 | 10 WebSocket tests |
| P2 | Security scan automation | QE | Week 2 Day 5 | OWASP ZAP in CI |

### Decision Points

**Go/No-Go Decision (End of Week 2)**:
- ✅ Must have: All P0 tests passing, no critical security issues
- ✅ Should have: P1 tests at 70% complete, performance baseline met
- ⚠️ Nice to have: P2 experimental tests

**Criteria for Launch**:
- Auth test suite: 95%+ coverage
- Data isolation: 100% verified
- Integration tests: All passing
- Performance: <200ms p95 response time
- Security: No high/critical findings

### Next Steps (Immediate)

1. **Today**: QE starts auth test suite (pull existing code from previous project)
2. **Tomorrow**: Dev 1 sets up integration test framework
3. **Day 3**: Team reviews OpenAPI spec, generates test skeleton
4. **Day 4**: QE demos property-based testing POC to team
5. **Day 5**: Retrospective on Week 1 progress

### Metrics to Track

- Test coverage % (target: 80%)
- Test execution time (target: <5 min)
- Defect detection rate (baseline: track for future)
- Production incidents (target: <2 per month)

### Communication Plan

- **Daily**: Standup with test status
- **Weekly**: Demo to product manager (show test results)
- **End of Week 2**: Go/No-Go decision meeting
- **Post-Launch**: Retrospective with Six Hats on testing effectiveness

---

## Synthesis: Key Decisions

### What We're Doing

1. **Risk-based approach**: Auth + data isolation first (Black Hat priorities)
2. **Quick wins**: Reuse existing test utilities (Yellow Hat opportunities)
3. **Innovation**: Property-based testing + contract testing (Green Hat ideas)
4. **Pragmatic**: Skip WebSocket deep testing for MVP, address in next sprint

### What We're NOT Doing (And Why)

- ❌ Comprehensive WebSocket testing (too risky for timeline, defer to next sprint)
- ❌ Chaos testing in production (not ready for that level of experimentation)
- ❌ 100% coverage (diminishing returns, 80% is realistic)
- ❌ Manual exploratory testing (no time, focus on automation)

### How This Addresses Concerns

- **Red Hat anxiety** about auth → P0 auth test suite
- **Black Hat risk** of data leaks → P0 tenant isolation tests
- **Yellow Hat opportunity** to reuse code → Leverage existing test utils
- **Green Hat innovation** → Property-based testing POC

### Success Definition

- Ship on time with <3 critical bugs in first month
- 80% automated test coverage
- <5 minute CI/CD test execution
- Team confident in quality (Red Hat validation)
- Reusable test framework for future APIs

---

## Lessons for Next Time

### What This Six Hats Session Revealed

1. **Red Hat** identified the real concern (auth complexity, data isolation) before we wasted time
2. **Black Hat** forced us to challenge "JWT is secure" assumption
3. **Green Hat** gave us property-based testing idea (wouldn't have thought of it)
4. **Yellow Hat** reminded us we have existing auth tests (saved 2 days)
5. **Blue Hat** prevented scope creep (deferred WebSocket testing)

### Time Investment vs Value

- **Time**: 45 minutes for Six Hats session
- **Value**:
  - Avoided 2 days of redundant work (reused existing tests)
  - Identified critical risk (data isolation) early
  - Aligned team on priorities
  - Created clear action plan

**ROI**: ~4x (45 min investment prevented 2+ days of rework)

---

**Session Date**: 2025-11-13
**Participants**: Backend Dev 1, Backend Dev 2, QE Engineer
**Duration**: 45 minutes
**Outcome**: Clear test strategy, team alignment, on-track for launch
