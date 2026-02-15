---
name: qcsd-ideation-swarm
description: "QCSD Ideation phase swarm for Quality Criteria sessions using HTSM v6.3, Risk Storming, and Testability analysis before development begins. Uses 5-tier browser cascade: Vibium → agent-browser → Playwright+Stealth → WebFetch → WebSearch-fallback."
category: qcsd-phases
priority: critical
version: 7.5.1
tokenEstimate: 3500
# DDD Domain Mapping (from QCSD-AGENTIC-QE-MAPPING-FRAMEWORK.md)
domains:
  primary:
    - domain: requirements-validation
      agents: [qe-quality-criteria-recommender, qe-requirements-validator]
    - domain: coverage-analysis
      agents: [qe-risk-assessor]
  conditional:
    - domain: security-compliance
      agents: [qe-security-auditor]
    - domain: visual-accessibility
      agents: [qe-accessibility-auditor]
    - domain: cross-domain
      agents: [qe-qx-partner]
    - domain: enterprise-integration
      agents: [qe-middleware-validator, qe-sap-rfc-tester, qe-sod-analyzer]
# Agent Inventory
agents:
  core: [qe-quality-criteria-recommender, qe-risk-assessor, qe-requirements-validator]
  conditional: [qe-accessibility-auditor, qe-security-auditor, qe-qx-partner, qe-middleware-validator, qe-sap-rfc-tester, qe-sod-analyzer]
  total: 9
  sub_agents: 0
skills: [testability-scoring, risk-based-testing, context-driven-testing, holistic-testing-pact]
# Execution Models (Task Tool is PRIMARY)
execution:
  primary: task-tool
  alternatives: [mcp-tools, cli]
swarm_pattern: true
parallel_batches: 2
last_updated: 2026-01-28
# v7.5.1 Changelog: Added prominent follow-up recommendation box at end of swarm execution (Phase URL-9)
# v7.5.0 Changelog: Added HAS_VIDEO flag detection with /a11y-ally follow-up recommendation for video caption generation
# v7.4.0 Changelog: Automated browser cascade via scripts/fetch-content.js with 30s per-tier timeouts
# v7.2.0 Changelog: Added 5-tier browser cascade (Vibium → agent-browser → Playwright+Stealth → WebFetch → WebSearch)
html_output: true
enforcement_level: strict
tags: [qcsd, ideation, htsm, quality-criteria, risk-storming, testability, swarm, parallel, ddd]
trust_tier: 3
validation:
  schema_path: schemas/output.json
  validator_path: scripts/validate-config.json
  eval_path: evals/qcsd-ideation-swarm.yaml

---

# QCSD Ideation Swarm v7.0

Shift-left quality engineering swarm for PI Planning and Sprint Planning.

---

## URL-Based Analysis Mode (v7.1)

When analyzing a live website URL, use this specialized execution pattern.

### Parameters

- `URL`: Website to analyze (required)
- `OUTPUT_FOLDER`: Where to save reports (default: `${PROJECT_ROOT}/Agentic QCSD/{domain}/` or `./Agentic QCSD/{domain}/`)

---

## ⛔ URL MODE: COMPLETE EXECUTION FLOW

**You MUST follow ALL phases in order. Skipping phases is a FAILURE.**

### PHASE URL-1: Setup and Content Fetch (AUTOMATED CASCADE)

**The browser cascade is now FULLY AUTOMATED via `scripts/fetch-content.js`.**

**Single command - automatic tier fallback with 30s timeout per tier:**

```bash
# SINGLE COMMAND - handles all tiers automatically:
# Use npx for installed package, or node with relative path for local development
npx aqe fetch-content "${URL}" "${OUTPUT_FOLDER}" --timeout 30000
# OR if running from project root:
node ./scripts/fetch-content.js "${URL}" "${OUTPUT_FOLDER}" --timeout 30000
```

**What the script does automatically:**
1. Creates output folder
2. Tries Playwright+Stealth (30s timeout)
3. Falls back to HTTP Fetch (30s timeout)
4. Falls back to WebSearch placeholder (30s timeout)
5. Saves `content.html`, `screenshot.png`, and `fetch-result.json`

**Execution:**

```javascript
// 1. Run the automated fetch cascade (use relative path from project root)
const fetchResult = Bash({
  command: `node ./scripts/fetch-content.js "${URL}" "${OUTPUT_FOLDER}" --timeout 30000`,
  timeout: 120000  // 2 min total max
})

// 2. Parse the JSON result from stdout
const result = JSON.parse(fetchResult.stdout)

// 3. Read the content
const content = Read({ file_path: `${OUTPUT_FOLDER}/content.html` })
const fetchMethod = result.tier
const contentSize = result.contentSize
```

**If script is not available, fall back to inline Playwright:**

```javascript
// FALLBACK: Only if scripts/fetch-content.js doesn't exist
Bash({ command: `mkdir -p "${OUTPUT_FOLDER}"` })

// Quick Playwright fetch (single tier, no cascade)
Bash({
  command: `cd /tmp && rm -rf qcsd-fetch && mkdir qcsd-fetch && cd qcsd-fetch && npm init -y && npm install playwright-extra puppeteer-extra-plugin-stealth playwright 2>/dev/null`,
  timeout: 60000
})

// ... minimal inline script as last resort
```

**MANDATORY: Output fetch method used:**
```
┌─────────────────────────────────────────────────────────────┐
│                    CONTENT FETCH RESULT                     │
├─────────────────────────────────────────────────────────────┤
│  Method Used: [vibium/agent-browser/playwright/webfetch/    │
│               websearch-fallback]                           │
│  Content Size: [X KB]                                       │
│  Status: [SUCCESS/DEGRADED]                                 │
│                                                             │
│  If DEGRADED (websearch-fallback), analysis is based on     │
│  public information, not live page inspection.              │
└─────────────────────────────────────────────────────────────┘
```

### PHASE URL-2: Programmatic Flag Detection (MANDATORY)

**You MUST detect flags from the fetched content. Do NOT skip this phase.**

```javascript
// Detect HAS_UI
const HAS_UI = (
  /<(form|button|input|select|textarea|img|video|canvas|nav|header|footer|aside)/i.test(content) ||
  /carousel|slider|modal|dialog|dropdown|menu|tab|accordion/i.test(content) ||
  /class=["'][^"']*btn|button|card|grid|flex/i.test(content)
);

// Detect HAS_SECURITY
const HAS_SECURITY = (
  /login|password|auth|token|session|credential|oauth|jwt|sso/i.test(content) ||
  /newsletter|subscribe|signup|email.*input|register/i.test(content) || // PII collection
  /payment|checkout|credit.*card|billing/i.test(content) ||
  /cookie|consent|gdpr|privacy/i.test(content)
);

// Detect HAS_UX
const HAS_UX = (
  /user|customer|visitor|journey|experience|engagement/i.test(content) ||
  /<form/i.test(content) && /<button/i.test(content) || // Interactive forms
  /onboarding|wizard|step.*step|progress/i.test(content) ||
  /feedback|rating|review|comment/i.test(content)
);

// Detect HAS_VIDEO (for a11y-ally follow-up recommendation)
const HAS_VIDEO = (
  /<video/i.test(content) ||
  /youtube\.com\/embed|vimeo\.com|wistia\.com/i.test(content) ||
  /\.mp4|\.webm|\.m3u8/i.test(content) ||
  /data-video-url|data-mobile-url|data-desktop-url/i.test(content)
);

// Detect HAS_MIDDLEWARE
const HAS_MIDDLEWARE = (
  /middleware|ESB|message.?broker|MQ|Kafka|RabbitMQ/i.test(content) ||
  /integration.?bus|API.?gateway|message.?queue|pub.?sub/i.test(content)
);

// Detect HAS_SAP_INTEGRATION
const HAS_SAP_INTEGRATION = (
  /\bSAP\b|RFC|BAPI|IDoc|OData|S\/4HANA/i.test(content) ||
  /\bEWM\b|\bECC\b|\bABAP\b|CDS.?view|Fiori/i.test(content)
);

// Detect HAS_AUTHORIZATION
const HAS_AUTHORIZATION = (
  /\bSoD\b|segregation.?of.?duties|role.?conflict/i.test(content) ||
  /authorization.?object|T-?code|user.?role/i.test(content) ||
  /access.?control.?matrix|\bGRC\b/i.test(content)
);
```

**You MUST output flag detection results before proceeding:**

```
┌─────────────────────────────────────────────────────────────┐
│                    FLAG DETECTION RESULTS                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  HAS_UI:       [TRUE/FALSE]                                 │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  HAS_SECURITY: [TRUE/FALSE]                                 │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  HAS_UX:       [TRUE/FALSE]                                 │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  HAS_VIDEO:    [TRUE/FALSE]                                 │
│  Evidence:     [video URLs found - for a11y follow-up]      │
│                                                             │
│  HAS_MIDDLEWARE: [TRUE/FALSE]                               │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  HAS_SAP_INTEGRATION: [TRUE/FALSE]                          │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  HAS_AUTHORIZATION: [TRUE/FALSE]                            │
│  Evidence:     [what triggered it - specific patterns]      │
│                                                             │
│  EXPECTED AGENTS:                                           │
│  - Core: 3 (always)                                         │
│  - Conditional: [count based on TRUE flags]                 │
│  - TOTAL: [3 + conditional count]                           │
│                                                             │
│  FOLLOW-UP RECOMMENDED:                                     │
│  - /a11y-ally: [YES if HAS_VIDEO=TRUE, else NO]            │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**❌ DO NOT proceed to Phase URL-3 without outputting flag detection results.**

### PHASE URL-3: Spawn Core Agents (PARALLEL)

**All 3 core agents MUST be spawned. Fewer is a FAILURE.**

Spawn ALL THREE in a single message:

```javascript
// Agent 1: Quality Criteria (HTSM v6.3)
Task({
  description: "QCSD Quality Criteria Analysis",
  prompt: `You are qe-quality-criteria-recommender analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Analyze ALL 10 HTSM v6.3 categories with weight and testability score for each.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/02-quality-criteria-analysis.md
2. Use the Write tool to save BEFORE completing
3. Report MUST be complete - no placeholders`,
  subagent_type: "qe-quality-criteria-recommender",
  run_in_background: true
})

// Agent 2: Risk Assessment (SFDIPOT)
Task({
  description: "QCSD Risk Assessment",
  prompt: `You are qe-risk-assessor analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Apply SFDIPOT framework: Structure, Function, Data, Interfaces, Platform, Operations, Time.
Identify minimum 10 risks with probability, impact, and score.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/04-risk-assessment.md
2. Use the Write tool to save BEFORE completing
3. Report MUST be complete - no placeholders`,
  subagent_type: "qe-risk-assessor",
  run_in_background: true
})

// Agent 3: Requirements Validator (Testability)
Task({
  description: "QCSD Testability Assessment",
  prompt: `You are qe-requirements-validator analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Apply 10 Principles of Testability. Score each principle 0-100.
Identify blockers and recommendations.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/03-testability-assessment.md
2. Use the Write tool to save BEFORE completing
3. Report MUST be complete - no placeholders`,
  subagent_type: "qe-requirements-validator",
  run_in_background: true
})
```

### PHASE URL-4: Spawn Conditional Agents (PARALLEL)

**Spawn agents based on flags detected in Phase URL-2.**

```javascript
// IF HAS_UI === TRUE
Task({
  description: "QCSD Accessibility Audit",
  prompt: `You are qe-accessibility-auditor analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Perform WCAG 2.2 AA compliance assessment.
Identify accessibility barriers, missing ARIA, color contrast issues.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/07-accessibility-audit.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-accessibility-auditor",
  run_in_background: true
})

// IF HAS_SECURITY === TRUE
Task({
  description: "QCSD Security Threat Model",
  prompt: `You are qe-security-auditor analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Apply STRIDE threat modeling framework.
Identify vulnerabilities, attack vectors, and mitigations.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/05-security-threat-model.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-security-auditor",
  run_in_background: true
})

// IF HAS_UX === TRUE
Task({
  description: "QCSD Quality Experience Analysis",
  prompt: `You are qe-qx-partner analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Analyze user journeys, experience quality, friction points.
Map key user flows and identify UX risks.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/08-quality-experience.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-qx-partner",
  run_in_background: true
})

// IF HAS_MIDDLEWARE === TRUE
Task({
  description: "QCSD Middleware Integration Quality Criteria",
  prompt: `You are qe-middleware-validator analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Assess middleware and integration bus quality criteria.
Identify message routing risks, transformation issues, and integration patterns.
Evaluate ESB, message broker, API gateway, and pub/sub quality concerns.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/09-middleware-quality.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-middleware-validator",
  run_in_background: true
})

// IF HAS_SAP_INTEGRATION === TRUE
Task({
  description: "QCSD SAP Integration Quality Criteria",
  prompt: `You are qe-sap-rfc-tester analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Assess SAP-specific quality criteria including RFC, BAPI, IDoc, OData, and Fiori concerns.
Identify S/4HANA integration risks, CDS view dependencies, and ABAP quality issues.
Evaluate SAP interface stability, data consistency, and transport risks.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/10-sap-quality.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-sap-rfc-tester",
  run_in_background: true
})

// IF HAS_AUTHORIZATION === TRUE
Task({
  description: "QCSD Authorization Quality Criteria",
  prompt: `You are qe-sod-analyzer analyzing ${URL}.

## WEBSITE CONTENT
${content}

## ANALYSIS REQUIREMENTS
Assess authorization and segregation of duties quality criteria.
Identify SoD conflicts, role misconfigurations, and access control matrix gaps.
Evaluate T-code restrictions, authorization object coverage, and GRC compliance risks.

## OUTPUT REQUIREMENTS (MANDATORY)
1. Write your complete analysis to: ${OUTPUT_FOLDER}/11-authorization-quality.md
2. Use the Write tool to save BEFORE completing`,
  subagent_type: "qe-sod-analyzer",
  run_in_background: true
})
```

### PHASE URL-5: Agent Count Validation

**Before proceeding, verify agent count:**

```
┌─────────────────────────────────────────────────────────────┐
│                   AGENT COUNT VALIDATION                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  CORE AGENTS (ALWAYS 3):                                    │
│    □ qe-quality-criteria-recommender - SPAWNED? [Y/N]       │
│    □ qe-risk-assessor - SPAWNED? [Y/N]                      │
│    □ qe-requirements-validator - SPAWNED? [Y/N]             │
│                                                             │
│  CONDITIONAL AGENTS (based on flags):                       │
│    □ qe-accessibility-auditor - SPAWNED? [Y/N] (HAS_UI)     │
│    □ qe-security-auditor - SPAWNED? [Y/N] (HAS_SECURITY)    │
│    □ qe-qx-partner - SPAWNED? [Y/N] (HAS_UX)                │
│    □ qe-middleware-validator - SPAWNED? [Y/N] (HAS_MIDDLEWARE)│
│    □ qe-sap-rfc-tester - SPAWNED? [Y/N] (HAS_SAP_INTEG)     │
│    □ qe-sod-analyzer - SPAWNED? [Y/N] (HAS_AUTHORIZATION)   │
│                                                             │
│  VALIDATION:                                                │
│    Expected agents: [3 + count of TRUE flags]               │
│    Actual spawned:  [count]                                 │
│    Status:          [PASS/FAIL]                             │
│                                                             │
│  If ACTUAL < EXPECTED, you have FAILED. Spawn missing       │
│  agents before proceeding.                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**❌ DO NOT proceed if validation FAILS.**

### PHASE URL-6: Wait for Agents and Verify Reports

After spawning, inform user and wait:

```
I've launched [N] agents in background:
- 📊 Quality Criteria Recommender: HTSM v6.3 analysis → 02-quality-criteria-analysis.md
- ⚠️ Risk Assessor: SFDIPOT analysis → 04-risk-assessment.md
- 🧪 Requirements Validator: Testability assessment → 03-testability-assessment.md
[IF HAS_UI]
- ♿ Accessibility Auditor: WCAG 2.2 audit → 07-accessibility-audit.md
[IF HAS_SECURITY]
- 🔒 Security Auditor: STRIDE threat model → 05-security-threat-model.md
[IF HAS_UX]
- 🎯 QX Partner: User experience analysis → 08-quality-experience.md

Each agent will write directly to: ${OUTPUT_FOLDER}/
```

Verify reports exist before synthesis:
```bash
ls -la "${OUTPUT_FOLDER}"
```

### PHASE URL-7: Invoke Related Skills (MANDATORY)

**After agent reports are complete, invoke these skills:**

```javascript
// Testability Scoring - applies formal scoring methodology
Skill({ skill: "testability-scoring", args: `${OUTPUT_FOLDER}/03-testability-assessment.md` })

// Risk-Based Testing - prioritizes test selection
Skill({ skill: "risk-based-testing", args: `${OUTPUT_FOLDER}/04-risk-assessment.md` })
```

**Required Skill Invocations:**

| Skill | When | Purpose |
|-------|------|---------|
| `testability-scoring` | After testability report | Formal 0-100 score calculation |
| `risk-based-testing` | After risk report | Test prioritization matrix |
| `context-driven-testing` | Always | Apply CDT principles to recommendations |
| `holistic-testing-pact` | Always | Validate test strategy completeness |

**❌ Analysis is INCOMPLETE without invoking related skills.**

### PHASE URL-8: Synthesis and Executive Summary

After all agents complete and skills are invoked:

1. **Read all agent reports**
2. **Generate Executive Summary** → `${OUTPUT_FOLDER}/01-executive-summary.md`
3. **Generate Consolidated Test Ideas** → `${OUTPUT_FOLDER}/06-test-ideas.md`
4. **Add Follow-up Recommendations** (if HAS_VIDEO=TRUE):
   ```markdown
   ## Recommended Follow-up Actions

   | Action | Skill/Command | Reason |
   |--------|---------------|--------|
   | Generate Video Captions | `/a11y-ally ${URL}` | Video detected without captions - WCAG 1.2.2 compliance |
   ```
5. **Store learnings in memory**:
   ```bash
   npx @claude-flow/cli@latest memory store \
     --key "qcsd-${domain}-pattern" \
     --value "[key learnings from this analysis]" \
     --namespace patterns
   ```

**IMPORTANT:** If HAS_VIDEO=TRUE, the Executive Summary MUST include a "Recommended Follow-up Actions" section recommending `/a11y-ally` for video caption generation. This is NOT automatic - it's a recommendation for the user to run separately.

### Report Filename Mapping

| Agent | Report Filename |
|-------|----------------|
| qe-quality-criteria-recommender | `02-quality-criteria-analysis.md` |
| qe-requirements-validator | `03-testability-assessment.md` |
| qe-risk-assessor | `04-risk-assessment.md` |
| qe-security-auditor | `05-security-threat-model.md` |
| qe-accessibility-auditor | `07-accessibility-audit.md` |
| qe-qx-partner | `08-quality-experience.md` |
| Synthesis | `01-executive-summary.md` |
| Synthesis | `06-test-ideas.md` |

### PHASE URL-9: Final Output with Follow-up Recommendations (MANDATORY)

**At the very end of swarm execution, ALWAYS output this completion summary:**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    QCSD IDEATION SWARM COMPLETE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  URL Analyzed: ${URL}                                               │
│  Reports Generated: 8                                               │
│  Output Folder: ${OUTPUT_FOLDER}                                    │
│                                                                     │
│  QUALITY SCORES:                                                    │
│  ├─ Risk Assessment:    [score]                                     │
│  ├─ Security Posture:   [score]                                     │
│  ├─ Quality Experience: [score]                                     │
│  ├─ Testability:        [score]                                     │
│  └─ Accessibility:      [score]                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**IF HAS_VIDEO=TRUE, ALSO output this prominent recommendation box:**

```
┌─────────────────────────────────────────────────────────────────────┐
│  ⚠️  FOLLOW-UP ACTION RECOMMENDED                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  VIDEO DETECTED WITHOUT CAPTIONS                                    │
│                                                                     │
│  To generate WCAG 1.2.2 compliant captions, run:                    │
│                                                                     │
│    /a11y-ally ${URL}                                                │
│                                                                     │
│  This will:                                                         │
│  • Download video and extract frames                                │
│  • Analyze each frame with Claude Vision                            │
│  • Generate captions.vtt and audiodesc.vtt                          │
│  • Save to docs/accessibility-scans/{page-slug}/                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**❌ DO NOT end the swarm without displaying the completion summary.**
**❌ DO NOT skip the follow-up recommendation box if HAS_VIDEO=TRUE.**

---

## DDD Domain Integration

This swarm operates across **3 primary domains** and **3 conditional domains**:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QCSD IDEATION - DOMAIN MAP                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PRIMARY DOMAINS (Always Active)                                             │
│  ┌─────────────────────────┐ ┌─────────────────────────┐                    │
│  │  requirements-validation │ │    coverage-analysis    │                    │
│  │  ─────────────────────── │ │  ───────────────────── │                    │
│  │  • qe-quality-criteria-  │ │  • qe-risk-assessor    │                    │
│  │    recommender (PRIMARY) │ │                        │                    │
│  │  • qe-requirements-      │ │                        │                    │
│  │    validator             │ │                        │                    │
│  └─────────────────────────┘ └─────────────────────────┘                    │
│                                                                              │
│  CONDITIONAL DOMAINS (Based on Epic Content)                                 │
│  ┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐                │
│  │security-complnce│ │visual-a11y      │ │  cross-domain   │                │
│  │─────────────────│ │─────────────────│ │─────────────────│                │
│  │qe-security-     │ │qe-accessibility-│ │  qe-qx-partner  │                │
│  │auditor          │ │auditor          │ │                 │                │
│  │[IF HAS_SECURITY]│ │[IF HAS_UI]      │ │[IF HAS_UX]      │                │
│  └─────────────────┘ └─────────────────┘ └─────────────────┘                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Execution Model Options

This skill supports **3 execution models**. Choose based on your environment:

| Model | When to Use | Pros | Cons |
|-------|-------------|------|------|
| **Task Tool** (PRIMARY) | Claude Code sessions | Full agent capabilities, parallel execution | Requires Claude Code |
| **MCP Tools** | MCP server available | Fleet coordination, memory persistence | Requires MCP setup |
| **CLI** | Terminal/scripts | Works anywhere, scriptable | Sequential only |

### Quick Start by Model

**Option A: Task Tool (RECOMMENDED)**
```
Just follow the skill phases below - uses Task() calls with run_in_background: true
```

**Option B: MCP Tools**
```javascript
// Initialize fleet for Ideation domains
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["requirements-validation", "coverage-analysis", "security-compliance"],
  maxAgents: 6
})

// Orchestrate ideation task
mcp__agentic-qe__task_orchestrate({
  task: "qcsd-ideation-analysis",
  strategy: "parallel"
})
```

**Option C: CLI**
```bash
# Initialize coordination
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 6

# Route task
npx @claude-flow/cli@latest hooks pre-task --description "QCSD Ideation for [Epic]"

# Execute agents
npx @claude-flow/cli@latest agent spawn --type qe-quality-criteria-recommender
npx @claude-flow/cli@latest agent spawn --type qe-risk-assessor
npx @claude-flow/cli@latest agent spawn --type qe-requirements-validator
```

---

## ⛔ ENFORCEMENT RULES - READ FIRST

**These rules are NON-NEGOTIABLE. Violation means skill execution failure.**

| Rule | Enforcement |
|------|-------------|
| **E1** | You MUST spawn ALL THREE core agents in Phase 2. No exceptions. |
| **E2** | You MUST put all parallel Task calls in a SINGLE message. |
| **E3** | You MUST STOP and WAIT after each batch. No proceeding early. |
| **E4** | You MUST spawn conditional agents if flags are TRUE. No skipping. |
| **E5** | You MUST apply GO/CONDITIONAL/NO-GO logic exactly as specified. |
| **E6** | You MUST generate the full report structure. No abbreviated versions. |
| **E7** | Each agent MUST read its reference files before analysis. |

**❌ PROHIBITED BEHAVIORS:**
- Summarizing instead of spawning agents
- Skipping agents "for brevity"
- Proceeding before background tasks complete
- Providing your own analysis instead of spawning specialists
- Omitting report sections
- Using placeholder text like "[details here]"

---

## PHASE 1: Analyze Epic Content

**⚠️ MANDATORY: You must complete this analysis before Phase 2.**

Scan the epic content and SET these flags. Do not skip any flag.

### Flag Detection (Check ALL THREE)

```
□ HAS_UI = FALSE
  Set TRUE if epic contains ANY of: UI, frontend, visual, design,
  component, screen, page, form, button, modal, dialog, dashboard,
  widget, interface, display, view, layout, CSS, styling

□ HAS_SECURITY = FALSE
  Set TRUE if epic contains ANY of: auth, security, credential, token,
  encrypt, PII, compliance, password, login, session, OAuth, JWT,
  permission, role, access control, RBAC, sensitive, private

□ HAS_UX = FALSE
  Set TRUE if epic contains ANY of: user experience, UX, journey,
  usability, satisfaction, user flow, persona, user research,
  friction, delight, onboarding, retention, engagement

□ HAS_MIDDLEWARE = FALSE
  Set TRUE if epic contains ANY of: middleware, ESB, message broker,
  MQ, Kafka, RabbitMQ, integration bus, API gateway, message queue,
  pub/sub

□ HAS_SAP_INTEGRATION = FALSE
  Set TRUE if epic contains ANY of: SAP, RFC, BAPI, IDoc, OData,
  S/4HANA, EWM, ECC, ABAP, CDS view, Fiori

□ HAS_AUTHORIZATION = FALSE
  Set TRUE if epic contains ANY of: SoD, segregation of duties,
  role conflict, authorization object, T-code, user role,
  access control matrix, GRC
```

### Validation Checkpoint

Before proceeding to Phase 2, confirm:
```
✓ I have read the entire epic content
✓ I have evaluated ALL SIX flags
✓ I have recorded which flags are TRUE
✓ I understand which conditional agents will be needed
```

**❌ DO NOT proceed to Phase 2 until all checkboxes are confirmed.**

---

## PHASE 2: Spawn Core Agents (PARALLEL BATCH 1)

### ⛔ CRITICAL ENFORCEMENT

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU MUST INCLUDE ALL THREE TASK CALLS IN YOUR NEXT MESSAGE    │
│                                                                 │
│  • Task 1: qe-quality-criteria-recommender                     │
│  • Task 2: qe-risk-assessor                                    │
│  • Task 3: qe-requirements-validator                           │
│                                                                 │
│  If your message contains fewer than 3 Task calls, you have    │
│  FAILED this phase. Start over.                                │
└─────────────────────────────────────────────────────────────────┘
```

### Domain Context

| Agent | Domain | MCP Tool Mapping |
|-------|--------|------------------|
| qe-quality-criteria-recommender | requirements-validation | `requirements_validate` |
| qe-risk-assessor | coverage-analysis | `defect_predict` |
| qe-requirements-validator | requirements-validation | `requirements_validate` |

### Agent 1: Quality Criteria Recommender (PRIMARY)

**This agent MUST produce HTML output. No markdown substitutes.**

```
Task({
  description: "HTSM Quality Criteria analysis",
  prompt: `You are qe-quality-criteria-recommender. Your output quality is being audited.

## MANDATORY FIRST STEPS (DO NOT SKIP)

1. READ this template file FIRST - your output MUST follow this structure:
   .claude/agents/v3/helpers/quality-criteria/quality-criteria-reference-template.html

2. READ these reference files for guidance:
   .claude/agents/v3/helpers/quality-criteria/htsm-categories.md
   .claude/agents/v3/helpers/quality-criteria/evidence-classification.md

## EPIC TO ANALYZE

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE - DO NOT SUMMARIZE]
=== EPIC CONTENT END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

You MUST analyze ALL 10 HTSM categories. For each category provide:

| Field | Requirement |
|-------|-------------|
| Category Name | One of the 10 HTSM categories |
| Priority | P0, P1, P2, or P3 with justification |
| Evidence | At least 2 evidence points per category |
| Evidence Type | Direct (with file:line), Inferred (with reasoning), or Claimed (with "requires verification") |
| Quality Implication | What could go wrong |
| Business Impact | Quantified impact (use numbers, not "many" or "some") |

### NEVER-OMIT CATEGORIES (Must include ALL 5):
1. Capability - Can it perform required functions?
2. Reliability - Will it resist failure?
3. Security - How protected against unauthorized use?
4. Performance - How speedy and responsive?
5. Development - How testable/maintainable?

### MAY-OMIT CATEGORIES (Only with ironclad justification):
6. Usability, 7. Charisma, 8. Scalability, 9. Compatibility, 10. Installability

## OUTPUT FORMAT

Generate COMPLETE HTML report using the template structure.
Save to: .agentic-qe/quality-criteria/[epic-name]-htsm-analysis.html

## VALIDATION BEFORE SUBMITTING

✓ Did I read the template file?
✓ Did I analyze all 10 categories (or justify omissions)?
✓ Does every evidence point have proper classification?
✓ Are business impacts quantified with numbers?
✓ Is output in HTML format using template structure?`,
  subagent_type: "qe-quality-criteria-recommender",
  run_in_background: true
})
```

### Agent 2: Risk Assessor

**This agent MUST identify at least 5 risks. Fewer is a failure.**

```
Task({
  description: "Risk Storming analysis",
  prompt: `You are qe-risk-assessor. Your output quality is being audited.

## METHODOLOGY

Apply risk-based-testing methodology systematically.

## EPIC TO ANALYZE

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE - DO NOT SUMMARIZE]
=== EPIC CONTENT END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### Risk Identification Requirements

You MUST identify risks in ALL FOUR categories:
1. **Technical Risks** - Architecture, integration, dependencies, complexity
2. **Business Risks** - Revenue impact, user impact, compliance, reputation
3. **Quality Risks** - Testability, maintainability, reliability concerns
4. **Integration Risks** - Third-party services, APIs, data flows

**MINIMUM: 5 total risks. Target: 10+ risks.**

### For EACH Risk, Provide:

| Field | Requirement |
|-------|-------------|
| Risk ID | R001, R002, etc. |
| Description | Specific, actionable description (not vague) |
| Category | Technical, Business, Quality, or Integration |
| Likelihood | 1-5 scale with justification |
| Impact | 1-5 scale with justification |
| Risk Score | Likelihood × Impact |
| Mitigation | Specific mitigation strategy |
| Owner | Suggested owner (Dev, QE, Product, Ops) |

### Critical Risk Threshold
- Score ≥ 15 = CRITICAL (must be flagged prominently)
- Score 10-14 = HIGH
- Score 5-9 = MEDIUM
- Score < 5 = LOW

## OUTPUT FORMAT

Markdown with:
1. Executive Summary (top 3 risks in bold)
2. Risk Matrix Table (sorted by score descending)
3. Critical Risks Section (if any score ≥ 15)
4. Mitigation Priority List

## VALIDATION BEFORE SUBMITTING

✓ Did I identify at least 5 risks?
✓ Did I cover all 4 risk categories?
✓ Does every risk have likelihood, impact, AND score?
✓ Are critical risks (≥15) clearly flagged?
✓ Does every risk have a specific mitigation?`,
  subagent_type: "qe-risk-assessor",
  run_in_background: true
})
```

### Agent 3: Requirements Validator

**This agent MUST provide testability score 0-100. No ranges.**

```
Task({
  description: "AC validation and testability scoring",
  prompt: `You are qe-requirements-validator. Your output quality is being audited.

## METHODOLOGY

Apply context-driven-testing and testability-scoring principles.

## ACCEPTANCE CRITERIA TO VALIDATE

=== ACCEPTANCE CRITERIA START ===
[PASTE THE COMPLETE ACCEPTANCE CRITERIA HERE - DO NOT SUMMARIZE]
=== ACCEPTANCE CRITERIA END ===

## REQUIRED OUTPUT (ALL SECTIONS MANDATORY)

### 1. Testability Score (MANDATORY - SINGLE NUMBER)

Score each of the 10 testability principles (0-10 each):

| Principle | Score | Evidence |
|-----------|-------|----------|
| Controllability | X/10 | Can we control inputs/state? |
| Observability | X/10 | Can we observe outputs/behavior? |
| Isolability | X/10 | Can we test in isolation? |
| Separation of Concerns | X/10 | Are responsibilities clear? |
| Understandability | X/10 | Is behavior clearly specified? |
| Automatability | X/10 | Can tests be automated? |
| Heterogeneity | X/10 | Works across environments? |
| Simplicity | X/10 | Is complexity manageable? |
| Stability | X/10 | Are requirements stable? |
| Information Availability | X/10 | Do we have needed info? |

**TOTAL TESTABILITY SCORE: XX/100**

### 2. AC Completeness Assessment

For EACH acceptance criterion:

| AC ID | Text | INVEST Score | Issues | Testable? |
|-------|------|--------------|--------|-----------|
| AC1 | ... | X/6 | ... | Yes/No |

INVEST Criteria:
- **I**ndependent (can be tested alone)
- **N**egotiable (not over-specified)
- **V**aluable (delivers value)
- **E**stimable (can estimate effort)
- **S**mall (testable in one session)
- **T**estable (clear pass/fail)

**AC COMPLETENESS: XX%** (ACs that are fully testable / total ACs)

### 3. Gaps Identified (MANDATORY)

List ALL gaps found:
- Missing scenarios
- Unclear requirements
- Untestable criteria
- Ambiguous language
- Missing edge cases

**MINIMUM: Identify at least 3 gaps or explicitly state "No gaps found after thorough analysis"**

### 4. Recommendations

Specific, actionable recommendations to improve testability.

## VALIDATION BEFORE SUBMITTING

✓ Did I score all 10 testability principles?
✓ Did I calculate a single testability score (not a range)?
✓ Did I assess every AC against INVEST?
✓ Did I calculate AC completeness percentage?
✓ Did I identify gaps (or explicitly confirm none)?`,
  subagent_type: "qe-requirements-validator",
  run_in_background: true
})
```

### Alternative: MCP Tools Execution

If using MCP instead of Task tool:

```javascript
// Option 1: Orchestrate via Queen Coordinator
mcp__agentic-qe__fleet_init({
  topology: "hierarchical",
  enabledDomains: ["requirements-validation", "coverage-analysis"],
  maxAgents: 6,
  lazyLoading: true
})

// Submit tasks to specific domains
mcp__agentic-qe__task_submit({
  type: "quality-criteria-analysis",
  priority: "p0",
  payload: {
    epicContent: epicContent,
    outputFormat: "html",
    htsmVersion: "6.3"
  }
})

mcp__agentic-qe__task_submit({
  type: "risk-assessment",
  priority: "p0",
  payload: {
    epicContent: epicContent,
    riskCategories: ["technical", "business", "quality", "integration"]
  }
})

mcp__agentic-qe__task_submit({
  type: "requirements-validation",
  priority: "p0",
  payload: {
    acceptanceCriteria: acContent,
    scoreTestability: true,
    validateInvest: true
  }
})

// Check task status
mcp__agentic-qe__task_list({ status: "pending" })
```

### Alternative: CLI Execution

If using CLI instead of Task tool:

```bash
# Initialize swarm for ideation
npx @claude-flow/cli@latest swarm init \
  --topology hierarchical \
  --max-agents 6 \
  --strategy specialized

# Pre-task hook for routing
npx @claude-flow/cli@latest hooks pre-task \
  --description "QCSD Ideation: Quality Criteria, Risk Assessment, AC Validation"

# Spawn agents (run in separate terminals or background)
npx @claude-flow/cli@latest agent spawn \
  --type qe-quality-criteria-recommender \
  --task "Analyze HTSM categories for epic" &

npx @claude-flow/cli@latest agent spawn \
  --type qe-risk-assessor \
  --task "Risk storming analysis" &

npx @claude-flow/cli@latest agent spawn \
  --type qe-requirements-validator \
  --task "AC validation and testability scoring" &

# Wait for completion
wait

# Check swarm status
npx @claude-flow/cli@latest swarm status
```

### Post-Spawn Confirmation

After sending all three Task calls, you MUST tell the user:

```
I've launched 3 core agents in parallel:

🎯 qe-quality-criteria-recommender [Domain: requirements-validation]
   - Analyzing all 10 HTSM v6.3 categories
   - Collecting evidence with classifications
   - Generating HTML report

⚠️ qe-risk-assessor [Domain: coverage-analysis]
   - Identifying Technical, Business, Quality, Integration risks
   - Scoring likelihood × impact
   - Prioritizing mitigations

✅ qe-requirements-validator [Domain: requirements-validation]
   - Scoring testability (10 principles)
   - Validating ACs against INVEST
   - Identifying gaps

⏳ WAITING for all agents to complete before proceeding...
```

**❌ DO NOT proceed to Phase 3 until you have sent this confirmation.**

---

## PHASE 3: Wait for Batch 1 Completion

### ⛔ ENFORCEMENT: NO EARLY PROCEEDING

```
┌─────────────────────────────────────────────────────────────────┐
│  YOU MUST WAIT FOR ALL THREE BACKGROUND TASKS TO COMPLETE      │
│                                                                 │
│  ❌ DO NOT summarize what agents "would" find                   │
│  ❌ DO NOT proceed to Phase 4 early                             │
│  ❌ DO NOT provide your own analysis as substitute              │
│                                                                 │
│  ✓ WAIT for actual agent results                               │
│  ✓ ONLY proceed when all three have returned                   │
└─────────────────────────────────────────────────────────────────┘
```

### Results Extraction Checklist

When results return, extract and record:

```
From qe-quality-criteria-recommender:
□ htsmCoverage = __/10 categories analyzed
□ p0Count = __ P0 priority items
□ evidenceQuality = Direct __%, Inferred __%, Claimed __%

From qe-risk-assessor:
□ totalRisks = __ risks identified
□ criticalRisks = __ risks with score ≥ 15
□ topRiskScore = __ (highest score)

From qe-requirements-validator:
□ testabilityScore = __/100
□ acCompleteness = __%
□ gapsIdentified = __ gaps
```

**❌ DO NOT proceed to Phase 4 until ALL fields are filled.**

---

## PHASE 4: Spawn Conditional Agents (PARALLEL BATCH 2)

### ⛔ ENFORCEMENT: NO SKIPPING CONDITIONAL AGENTS

```
┌─────────────────────────────────────────────────────────────────┐
│  IF A FLAG IS TRUE, YOU MUST SPAWN THAT AGENT                  │
│                                                                 │
│  HAS_UI = TRUE           → MUST spawn qe-accessibility-auditor  │
│  HAS_SECURITY = TRUE     → MUST spawn qe-security-auditor      │
│  HAS_UX = TRUE           → MUST spawn qe-qx-partner            │
│  HAS_MIDDLEWARE = TRUE   → MUST spawn qe-middleware-validator   │
│  HAS_SAP_INTEGRATION = TRUE → MUST spawn qe-sap-rfc-tester     │
│  HAS_AUTHORIZATION = TRUE → MUST spawn qe-sod-analyzer         │
│                                                                 │
│  Skipping a flagged agent is a FAILURE of this skill.          │
└─────────────────────────────────────────────────────────────────┘
```

### Conditional Domain Mapping

| Flag | Agent | Domain | MCP Tool |
|------|-------|--------|----------|
| HAS_UI | qe-accessibility-auditor | visual-accessibility | `accessibility_test` |
| HAS_SECURITY | qe-security-auditor | security-compliance | `security_scan_comprehensive` |
| HAS_UX | qe-qx-partner | cross-domain | `task_orchestrate` |
| HAS_MIDDLEWARE | qe-middleware-validator | enterprise-integration | `task_orchestrate` |
| HAS_SAP_INTEGRATION | qe-sap-rfc-tester | enterprise-integration | `task_orchestrate` |
| HAS_AUTHORIZATION | qe-sod-analyzer | enterprise-integration | `task_orchestrate` |

### Decision Tree

```
IF HAS_UI == FALSE AND HAS_SECURITY == FALSE AND HAS_UX == FALSE AND HAS_MIDDLEWARE == FALSE AND HAS_SAP_INTEGRATION == FALSE AND HAS_AUTHORIZATION == FALSE:
    → Skip to Phase 5 (no conditional agents needed)
    → State: "No conditional agents needed based on epic analysis"

ELSE:
    → Spawn ALL applicable agents in ONE message
    → Count how many you're spawning: __
```

### IF HAS_UI: Accessibility Auditor (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Early accessibility review",
  prompt: `You are qe-accessibility-auditor. Your output quality is being audited.

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. UI Components Inventory

List EVERY UI component mentioned or implied:
| Component | Type | A11y Risk Level |
|-----------|------|-----------------|

### 2. WCAG 2.1 AA Risk Assessment

For each component, assess against:
- Perceivable (text alternatives, captions, adaptable, distinguishable)
- Operable (keyboard, timing, seizures, navigation)
- Understandable (readable, predictable, input assistance)
- Robust (compatible with assistive tech)

### 3. Keyboard Navigation Requirements

List ALL interactions that MUST support keyboard:
- [ ] Requirement 1
- [ ] Requirement 2
- ...

### 4. Screen Reader Considerations

What must be announced? What ARIA roles needed?

### 5. Findings Summary

| Finding | Severity | WCAG Criterion | Recommendation |
|---------|----------|----------------|----------------|

Severity: Critical (blocker), Major (significant barrier), Minor (inconvenience)

**MINIMUM: 3 findings or explicit "No accessibility risks identified after thorough analysis"**`,
  subagent_type: "qe-accessibility-auditor",
  run_in_background: true
})
```

### IF HAS_SECURITY: Security Auditor (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Security threat modeling",
  prompt: `You are qe-security-auditor. Your output quality is being audited.

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. STRIDE Threat Model

Analyze against ALL SIX categories:

| Threat Type | Applicable? | Threats Identified | Mitigations |
|-------------|-------------|-------------------|-------------|
| **S**poofing | Yes/No | ... | ... |
| **T**ampering | Yes/No | ... | ... |
| **R**epudiation | Yes/No | ... | ... |
| **I**nformation Disclosure | Yes/No | ... | ... |
| **D**enial of Service | Yes/No | ... | ... |
| **E**levation of Privilege | Yes/No | ... | ... |

### 2. Authentication/Authorization Requirements

- Auth method required: ___
- Session management: ___
- Permission model: ___

### 3. Data Protection Concerns

| Data Type | Classification | Protection Required |
|-----------|---------------|---------------------|
| ... | PII/Sensitive/Public | Encryption/Masking/None |

### 4. Compliance Implications

Check ALL that apply:
- [ ] GDPR (EU user data)
- [ ] CCPA (California user data)
- [ ] SOC 2 (security controls)
- [ ] HIPAA (health data)
- [ ] PCI-DSS (payment data)
- [ ] Other: ___

### 5. Security Testing Requirements

What security tests MUST be performed?

**MINIMUM: Identify threats in at least 3 STRIDE categories**`,
  subagent_type: "qe-security-auditor",
  run_in_background: true
})
```

### IF HAS_UX: QX Partner (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Quality Experience analysis",
  prompt: `You are qe-qx-partner. Your output quality is being audited.

## METHODOLOGY

Apply holistic-testing-pact methodology (PACT principles).

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. PACT Analysis

| Dimension | Analysis |
|-----------|----------|
| **P**eople | Who are the users? Personas? Needs? |
| **A**ctivities | What are they trying to do? Goals? |
| **C**ontexts | Where/when/how do they use this? |
| **T**echnologies | What tech constraints exist? |

### 2. User Personas Affected

| Persona | Impact Level | Key Concerns |
|---------|--------------|--------------|
| ... | High/Medium/Low | ... |

### 3. User Journey Impact

Map affected touchpoints:
```
[Entry] → [Step 1] → [Step 2] → [Exit]
           ↑ Impact    ↑ Impact
```

### 4. Quality Experience Risks

| QX Risk | User Feeling | Business Impact |
|---------|--------------|-----------------|
| ... | Frustrated/Confused/Delighted | ... |

### 5. UX Testing Recommendations

What UX-specific tests are needed?
- Usability testing needs
- User research gaps
- A/B testing candidates

**MINIMUM: Identify 3 QX risks or explicit "No QX risks after thorough analysis"**`,
  subagent_type: "qe-qx-partner",
  run_in_background: true
})
```

### IF HAS_MIDDLEWARE: Middleware Validator (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Middleware integration quality criteria",
  prompt: `You are qe-middleware-validator. Your output quality is being audited.

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Middleware Component Inventory

List EVERY middleware/integration component mentioned or implied:
| Component | Type | Quality Risk Level |
|-----------|------|-------------------|

### 2. Message Flow Quality Assessment

For each integration flow:
- Message routing reliability
- Transformation accuracy
- Dead letter queue handling
- Protocol mediation risks
- Retry and idempotency concerns

### 3. Integration Pattern Risks

| Pattern | Risk | Impact | Recommendation |
|---------|------|--------|----------------|

### 4. Quality Criteria Recommendations

What middleware-specific quality criteria MUST be validated?

**MINIMUM: 3 findings or explicit "No middleware quality risks after thorough analysis"**`,
  subagent_type: "qe-middleware-validator",
  run_in_background: true
})
```

### IF HAS_SAP_INTEGRATION: SAP RFC Tester (MANDATORY WHEN FLAGGED)

```
Task({
  description: "SAP-specific quality criteria",
  prompt: `You are qe-sap-rfc-tester. Your output quality is being audited.

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. SAP Interface Inventory

List EVERY SAP interface mentioned or implied:
| Interface | Type (RFC/BAPI/IDoc/OData) | Direction | Quality Risk |
|-----------|---------------------------|-----------|-------------|

### 2. SAP-Specific Quality Criteria

Assess quality concerns for:
- RFC/BAPI call reliability and error handling
- IDoc processing and status monitoring
- OData service contract stability
- S/4HANA migration compatibility
- CDS view dependencies and performance
- Fiori UI quality concerns (if applicable)

### 3. SAP Transport and Data Consistency

| Concern | Risk | Impact | Recommendation |
|---------|------|--------|----------------|

### 4. Quality Criteria Recommendations

What SAP-specific quality criteria MUST be validated?

**MINIMUM: 3 findings or explicit "No SAP quality risks after thorough analysis"**`,
  subagent_type: "qe-sap-rfc-tester",
  run_in_background: true
})
```

### IF HAS_AUTHORIZATION: SoD Analyzer (MANDATORY WHEN FLAGGED)

```
Task({
  description: "Authorization and SoD quality criteria",
  prompt: `You are qe-sod-analyzer. Your output quality is being audited.

## EPIC CONTENT

=== EPIC CONTENT START ===
[PASTE THE COMPLETE EPIC CONTENT HERE]
=== EPIC CONTENT END ===

## REQUIRED ANALYSIS (ALL SECTIONS MANDATORY)

### 1. Authorization Scope Inventory

List EVERY authorization concern mentioned or implied:
| Concern | Type | Risk Level |
|---------|------|-----------|

### 2. Segregation of Duties Assessment

For each SoD concern:
- Role conflict identification
- Authorization object coverage
- T-code restriction analysis
- Access control matrix completeness
- GRC compliance requirements

### 3. Authorization Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|

### 4. Quality Criteria Recommendations

What authorization-specific quality criteria MUST be validated?

**MINIMUM: 3 findings or explicit "No authorization quality risks after thorough analysis"**`,
  subagent_type: "qe-sod-analyzer",
  run_in_background: true
})
```

### Alternative: MCP Tools for Conditional Agents

```javascript
// IF HAS_UI - Enable visual-accessibility domain
if (HAS_UI) {
  mcp__agentic-qe__accessibility_test({
    url: targetUrl,  // if web-based
    standard: "WCAG21AA"
  })
}

// IF HAS_SECURITY - Enable security-compliance domain
if (HAS_SECURITY) {
  mcp__agentic-qe__security_scan_comprehensive({
    target: "src/",
    sast: true,
    dast: false  // No runtime yet in ideation
  })
}

// IF HAS_UX - Cross-domain analysis
if (HAS_UX) {
  mcp__agentic-qe__task_orchestrate({
    task: "qx-analysis",
    strategy: "adaptive"
  })
}
```

### Alternative: CLI for Conditional Agents

```bash
# IF HAS_UI
if [ "$HAS_UI" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-accessibility-auditor \
    --task "WCAG 2.1 AA assessment" &
fi

# IF HAS_SECURITY
if [ "$HAS_SECURITY" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-security-auditor \
    --task "STRIDE threat modeling" &
fi

# IF HAS_UX
if [ "$HAS_UX" = "TRUE" ]; then
  npx @claude-flow/cli@latest agent spawn \
    --type qe-qx-partner \
    --task "PACT quality experience analysis" &
fi

# Wait for conditional agents
wait
```

### Post-Spawn Confirmation (If Applicable)

```
I've launched [N] conditional agent(s) in parallel:

[IF HAS_UI] ♿ qe-accessibility-auditor [Domain: visual-accessibility] - WCAG 2.1 AA assessment
[IF HAS_SECURITY] 🔒 qe-security-auditor [Domain: security-compliance] - STRIDE threat modeling
[IF HAS_UX] 💫 qe-qx-partner [Domain: cross-domain] - PACT quality experience analysis

⏳ WAITING for conditional agents to complete...
```

---

## PHASE 5: Synthesize Results & Determine Recommendation

### ⛔ ENFORCEMENT: EXACT DECISION LOGIC

**You MUST apply this logic EXACTLY. No interpretation.**

```
STEP 1: Check NO-GO conditions (ANY triggers NO-GO)
─────────────────────────────────────────────────
IF testabilityScore < 40        → NO-GO (reason: "Testability critically low")
IF htsmCoverage < 6             → NO-GO (reason: "Insufficient quality coverage")
IF acCompleteness < 50          → NO-GO (reason: "Acceptance criteria incomplete")
IF criticalRisks > 2            → NO-GO (reason: "Too many critical risks")

STEP 2: Check GO conditions (ALL required for GO)
─────────────────────────────────────────────────
IF testabilityScore >= 80
   AND htsmCoverage >= 8
   AND acCompleteness >= 90
   AND criticalRisks == 0       → GO

STEP 3: Default
─────────────────────────────────────────────────
ELSE                            → CONDITIONAL
```

### Decision Recording

```
METRICS:
- testabilityScore = __/100
- htsmCoverage = __/10
- acCompleteness = __%
- criticalRisks = __

NO-GO CHECK:
- testabilityScore < 40? __ (YES/NO)
- htsmCoverage < 6? __ (YES/NO)
- acCompleteness < 50? __ (YES/NO)
- criticalRisks > 2? __ (YES/NO)

GO CHECK (only if no NO-GO triggered):
- testabilityScore >= 80? __ (YES/NO)
- htsmCoverage >= 8? __ (YES/NO)
- acCompleteness >= 90? __ (YES/NO)
- criticalRisks == 0? __ (YES/NO)

FINAL RECOMMENDATION: [GO / CONDITIONAL / NO-GO]
REASON: ___
```

---

## PHASE 6: Generate Ideation Report

### ⛔ ENFORCEMENT: COMPLETE REPORT STRUCTURE

**ALL sections below are MANDATORY. No abbreviations.**

```markdown
# QCSD Ideation Report: [Epic Name]

**Generated**: [Date/Time]
**Recommendation**: [GO / CONDITIONAL / NO-GO]
**Agents Executed**: [List all agents that ran]

---

## Executive Summary

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| HTSM Coverage | X/10 | ≥8 | ✅/⚠️/❌ |
| Testability Score | X% | ≥80% | ✅/⚠️/❌ |
| AC Completeness | X% | ≥90% | ✅/⚠️/❌ |
| Critical Risks | X | 0 | ✅/⚠️/❌ |

**Recommendation Rationale**: [1-2 sentences explaining why GO/CONDITIONAL/NO-GO]

---

## Quality Criteria Analysis (HTSM v6.3)

[EMBED or LINK the HTML report from qe-quality-criteria-recommender]

### Priority Items Summary

| Priority | Count | Categories |
|----------|-------|------------|
| P0 (Critical) | X | [list] |
| P1 (High) | X | [list] |
| P2 (Medium) | X | [list] |
| P3 (Low) | X | [list] |

### Cross-Cutting Concerns
[List any concerns that span multiple categories]

---

## Risk Assessment

### Risk Matrix

| ID | Risk | Category | L | I | Score | Mitigation |
|----|------|----------|---|---|-------|------------|
[ALL risks from qe-risk-assessor, sorted by score]

### Critical Risks (Score ≥ 15)
[Highlight critical risks with detailed mitigation plans]

### Risk Distribution
- Technical: X risks
- Business: X risks
- Quality: X risks
- Integration: X risks

---

## Requirements Validation

### Testability Score: X/100

| Principle | Score | Notes |
|-----------|-------|-------|
[All 10 principles from qe-requirements-validator]

### AC Completeness: X%

| AC | Status | Issues |
|----|--------|--------|
[All ACs evaluated]

### Gaps Identified
1. [Gap 1]
2. [Gap 2]
[All gaps from qe-requirements-validator]

---

## Conditional Analysis

[INCLUDE ONLY IF APPLICABLE - based on which conditional agents ran]

### Accessibility Review (IF HAS_UI)
[Full output from qe-accessibility-auditor]

### Security Assessment (IF HAS_SECURITY)
[Full output from qe-security-auditor]

### Quality Experience (IF HAS_UX)
[Full output from qe-qx-partner]

---

## Recommended Next Steps

### Immediate Actions (Before Development)
- [ ] [Action based on findings]
- [ ] [Action based on findings]

### During Development
- [ ] [Action based on findings]

### Pre-Release
- [ ] [Action based on findings]

---

## Appendix: Agent Outputs

[Link to or embed full outputs from each agent]

---

*Generated by QCSD Ideation Swarm v6.1*
*Execution Model: Task Tool Parallel Swarm*
```

### Report Validation Checklist

Before presenting report:
```
✓ Executive Summary table is complete with all 4 metrics
✓ Recommendation matches decision logic output
✓ Quality Criteria section includes priority summary
✓ Risk Matrix includes ALL identified risks
✓ Testability score shows all 10 principles
✓ All gaps are listed
✓ Conditional sections included for all spawned agents
✓ Next steps are specific and actionable (not generic)
```

**❌ DO NOT present an incomplete report.**

---

## PHASE 7: Store Learnings & Persist State

### Purpose

Store ideation findings for:
- Cross-phase feedback loops (Production → next Ideation cycle)
- Historical analysis of GO/CONDITIONAL/NO-GO decisions
- Pattern learning across epics

### Option A: MCP Memory Tools (RECOMMENDED)

```javascript
// Store ideation findings
mcp__agentic-qe__memory_store({
  key: `qcsd-ideation-${epicId}-${Date.now()}`,
  namespace: "qcsd-ideation",
  value: {
    epicId: epicId,
    epicName: epicName,
    recommendation: recommendation,  // GO, CONDITIONAL, NO-GO
    metrics: {
      htsmCoverage: htsmCoverage,
      testabilityScore: testabilityScore,
      acCompleteness: acCompleteness,
      criticalRisks: criticalRisks
    },
    domains: {
      requirementsValidation: true,
      coverageAnalysis: true,
      securityCompliance: HAS_SECURITY,
      visualAccessibility: HAS_UI,
      crossDomain: HAS_UX
    },
    agentsInvoked: agentList,
    timestamp: new Date().toISOString()
  }
})

// Share learnings with learning coordinator for cross-domain patterns
mcp__agentic-qe__memory_share({
  sourceAgentId: "qcsd-ideation-swarm",
  targetAgentIds: ["qe-learning-coordinator", "qe-pattern-learner"],
  knowledgeDomain: "ideation-patterns"
})

// Query previous ideation results for similar epics
mcp__agentic-qe__memory_query({
  pattern: "qcsd-ideation-*",
  namespace: "qcsd-ideation"
})
```

### Option B: CLI Memory Commands

```bash
# Store ideation findings
npx @claude-flow/cli@latest memory store \
  --key "qcsd-ideation-${EPIC_ID}" \
  --value '{"recommendation":"GO","testabilityScore":85,"htsmCoverage":9}' \
  --namespace qcsd-ideation

# Search for similar epics
npx @claude-flow/cli@latest memory search \
  --query "ideation recommendation" \
  --namespace qcsd-ideation

# List all ideation records
npx @claude-flow/cli@latest memory list \
  --namespace qcsd-ideation

# Post-task hook for learning
npx @claude-flow/cli@latest hooks post-task \
  --task-id "qcsd-ideation-${EPIC_ID}" \
  --success true
```

### Option C: Direct File Storage (Fallback)

If MCP/CLI not available, save to `.agentic-qe/`:

```bash
# Output directory structure
.agentic-qe/
├── quality-criteria/
│   └── [epic-name]-htsm-analysis.html
├── ideation-reports/
│   └── [epic-name]-ideation-report.md
└── learnings/
    └── [epic-id]-ideation-metrics.json
```

---

## Quick Reference

### Enforcement Summary

| Phase | Must Do | Failure Condition |
|-------|---------|-------------------|
| 1 | Check ALL 3 flags | Missing flag evaluation |
| 2 | Spawn ALL 3 core agents in ONE message | Fewer than 3 Task calls |
| 3 | WAIT for completion | Proceeding before results |
| 4 | Spawn ALL flagged conditional agents | Skipping a TRUE flag |
| 5 | Apply EXACT decision logic | Wrong recommendation |
| 6 | Generate COMPLETE report | Missing sections |
| 7 | Store learnings (if MCP/CLI available) | Pattern loss |

### Quality Gate Thresholds

| Metric | GO | CONDITIONAL | NO-GO |
|--------|-----|-------------|-------|
| Testability | ≥80% | 40-79% | <40% |
| HTSM Coverage | ≥8/10 | 6-7/10 | <6/10 |
| AC Completeness | ≥90% | 50-89% | <50% |
| Critical Risks | 0 | 1-2 | >2 |

### Domain-to-Agent Mapping

| Domain | Agent | Primary Phase |
|--------|-------|---------------|
| requirements-validation | qe-quality-criteria-recommender | Ideation (P) |
| requirements-validation | qe-requirements-validator | Ideation (P) |
| coverage-analysis | qe-risk-assessor | Ideation (P) |
| security-compliance | qe-security-auditor | Ideation (S - conditional) |
| visual-accessibility | qe-accessibility-auditor | Ideation (S - conditional) |
| cross-domain | qe-qx-partner | Ideation (S - conditional) |

### Execution Model Quick Reference

| Model | Initialization | Agent Spawn | Memory Store |
|-------|---------------|-------------|--------------|
| **Task Tool** | N/A | `Task({ subagent_type, run_in_background: true })` | N/A (use MCP) |
| **MCP Tools** | `fleet_init({})` | `task_submit({})` | `memory_store({})` |
| **CLI** | `swarm init` | `agent spawn` | `memory store` |

### MCP Tools Quick Reference

```javascript
// Initialization
mcp__agentic-qe__fleet_init({ topology: "hierarchical", enabledDomains: [...], maxAgents: 6 })

// Task submission
mcp__agentic-qe__task_submit({ type: "...", priority: "p0", payload: {...} })
mcp__agentic-qe__task_orchestrate({ task: "...", strategy: "parallel" })

// Status
mcp__agentic-qe__fleet_status({ verbose: true })
mcp__agentic-qe__task_list({ status: "pending" })

// Memory
mcp__agentic-qe__memory_store({ key: "...", value: {...}, namespace: "qcsd-ideation" })
mcp__agentic-qe__memory_query({ pattern: "qcsd-*", namespace: "qcsd-ideation" })
mcp__agentic-qe__memory_share({ sourceAgentId: "...", targetAgentIds: [...], knowledgeDomain: "..." })
```

### CLI Quick Reference

```bash
# Initialization
npx @claude-flow/cli@latest swarm init --topology hierarchical --max-agents 6

# Agent operations
npx @claude-flow/cli@latest agent spawn --type [agent-type] --task "[description]"
npx @claude-flow/cli@latest hooks pre-task --description "[task]"
npx @claude-flow/cli@latest hooks post-task --task-id "[id]" --success true

# Status
npx @claude-flow/cli@latest swarm status

# Memory
npx @claude-flow/cli@latest memory store --key "[key]" --value "[json]" --namespace qcsd-ideation
npx @claude-flow/cli@latest memory search --query "[query]" --namespace qcsd-ideation
npx @claude-flow/cli@latest memory list --namespace qcsd-ideation
```

### Swarm Topology

```
              QCSD IDEATION SWARM v7.0
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
│Quality  │   │   Risk    │   │    AC     │
│Criteria │   │ Assessor  │   │ Validator │
│ (HTML)  │   │           │   │           │
│─────────│   │───────────│   │───────────│
│req-valid│   │cov-anlysis│   │req-valid  │
└────┬────┘   └─────┬─────┘   └─────┬─────┘
     │               │               │
     └───────────────┼───────────────┘
                     │
              [QUALITY GATE]
                     │
     ┌───────────────┼───────────────┐
     │               │               │
┌────▼────┐   ┌─────▼─────┐   ┌─────▼─────┐
│  A11y   │   │ Security  │   │    QX     │
│[IF UI]  │   │[IF AUTH]  │   │ [IF UX]   │
│─────────│   │───────────│   │───────────│
│vis-a11y │   │sec-compli │   │cross-dom  │
└─────────┘   └───────────┘   └───────────┘
```

---

## Inventory Summary

| Resource Type | Count | Primary | Conditional |
|---------------|:-----:|:-------:|:-----------:|
| **Agents** | 9 | 3 | 6 |
| **Sub-agents** | 0 | - | - |
| **Skills** | 4 | 4 | - |
| **Domains** | 6 | 2 | 4 |

**Skills Used:**
1. `testability-scoring` - 10 testability principles
2. `risk-based-testing` - Risk prioritization
3. `context-driven-testing` - Context-appropriate strategy
4. `holistic-testing-pact` - PACT methodology (People, Activities, Contexts, Technologies)

---

## Key Principle

**Quality is built in from the start, not tested in at the end.**

This swarm provides:
1. **What quality criteria matter?** → HTSM Analysis (10 categories)
2. **What risks exist?** → Risk Storming (4 categories)
3. **Are requirements testable?** → AC Validation (10 principles)
4. **Is it accessible/secure/good UX?** → Conditional specialists
5. **Should we proceed?** → GO/CONDITIONAL/NO-GO decision
6. **What did we learn?** → Memory persistence for future cycles
