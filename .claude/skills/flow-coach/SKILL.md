---
name: flow-coach
description: >-
  Self-learning claude-flow v3 orchestration coach with DAXIOM intelligence + Guidance governance.
  Compiles CLAUDE.md into enforceable policy, retrieves task-scoped rules, enforces gates, tracks trust,
  and proves decisions. Enables 10x-100x longer autonomy. Queries PostgreSQL for proven patterns before
  recommending, learns from outcomes, and saves intelligence for future agents. Falls back to defaults
  when no patterns match (< 0.5 similarity). NEVER auto-executes.
trust_tier: 3
domain: orchestration
author: ruvnet
version: 2.0.0
---

# Flow Coach

Interactive claude-flow v3 orchestration coach. Analyzes your task, selects optimal topology and SPARC
phases, applies coaching shortcuts, then **pauses for your approval** before executing anything.

## Arguments

- `<task_description>` -- Natural-language description of the work to be done.
  If omitted, prompt the user.

Inline shortcuts can appear anywhere in the description:
`just recommend`, `simpler setup`, `full power`, `enable security`, `enable testing`,
`enable sparc`, `sparc lite`, `new project setup`, `pre-release check`.

---

## Phase 1 -- Task Analysis

Parse the `<task_description>` and extract:

1. **Task type** -- one of: `new-feature`, `bug-fix`, `refactoring`, `prototype`,
   `documentation`, `research`, `deployment`, `migration`, `unknown`.
2. **Domain keywords** -- scan for matches against domain trigger lists
   (auth/oauth/jwt, api/rest/graphql, fix/bug/error, refactor, deploy/release,
   database/migration).
3. **Complexity estimate** -- `low` (<3 files, single concern), `medium` (3-10 files,
   multiple concerns), `high` (>10 files or cross-cutting).
4. **Active shortcuts** -- scan description for shortcut phrases (case-insensitive).

If the AQE routing hook has already fired, read its `[TASK_MODEL_RECOMMENDATION]`
output as an additional signal (recommended agent type + confidence).

---

## Phase 2 -- Dynamic Topology Selection

Apply the first matching rule from top to bottom. If a shortcut overrides topology,
the shortcut wins (see Phase 4).

| # | Condition | Topology | Agents |
|---|-----------|----------|--------|
| 1 | Research-only, no code changes expected | `mesh` | 3-5 |
| 2 | Sequential stages (build -> test -> deploy) | `ring` | = stage count |
| 3 | Centralized control needed (single coordinator) | `star` | 4-6 |
| 4 | Multiple independent teams or modules | `hierarchical-mesh` | 8-15 |
| 5 | Quick fix / small change, complexity = low | `mesh` | 2-3 |
| 6 | Complex multi-day effort, complexity = high | `hierarchical-mesh` | 8-15 |
| 7 | Standard complexity | `hierarchical` | 5-8 |
| 8 | Task description mentions learning or adaptive | `adaptive` | 5-8 |
| 9 | Default (nothing else matched) | `hierarchical` | 6 |

Record which rule matched and why (used by `[X] Explain` later).

---

## Phase 3 -- SPARC Phase Auto-Selection

Based on the task type from Phase 1, select the applicable SPARC phases and
any extra agent roles to spawn.

| Task Type | SPARC Phases | Extra Agents |
|-----------|-------------|--------------|
| `new-feature` | Spec -> Pseudo -> Arch -> Refine -> Complete | (from domain triggers) |
| `bug-fix` | Refine | debugger, tester |
| `refactoring` | Arch -> Refine -> Complete | analyzer, reviewer |
| `prototype` | Pseudo -> Refine | researcher |
| `documentation` | Complete | documenter |
| `research` | *(no SPARC)* | researcher, analyzer |
| `deployment` | Complete | devops |
| `migration` | Arch -> Refine -> Complete | analyzer, tester |
| `unknown` | Spec -> Pseudo -> Arch -> Refine -> Complete | *(none extra)* |

### Domain Trigger Overrides

Scan the task description for domain keywords. Each match adds agents and/or
forces additional SPARC phases. Multiple triggers stack.

| Keywords | Extra Agents | Phase Override |
|----------|-------------|----------------|
| `auth`, `oauth`, `jwt`, `login`, `session` | + security-auditor | Force full SPARC (all 5 phases) |
| `api`, `rest`, `graphql`, `endpoint` | + documenter, + tester | Force full SPARC |
| `fix`, `bug`, `error`, `broken`, `crash` | + tester | Ensure Refine phase present |
| `refactor`, `restructure`, `clean up` | *(none)* | Ensure Arch -> Complete phases |
| `deploy`, `release`, `ship`, `publish` | *(none)* | Apply pre-release check pattern |
| `database`, `migration`, `schema`, `sql` | *(none)* | Ensure Arch phase present |

Merge phases using set-union (no duplicates, maintain canonical order:
Spec -> Pseudo -> Arch -> Refine -> Complete).

---

## Phase 4 -- Coaching Shortcuts

Process any shortcuts detected in Phase 1. Shortcuts are applied **after** Phases 2-3,
and later shortcuts override earlier ones for topology/agents. Phase lists merge via
set-union.

| Shortcut | Topology Override | Agent Override | Phase Override |
|----------|-------------------|---------------|----------------|
| `just recommend` | *(no change)* | *(no change)* | *(no change)* -- suppress verbose output, show assessment box only |
| `simpler setup` | `mesh` | 3 agents | *(clear all SPARC phases and domain triggers)* |
| `full power` | `hierarchical-mesh` | 15 agents | Full SPARC + all domain triggers active |
| `enable security` | *(no change)* | + security-auditor, + security-scanner | + Spec, + Arch |
| `enable testing` | *(no change)* | + tester, + coverage-specialist | + Refine |
| `enable sparc` | *(no change)* | *(no change)* | Full SPARC (all 5 phases) |
| `sparc lite` | *(no change)* | *(no change)* | Pseudo + Refine only |
| `new project setup` | `hierarchical-mesh` | 10 agents | Full SPARC + security + testing + documenter |
| `pre-release check` | `star` | 5 agents: security-scanner, perf-analyzer, coverage-specialist, tester, reviewer | *(clear SPARC -- validation only)* |

---

## Phase 5 -- Assessment Display

Compile the results of Phases 1-4 into a single assessment box. If the `just recommend`
shortcut is active, skip reasoning text and output only the box.

Format:

```
+---------------------------------------------------------------+
|                    FLOW COACH ASSESSMENT                      |
+---------------------------------------------------------------+
| Task:       <first 60 chars of description>                   |
| Type:       <task_type>                                       |
| Complexity: <low | medium | high>                             |
+---------------------------------------------------------------+
| Topology:   <selected topology>  (Rule #<N>)                  |
| Agents:     <count>                                           |
| Roles:      <comma-separated list of agent types>             |
+---------------------------------------------------------------+
| SPARC:      <phase list or "None">                            |
| Domains:    <triggered domains or "None">                     |
| Shortcuts:  <active shortcuts or "None">                      |
+---------------------------------------------------------------+
```

Then present the interactive menu (Phase 6).

---

## Phase 6 -- Interactive Pause

**CRITICAL: Do NOT call `swarm_init`, `agent_spawn`, or the Task tool before the user
selects [E]. This is a hard gate.**

Present:

```
What would you like to do?

[E] Execute  - Launch swarm with this configuration
[M] Modify   - Change topology, agent count, or strategy
[A] Add      - Add agents, SPARC phases, or domains
[R] Remove   - Remove agents, phases, or features
[X] Explain  - Explain why this configuration was chosen
```

Use the `AskUserQuestion` tool with these five options.

### On `[E]` Execute

Proceed to Phase 7.

### On `[M]` Modify

Ask what to change (topology, agent count, strategy). Apply the change,
recompute any dependent values, then re-display the assessment box (Phase 5)
and re-present this menu.

### On `[A]` Add

Ask what to add (agent roles, SPARC phases, domain triggers). Merge into
current config using set-union, then re-display and re-present.

### On `[R]` Remove

Ask what to remove. Remove from current config, then re-display and re-present.

### On `[X]` Explain

Output a structured explanation:

1. **Topology rationale** -- which rule matched and why.
2. **SPARC rationale** -- task type mapping + domain overrides.
3. **Agent rationale** -- base roles + domain-triggered additions.
4. **Shortcut effects** -- what each active shortcut changed.
5. **AQE signal** -- if AQE routing fired, show recommended agent + confidence.

Then re-present the interactive menu (do NOT execute).

---

## Phase 7 -- Execution

Only reached when user selected `[E]`.

### Step 1: Initialize Swarm

Call `mcp__claude-flow__swarm_init` with the selected topology and max agents:

```
mcp__claude-flow__swarm_init({
  topology: "<selected_topology>",
  maxAgents: <agent_count>,
  strategy: "specialized"
})
```

### Step 2: Spawn Agents

For each role in the agent list, spawn via the Task tool with `run_in_background: true`.
All spawns go in a **single message** (parallel execution per CLAUDE.md rules).

```
Task({
  description: "<role> agent for <task>",
  prompt: "<task-specific instructions for this role>",
  subagent_type: "<matching agent type>",
  run_in_background: true
})
```

### Step 3: SPARC Orchestration (if phases selected)

If SPARC phases are active, coordinate them in order. Each phase gates the next --
the previous phase must complete before the next begins.

Phase-to-agent mapping:
- **Spec** -> `specification` agent
- **Pseudo** -> `pseudocode` agent
- **Arch** -> `architecture` agent
- **Refine** -> `refinement` agent
- **Complete** -> `sparc-coder` agent

### Step 4: Report

After all agents complete, summarize results and present:

```
Swarm execution complete.
- Topology: <topology>
- Agents spawned: <count>
- SPARC phases completed: <list>
- Duration: <elapsed>
```

---

## Rules

- NEVER call `swarm_init`, `agent_spawn`, or spawn Task tool agents before user selects `[E]`
- NEVER auto-execute -- always pause at Phase 6
- ALWAYS display the assessment box before the interactive menu
- When `just recommend` is active, suppress verbose analysis -- box + menu only
- Shortcuts stack: later shortcuts override topology/agents; phases merge via set-union
- Domain triggers always stack (set-union for agents and phases)
- If blocked or uncertain, stop and ask the user -- do not guess
- Respect CLAUDE.md concurrency rules: all agent spawns in one message
- If AQE routing has fired, reference its output but do not let it override the user's menu choice

### CLI Path Rule (CRITICAL)

**NEVER use `npx @claude-flow/cli@latest` or `npx claude-flow@alpha` for Bash commands.**
These pull from npm and are NOT initialized in this project.

The claude-flow MCP tools (`mcp__claude-flow__swarm_init`, `mcp__claude-flow__memory_store`, etc.)
are the **primary** interface — they route through the local patched install automatically.

If a Bash CLI fallback is needed, use the **local install path**:

```bash
# CORRECT — local patched install
node /Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@claude-flow/cli/bin/cli.js <command>

# WRONG — pulls from npm, not initialized
npx @claude-flow/cli@latest <command>
npx claude-flow@alpha <command>
```

**Priority order:**
1. MCP tools (always preferred — use `mcp__claude-flow__*`)
2. Local CLI via Bash (fallback only)
3. NEVER use npx for claude-flow commands

---

## Examples

### Example 1: REST API with Auth

```
/flow-coach build a REST API with auth
```

**Expected assessment:**
- Type: `new-feature`
- Topology: `hierarchical` (Rule #7, standard complexity)
- Agents: 5-8, including security-auditor (auth trigger), documenter + tester (api trigger)
- SPARC: Full (Spec -> Pseudo -> Arch -> Refine -> Complete) -- forced by auth + api triggers
- Interactive menu displayed, NO auto-execution

### Example 2: Quick Bug Fix

```
/flow-coach just recommend fix login bug
```

**Expected assessment:**
- Type: `bug-fix`
- Topology: `mesh` (Rule #5, quick fix)
- Agents: 2-3, including tester (fix/bug trigger)
- SPARC: Refine only
- `just recommend` active: box only, no verbose analysis
- Interactive menu displayed

### Example 3: Full Power

```
/flow-coach full power implement payment system
```

**Expected assessment:**
- Type: `new-feature`
- Topology: `hierarchical-mesh` (shortcut override)
- Agents: 15 (shortcut override)
- SPARC: Full + all domain triggers active
- Interactive menu displayed
