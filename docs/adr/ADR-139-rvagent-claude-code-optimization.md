# ADR-139: RVAgent Optimization Using Decompiled Claude Code Intelligence

## Status

Proposed

## Date

2026-04-03

## Context

ruDevolution's decompilation of Claude Code v2.1.91 revealed 34,759 declarations, 498 environment variables, 6 permission modes, 25+ tools, and multiple unreleased features. This intelligence enables RVAgent (claude-flow) to optimize its integration with Claude Code at a depth no other tool achieves.

### What the Decompilation Revealed

| Discovery | Optimization Opportunity |
|-----------|------------------------|
| Agent loop is async generator (`s$`) yielding 13 event types | Match event handling exactly — no guessing |
| `CLAUDE_CODE_SUBAGENT_MODEL` env var | Override model per subagent for cost optimization |
| `CLAUDE_CODE_PLAN_V2_AGENT_COUNT` / `EXPLORE_AGENT_COUNT` | Control Plan V2 parallelism directly |
| `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | Enable multi-agent swarm mode |
| `CLAUDE_CODE_TEAMMATE_COMMAND` | Inject teammate discovery |
| `CLAUDE_CODE_IS_COWORK` | Enable cowork collaboration |
| `CLAUDE_CODE_IDLE_THRESHOLD_MINUTES` | Control when auto-dream activates |
| `CLAUDE_CODE_ENABLE_TASKS` | Enable built-in task management |
| `CLAUDE_CODE_DISABLE_CRON` | Ensure cron scheduling stays active |
| `CLAUDE_CODE_REPL` | REPL mode for interactive sessions |
| `CLAUDE_CODE_BRIEF` | Compressed output for faster agent communication |
| `CLAUDE_CODE_EMIT_TOOL_USE_SUMMARIES` | Get tool execution summaries |
| `CLAUDE_CODE_EMIT_SESSION_STATE_EVENTS` | Subscribe to session state changes |
| Permission modes: `acceptEdits`, `bypassPermissions`, `auto` | Set optimal permission per agent type |
| `clear_tool_uses_20250919` API feature | Trigger context compaction programmatically |
| Deferred tool loading via `ToolSearch` | Lazy-load tools to reduce token usage |
| `promptCacheSharingEnabled` | Share prompt cache across agents |
| `CLAUDE_CODE_EFFORT_LEVEL` | Control reasoning depth per task |

## Decision

Optimize RVAgent across 5 dimensions using decompiled intelligence.

### 1. Environment Variable Injection

RVAgent hooks set optimal env vars before each Claude Code session based on task type:

```json
{
  "hooks": {
    "SessionStart": [{
      "matcher": "",
      "hooks": [{
        "type": "command",
        "command": "npx @claude-flow/cli@latest hooks session-start --optimize"
      }]
    }]
  }
}
```

The optimizer sets:

| Task Type | Env Vars Set |
|-----------|-------------|
| **Coding** | `EFFORT_LEVEL=high`, `BRIEF=0`, `ENABLE_TASKS=1` |
| **Research** | `EFFORT_LEVEL=max`, `SUBAGENT_MODEL=haiku`, spawn explorers |
| **Quick fix** | `EFFORT_LEVEL=low`, `BRIEF=1`, `FAST_MODE=1` |
| **Planning** | `PLAN_V2_AGENT_COUNT=5`, `PLAN_MODE_REQUIRED=1` |
| **Background** | `IDLE_THRESHOLD_MINUTES=1`, `DISABLE_BACKGROUND_TASKS=0` |
| **Swarm** | `EXPERIMENTAL_AGENT_TEAMS=1`, `IS_COWORK=1` |

### 2. Agent-Booster Fast Path (ADR-026 Tier 1)

From the decompilation, we know the exact tool dispatch path:

```
Agent Loop (s$) → tool_use content block → tool name lookup → validateInput → call
```

The Agent Booster intercepts at the `validateInput` step:
- Simple transforms (var→const, add types) → handle in WASM, skip LLM entirely
- Known patterns (from 210 training patterns) → return cached result
- Complex reasoning → pass through to Claude

This maps directly to the 3-tier model routing:
- **Tier 1**: WASM booster (<1ms) — pattern matches from decompiled tool schemas
- **Tier 2**: Haiku (~500ms) — simple tasks identified by decompiled complexity hints
- **Tier 3**: Sonnet/Opus (2-5s) — complex reasoning

### 3. Permission Mode Optimization

The decompilation revealed 6 permission modes and their exact behavior:

| Mode | What It Allows | RVAgent Use Case |
|------|---------------|------------------|
| `bypassPermissions` | Everything | Trusted automated pipelines |
| `acceptEdits` | All file edits, asks for Bash | Code generation agents |
| `auto` | AI decides | General purpose |
| `default` | Asks for everything | User-facing sessions |
| `dontAsk` | Denies instead of asking | CI/CD (fail-safe) |
| `plan` | Plan only, no execution | Architecture review |

RVAgent sets the optimal mode per agent type:
```javascript
function getPermissionMode(agentType) {
    switch (agentType) {
        case 'coder': return 'acceptEdits';
        case 'reviewer': return 'plan';
        case 'tester': return 'auto';
        case 'deployer': return 'bypassPermissions'; // trusted
        default: return 'default';
    }
}
```

### 4. Context Window Optimization

The decompilation revealed the exact compaction algorithm:
- `clear_tool_uses_20250919` API feature triggers server-side compaction
- Micro-compaction removes stale tool results
- `preCompactTokenCount` / `postCompactTokenCount` track effectiveness
- `AUTO_COMPACT_WINDOW` controls the threshold

RVAgent optimizations:
- **Pre-load CLAUDE.md cache**: Put stable content (tool schemas, rules) first for prompt cache hits
- **Deferred tool loading**: Use `ToolSearch` to lazy-load tool schemas (saves ~2000 tokens per unused tool)
- **Strategic compaction**: Trigger compaction before context-heavy operations
- **Brief mode**: Enable `CLAUDE_CODE_BRIEF` for inter-agent communication (25-word limit between tool calls)

### 5. Unreleased Feature Exploitation

Features found in the binary that RVAgent can activate:

**Agent Teams (cowork mode)**:
```bash
export CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1
export CLAUDE_CODE_IS_COWORK=1
export CLAUDE_CODE_USE_COWORK_PLUGINS=1
```
This enables Claude Code's native multi-agent collaboration — potentially replacing some of RVAgent's swarm coordination.

**KAIROS Autonomous Mode**:
```bash
# Don't disable cron
unset CLAUDE_CODE_DISABLE_CRON
# Set idle threshold low for faster dream activation
export CLAUDE_CODE_IDLE_THRESHOLD_MINUTES=2
```

**Plan V2 Multi-Agent**:
```bash
export CLAUDE_CODE_PLAN_V2_AGENT_COUNT=8
export CLAUDE_CODE_PLAN_V2_EXPLORE_AGENT_COUNT=4
```
This tells Claude Code to spawn 8 planning agents and 4 explore agents — native parallelism.

**Task System**:
```bash
export CLAUDE_CODE_ENABLE_TASKS=1
```
Built-in task tracking that RVAgent can monitor via session state events.

## Implementation

### Phase 1: Env Var Optimizer Hook (1 day)

Create `@claude-flow/optimizer` that:
1. Reads task type from hook context
2. Sets optimal env vars
3. Configures permission mode
4. Enables relevant unreleased features

### Phase 2: Agent Booster Integration (3 days)

Wire WASM booster with decompiled tool schemas:
1. Extract all 25+ tool `inputSchema` definitions from decompiled source
2. Build WASM-side validation that matches Claude Code's `validateInput` exactly
3. Cache known tool responses for repeated patterns

### Phase 3: Context Optimization (2 days)

1. Restructure CLAUDE.md for cache-first layout
2. Implement deferred tool loading in MCP server
3. Add compaction triggers to session hooks
4. Enable brief mode for inter-agent comms

### Phase 4: Unreleased Features (1 day)

1. Enable Agent Teams for swarm tasks
2. Enable Plan V2 for architecture planning
3. Enable Tasks for progress tracking
4. Test KAIROS activation via settings

## Consequences

### Positive

- **10-40x faster** on Tier 1 tasks (WASM booster handles pattern matches)
- **60-80% token reduction** from deferred tool loading + cache optimization
- **Native multi-agent** via Agent Teams (no external orchestration needed)
- **Optimal permission mode** per agent type (no over-asking, no under-permitting)
- **Autonomous operation** via KAIROS + cron for background work

### Negative

- Depends on specific Claude Code version (env vars may change)
- Unreleased features may be removed in future versions
- Agent Teams may conflict with RVAgent's own swarm coordination

### Risks

| Risk | Mitigation |
|------|------------|
| Env vars change between versions | Pin to version, re-decompile on update |
| Unreleased features break | Feature-flag each optimization, graceful fallback |
| Anthropic blocks env var overrides | Monitor for changes, have pattern-only fallback |

## References

- [ADR-026: 3-Tier Model Routing](./CLAUDE.md)
- [ADR-134: RuVector Deep Integration](./ADR-134-ruvector-claude-code-deep-integration.md)
- [ADR-135: MinCut Decompiler](./ADR-135-mincut-decompiler-with-witness-chains.md)
- [Research: Extension Points](../research/claude-code-rvsource/13-extension-points.md)
- [Research: Core Module Analysis](../research/claude-code-rvsource/15-core-module-analysis.md)
- [ruDevolution releases](https://github.com/ruvnet/rudevolution/releases)
