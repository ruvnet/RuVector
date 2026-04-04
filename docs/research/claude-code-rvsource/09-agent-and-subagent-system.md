# Claude Code CLI: Agent and Subagent System

## Agent Architecture

### Core Agent Types

| Type | Internal Reference | Purpose |
|------|-------------------|---------|
| `default` | Built-in | Standard Claude Code agent |
| `advisor` | AgentModel/AgentConfig | Advisory/review agent |
| `reviewer` | AgentPlugin | Code review agent |
| `custom` | AgentFromJson/AgentFromMarkdown | User-defined agents |

### Agent Definition Sources

Agents can be defined from multiple sources:

1. **Built-in agents**: Compiled into the binary
2. **JSON definitions**: `.claude/agents/*.json`
3. **Markdown definitions**: `.claude/agents/*.md`
4. **CLI `--agents` flag**: Inline JSON
5. **`--agent` flag**: Select a named agent
6. **Plugin agents**: From marketplace plugins

### Agent Definition Schema

```json
{
  "reviewer": {
    "description": "Reviews code for quality and correctness",
    "prompt": "You are a code reviewer...",
    "model": "claude-sonnet-4-6",
    "tools": ["Read", "Glob", "Grep"],
    "hooks": { ... }
  }
}
```

### Agent State Machine

```
AgentBaseInternalState
     |
     v
AgentBusy -> AgentContext -> AgentMode
     |
     v
AgentSubmit -> API Call -> Response Processing
     |
     v
AgentOutputTool -> Tool Execution -> Result
```

### Agent Configuration

| Property | Purpose |
|----------|---------|
| `AgentConfig` | Full agent configuration |
| `AgentDefinition` | Agent spec (name, prompt, tools) |
| `AgentOptions` | Runtime options |
| `AgentModel` | Model override per agent |
| `AgentColorMap` | Visual differentiation |
| `AgentPrefix` | Message prefix per agent |
| `AgentLanguages` | Language settings |
| `AgentOperatingSystem` | OS-specific behavior |
| `AgentPolicy` | Policy constraints |
| `AgentMiddleware` | Processing middleware |
| `AgentProgressSummariesEnabled` | Progress tracking |

### Agent Factory

`AgentFactoryFromOptions` creates agents from configuration:
- Resolves agent definitions from cache: `AgentDefinitionsCache`
- Applies overrides: `AgentDefinitionsWithOverrides`
- Sets up agent context with tools, permissions, hooks

## Subagent / Task System

### Task Tool

The `Task` tool spawns subagents for parallel work:

```
Main Agent
  |-- Task("Review auth module") -> Subagent 1
  |-- Task("Write unit tests")   -> Subagent 2
  |-- Task("Update docs")        -> Subagent 3
```

### Subagent Types

| Type | Purpose |
|------|---------|
| `SubagentStart` | Subagent creation event |
| `SubagentStop` | Subagent completion event |
| `SubagentEventReader` | Read subagent events |
| `SubagentInternalEvents` | Internal event bus |
| `SubagentTranscripts` | Conversation history |
| `SubagentTranscriptsFromDisk` | Persisted transcripts |
| `SubagentStartHooks` | Hooks for subagent creation |

### Task Lifecycle

```
TaskCreated
     |
     v
TaskAssignment -> TaskAgent
     |
     v
TaskActiveQ (queue management)
     |
     v
TaskAbort / TaskCompleted
     |
     v
TaskAbortReasonInfo (if aborted)
```

### Task Configuration

| Config | Purpose |
|--------|---------|
| `CLAUDE_CODE_ENABLE_TASKS` | Enable task system |
| `CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY` | Max parallel tools |
| `TASK_MAX_OUTPUT_LENGTH` | Max task output |
| `CLAUDE_CODE_PLAN_V2_AGENT_COUNT` | Plan mode agent count |
| `CLAUDE_CODE_PLAN_V2_EXPLORE_AGENT_COUNT` | Exploration agents |
| `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS` | Experimental teams |

### Agent Cost Steering

`CLAUDE_CODE_AGENT_COST_STEER` -- Controls cost optimization for
multi-agent scenarios, balancing between model quality and cost.

### Agent List in Messages

`CLAUDE_CODE_AGENT_LIST_IN_MESSAGES` -- Include agent list in
conversation messages for context.

## Skill System

Skills are reusable capability modules that extend Claude Code:

### Skill Discovery

- `.claude/skills/` directory
- `.claude/commands/` directory (legacy)
- `/skills` slash command -- List available
- `--disable-slash-commands` -- Disable all skills
- `SLASH_COMMAND_TOOL_CHAR_BUDGET` -- Character budget for skill content

### Skill Definition Format

Skills are defined as Markdown files with YAML frontmatter:

```markdown
---
name: deploy
description: Deploy the application
tools: [Bash, Read]
---

# Deploy Skill

Instructions for deployment workflow...
```

### Plugin System

Plugins extend Claude Code through marketplace:

| Component | Purpose |
|-----------|---------|
| `PluginAgent` | Agent provided by plugin |
| `PluginCache` | Plugin artifact cache |
| `PluginConfiguration` | Plugin settings |
| `PluginControl` | Plugin lifecycle management |
| `PluginAutoupdate` | Auto-update mechanism |
| `PluginEditableScopes` | Configurable scopes |
| `PluginAffectingSettingsSnapshot` | Settings impact tracking |

### Marketplace Integration

| Component | Purpose |
|-----------|---------|
| `MarketplaceFromGcs` | GCS-hosted marketplace |
| `MarketplaceIsPrivate` | Private marketplace flag |
| `MarketplaceAutoInstalled` | Auto-installed plugins |
| `MarketplaceModelEndpoint` | Model endpoint plugins |
| `MarketplaceModelEndpoints` | Multiple endpoints |
| `MarketplaceDependenciesOn` | Dependency tracking |

Plugin configuration:
- `CLAUDE_CODE_PLUGIN_CACHE_DIR` -- Cache location
- `CLAUDE_CODE_PLUGIN_GIT_TIMEOUT_MS` -- Git operation timeout
- `CLAUDE_CODE_PLUGIN_KEEP_MARKETPLACE_ON_FAILURE` -- Resilience
- `CLAUDE_CODE_PLUGIN_SEED_DIR` -- Seed directory
- `CLAUDE_CODE_PLUGIN_USE_ZIP_CACHE` -- Zip caching
- `FORCE_AUTOUPDATE_PLUGINS` -- Force updates
- `CLAUDE_CODE_SYNC_PLUGIN_INSTALL` -- Synchronous install
- `CLAUDE_CODE_USE_COWORK_PLUGINS` -- Cowork plugin support
