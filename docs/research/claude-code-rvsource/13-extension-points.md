# Claude Code CLI: Extension Points

## Overview

Claude Code has multiple extension mechanisms, each serving different
integration depths. This document catalogs all identified extension points.

## 1. MCP Servers (Primary Extension)

**Integration depth**: Add tools, resources, and prompts.

```json
// .mcp.json
{
  "mcpServers": {
    "my-server": {
      "command": "npx",
      "args": ["-y", "my-mcp-server"],
      "env": { "API_KEY": "..." }
    }
  }
}
```

**What you can extend**:
- Custom tools (via `tools/list` + `tools/call`)
- Resources (via `resources/list` + `resources/read`)
- Prompt templates (via `prompts/list` + `prompts/get`)
- Auto-completion (via `completion/complete`)

**Transports**: stdio, SSE, HTTP, WebSocket

## 2. Hooks (Lifecycle Automation)

**Integration depth**: React to and control tool execution.

```json
// .claude/settings.json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Edit|Write",
      "hooks": [{ "type": "command", "command": "lint-check $FILE" }]
    }],
    "PostToolUse": [{
      "matcher": "Bash",
      "hooks": [{ "type": "http", "url": "https://api.example.com/notify" }]
    }],
    "Stop": [{
      "matcher": "",
      "hooks": [{ "type": "command", "command": "run-tests" }]
    }]
  }
}
```

**Events**: PreToolUse, PostToolUse, PreToolUseFailure, PostToolUseFailure,
Notification, Stop, SubagentStop

**Hook types**: command (shell), http (webhook)

**Capabilities**: Hooks can approve/deny tool execution (PreToolUse) and
prevent agent stop (Stop).

## 3. Custom Agents

**Integration depth**: Define specialized AI personas.

### JSON Format
```json
// .claude/agents/reviewer.json
{
  "description": "Code review specialist",
  "prompt": "You are a thorough code reviewer...",
  "model": "claude-sonnet-4-6",
  "tools": ["Read", "Glob", "Grep"],
  "hooks": { ... }
}
```

### Markdown Format
```markdown
// .claude/agents/reviewer.md
---
name: reviewer
description: Code review specialist
model: claude-sonnet-4-6
tools: [Read, Glob, Grep]
---

You are a thorough code reviewer...
```

### CLI Definition
```bash
claude --agents '{"reviewer": {"description": "...", "prompt": "..."}}'
claude --agent reviewer
```

## 4. Skills (Slash Commands)

**Integration depth**: Reusable task workflows.

```markdown
// .claude/skills/deploy/SKILL.md
---
name: deploy
description: Deploy the application
tools: [Bash, Read]
---

# Deploy Skill

Follow these steps to deploy...
```

Skills are invoked via `/skill-name` in the interactive REPL.

## 5. Plugins (Marketplace)

**Integration depth**: Distributable packages of agents, tools, and config.

Plugins are installed from marketplaces and can provide:
- MCP servers (additional tools)
- Agent definitions
- Skill definitions
- Configuration presets
- Model endpoint definitions

Configuration:
- `enabledPlugins` -- Active plugins
- `pluginConfigs` -- Per-plugin settings
- `blockedMarketplaces` -- Denylist
- `extraKnownMarketplaces` -- Additional sources

## 6. CLAUDE.md System Prompt

**Integration depth**: Customize AI behavior per project.

```markdown
// CLAUDE.md (project root)
# Project Instructions

## Rules
- Always use TypeScript
- Follow TDD

## Architecture
- Use DDD bounded contexts
- Keep files under 500 lines
```

Discovered locations (all loaded and merged):
1. `~/.claude/CLAUDE.md` (global user)
2. `CLAUDE.md` in project root
3. `CLAUDE.md` in parent directories
4. Additional via `--add-dir`

## 7. Settings Override Chain

**Integration depth**: Configure behavior at multiple scopes.

| Scope | File | Purpose |
|-------|------|---------|
| User | `~/.claude/settings.json` | Personal defaults |
| Project | `.claude/settings.json` | Team shared config |
| Local | `.claude/settings.local.json` | Personal project overrides |
| Managed | Enterprise policy | Organization rules |

## 8. Environment Variables

**Integration depth**: Runtime behavior control.

498+ recognized environment variables provide fine-grained control
over every subsystem. Key categories:
- `ANTHROPIC_*` -- API and model configuration
- `CLAUDE_CODE_*` -- Feature flags and behavior
- `CLAUDE_CODE_DISABLE_*` -- Feature disabling
- `CLAUDE_CODE_ENABLE_*` -- Feature enabling
- `MCP_*` -- MCP configuration
- `OTEL_*` -- Observability

## 9. API Key Helper

**Integration depth**: Custom authentication.

```json
{
  "apiKeyHelper": "/path/to/auth-script"
}
```

The script must output the API key to stdout. Also:
- `awsAuthRefresh` -- AWS credential refresh script
- `awsCredentialExport` -- AWS credential export script
- `gcpAuthRefresh` -- GCP auth refresh command
- `otelHeadersHelper` -- OTEL headers generation

## 10. Keybindings

**Integration depth**: Custom keyboard shortcuts.

```json
// ~/.claude/keybindings.json
{
  "bindings": [
    { "key": "ctrl+s", "command": "submit" },
    { "key": "ctrl+shift+p", "command": "slash-command" }
  ]
}
```

## 11. Agent SDK (Programmatic)

**Integration depth**: Embed Claude Code in applications.

```typescript
import { query } from "@anthropic-ai/claude-agent-sdk";

for await (const message of query({
  prompt: "Explain this code",
  options: {
    cwd: "/path/to/project",
    allowedTools: ["Read", "Glob", "Grep"],
  },
})) {
  if ("result" in message) {
    console.log(message.result);
  }
}
```

Environment variables for SDK:
- `CLAUDE_AGENT_SDK_CLIENT_APP`
- `CLAUDE_AGENT_SDK_DISABLE_BUILTIN_AGENTS`
- `CLAUDE_AGENT_SDK_MCP_NO_PREFIX`
- `CLAUDE_AGENT_SDK_VERSION`

## 12. Scheduled Tasks (Cron)

**Integration depth**: Automated recurring tasks.

```json
// ~/.claude/scheduled_tasks.json
{
  "tasks": [
    {
      "schedule": "0 9 * * 1",
      "command": "Review open PRs"
    }
  ]
}
```

Controlled by `CLAUDE_CODE_DISABLE_CRON`.

## 13. Remote Control

**Integration depth**: External process control.

`claude remote-control` and `claude serve` expose an SSE/HTTP interface
for external tools to drive Claude Code programmatically.

- `CLAUDE_CODE_SSE_PORT` -- SSE server port
- `--output-format stream-json` -- Streaming JSON protocol
- `--input-format stream-json` -- Streaming JSON input

## Extension Point Comparison

| Extension | Complexity | Capability | Distribution |
|-----------|-----------|------------|--------------|
| CLAUDE.md | Low | Prompt only | Git |
| Env vars | Low | Config only | Shell |
| Settings | Low | Config only | File |
| Keybindings | Low | UI only | File |
| Skills | Medium | Workflows | Git |
| Hooks | Medium | Automation | Settings |
| Custom agents | Medium | Personas | Git/Settings |
| MCP servers | High | Full tools | Package |
| Plugins | High | Everything | Marketplace |
| Agent SDK | High | Embedding | npm |
| Cron tasks | Medium | Scheduling | File |
| Remote control | High | External API | Process |
