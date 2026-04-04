# Claude Code CLI: Hooks System

## Overview

Hooks are custom commands that run at specific points in the tool execution
lifecycle. They enable automation, validation, and integration with external
systems.

## Hook Event Types

Six hook events are supported:

| Event | Trigger | Can Block |
|-------|---------|-----------|
| `PreToolUse` | Before a tool executes | Yes (can deny) |
| `PostToolUse` | After a tool executes | No |
| `PreToolUseFailure` | Before reporting tool failure | No |
| `PostToolUseFailure` | After tool failure | No |
| `Notification` | On system notifications | No |
| `Stop` | When agent stops (end_turn) | Yes (can continue) |
| `SubagentStop` | When a subagent stops | Yes (can continue) |

## Hook Configuration

Hooks are defined in settings.json under the `hooks` key:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Edit|Write",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'File being modified: $TOOL_INPUT'"
          }
        ]
      }
    ],
    "PostToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo 'Command completed'"
          }
        ]
      }
    ]
  }
}
```

### Matcher Syntax

- Empty string `""` -- Match all tools
- `"Bash"` -- Match specific tool
- `"Edit|Write"` -- Match multiple tools (pipe-separated)

### Hook Types

| Type | Description |
|------|-------------|
| `command` | Execute a shell command |
| `http` | Make an HTTP request |

### HTTP Hooks

```json
{
  "type": "http",
  "url": "https://example.com/webhook",
  "method": "POST",
  "headers": { "Authorization": "Bearer $TOKEN" }
}
```

HTTP hook security:
- `allowedHttpHookUrls` -- URL pattern allowlist (supports `*` wildcard)
- `httpHookAllowedEnvVars` -- Env vars allowed in header interpolation

## Hook Execution

### Lifecycle

```
Tool Use Request
     |
     v
PreToolUse hooks (matcher: tool name)
     |
     +-- Hook returns { action: "deny" } -> Tool blocked
     +-- Hook returns { action: "allow" } -> Skip permission check
     +-- Hook returns nothing -> Continue to permission check
     |
     v
Tool executes
     |
     v
PostToolUse hooks (matcher: tool name)
     |
     +-- Can log, notify, but cannot block
```

### Hook Functions

- `runHooks` -- Main hook execution engine
- `executeHooksOutsideREPL` -- Run hooks in non-interactive mode

### Stop Hooks

Stop hooks fire when the agent decides to stop (end_turn):

```json
{
  "Stop": [
    {
      "matcher": "",
      "hooks": [
        {
          "type": "command",
          "command": "echo 'Agent completed turn'"
        }
      ]
    }
  ]
}
```

## Security Controls

### Managed Hooks

- `allowManagedHooksOnly` -- When true, only hooks from managed (enterprise)
  settings are executed
- `disableAllHooks` -- Completely disable hook execution

### Hook Context

Hooks receive context about the tool execution:
- Tool name
- Tool input parameters
- Tool output (PostToolUse only)
- Session information
- `CLAUDE_CODE_SAVE_HOOK_ADDITIONAL_CONTEXT` -- Include extra context

### Hook Timeout

`CLAUDE_CODE_SESSIONEND_HOOKS_TIMEOUT_MS` -- Timeout for session-end hooks

## Agent Hooks

Agent-specific hook schema: `AgentHookSchema`

Hooks can be defined per-agent, allowing different automation for
different agent roles (e.g., reviewer agent has different PostToolUse
hooks than coder agent).

## Integration with Compact

`CompactHooks` -- Hooks that run during context compaction, allowing
external systems to be notified when conversation history is compressed.

## Hook Events in Stream Output

When using `--output-format stream-json`:
- `--include-hook-events` includes all hook lifecycle events in the output
- Useful for monitoring/debugging hook execution in CI/CD pipelines
