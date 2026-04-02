# Claude Code CLI: Permission System

## Permission Modes

Six distinct permission modes control tool execution:

| Mode | Description |
|------|-------------|
| `default` | Ask user for each permission (interactive) |
| `acceptEdits` | Auto-approve file edits, ask for everything else |
| `plan` | Read-only mode, no modifications allowed |
| `dontAsk` | Skip permission prompts but respect rules |
| `bypassPermissions` | Skip all permission checks (sandbox only) |
| `auto` | Auto-approve based on configured rules |

Set via `--permission-mode` CLI flag or `--dangerously-skip-permissions`.

## Permission Architecture

### Core Types (30+ identified)

| Type | Purpose |
|------|---------|
| `Permission` | Base permission type |
| `PermissionRequest` | Incoming tool use request |
| `PermissionResponse` | Approval/denial result |
| `PermissionResult` | Final permission outcome |
| `PermissionContext` | Contextual info for decision |
| `PermissionMatcher` | Pattern matching for rules |
| `PermissionCallbacks` | UI callbacks for prompts |
| `PermissionPrompt` | User prompt configuration |
| `PermissionPoller` | Async permission polling |
| `PermissionSync` | Permission state sync |

### Permission Flow

```
tool_use request
     |
     v
PermissionRequest created
     |
     v
PreToolUse hooks run
     |  (hooks can approve/deny/modify)
     v
Permission rules checked
     |  (allow/deny/ask patterns)
     v
PermissionMode evaluation
     |
     +-- bypassPermissions -> ALLOW
     +-- plan -> DENY (if write operation)
     +-- auto -> Check configured rules
     +-- default -> PermissionPrompt to user
     +-- acceptEdits -> ALLOW edits, ask rest
     +-- dontAsk -> Default to deny
     |
     v
Permission granted or denied
     |
     v
PostToolUse hooks run (if executed)
```

### Permission Rules

Configured in settings under `permissions`:

```json
{
  "permissions": {
    "allow": ["Read", "Glob", "Grep"],
    "deny": ["Bash(rm *)"],
    "ask": ["Write", "Edit"]
  }
}
```

Rules support tool name patterns:
- `"Bash"` -- Match all Bash tool uses
- `"Bash(git:*)"` -- Match Bash with git commands
- `"Edit"` -- Match all Edit tool uses
- `"Read(~/.zshrc)"` -- Match specific file reads

### Permission Errors

| Error Type | Meaning |
|------------|---------|
| `PermissionDenied` | Explicitly denied by rule |
| `PermissionDeniedError` | Error thrown on denial |
| `PermissionDeniedHooks` | Denied by a hook |
| `PermissionCancelled` | User cancelled the prompt |

### Sandbox Integration

The permission system integrates with OS-level sandboxing:

**Linux**: `bubblewrap` (bwrap)
- Filesystem isolation
- Network access control
- Process namespace separation

**macOS**: `sandbox-exec` / `seatbelt`
- App Sandbox profiles
- File system restrictions
- Network policy enforcement

Sandbox-related types:
- `SandboxManager` -- Manages sandbox lifecycle
- `SandboxPermissions` -- Sandbox-level permissions
- `SandboxViolationStore` -- Tracks sandbox violations
- `SandboxRuntimeConfig` / `SandboxRuntimeConfigSchema` -- Runtime config
- `SandboxedBash` -- Bash execution within sandbox
- `SandboxSettings` / `SandboxSettingsLockedByPolicy` -- Policy controls
- `SandboxAutoAllowEnabled` -- Auto-allow certain sandbox operations
- `SandboxDomainsOnly` -- Restrict to approved domains

### Auto Mode Permissions

`PermissionsForAutoMode` provides a curated set of auto-approved
operations for `--permission-mode auto`:
- File reads in project directory
- Standard git operations
- Build/test commands
- Non-destructive shell commands

### Managed Settings Lock

Enterprise/team settings can lock permissions:
- `allowManagedPermissionRulesOnly` -- Only use managed rules
- `allowManagedHooksOnly` -- Only allow managed hooks
- `allowManagedMcpServersOnly` -- Only use managed MCP servers

These ensure organizational security policies cannot be overridden
by individual users.
