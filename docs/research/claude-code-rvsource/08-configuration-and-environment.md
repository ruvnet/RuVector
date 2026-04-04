# Claude Code CLI: Configuration and Environment

## Configuration Hierarchy

Settings are loaded from multiple sources with increasing priority:

```
1. Built-in defaults (lowest)
2. User settings:        ~/.claude/settings.json
3. Project settings:     .claude/settings.json
4. Local settings:       .claude/settings.local.json
5. Managed settings:     Enterprise/team (highest for locked fields)
6. CLI flags:            --model, --permission-mode, etc. (runtime override)
7. Environment variables: ANTHROPIC_MODEL, etc. (runtime override)
```

`--setting-sources` controls which sources load: `user`, `project`, `local`.

## Settings Schema (76 Properties)

### Model Configuration

| Setting | Type | Description |
|---------|------|-------------|
| `model` | string | Override default model |
| `advisorModel` | string | Model for advisor tool |
| `availableModels` | array | Allowlisted models (e.g., "opus", "sonnet") |
| `modelOverrides` | object | Map Anthropic model IDs to provider-specific IDs |
| `effortLevel` | string | Persisted effort level |
| `fastMode` | boolean | Enable fast mode |
| `fastModePerSessionOptIn` | boolean | Fast mode does not persist across sessions |
| `alwaysThinkingEnabled` | boolean | Control thinking/reasoning |

### Agent Configuration

| Setting | Type | Description |
|---------|------|-------------|
| `agent` | string | Default agent for main thread |
| `permissions` | object | Tool permission rules (allow/deny/ask) |
| `hooks` | object | Hook event handlers |
| `env` | object | Environment variables for sessions |

### MCP Configuration

| Setting | Type | Description |
|---------|------|-------------|
| `allowedMcpServers` | array | Enterprise MCP server allowlist |
| `deniedMcpServers` | array | Enterprise MCP server denylist |
| `enableAllProjectMcpServers` | boolean | Auto-approve project MCP servers |
| `enabledMcpjsonServers` | array | Approved .mcp.json servers |
| `disabledMcpjsonServers` | array | Rejected .mcp.json servers |
| `allowManagedMcpServersOnly` | boolean | Lock to managed servers only |

### Plugin/Marketplace

| Setting | Type | Description |
|---------|------|-------------|
| `enabledPlugins` | object | Active plugins (plugin-id@marketplace-id) |
| `pluginConfigs` | object | Per-plugin configuration |
| `blockedMarketplaces` | array | Blocked marketplace sources |
| `extraKnownMarketplaces` | object | Additional marketplaces |
| `allowedChannelPlugins` | array | Channel plugin allowlist |
| `channelsEnabled` | boolean | Enable channel notifications |
| `pluginTrustMessage` | string | Custom trust warning message |

### Session/Memory

| Setting | Type | Description |
|---------|------|-------------|
| `autoMemoryEnabled` | boolean | Enable auto-memory |
| `autoMemoryDirectory` | string | Custom memory storage path |
| `autoDreamEnabled` | boolean | Background memory consolidation |
| `cleanupPeriodDays` | integer | Transcript retention (default: 30) |
| `autoCompactWindow` | integer | Auto-compact window size |
| `plansDirectory` | string | Custom plan file directory |

### Security/Enterprise

| Setting | Type | Description |
|---------|------|-------------|
| `allowManagedHooksOnly` | boolean | Lock to managed hooks |
| `allowManagedPermissionRulesOnly` | boolean | Lock to managed permissions |
| `disableAllHooks` | boolean | Disable all hooks |
| `disableAutoMode` | string | Disable auto permission mode |
| `forceLoginMethod` | string | Force "claudeai" or "console" auth |
| `forceLoginOrgUUID` | string/array | Required org UUID for OAuth |
| `minimumVersion` | string | Prevent downgrades |

### Display/UX

| Setting | Type | Description |
|---------|------|-------------|
| `language` | string | Response language preference |
| `outputStyle` | string | Output style for responses |
| `prefersReducedMotion` | boolean | Reduce animations |
| `feedbackSurveyRate` | number | Survey probability (0-1) |
| `attribution` | object | Commit/PR attribution text |
| `includeGitInstructions` | boolean | Include git workflow prompts |
| `promptSuggestionEnabled` | boolean | Enable prompt suggestions |
| `fileSuggestion` | object | @ mention file suggestions |

### Remote

| Setting | Type | Description |
|---------|------|-------------|
| `remote` | object | Remote session configuration |
| `defaultShell` | string | Default shell (defaults to bash) |
| `apiKeyHelper` | string | Script path for auth values |
| `awsAuthRefresh` | string | AWS auth refresh script |
| `awsCredentialExport` | string | AWS credential export script |
| `gcpAuthRefresh` | string | GCP auth refresh command |
| `otelHeadersHelper` | string | OTEL headers script |

## Environment Variables (498 Recognized)

### Critical Anthropic Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_API_KEY` | Primary API key |
| `ANTHROPIC_AUTH_TOKEN` | Auth token |
| `ANTHROPIC_BASE_URL` | Custom API base URL |
| `ANTHROPIC_MODEL` | Model override |
| `ANTHROPIC_BETAS` | Beta feature flags |
| `ANTHROPIC_CUSTOM_HEADERS` | Custom API headers |

### Provider Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_BEDROCK_BASE_URL` | Bedrock endpoint |
| `CLAUDE_CODE_USE_BEDROCK` | Enable Bedrock |
| `CLAUDE_CODE_USE_VERTEX` | Enable Vertex AI |
| `CLAUDE_CODE_USE_FOUNDRY` | Enable Azure Foundry |
| `CLAUDE_CODE_USE_ANTHROPIC_AWS` | Enable Anthropic AWS |
| `AWS_ACCESS_KEY_ID` | AWS credentials |
| `AWS_SECRET_ACCESS_KEY` | AWS credentials |
| `AWS_SESSION_TOKEN` | AWS session |
| `AWS_REGION` | AWS region |
| `ANTHROPIC_VERTEX_PROJECT_ID` | GCP project |
| `ANTHROPIC_FOUNDRY_BASE_URL` | Azure endpoint |
| `ANTHROPIC_FOUNDRY_API_KEY` | Azure key |

### Model Configuration Variables

| Variable | Purpose |
|----------|---------|
| `ANTHROPIC_DEFAULT_SONNET_MODEL` | Default Sonnet model ID |
| `ANTHROPIC_DEFAULT_OPUS_MODEL` | Default Opus model ID |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | Default Haiku model ID |
| `ANTHROPIC_SMALL_FAST_MODEL` | Small/fast model for subtasks |
| `CLAUDE_CODE_SUBAGENT_MODEL` | Model for subagents |

### Feature Flags (CLAUDE_CODE_*)

Over 120 `CLAUDE_CODE_*` variables control features:

**Disable flags**: `CLAUDE_CODE_DISABLE_*`
- `_AUTO_MEMORY`, `_BACKGROUND_TASKS`, `_CLAUDE_MDS`, `_CRON`
- `_FAST_MODE`, `_FILE_CHECKPOINTING`, `_GIT_INSTRUCTIONS`
- `_THINKING`, `_VIRTUAL_SCROLL`, `_MOUSE`, `_TERMINAL_TITLE`
- `_ATTACHMENTS`, `_EXPERIMENTAL_BETAS`, `_FEEDBACK_SURVEY`
- `_1M_CONTEXT`, `_ADAPTIVE_THINKING`, `_POLICY_SKILLS`

**Enable flags**: `CLAUDE_CODE_ENABLE_*`
- `_CFC`, `_FINE_GRAINED_TOOL_STREAMING`, `_PROMPT_SUGGESTION`
- `_SDK_FILE_CHECKPOINTING`, `_TASKS`, `_TELEMETRY`
- `_TOKEN_USAGE_ATTACHMENT`, `_XAA`

**Configuration**: `CLAUDE_CODE_*`
- `_EFFORT_LEVEL`, `_MAX_OUTPUT_TOKENS`, `_MAX_RETRIES`
- `_SHELL`, `_SHELL_PREFIX`, `_TMPDIR`
- `_GLOB_TIMEOUT_SECONDS`, `_GLOB_HIDDEN`, `_GLOB_NO_IGNORE`
- `_SCROLL_SPEED`, `_IDLE_THRESHOLD_MINUTES`
- `_OAUTH_TOKEN`, `_OAUTH_CLIENT_ID`, `_OAUTH_SCOPES`

### Bash Sandbox Variables

| Variable | Purpose |
|----------|---------|
| `BASH_MAX_OUTPUT_LENGTH` | Max Bash tool output |
| `CLAUDE_CODE_BUBBLEWRAP` | Bubblewrap sandbox config |
| `CLAUDE_CODE_BASH_SANDBOX_SHOW_INDICATOR` | Show sandbox indicator |
| `CLAUDE_CODE_BASH_MAINTAIN_PROJECT_WORKING_DIR` | Maintain cwd |

## Claude Home Directory Structure

```
~/.claude/
  settings.json           # User settings
  keybindings.json        # Custom keybindings
  CLAUDE.md               # Global user instructions
  scheduled_tasks.json    # Cron-like tasks
  agents/                 # Custom agent definitions
  commands/               # Custom slash commands
  debug/                  # Debug logs
  local/                  # Local data
    claude                # Local state
    node_modules/         # Cached modules
  plans/                  # Saved plans
  plugins/                # Installed plugins
    data/                 # Plugin data
  projects/               # Per-project data
    <hash>/               # Project-specific
      memory/MEMORY.md    # Auto-memory
  remote/                 # Remote session data
  rules/                  # Custom rules
```

## Project Configuration Files

```
.claude/
  settings.json           # Project settings
  settings.local.json     # Local overrides (gitignored)
  agents/                 # Project-specific agents
  skills/
    deploy/SKILL.md       # Skill definitions

.mcp.json                 # MCP server definitions
.claudeignore             # Files to ignore
CLAUDE.md                 # Project instructions
```
