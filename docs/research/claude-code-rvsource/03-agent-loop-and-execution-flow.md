# Claude Code CLI: Agent Loop and Execution Flow

## Entry Points

### CLI Entry

The binary boots through Bun's SEA mechanism into a Commander.js-style CLI:

```
claude [options] [command] [prompt]
```

**Subcommands** (24 registered):
- `auth`, `login`, `logout` -- Authentication
- `config` (list/show/set/defaults) -- Configuration management
- `mcp` (add-from-claude-desktop, list, serve, status, setup) -- MCP server management
- `doctor` -- Diagnostics
- `update` -- Self-update
- `agents` -- Agent management
- `marketplace`, `plugin` -- Plugin/skill management
- `remote-control`, `serve` -- Remote/SSE modes
- `auto-mode` -- Auto permission mode
- `setup-token`, `xaa` -- Auth helpers
- `critique`, `reset-project-choices`, `clear` -- Misc utilities

### VS Code Extension Entry

`extension.js` registers VS Code commands and creates a WebView panel.
The WebView (`webview/index.js`) is a React 18.3.1 application that
communicates with the extension host via `postMessage`.

VS Code commands (21 registered):
- `claude-vscode.editor.open` / `openLast` / `primaryEditor.open`
- `claude-vscode.window.open`
- `claude-vscode.createWorktree`
- `claude-vscode.insertAtMention`
- `claude-vscode.focus` / `blur`
- `claude-vscode.terminal.open.keyboard`

## Main Loop Architecture

### Core Loop Functions

| Function | Purpose |
|----------|---------|
| `mainLoop` | Primary conversation loop |
| `mainLoopModel` | Model selection for current session |
| `mainLoopModelForSession` | Session-level model override |
| `mainLoopModelOverride` | CLI/env model override |
| `mainLoopTokens` | Token tracking for main loop |
| `processMessagesForTeleportResume` | Handle session teleport/resume |

### Execution Flow

```
1. CLI Parse / VS Code Activation
     |
2. Configuration Loading
   - User settings (~/.claude/settings.json)
   - Project settings (.claude/settings.json, .claude/settings.local.json)
   - CLAUDE.md files (auto-discovered up directory tree)
   - Environment variables (498+ recognized)
     |
3. Authentication
   - AnthropicApiKeyAuth (API key from env/helper)
   - ConsoleOAuthFlow (Anthropic Console OAuth)
   - BedrockClient (AWS Bedrock)
   - VertexAI (Google Cloud)
   - Foundry (Azure)
     |
4. MCP Server Connection
   - Read .mcp.json, settings, --mcp-config
   - Connect to MCP servers (stdio, SSE, HTTP transports)
   - Discover tools via tools/list
     |
5. Session Init / Resume
   - Create new session or resume existing
   - Load conversation history
   - Apply CLAUDE.md system prompt
     |
6. Main Loop (per turn):
   a. Collect user input (interactive) or read prompt (--print)
   b. Build message array with system prompt + conversation history
   c. Call Anthropic Messages API (streaming)
   d. Process response stream:
      - Text blocks -> render to terminal/UI
      - Thinking blocks -> display if enabled
      - tool_use blocks -> queue for execution
   e. For each tool_use:
      - Check permissions (PreToolUse hooks)
      - Execute tool
      - Run PostToolUse hooks
      - Append tool_result to messages
   f. If model stopped with tool_use (not end_turn):
      - Loop back to step (c) with updated messages
   g. If model stopped with end_turn:
      - Display final response
      - Wait for next user input
     |
7. Context Management
   - Track token usage
   - Auto-compact when context window fills
   - File checkpointing for recovery
     |
8. Session End
   - Run session-end hooks
   - Persist session to disk
   - Cleanup MCP connections
```

### Streaming Architecture

The API response is streamed using Server-Sent Events (SSE):

```
StreamEvent -> StreamEventBuffer -> StreamHandler -> UI Renderer
```

Key streaming patterns:
- `StreamFriendlyUIEnabled` -- Controls rich terminal rendering
- `StreamInterval` -- Configurable update frequency
- `StreamFinished` / `StreamFinishing` -- Completion states
- Partial message chunks available via `--include-partial-messages`

### Conversation Management

- `ConversationManager` -- Manages conversation state
- `ConversationChain` -- Message chain/history
- `MessageBuffer` / `MessageBufferTracker` -- Buffer management
- `MessageByUuid` -- Message lookup
- `MessageExistInSession` -- Deduplication

### Slash Commands (39 Built-in)

Interactive commands processed before sending to the API:

| Command | Purpose |
|---------|---------|
| `/compact` | Force context compaction |
| `/clear` | Clear conversation |
| `/model` | Switch model |
| `/effort` | Set reasoning effort level |
| `/fast` | Toggle fast mode |
| `/permissions` | View/edit permission rules |
| `/hooks` | View configured hooks |
| `/mcp` | MCP server status |
| `/memory` | View/edit auto-memory |
| `/status` | Session status |
| `/review-pr` | PR review workflow |
| `/commit` | Commit workflow |
| `/init` | Project initialization |
| `/login` / `/logout` | Authentication |
| `/doctor` | Run diagnostics |
| `/bug` | Report a bug |
| `/cost` | Show cost information |
| `/extra-usage` | Usage details |
| `/feedback` | Submit feedback |
| `/config` | Configuration |
| `/skills` | List available skills |
| `/agents` | Agent management |
| `/tasks` | Task/subagent list |
| `/worker` | Background workers |
| `/branch` | Create feature branch |
| `/teleport` | Session teleport |
| `/rewind` | Rewind conversation |
| `/resume` | Resume session |
| `/loop` | Recurring command |
| `/chrome` | Chrome integration |
| `/powerup` | Power-up features |
| `/deploy` | Deployment workflow |
| `/dev` | Development mode |
| `/issue` | Issue tracker |
| `/metrics` | Performance metrics |
| `/dashboard` | Dashboard view |
| `/ultrareview` | Deep code review |
| `/prompt` | Prompt management |
| `/install-github-app` | GitHub App setup |

### Output Modes

```
--output-format text         # Default: human-readable
--output-format json         # Single JSON result
--output-format stream-json  # Real-time streaming JSON
```

Input modes:
```
--input-format text          # Default: plain text prompt
--input-format stream-json   # Real-time streaming JSON input
```
