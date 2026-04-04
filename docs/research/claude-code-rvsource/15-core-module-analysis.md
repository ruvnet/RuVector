# 15 - Core Module Analysis

Deep analysis of each core module's actual implementation, traced from the minified source.

## 1. Agent Loop (`s$` - async generator)

The agent loop is the central execution engine. It is an **async generator function** named `s$`
in the minified code, yielding events as it processes messages.

### Signature (reconstructed)

```typescript
async function* agentLoop({
  messages: Message[],
  systemPrompt: string,
  userContext: Record<string, unknown>,
  systemContext: Record<string, unknown>,
  canUseTool: PermissionChecker,
  toolUseContext: ToolUseContext,
  autoCompactTracking?: CompactTracker,
  fallbackModel?: string,
  stopHookActive?: boolean,
  querySource: QuerySource
}): AsyncGenerator<AgentEvent>
```

### Yield Event Types

The loop yields these event types (traced from `yield{type:"..."}` patterns):

| Event Type | Count in Code | Purpose |
|------------|---------------|---------|
| `stream_request_start` | 1 | Signals API request beginning |
| `stream_event` | 2 | SSE events from Anthropic API |
| `assistant` | 2 | Complete assistant message |
| `user` | 5 | Synthetic user messages (tool results) |
| `system` | 5 | System events (compact boundary, etc.) |
| `result` | 6 | Tool execution results |
| `progress` | 1 | Progress updates |
| `stop` | 3 | Stop signals |
| `stopReason` | 1 | Why the loop stopped |
| `hookPermissionResult` | 3 | Hook permission decisions |
| `preventContinuation` | 1 | Blocks further turns |
| `tool_progress` | 1 | Tool execution progress |
| `message` | 3 | General message events |

### Core Loop Flow (reconstructed from call sites)

```
s$() entry
  ├── yield {type: "stream_request_start"}
  ├── Build API request (system prompt, messages, tools, betas)
  ├── Call Anthropic Messages API (streaming)
  │   ├── for each SSE event:
  │   │   ├── yield {type: "stream_event", event}
  │   │   ├── Accumulate content blocks (text, tool_use, thinking)
  │   │   └── Track token usage
  │   └── yield {type: "assistant", message}
  ├── If response contains tool_use blocks:
  │   ├── For each tool_use:
  │   │   ├── Check permission via canUseTool()
  │   │   ├── If denied: yield {type: "hookPermissionResult"}
  │   │   ├── Execute tool.call()
  │   │   └── yield {type: "result"}
  │   ├── Assemble tool_result messages
  │   ├── yield {type: "user", tool_results}
  │   └── Recursive: yield* s$({messages: [...messages, ...newMessages], ...})
  ├── If stop_reason == "end_turn":
  │   └── yield {type: "stop"}
  ├── If auto-compact triggered:
  │   └── Run compaction, yield {type: "system", subtype: "compact_boundary"}
  └── Run stop hooks if active
```

### Recursive Self-Call Pattern

The loop is **self-recursive via `yield*`**. After processing tool results, it calls itself
with the updated message history. This creates a chain of generator delegations that unwinds
as the conversation progresses. The recursion terminates when:
- The model returns `end_turn` stop reason
- `max_tokens` is reached
- The abort controller signals
- A stop hook intervenes
- Budget limit is exceeded

## 2. Tool Dispatch System

### Tool Interface (reconstructed)

```typescript
interface Tool {
  name: string;
  userFacingName(): string;
  isEnabled(): boolean;
  inputSchema: ZodSchema;
  outputSchema?: ZodSchema;
  validateInput(input: unknown, context: ToolContext): Promise<ValidationResult>;
  call(input: unknown, context: ToolContext, canUseTool, message): Promise<ToolResult>;
  prompt?(context: PromptContext): Promise<string>;
  getPath?(input: unknown): string;  // For file-based tools
  renderToolUseMessage?(input, context): ReactElement;
  renderToolResultMessage?(result, context): ReactElement;
  renderToolUseErrorMessage?(error, context): ReactElement;
  mapToolResultToToolResultBlockParam(result, toolUseId): ToolResultBlock;
}
```

### Built-in Tools (traced from string literals)

| Tool Name | Minified Class | Key Behaviors |
|-----------|---------------|---------------|
| `BashTool` | Named in string | Executes shell commands; sandboxed via seatbelt (macOS) or SOCKS bridge (Linux) |
| `FileReadTool` | `I6` (inferred) | Reads files with offset/limit; validates path permissions |
| `FileWriteTool` | Named in string | Writes files; validates path permissions for write |
| `FileEditTool` | Named in string | String replacement in files; validates edit permissions |
| `Glob` | Referenced | Pattern-based file search |
| `Grep` | Referenced | Content search via vendored ripgrep |
| `WebFetch` | Referenced | HTTP fetch with domain permissions |
| `WebSearch` | Referenced | Web search (no wildcard support) |
| `Agent` / subagent | Referenced | Spawns child agent with own loop |
| `AgentOutputTool` | Named in string | Reads background agent output |
| `TodoWrite` | Referenced | Manages todo list state |
| `NotebookEdit` | Referenced | Jupyter notebook cell editing |
| `MCP tools` | Dynamic | Tools from MCP servers, prefixed `mcp__serverName__toolName` |
| `KillShell` | Referenced | Kills background shell process |
| `ListMcpResources` | Named in string | Lists MCP server resources |
| `ReadMcpResource` | Named in string | Reads specific MCP resource |
| `ExitPlanMode` | Referenced | Exits plan mode, optionally launches swarm |
| `AskUserQuestion` | Referenced | Prompts user for input |

### Tool Dispatch Flow

```
Model returns tool_use content block
  ├── Find tool by name in tools array
  │   ├── Built-in tools: direct lookup
  │   └── MCP tools: parse "mcp__server__tool" prefix → route to MCP client
  ├── Check tool.isEnabled()
  ├── Run tool.validateInput(input, context)
  │   ├── Path validation (file tools): check against permission rules
  │   ├── URL validation (WebFetch): domain whitelist check
  │   └── Schema validation via Zod
  ├── Permission check via canUseTool(toolName, input)
  │   ├── Check always-allow rules
  │   ├── Check deny rules
  │   ├── If mode == "bypassPermissions": allow
  │   ├── If mode == "dontAsk": allow
  │   ├── If mode == "plan": deny tool execution
  │   └── If mode == "default": prompt user
  ├── Execute tool.call(input, context, canUseTool, message)
  └── Map result via mapToolResultToToolResultBlockParam()
```

## 3. Permission Checker

### Permission Modes (from `ET` enum)

```typescript
type PermissionMode = "acceptEdits" | "bypassPermissions" | "default" | "dontAsk" | "plan";
```

### Permission Context (from `Ll()` / `xC()`)

```typescript
interface ToolPermissionContext {
  mode: PermissionMode;
  alwaysAllowRules: {
    command: string[];  // e.g., ["Bash(npm run:*)"]
  };
  // deny rules for file paths, domains, etc.
}
```

### Permission Rule Format (from string literals)

```
"Bash"                              → allows all bash commands
"Bash(npm run:*)"                   → allows any npm run command
"Bash(npm install express)"         → allows exact command
"Bash(git:*)"                       → allows all git commands
"Bash(git:*) Edit"                  → allows git + file editing
"WebFetch(domain:example.com)"      → allows fetch to specific domain
"WebFetch(domain:*.google.com)"     → allows fetch to google subdomains
```

### Sandbox Implementation

Two sandbox backends exist:

1. **macOS**: Uses `sandbox-exec -p` (seatbelt) with a generated profile
2. **Linux**: Uses a SOCKS bridge process for network isolation

Key sandbox config fields:
```typescript
sandbox: {
  enabled: boolean;
  autoAllowBashIfSandboxed: boolean;
  allowUnsandboxedCommands: boolean;
  excludedCommands: string[];
  ignoreViolations: boolean;
  enableWeakerNestedSandbox: boolean;  // For Docker
  ripgrep?: RipgrepConfig;
}
```

## 4. Context Window Management

### Compaction System (reconstructed from "tengu_compact" telemetry)

The compaction system manages context window overflow through summarization.

```typescript
// Auto-compact triggers when input tokens exceed threshold
// Threshold is derived from API_TARGET_INPUT_TOKENS env var
// Default: uses clear_tool_uses_20250919 API feature

interface CompactionConfig {
  type: "clear_tool_uses_20250919";
  trigger: { type: "input_tokens"; value: number };
  clear_at_least: { type: "input_tokens"; value: number };
}
```

### Compaction Flow

```
Token count exceeds threshold
  ├── Set spinner: "Running SessionStart hooks..."
  ├── Call API with querySource: "compact"
  │   ├── Request summary of conversation
  │   └── Track: tengu_compact telemetry
  ├── Validate summary (not empty, not error, not too_long)
  ├── Replace message history with:
  │   ├── Summary message
  │   ├── Compact boundary marker
  │   ├── File reference attachments (re-read key files)
  │   └── Hook results
  ├── Micro-compaction: clear individual tool uses that are stale
  ├── Track: preCompactTokenCount, postCompactTokenCount
  └── Continue agent loop with reduced context
```

### File Reference Restoration

After compaction, files referenced in the conversation are re-read to maintain context:
```
Post-compact file restore:
  ├── For each tracked file:
  │   ├── Read with fileReadingLimits.maxTokens = iE5
  │   ├── Track: tengu_post_compact_file_restore_success/error
  │   └── Add as attachment
  └── Clear stale entries from readFileState cache
```

## 5. MCP Client

### Connection Lifecycle

```typescript
// MCP server config sources:
// 1. .mcp.json in project directory
// 2. Settings (localSettings, policySettings)
// 3. Dynamic config at runtime

// Connection types:
// - stdio: spawn child process
// - SSE: HTTP server-sent events
// - WebSocket: ws:// connection
// - MCPB: bundled MCP packages (downloaded + extracted)
```

### MCP Tool Integration

MCP tools are namespaced as `mcp__<serverName>__<toolName>`:

```typescript
function parseMcpToolName(fullName: string): { serverName: string; toolName: string } {
  // "mcp__claude-flow__memory_store" → { serverName: "claude-flow", toolName: "memory_store" }
}
```

### MCP Protocol Messages (traced from code)

```
initialize → capabilities exchange
listTools  → discover available tools
callTool   → execute a tool with JSON input
notifications/ → server-sent notifications
resources/list → list available resources
resources/read → read a specific resource
```

### MCPB (MCP Bundles)

MCPB files are packaged MCP servers that can be downloaded and extracted:
```
Download MCPB → extract to ~/.claude/ → load config → start server
```

## 6. Streaming Handler

### SSE Event Processing (from content_block_delta handler)

```typescript
// Event types processed:
switch (event.type) {
  case "message_start":     // Initialize response accumulator
  case "content_block_start": // New content block (text, tool_use, thinking)
  case "content_block_delta": // Incremental update
    switch (delta.type) {
      case "text_delta":        // Text chunk → append to current block
      case "input_json_delta":  // Tool input JSON chunk → accumulate
      case "thinking_delta":    // Extended thinking chunk
    }
  case "content_block_stop":  // Block complete
  case "message_delta":       // Usage update (stop_reason, tokens)
  case "message_stop":        // Response complete
}
```

### Stream Mode States

The UI tracks stream mode for rendering:
```
"requesting"  → Waiting for first token
"responding"  → Receiving text content
"tool-input"  → Receiving tool use JSON
```

### Token Tracking (from usage accumulator)

```typescript
// Per-response tracking:
usage: {
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
  server_tool_use?: { web_search_requests: number };
}
```
