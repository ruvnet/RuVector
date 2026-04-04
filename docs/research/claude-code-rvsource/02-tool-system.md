# Claude Code CLI: Tool System Architecture

## Built-in Tools

Claude Code registers tools with the Anthropic Messages API using the standard
tool_use / tool_result content block protocol.

### Tool Registry

The internal tool name map (display names for UI):

```javascript
{
  Read: "Reading",
  Write: "Writing",
  Edit: "Editing",
  MultiEdit: "Editing",
  Bash: "Running",
  Glob: "Searching",
  Grep: "Searching",
  WebFetch: "Fetching",
  WebSearch: "Searching",
  Task: "Running task",
  FileReadTool: "Reading",
  FileWriteTool: "Writing",
  FileEditTool: "Editing",
  GlobTool: "Searching",
  GrepTool: "Searching",
  BashTool: "Running",
  NotebookEditTool: "Editing notebook",
  LSP: "LSP"
}
```

### Tool Name Variables (Internal References)

| Variable | Tool Name |
|----------|-----------|
| `Yq` | Bash |
| `Bq` | Read |
| `c7` | Write |
| `wK` | Edit |
| `Gf` | Glob |
| `f9` | Grep |
| `sw` | WebFetch |
| `nE` | WebSearch |
| `IL` | NotebookEdit |
| `WX` | ToolSearch |
| `lI` | TodoWrite |
| `nj` | Skill |
| `tXH` | SendUserMessage |

### Tool Categories

**File Operations**:
- `Read` / `FileReadTool` -- Read files, images, PDFs, notebooks
- `Write` / `FileWriteTool` -- Create or overwrite files
- `Edit` / `FileEditTool` -- String replacement edits in existing files
- `MultiEdit` -- Multiple edits in a single tool call
- `NotebookEdit` / `NotebookEditTool` -- Jupyter notebook cell operations

**Search**:
- `Glob` / `GlobTool` -- File pattern matching (fast, sorted by mtime)
- `Grep` / `GrepTool` -- Content search via ripgrep with regex support

**Execution**:
- `Bash` / `BashTool` -- Shell command execution with sandbox support
- `Task` -- Spawn subagent for parallel work

**Web**:
- `WebFetch` -- HTTP fetch with content extraction
- `WebSearch` -- Web search integration

**Agent/System**:
- `ToolSearch` -- Discover deferred tools (MCP tools loaded on demand)
- `Skill` -- Execute registered skills/slash commands
- `SendUserMessage` -- Agent-to-user communication (brief mode)
- `TodoRead` / `TodoWrite` -- Task list management
- `EnterWorktree` / `ExitWorktree` -- Git worktree management
- `LSP` -- Language Server Protocol integration

### Tool Schema Pattern

Each tool is defined with:
```javascript
{
  name: "ToolName",
  description: `Template literal with ${otherToolNames} interpolated`,
  input_schema: {
    type: "object",
    properties: { ... },
    required: [ ... ]
  }
}
```

Tool descriptions dynamically reference other tool names so the model knows
the full toolkit available. For example, Bash's description says "Avoid using
this tool to run grep, find, cat... instead use the appropriate dedicated tool."

### Tool Validation

Tools have category-based validation:
- **filePatternTools**: `["Read", "Write", "Edit", "Glob", "NotebookRead", "NotebookEdit"]`
  - Validate file paths
- **bashPrefixTools**: `["Bash"]`
  - Validate command prefixes against allowed/denied patterns
- **customValidation**: Per-tool validators (e.g., WebSearch rejects wildcards)

### Tool Permission Flow

```
User prompt -> Model generates tool_use -> Permission check -> Execute -> tool_result
```

Each tool call goes through the permission system before execution.
See `04-permission-system.md` for details.

### MCP Tool Integration

MCP (Model Context Protocol) tools extend the built-in set:
- MCP tools are prefixed with `mcp__<server-name>__<tool-name>`
- `ReadMcpResourceTool` -- Read MCP server resources
- MCP tools can be deferred (loaded on-demand via `ToolSearch`)
- Tool output size controlled by `MAX_MCP_OUTPUT_TOKENS` env var
- Large outputs can be saved to files (`ENABLE_MCP_LARGE_OUTPUT_FILES`)

### Tool Concurrency

`CLAUDE_CODE_MAX_TOOL_USE_CONCURRENCY` controls parallel tool execution.
Multiple tool calls in a single response can execute concurrently.

### Content Block Types

The API supports these content block types in messages:

| Type | Direction | Purpose |
|------|-----------|---------|
| `text` | Both | Text content |
| `thinking` | Response | Extended thinking |
| `redacted_thinking` | Response | Safety-redacted thinking |
| `tool_use` | Response | Tool call request |
| `tool_result` | Request | Tool execution result |
| `image` | Request | Image content |
| `file` | Request | File content |
| `server_tool_use` | Response | Server-side tool calls |
| `web_search_tool_result` | Response | Web search results |
| `code_execution_tool_result` | Response | Code execution results |
| `mcp_tool_use` | Response | MCP tool calls |
| `mcp_tool_result` | Response | MCP tool results |
