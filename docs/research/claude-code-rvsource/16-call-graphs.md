# 16 - Call Graphs

Call graphs traced from actual minified source patterns in `cli.js` (v2.0.62).
Function names shown as `minified` -> `reconstructedName`.

## 1. Main Boot Sequence

```mermaid
graph TD
    A["cli.js entry (#!/usr/bin/env node)"] --> B["Parse CLI arguments (process.argv)"]
    B --> C{"Mode?"}
    C -->|"Interactive REPL"| D["Initialize Ink React app"]
    C -->|"--print / -p"| E["SDK non-interactive mode"]
    C -->|"MCP server"| F["Start MCP stdio/SSE server"]
    C -->|"Sub-command"| G["Route to command handler"]

    D --> H["Ll() - createInitialAppState"]
    H --> I["gG() - AppStateProvider"]
    I --> J["Load settings: qq()"]
    J --> K["Load permissions: xC()"]
    K --> L["Initialize MCP clients"]
    L --> M["Load plugins"]
    M --> N["LoA() - resolveThinkingEnabled"]
    N --> O["Render main React component"]
    O --> P["Wait for user input"]

    E --> Q["Build SDK context"]
    Q --> R["JP() - prepareQuery"]
    R --> S["s$() - agentLoop"]
    S --> T["Yield results to SDK caller"]
```

## 2. Agent Loop (s$ function)

```mermaid
graph TD
    A["s$() entry"] --> B["yield stream_request_start"]
    B --> C["Build API params"]
    C --> D["qh2() - countTokens"]
    D --> E{"Exceeds compact threshold?"}
    E -->|Yes| F["Run compaction"]
    F --> G["yield system:compact_boundary"]
    G --> C
    E -->|No| H["d39() - telemetry: tengu_api_query"]
    H --> I["Call Anthropic Messages API (streaming)"]

    I --> J["Process SSE stream"]
    J --> K{"Event type?"}
    K -->|"content_block_delta"| L["Accumulate text/tool_use/thinking"]
    K -->|"message_stop"| M["yield assistant message"]

    M --> N{"Has tool_use blocks?"}
    N -->|No| O["yield stop"]
    N -->|Yes| P["Process each tool_use"]

    P --> Q["Permission check: canUseTool()"]
    Q --> R{"Allowed?"}
    R -->|Denied| S["yield hookPermissionResult"]
    R -->|Ask user| T["Prompt for approval"]
    R -->|Allowed| U["tool.validateInput()"]

    U --> V{"Valid?"}
    V -->|No| W["Return error result"]
    V -->|Yes| X["tool.call()"]
    X --> Y["yield result"]

    Y --> Z["Assemble tool_result messages"]
    Z --> AA["yield user (tool results)"]
    AA --> AB["yield* s$() RECURSIVE CALL"]
    AB --> AC["Continue processing"]
```

## 3. Tool Dispatch Flow

```mermaid
graph TD
    A["Tool use from API response"] --> B{"Tool name prefix?"}
    B -->|"mcp__*"| C["Parse: mcp__server__tool"]
    B -->|Built-in| D["Lookup in tools array"]

    C --> E["Find MCP client by server name"]
    E --> F["MCP callTool(name, args)"]
    F --> G["Return MCP result"]

    D --> H{"Tool type?"}
    H -->|"BashTool"| I["BashTool.call()"]
    H -->|"FileReadTool"| J["FileReadTool.call()"]
    H -->|"FileEditTool"| K["FileEditTool.call()"]
    H -->|"FileWriteTool"| L["FileWriteTool.call()"]
    H -->|"Agent"| M["Spawn sub-agent"]
    H -->|Other| N["Generic tool.call()"]

    I --> I1["Check sandbox config"]
    I1 --> I2{"Sandbox enabled?"}
    I2 -->|macOS| I3["sandbox-exec -p profile cmd"]
    I2 -->|Linux| I4["SOCKS bridge execution"]
    I2 -->|Disabled| I5["Direct child_process.exec"]

    J --> J1["Validate path permissions"]
    J1 --> J2["fs.readFile with offset/limit"]
    J2 --> J3["Return content with line numbers"]

    M --> M1["Create agent context"]
    M1 --> M2["Fork messages if needed: bV0()"]
    M2 --> M3["Resolve agent model: JsA()"]
    M3 --> M4["yield* mVA() - agent generator"]
    M4 --> M5["Runs own s$() loop"]
```

## 4. Permission Check Flow

```mermaid
graph TD
    A["canUseTool(toolName, input)"] --> B["Get permission context"]
    B --> C{"Permission mode?"}

    C -->|"bypassPermissions"| D["ALLOW immediately"]
    C -->|"dontAsk"| E["ALLOW (agent mode)"]
    C -->|"plan"| F["DENY (read-only)"]
    C -->|"default"| G["Check rules"]
    C -->|"acceptEdits"| H["Allow edits, ask for others"]

    G --> I{"Match always-allow rules?"}
    I -->|Yes| J["ALLOW"]
    I -->|No| K{"Match deny rules?"}
    K -->|Yes| L["DENY"]
    K -->|No| M{"Is file operation?"}

    M -->|Yes| N["tD() - check path permission"]
    N --> O{"Path allowed?"}
    O -->|"allow"| P["ALLOW"]
    O -->|"deny"| Q["DENY"]
    O -->|"ask"| R["Prompt user"]

    M -->|No| R
    R --> S{"User response?"}
    S -->|Allow once| T["ALLOW"]
    S -->|Allow always| U["Add to always-allow rules"]
    S -->|Deny| V["DENY"]
```

## 5. MCP Server Lifecycle

```mermaid
graph TD
    A["Load MCP config"] --> B{"Config source?"}
    B -->|".mcp.json"| C["Read project .mcp.json"]
    B -->|"Settings"| D["Read from localSettings"]
    B -->|"MCPB"| E["Download + extract MCPB bundle"]

    C --> F["Normalize server names: UG()"]
    D --> F
    E --> F

    F --> G{"Transport type?"}
    G -->|"stdio"| H["Spawn child process"]
    G -->|"sse"| I["HTTP SSE connection"]
    G -->|"websocket"| J["WebSocket connection"]

    H --> K["Send: initialize"]
    I --> K
    J --> K

    K --> L["Receive: capabilities"]
    L --> M["Send: tools/list"]
    M --> N["Register tools as mcp__server__tool"]

    N --> O["Tool invocation"]
    O --> P["Send: tools/call"]
    P --> Q["Receive: tool result"]
    Q --> R["Return to agent loop"]

    K --> S["Send: resources/list"]
    S --> T["Register resources"]

    R --> U{"Server error?"}
    U -->|"Connection lost"| V["Restart MCP server process"]
    U -->|"Timeout"| W["k91() - get MCP_TIMEOUT"]
```

## 6. Compaction Flow

```mermaid
graph TD
    A["Token count check"] --> B{"input_tokens > threshold?"}
    B -->|No| C["Continue normally"]
    B -->|Yes| D["Start compaction"]

    D --> E["Set spinner: compacting"]
    E --> F["Call API: querySource='compact'"]
    F --> G["Stream summary response"]
    G --> H{"Summary valid?"}

    H -->|"Empty"| I["tengu_compact_failed: no_summary"]
    H -->|"API error prefix"| J["tengu_compact_failed: api_error"]
    H -->|"Prompt too long"| K["tengu_compact_failed: prompt_too_long"]
    H -->|"Valid"| L["Build new message history"]

    L --> M["c12() - refresh readFileState"]
    M --> N["Re-read tracked files"]
    N --> O["z$('compact') - run SessionStart hooks"]
    O --> P["Assemble: summary + boundary + attachments + hooks"]
    P --> Q["Track: tengu_compact telemetry"]
    Q --> R["Return compacted messages"]

    R --> S["Micro-compaction check"]
    S --> T{"Stale tool uses?"}
    T -->|Yes| U["Remove old tool_use blocks"]
    T -->|No| V["Done"]
    U --> V
```

## 7. Sub-Agent Spawn Flow

```mermaid
graph TD
    A["Agent tool invoked"] --> B["mVA() - agent generator"]
    B --> C["Resolve model: JsA()"]
    C --> D["Create agent ID: vQA()"]
    D --> E["Fork context: bV0()"]
    E --> F["Build agent system prompt"]
    F --> G["Create tool use context: x_A()"]

    G --> H{"Agent permission mode?"}
    H -->|"Defined on agent"| I["Use agent's permissionMode"]
    H -->|"Async agent"| J["Override permission context"]
    H -->|"Inherit"| K["Use parent's context"]

    I --> L["s$() - run agent's own loop"]
    J --> L
    K --> L

    L --> M["Yield events back to parent"]
    M --> N{"Event type?"}
    N -->|"assistant"| O["Collect response"]
    N -->|"progress"| P["Forward to parent"]
    N -->|"user"| Q["Internal to agent"]
```

## 8. Slash Command Dispatch

```
User types /<command>
  ├── Parse command name and args
  ├── Look up in commands array
  │   ├── /clear    → Clear conversation history
  │   ├── /compact  → Trigger manual compaction
  │   ├── /config   → Open config panel (React component)
  │   ├── /context  → Show context usage grid
  │   ├── /cost     → Show session cost/duration
  │   ├── /doctor   → Run diagnostics
  │   ├── /help     → Show help
  │   ├── /memory   → Edit CLAUDE.md files
  │   ├── /mcp      → Manage MCP servers
  │   ├── /model    → Switch model (YI1 component)
  │   ├── /plan     → View/open session plan
  │   ├── /resume   → Resume conversation
  │   ├── /review   → Review pull request
  │   ├── /status   → Show version, model, account info
  │   ├── /vim      → Toggle vim mode
  │   └── 30+ more commands
  ├── Execute command handler
  │   ├── Some return React components (config, context)
  │   ├── Some modify app state (model, vim)
  │   └── Some yield messages (compact, review)
  └── Return: {newMessages, contextModifier, allowedTools, model}
```
