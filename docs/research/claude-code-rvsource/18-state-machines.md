# 18 - State Machine Diagrams

State machines extracted from actual code patterns in `cli.js` (v2.0.62).

## 1. Agent Loop State Machine

The agent loop (`s$`) is the primary state machine. It operates as an async generator
that transitions through states by yielding events.

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
              ┌──────────┐                                    │
              │  IDLE     │  (waiting for user input)         │
              └────┬─────┘                                    │
                   │ User submits query                       │
                   ▼                                          │
              ┌──────────┐                                    │
              │ PREPARING │  JP() - prepareQuery              │
              │           │  Count tokens, build system prompt │
              └────┬─────┘                                    │
                   │                                          │
                   ▼                                          │
              ┌──────────┐     ┌──────────┐                   │
              │ CHECKING  │────►│COMPACTING│                   │
              │ CONTEXT   │    │          │  Run summarization │
              └────┬─────┘    └────┬─────┘                   │
                   │               │ Done                     │
                   │◄──────────────┘                          │
                   ▼                                          │
              ┌──────────┐                                    │
              │ STREAMING │  Call Anthropic Messages API       │
              │           │  Process SSE events                │
              └────┬─────┘                                    │
                   │ Response complete                        │
                   ▼                                          │
              ┌──────────┐                                    │
              │ EVALUATING│  Check stop_reason                 │
              └────┬─────┘                                    │
                   │                                          │
          ┌────────┼────────┐                                 │
          ▼        ▼        ▼                                 │
    ┌─────────┐ ┌──────┐ ┌──────────┐                        │
    │end_turn │ │max_  │ │tool_use  │                        │
    │         │ │tokens│ │          │                        │
    └────┬────┘ └──┬───┘ └────┬─────┘                        │
         │         │          │                               │
         ▼         ▼          ▼                               │
    ┌─────────┐ ┌──────┐ ┌──────────┐                        │
    │ DONE    │ │ DONE │ │DISPATCHING│  Process tool calls    │
    └─────────┘ └──────┘ │ TOOLS    │                        │
                          └────┬─────┘                        │
                               │                              │
                     ┌─────────┼─────────┐                    │
                     ▼         ▼         ▼                    │
              ┌──────────┐ ┌──────┐ ┌──────────┐             │
              │PERMISSION│ │EXEC  │ │MCP CALL  │             │
              │CHECK     │ │TOOL  │ │          │             │
              └────┬─────┘ └──┬───┘ └────┬─────┘             │
                   │          │          │                    │
                   ▼          ▼          ▼                    │
              ┌──────────────────────────────┐                │
              │ ASSEMBLING TOOL RESULTS      │                │
              └────────────┬─────────────────┘                │
                           │                                  │
                           │ yield* s$() (recursive)          │
                           └──────────────────────────────────┘
```

### State Transitions Table

| From | To | Trigger | Action |
|------|----|---------|--------|
| IDLE | PREPARING | User input | `JP()` / `c_7()` |
| PREPARING | CHECKING_CONTEXT | Query built | `qh2()` token count |
| CHECKING_CONTEXT | COMPACTING | Tokens > threshold | Compaction API call |
| CHECKING_CONTEXT | STREAMING | Tokens OK | `s$()` entry |
| COMPACTING | CHECKING_CONTEXT | Summary received | Replace messages |
| STREAMING | EVALUATING | `message_stop` SSE | Accumulate response |
| EVALUATING | DONE | `end_turn` | Yield stop |
| EVALUATING | DONE | `max_tokens` | Yield stop |
| EVALUATING | DISPATCHING_TOOLS | `tool_use` blocks | Process tools |
| DISPATCHING_TOOLS | PERMISSION_CHECK | Each tool | `canUseTool()` |
| PERMISSION_CHECK | EXEC_TOOL | Allowed | `tool.call()` |
| PERMISSION_CHECK | ASSEMBLING | Denied | Error result |
| EXEC_TOOL | ASSEMBLING | Tool done | Collect result |
| ASSEMBLING | STREAMING | `yield* s$()` | Recursive loop |

## 2. Permission System State Machine

```
                 ┌───────────────────┐
                 │  PERMISSION CHECK  │
                 │  canUseTool()      │
                 └────────┬──────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ MODE:    │ │ MODE:    │ │ MODE:    │
        │ bypass   │ │ default  │ │ plan     │
        │ dontAsk  │ │ acceptEd │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        ┌────────┐   ┌──────────┐  ┌────────┐
        │ ALLOW  │   │CHECK RULES│  │ DENY   │
        └────────┘   └────┬─────┘  └────────┘
                          │
              ┌───────────┼───────────┐
              ▼           ▼           ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ MATCH    │ │ MATCH    │ │ NO MATCH │
        │ ALLOW    │ │ DENY     │ │          │
        │ RULE     │ │ RULE     │ │          │
        └────┬─────┘ └────┬─────┘ └────┬─────┘
             │             │             │
             ▼             ▼             ▼
        ┌────────┐   ┌────────┐   ┌──────────┐
        │ ALLOW  │   │ DENY   │   │ ASK USER │
        └────────┘   └────────┘   └────┬─────┘
                                       │
                              ┌────────┼────────┐
                              ▼        ▼        ▼
                        ┌────────┐┌────────┐┌──────────┐
                        │ ALLOW  ││ ALLOW  ││  DENY    │
                        │ ONCE   ││ ALWAYS ││          │
                        └────────┘└───┬────┘└──────────┘
                                      │
                                      ▼
                               ┌──────────────┐
                               │ ADD TO ALWAYS │
                               │ ALLOW RULES   │
                               └──────────────┘
```

### Permission Rule Matching

```
Rule Format: "ToolName" or "ToolName(pattern)" or "ToolName(pattern:wildcard)"

Examples:
  "Bash"                    → matches all Bash commands
  "Bash(npm run:*)"         → matches "npm run build", "npm run test", etc.
  "Bash(git commit:*)"      → matches "git commit -m ..."
  "WebFetch(domain:*.com)"  → matches any .com domain
  "Edit"                    → matches all file edits

Matching algorithm:
  1. Exact tool name match
  2. If rule has pattern: extract command from input
  3. If rule has wildcard (*): prefix match
  4. If rule has glob: glob match (limited on Linux)
```

## 3. Session Lifecycle State Machine

```
              ┌──────────┐
              │  INIT    │
              └────┬─────┘
                   │ Load settings, credentials
                   ▼
              ┌──────────┐
              │ AUTH     │  Validate API key / OAuth
              └────┬─────┘
                   │
           ┌───────┼───────┐
           ▼       ▼       ▼
     ┌────────┐ ┌──────┐ ┌──────┐
     │ CLI    │ │ SDK  │ │ MCP  │
     │ REPL   │ │ MODE │ │SERVER│
     └───┬────┘ └──┬───┘ └──┬───┘
         │         │        │
         ▼         ▼        ▼
    ┌──────────────────────────┐
    │      ACTIVE SESSION      │
    │  ┌─────────────────────┐ │
    │  │ AppState (Ll())     │ │
    │  │ - mainLoopModel     │ │
    │  │ - permissionContext  │ │
    │  │ - mcp clients       │ │
    │  │ - plugins           │ │
    │  │ - todos             │ │
    │  │ - fileHistory       │ │
    │  │ - thinkingEnabled   │ │
    │  └─────────────────────┘ │
    └──────────┬───────────────┘
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
  ┌────────┐ ┌──────┐ ┌──────────┐
  │QUERYING│ │IDLE  │ │COMPACTING│
  │(s$ loop│ │      │ │          │
  └───┬────┘ └──────┘ └──────────┘
      │
      ▼
  ┌──────────┐
  │SAVING    │  History → ~/.claude/history.jsonl
  │          │  Session → ~/.claude/sessions/
  └────┬─────┘
       │
       ▼
  ┌──────────┐
  │  EXIT    │  Cleanup MCP connections
  └──────────┘
```

## 4. Streaming Response State Machine

```
         ┌──────────────┐
         │ REQUESTING   │  (spinner active)
         └──────┬───────┘
                │ First SSE event
                ▼
         ┌──────────────┐
         │ message_start│  Initialize accumulator
         └──────┬───────┘
                │
                ▼
    ┌───────────────────────┐
    │ PROCESSING BLOCKS     │◄──────────────┐
    └───────────┬───────────┘               │
                │                           │
       ┌────────┼────────┐                  │
       ▼        ▼        ▼                  │
  ┌────────┐┌────────┐┌──────────┐          │
  │ TEXT   ││TOOL USE││THINKING  │          │
  │ BLOCK  ││ BLOCK  ││ BLOCK    │          │
  └───┬────┘└───┬────┘└────┬─────┘          │
      │         │          │                │
      ▼         ▼          ▼                │
  ┌────────┐┌────────┐┌──────────┐          │
  │text_   ││input_  ││thinking_ │          │
  │delta   ││json_  ││delta     │  Repeat   │
  │chunks  ││delta  ││chunks    │──────────►│
  └───┬────┘└───┬────┘└────┬─────┘          │
      │         │          │                │
      ▼         ▼          ▼                │
  ┌──────────────────────────┐              │
  │ content_block_stop       │──────────────┘
  └──────────┬───────────────┘
             │ All blocks done
             ▼
  ┌──────────────────┐
  │ message_delta    │  Usage info, stop_reason
  └──────────┬───────┘
             ▼
  ┌──────────────────┐
  │ message_stop     │  Response complete
  └──────────────────┘
```

### Stream Mode Transitions (UI)

```
"requesting"  ──[first text_delta]──►  "responding"
"responding"  ──[tool_use block]──►    "tool-input"
"tool-input"  ──[block_stop]──►        "responding"  (if more text)
"responding"  ──[message_stop]──►      "requesting"  (next turn)
```

## 5. MCP Connection State Machine

```
         ┌──────────────┐
         │ DISCONNECTED │
         └──────┬───────┘
                │ Config loaded
                ▼
         ┌──────────────┐
         │ CONNECTING   │  Spawn process / HTTP connect
         └──────┬───────┘
                │
         ┌──────┼──────┐
         ▼      ▼      ▼
    ┌────────┐┌─────┐┌──────┐
    │ STDIO  ││ SSE ││ WS   │
    └───┬────┘└──┬──┘└──┬───┘
        │        │      │
        └────────┼──────┘
                 ▼
         ┌──────────────┐
         │ INITIALIZING │  Send: initialize
         └──────┬───────┘
                │ Receive: capabilities
                ▼
         ┌──────────────┐
         │ READY        │  List tools, resources
         └──────┬───────┘
                │
        ┌───────┼───────┐
        ▼       ▼       ▼
   ┌────────┐┌──────┐┌──────────┐
   │TOOL    ││IDLE  ││RESOURCE  │
   │CALL    ││      ││READ     │
   └───┬────┘└──────┘└────┬─────┘
       │                   │
       └───────┬───────────┘
               ▼
        ┌──────────────┐
        │ ERROR        │
        └──────┬───────┘
               │ Retry / restart
               ▼
        ┌──────────────┐
        │ RESTARTING   │  "Restarting MCP server process"
        └──────┬───────┘
               │
               ▼
        ┌──────────────┐
        │ CONNECTING   │  (back to start)
        └──────────────┘
```

## 6. Sandbox State Machine (Linux)

```
         ┌──────────────┐
         │ CHECK CONFIG │  IQ()?.sandbox?.enabled
         └──────┬───────┘
                │
         ┌──────┼──────┐
         ▼      ▼      ▼
    ┌────────┐┌─────┐┌──────────┐
    │DISABLED││macOS││ LINUX    │
    └────────┘│SEATB││ SOCKS    │
              └──┬──┘└────┬─────┘
                 │        │
                 ▼        ▼
         ┌────────────────────┐
         │ COMMAND RECEIVED   │
         └────────┬───────────┘
                  │
         ┌────────┼────────┐
         ▼        ▼        ▼
    ┌────────┐┌──────┐┌──────────┐
    │EXCLUDED││SANDBO││DANGEROUS │
    │COMMAND ││XED   ││LY DISABLE│
    └───┬────┘└──┬───┘└────┬─────┘
        │        │         │
        ▼        ▼         ▼
    ┌────────┐┌────────┐┌────────┐
    │DIRECT  ││WRAPPED ││DIRECT  │
    │EXEC    ││EXEC    ││EXEC    │
    └────────┘└────────┘└────────┘
```

## 7. Model Selection State Machine

```
    ┌──────────────────────────────┐
    │ ye() - resolveModel          │
    └──────────────┬───────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐   ┌──────────┐   ┌──────────┐
│ ENV    │   │ USER     │   │ DEFAULT  │
│ OVERRIDE│  │ SELECTED │   │          │
│ANTHROPIC│  │(AppState)│   │ Fg1      │
│_MODEL  │   └────┬─────┘   └────┬─────┘
└───┬────┘        │              │
    │        ┌────┼────┐         │
    │        ▼    ▼    ▼         │
    │   ┌──────┐┌───┐┌──────┐   │
    │   │inherit││   ││direct│   │
    │   └──┬───┘│   │└──┬───┘   │
    │      │    │   │   │       │
    │      ▼    │   │   │       │
    │   ┌──────────┐│   │       │
    │   │PERMISSION│◄───┘       │
    │   │MODE CHECK│            │
    │   └────┬─────┘            │
    │        │                  │
    │   ┌────┼────┐             │
    │   ▼    ▼    ▼             │
    │ ┌────┐┌────┐┌────┐        │
    │ │plan││def ││bypass       │
    │ │mode││    ││    │        │
    │ └─┬──┘└─┬──┘└─┬──┘        │
    │   │     │     │           │
    └───┴─────┴─────┴───────────┘
                │
                ▼
         ┌──────────┐
         │ YF()     │  Normalize model string
         │ RESOLVED │
         └──────────┘
```

## Key Observations

1. **The agent loop is recursive**, not iterative. Each tool execution triggers
   a new `yield*` delegation that creates a stack of generators. This enables
   natural backpressure and cancellation via abort controllers.

2. **Compaction is a sub-state of the main loop**, triggered by token counting
   before each API call. It can itself fail and has retry/fallback logic.

3. **Permission checking is synchronous within the loop** but can suspend
   for user input (interactive mode) or return immediately (bypass/dontAsk modes).

4. **Streaming is event-driven** with a well-defined state machine matching
   the Anthropic Messages API SSE protocol exactly.

5. **MCP connections are persistent** with automatic restart on failure,
   following the MCP specification's connection lifecycle.
