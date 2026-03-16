# ADR-111: Ruvocal UI Integration with rvAgent

| Field       | Value                                           |
|-------------|------------------------------------------------|
| **Status**  | In Progress                                     |
| **Date**    | 2026-03-15                                      |
| **Updated** | 2026-03-15                                      |
| **Authors** | ruvnet                                          |
| **Series**  | ADR-093 (DeepAgents), ADR-104 (MCP Skills), ADR-108 (ruvbot) |
| **Related** | ADR-106 (RuViX Kernel), ADR-105 (MCP Implementation) |
| **Source**  | https://github.com/ruvnet/ruflo/tree/main/ruflo/src/ruvocal |

## Context

The RuVector ecosystem requires a modern, production-ready chat interface for interacting with rvAgent. **Ruvocal** (from the ruflo project) is a SvelteKit-based chat UI originally designed for HuggingChat that provides:

1. **Modern Chat Interface**: Real-time streaming, markdown rendering, code highlighting
2. **Multi-Model Support**: OpenAI-compatible API abstraction
3. **MCP Bridge**: Existing Model Context Protocol integration for tool calling
4. **Conversation Management**: Persistent chat history with MongoDB
5. **Theming System**: Customizable branding and appearance
6. **LLM Router**: Intelligent model selection (Arch-Router-1.5B)

### Why Ruvocal?

| Feature | Ruvocal | Build from Scratch |
|---------|---------|-------------------|
| Development Time | Days (fork + adapt) | Weeks |
| Chat UI Components | Complete | Build all |
| Streaming Support | Built-in | Implement SSE/WS |
| MCP Integration | Has mcp-bridge | Build from spec |
| Mobile Responsive | Yes | Design + build |
| Dark Mode | Yes | Implement |
| Code Highlighting | Shiki/Prism | Choose + integrate |
| Message History | MongoDB | Design schema |

### Current rvAgent Architecture

```
rvAgent Crates:
├── rvagent-core      # State, config, witness
├── rvagent-tools     # Tool definitions, registry
├── rvagent-middleware # 9-layer pipeline
├── rvagent-backends  # Filesystem, sandbox
├── rvagent-subagents # Orchestration, validation
├── rvagent-mcp       # MCP server (ADR-104)
└── rvagent-acp       # ACP server
```

Missing: A production chat UI that connects users to rvAgent's capabilities.

---

## Decision

Integrate Ruvocal as the official web UI for rvAgent, adapting it to connect to rvAgent's MCP server while preserving its existing features.

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Ruvocal + rvAgent Integration                         │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        Ruvocal (SvelteKit UI)                            │ │
│  │                                                                           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐  │ │
│  │  │   Chat UI    │  │   Sidebar    │  │   Settings   │  │   Themes    │  │ │
│  │  │  Components  │  │  Navigation  │  │    Panel     │  │   System    │  │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘  │ │
│  │         │                  │                 │                           │ │
│  │         └──────────────────┼─────────────────┘                           │ │
│  │                            ▼                                              │ │
│  │  ┌──────────────────────────────────────────────────────────────────────┐│ │
│  │  │                       APIClient Layer                                 ││ │
│  │  │                                                                        ││ │
│  │  │   • OpenAI-compatible endpoints                                       ││ │
│  │  │   • Streaming response handling                                        ││ │
│  │  │   • Tool call marshaling                                               ││ │
│  │  │   • Error recovery                                                     ││ │
│  │  └──────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       MCP Bridge (ruvocal/mcp-bridge)                    │ │
│  │                                                                           │ │
│  │   ┌────────────────┐  ┌────────────────┐  ┌────────────────┐            │ │
│  │   │  stdio-kernel  │  │  Tool Router   │  │  Result Parser │            │ │
│  │   └────────────────┘  └────────────────┘  └────────────────┘            │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼ MCP Protocol                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         rvAgent MCP Server                               │ │
│  │                                                                           │ │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │   │ Tool Runtime │  │   Witness    │  │  Middleware  │  │   State     │ │ │
│  │   │  (46 tools)  │  │    Chain     │  │   Pipeline   │  │  Backend    │ │ │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      RuVector Native Backends                            │ │
│  │                                                                           │ │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌─────────────┐ │ │
│  │   │  Filesystem  │  │   Sandbox    │  │     RVF      │  │    HNSW     │ │ │
│  │   │   Backend    │  │   Backend    │  │   Runtime    │  │   Memory    │ │ │
│  │   └──────────────┘  └──────────────┘  └──────────────┘  └─────────────┘ │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Integration Points

### 1. Repository Structure

Fork ruvocal into the RuVector monorepo:

```
ruvector/
├── crates/
│   └── rvAgent/          # Existing
├── ui/
│   └── ruvocal/          # NEW - Forked from ruflo
│       ├── src/
│       │   ├── lib/
│       │   │   ├── components/    # Chat UI components
│       │   │   ├── stores/        # Svelte state stores
│       │   │   ├── server/        # Server-side API handlers
│       │   │   └── utils/         # Utilities
│       │   └── routes/            # SvelteKit pages
│       ├── mcp-bridge/            # MCP stdio kernel
│       ├── static/                # Assets (rebrand to RuVector)
│       ├── .env.example           # Configuration template
│       ├── Dockerfile             # Container build
│       └── package.json
└── docs/adr/
    └── ADR-111-...       # This document
```

### 2. MCP Bridge Configuration

Adapt the existing mcp-bridge to connect to rvAgent:

```javascript
// ui/ruvocal/mcp-bridge/rvagent-kernel.js

import { spawn } from 'child_process';
import { McpClient } from '@modelcontextprotocol/sdk';

export class RvAgentKernel {
  constructor(config) {
    this.config = {
      // rvAgent MCP server binary or via cargo
      command: config.command || 'cargo',
      args: config.args || ['run', '-p', 'rvagent-mcp', '--', 'stdio'],
      cwd: config.cwd || process.env.RVAGENT_PATH,
      ...config
    };
  }

  async connect() {
    // Spawn rvAgent MCP server in stdio mode
    this.process = spawn(this.config.command, this.config.args, {
      cwd: this.config.cwd,
      stdio: ['pipe', 'pipe', 'inherit']
    });

    this.client = new McpClient();
    await this.client.connect(this.process.stdin, this.process.stdout);

    // List available tools from rvAgent
    const tools = await this.client.listTools();
    console.log(`Connected to rvAgent with ${tools.length} tools`);

    return tools;
  }

  async executeTool(name, args) {
    return await this.client.executeTool(name, args);
  }

  async shutdown() {
    await this.client.close();
    this.process.kill();
  }
}
```

### 3. APIClient Adaptation

Modify Ruvocal's APIClient to route tool calls through rvAgent:

```typescript
// ui/ruvocal/src/lib/APIClient.ts

import type { RvAgentKernel } from '../mcp-bridge/rvagent-kernel';

export class RvAgentAPIClient {
  private kernel: RvAgentKernel;
  private baseUrl: string;

  constructor(config: {
    openaiBaseUrl: string;
    rvagentKernel: RvAgentKernel;
  }) {
    this.baseUrl = config.openaiBaseUrl;
    this.kernel = config.rvagentKernel;
  }

  // Chat completion with tool calling
  async chat(messages: Message[], options: ChatOptions): Promise<AsyncIterable<StreamChunk>> {
    const response = await fetch(`${this.baseUrl}/chat/completions`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${options.apiKey}`
      },
      body: JSON.stringify({
        model: options.model,
        messages,
        tools: await this.getAvailableTools(),
        stream: true
      })
    });

    return this.processStream(response);
  }

  // Get tools from rvAgent MCP server
  async getAvailableTools(): Promise<Tool[]> {
    const mcpTools = await this.kernel.client.listTools();

    // Convert MCP tool format to OpenAI function format
    return mcpTools.map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema
      }
    }));
  }

  // Execute tool calls via rvAgent
  async executeToolCall(toolCall: ToolCall): Promise<ToolResult> {
    const result = await this.kernel.executeTool(
      toolCall.function.name,
      JSON.parse(toolCall.function.arguments)
    );

    return {
      tool_call_id: toolCall.id,
      role: 'tool',
      content: JSON.stringify(result)
    };
  }
}
```

### 4. Environment Configuration

```bash
# ui/ruvocal/.env.example

# LLM Provider (OpenAI-compatible API)
OPENAI_BASE_URL=http://localhost:8080/v1
OPENAI_API_KEY=sk-rvector-local

# rvAgent MCP Server
RVAGENT_PATH=/path/to/ruvector/crates/rvAgent
RVAGENT_MCP_MODE=stdio
RVAGENT_WITNESS_ENABLED=true

# Database (conversation history)
MONGODB_URL=mongodb://localhost:27017/ruvocal
# Or use embedded mode for development:
# MONGODB_DB_PATH=./db

# Theming
PUBLIC_APP_NAME=RuVector Agent
PUBLIC_APP_ASSETS=ruvector
PUBLIC_APP_DESCRIPTION=Intelligent AI Agent with RuVector

# Feature Flags
ENABLE_MCP_TOOLS=true
ENABLE_LLM_ROUTER=false
ENABLE_WITNESS_UI=true
```

### 5. Svelte Store Integration

Create a store for rvAgent state:

```typescript
// ui/ruvocal/src/lib/stores/rvagent.ts

import { writable, derived } from 'svelte/store';
import type { WitnessEntry, AgentState } from '$lib/types/rvagent';

// Witness chain visualization
export const witnessChain = writable<WitnessEntry[]>([]);

// Agent execution state
export const agentState = writable<AgentState>({
  status: 'idle',
  currentTool: null,
  progress: 0
});

// Available tools from rvAgent
export const availableTools = writable<Tool[]>([]);

// Derived: active tool calls in current conversation
export const activeToolCalls = derived(
  witnessChain,
  $chain => $chain.filter(entry => entry.status === 'executing')
);

// Actions
export function addWitnessEntry(entry: WitnessEntry) {
  witnessChain.update(chain => [...chain, entry]);
}

export function updateAgentStatus(status: AgentState['status'], tool?: string) {
  agentState.update(state => ({
    ...state,
    status,
    currentTool: tool || null
  }));
}
```

### 6. Witness Chain UI Component

Add a component to visualize rvAgent's witness chain:

```svelte
<!-- ui/ruvocal/src/lib/components/WitnessChain.svelte -->

<script lang="ts">
  import { witnessChain, agentState } from '$lib/stores/rvagent';
  import { slide } from 'svelte/transition';

  export let expanded = false;
</script>

<div class="witness-panel" class:expanded>
  <button
    class="toggle-btn"
    on:click={() => expanded = !expanded}
    aria-label="Toggle witness chain"
  >
    <span class="icon">🔗</span>
    <span class="count">{$witnessChain.length}</span>
  </button>

  {#if expanded}
    <div class="witness-list" transition:slide>
      {#each $witnessChain as entry, i (entry.id)}
        <div class="witness-entry" class:executing={entry.status === 'executing'}>
          <span class="index">#{i + 1}</span>
          <span class="tool-name">{entry.toolName}</span>
          <span class="hash" title={entry.hash}>
            {entry.hash.slice(0, 8)}...
          </span>
          <span class="status {entry.status}">
            {entry.status === 'completed' ? '✓' : entry.status === 'executing' ? '⏳' : '✗'}
          </span>
        </div>
      {/each}
    </div>
  {/if}
</div>

<style>
  .witness-panel {
    position: fixed;
    bottom: 1rem;
    right: 1rem;
    background: var(--bg-secondary);
    border-radius: 0.5rem;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    z-index: 1000;
  }

  .witness-entry {
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem;
    border-bottom: 1px solid var(--border-color);
    font-family: monospace;
    font-size: 0.75rem;
  }

  .witness-entry.executing {
    background: var(--accent-bg);
    animation: pulse 1s infinite;
  }

  .status.completed { color: var(--success); }
  .status.failed { color: var(--error); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.7; }
  }
</style>
```

---

## Tool Categories in UI

Organize rvAgent tools into user-friendly categories:

| Category | Tools | UI Representation |
|----------|-------|-------------------|
| **Files** | read_file, write_file, list_directory | File explorer panel |
| **Code** | search_code, edit_file, run_tests | Code editor integration |
| **Shell** | execute_command, bash | Terminal panel |
| **Memory** | semantic_search, store_memory | Knowledge sidebar |
| **Web** | web_fetch, web_search | Browser preview |
| **Git** | git_status, git_commit, git_diff | Version control panel |

---

## Deployment Options

### Option 1: Development (Local)

```bash
cd ui/ruvocal
npm install
npm run dev -- --open

# In another terminal
cd crates/rvAgent
cargo run -p rvagent-mcp -- stdio
```

### Option 2: Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  ruvocal:
    build:
      context: ./ui/ruvocal
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    environment:
      - RVAGENT_MCP_MODE=socket
      - RVAGENT_HOST=rvagent
      - RVAGENT_PORT=9000
    depends_on:
      - rvagent
      - mongodb

  rvagent:
    build:
      context: .
      dockerfile: crates/rvAgent/Dockerfile
    command: ["rvagent-mcp", "socket", "--port", "9000"]
    volumes:
      - ./workspace:/workspace

  mongodb:
    image: mongo:7
    volumes:
      - mongodb_data:/data/db

volumes:
  mongodb_data:
```

### Option 3: Cloud Run (Production)

```yaml
# cloudbuild.yaml
steps:
  # Build rvAgent MCP server
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/rvagent-mcp', '-f', 'crates/rvAgent/Dockerfile', '.']

  # Build Ruvocal UI
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/ruvocal-ui', '-f', 'ui/ruvocal/Dockerfile', '.']

  # Deploy
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'gcloud'
    args: ['run', 'deploy', 'ruvocal', '--image', 'gcr.io/$PROJECT_ID/ruvocal-ui', '--region', 'us-central1']
```

---

## Rebranding Checklist

| Item | Location | Change |
|------|----------|--------|
| App Name | `.env` | `PUBLIC_APP_NAME=RuVector Agent` |
| Logo | `static/logo.svg` | RuVector logo |
| Favicon | `static/favicon.ico` | RuVector icon |
| Colors | `tailwind.config.cjs` | RuVector palette |
| Footer | `src/routes/+layout.svelte` | RuVector attribution |
| Title | `src/app.html` | `<title>RuVector Agent</title>` |
| Manifest | `static/manifest.json` | PWA metadata |

---

## Security Considerations

### Tool Execution Sandboxing

All tool execution goes through rvAgent's sandbox backend (ADR-103 C5):

```rust
// rvAgent enforces sandbox policy
pub struct SandboxPolicy {
    allowed_paths: Vec<PathBuf>,
    denied_commands: Vec<String>,
    max_execution_time: Duration,
    memory_limit: usize,
}
```

### Authentication Flow

```
┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐
│  User   │────▶│ Ruvocal │────▶│  Auth   │────▶│ rvAgent │
│ Browser │     │   UI    │     │ Service │     │   MCP   │
└─────────┘     └─────────┘     └─────────┘     └─────────┘
     │               │               │               │
     │  1. Login     │               │               │
     │──────────────▶│               │               │
     │               │ 2. Verify     │               │
     │               │──────────────▶│               │
     │               │ 3. JWT token  │               │
     │               │◀──────────────│               │
     │ 4. Session    │               │               │
     │◀──────────────│               │               │
     │               │ 5. Tool call + JWT            │
     │               │──────────────────────────────▶│
     │               │ 6. Verify & execute           │
     │               │◀──────────────────────────────│
```

### Input Validation

Ruvocal uses rvAgent's SubAgentResultValidator (ADR-103 C8) for all responses:

- Response length limits
- Injection pattern detection
- Control character stripping
- Prototype pollution prevention

---

## Implementation Phases

### Phase 1: Fork & Setup (Week 1) ✅ COMPLETE

- [x] Fork ruvocal to `ui/ruvocal/`
- [x] Remove HuggingFace-specific code
- [x] Update dependencies
- [x] Configure MCP connection (via π.ruv.io brain server)
- [x] Basic chat flow working

### Phase 2: Integration (Week 2) 🔄 IN PROGRESS

- [x] MCP bridge to π brain server (alternative to direct rvAgent)
- [x] Connect APIClient to π Brain tools (91 MCP tools available)
- [ ] Add witness chain visualization (NOT STARTED)
- [x] Tool category organization (mcpExamples updated)
- [x] Error handling + recovery (evidence_links transform, witness_hash fallback)

### Phase 3: Polish (Week 3) 🔄 IN PROGRESS

- [x] Rebranding (logos, colors, text) - Gold #e8a634, Dark #020205
- [x] Dark mode default (app.html, switchTheme.ts)
- [x] Foundation-inspired animated background (FoundationBackground.svelte)
- [x] Thinking block collapse (THINK_BLOCK_REGEX added)
- [ ] Mobile responsiveness testing (NOT STARTED)
- [ ] Accessibility audit (NOT STARTED)
- [ ] Performance optimization (NOT STARTED)

### Phase 4: Production (Week 4) ⏳ PENDING

- [ ] Docker images
- [ ] Cloud Run deployment (π.ruv.io deployed, UI needs separate deploy)
- [ ] CI/CD pipeline
- [ ] Documentation
- [ ] User guide

---

## Current Implementation Status

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| RuVocal UI Fork | ✅ Complete | `ui/ruvocal/` | SvelteKit 2 + Svelte 5 |
| MCP Bridge | ✅ Working | π.ruv.io | 91 tools via brain server |
| Dark Mode | ✅ Complete | `app.html`, `switchTheme.ts` | Default theme |
| Foundation Background | ✅ Complete | `FoundationBackground.svelte` | Canvas particle animation |
| Thinking Collapse | ✅ Complete | `ChatMessage.svelte` | THINK_BLOCK_REGEX |
| Gold Color Scheme | ✅ Complete | Tailwind config | #e8a634 primary |
| Query Suggestions | ✅ Complete | `mcpExamples.ts` | π Brain focused |
| brain_page_delta | ✅ Fixed | `routes.rs` | evidence_links transform |
| Witness Chain UI | ❌ Missing | - | Not implemented |
| Direct rvAgent MCP | ❌ Missing | - | Uses π brain instead |
| rvAgent Kernel | ❌ Missing | - | Planned for Phase 2 |

---

## Consequences

### Positive

1. **Rapid Development**: Leveraging mature chat UI saves weeks of development
2. **Feature-Rich**: Streaming, code highlighting, themes included
3. **MCP Native**: Existing mcp-bridge reduces integration effort
4. **Modern Stack**: SvelteKit provides excellent DX and performance
5. **Witness Transparency**: Users can see tool execution chain

### Negative

1. **Maintenance Burden**: Must track upstream ruvocal changes
2. **Node.js Dependency**: UI requires Node.js runtime
3. **MongoDB Dependency**: Conversation persistence requires database

### Mitigations

- Pin to specific ruvocal version, selectively merge updates
- Embed MongoDB option reduces ops burden
- Consider future Rust-native UI (Dioxus, Leptos) for full-stack Rust

---

## Related ADRs

| ADR | Relevance |
|-----|-----------|
| ADR-093 | DeepAgents Rust conversion overview |
| ADR-104 | rvAgent MCP Skills & Topology |
| ADR-105 | MCP Implementation Details |
| ADR-106 | RuViX Kernel Integration |
| ADR-108 | ruvbot Integration Architecture |
| ADR-103 C5 | Sandbox Contract |
| ADR-103 C8 | SubAgent Result Validation |

---

## References

- [Ruvocal Source (ruflo)](https://github.com/ruvnet/ruflo/tree/main/ruflo/src/ruvocal)
- [MCP Specification](https://spec.modelcontextprotocol.io/)
- [SvelteKit Documentation](https://kit.svelte.dev/)
- [rvAgent MCP Server](../crates/rvAgent/rvagent-mcp/)

---

## Appendix: Ruvocal Component Mapping

| Ruvocal Component | Purpose | rvAgent Integration |
|-------------------|---------|---------------------|
| `lib/APIClient.ts` | LLM communication | Add rvAgent tool routing |
| `lib/buildPrompt.ts` | Prompt construction | Include system prompt from rvAgent |
| `lib/components/ChatMessage.svelte` | Message rendering | Add tool call visualization |
| `lib/stores/` | State management | Add rvAgent state stores |
| `routes/conversation/` | Chat pages | Integrate witness panel |
| `mcp-bridge/` | Tool execution | Replace with rvAgent kernel |
| `server/` | API handlers | Add rvAgent health endpoints |
