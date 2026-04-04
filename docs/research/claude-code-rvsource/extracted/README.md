# Extracted Source - Claude Code v2.1.91

Decompiled source modules from `@anthropic-ai/claude-code@2.1.91`.

## Directory Structure

```
extracted/
  source/                   # Source code only (no .rvf files)
    core/                   # Core execution engine
      agent-loop.js         # Main async generator
      context-manager.js    # Token counting and compaction
      streaming-handler.js  # SSE event processing
      session.js            # Session management
    tools/                  # Tool system
      tool-dispatch.js      # Tool registry and routing
      mcp/
        mcp-client.js       # MCP protocol client
    permissions/            # Permission system
      permission-system.js  # Permission checker and sandbox
    ui/                     # User interface
      commands.js           # Slash commands
      command-defs.js       # Command definitions
    config/                 # Configuration
      config.js             # Settings schema
      env-vars.js           # Environment variables
      model-provider.js     # Model selection/routing
    telemetry/              # Observability
      telemetry.js          # OpenTelemetry integration
      telemetry-events.js   # Event definitions
    types/                  # Type info
      class-hierarchy.js    # Class declarations
      api-endpoints.js      # API endpoints
    uncategorized/          # Remaining bundle code
      uncategorized.js

  rvf/                      # RVF containers only (no .js files)
    master.rvf              # All vectors combined
    core.rvf                # Core modules only
    tools.rvf               # Tool modules only
    permissions.rvf         # Permission modules only
    config.rvf              # Configuration modules only
    telemetry.rvf           # Telemetry modules only
    ui.rvf                  # UI modules only
    types.rvf               # Type modules only
    uncategorized.rvf       # Uncategorized modules

  metrics.json              # Overall metrics
```

## Metrics

| Metric | Value |
|--------|-------|
| Version | 2.1.91 |
| Bundle size | 12.6 MB |
| Classes | 1632 |
| Functions | 19906 |
| Modules | 17 |
| Coverage | 100.6% |
| Extracted | 2026-04-03T03:17:18.261Z |

## RVF Containers

Source and RVF files are cleanly separated:
- `rvf/master.rvf` - Master RVF (all modules, all vectors)
- `rvf/core.rvf` - Core execution modules only
- `rvf/tools.rvf` - Tool system modules only
- `rvf/permissions.rvf` - Permission modules only
- `rvf/config.rvf` - Configuration modules only
- `rvf/telemetry.rvf` - Telemetry modules only

Each RVF container has an accompanying `.manifest.json` sidecar.
