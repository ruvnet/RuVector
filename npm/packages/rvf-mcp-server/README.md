<p align="center"><img src="https://raw.githubusercontent.com/ruvnet/RuVector/main/assets/ruvector/packages/rvf-mcp-server.svg" alt="@ruvector/rvf-mcp-server: MCP server for RuVector Format vector stores — stdio and SSE transports for AI agents." width="100%"></p>

# @ruvector/rvf-mcp-server

MCP (Model Context Protocol) server for RuVector Format (RVF) vector stores. Exposes RVF capabilities to AI agents like Claude Code, Cursor, and other MCP-compatible tools.

## Install

```bash
npx @ruvector/rvf-mcp-server --transport stdio
```

## Claude Code Integration

Add to your MCP config:

```json
{
  "mcpServers": {
    "rvf": {
      "command": "npx",
      "args": ["@ruvector/rvf-mcp-server", "--transport", "stdio"]
    }
  }
}
```

## Transports

```bash
# stdio (for Claude Code, Cursor, etc.)
npx @ruvector/rvf-mcp-server --transport stdio

# SSE (for web clients)
npx @ruvector/rvf-mcp-server --transport sse --port 3100
```

The SSE transport binds `127.0.0.1` by default and serves one session at a time. It has no authentication; use `--host <address>` to bind elsewhere only behind your own auth proxy. Point clients at `http://127.0.0.1:3100/sse` (on dual-stack hosts `localhost` may resolve to `::1`; use `--host ::1` if your client needs it).

## Storage

Stores are real `.rvf` files written through [`@ruvector/rvf`](https://www.npmjs.com/package/@ruvector/rvf), so they survive server restarts and queries go through the native HNSW index instead of a linear scan. Every tool `path` is resolved inside the data directory: `--data-dir <dir>`, else `$RVF_MCP_DATA_DIR`, else `~/.rvf-mcp-server`. Absolute paths, `..` segments and symlinks that lead outside it are rejected. Per-vector metadata, which the native store does not hold, lives next to each store in `<store>.mcp-meta.jsonl`. Metadata filters are applied to an over-fetched index result.

## MCP Tools

| Tool | Description |
|------|-------------|
| `rvf_create_store` | Create a new RVF vector store |
| `rvf_open_store` | Open an existing store |
| `rvf_close_store` | Close and release writer lock |
| `rvf_ingest` | Insert vectors with optional metadata |
| `rvf_query` | k-NN similarity search with filters |
| `rvf_delete` | Delete vectors by ID |
| `rvf_delete_filter` | Delete vectors matching a filter |
| `rvf_compact` | Compact store to reclaim space |
| `rvf_status` | Get store status |
| `rvf_list_stores` | List all open stores |

## MCP Resources

| URI | Description |
|-----|-------------|
| `rvf://stores` | JSON listing of all open stores |

## MCP Prompts

| Prompt | Description |
|--------|-------------|
| `rvf-search` | Natural language similarity search |
| `rvf-ingest` | Data ingestion with auto-embedding |

## Requirements

- Node.js >= 18.0.0

## License

MIT
