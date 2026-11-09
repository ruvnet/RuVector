#!/usr/bin/env node
/**
 * RVF MCP Server CLI — start the server in stdio or SSE mode.
 *
 * Usage:
 *   rvf-mcp-server                          # stdio (default)
 *   rvf-mcp-server --transport stdio        # stdio explicitly
 *   rvf-mcp-server --transport sse          # SSE on port 3100
 *   rvf-mcp-server --transport sse --port 8080
 *   rvf-mcp-server --data-dir ./stores      # where .rvf stores live
 */

import { createRequire } from 'node:module';
import { createServer } from './transports.js';

const { version } = createRequire(import.meta.url)('../package.json') as { version: string };

function parseArgs(): { transport: 'stdio' | 'sse'; port: number; host?: string; dataDir?: string } {
  const args = process.argv.slice(2);
  let transport: 'stdio' | 'sse' = 'stdio';
  let port = 3100;
  let host: string | undefined;
  let dataDir: string | undefined;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--transport' || args[i] === '-t') {
      const val = args[++i];
      if (val === 'sse' || val === 'stdio') {
        transport = val;
      } else {
        console.error(`Unknown transport: ${val}. Use 'stdio' or 'sse'.`);
        process.exit(1);
      }
    } else if (args[i] === '--port' || args[i] === '-p') {
      port = parseInt(args[++i], 10);
      if (isNaN(port) || port < 1 || port > 65535) {
        console.error('Port must be between 1 and 65535');
        process.exit(1);
      }
    } else if (args[i] === '--host' || args[i] === '-H') {
      host = args[++i];
      if (!host) {
        console.error('--host requires an address');
        process.exit(1);
      }
    } else if (args[i] === '--data-dir' || args[i] === '-d') {
      dataDir = args[++i];
      if (!dataDir) {
        console.error('--data-dir requires a directory');
        process.exit(1);
      }
    } else if (args[i] === '--help' || args[i] === '-h') {
      console.log(`
RVF MCP Server — Model Context Protocol server for RuVector Format

Usage:
  rvf-mcp-server [options]

Options:
  -t, --transport <stdio|sse>  Transport mode (default: stdio)
  -p, --port <number>          SSE port (default: 3100)
  -H, --host <address>         SSE bind address (default: 127.0.0.1; the SSE
                               endpoint is unauthenticated, so only bind a
                               public address behind your own auth proxy)
  -d, --data-dir <dir>         Directory holding the .rvf stores (default:
                               $RVF_MCP_DATA_DIR, else ~/.rvf-mcp-server).
                               Tool paths are resolved inside it and may
                               not escape it.
  -h, --help                   Show this help message

MCP Tools:
  rvf_create_store   Create a new vector store
  rvf_open_store     Open an existing store
  rvf_close_store    Close a store
  rvf_ingest         Insert vectors
  rvf_query          k-NN similarity search
  rvf_delete         Delete vectors by ID
  rvf_delete_filter  Delete by metadata filter
  rvf_compact        Reclaim dead space
  rvf_status         Store status
  rvf_list_stores    List open stores

stdio config (.mcp.json):
  {
    "mcpServers": {
      "rvf": {
        "command": "npx",
        "args": ["-y", "@ruvector/rvf-mcp-server", "--data-dir", "/abs/path/to/stores"]
      }
    }
  }
`);
      process.exit(0);
    }
  }

  return { transport, port, host, dataDir };
}

async function main(): Promise<void> {
  const { transport, port, host, dataDir } = parseArgs();

  if (transport === 'stdio') {
    // Suppress stdout logging in stdio mode (MCP uses stdout)
    console.error('RVF MCP Server starting (stdio transport)...');
  }

  const server = await createServer(transport, port, { version, ...(dataDir ? { dataDir } : {}) }, { host });

  // Close stores (flushes id maps and metadata, releases locks) on shutdown.
  const shutdown = (message: string) => {
    console.error(message);
    server.close().catch((err) => console.error('Close failed:', err)).finally(() => process.exit(0));
  };
  process.on('SIGINT', () => shutdown('\nRVF MCP Server shutting down...'));
  process.on('SIGTERM', () => shutdown('RVF MCP Server terminated.'));
}

main().catch((err) => {
  console.error('Fatal:', err);
  process.exit(1);
});
