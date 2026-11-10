/**
 * Transport factory functions for stdio and SSE modes.
 */

import { RvfMcpServer, type RvfMcpServerOptions } from './server.js';

/**
 * Create and start an RVF MCP server over stdio transport.
 *
 * Usage in .mcp.json:
 * ```json
 * {
 *   "mcpServers": {
 *     "rvf": {
 *       "command": "node",
 *       "args": ["dist/cli.js", "--transport", "stdio"]
 *     }
 *   }
 * }
 * ```
 */
export async function createStdioServer(
  options?: RvfMcpServerOptions,
): Promise<RvfMcpServer> {
  const { StdioServerTransport } = await import(
    '@modelcontextprotocol/sdk/server/stdio.js'
  );

  const server = new RvfMcpServer(options);
  const transport = new StdioServerTransport();
  await server.connect(transport);
  return server;
}

/** Options for the SSE transport. */
export interface SseServerOptions {
  /**
   * Interface to bind. Default: '127.0.0.1' (loopback only). The SSE
   * endpoint has no authentication, so binding a non-loopback address
   * exposes every tool to anyone who can reach the port.
   */
  host?: string;
}

const LOOPBACK_HOSTS = new Set(['127.0.0.1', 'localhost', '::1']);

/**
 * Create and start an RVF MCP server over SSE transport.
 *
 * Starts an Express HTTP server with SSE endpoint at `/sse`
 * and message endpoint at `/messages`. One SSE session is served at a
 * time; POSTs are routed by the `sessionId` the server hands out.
 *
 * @param port    HTTP port. Default: 3100.
 * @param options Server options.
 * @param sseOptions Transport options (bind host).
 */
export async function createSseServer(
  port = 3100,
  options?: RvfMcpServerOptions,
  sseOptions?: SseServerOptions,
): Promise<RvfMcpServer> {
  const { SSEServerTransport } = await import(
    '@modelcontextprotocol/sdk/server/sse.js'
  );
  const express = (await import('express')).default;

  const host = sseOptions?.host ?? '127.0.0.1';
  const loopback = LOOPBACK_HOSTS.has(host);
  // DNS-rebinding protection: on loopback, only accept our own Host header.
  const hostLabel = host.includes(':') ? `[${host}]` : host;
  const transportOptions = loopback
    ? {
        enableDnsRebindingProtection: true,
        allowedHosts: ['127.0.0.1', 'localhost', '[::1]'].map((h) => `${h}:${port}`),
      }
    : undefined;

  const app = express();
  const server = new RvfMcpServer(options);

  let sseTransport: InstanceType<typeof SSEServerTransport> | null = null;

  // SSE endpoint — client connects here
  app.get('/sse', async (req, res) => {
    // The SDK checks Host only on POST; check it here too so a rebound page
    // cannot occupy the single session slot.
    if (transportOptions && !transportOptions.allowedHosts.includes(req.headers.host ?? '')) {
      res.status(403).json({ error: 'Invalid Host header' });
      return;
    }
    if (sseTransport) {
      // A second connect would otherwise replace the live session and
      // leave the first client (and the server) wired to a dead transport.
      res.status(409).json({ error: 'An SSE session is already active' });
      return;
    }
    const transport = new SSEServerTransport('/messages', res, transportOptions);
    sseTransport = transport;
    res.on('close', () => {
      if (sseTransport === transport) sseTransport = null;
    });
    try {
      await server.connect(transport);
    } catch (err) {
      if (sseTransport === transport) sseTransport = null;
      console.error('SSE connection error:', err instanceof Error ? err.message : err);
      if (!res.headersSent) res.status(500).end();
    }
  });

  // Message endpoint — client sends JSON-RPC here
  app.post('/messages', async (req, res) => {
    const transport = sseTransport;
    if (!transport || req.query.sessionId !== transport.sessionId) {
      res.status(404).json({ error: 'Unknown or expired session' });
      return;
    }
    try {
      // Throws (after writing a 500) if the stream closed meanwhile; an
      // unhandled rejection here would terminate the process.
      await transport.handlePostMessage(req, res);
    } catch (err) {
      if (!res.headersSent) res.status(500).end();
    }
  });

  // Health check
  app.get('/health', (_req, res) => {
    res.json({
      status: 'ok',
      server: options?.name ?? 'rvf-mcp-server',
      stores: server.storeCount,
    });
  });

  await new Promise<void>((resolve, reject) => {
    const http = app.listen(port, host, () => resolve());
    http.once('error', reject);
  });
  const base = `http://${hostLabel}:${port}`;
  console.error(`RVF MCP Server (SSE) listening on ${base}`);
  console.error(`  SSE endpoint:     ${base}/sse`);
  console.error(`  Message endpoint: ${base}/messages`);
  console.error(`  Health check:     ${base}/health`);
  if (!loopback) {
    console.error('  WARNING: bound to a non-loopback address with no authentication.');
  }

  return server;
}

/**
 * Create a server with the specified transport type.
 */
export async function createServer(
  transport: 'stdio' | 'sse' = 'stdio',
  port = 3100,
  options?: RvfMcpServerOptions,
  sseOptions?: SseServerOptions,
): Promise<RvfMcpServer> {
  if (transport === 'sse') {
    return createSseServer(port, options, sseOptions);
  }
  return createStdioServer(options);
}
