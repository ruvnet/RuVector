/**
 * π Brain MCP Server
 *
 * Proxies MCP tool calls to the π REST API at pi.ruv.io
 */

import { PiBrainClient } from './client.js';

const client = new PiBrainClient();

const TOOLS = [
  {
    name: 'brain_share',
    description: 'Share a learning with the π collective intelligence',
    inputSchema: {
      type: 'object' as const,
      properties: {
        category: {
          type: 'string',
          description:
            'Category: architecture, pattern, solution, convention, security, performance, tooling, debug',
        },
        title: { type: 'string', description: 'Title of the knowledge' },
        content: { type: 'string', description: 'Content body' },
        tags: {
          type: 'array',
          items: { type: 'string' },
          description: 'Tags for categorization',
        },
        code_snippet: {
          type: 'string',
          description: 'Optional code snippet',
        },
      },
      required: ['category', 'title', 'content'],
    },
  },
  {
    name: 'brain_search',
    description: 'Semantic search across shared knowledge in π',
    inputSchema: {
      type: 'object' as const,
      properties: {
        query: { type: 'string', description: 'Search query' },
        category: { type: 'string', description: 'Filter by category' },
        tags: {
          type: 'string',
          description: 'Filter by tags (comma-separated)',
        },
        limit: { type: 'number', description: 'Max results' },
      },
      required: ['query'],
    },
  },
  {
    name: 'brain_get',
    description: 'Get a specific memory from π by ID',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: { type: 'string', description: 'Memory ID' },
      },
      required: ['id'],
    },
  },
  {
    name: 'brain_list',
    description: 'List memories in π',
    inputSchema: {
      type: 'object' as const,
      properties: {
        category: { type: 'string', description: 'Filter by category' },
        limit: { type: 'number', description: 'Max results' },
      },
    },
  },
  {
    name: 'brain_vote',
    description: 'Vote on a memory quality (Bayesian update)',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: { type: 'string', description: 'Memory ID' },
        direction: {
          type: 'string',
          enum: ['up', 'down'],
          description: 'Vote direction',
        },
      },
      required: ['id', 'direction'],
    },
  },
  {
    name: 'brain_delete',
    description: 'Delete a memory from π',
    inputSchema: {
      type: 'object' as const,
      properties: {
        id: { type: 'string', description: 'Memory ID' },
      },
      required: ['id'],
    },
  },
  {
    name: 'brain_transfer',
    description: 'Transfer learning priors between domains',
    inputSchema: {
      type: 'object' as const,
      properties: {
        source_domain: { type: 'string', description: 'Source domain' },
        target_domain: { type: 'string', description: 'Target domain' },
      },
      required: ['source_domain', 'target_domain'],
    },
  },
  {
    name: 'brain_drift',
    description: 'Check knowledge drift in π',
    inputSchema: {
      type: 'object' as const,
      properties: {
        domain: { type: 'string', description: 'Domain to check' },
      },
    },
  },
  {
    name: 'brain_partition',
    description: 'View knowledge topology via MinCut partitioning',
    inputSchema: {
      type: 'object' as const,
      properties: {
        domain: { type: 'string', description: 'Domain to partition' },
      },
    },
  },
  {
    name: 'brain_status',
    description: 'Get π system status',
    inputSchema: {
      type: 'object' as const,
      properties: {},
    },
  },
];

async function handleToolCall(
  name: string,
  args: Record<string, unknown>,
): Promise<unknown> {
  switch (name) {
    case 'brain_share':
      return client.share({
        category: args.category as string,
        title: args.title as string,
        content: args.content as string,
        tags: (args.tags as string[]) ?? [],
        code_snippet: args.code_snippet as string | undefined,
      });
    case 'brain_search':
      return client.search({
        query: args.query as string,
        category: args.category as string | undefined,
        tags: args.tags as string | undefined,
        limit: args.limit as number | undefined,
      });
    case 'brain_get':
      return client.get(args.id as string);
    case 'brain_list':
      return client.list(
        args.category as string | undefined,
        args.limit as number | undefined,
      );
    case 'brain_vote':
      return client.vote(args.id as string, args.direction as 'up' | 'down');
    case 'brain_delete':
      return client.delete(args.id as string);
    case 'brain_transfer':
      return client.transfer(
        args.source_domain as string,
        args.target_domain as string,
      );
    case 'brain_drift':
      return client.drift(args.domain as string | undefined);
    case 'brain_partition':
      return client.partition(args.domain as string | undefined);
    case 'brain_status':
      return client.status();
    default:
      throw new Error(`Unknown tool: ${name}`);
  }
}

export async function startMcpServer(
  transport: 'stdio' | 'sse' = 'stdio',
  port = 3100,
) {
  // Use raw JSON-RPC over stdio/SSE (no SDK dependency needed for simple protocol)
  if (transport === 'stdio') {
    const readline = await import('readline');
    const rl = readline.createInterface({ input: process.stdin });

    rl.on('line', async (line: string) => {
      try {
        const req = JSON.parse(line);
        const res = await handleJsonRpc(req);
        if (res) {
          process.stdout.write(JSON.stringify(res) + '\n');
        }
      } catch (e) {
        const err = {
          jsonrpc: '2.0',
          id: null,
          error: {
            code: -32700,
            message: `Parse error: ${(e as Error).message}`,
          },
        };
        process.stdout.write(JSON.stringify(err) + '\n');
      }
    });

    console.error('π Brain MCP Server started (stdio)');
  } else {
    // SSE mode - point to hosted SSE on pi.ruv.io
    console.error(
      `π Brain MCP Server — use hosted SSE at https://pi.ruv.io/sse`,
    );
    console.error(
      `Or connect via: claude mcp add π --url https://pi.ruv.io/sse`,
    );
  }

  // Keep alive
  await new Promise(() => {});
}

async function handleJsonRpc(req: {
  jsonrpc: string;
  id: unknown;
  method: string;
  params?: unknown;
}) {
  switch (req.method) {
    case 'initialize':
      return {
        jsonrpc: '2.0',
        id: req.id,
        result: {
          protocolVersion: '2024-11-05',
          capabilities: { tools: { listChanged: false } },
          serverInfo: { name: 'pi-brain', version: '0.1.0' },
        },
      };
    case 'initialized':
      return { jsonrpc: '2.0', id: req.id, result: {} };
    case 'tools/list':
      return { jsonrpc: '2.0', id: req.id, result: { tools: TOOLS } };
    case 'tools/call': {
      const params = req.params as {
        name: string;
        arguments: Record<string, unknown>;
      };
      try {
        const result = await handleToolCall(
          params.name,
          params.arguments ?? {},
        );
        return {
          jsonrpc: '2.0',
          id: req.id,
          result: {
            content: [
              { type: 'text', text: JSON.stringify(result, null, 2) },
            ],
          },
        };
      } catch (e) {
        return {
          jsonrpc: '2.0',
          id: req.id,
          result: {
            content: [
              { type: 'text', text: `Error: ${(e as Error).message}` },
            ],
            isError: true,
          },
        };
      }
    }
    case 'shutdown':
      return { jsonrpc: '2.0', id: req.id, result: {} };
    default:
      return {
        jsonrpc: '2.0',
        id: req.id,
        error: {
          code: -32601,
          message: `Method not found: ${req.method}`,
        },
      };
  }
}
