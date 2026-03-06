#!/usr/bin/env node
/**
 * π Brain CLI
 *
 * Usage:
 *   npx @ruvector/pi-brain health
 *   npx @ruvector/pi-brain share --category pattern --title "My Pattern" --content "..."
 *   npx @ruvector/pi-brain search "authentication patterns"
 *   npx @ruvector/pi-brain list --category architecture --limit 10
 *   npx @ruvector/pi-brain status
 *   npx @ruvector/pi-brain mcp                    # Start MCP stdio server
 *   npx @ruvector/pi-brain mcp --transport sse    # Start MCP SSE server
 */

import { PiBrainClient } from './client.js';

const client = new PiBrainClient();

async function main() {
  const args = process.argv.slice(2);
  const command = args[0];

  if (!command || command === '--help' || command === '-h') {
    console.log(`
π Brain — RuVector Shared Intelligence

Usage:
  pi-brain <command> [options]

Commands:
  health                          Check system health
  share                           Share knowledge with the collective
    --category <cat>              Category (architecture, pattern, solution, etc.)
    --title <title>               Title of the memory
    --content <content>           Content body
    --tags <tag1,tag2>            Comma-separated tags
  search <query>                  Semantic search
    --category <cat>              Filter by category
    --limit <n>                   Max results (default: 10)
  get <id>                        Get a memory by ID
  list                            List memories
    --category <cat>              Filter by category
    --limit <n>                   Max results
  vote <id> <up|down>             Vote on a memory
  delete <id>                     Delete a memory
  transfer <source> <target>      Transfer knowledge between domains
  drift [domain]                  Check knowledge drift
  partition [domain]              View knowledge topology
  status                          System status
  mcp                             Start MCP server (stdio)
    --transport <stdio|sse>       Transport mode
    --port <n>                    SSE port (default: 3100)

Environment:
  PI=<key>            Your π identity key
  BRAIN_URL=<url>     Custom backend URL (default: https://pi.ruv.io)
`);
    process.exit(0);
  }

  try {
    switch (command) {
      case 'health':
        console.log(JSON.stringify(await client.health(), null, 2));
        break;

      case 'share': {
        const category = getArg(args, '--category') ?? 'pattern';
        const title = getArg(args, '--title');
        const content = getArg(args, '--content');
        if (!title || !content) {
          console.error('Error: --title and --content are required');
          process.exit(1);
        }
        const tags = getArg(args, '--tags')?.split(',') ?? [];
        console.log(
          JSON.stringify(
            await client.share({ category, title, content, tags }),
            null,
            2,
          ),
        );
        break;
      }

      case 'search': {
        const query = args[1];
        if (!query) {
          console.error('Error: search query required');
          process.exit(1);
        }
        const category = getArg(args, '--category');
        const limit = getArg(args, '--limit');
        console.log(
          JSON.stringify(
            await client.search({
              query,
              category: category ?? undefined,
              limit: limit ? parseInt(limit) : undefined,
            }),
            null,
            2,
          ),
        );
        break;
      }

      case 'get':
        if (!args[1]) {
          console.error('Error: ID required');
          process.exit(1);
        }
        console.log(JSON.stringify(await client.get(args[1]), null, 2));
        break;

      case 'list': {
        const cat = getArg(args, '--category');
        const lim = getArg(args, '--limit');
        console.log(
          JSON.stringify(
            await client.list(
              cat ?? undefined,
              lim ? parseInt(lim) : undefined,
            ),
            null,
            2,
          ),
        );
        break;
      }

      case 'vote':
        if (!args[1] || !args[2]) {
          console.error('Error: ID and direction (up/down) required');
          process.exit(1);
        }
        console.log(
          JSON.stringify(
            await client.vote(args[1], args[2] as 'up' | 'down'),
            null,
            2,
          ),
        );
        break;

      case 'delete':
        if (!args[1]) {
          console.error('Error: ID required');
          process.exit(1);
        }
        console.log(JSON.stringify(await client.delete(args[1]), null, 2));
        break;

      case 'transfer':
        if (!args[1] || !args[2]) {
          console.error('Error: source and target domains required');
          process.exit(1);
        }
        console.log(
          JSON.stringify(await client.transfer(args[1], args[2]), null, 2),
        );
        break;

      case 'drift':
        console.log(JSON.stringify(await client.drift(args[1]), null, 2));
        break;

      case 'partition':
        console.log(JSON.stringify(await client.partition(args[1]), null, 2));
        break;

      case 'status':
        console.log(JSON.stringify(await client.status(), null, 2));
        break;

      case 'mcp': {
        const { startMcpServer } = await import('./mcp.js');
        const transport = (getArg(args, '--transport') ?? 'stdio') as
          | 'stdio'
          | 'sse';
        const port = parseInt(getArg(args, '--port') ?? '3100');
        await startMcpServer(transport, port);
        break;
      }

      default:
        console.error(
          `Unknown command: ${command}. Run pi-brain --help for usage.`,
        );
        process.exit(1);
    }
  } catch (err) {
    console.error(`Error: ${(err as Error).message}`);
    process.exit(1);
  }
}

function getArg(args: string[], flag: string): string | null {
  const i = args.indexOf(flag);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : null;
}

main();
