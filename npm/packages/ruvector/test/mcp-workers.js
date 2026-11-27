#!/usr/bin/env node
/**
 * Exercise the installed worker CLI through the public MCP stdio launcher.
 * PATH fixtures make the pinned npx banner and the installed CLI deterministic.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

const CLI = path.join(__dirname, '..', 'bin', 'cli.js');
const BANNER = 'Agentic Flow v3.0.0-alpha.2\n\nUSAGE:\n  npx agentic-flow [COMMAND] [OPTIONS]';

async function main() {
  if (process.platform === 'win32') {
    console.log('SKIP  MCP worker CLI fixtures require Unix executable scripts');
    return;
  }

  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-mcp-workers-'));
  const workerCli = path.join(dir, 'agentic-flow');
  const npxMarker = path.join(dir, 'used-npx');
  const client = new Client({ name: 'workers-test', version: '1.0.0' });
  let failures = 0;

  async function check(name, fn) {
    try {
      await fn();
      console.log(`  PASS  ${name}`);
    } catch (error) {
      failures++;
      console.error(`  FAIL  ${name}\n        ${error.message}`);
    }
  }

  async function call(name, args = {}) {
    const result = await client.callTool({ name, arguments: args }, undefined, { timeout: 15_000 });
    return { result, payload: JSON.parse(result.content[0].text) };
  }

  function assertToolError(response, errorPattern) {
    assert.strictEqual(response.payload.success, false, JSON.stringify(response.payload));
    assert.strictEqual(response.result.isError, true);
    assert.match(response.payload.error, errorPattern);
  }

  try {
    fs.symlinkSync(process.execPath, path.join(dir, 'node'));
    fs.writeFileSync(path.join(dir, 'npx'), `#!/usr/bin/env node
const assert = require('assert');
require('fs').writeFileSync(${JSON.stringify(npxMarker)}, '');
assert.strictEqual(process.argv[2], 'agentic-flow@3.0.0-alpha.2');
if (process.argv[5] === 'process error') {
  console.error('worker process failed');
  process.exit(7);
}
console.log(${JSON.stringify(BANNER)});
`, { mode: 0o755 });
    fs.writeFileSync(workerCli, `#!/usr/bin/env node
const assert = require('assert');
const [group, command, ...args] = process.argv.slice(2);
assert.strictEqual(group, 'workers');
if (command === 'dispatch') {
  const prompt = args[0] === '--' ? args[1] : args[0];
  assert.strictEqual(args.length, args[0] === '--' ? 2 : 1);
  if (prompt === 'unsupported') {
    console.log(${JSON.stringify(BANNER)});
  } else if (args[0] === '--help') {
    console.log('Usage: workers dispatch [options] <prompt>');
  } else if (prompt === 'process error') {
    console.error('worker process failed');
    process.exit(7);
  } else if (prompt === 'probe') {
    console.log('No triggers detected in prompt');
  } else {
    console.log('Worker queued: ' + prompt);
  }
} else if (command === 'status') {
  assert.deepStrictEqual(args, ['worker-7']);
  console.log('Worker worker-7 is ready');
} else if (command === 'results') {
  assert.ok(args.length === 0 || (args.length === 1 && args[0] === '--json'));
  console.log(args.length ? '[{"id":"worker-7"}]' : 'Result for worker-7');
} else {
  throw new Error('Unexpected command: ' + command);
}
`, { mode: 0o755 });

    await client.connect(new StdioClientTransport({
      command: process.execPath,
      args: [CLI, 'mcp', 'start'],
      cwd: dir,
      env: {
        PATH: dir,
        RUVECTOR_MCP_ALLOW: 'workers_dispatch,workers_status,workers_results',
        RUVECTOR_MCP_DENY: '',
        RUVECTOR_MCP_PROFILE: '',
      },
      stderr: 'ignore',
    }));
    const { tools } = await client.listTools();
    assert.ok(tools.some((tool) => tool.name === 'workers_dispatch'));

    await check('workers_dispatch rejects an exit-zero usage banner', async () => {
      assertToolError(await call('workers_dispatch', { prompt: 'unsupported' }), /agentic-flow.*usage/i);
    });

    await check('workers_dispatch uses the installed CLI and passes the prompt intact', async () => {
      const prompt = 'audit "function names" & docs';
      const { payload } = await call('workers_dispatch', { prompt });
      assert.deepStrictEqual(payload, {
        success: true,
        message: 'Worker dispatched',
        output: `Worker queued: ${prompt}`,
      });
      assert.strictEqual(fs.existsSync(npxMarker), false, 'worker calls must not invoke npx');
    });

    await check('workers_dispatch treats option-like prompts as text and reports no-trigger output', async () => {
      const literal = await call('workers_dispatch', { prompt: '--help' });
      assert.strictEqual(literal.payload.output, 'Worker queued: --help');
      const { payload } = await call('workers_dispatch', { prompt: 'probe' });
      assert.strictEqual(payload.success, true);
      assert.strictEqual(payload.message, 'No triggers detected in prompt');
      assert.strictEqual(payload.output, 'No triggers detected in prompt');
    });

    await check('workers_dispatch reports a nonzero child exit as a tool error', async () => {
      assertToolError(await call('workers_dispatch', { prompt: 'process error' }), /worker process failed/i);
    });

    await check('workers_status and workers_results use the installed CLI and preserve output formats', async () => {
      assert.deepStrictEqual((await call('workers_status', { workerId: 'worker-7' })).payload, {
        success: true, status: 'Worker worker-7 is ready',
      });
      assert.deepStrictEqual((await call('workers_results')).payload, {
        success: true, results: 'Result for worker-7',
      });
      assert.deepStrictEqual((await call('workers_results', { json: true })).payload, {
        success: true, results: [{ id: 'worker-7' }],
      });
    });

    fs.writeFileSync(workerCli, `#!/usr/bin/env node
console.log(${JSON.stringify('\x1b[1mUsage: workers [options] [command]\x1b[0m')});
`);
    await check('progress tools reject usage even when JSON results are requested', async () => {
      assertToolError(await call('workers_status'), /agentic-flow.*usage/i);
      assertToolError(await call('workers_results', { json: true }), /agentic-flow.*usage/i);
    });

    fs.unlinkSync(workerCli);
    await check('worker tools name the missing agentic-flow executable', async () => {
      for (const name of ['workers_dispatch', 'workers_status', 'workers_results']) {
        assertToolError(await call(name, { prompt: 'probe' }), /agentic-flow.*not found.*PATH/i);
      }
    });
  } finally {
    await client.close();
    fs.rmSync(dir, { recursive: true, force: true });
  }

  if (failures) process.exitCode = 1;
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
