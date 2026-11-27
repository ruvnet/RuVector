#!/usr/bin/env node

/**
 * Regression coverage for issues #710 and #715.
 *
 * The CLI launcher must explicitly start the required MCP module, and stdout
 * must remain exclusively JSON-RPC while the stdio transport is active.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');
const { Client } = require('@modelcontextprotocol/sdk/client/index.js');
const { StdioClientTransport } = require('@modelcontextprotocol/sdk/client/stdio.js');

const PACKAGE_ROOT = path.join(__dirname, '..');
const CLI = path.join(PACKAGE_ROOT, 'bin', 'cli.js');
const MCP_SERVER = path.join(PACKAGE_ROOT, 'bin', 'mcp-server.js');
const LOADER_PATHS = [
  path.join(PACKAGE_ROOT, 'src', 'core', 'onnx', 'loader.js'),
  path.join(PACKAGE_ROOT, 'src', 'core', 'onnx', 'pkg', 'loader.js'),
];

function waitForInitialize(child, timeoutMs = 10_000) {
  return new Promise((resolve, reject) => {
    let stdout = '';
    let stderr = '';
    const timer = setTimeout(() => {
      reject(new Error(`Timed out waiting for MCP initialize response.\nstdout: ${stdout}\nstderr: ${stderr}`));
    }, timeoutMs);

    child.stdout.on('data', (chunk) => {
      stdout += chunk.toString();
      for (const line of stdout.split(/\r?\n/).filter(Boolean)) {
        try {
          const message = JSON.parse(line);
          if (message.id === 1) {
            clearTimeout(timer);
            resolve({ message, stdout, stderr });
            return;
          }
        } catch {
          // Preserve the output; the stdout purity assertion below reports it.
        }
      }
    });
    child.stderr.on('data', (chunk) => {
      stderr += chunk.toString();
    });
    child.once('error', (error) => {
      clearTimeout(timer);
      reject(error);
    });
    child.once('exit', (code, signal) => {
      clearTimeout(timer);
      reject(new Error(`MCP process exited before initialize: code=${code} signal=${signal}\nstderr: ${stderr}`));
    });
  });
}

async function testVectorSearchThreshold() {
  const cwd = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-mcp-search-'));
  const client = new Client({ name: 'ruvector-mcp-search-test', version: '1.0.0' });
  const transport = new StdioClientTransport({
    command: process.execPath,
    args: [CLI, 'mcp', 'start'],
    cwd,
    env: { ...process.env, HOME: cwd, NO_COLOR: '1' },
    stderr: 'pipe',
  });
  transport.stderr.resume();

  try {
    await client.connect(transport);
    const call = async (name, args) => JSON.parse(
      (await client.callTool({ name, arguments: { path: './vectors.db', ...args } })).content[0].text,
    );
    const created = await call('vectordb_create', { dimension: 2 });
    assert.strictEqual(created.success, true, JSON.stringify(created));
    const inserted = await call('vectordb_insert', { entries: [
      { id: 'same', vector: [1, 0], metadata: { group: 'a' } },
      { id: 'opposite', vector: [-1, 0], metadata: { group: 'b' } },
    ] });
    assert.strictEqual(inserted.success, true, JSON.stringify(inserted));

    const query = { vector: [1, 0], k: 2 };
    const all = await call('vectordb_search', query);
    assert.strictEqual(all.success, true, JSON.stringify(all));
    assert.deepStrictEqual(all.results.map(row => row.id), ['same', 'opposite']);
    const filtered = await call('vectordb_search', { ...query, threshold: 1 });
    assert.deepStrictEqual(filtered, { success: true, results: [all.results[0]] },
      `vectordb_search threshold 1 returned ${JSON.stringify(filtered)}`);
    assert.deepStrictEqual(
      await call('vectordb_search', { ...query, threshold: 0, filter: { group: 'b' } }),
      { success: true, results: [] },
    );
    const invalid = await call('vectordb_search', { ...query, threshold: '1' });
    assert.strictEqual(invalid.success, false);
    assert.match(invalid.error, /threshold/i);

    const { tools } = await client.listTools();
    const schema = tools.find(tool => tool.name === 'vectordb_search').inputSchema;
    assert.strictEqual(schema.properties.threshold.type, 'number');
    assert.ok(!schema.required.includes('threshold'));
  } finally {
    await client.close();
    fs.rmSync(cwd, { recursive: true, force: true });
  }
}

async function main() {
  // Inspect the module export without requiring it in this test process.
  // Requiring it initializes the real intelligence worker pool, which is
  // intentionally long-lived and would keep this regression test alive.
  const serverSource = fs.readFileSync(MCP_SERVER, 'utf8');
  assert.match(
    serverSource,
    /module\.exports\s*=\s*\{[^}]*\bmain\b/,
    'mcp-server.js must export main() for the CLI launcher',
  );

  for (const loaderPath of LOADER_PATHS) {
    const source = fs.readFileSync(loaderPath, 'utf8');
    assert.doesNotMatch(
      source,
      /console\.log\((?:`Loading model:|`\s+(?:Disk )?Cache hit:|`\s+Downloading:|'Model cache cleared')/,
      `${path.relative(PACKAGE_ROOT, loaderPath)} must send diagnostics to stderr`,
    );
  }

  const child = spawn(process.execPath, [CLI, 'mcp', 'start'], {
    cwd: PACKAGE_ROOT,
    env: { ...process.env, NO_COLOR: '1' },
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  try {
    child.stdin.write(`${JSON.stringify({
      jsonrpc: '2.0',
      id: 1,
      method: 'initialize',
      params: {
        protocolVersion: '2024-11-05',
        capabilities: {},
        clientInfo: { name: 'ruvector-mcp-regression', version: '1.0.0' },
      },
    })}\n`);

    const { message, stdout } = await waitForInitialize(child);
    assert.ok(message.result, `initialize returned an error: ${JSON.stringify(message)}`);

    const frames = stdout.split(/\r?\n/).filter(Boolean);
    assert.ok(frames.length > 0, 'MCP server must emit an initialize response');
    for (const frame of frames) {
      assert.doesNotThrow(() => JSON.parse(frame), `stdout contained non-JSON-RPC data: ${frame}`);
    }
  } finally {
    child.stdin.end();
    child.kill('SIGTERM');
  }

  console.log('MCP stdio launcher and stdout hygiene checks passed');

  console.log('MCP vectordb_search honors its optional similarity threshold');
  await testVectorSearchThreshold();
  console.log('  PASS  MCP vectordb_search honors its optional similarity threshold');
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
