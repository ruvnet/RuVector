#!/usr/bin/env node

// Regression for #1026: MCP must own its VectorDb path and report lock failures.
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { spawn } = require('child_process');

const serverPath = path.join(__dirname, '..', 'bin', 'mcp-server.js');

function startServer(cwd, env = {}) {
  const environment = { ...process.env, HOME: path.join(cwd, 'home'), RUVECTOR_EMBEDDER: 'hash', NO_COLOR: '1' };
  delete environment.RUVECTOR_STORAGE_PATH;
  Object.assign(environment, env);
  const child = spawn(process.execPath, [serverPath], {
    cwd,
    env: environment,
    stdio: ['pipe', 'pipe', 'pipe'],
  });
  let buffer = '';
  let stderr = '';
  let nextId = 1;
  const pending = new Map();

  child.stdout.on('data', (chunk) => {
    buffer += chunk.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop();
    for (const line of lines) {
      let message;
      try { message = JSON.parse(line); } catch { continue; }
      const request = pending.get(message.id);
      if (request) {
        pending.delete(message.id);
        clearTimeout(request.timer);
        request.resolve(message);
      }
    }
  });
  child.stderr.on('data', (chunk) => { stderr += chunk.toString(); });
  child.on('exit', (code) => {
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(new Error(`MCP server exited (${code}): ${stderr}`));
    }
    pending.clear();
  });
  child.on('error', (error) => {
    for (const request of pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    pending.clear();
  });

  function request(method, params) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      const timer = setTimeout(() => {
        pending.delete(id);
        reject(new Error(`Timed out waiting for ${method}: ${stderr}`));
      }, 20_000);
      pending.set(id, { resolve, reject, timer });
      child.stdin.write(`${JSON.stringify({ jsonrpc: '2.0', id, method, params })}\n`);
    });
  }

  async function stop() {
    if (child.exitCode !== null || child.signalCode !== null) return;
    const exited = new Promise((resolve) => child.once('exit', resolve));
    child.stdin.end();
    child.kill('SIGTERM');
    await exited;
  }

  return { request, stop, stderr: () => stderr };
}

async function capabilities(server) {
  const init = await server.request('initialize', {
    protocolVersion: '2024-11-05',
    capabilities: {},
    clientInfo: { name: 'ruvector-vector-store-test', version: '1.0.0' },
  });
  assert.ok(init.result, `initialize failed: ${JSON.stringify(init)}`);
  const reply = await server.request('tools/call', { name: 'hooks_capabilities', arguments: {} });
  assert.ok(reply.result, `hooks_capabilities failed: ${JSON.stringify(reply)}`);
  return JSON.parse(reply.result.content[0].text).capabilities;
}

async function main() {
  const project = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-mcp-store-'));
  const first = startServer(project);
  let second;
  let override;
  try {
    const firstCapabilities = await capabilities(first);
    assert.strictEqual(firstCapabilities.vectorDb, true, 'first server should have VectorDb');
    const remembered = await first.request('tools/call', {
      name: 'hooks_remember',
      arguments: { content: 'storage path regression', type: 'test' },
    });
    assert.ok(remembered.result, `hooks_remember failed: ${JSON.stringify(remembered)}`);

    second = startServer(project);
    const secondCapabilities = await capabilities(second);

    const projectRootDb = path.join(project, 'ruvector.db');
    const ownedDb = path.join(project, '.ruvector', 'vectors.db');
    const errors = [];
    if (fs.existsSync(projectRootDb)) errors.push('server wrote ruvector.db into project root');
    if (!fs.existsSync(ownedDb)) errors.push('server-owned .ruvector/vectors.db was not created');
    if (secondCapabilities.vectorDb !== false) errors.push('second server claimed VectorDb after lock failure');
    if (!/Cannot acquire lock/.test(second.stderr())) errors.push('second server did not report the lock failure on stderr');
    if (fs.existsSync(ownedDb) && !fs.readFileSync(ownedDb).includes(Buffer.from('"hnsw_config":{'))) {
      errors.push('server-owned VectorDb did not persist HNSW configuration');
    }

    const otherProject = path.join(project, 'other-project');
    fs.mkdirSync(otherProject);
    const selectedDb = path.join(project, 'custom', 'selected.db');
    override = startServer(otherProject, { RUVECTOR_STORAGE_PATH: selectedDb });
    const overrideCapabilities = await capabilities(override);
    if (overrideCapabilities.vectorDb !== true || !fs.existsSync(selectedDb)) {
      errors.push('RUVECTOR_STORAGE_PATH did not select a working VectorDb path');
    }
    if (fs.existsSync(path.join(otherProject, 'ruvector.db'))) {
      errors.push('override server wrote ruvector.db into project root');
    }

    assert.deepStrictEqual(errors, []);
  } finally {
    await Promise.all([override?.stop(), second?.stop(), first.stop()]);
    fs.rmSync(project, { recursive: true, force: true });
  }
  console.log('MCP VectorDb path, lock reporting, and HNSW checks passed (#1026)');
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
