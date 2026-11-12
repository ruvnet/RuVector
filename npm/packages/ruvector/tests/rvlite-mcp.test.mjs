/**
 * rvlite_sql / rvlite_cypher / rvlite_sparql over MCP stdio, against the real
 * rvlite package (a devDependency, so this never runs against a mock).
 *
 * Regression: the handlers called `new rvlite.Database(...)`, which rvlite
 * does not export (the class is `RvLite`), did not await its async query
 * methods, and stripped ( ) { } ' " from the query as if it were a shell
 * argument, so every call failed or returned an empty object.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const SERVER = path.join(__dirname, '..', 'bin', 'mcp-server.js');

async function mcpSession(cwd) {
  const p = spawn(process.execPath, [SERVER], {
    cwd, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, HOME: cwd },
  });
  let buf = '';
  let nextId = 0;
  const waiting = new Map();
  p.stdout.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      let m;
      try { m = JSON.parse(line); } catch { continue; }
      if (waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
    }
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => reject(new Error(`MCP ${method} timed out`)), 60000);
    waiting.set(id, (m) => { clearTimeout(timer); resolve(m); });
    p.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
  await request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'rvlite-test', version: '1' } });
  p.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
  return {
    async call(name, args) {
      const m = await request('tools/call', { name, arguments: args });
      const text = m.result?.content?.[0]?.text ?? JSON.stringify(m.error);
      return { isError: !!m.result?.isError, body: JSON.parse(text) };
    },
    close() { p.kill(); },
  };
}

test('rvlite is the real package, and it exports RvLite (not Database)', async () => {
  const mod = await import('rvlite');
  assert.equal(typeof mod.RvLite, 'function');
  assert.equal(mod.Database, undefined);
  assert.ok(require.resolve('rvlite').includes(`${path.sep}node_modules${path.sep}rvlite${path.sep}`));
});

test('MCP rvlite_* tools run real queries', async (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-rvlite-mcp-'));
  const mcp = await mcpSession(dir);
  t.after(() => { mcp.close(); fs.rmSync(dir, { recursive: true, force: true }); });

  // Parentheses, quotes and brackets must reach the parser intact.
  const sql = await mcp.call('rvlite_sql', { query: "CREATE TABLE docs (id TEXT, embedding VECTOR(3))" });
  assert.equal(sql.isError, false, JSON.stringify(sql.body));
  assert.equal(sql.body.success, true, JSON.stringify(sql.body));
  assert.equal(sql.body.query_type, 'sql');
  assert.ok(Array.isArray(sql.body.results.rows), JSON.stringify(sql.body));

  const cypher = await mcp.call('rvlite_cypher', { query: "CREATE (n:Person {name: 'Ada'}) RETURN n" });
  assert.equal(cypher.body.success, true, JSON.stringify(cypher.body));
  assert.deepEqual(cypher.body.results.columns, ['n']);
  assert.equal(cypher.body.row_count, 1);

  const sparql = await mcp.call('rvlite_sparql', { query: 'SELECT ?s WHERE { ?s ?p ?o }' });
  assert.equal(sparql.body.success, true, JSON.stringify(sparql.body));
  assert.equal(sparql.body.results.type, 'select');
  assert.deepEqual(sparql.body.results.variables, ['s']);

  // Parser errors come back as tool errors, not as a fake success.
  const bad = await mcp.call('rvlite_sql', { query: 'NOT A STATEMENT' });
  assert.equal(bad.isError, true);
  assert.equal(bad.body.success, false);

  // db_path stays confined to the working directory.
  const escape = await mcp.call('rvlite_sql', { query: 'SELECT * FROM docs', db_path: '/etc/passwd' });
  assert.equal(escape.isError, true);
  assert.match(escape.body.error, /outside the working directory/);

  // An in-root db_path that does not exist yet opens an empty database.
  const fresh = await mcp.call('rvlite_cypher', { query: 'MATCH (n) RETURN n', db_path: 'graph.rvf' });
  assert.equal(fresh.body.success, true, JSON.stringify(fresh.body));
  assert.equal(fs.existsSync(path.join(dir, 'graph.rvf')), false, 'a query must not write the RVF file');
});
