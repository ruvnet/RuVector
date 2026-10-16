import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RuGraph, RuGraphError } from '../index.mjs';

function child(source, timeoutMs = 1000) {
  return new RuGraph({ binary: process.execPath, timeoutMs }).invoke(['-e', source], '');
}

test('unsafe integers are rejected on input before process execution', () => {
  const client = new RuGraph({ binary: '/no/such/binary' });
  assert.throws(() => client.run({ now: Number.MAX_SAFE_INTEGER + 1 }), error => error.code === 'UNSAFE_INTEGER');
  assert.throws(() => client.run({ now: 1n }), error => error instanceof RuGraphError && error.code === 'INVALID_INPUT');
  const cycle = {}; cycle.self = cycle;
  assert.throws(() => client.run(cycle), error => error instanceof RuGraphError && error.code === 'INVALID_INPUT');
});

test('Rust u64 values cannot silently round during real compilation', async () => {
  const client = new RuGraph();
  const source = 'program large { effect graph.read; effect decision.emit; budget 10000; let n = nodes "Service"; let e = observe n "Failing" within 18446744073709551615; let d = assess e; emit d; }';
  await assert.rejects(client.compile(source), error => error.code === 'UNSAFE_INTEGER');
});

test('combined stdout and stderr are bounded while the child runs', async () => {
  await assert.rejects(child('const b = Buffer.alloc(9*1024*1024, 65); process.stdout.write(b); process.stderr.write(b);', 5000), error => error.code === 'OUTPUT_LIMIT');
});

test('process timeout and spawn failure return structured errors', async () => {
  await assert.rejects(child('setInterval(() => {}, 1000)', 30), error => error.code === 'TIMEOUT');
  await assert.rejects(new RuGraph({ binary: '/no/such/binary' }).check(''), error => error.code === 'EXECUTABLE_ERROR');
});

test('malformed error JSON and invalid output encoding fail closed', async () => {
  for (const data of ['null', '[]', '{"code":1,"message":2}', '{"code":"X","message":"bad","position":-1}']) {
    await assert.rejects(child(`process.stderr.write(${JSON.stringify(data)}); process.exitCode = 2;`), error => error.code === 'PROCESS_ERROR');
  }
  await assert.rejects(child('process.stdout.write(Buffer.from([255]));'), error => error.code === 'INVALID_OUTPUT');
  await assert.rejects(child('process.stdout.write("[]");'), error => error.code === 'INVALID_OUTPUT');
  await assert.rejects(child('process.stdout.write("{\\"x\\":1e999}");'), error => error.code === 'INVALID_OUTPUT');
});

test('source and model configuration reject invalid types and NUL', async () => {
  const client = new RuGraph();
  await assert.rejects(client.check({ source: 'x' }), error => error.code === 'INVALID_INPUT');
  assert.throws(() => new RuGraph({ binary: 'bad\0path' }), error => error.code === 'INVALID_INPUT');
  assert.throws(() => client.run({}, { modelsPath: 'bad\0path' }), error => error.code === 'INVALID_INPUT');
});
