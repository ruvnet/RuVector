import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { createTypesafe, choice } = require('../dist/index.js');
const { main } = require('../dist/cli/main.js');
const fake = require('./fixtures/fake-binding.cjs');

function fixture(t) {
  const dir = mkdtempSync(join(tmpdir(), 'typesafe-wasm-onnx-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const modelDir = join(dir, 'models');
  mkdirSync(join(modelDir, 'selected'), { recursive: true });
  writeFileSync(join(modelDir, 'selected', 'model.onnx'), 'model bytes');
  writeFileSync(join(modelDir, 'selected', 'tokenizer.json'), 'tokenizer bytes');
  const manifest = join(dir, 'manifest.json');
  const questions = join(dir, 'questions.json');
  writeFileSync(questions, JSON.stringify({ dept: { type: 'choice', criteria: { billing: 'b', fraud: 'f' } } }));
  const selected = {
    name: 'selected', file: 'selected/model.onnx', tokenizer_file: 'selected/tokenizer.json',
    sha256: 'a'.repeat(64), tokenizer_sha256: 'b'.repeat(64), dims: 384,
    license: 'MIT', source_url: 'https://example.com/model', added: '2026-09-01',
    review_by: '2027-09-01', pooling: 'cls', max_tokens: 256,
  };
  writeFileSync(manifest, JSON.stringify({ models: [
    { ...selected, name: 'other', file: 'other/model.onnx' }, selected,
  ] }));
  return { modelDir, manifest, questions, selected };
}

function wasmBinding(calls) {
  class Engine extends fake.Engine {
    constructor(options) {
      if (JSON.parse(options).embedder.kind === 'onnx') throw 'onnx embedder requires Engine.fromBytes';
      super(options);
    }

    static fromBytes(options, model, tokenizer) {
      calls.push({ options: JSON.parse(options), model, tokenizer });
      return new fake.Engine('{"embedder":"hash"}');
    }
  }
  return { Engine, version: fake.version, backend: 'wasm' };
}

test('createTypesafe loads the selected ONNX model and tokenizer through WASM fromBytes', async (t) => {
  const { modelDir, manifest, selected } = fixture(t);
  const calls = [];
  const ts = createTypesafe({ binding: wasmBinding(calls),
    embedder: { kind: 'onnx', modelDir, manifest, model: 'selected' } });
  const result = await ts.decide('charge', { dept: choice({ billing: 'b', fraud: 'f' }) });
  assert.equal(result.dept.choice, 'billing');
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].options.manifest, selected);
  assert.deepEqual(calls[0].model, readFileSync(join(modelDir, selected.file)));
  assert.deepEqual(calls[0].tokenizer, readFileSync(join(modelDir, selected.tokenizer_file)));
});

test('CLI ONNX failure reports the model-loading error instead of unknown failure', async (t) => {
  const { modelDir, manifest, questions } = fixture(t);
  const manifestJson = JSON.parse(readFileSync(manifest, 'utf8'));
  writeFileSync(manifest, JSON.stringify({ models: [manifestJson.models[1]] }));
  rmSync(join(modelDir, 'selected', 'model.onnx'));
  const lines = [];
  const result = await main(['decide', '--state', 'charge', '--questions', questions,
    '--embedder', 'onnx', '--model-dir', modelDir, '--manifest', manifest], {
    binding: wasmBinding([]), stdout: (s) => lines.push(s), stderr: (s) => lines.push(s),
  });
  assert.equal(result.code, 1);
  assert.match(lines.join('\n'), /error:.*(model|ENOENT|missing)/);
  assert.doesNotMatch(lines.join('\n'), /unknown failure/);
});

test('CLI --model selects a manifest entry for WASM', async (t) => {
  const { modelDir, manifest, questions } = fixture(t);
  const calls = [];
  const lines = [];
  const result = await main(['decide', '--state', 'charge', '--questions', questions,
    '--embedder', 'onnx', '--model-dir', modelDir, '--manifest', manifest,
    '--model', 'selected'], {
    binding: wasmBinding(calls), stdout: (s) => lines.push(s), stderr: (s) => lines.push(s),
  });
  assert.equal(result.code, 0, lines.join('\n'));
  assert.equal(calls[0].options.manifest.name, 'selected');
});

test('WASM rejects an ambiguous manifest and a path escaping modelDir', (t) => {
  const { modelDir, manifest } = fixture(t);
  const calls = [];
  const binding = wasmBinding(calls);
  assert.throws(() => createTypesafe({ binding, embedder: { kind: 'onnx', modelDir, manifest } }),
    /set embedder.model explicitly/);
  const contents = JSON.parse(readFileSync(manifest, 'utf8'));
  contents.models[1].file = '../secret.onnx';
  writeFileSync(manifest, JSON.stringify(contents));
  assert.throws(() => createTypesafe({ binding, embedder: {
    kind: 'onnx', modelDir, manifest, model: 'selected',
  } }), /unsafe file path/);
  assert.equal(calls.length, 0);
});

test('WASM string error envelope becomes a useful TypesafeError', (t) => {
  const { modelDir, manifest } = fixture(t);
  const binding = wasmBinding([]);
  binding.Engine.fromBytes = () => { throw '{"error":{"kind":"embedder","message":"wasm-onnx feature is missing"}}'; };
  assert.throws(() => createTypesafe({ binding, embedder: {
    kind: 'onnx', modelDir, manifest, model: 'selected',
  } }), /wasm-onnx feature is missing/);
});

test('WASM rejects symlink escapes and unsupported tuning before loading weights', (t) => {
  const { modelDir, manifest } = fixture(t);
  const binding = wasmBinding([]);
  assert.throws(() => createTypesafe({ binding, embedder: {
    kind: 'onnx', modelDir, manifest, model: 'selected',
  }, engine: { logitScale: 10 } }), /does not support engine tuning/);
  rmSync(join(modelDir, 'selected', 'model.onnx'));
  symlinkSync(join(modelDir, '..', 'manifest.json'), join(modelDir, 'selected', 'model.onnx'));
  assert.throws(() => createTypesafe({ binding, embedder: {
    kind: 'onnx', modelDir, manifest, model: 'selected',
  } }), /escapes modelDir/);
});
