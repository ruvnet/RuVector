import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtemp, readFile, rm, writeFile, mkdir, symlink } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Miniflare } from 'miniflare';
import { buildWorker } from '../examples/cloudflare/build.mjs';

const token = 'e-means-workerd-test-not-for-production';
const config = { compatibilityDate: '2026-07-30', modules: true,
  modulesRules: [{ type: 'CompiledWasm', include: ['**/*.wasm'], fallthrough: true }] };

test('packed npm static Rust WASM executes in real workerd without Node compatibility', { timeout: 60_000 }, async t => {
  const temporary = await mkdtemp(join(tmpdir(), 'e-means-workerd-'));
  const packageDir = fileURLToPath(new URL('../', import.meta.url));
  const run = (args, cwd) => {
    const result = spawnSync('npm', args, { cwd, encoding: 'utf8', env: { ...process.env, npm_config_workspaces: 'false' } });
    assert.equal(result.status, 0, `${result.stdout}\n${result.stderr}`);
    return result.stdout;
  };
  const archive = JSON.parse(run(['pack', '--json', '--workspaces=false', '--pack-destination', temporary], packageDir))[0];
  const consumer = join(temporary, 'consumer');
  await mkdir(consumer);
  await writeFile(join(consumer, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  run(['install', '--ignore-scripts', '--no-audit', '--no-fund', '--offline', join(temporary, archive.filename)], consumer);
  // esbuild is caller-supplied build tooling, not an installed runtime dependency.
  await symlink(dirname(dirname(fileURLToPath(import.meta.resolve('esbuild')))), join(consumer, 'node_modules/esbuild'), 'dir');
  const installed = join(consumer, 'node_modules/@ruvector/e-means');
  const builder = await import(pathToFileURL(join(installed, 'examples/cloudflare/build.mjs')).href);
  const output = join(temporary, 'output');
  const artifact = await builder.buildWorker(output);
  const source = await readFile(artifact.script, 'utf8');
  assert.match(source, /\.\/e-means\.wasm/);
  assert.doesNotMatch(source, /node:fs|node:crypto|node:os/);
  const worker = new Miniflare({ ...config, modulesRoot: output, scriptPath: artifact.script,
    bindings: { E_MEANS_BEARER_TOKEN: token } });
  t.after(async () => { await worker.dispose(); await rm(temporary, { recursive: true, force: true }); });
  const invoke = (path, value, authorization = `Bearer ${token}`) => worker.dispatchFetch(`https://worker.test${path}`, {
    method: 'POST', headers: { 'content-type': 'application/json', authorization }, body: JSON.stringify(value),
  });
  const health = await worker.dispatchFetch('https://worker.test/health');
  assert.equal(health.status, 200);
  assert.equal((await health.json()).stateless, true);
  const fitted = await invoke('/fit', { data: [1, 2, 3, 4, 5, 6], dimensions: 2, options: { clusters: 1 } });
  assert.equal(fitted.status, 200, await fitted.clone().text());
  const model = await fitted.json();
  assert.deepEqual(model.centroids, [3, 4]);
  assert.deepEqual(model.assignments, [0, 0, 0]);
  assert.equal(model.mse, 16 / 3);
  const predicted = await invoke('/predict', { data: [1, 1, 9, 9], dimensions: 2, centroids: [0, 0, 10, 10] });
  assert.deepEqual(await predicted.json(), { assignments: [0, 1] });
  assert.equal((await invoke('/fit', {}, 'Bearer incorrect')).status, 401);
  assert.equal((await invoke('/fit', { data: [1, 2], dimensions: 0, options: { clusters: 1 } })).status, 422);
});

test('R2 codebook helpers use the real local binding and Vectorize contract remains explicit', { timeout: 60_000 }, async t => {
  const temporary = await mkdtemp(join(tmpdir(), 'e-means-r2-workerd-'));
  const entry = join(temporary, 'entry.mjs');
  const adapter = fileURLToPath(new URL('../integrations/cloudflare.mjs', import.meta.url));
  await writeFile(entry, `
import { persistCodebook, loadCodebook } from ${JSON.stringify(adapter)};
export default { async fetch(request, env) {
  const original = { dimensions: 2, centroids: new Float32Array([1, 2, 3, 4]) };
  const receipt = await persistCodebook(env.CODEBOOKS, 'test/codebook', original);
  const restored = await loadCodebook(env.CODEBOOKS, 'test/codebook');
  return Response.json({ receipt, centroids: Array.from(restored.centroids), clusters: restored.clusters });
} };
`);
  const artifact = await buildWorker(join(temporary, 'out'), entry);
  const worker = new Miniflare({ ...config, modulesRoot: join(temporary, 'out'), scriptPath: artifact.script, r2Buckets: ['CODEBOOKS'] });
  t.after(async () => { await worker.dispose(); await rm(temporary, { recursive: true, force: true }); });
  const response = await worker.dispatchFetch('https://worker.test/storage');
  assert.equal(response.status, 200, await response.clone().text());
  const result = await response.json();
  assert.deepEqual(result.centroids, [1, 2, 3, 4]);
  assert.equal(result.clusters, 2);
  const bucket = await worker.getR2Bucket('CODEBOOKS');
  const stored = JSON.parse(await (await bucket.get('test/codebook')).text());
  assert.equal(stored.schema, 'ruvector.e-means-codebook/1');
  assert.equal(Object.hasOwn(stored, 'assignments'), false);
});
