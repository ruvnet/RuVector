import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const repository = fileURLToPath(new URL('../../../../', import.meta.url));
const packageRoot = fileURLToPath(new URL('../', import.meta.url));
const temporary = await mkdtemp(join(tmpdir(), 'e-means-ecosystem-'));
function run(command, args, cwd = temporary) {
  const result = spawnSync(command, args, {
    cwd, encoding: 'utf8', timeout: 180_000,
    env: { ...process.env, npm_config_workspaces: 'false' },
  });
  assert.equal(result.status, 0, `${command} ${args.join(' ')}\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}
try {
  await writeFile(join(temporary, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  const packages = [
    { path: packageRoot, check: true },
    { path: join(repository, 'npm/packages/core') },
    { path: join(repository, 'npm/packages/ruvector') },
    { path: join(repository, 'npm/packages/all') },
    { path: join(repository, 'crates/ruvector-wasm') },
  ];
  const archives = [];
  for (const item of packages) {
    const args = ['pack', '--json', '--workspaces=false', '--pack-destination', temporary];
    // The shared artifact's normal prepack executes its actual WASM. Consumer
    // prepack lanes build unrelated database SDKs; this test qualifies only the
    // newly shipped clustering subpaths, not those packages' legacy root builds.
    if (!item.check) args.push('--ignore-scripts');
    const [packed] = JSON.parse(run('npm', args, item.path));
    const paths = packed.files.map(file => file.path);
    if (item.check) {
      for (const name of ['runtime.mjs', 'wasm/ruvector_e_means_bg.wasm', 'integrations/metaharness.mjs', 'integrations/cloudflare.mjs']) {
        assert.ok(paths.includes(name), `missing shared artifact ${name}`);
      }
    } else {
      for (const name of ['clustering.mjs', 'clustering.d.mts', 'clustering/quantization.mjs', 'clustering/ivf.mjs', 'clustering/metaharness.mjs', 'clustering/cloudflare.mjs']) {
        assert.ok(paths.includes(name), `missing consumer artifact ${name}`);
      }
    }
    archives.push(join(temporary, packed.filename));
  }
  run('npm', ['install', '--ignore-scripts', '--omit=optional', '--no-audit', '--no-fund', ...archives]);
  await writeFile(join(temporary, 'consumer.mjs'), `
    import assert from 'node:assert/strict';
    import { readFile } from 'node:fs/promises';
    const cases = [
      ['@ruvector/core/clustering.mjs', '@ruvector/core/clustering/'],
      ['ruvector/clustering.mjs', 'ruvector/clustering/'],
      ['@ruvector/all/clustering.mjs', '@ruvector/all/clustering/'],
      ['@ruvector/wasm/clustering', '@ruvector/wasm/clustering/'],
    ];
    for (const [root, prefix] of cases) {
      const extension = root.endsWith('.mjs') ? '.mjs' : '';
      const core = await import(root);
      await core.init();
      const data = new Float32Array([0, 0, 0, 2, 10, 10, 10, 12]);
      const quantization = await import(prefix + 'quantization' + extension);
      const book = quantization.trainProductQuantizer(data, 2, 2, { clusters: 1, epochs: 1 });
      const codes = quantization.encodeProductQuantizer(book, data);
      assert.ok(codes instanceof Uint8Array);
      assert.deepEqual([...quantization.decodeProductQuantizer(book, codes)], [5,6,5,6,5,6,5,6]);
      const ivf = await import(prefix + 'ivf' + extension);
      const partitions = ivf.partition(data, 2, { clusters: 1 });
      assert.deepEqual([...ivf.postingList(partitions, 0)], [0,1,2,3]);
      const meta = await import(prefix + 'metaharness' + extension);
      const evaluator = meta.createClusteringEvaluator({ training: data,
        holdout: new Float32Array([5,6]), dimensions: 2,
        baseline: { clusters: 1, epochs: 1, batchSize: 4 },
        domain: { batchSize: { min: 1, max: 4, type: 'int', scale: 'linear' } },
        budget: { maxEvaluations: 1, maxDistanceCoordinates: 100000, timeoutMs: 30000 } });
      const score = await evaluator.evaluate(evaluator.baselineGenome, 'installed-baseline');
      assert.equal(score.raw.holdoutMse, 0);
      const edge = await import(prefix + 'cloudflare' + extension);
      const bytes = await readFile(new URL(import.meta.resolve('@ruvector/e-means/wasm')));
      const wasmModule = await WebAssembly.compile(bytes);
      const token = 'local-installed-consumer-test-token';
      const worker = edge.createClusteringWorker({ wasmModule, bearerToken: token });
      const response = await worker.fetch(new Request('https://local.test/fit', {
        method: 'POST', headers: { authorization: 'Bearer ' + token, 'content-type': 'application/json' },
        body: JSON.stringify({ data: [1,2,3,4], dimensions: 2, options: { clusters: 1 } }),
      }));
      assert.equal(response.status, 200);
      assert.deepEqual((await response.json()).centroids, [2,3]);
      console.log(root + ': PQ, IVF, heldout evaluator and Worker facades passed');
    }
  `);
  process.stdout.write(run(process.execPath, ['consumer.mjs']));
  const installed = JSON.parse(await readFile(join(temporary, 'node_modules/@ruvector/e-means/package.json')));
  assert.equal(installed.version, '0.1.1');
  console.log('Installed ecosystem subpaths passed; consumer root/prepack release lanes were not qualified');
} finally {
  await rm(temporary, { recursive: true, force: true });
}
