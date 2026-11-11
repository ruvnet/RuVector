import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { offlineToolingLock } from './offline-tooling-lock.mjs';

const packageDir = fileURLToPath(new URL('../', import.meta.url));
const temp = await mkdtemp(join(tmpdir(), 'e-means-consumer-'));
function run(command, args, cwd = temp) {
  const result = spawnSync(command, args, { cwd, encoding: 'utf8', env: { ...process.env, npm_config_workspaces: 'false' } });
  assert.equal(result.status, 0, `${command} ${args.join(' ')}\n${result.stdout}\n${result.stderr}`);
  return result.stdout;
}
try {
  const pack = JSON.parse(run('npm', ['pack', '--json', '--pack-destination', temp, '--workspaces=false'], packageDir));
  const files = pack[0].files.map((entry) => entry.path);
  assert.ok(files.includes('wasm/ruvector_e_means_bg.wasm'));
  assert.ok(files.includes('wasm/ruvector_e_means.js'));
  assert.ok(files.includes('e-means.d.ts'));
  for (const path of ['runtime.mjs', 'runtime.d.ts', 'integrations/metaharness.mjs', 'integrations/metaharness.d.ts', 'integrations/clustering-validation.mjs']) {
    assert.ok(files.includes(path), `archive missing ${path}`);
  }
  assert.ok(!files.some((path) => path.endsWith('.rs') || path.includes('node_modules')));
  await writeFile(join(temp, 'package.json'), JSON.stringify({ private: true, type: 'module' }));
  run('npm', ['install', '--ignore-scripts', '--no-audit', '--no-fund', '--offline', join(temp, pack[0].filename)]);
  await writeFile(join(temp, 'consumer.mjs'), `
    import assert from 'node:assert/strict';
    import { init, fit, predict } from '@ruvector/e-means';
    import { createClusteringEvaluator, tuneClustering } from '@ruvector/e-means/metaharness';
    import { fit as runtimeFit } from '@ruvector/e-means/runtime';
    await init();
    const data = new Float32Array([1, 1, 3, 3]);
    const result = fit(data, 2, { clusters: 1 });
    assert.deepEqual([...result.centroids], [2, 2]);
    assert.equal(result.mse, 2);
    assert.deepEqual([...predict(data, 2, result.centroids)], [0, 0]);
    assert.equal(runtimeFit(data, 2, { clusters: 1 }).mse, 2);
    const evaluation = { training: new Float32Array([0,0,0,2,10,10,10,12]), holdout: new Float32Array([0,1,10,11]), dimensions:2,
      baseline:{clusters:2,epochs:2,batchSize:2,peakStep:0.5}, domain:{clusters:{min:1,max:2,type:'int',scale:'linear'}},
      budget:{maxEvaluations:3,maxDistanceCoordinates:100000} };
    const evaluator = createClusteringEvaluator(evaluation);
    assert.equal((await evaluator.evaluate(evaluator.baselineGenome, 'baseline')).raw.holdoutMse, 0);
    await assert.rejects(tuneClustering({...evaluation,generations:1,childrenPerGeneration:2}), /optional peer/);
    console.log('Installed archive runtime/evaluator passed without the optional SDK');
  `);
  process.stdout.write(run(process.execPath, ['consumer.mjs']));
  const installed = JSON.parse(await readFile(join(temp, 'node_modules/@ruvector/e-means/package.json')));
  assert.equal(installed.version, '0.1.1');
  // npm ci cached the source lock's tarballs, not necessarily registry package
  // metadata. Transfer only the exact SDK/tool closure and the actual phase-one
  // shared archive record, then install without any name/version resolution.
  const sourceLock = JSON.parse(await readFile(join(packageDir, 'package-lock.json')));
  const consumerLock = JSON.parse(await readFile(join(temp, 'package-lock.json')));
  const consumerManifest = JSON.parse(await readFile(join(temp, 'package.json')));
  const offline = offlineToolingLock(sourceLock, consumerLock, consumerManifest);
  await writeFile(join(temp, 'package.json'), JSON.stringify(offline.manifest));
  await writeFile(join(temp, 'package-lock.json'), JSON.stringify(offline.lock));
  run('npm', ['ci', '--offline', '--ignore-scripts', '--no-audit', '--no-fund']);
  await writeFile(join(temp, 'meta-consumer.mjs'), `
    import assert from 'node:assert/strict';
    import { tuneClustering } from '@ruvector/e-means/metaharness';
    import { evolveNumeric } from '@metaharness/darwin';
    assert.equal(typeof evolveNumeric, 'function');
    const result = await tuneClustering({
      training:new Float32Array([0,0,0,2,10,10,10,12]), holdout:new Float32Array([0,1,10,11]), dimensions:2,
      baseline:{clusters:2,epochs:2,batchSize:2,peakStep:0.5}, domain:{clusters:{min:1,max:2,type:'int',scale:'linear'}},
      budget:{maxEvaluations:3,maxDistanceCoordinates:100000}, generations:1,childrenPerGeneration:2 });
    assert.equal(result.engine.version,'0.13.0');
    assert.equal(result.trials.length,3);
    assert.equal(result.baseline.score.raw.holdoutMse,0);
    assert.equal(result.sourceApplied,false);
    assert.equal(result.temporaryArchiveRemoved,true);
    console.log('Offline installed MetaHarness/WASM consumer passed');
  `);
  process.stdout.write(run(process.execPath, ['meta-consumer.mjs']));
  await writeFile(join(temp, 'types.mts'), `
    import {createClusteringEvaluator,tuneClustering,type ClusteringEvaluationOptions} from '@ruvector/e-means/metaharness';
    import {init,fit} from '@ruvector/e-means/runtime';
    const config:ClusteringEvaluationOptions={training:new Float32Array([0,1]),holdout:new Float32Array([2]),dimensions:1,
      baseline:{clusters:1,epochs:1},domain:{epochs:{min:1,max:2,type:'int',scale:'linear'}},budget:{maxEvaluations:2,maxDistanceCoordinates:1000}};
    const evaluator=createClusteringEvaluator(config);
    const result=await tuneClustering({...config,generations:1,childrenPerGeneration:1});
    const notApplied:false=result.sourceApplied;
    await init(new WebAssembly.Module(new Uint8Array()));
    const assignments:Uint32Array=fit(config.training,1,config.baseline).assignments;
    void [evaluator,notApplied,assignments];
    // @ts-expect-error explicit source required by Node-free runtime
    await init();
  `);
  run(process.execPath, ['node_modules/typescript/bin/tsc', '--noEmit', '--strict', '--target', 'es2022', '--module', 'nodenext', '--moduleResolution', 'nodenext', '--lib', 'es2022,dom', 'types.mts']);
  console.log('Offline installed integration declarations passed');
} finally {
  await rm(temp, { recursive: true, force: true });
}
