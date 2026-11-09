/**
 * Export surface: every 3.0.0 export and every 4.0.1 export must resolve
 * from @ruvector/ruvllm (self-referenced through package.json "exports",
 * so this exercises the same map consumers use).
 *
 * 4.0.1 (the GPU job loop) lives in @ruvector/ruvllm-gpu-loop; when it is
 * installed every subpath must expose the 4.0.1 names, otherwise importing a
 * subpath must fail with RUVLLM_GPU_LOOP_MISSING. Set RUVLLM_REQUIRE_GPU_LOOP=1
 * to make a missing package a test failure (release verification).
 */
const { test, describe } = require('node:test');
const assert = require('node:assert');
const surface = require('./fixtures/export-surface.json');
const pkg = require('../package.json');

function gpuLoopInstalled() {
  try {
    require.resolve('@ruvector/ruvllm-gpu-loop/package.json');
    return true;
  } catch {
    return false;
  }
}

describe('3.0.0 export surface', () => {
  for (const [subpath, names] of Object.entries(surface['3.0.0'])) {
    const specifier = subpath === '.' ? pkg.name : `${pkg.name}/${subpath.slice(2)}`;

    test(`require('${specifier}') exposes all ${names.length} 3.0.0 exports`, () => {
      const mod = require(specifier);
      const missing = names.filter((n) => !(n in mod));
      assert.deepStrictEqual(missing, []);
    });

    test(`import('${specifier}') exposes all ${names.length} 3.0.0 exports`, async () => {
      const mod = await import(specifier);
      const missing = names.filter((n) => !(n in mod));
      assert.deepStrictEqual(missing, []);
    });
  }

  test('symbols ruflo uses are constructible', () => {
    const m = require(pkg.name);
    assert.ok(m.DEFAULT_SONA_CONFIG && typeof m.DEFAULT_SONA_CONFIG === 'object');
    assert.ok(new m.SonaCoordinator(m.DEFAULT_SONA_CONFIG));
    assert.ok(new m.ContrastiveTrainer({ batchSize: 4, margin: 0.5 }));
    const adapter = new m.LoraAdapter(8, 8, 2);
    assert.ok(new m.TrainingPipeline({ adapter, learningRate: 0.001, batchSize: 4 }));
    assert.ok(new m.TrainingPipeline({}));
  });

  test('package.json is exported (capability detection reads name/version)', () => {
    const manifest = require(`${pkg.name}/package.json`);
    assert.strictEqual(manifest.name, '@ruvector/ruvllm');
    assert.strictEqual(manifest.version, pkg.version);
  });
});

describe('4.0.1 export surface (via @ruvector/ruvllm-gpu-loop)', () => {
  const installed = gpuLoopInstalled();
  if (process.env.RUVLLM_REQUIRE_GPU_LOOP === '1') {
    test('@ruvector/ruvllm-gpu-loop is installed', () => assert.ok(installed));
  }

  for (const [subpath, names] of Object.entries(surface['4.0.1'].subpaths)) {
    const specifier = `${pkg.name}/${subpath.slice(2)}`;

    test(`import('${specifier}')`, async () => {
      if (!installed) {
        await assert.rejects(import(specifier), { code: 'RUVLLM_GPU_LOOP_MISSING' });
        return;
      }
      const mod = await import(specifier);
      const missing = names.filter((n) => mod[n] === undefined);
      assert.deepStrictEqual(missing, []);
      const original = await import(`@ruvector/ruvllm-gpu-loop${subpath === './engine' ? '' : subpath.slice(1)}`);
      for (const n of names) assert.strictEqual(mod[n], original[n], `${subpath} ${n} is the gpu-loop export`);
    });

    test(`require('${specifier}')`, () => {
      if (!installed) {
        assert.throws(() => require(specifier), { code: 'RUVLLM_GPU_LOOP_MISSING' });
        return;
      }
      let mod;
      try {
        mod = require(specifier);
      } catch (error) {
        // require(esm) is unavailable before Node 20.19 / 22.12.
        assert.strictEqual(error.code, 'RUVLLM_GPU_LOOP_ESM_ONLY');
        return;
      }
      const missing = names.filter((n) => mod[n] === undefined);
      assert.deepStrictEqual(missing, []);
    });
  }

  test('4.0.1 bins are still provided', () => {
    for (const bin of surface['4.0.1'].bins) {
      if (bin === 'ruvllm') continue; // ruvllm is the runtime CLI again; 4.0.1's is `ruvllm gpu`
      assert.ok(pkg.bin[bin], `bin ${bin}`);
    }
    assert.strictEqual(pkg.bin.ruvllm, './bin/cli.js');
  });
});
