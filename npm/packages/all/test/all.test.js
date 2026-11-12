'use strict';

// node --test. Installs nothing: members may or may not be present in
// node_modules, so access tests accept either a loaded module or a
// module-not-found error — what they assert is *when* loading happens.

const test = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const Module = require('node:module');
const { execFile } = require('node:child_process');
const { promisify } = require('node:util');

const ROOT = path.join(__dirname, '..');
const pkg = require(path.join(ROOT, 'package.json'));
const INDEX = path.join(ROOT, 'index.js');
const NOT_FOUND = new Set(['MODULE_NOT_FOUND', 'ERR_MODULE_NOT_FOUND']);

function freshRequireWithSpy() {
  delete require.cache[require.resolve(INDEX)];
  const requested = [];
  const orig = Module._load;
  Module._load = function (request, ...rest) {
    requested.push(request);
    return orig.call(this, request, ...rest);
  };
  try {
    const all = require(INDEX);
    return { all, requested, restore: () => { Module._load = orig; } };
  } catch (e) {
    Module._load = orig;
    throw e;
  }
}

test('package.json shape', () => {
  assert.equal(pkg.name, '@ruvector/all');
  assert.deepEqual(pkg.files, ['index.js', 'index.d.ts', 'README.md', 'clustering.mjs', 'clustering.d.mts', 'clustering/']);
  assert.equal(pkg.optionalDependencies, undefined);
  for (const name of Object.keys(pkg.dependencies)) {
    // Platform binaries resolve via each member's optionalDependencies.
    assert.doesNotMatch(name, /-(linux|darwin|win32|android|freebsd)-/, `${name} is platform-suffixed`);
    // Caret ranges, or an exact pin for new 0.x edges (lane policy).
    assert.match(pkg.dependencies[name], /^\^?\d+\.\d+\.\d+$/, `${name} must use a caret range or exact pin`);
  }
});

test('manifest covers exactly the declared dependencies', () => {
  const all = require(INDEX);
  const fromManifest = all.members.map((m) => m.pkg).sort();
  assert.deepEqual(fromManifest, Object.keys(pkg.dependencies).sort());
});

test('requiring @ruvector/all loads no member', () => {
  const { all, requested, restore } = freshRequireWithSpy();
  try {
    const memberPkgs = new Set(all.members.map((m) => m.pkg));
    assert.deepEqual(requested.filter((r) => memberPkgs.has(r)), []);
    for (const m of all.members) assert.equal(all.isLoaded(m.key), false, m.key);
  } finally {
    restore();
  }
});

test('enumerating exports does not trigger getters', () => {
  const { all, requested, restore } = freshRequireWithSpy();
  try {
    Object.keys(all);
    Object.getOwnPropertyNames(all);
    const memberPkgs = new Set(all.members.map((m) => m.pkg));
    assert.deepEqual(requested.filter((r) => memberPkgs.has(r)), []);
  } finally {
    restore();
  }
});

test('CJS getter loads only its own module, on access', () => {
  const { all, requested, restore } = freshRequireWithSpy();
  try {
    const memberPkgs = new Set(all.members.map((m) => m.pkg));
    try {
      void all.diskann;
    } catch (e) {
      if (!NOT_FOUND.has(e.code)) throw e;
    }
    assert.deepEqual(requested.filter((r) => memberPkgs.has(r)), ['@ruvector/diskann']);
  } finally {
    restore();
  }
});

test('ESM loaders are lazy and return promises', async () => {
  const { all, restore } = freshRequireWithSpy();
  try {
    for (const m of all.members.filter((x) => x.kind === 'esm')) {
      assert.equal(typeof all[m.loader], 'function', m.loader);
    }
    assert.equal(all.isLoaded('emergentTime'), false);
    const p = all.loadEmergentTime();
    assert.ok(p instanceof Promise);
    assert.equal(all.isLoaded('emergentTime'), true);
    assert.equal(all.isLoaded('graphWasm'), false, 'other ESM members stay unloaded');
    try {
      await p;
    } catch (e) {
      if (!NOT_FOUND.has(e.code)) throw e;
      assert.equal(all.isLoaded('emergentTime'), false, 'failed import is not cached');
    }
  } finally {
    restore();
  }
});

test('load() rejects unknown keys', async () => {
  const all = require(INDEX);
  await assert.rejects(all.load('nope'), TypeError);
  assert.throws(() => all.isLoaded('nope'), TypeError);
});

test('index.d.ts declares exactly the runtime exports', () => {
  const all = require(INDEX);
  const dts = fs.readFileSync(path.join(ROOT, 'index.d.ts'), 'utf8');
  const declared = [...dts.matchAll(/^export declare (?:const|function) (\w+)/gm)].map((m) => m[1]).sort();
  assert.deepEqual(declared, Object.keys(all).sort());

  // Each member's typed import must point at the right package.
  for (const m of all.members) {
    const name = m.kind === 'cjs' ? m.key : m.loader;
    const re = new RegExp(`export declare (?:const|function) ${name}\\b[^\\n]*import\\('${m.pkg.replace('/', '\\/')}'\\)`);
    assert.match(dts, re, `${name} → ${m.pkg}`);
    assert.match(dts, new RegExp(`^\\s+${m.key}: typeof import\\('${m.pkg.replace('/', '\\/')}'\\);`, 'm'), `MemberModules.${m.key}`);
  }
});

test('every dependency range resolves on npm', { timeout: 120_000 }, async (t) => {
  if (process.env.RUVECTOR_ALL_OFFLINE) return t.skip('RUVECTOR_ALL_OFFLINE set');
  const run = promisify(execFile);
  const results = await Promise.all(
    Object.entries(pkg.dependencies).map(async ([name, range]) => {
      try {
        // Pass the range verbatim, like regression-guard.yml does.
        const { stdout } = await run('npm', ['view', `${name}@${range}`, 'version', '--json'], { timeout: 60_000 });
        return { name, range, ok: stdout.trim().length > 0 };
      } catch (e) {
        const out = `${e.stdout || ''}${e.stderr || ''}`;
        return { name, range, ok: false, notFound: /E404|is not in this registry/.test(out), out };
      }
    }),
  );
  const missing = results.filter((r) => r.notFound);
  assert.deepEqual(missing.map((r) => `${r.name}@${r.range}`), [], 'not published on npm');
  const transient = results.filter((r) => !r.ok && !r.notFound);
  if (transient.length) t.diagnostic(`could not reach npm for: ${transient.map((r) => r.name).join(', ')}`);
});
