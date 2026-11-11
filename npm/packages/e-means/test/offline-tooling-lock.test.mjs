import assert from 'node:assert/strict';
import test from 'node:test';
import { offlineToolingLock, toolingTarballURLs } from './offline-tooling-lock.mjs';

function fixture() {
  const record = (name, version, dependencies = {}) => ({ version, dev: true, dependencies,
    resolved: `https://registry.npmjs.org/${name}/-/${name.split('/').at(-1)}-${version}.tgz`, integrity: 'sha512-fixture' });
  const source = { lockfileVersion: 3, packages: {
    'node_modules/@metaharness/darwin': record('@metaharness/darwin', '0.13.0', { '@babel/parser': '^7.29.13' }),
    'node_modules/@babel/parser': record('@babel/parser', '7.29.13', { '@babel/types': '^7.29.11' }),
    'node_modules/@babel/types': record('@babel/types', '7.29.11', { '@babel/helper-string-parser': '^7.29.11', '@babel/helper-validator-identifier': '^7.29.7' }),
    'node_modules/@babel/helper-string-parser': record('@babel/helper-string-parser', '7.29.13'),
    'node_modules/@babel/helper-validator-identifier': record('@babel/helper-validator-identifier', '7.29.7'),
    'node_modules/typescript': record('typescript', '5.9.3'),
    'node_modules/unrelated-build-tool': record('unrelated-build-tool', '1.0.0'),
  } };
  const manifest = { private: true, type: 'module', dependencies: { '@ruvector/e-means': 'file:actual-shared.tgz' } };
  const own = { version: '0.1.0', resolved: 'file:actual-shared.tgz', integrity: 'sha512-actual-shared-fixture',
    peerDependencies: { '@metaharness/darwin': '0.13.0 || 0.13.1' }, peerDependenciesMeta: { '@metaharness/darwin': { optional: true } } };
  const consumer = { lockfileVersion: 3, packages: { '': { dependencies: manifest.dependencies }, 'node_modules/@ruvector/e-means': own } };
  return { source, consumer, manifest };
}

test('minimal tooling graph preserves every source record and the actual packed shared record', () => {
  const { source, consumer, manifest } = fixture();
  const original = structuredClone({ source, consumer, manifest });
  const result = offlineToolingLock(source, consumer, manifest);
  assert.equal(Object.keys(result.lock.packages).length, 8); // root + own + six registry records
  assert.ok(!Object.hasOwn(result.lock.packages, 'node_modules/unrelated-build-tool'));
  assert.deepEqual(result.lock.packages['node_modules/@ruvector/e-means'], consumer.packages['node_modules/@ruvector/e-means']);
  for (const [key, value] of Object.entries(result.lock.packages)) {
    if (key && key !== 'node_modules/@ruvector/e-means') assert.deepEqual(value, source.packages[key]);
  }
  assert.deepEqual(result.manifest.devDependencies, { '@metaharness/darwin': '0.13.0', typescript: '5.9.3' });
  assert.deepEqual(result.lock.packages[''].devDependencies, result.manifest.devDependencies);
  assert.deepEqual({ source, consumer, manifest }, original);
});

test('missing transitive records, changed tools, missing integrity and non-registry bytes fail closed', () => {
  for (const mutation of ['transitive', 'version', 'integrity', 'registry', 'peer', 'shared']) {
    const { source, consumer, manifest } = fixture();
    if (mutation === 'transitive') delete source.packages['node_modules/@babel/types'];
    if (mutation === 'version') source.packages['node_modules/typescript'].version = '5.9.2';
    if (mutation === 'integrity') delete source.packages['node_modules/@babel/parser'].integrity;
    if (mutation === 'registry') source.packages['node_modules/@babel/parser'].resolved = 'https://example.com/parser.tgz';
    if (mutation === 'peer') source.packages['node_modules/@babel/parser'].peerDependencies = { missing: '*' };
    if (mutation === 'shared') consumer.packages['node_modules/@ruvector/e-means'].resolved = 'https://registry.npmjs.org/shared.tgz';
    assert.throws(() => offlineToolingLock(source, consumer, manifest), undefined, mutation);
  }
});

test('cache warming selects only six locked tarball URLs and rejects metadata endpoints', () => {
  const { source } = fixture();
  const urls = toolingTarballURLs(source);
  assert.equal(urls.length, 6);
  assert.deepEqual(urls, Object.values(source.packages).slice(0, 6).map(row => row.resolved));
  for (const url of ['https://registry.npmjs.org/@babel/parser', 'https://registry.npmjs.org/parser/-/parser.tgz?metadata=true']) {
    const modified = structuredClone(source);
    modified.packages['node_modules/@babel/parser'].resolved = url;
    assert.throws(() => toolingTarballURLs(modified), /tarball URL/);
  }
  source.packages['node_modules/@babel/parser'].dependencies.unexpected = '*';
  source.packages['node_modules/unexpected'] = { ...source.packages['node_modules/typescript'], resolved: 'https://registry.npmjs.org/unexpected/-/unexpected-1.0.0.tgz' };
  assert.throws(() => toolingTarballURLs(source), /six minimal/);
});
