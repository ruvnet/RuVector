import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const tooling = { '@metaharness/darwin': '0.13.0', typescript: '5.9.3' };

function toolingPackages(source) {
  assert.equal(source.lockfileVersion, 3, 'source tooling requires a v3 lock');
  const packages = {};
  function resolve(from, name) {
    for (let prefix = from; ; ) {
      const key = `${prefix ? `${prefix}/` : ''}node_modules/${name}`;
      if (Object.hasOwn(source.packages, key)) return key;
      if (!prefix) throw new Error(`Source lock lacks dependency ${name} from ${from}`);
      const parent = prefix.lastIndexOf('/node_modules/');
      prefix = parent < 0 ? '' : prefix.slice(0, parent);
    }
  }
  function include(key) {
    if (Object.hasOwn(packages, key)) return;
    const record = source.packages[key];
    assert.ok(record && !record.link && record.version, `Missing locked tooling ${key}`);
    const url = new URL(record.resolved);
    assert.equal(url.protocol, 'https:');
    assert.equal(url.hostname, 'registry.npmjs.org');
    assert.ok(!url.username && !url.password && !url.search && !url.hash && url.pathname.includes('/-/') && url.pathname.endsWith('.tgz'), 'Expected a canonical registry tarball URL');
    assert.ok(record.integrity?.startsWith('sha512-'), `Missing integrity for ${key}`);
    // Current pinned tools have no optional or peer closure. Reject a changed
    // graph explicitly instead of silently resolving missing registry metadata.
    assert.equal(Object.keys(record.optionalDependencies ?? {}).length, 0, `Unsupported optional tooling dependency ${key}`);
    assert.equal(Object.keys(record.peerDependencies ?? {}).length, 0, `Unsupported peer tooling dependency ${key}`);
    packages[key] = structuredClone(record);
    for (const name of Object.keys(record.dependencies ?? {})) include(resolve(key, name));
  }
  for (const [name, version] of Object.entries(tooling)) {
    const key = `node_modules/${name}`;
    assert.equal(source.packages[key]?.version, version, `Source lock must pin ${name}@${version}`);
    include(key);
  }
  return packages;
}

/** Only the six exact SDK/compiler closure tarballs; never package metadata. */
export function toolingTarballURLs(source) {
  const urls = Object.values(toolingPackages(source)).map(record => record.resolved);
  assert.equal(urls.length, 6, 'Expected six minimal tooling tarballs');
  assert.equal(new Set(urls).size, 6, 'Tooling tarballs must be unique');
  return urls;
}

/** Reuse exact registry records; never resolve package names or repack tooling. */
export function offlineToolingLock(source, consumer, manifest) {
  assert.equal(consumer.lockfileVersion, 3, 'consumer requires a v3 lock');
  assert.deepEqual(Object.keys(manifest.dependencies ?? {}), ['@ruvector/e-means']);
  assert.deepEqual(consumer.packages[''].dependencies, manifest.dependencies);
  const own = consumer.packages['node_modules/@ruvector/e-means'];
  assert.equal(own?.version, '0.1.0', 'retain the actually installed shared archive');
  assert.ok(own.resolved?.startsWith('file:') && own.integrity?.startsWith('sha512-'));
  const packages = { '': { ...consumer.packages[''], devDependencies: tooling },
    'node_modules/@ruvector/e-means': own, ...toolingPackages(source) };

  return { manifest: { ...manifest, devDependencies: tooling },
    lock: { ...consumer, packages } };
}

// This is an explicit prefetch step. All subsequent consumer installs stay
// offline and enforce the original source lock's integrity hashes via npm ci.
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  assert.equal(process.argv[2], 'warm');
  assert.equal(process.argv.length, 4);
  const source = JSON.parse(await readFile(process.argv[3], 'utf8'));
  for (const url of toolingTarballURLs(source)) {
    const result = spawnSync('npm', ['cache', 'add', url, '--ignore-scripts', '--workspaces=false'], { stdio: 'inherit' });
    assert.equal(result.status, 0, `Failed to warm locked tooling tarball: ${url}`);
  }
  console.log('Warmed six lock-bound tooling tarballs without package metadata');
}
