import assert from 'node:assert/strict';

const tooling = { '@metaharness/darwin': '0.13.0', typescript: '5.9.3' };

/** Reuse exact registry records; never resolve package names or repack tooling. */
export function offlineToolingLock(source, consumer, manifest) {
  assert.equal(source.lockfileVersion, 3, 'source tooling requires a v3 lock');
  assert.equal(consumer.lockfileVersion, 3, 'consumer requires a v3 lock');
  assert.deepEqual(Object.keys(manifest.dependencies ?? {}), ['@ruvector/e-means']);
  assert.deepEqual(consumer.packages[''].dependencies, manifest.dependencies);
  const own = consumer.packages['node_modules/@ruvector/e-means'];
  assert.equal(own?.version, '0.1.0', 'retain the actually installed shared archive');
  assert.ok(own.resolved?.startsWith('file:') && own.integrity?.startsWith('sha512-'));
  const packages = { '': { ...consumer.packages[''], devDependencies: tooling },
    'node_modules/@ruvector/e-means': own };

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
  return { manifest: { ...manifest, devDependencies: tooling },
    lock: { ...consumer, packages } };
}
