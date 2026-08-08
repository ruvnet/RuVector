/**
 * The registry wire format is a contract shared with `crates/rvforge-registry`
 * and the Reader, so these tests pin the rules the parity test will reconcile
 * against: what an object id is computed over, where an object lives, and what
 * the release index and transparency log refuse.
 */

import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join, sep } from 'node:path';

import { ForgeError, ForgeErrorCode } from '../src/errors';
import { canonicalJson } from '../src/hash';
import {
  appendLogEntry,
  appendRelease,
  digestOf,
  getObject,
  inclusionProof,
  initRegistry,
  objectId,
  objectPath,
  putObject,
  readLog,
  readReleaseIndex,
  readTreeHead,
  verifyInclusionProof,
  verifyLog,
  type Release,
} from '../src/registry';

let root: string;

beforeEach(async () => {
  root = await mkdtemp(join(tmpdir(), 'rvforge-registry-'));
  await initRegistry(root);
});

afterEach(async () => {
  await rm(root, { recursive: true, force: true });
});

const release = (overrides: Partial<Release> = {}): Release => {
  const base = {
    schemaVersion: 1 as const,
    type: 'release' as const,
    package: { name: 'analyst', publisherId: `sha256:${'1'.repeat(64)}` },
    version: '1.0.0',
    predecessor: null,
    rvfIdentity: `sha256:${'2'.repeat(64)}`,
    rvfSize: 4096,
    capabilityManifest: `sha256:${'3'.repeat(64)}`,
    runtimeProfiles: ['wasm'],
    compatibility: { rvmVersionMin: '0.1.0', stateSchemaVersion: 1, witnessSchemaVersion: 1 },
    modelManifest: { location: 'embedded', digests: [] },
    softwareInventory: `sha256:${'4'.repeat(64)}`,
    evaluationReport: null,
    securityReport: null,
    provenance: `sha256:${'5'.repeat(64)}`,
    trustLevel: 'published' as const,
    rollbackSafeUntilStateSchema: 2,
    listing: { category: 'developer-tools', description: 'd', priceModel: 'free' },
    publishedAt: '2026-08-03T00:00:00.000Z',
    signatures: [],
    ...overrides,
  };
  return { ...base, releaseId: objectId(base) };
};

describe('object identity', () => {
  it('excludes signatures and the object’s own id field', () => {
    const object = { schemaVersion: 1, type: 'release', version: '1.0.0' };
    const bare = objectId(object);
    expect(bare).toBe(objectId({ ...object, releaseId: 'anything', signatures: [{ sig: 'x' }] }));
    expect(bare).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it('is order-independent, because the encoding is canonical', () => {
    expect(objectId({ b: 2, a: 1 })).toBe(objectId({ a: 1, b: 2 }));
  });

  it('changes when any other field changes', () => {
    expect(objectId({ a: 1 })).not.toBe(objectId({ a: 2 }));
    expect(release().releaseId).not.toBe(release({ version: '1.0.1' }).releaseId);
    expect(release().releaseId).not.toBe(release({ predecessor: `sha256:${'9'.repeat(64)}` }).releaseId);
  });
});

describe('object storage', () => {
  it('shards by the first two hex characters of the digest', async () => {
    const object = release();
    const { path } = await putObject(root, object.releaseId, object);
    const digest = digestOf(object.releaseId);
    expect(path).toBe(objectPath(root, object.releaseId));
    expect(path.endsWith(join('objects', 'sha256', digest.slice(0, 2), `${digest}.json`))).toBe(true);
  });

  it('stores canonical JSON with a trailing newline', async () => {
    const object = release();
    const { path } = await putObject(root, object.releaseId, object);
    expect(await readFile(path, 'utf8')).toBe(`${canonicalJson(object)}\n`);
  });

  it('is idempotent, and reports whether the object was already there', async () => {
    const object = release();
    expect((await putObject(root, object.releaseId, object)).existed).toBe(false);
    expect((await putObject(root, object.releaseId, object)).existed).toBe(true);
    expect(await getObject<Release>(root, object.releaseId)).toEqual(object);
  });

  it('returns null for an object that was never written', async () => {
    expect(await getObject(root, `sha256:${'e'.repeat(64)}`)).toBeNull();
  });

  it('refuses an id that is not a sha256 digest', () => {
    expect(() => objectPath(root, 'sha256:not-a-digest')).toThrow(ForgeError);
    try {
      objectPath(root, `../../escape${sep}`);
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.REGISTRY);
    }
  });
});

describe('release lineage', () => {
  it('appends a first release with a null predecessor', async () => {
    const first = release();
    await appendRelease(root, first);
    expect(await readReleaseIndex(root, first.package.publisherId, 'analyst')).toEqual([
      {
        releaseId: first.releaseId,
        version: '1.0.0',
        predecessor: null,
        publishedAt: first.publishedAt,
      },
    ]);
  });

  it('refuses a second release that does not name the head', async () => {
    const first = release();
    await appendRelease(root, first);
    const orphan = release({ version: '1.1.0', predecessor: null });

    try {
      await appendRelease(root, orphan);
      throw new Error('expected a lineage error');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.LINEAGE);
      expect((err as ForgeError).message).toMatch(/head of analyst/);
    }
  });

  it('refuses a repeated version even when the predecessor is right', async () => {
    const first = release();
    await appendRelease(root, first);
    try {
      await appendRelease(root, release({ predecessor: first.releaseId }));
      throw new Error('expected a lineage error');
    } catch (err) {
      expect((err as ForgeError).code).toBe(ForgeErrorCode.LINEAGE);
      expect((err as ForgeError).message).toMatch(/immutable/);
    }
  });

  it('reads an empty index for a package with no releases', async () => {
    expect(await readReleaseIndex(root, `sha256:${'1'.repeat(64)}`, 'nothing')).toEqual([]);
  });
});

describe('transparency log', () => {
  const entry = (object: string, timestamp: string) =>
    appendLogEntry(root, { object, objectType: 'release', timestamp });

  it('numbers entries from zero and advances the tree head', async () => {
    const first = await entry(`sha256:${'a'.repeat(64)}`, '2026-08-03T00:00:00.000Z');
    const second = await entry(`sha256:${'b'.repeat(64)}`, '2026-08-04T00:00:00.000Z');

    expect(first.entry.index).toBe(0);
    expect(second.entry.index).toBe(1);
    expect(second.entry.treeHead).not.toBe(first.entry.treeHead);
    expect((await readTreeHead(root))?.rootHash).toBe(second.head.rootHash);
    expect((await readTreeHead(root))?.treeSize).toBe(2);
    // Unsigned, but the slot is present: this registry holds no registry key.
    expect((await readTreeHead(root))?.signatures).toEqual([]);
  });

  it('proves inclusion of every entry against the head', async () => {
    for (const c of ['a', 'b', 'c', 'd', 'e']) {
      await entry(`sha256:${c.repeat(64)}`, '2026-08-03T00:00:00.000Z');
    }
    const entries = await readLog(root);
    const head = (await readTreeHead(root))?.rootHash as string;

    for (const logged of entries) {
      const proof = inclusionProof(entries, logged.object);
      expect(proof).not.toBeNull();
      expect(proof?.treeSize).toBe(5);
      expect(verifyInclusionProof(proof as NonNullable<typeof proof>, head)).toBe(true);
    }
  });

  it('refuses a proof for the wrong position or an unlogged object', async () => {
    for (const c of ['a', 'b', 'c']) {
      await entry(`sha256:${c.repeat(64)}`, '2026-08-03T00:00:00.000Z');
    }
    const entries = await readLog(root);
    const head = (await readTreeHead(root))?.rootHash as string;

    expect(inclusionProof(entries, `sha256:${'9'.repeat(64)}`)).toBeNull();
    const proof = inclusionProof(entries, entries[0].object) as NonNullable<
      ReturnType<typeof inclusionProof>
    >;
    expect(verifyInclusionProof({ ...proof, leafIndex: 2 }, head)).toBe(false);
    expect(verifyInclusionProof(proof, `sha256:${'0'.repeat(64)}`)).toBe(false);
  });

  it('verifies a log it wrote', async () => {
    await entry(`sha256:${'a'.repeat(64)}`, '2026-08-03T00:00:00.000Z');
    await entry(`sha256:${'b'.repeat(64)}`, '2026-08-04T00:00:00.000Z');
    const verification = verifyLog(await readLog(root));
    expect(verification.ok).toBe(true);
    expect(verification.size).toBe(2);
  });

  it('detects a removed entry through the head chain', async () => {
    for (const c of ['a', 'b', 'c']) {
      await entry(`sha256:${c.repeat(64)}`, '2026-08-03T00:00:00.000Z');
    }
    const entries = await readLog(root);
    const verification = verifyLog([entries[0], entries[2]]);
    expect(verification.ok).toBe(false);
    expect(verification.problems.map((p) => p.reason)).toEqual(
      expect.arrayContaining(['index', 'tree-head']),
    );
  });

  it('reads an empty log and a null head before anything is published', async () => {
    expect(await readLog(root)).toEqual([]);
    expect(await readTreeHead(root)).toBeNull();
    expect(verifyLog([])).toEqual({ ok: true, size: 0, treeHead: null, problems: [] });
  });
});
