import { chmod, mkdtemp, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runInit } from '../src/config';
import { ForgeError, ForgeErrorCode } from '../src/errors';
import { generatePublisherKey, loadPublisherKey, verifyObjectId, writePublisherKey } from '../src/keys';
import { publisherIdFor, type RvforgeProject } from '../src/project';
import { renderPublishSummary, runPublish } from '../src/publish';
import {
  RELEASES_FILENAME,
  digestOf,
  getObject,
  inclusionProof,
  objectPath,
  packagePath,
  publisherPath,
  readLog,
  readReleaseIndex,
  readTreeHead,
  readWitnessChain,
  verifyInclusionProof,
  verifyLog,
  type PublisherRecord,
  type Release,
} from '../src/registry';
import { verifyReceiptChain, type WitnessReceipt } from '../src/witness';
import { buildRvfFixture } from './fixtures/rvf-fixture';
import { fixtureProject } from './fixtures/project-fixture';

jest.setTimeout(30_000);

const POSIX = process.platform !== 'win32';

let root: string;
let rvfPath: string;
let keyFile: string;

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'rvforge-publish-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
  keyFile = join(root, 'publisher.key');
  await writePublisherKey(keyFile, generatePublisherKey());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

/** A fresh registry directory per test, so lineage assertions are isolated. */
const registry = (name: string): string => join(root, `registry-${name}`);

const publish = (
  registryDir: string,
  project: RvforgeProject = fixtureProject(),
  overrides: { keyFile?: string; publishedAt?: string } = {},
) =>
  runPublish({
    rvfPath,
    project,
    registryDir,
    keyFile: overrides.keyFile ?? keyFile,
    publishedAt: overrides.publishedAt ?? '2026-08-03T00:00:00.000Z',
  });

async function expectCode(promise: Promise<unknown>, code: ForgeErrorCode): Promise<ForgeError> {
  try {
    await promise;
  } catch (err) {
    expect(err).toBeInstanceOf(ForgeError);
    expect((err as ForgeError).code).toBe(code);
    return err as ForgeError;
  }
  throw new Error(`expected a ForgeError with code ${code}`);
}

const exists = async (path: string): Promise<boolean> =>
  stat(path).then(
    () => true,
    () => false,
  );

describe('publish — round trip', () => {
  it('writes every object at its content address', async () => {
    const dir = registry('roundtrip');
    const result = await publish(dir);

    expect(result.objects.map((o) => o.type).sort()).toEqual([
      'capability-manifest',
      'pack-provenance',
      'publisher',
      'release',
      'software-inventory',
    ]);
    for (const object of result.objects) {
      expect(object.path).toBe(objectPath(dir, object.id));
      expect(await exists(object.path)).toBe(true);
      const digest = digestOf(object.id);
      expect(object.path).toContain(join('objects', 'sha256', digest.slice(0, 2), `${digest}.json`));
    }
  });

  it('stores the release under its own id and links its manifest', async () => {
    const dir = registry('stored');
    const result = await publish(dir);

    const stored = await getObject<Release>(dir, result.release.releaseId);
    expect(stored?.releaseId).toBe(result.release.releaseId);
    expect(stored?.capabilityManifest).toBe(result.capabilityManifest.manifestId);
    expect(await getObject(dir, result.capabilityManifest.manifestId)).not.toBeNull();
    expect(await getObject(dir, result.release.softwareInventory)).not.toBeNull();
    expect(await getObject(dir, result.release.provenance)).not.toBeNull();
  });

  it('signs the release with the publisher key, verifiably', async () => {
    const dir = registry('signed');
    const result = await publish(dir);
    const key = await loadPublisherKey(keyFile);

    expect(result.release.signatures).toHaveLength(1);
    const [signature] = result.release.signatures;
    expect(signature.role).toBe('publisher');
    expect(signature.keyId).toBe(key.keyId);
    expect(verifyObjectId(key.publicKey, result.release.releaseId, signature.sig)).toBe(true);
    expect(verifyObjectId(key.publicKey, `${result.release.releaseId}x`, signature.sig)).toBe(false);
  });

  it('registers the publisher with its public key on the first publish', async () => {
    const dir = registry('publisher');
    const result = await publish(dir);
    const key = await loadPublisherKey(keyFile);

    expect(result.publisherRecord.created).toBe(true);
    expect(result.publisherIdentity).toBe(publisherIdFor(fixtureProject().publisher));

    const record = await getObject<PublisherRecord>(dir, result.publisherRecord.id);
    expect(record?.publicKeys).toEqual([
      expect.objectContaining({ keyId: key.keyId, alg: 'ed25519', revokedAt: null }),
    ]);
    // The pointer is filed under the stable identity; the record names its own
    // content address, which is what the release resolves.
    expect(await exists(publisherPath(dir, result.publisherIdentity))).toBe(true);
    expect(result.publisherId).toBe(result.publisherRecord.id);
    expect(record?.publisherId).toBe(result.publisherRecord.id);
    expect(result.release.package.publisherId).toBe(result.publisherRecord.id);
  });

  it('names a publisherId that resolves to the stored record', async () => {
    const dir = registry('publisher-resolves');
    const result = await publish(dir);

    // The rule the Rust registry enforces at publish time: a release's
    // package.publisherId must be an object address, not a separate identity.
    const record = await getObject<PublisherRecord>(dir, result.release.package.publisherId);
    expect(record).not.toBeNull();
    expect(record?.type).toBe('publisher');
    expect(result.publisherId).not.toBe(result.publisherIdentity);
  });

  it('appends to the package release index', async () => {
    const dir = registry('index');
    const result = await publish(dir);

    const indexPath = join(
      packagePath(dir, result.publisherId, fixtureProject().listing.name),
      RELEASES_FILENAME,
    );
    expect(result.releasesPath).toBe(indexPath);

    const index = await readReleaseIndex(dir, result.publisherId, 'analyst');
    expect(index).toEqual([
      {
        releaseId: result.release.releaseId,
        version: '1.0.0',
        predecessor: null,
        publishedAt: '2026-08-03T00:00:00.000Z',
      },
    ]);
  });

  it('appends a transparency log entry and advances the tree head', async () => {
    const dir = registry('log');
    const result = await publish(dir);

    const entries = await readLog(dir);
    // Two entries: the publisher registration, then the release.
    expect(entries.map((e) => e.objectType)).toEqual(['publisher', 'release']);
    expect(entries[1].object).toBe(result.release.releaseId);
    expect(result.logEntry.index).toBe(1);
    expect(verifyLog(entries).ok).toBe(true);

    const head = await readTreeHead(dir);
    expect(head?.treeSize).toBe(2);
    expect(head?.rootHash).toBe(entries[1].treeHead);

    const proof = inclusionProof(entries, result.release.releaseId);
    expect(proof).not.toBeNull();
    expect(verifyInclusionProof(proof as NonNullable<typeof proof>, head?.rootHash as string)).toBe(
      true,
    );
  });

  it('detects a tampered log entry', async () => {
    const dir = registry('tampered-log');
    await publish(dir);
    const entries = await readLog(dir);
    entries[1] = { ...entries[1], object: `sha256:${'f'.repeat(64)}` };
    expect(verifyLog(entries).ok).toBe(false);
    expect(verifyLog(entries).problems.map((p) => p.reason)).toContain('entry-hash');
  });

  it('emits a witness receipt, stored and indexed under its subject', async () => {
    const dir = registry('receipt');
    const result = await publish(dir);

    expect(result.receipt.event).toBe('publish');
    expect(result.receipt.outcome).toBe('pass');
    expect(result.receipt.subject).toBe(result.release.rvfIdentity);
    expect(result.receipt.prevReceipt).toBeNull();
    // Evidence is one `details` string — the shape registry-model.md gives it.
    expect(Object.keys(result.receipt.evidence)).toEqual(['details']);
    expect(result.receipt.evidence.details).toContain(result.release.releaseId);

    expect(await getObject(dir, result.receipt.receiptId)).not.toBeNull();
    expect(await readWitnessChain(dir, result.release.rvfIdentity)).toEqual([
      result.receipt.receiptId,
    ]);
  });

  it('chains a second receipt for the same subject onto the first', async () => {
    const dir = registry('receipt-chain');
    const first = await publish(dir);
    const second = await publish(dir, fixtureProject({ version: '1.1.0' }), {
      publishedAt: '2026-08-04T00:00:00.000Z',
    });

    const subject = first.release.rvfIdentity;
    expect(second.receipt.subject).toBe(subject);
    expect(second.receipt.prevReceipt).toBe(first.receipt.receiptId);

    const chain = await readWitnessChain(dir, subject);
    expect(chain).toEqual([first.receipt.receiptId, second.receipt.receiptId]);

    const receipts = await Promise.all(
      chain.map((id) => getObject<WitnessReceipt>(dir, id) as Promise<WitnessReceipt>),
    );
    expect(verifyReceiptChain(receipts).ok).toBe(true);
  });

  it('prints the P4 summary without inventing an evaluation score', async () => {
    const dir = registry('summary');
    const { summary } = await publish(dir);
    const lines = renderPublishSummary(summary);

    expect(summary.evaluationScore).toBeNull();
    expect(lines[0]).toBe('Validation passed');
    expect(lines).toContain('Security profile: restricted');
    expect(lines).toContain('Supported runtimes: WASM');
    expect(lines).toContain('Supported systems: Linux, Windows, macOS');
    expect(lines).toContain('Network access: none');
    expect(lines).toContain('Ready to publish');
    expect(lines.join('\n')).toMatch(/Evaluation score: not evaluated: requires quarantined runtime/);
  });
});

describe('publish — lineage', () => {
  it('links a second release to its predecessor', async () => {
    const dir = registry('lineage');
    const first = await publish(dir);
    const second = await publish(dir, fixtureProject({ version: '1.1.0' }), {
      publishedAt: '2026-08-04T00:00:00.000Z',
    });

    expect(first.predecessor).toBeNull();
    expect(second.predecessor).toBe(first.release.releaseId);
    expect(second.release.predecessor).toBe(first.release.releaseId);
    expect(second.release.releaseId).not.toBe(first.release.releaseId);

    const index = await readReleaseIndex(dir, first.publisherId, 'analyst');
    expect(index.map((e) => e.version)).toEqual(['1.0.0', '1.1.0']);
    expect(index[1].predecessor).toBe(index[0].releaseId);
  });

  it('reuses the existing publisher record on the second publish', async () => {
    const dir = registry('publisher-reuse');
    const first = await publish(dir);
    const second = await publish(dir, fixtureProject({ version: '1.1.0' }));

    expect(second.publisherRecord.created).toBe(false);
    expect(second.publisherRecord.id).toBe(first.publisherRecord.id);
  });

  it('refuses to republish a version, because releases are immutable', async () => {
    const dir = registry('immutable');
    await publish(dir);
    const err = await expectCode(publish(dir), ForgeErrorCode.LINEAGE);
    expect(err.message).toMatch(/already published/);
  });
});

describe('publish — key handling', () => {
  it('refuses a key the publisher has not registered', async () => {
    const dir = registry('wrong-key');
    await publish(dir);

    const otherKeyFile = join(root, 'other.key');
    if (!(await exists(otherKeyFile))) await writePublisherKey(otherKeyFile, generatePublisherKey());

    const err = await expectCode(
      publish(dir, fixtureProject({ version: '2.0.0' }), { keyFile: otherKeyFile }),
      ForgeErrorCode.KEY,
    );
    expect(err.message).toMatch(/not a registered, unrevoked key/);
  });

  it('refuses to publish with no key at all', async () => {
    await expectCode(
      runPublish({ rvfPath, project: fixtureProject(), registryDir: registry('no-key') }),
      ForgeErrorCode.KEY,
    );
  });

  it('reports a missing key file as not found', async () => {
    await expectCode(
      publish(registry('missing-key'), fixtureProject(), { keyFile: join(root, 'absent.key') }),
      ForgeErrorCode.NOT_FOUND,
    );
  });

  (POSIX ? it : it.skip)('refuses a world-readable key file', async () => {
    const loose = join(root, 'loose.key');
    await writePublisherKey(loose, generatePublisherKey());
    await chmod(loose, 0o644);

    const err = await expectCode(
      publish(registry('loose-key'), fixtureProject(), { keyFile: loose }),
      ForgeErrorCode.KEY,
    );
    expect(err.message).toMatch(/readable beyond its owner/);
    expect(err.message).toMatch(/chmod 600/);
  });

  (POSIX ? it : it.skip)('writes a generated key with owner-only permissions', async () => {
    const path = join(root, 'fresh.key');
    await writePublisherKey(path, generatePublisherKey());
    expect((await stat(path)).mode & 0o777).toBe(0o600);
  });

  it('never overwrites an existing key file', async () => {
    await expectCode(writePublisherKey(keyFile, generatePublisherKey()), ForgeErrorCode.KEY);
  });

  it('keeps key material out of every published object and out of the result', async () => {
    const dir = registry('no-leak');
    const result = await publish(dir);
    const secret = JSON.parse(await readFile(keyFile, 'utf8')) as { privateKey: string };

    const serialised = JSON.stringify(result);
    expect(serialised).not.toContain(secret.privateKey);
    for (const object of result.objects) {
      expect(await readFile(object.path, 'utf8')).not.toContain(secret.privateKey);
    }
  });
});

describe('init --keygen', () => {
  it('scaffolds a project file and a key, and reports the path rather than the key', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rvforge-init-'));
    try {
      const result = await runInit(join(dir, 'forge.config.json'), { keygen: true });

      expect(result.projectPath).toBe(join(dir, 'rvforge.json'));
      expect(result.keyFile).toBe(join(dir, 'rvforge-publisher.key'));
      expect(result.keyId).toMatch(/^ed25519:[0-9a-f]{64}$/);
      expect(result.project?.publisher.keyFile).toBe(result.keyFile);
      // The result carries the path and the key id — never the key itself.
      expect(JSON.stringify(result)).not.toContain(
        (JSON.parse(await readFile(result.keyFile as string, 'utf8')) as { privateKey: string }).privateKey,
      );
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('publishes end to end from a scaffolded project', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rvforge-e2e-'));
    try {
      const init = await runInit(join(dir, 'forge.config.json'), { keygen: true });
      const project = fixtureProject({
        publisher: { ...fixtureProject().publisher, keyFile: init.keyFile ?? null },
      });
      const result = await runPublish({
        rvfPath,
        project,
        registryDir: join(dir, 'registry'),
        publishedAt: '2026-08-03T00:00:00.000Z',
      });

      expect(result.keyId).toBe(init.keyId);
      expect(result.release.signatures[0].keyId).toBe(init.keyId);
      expect(verifyLog(await readLog(result.registryDir)).ok).toBe(true);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
