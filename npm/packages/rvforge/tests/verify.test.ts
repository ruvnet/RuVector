import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runBuild } from '../src/build';
import { defaultConfig } from '../src/config';
import { ForgeError, ForgeErrorCode } from '../src/errors';
import { manifestFileContents } from '../src/manifest';
import { runVerify } from '../src/verify';
import type { BuildManifest, ForgeConfig } from '../src/types';
import { buildRvfFixture } from './fixtures/rvf-fixture';

jest.setTimeout(30_000);

/** Where the single configured target's embedded payload is staged. */
const PAYLOAD = 'bundles/linux-x64/rvf/agent.rvf';
const READER_SLOT = 'bundles/linux-x64/reader/reader-slot.json';

let root: string;
let rvfPath: string;

const config = (): ForgeConfig =>
  defaultConfig({
    app: {
      name: 'Fixture Agent',
      version: '0.1.0',
      publisher: 'RuVector Tests',
      identifier: 'dev.ruvector.fixture',
    },
    targets: ['linux-x64'],
  });

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'forge-verify-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

async function build(name: string) {
  return runBuild({ config: config(), rvfPath, outDir: join(root, name), force: true });
}

async function expectCode(promise: Promise<unknown>, code: ForgeErrorCode): Promise<ForgeError> {
  try {
    await promise;
  } catch (err) {
    expect(err).toBeInstanceOf(ForgeError);
    expect((err as ForgeError).code).toBe(code);
    return err as ForgeError;
  }
  throw new Error(`expected ${code}`);
}

describe('build output', () => {
  it('stages a bundle with a manifest, inventory, payload, provenance, and checksums', async () => {
    const result = await build('out-happy');

    expect(result.packaged).toBe(false);
    expect(result.provenance.status).toBe('staged');
    expect(result.provenance.signing.mode).toBe('unsigned-development');
    expect(result.provenance.files.map((f) => f.path).sort()).toEqual([
      'build-manifest.json',
      READER_SLOT,
      PAYLOAD,
      'inventory.json',
    ].sort());

    // Embedded mode copies the RVF byte for byte: the embedded hash is the
    // input hash (ADR-283 invariant 1).
    const embedded = result.provenance.files.find((f) => f.path === PAYLOAD);
    expect(embedded?.sha256).toBe(result.manifest.rvf.sha256);
  });

  it('leaves no output directory behind when the build fails', async () => {
    const outDir = join(root, 'out-failed');
    const broken = { ...config(), app: { ...config().app, version: 'not-semver' } };
    await expectCode(runBuild({ config: broken, rvfPath, outDir }), ForgeErrorCode.MANIFEST);
    await expect(readFile(join(outDir, 'provenance.json'))).rejects.toThrow();
  });

  it('refuses to overwrite an existing output directory without --force', async () => {
    const outDir = join(root, 'out-exists');
    await runBuild({ config: config(), rvfPath, outDir });
    await expectCode(runBuild({ config: config(), rvfPath, outDir }), ForgeErrorCode.IO);
  });
});

describe('verify', () => {
  it('verifies an untouched build directory', async () => {
    const built = await build('out-verify-ok');
    const result = await runVerify(built.outDir);

    expect(result.verified).toBe(true);
    expect(result.failures).toEqual([]);
    expect(result.manifestSha256.ok).toBe(true);
    expect(result.rvfIdentity).toBe(built.manifest.rvf.sha256);
    expect(result.files).toHaveLength(4);
  });

  it('verifies a single artifact inside a build directory', async () => {
    const built = await build('out-verify-single');
    const result = await runVerify(join(built.outDir, PAYLOAD));
    expect(result.target?.path).toBe(PAYLOAD);
    expect(result.target?.ok).toBe(true);
  });

  it('fails when the embedded RVF is tampered with', async () => {
    const built = await build('out-verify-tampered');
    const payload = join(built.outDir, PAYLOAD);
    const bytes = await readFile(payload);
    bytes[0] ^= 0xff;
    await writeFile(payload, bytes);

    const err = await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(err.message).toMatch(/rvf\/agent\.rvf/);
  });

  it('fails when an artifact was removed', async () => {
    const built = await build('out-verify-missing');
    await rm(join(built.outDir, 'inventory.json'));
    const err = await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(err.details.failedFiles).toEqual([{ path: 'inventory.json', reason: 'missing' }]);
  });

  it('fails when the manifest is rewritten and its provenance digest updated to match', async () => {
    const built = await build('out-verify-manifest-swap');
    const manifestPath = join(built.outDir, 'build-manifest.json');

    // The strongest tamper available to an attacker with write access: change
    // the manifest AND fix up the file digest recorded for it. The recomputed
    // canonical manifest hash still fails, because that is derived from
    // content rather than copied from the record.
    const manifest = JSON.parse(await readFile(manifestPath, 'utf8')) as BuildManifest;
    manifest.capabilityPolicy.allow.network = ['*'];
    const rewritten = manifestFileContents(manifest);
    await writeFile(manifestPath, rewritten);

    const provenancePath = join(built.outDir, 'provenance.json');
    const provenance = JSON.parse(await readFile(provenancePath, 'utf8'));
    const { createHash } = await import('node:crypto');
    provenance.files = provenance.files.map((f: { path: string }) =>
      f.path === 'build-manifest.json'
        ? {
            path: f.path,
            sha256: createHash('sha256').update(rewritten).digest('hex'),
            size: Buffer.byteLength(rewritten),
          }
        : f,
    );
    await writeFile(provenancePath, JSON.stringify(provenance));

    const err = await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(err.message).toMatch(/manifest hash/);
  });

  it('reports a missing provenance record', async () => {
    await expectCode(runVerify(root), ForgeErrorCode.NOT_FOUND);
  });

  it('rejects a file that is not listed in the provenance record', async () => {
    const built = await build('out-verify-foreign');
    const foreign = join(built.outDir, 'unexpected.exe');
    await writeFile(foreign, 'not from this build');
    await expectCode(runVerify(foreign), ForgeErrorCode.VERIFY_FAILED);
  });
});
