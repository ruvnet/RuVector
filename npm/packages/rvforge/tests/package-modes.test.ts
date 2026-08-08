import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runBuild } from '../src/build';
import { defaultConfig } from '../src/config';
import { ForgeError, ForgeErrorCode } from '../src/errors';
import { sha256File } from '../src/hash';
import {
  assertEmbeddedRvfIdentical,
  assertLocator,
  buildLocator,
  verifyLocator,
  type RvfLocator,
} from '../src/package-modes';
import { runVerify } from '../src/verify';
import type { ForgeConfig, PackagingMode } from '../src/types';
import { buildRvfFixture } from './fixtures/rvf-fixture';

jest.setTimeout(30_000);

const DISTRIBUTION_URL = 'https://dist.example/agent.rvf';
const TARGETS = ['linux-x64', 'windows-x64'];

let root: string;
let rvfPath: string;

const config = (mode: PackagingMode = 'embedded'): ForgeConfig =>
  defaultConfig({
    app: {
      name: 'Fixture Agent',
      version: '0.1.0',
      publisher: 'RuVector Tests',
      identifier: 'dev.ruvector.fixture',
    },
    targets: TARGETS as ForgeConfig['targets'],
    packaging: mode === 'embedded' ? { mode } : { mode, distributionUrl: DISTRIBUTION_URL },
  });

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'forge-modes-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

const build = (name: string, mode: PackagingMode = 'embedded') =>
  runBuild({ config: config(mode), rvfPath, outDir: join(root, name), force: true });

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

describe('embedded mode', () => {
  it('stages one bundle per target with a reader slot and the payload', async () => {
    const built = await build('embedded-layout');

    expect(built.layout.mode).toBe('embedded');
    expect(built.layout.bundles.map((b) => b.target)).toEqual(['linux-x64', 'windows-x64']);
    for (const bundle of built.layout.bundles) {
      expect(bundle.rvfPath).toBe(`bundles/${bundle.target}/rvf/agent.rvf`);
      expect(bundle.locatorPath).toBeUndefined();
      expect(bundle.readerSlotPath).toBe(`bundles/${bundle.target}/reader/reader-slot.json`);
    }
  });

  it('embeds byte-identical bytes in every target bundle (core invariant 1)', async () => {
    const built = await build('embedded-identical');

    const digests = await Promise.all(
      built.layout.bundles.map((b) => sha256File(join(built.outDir, b.rvfPath!))),
    );
    expect(new Set(digests).size).toBe(1);
    expect(digests[0]).toBe(built.manifest.rvf.sha256);
    expect(digests[0]).toBe(await sha256File(rvfPath));
  });

  it('fails the build when one target bundle carries different bytes', async () => {
    const built = await build('embedded-corrupt');
    const [, second] = built.layout.bundles;

    const corrupted = join(built.outDir, second.rvfPath!);
    const bytes = await readFile(corrupted);
    bytes[0] ^= 0xff;
    await writeFile(corrupted, bytes);

    const err = await expectCode(
      assertEmbeddedRvfIdentical(built.outDir, built.layout),
      ForgeErrorCode.VERIFY_FAILED,
    );
    expect(err.message).toMatch(/every platform package must embed identical bytes/);
    expect(err.details.diverged).toEqual([
      { target: second.target, sha256: await sha256File(corrupted) },
    ]);

    // The same corruption is caught downstream by the provenance digests, so
    // an already-shipped bundle does not depend on the build-time check alone.
    await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
  });

  it('reports a bundle whose payload never got staged', async () => {
    const built = await build('embedded-missing-slot');
    const layout = {
      ...built.layout,
      bundles: built.layout.bundles.map((b) => ({ ...b, rvfPath: undefined })),
    };
    await expectCode(assertEmbeddedRvfIdentical(built.outDir, layout), ForgeErrorCode.VERIFY_FAILED);
  });

  it('does not stage a locator', async () => {
    const built = await build('embedded-no-locator');
    await expect(readFile(join(built.outDir, 'bundles/linux-x64/rvf/locator.json'))).rejects.toThrow();
  });
});

describe('thin mode', () => {
  it('stages a signed locator instead of the payload', async () => {
    const built = await build('thin-layout', 'thin');

    expect(built.layout.mode).toBe('thin');
    expect(built.layout.rvfSha256).toBeNull();
    for (const bundle of built.layout.bundles) {
      expect(bundle.locatorPath).toBe(`bundles/${bundle.target}/rvf/locator.json`);
      expect(bundle.rvfPath).toBeUndefined();
      await expect(readFile(join(built.outDir, `bundles/${bundle.target}/rvf/agent.rvf`))).rejects.toThrow();
    }
  });

  it('round-trips a staged locator against the build manifest', async () => {
    const built = await build('thin-roundtrip', 'thin');
    const expected = {
      sha256: built.manifest.rvf.sha256,
      size: built.manifest.rvf.size,
      fileId: built.manifest.rvf.fileId,
      capabilityPolicyHash: built.manifest.capabilityPolicyHash,
    };

    for (const bundle of built.layout.bundles) {
      const locator = JSON.parse(
        await readFile(join(built.outDir, bundle.locatorPath!), 'utf8'),
      ) as RvfLocator;

      expect(locator.type).toBe('rvf-locator');
      expect(locator.url).toBe(DISTRIBUTION_URL);
      expect(locator.rvf.sha256).toBe(built.manifest.rvf.sha256);
      expect(verifyLocator(locator, expected)).toEqual({ ok: true, problems: [] });
      expect(() => assertLocator(locator, expected)).not.toThrow();
    }
  });

  it('carries a signature slot and never key material', async () => {
    const built = await build('thin-signature-slot', 'thin');
    const locator = JSON.parse(
      await readFile(join(built.outDir, built.layout.bundles[0].locatorPath!), 'utf8'),
    ) as RvfLocator;

    expect(locator.signature).toEqual({
      alg: 'ed25519',
      provider: null,
      keyRef: null,
      sig: null,
      status: 'unsigned-slot',
    });
    const raw = await readFile(join(built.outDir, built.layout.bundles[0].locatorPath!), 'utf8');
    expect(raw).not.toMatch(/BEGIN [A-Z ]*PRIVATE KEY/);
  });

  it('rejects a locator pointing at a different RVF', async () => {
    const built = await build('thin-wrong-rvf', 'thin');
    const expected = {
      sha256: built.manifest.rvf.sha256,
      size: built.manifest.rvf.size,
      fileId: built.manifest.rvf.fileId,
      capabilityPolicyHash: built.manifest.capabilityPolicyHash,
    };
    const tampered = { ...buildLocator(built.manifest), rvf: { ...built.manifest.rvf, sha256: 'f'.repeat(64) } };

    const check = verifyLocator(tampered, expected);
    expect(check.ok).toBe(false);
    expect(check.problems.join(' ')).toMatch(/rvf\.sha256/);

    try {
      assertLocator(tampered, expected);
      throw new Error('expected FORGE_E_VERIFY_FAILED');
    } catch (err) {
      expect(err).toBeInstanceOf(ForgeError);
      expect((err as ForgeError).code).toBe(ForgeErrorCode.VERIFY_FAILED);
    }
  });

  it('rejects a locator with an inline signature', () => {
    const manifest = { rvf: { sha256: 'a'.repeat(64), size: 10, fileId: 'b'.repeat(32) }, capabilityPolicyHash: 'c'.repeat(64) };
    const check = verifyLocator(
      {
        schemaVersion: 1,
        type: 'rvf-locator',
        url: DISTRIBUTION_URL,
        rvf: { sha256: manifest.rvf.sha256, sizeBytes: 10, fileId: manifest.rvf.fileId },
        capabilityPolicyHash: manifest.capabilityPolicyHash,
        signature: { alg: 'ed25519', provider: null, keyRef: null, sig: 'inline', status: 'unsigned-slot' },
      },
      { sha256: manifest.rvf.sha256, size: 10, fileId: manifest.rvf.fileId, capabilityPolicyHash: manifest.capabilityPolicyHash },
    );
    expect(check.ok).toBe(false);
    expect(check.problems.join(' ')).toMatch(/signature slots only/);
  });

  it('rejects a non-object and a locator with no identity block', () => {
    const expected = { sha256: 'a'.repeat(64), size: 1, fileId: 'b'.repeat(32), capabilityPolicyHash: 'c'.repeat(64) };
    expect(verifyLocator(null, expected).ok).toBe(false);
    expect(verifyLocator({ schemaVersion: 1, type: 'rvf-locator', url: DISTRIBUTION_URL }, expected).problems)
      .toContain('locator carries no rvf identity block');
  });
});

describe('mode selection', () => {
  it('lets --mode override the configured mode for one build', async () => {
    const built = await runBuild({
      config: { ...config('thin'), packaging: { mode: 'thin', distributionUrl: DISTRIBUTION_URL } },
      rvfPath,
      mode: 'embedded',
      outDir: join(root, 'mode-override'),
      force: true,
    });
    expect(built.manifest.packaging.mode).toBe('embedded');
    expect(built.layout.bundles[0].rvfPath).toBeDefined();
  });

  it('refuses shared-reader, which the compatibility matrix marks planned', async () => {
    await expectCode(
      runBuild({
        config: config(),
        rvfPath,
        mode: 'shared-reader',
        outDir: join(root, 'mode-shared-reader'),
        force: true,
      }),
      ForgeErrorCode.UNSUPPORTED_TARGET,
    );
  });
});
