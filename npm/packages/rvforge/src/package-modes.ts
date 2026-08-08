/**
 * Packaging modes: how the RVF reaches each target's bundle (FR001–FR003).
 *
 * - **embedded** — the bundle contains the complete RVF. The *same bytes* go
 *   into every target's bundle, which is core invariant 1: the embedded RVF
 *   hash must be identical across every platform package. Forge stages the
 *   copies and then re-hashes each one; a build whose copies diverge fails
 *   rather than shipping platform packages that disagree about what the agent
 *   is.
 * - **thin** — the bundle contains a signed RVF locator instead of the
 *   payload: where to fetch it, which digest to expect, and a slot for the
 *   signature. The reader downloads and verifies before execution.
 *
 * Every bundle also gets a *reader placeholder slot*: a JSON descriptor of the
 * RVF Reader the packaging layer will drop in. It is deliberately not an
 * executable — nothing in a staged bundle is runnable, and forge never
 * executes RVF content (ADR-283 invariant 2).
 */

import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { basename, dirname, join } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJsonFile, sha256File } from './hash';
import type { BuildManifest, BuildTarget, PackagingMode } from './types';

/** Directory, relative to the build root, holding the per-target bundles. */
export const BUNDLES_DIR = 'bundles';

/** One target's staged bundle. Paths are relative to the build root. */
export interface BundleEntry {
  target: BuildTarget;
  mode: PackagingMode;
  /** Embedded mode: the staged RVF copy. */
  rvfPath?: string;
  /** Thin and shared-reader modes: the signed locator. */
  locatorPath?: string;
  /** Where the packaging layer places the RVF Reader. */
  readerSlotPath: string;
}

/** Deterministic record of where everything sits in the staged output. */
export interface BundleLayout {
  schemaVersion: 1;
  mode: PackagingMode;
  /** SHA256 every embedded copy must equal; null in non-embedded modes. */
  rvfSha256: string | null;
  bundles: BundleEntry[];
}

/** A signed RVF locator (FR002). `signature` is a slot — never key material. */
export interface RvfLocator {
  schemaVersion: 1;
  type: 'rvf-locator';
  mode: PackagingMode;
  url: string | null;
  rvf: { sha256: string; sizeBytes: number; fileId: string };
  capabilityPolicyHash: string;
  updateChannel: string | null;
  /**
   * Signature *slot*. Forge records which key the signing worker will use and
   * leaves `sig` null; private key material never enters a forge process.
   */
  signature: {
    alg: 'ed25519';
    provider: string | null;
    keyRef: string | null;
    sig: null;
    status: 'unsigned-slot';
  };
}

/** The reader placeholder slot written into every bundle. */
export interface ReaderSlot {
  schemaVersion: 1;
  type: 'reader-slot';
  target: BuildTarget;
  runtimeProfile: string;
  rvmVersion: string;
  rvmCommit: string;
  /** Nothing executable is staged: the packaging layer fills this slot. */
  status: 'placeholder';
}

const bundleDir = (target: BuildTarget): string => `${BUNDLES_DIR}/${target}`;

/**
 * Stage one bundle per target and return the layout plus every relative path
 * written, for the inventory and provenance records.
 */
export async function stageBundles(input: {
  stagingDir: string;
  rvfPath: string;
  manifest: BuildManifest;
}): Promise<{ layout: BundleLayout; files: string[] }> {
  const { stagingDir, rvfPath, manifest } = input;
  const mode = manifest.packaging.mode;
  const embedded = mode === 'embedded';
  const payloadName = basename(rvfPath);

  const bundles: BundleEntry[] = [];
  const files: string[] = [];

  for (const target of manifest.targets) {
    const dir = bundleDir(target);
    const readerSlotPath = `${dir}/reader/reader-slot.json`;
    await writeJson(stagingDir, readerSlotPath, readerSlot(target, manifest));
    files.push(readerSlotPath);

    if (embedded) {
      const rvfRelative = `${dir}/rvf/${payloadName}`;
      await mkdir(join(stagingDir, dirname(rvfRelative)), { recursive: true });
      await copyFile(rvfPath, join(stagingDir, rvfRelative));
      files.push(rvfRelative);
      bundles.push({ target, mode, rvfPath: rvfRelative, readerSlotPath });
      continue;
    }

    const locatorPath = `${dir}/rvf/locator.json`;
    await writeJson(stagingDir, locatorPath, buildLocator(manifest));
    files.push(locatorPath);
    bundles.push({ target, mode, locatorPath, readerSlotPath });
  }

  const layout: BundleLayout = {
    schemaVersion: 1,
    mode,
    rvfSha256: embedded ? manifest.rvf.sha256 : null,
    bundles,
  };
  return { layout, files: files.sort() };
}

/**
 * Core invariant 1: every target's embedded RVF is byte-identical.
 *
 * Re-hashes each staged copy rather than trusting that `copyFile` did what it
 * was asked. A divergence here means one platform's package would carry a
 * different agent from another's, which is exactly the failure the invariant
 * exists to catch, so the build fails instead of continuing.
 *
 * @throws {ForgeError} `FORGE_E_VERIFY_FAILED` when any copy diverges.
 */
export async function assertEmbeddedRvfIdentical(
  stagingDir: string,
  layout: BundleLayout,
): Promise<{ target: BuildTarget; sha256: string }[]> {
  if (layout.mode !== 'embedded') return [];
  const expected = layout.rvfSha256;
  if (!expected) {
    throw new ForgeError(
      ForgeErrorCode.INTERNAL,
      'Embedded layout carries no expected RVF digest.',
    );
  }

  const observed: { target: BuildTarget; sha256: string }[] = [];
  for (const bundle of layout.bundles) {
    if (!bundle.rvfPath) {
      throw new ForgeError(
        ForgeErrorCode.VERIFY_FAILED,
        `Embedded bundle for ${bundle.target} has no staged RVF.`,
        { target: bundle.target },
      );
    }
    observed.push({ target: bundle.target, sha256: await sha256File(join(stagingDir, bundle.rvfPath)) });
  }

  const diverged = observed.filter((entry) => entry.sha256 !== expected);
  if (diverged.length > 0) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `Embedded RVF differs between target bundles — every platform package must embed identical bytes ` +
        `(expected ${expected}). Diverged: ${diverged.map((d) => d.target).join(', ')}.`,
      {
        expectedSha256: expected,
        diverged: diverged.map((d) => ({ target: d.target, sha256: d.sha256 })),
      },
    );
  }
  return observed;
}

/** Build the locator a thin-mode bundle carries in place of the payload. */
export function buildLocator(manifest: BuildManifest): RvfLocator {
  const signing = manifest.signing.linux ?? manifest.signing.macos ?? manifest.signing.windows;
  return {
    schemaVersion: 1,
    type: 'rvf-locator',
    mode: manifest.packaging.mode,
    url: manifest.packaging.distributionUrl ?? null,
    rvf: {
      sha256: manifest.rvf.sha256,
      sizeBytes: manifest.rvf.size,
      fileId: manifest.rvf.fileId,
    },
    capabilityPolicyHash: manifest.capabilityPolicyHash,
    updateChannel: manifest.packaging.updateChannel ?? null,
    signature: {
      alg: 'ed25519',
      provider: signing?.provider ?? null,
      keyRef: signing?.keyRef ?? null,
      sig: null,
      status: 'unsigned-slot',
    },
  };
}

export interface LocatorCheck {
  ok: boolean;
  problems: string[];
}

/**
 * Check a locator against the identity it claims to point at.
 *
 * This is the round-trip the reader performs before it fetches anything: a
 * locator that names the wrong digest, size, file id, or capability policy is
 * not a shortcut to the RVF, it is a different artifact.
 */
export function verifyLocator(
  value: unknown,
  expected: { sha256: string; size: number; fileId: string; capabilityPolicyHash: string },
): LocatorCheck {
  const problems: string[] = [];
  const locator = value as Partial<RvfLocator> | null;

  if (!locator || typeof locator !== 'object') {
    return { ok: false, problems: ['locator is not a JSON object'] };
  }
  if (locator.type !== 'rvf-locator') problems.push(`type is "${String(locator.type)}", expected "rvf-locator"`);
  if (locator.schemaVersion !== 1) problems.push(`unsupported schemaVersion ${String(locator.schemaVersion)}`);
  if (!locator.url) problems.push('no distribution URL — a thin package cannot fetch its RVF');

  const rvf = locator.rvf;
  if (!rvf) {
    problems.push('locator carries no rvf identity block');
  } else {
    if (rvf.sha256 !== expected.sha256) problems.push(`rvf.sha256 ${rvf.sha256} does not match ${expected.sha256}`);
    if (rvf.sizeBytes !== expected.size) problems.push(`rvf.sizeBytes ${rvf.sizeBytes} does not match ${expected.size}`);
    if (rvf.fileId !== expected.fileId) problems.push(`rvf.fileId ${rvf.fileId} does not match ${expected.fileId}`);
  }
  if (locator.capabilityPolicyHash !== expected.capabilityPolicyHash) {
    problems.push('capabilityPolicyHash does not match the build manifest');
  }
  if (locator.signature && locator.signature.sig !== null) {
    problems.push('locator carries an inline signature; forge emits signature slots only');
  }
  return { ok: problems.length === 0, problems };
}

/**
 * Assert a locator round-trips.
 *
 * @throws {ForgeError} `FORGE_E_VERIFY_FAILED` when it does not.
 */
export function assertLocator(
  value: unknown,
  expected: { sha256: string; size: number; fileId: string; capabilityPolicyHash: string },
): void {
  const check = verifyLocator(value, expected);
  if (!check.ok) {
    throw new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `RVF locator does not match the build it belongs to: ${check.problems.join('; ')}.`,
      { problems: check.problems },
    );
  }
}

function readerSlot(target: BuildTarget, manifest: BuildManifest): ReaderSlot {
  return {
    schemaVersion: 1,
    type: 'reader-slot',
    target,
    runtimeProfile: manifest.runtime.runtimeProfile,
    rvmVersion: manifest.runtime.rvmVersion,
    rvmCommit: manifest.runtime.rvmCommit,
    status: 'placeholder',
  };
}

async function writeJson(root: string, relative: string, value: unknown): Promise<void> {
  const abs = join(root, relative);
  await mkdir(dirname(abs), { recursive: true });
  await writeFile(abs, canonicalJsonFile(value), 'utf8');
}
