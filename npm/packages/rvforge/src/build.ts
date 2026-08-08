/**
 * Local build.
 *
 * Produces everything that does not require a native packaging toolchain: the
 * canonical build manifest, one staged bundle per target, a software
 * inventory, SHA256 checksums, a provenance record, and a witness receipt.
 * When Tauri is not installed the result is explicitly labelled `staged` and
 * no installer is claimed.
 *
 * Four invariants shape the implementation:
 *
 * - **The embedded RVF is identical everywhere** (invariant 1). Embedded mode
 *   stages the same bytes into every target's bundle and then re-hashes each
 *   copy; a divergence fails the build.
 * - **Nothing executes the RVF** (invariant 2). Embedded mode copies bytes,
 *   thin mode writes a locator, and the reader slot is a placeholder
 *   descriptor rather than a binary. The file is read, not interpreted.
 * - **Unsupported combinations fail before any work** (ADR-291 §2). The
 *   compatibility gate runs before the RVF is even opened.
 * - **A failed build leaves no partially trusted artifact** (invariant 7).
 *   Everything is assembled in a temporary directory and moved into place in
 *   one step; a failure removes the temporary directory instead.
 */

import { execFile } from 'node:child_process';
import { mkdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

import { assertCompatible, compatibilityMatrixRevision } from './compat';
import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJsonFile, sha256File, sha256String } from './hash';
import { buildInventory, type SoftwareInventory } from './inventory';
import { manifestFileContents, manifestHash, buildManifest } from './manifest';
import {
  assertEmbeddedRvfIdentical,
  stageBundles,
  type BundleLayout,
} from './package-modes';
import { resolveTargets } from './types';
import { validateRvf, type ValidateResult } from './validate';
import { FORGE_PACKAGE_NAME, forgeVersion } from './version';
import { appendReceipt, type WitnessReceipt } from './witness';
import type {
  BuildManifest,
  PackagingMode,
  ProvenanceFile,
  ProvenanceRecord,
  ForgeConfig,
} from './types';

const execFileAsync = promisify(execFile);

export const MANIFEST_FILE = 'build-manifest.json';
export const INVENTORY_FILE = 'inventory.json';
export const PROVENANCE_FILE = 'provenance.json';
export const CHECKSUMS_FILE = 'checksums.txt';

export interface BuildOptions {
  config: ForgeConfig;
  /** Overrides `config.rvf`. */
  rvfPath?: string;
  /** Overrides `config.targets`. */
  targets?: readonly string[];
  /** Overrides `config.packaging.mode` — the CLI's `--mode`. */
  mode?: PackagingMode;
  outDir: string;
  /** Replace an existing output directory. */
  force?: boolean;
  /** Permit unsigned executable segments (development builds). */
  allowUnsigned?: boolean;
}

export interface BuildResult {
  outDir: string;
  manifest: BuildManifest;
  manifestSha256: string;
  provenance: ProvenanceRecord;
  validation: ValidateResult;
  inventory: SoftwareInventory;
  layout: BundleLayout;
  receipt: WitnessReceipt;
  /** True when a packaging toolchain produced installers. */
  packaged: boolean;
  toolchain: { tauri: string | null };
  warnings: string[];
}

export async function runBuild(options: BuildOptions): Promise<BuildResult> {
  const config = withMode(options.config, options.mode);
  const rvfPath = resolve(options.rvfPath ?? config.rvf);
  const outDir = resolve(options.outDir);
  const matrixRevision = compatibilityMatrixRevision();

  // Gate the combination before the RVF is opened: an unsupported request
  // should cost nothing (ADR-291 §2).
  assertCompatible({
    mode: config.packaging.mode,
    targets: resolveTargets(options.targets ?? config.targets),
    runtimeProfile: config.runtime.profile,
  });

  const validation = await validateRvf(rvfPath, { deep: true, allowUnsigned: options.allowUnsigned });

  const manifest = buildManifest({
    app: config.app,
    identity: validation.identity,
    targets: options.targets ?? config.targets,
    packaging: config.packaging,
    runtime: config.runtime,
    capabilityPolicy: config.capabilityPolicy,
    signing: config.signing,
  });

  await prepareOutDir(outDir, options.force);
  const stagingDir = `${outDir}.staging-${process.pid}`;

  try {
    await mkdir(stagingDir, { recursive: true });
    await writeText(stagingDir, MANIFEST_FILE, manifestFileContents(manifest));

    const staged = await stageBundles({ stagingDir, rvfPath, manifest });
    await assertEmbeddedRvfIdentical(stagingDir, staged.layout);

    const inventory = await buildInventory({
      stagingDir,
      manifest,
      validation,
      layout: staged.layout,
      files: [MANIFEST_FILE, ...staged.files],
      compatibilityMatrixRevision: matrixRevision,
    });
    await writeText(stagingDir, INVENTORY_FILE, canonicalJsonFile(inventory));

    const tauri = await detectTauri();
    const packaged = false; // Installer generation lands with the Tauri packaging layer.
    const warnings = [...validation.warnings, toolchainWarning(tauri)];

    const provenanceFiles = await hashFiles(stagingDir, [
      MANIFEST_FILE,
      INVENTORY_FILE,
      ...staged.files,
    ]);
    const provenance = await buildProvenance({
      manifest,
      files: provenanceFiles,
      packaged,
      tauri,
      matrixRevision,
    });
    await writeText(stagingDir, PROVENANCE_FILE, canonicalJsonFile(provenance));
    await writeText(stagingDir, CHECKSUMS_FILE, checksumsFile(provenanceFiles));

    // The witness chain starts here. `receipts.jsonl` is deliberately absent
    // from the provenance record: it is append-only and grows on every
    // subsequent verify, so listing its digest would make the build's own
    // provenance fail the moment the chain is extended.
    const receipt = await appendReceipt(stagingDir, {
      subject: manifest.rvf.sha256,
      event: 'build',
      outcome: 'pass',
      actor: { kind: 'builder', id: `${FORGE_PACKAGE_NAME}@${forgeVersion()}` },
      evidence: {
        manifestSha256: provenance.manifestSha256,
        packagingMode: manifest.packaging.mode,
        targets: manifest.targets.join(','),
        fileCount: provenanceFiles.length,
        status: provenance.status,
        compatibilityMatrixRevision: matrixRevision,
      },
    });

    await rename(stagingDir, outDir);

    return {
      outDir,
      manifest,
      manifestSha256: provenance.manifestSha256,
      provenance,
      validation,
      inventory,
      layout: staged.layout,
      receipt,
      packaged,
      toolchain: { tauri },
      warnings,
    };
  } catch (err) {
    await rm(stagingDir, { recursive: true, force: true });
    throw err;
  }
}

/**
 * Split `build`/`submit` positionals into an optional RVF path and targets.
 *
 * The grammar is `build [agent.rvf] [targets...]`, which is ambiguous for a
 * single operand. Resolving it by extension rather than by position means
 * `forge build linux-x64` reports an unsupported target if it is misspelled,
 * instead of an unreadable "no such file: linux-x64".
 */
export function splitBuildOperands(positionals: readonly string[]): {
  rvfPath?: string;
  targets?: string[];
} {
  const [first, ...rest] = positionals;
  if (first === undefined) return {};
  if (first.toLowerCase().endsWith('.rvf')) {
    return { rvfPath: first, ...(rest.length > 0 ? { targets: rest } : {}) };
  }
  return { targets: [...positionals] };
}

/** Apply a `--mode` override without mutating the caller's config. */
function withMode(config: ForgeConfig, mode?: PackagingMode): ForgeConfig {
  if (!mode || mode === config.packaging.mode) return config;
  return { ...config, packaging: { ...config.packaging, mode } };
}

function toolchainWarning(tauri: string | null): string {
  return tauri
    ? `Tauri ${tauri} is available but local installer generation is not wired up yet; the bundle is staged only.`
    : 'Tauri was not found on PATH; forge staged the bundle and produced no installers. ' +
        'Install the Tauri CLI, or use "forge submit" to build on hosted workers.';
}

async function prepareOutDir(outDir: string, force?: boolean): Promise<void> {
  try {
    await stat(outDir);
  } catch {
    await mkdir(dirname(outDir), { recursive: true });
    return;
  }
  if (!force) {
    throw new ForgeError(
      ForgeErrorCode.IO,
      `Output directory ${outDir} already exists. Pass --force to replace it.`,
      { outDir },
    );
  }
  await rm(outDir, { recursive: true, force: true });
}

async function buildProvenance(input: {
  manifest: BuildManifest;
  files: ProvenanceFile[];
  packaged: boolean;
  tauri: string | null;
  matrixRevision: string;
}): Promise<ProvenanceRecord> {
  const { manifest, files, packaged } = input;
  const signed = Object.keys(manifest.signing).length > 0;
  return {
    schemaVersion: 1,
    createdAt: new Date().toISOString(),
    rvfIdentity: manifest.rvf.sha256,
    manifestSha256: manifestHash(manifest),
    compatibilityMatrixRevision: input.matrixRevision,
    builder: {
      tool: FORGE_PACKAGE_NAME,
      version: forgeVersion(),
      node: process.version,
      platform: `${process.platform}-${process.arch}`,
      sourceRevision: await gitRevision(),
    },
    status: packaged ? 'built' : 'staged',
    packaging: {
      mode: manifest.packaging.mode,
      status: packaged ? 'packaged' : 'staged',
      ...(packaged ? {} : { reason: input.tauri ? 'installer-generation-not-implemented' : 'tauri-unavailable' }),
    },
    // Local builds are development builds: forge holds signing references, and
    // the signing operation itself happens on a worker with HSM/KMS access.
    signing: { mode: signed ? 'signed' : 'unsigned-development', refs: manifest.signing },
    targets: manifest.targets,
    files,
  };
}

async function hashFiles(root: string, relativePaths: readonly string[]): Promise<ProvenanceFile[]> {
  const entries: ProvenanceFile[] = [];
  for (const rel of [...new Set(relativePaths)].sort()) {
    const abs = join(root, rel);
    const info = await stat(abs);
    entries.push({ path: rel.split('\\').join('/'), sha256: await sha256File(abs), size: info.size });
  }
  return entries;
}

/** `sha256sum`-compatible output so standard tooling can check it. */
function checksumsFile(files: readonly ProvenanceFile[]): string {
  return `${files.map((f) => `${f.sha256}  ${f.path}`).join('\n')}\n`;
}

async function writeText(root: string, relative: string, contents: string): Promise<string> {
  const abs = join(root, relative);
  await mkdir(dirname(abs), { recursive: true });
  await writeFile(abs, contents, 'utf8');
  return relative;
}

/** Tauri CLI version, or null when it is not on PATH. */
async function detectTauri(): Promise<string | null> {
  for (const [cmd, args] of [
    ['cargo-tauri', ['--version']],
    ['tauri', ['--version']],
  ] as const) {
    try {
      const { stdout } = await execFileAsync(cmd, [...args], { timeout: 5_000 });
      return stdout.trim();
    } catch {
      continue;
    }
  }
  return null;
}

async function gitRevision(): Promise<string | null> {
  try {
    const { stdout } = await execFileAsync('git', ['rev-parse', 'HEAD'], { timeout: 5_000 });
    const rev = stdout.trim();
    return /^[0-9a-f]{40}$/.test(rev) ? rev : null;
  } catch {
    return null;
  }
}

/** SHA256 of a manifest file's exact bytes, for cross-checking a record. */
export const manifestFileHash = (contents: string): string => sha256String(contents);
