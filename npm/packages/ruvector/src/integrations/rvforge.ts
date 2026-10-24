import { IntegrationError, loadOptionalPackage } from './optional-package';
import { copyFile, mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const PACKAGE = '@ruvector/rvforge';

export interface RvfArtifactValidationOptions {
  /** Trusted whole-file SHA-256, obtained separately from the file being checked. */
  expectedSha256: string;
  /** Development-only opt-in for unsigned executable segments. Defaults to false. */
  allowUnsigned?: boolean;
}

export interface RvfArtifactValidation {
  path: string;
  size: number;
  sha256: string;
  fileId: string;
  formatVersion: number;
  /** Presence only: this does not authenticate a signer or verify a signature. */
  signaturePresent: boolean;
  segmentCount: number;
  expectedSha256Matched: true;
  warnings: string[];
}

/**
 * Inspect an RVF with the published RVForge validator and hash its entire file.
 * Install `@ruvector/rvforge` as a development dependency to use this build-time
 * extension. Importing RuVector does not load RVForge or install anything.
 *
 * RVForge checks structure and the root CRC. It does not authenticate signatures
 * or check every segment's payload hash. Requires a separately trusted SHA-256 to
 * reject payload changes as well. This operation does not execute RVF content,
 * create installers, or publish an artifact. Upstream validation errors propagate.
 */
export async function validateRvfArtifact(
  path: string,
  options: RvfArtifactValidationOptions,
): Promise<RvfArtifactValidation> {
  if (typeof path !== 'string' || path.trim().length === 0) {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE, 'An RVF file path is required.');
  }
  const expected = options?.expectedSha256;
  if (typeof expected !== 'string' || !/^[a-fA-F0-9]{64}$/.test(expected)) {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE,
      'expectedSha256 must contain exactly 64 hexadecimal characters.');
  }
  if (options.allowUnsigned !== undefined && typeof options.allowUnsigned !== 'boolean') {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE, 'allowUnsigned must be a boolean.');
  }
  const forge = await loadOptionalPackage(PACKAGE, PACKAGE, true);
  if (typeof forge.validateRvf !== 'function') {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE,
      'Installed @ruvector/rvforge must expose validateRvf().');
  }
  const result = await forge.validateRvf(path, { deep: true, allowUnsigned: options.allowUnsigned ?? false });
  if (result?.ok !== true || result.deep !== true || !/^[a-f0-9]{64}$/.test(result.identity?.sha256 ?? '') ||
      typeof result.identity?.fileId !== 'string' || typeof result.identity?.formatVersion !== 'number' ||
      typeof result.identity?.signaturePresent !== 'boolean' || !Number.isSafeInteger(result.size) ||
      result.size < 0 || !Number.isSafeInteger(result.segments?.count) || result.segments.count < 0 ||
      typeof result.path !== 'string' || !Array.isArray(result.warnings) ||
      result.warnings.some((warning: unknown) => typeof warning !== 'string')) {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE,
      'Installed @ruvector/rvforge returned an unsupported deep-validation result.');
  }
  if (result.identity.sha256 !== expected.toLowerCase()) {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE,
      `RVF SHA-256 mismatch: expected ${expected.toLowerCase()}, received ${result.identity.sha256}.`);
  }
  return {
    path: result.path,
    size: result.size,
    sha256: result.identity.sha256,
    fileId: result.identity.fileId,
    formatVersion: result.identity.formatVersion,
    signaturePresent: result.identity.signaturePresent,
    segmentCount: result.segments.count,
    expectedSha256Matched: true,
    warnings: [...result.warnings],
  };
}

export interface RvfArtifactStageOptions extends RvfArtifactValidationOptions {
  rvfPath: string;
  /** A new directory; existing output is never overwritten by this adapter. */
  outDir: string;
  app: { name: string; version: string; publisher: string; identifier: string };
  targets: readonly string[];
  runtime: { profile: 'wasm' | 'native' | 'microvm' | 'rvm'; rvmVersion: string; rvmCommit: string };
}

export interface RvfArtifactStageResult {
  outDir: string;
  sha256: string;
  manifestSha256: string;
  /** RVForge 0.2.0 stages bundles with reader placeholders; it creates no installers. */
  packaged: boolean;
  status: 'staged' | 'packaged';
  bundles: Array<{ target: string; rvfPath: string; readerSlotPath: string }>;
  warnings: string[];
}

/** Stage identical embedded RVF bundles, manifests and provenance with RVForge.
 * No upload or release is performed. A private, hash-checked input snapshot
 * prevents source changes during packaging from substituting unapproved bytes. */
export async function stageRvfArtifact(options: RvfArtifactStageOptions): Promise<RvfArtifactStageResult> {
  await validateRvfArtifact(options?.rvfPath, options);
  if (typeof options.outDir !== 'string' || !options.outDir.trim() ||
      !Array.isArray(options.targets) || !options.targets.length ||
      options.targets.some(target => typeof target !== 'string' || !target.trim())) {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE, 'An output directory and build targets are required.');
  }
  if (['name', 'version', 'publisher', 'identifier'].some(key => {
    const value = options.app?.[key as keyof RvfArtifactStageOptions['app']];
    return typeof value !== 'string' || !value.trim();
  }) || !options.runtime || ['profile', 'rvmVersion', 'rvmCommit'].some(key => {
    const value = options.runtime?.[key as keyof RvfArtifactStageOptions['runtime']];
    return typeof value !== 'string' || !value.trim();
  })) {
    throw new IntegrationError('ERR_INTEGRATION_INVALID', PACKAGE, 'Explicit application and runtime metadata are required.');
  }
  const forge = await loadOptionalPackage(PACKAGE, PACKAGE, true);
  if (typeof forge.runBuild !== 'function' || typeof forge.defaultConfig !== 'function' ||
      typeof forge.validateConfig !== 'function') {
    throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE,
      'Installed @ruvector/rvforge must expose runBuild(), defaultConfig() and validateConfig().');
  }
  // Use RVForge's configuration validator, including app and runtime metadata.
  const config = forge.validateConfig(forge.defaultConfig({
    app: options.app, rvf: options.rvfPath, targets: options.targets,
    packaging: { mode: 'embedded' }, runtime: options.runtime,
  }));
  const scratch = await mkdtemp(join(tmpdir(), 'ruvector-rvforge-'));
  try {
    const snapshot = join(scratch, 'artifact.rvf');
    await copyFile(options.rvfPath, snapshot);
    await validateRvfArtifact(snapshot, options);
    const result = await forge.runBuild({
      config, rvfPath: snapshot, outDir: options.outDir,
      mode: 'embedded', force: false, allowUnsigned: options.allowUnsigned ?? false,
    });
    if (result?.validation?.identity?.sha256 !== options.expectedSha256.toLowerCase() ||
        typeof result.packaged !== 'boolean' || typeof result.outDir !== 'string' ||
        !/^[a-f0-9]{64}$/.test(result.manifestSha256 ?? '') ||
        !Array.isArray(result.layout?.bundles) || !Array.isArray(result.warnings) ||
        result.layout.bundles.some((bundle: any) => typeof bundle.target !== 'string' ||
          typeof bundle.rvfPath !== 'string' || typeof bundle.readerSlotPath !== 'string')) {
      throw new IntegrationError('ERR_INTEGRATION_API', PACKAGE, 'RVForge returned an unsupported staging result.');
    }
    return {
      outDir: result.outDir, sha256: result.validation.identity.sha256,
      manifestSha256: result.manifestSha256, packaged: result.packaged,
      status: result.packaged ? 'packaged' : 'staged',
      bundles: result.layout.bundles.map((bundle: any) => ({
        target: bundle.target, rvfPath: bundle.rvfPath, readerSlotPath: bundle.readerSlotPath,
      })),
      warnings: [...result.warnings],
    };
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }
}
