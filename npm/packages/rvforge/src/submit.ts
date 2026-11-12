/**
 * Submit a build to the hosted service.
 *
 * The RVF is validated locally first (requirements §5.3) and the estimate is
 * produced before anything leaves the machine (§5.6). The upload itself is
 * gated on explicit confirmation so an unattended run cannot ship a
 * confidential RVF to a remote worker by accident.
 */

import { resolve } from 'node:path';

import { ForgeApiClient, type BuildRecord, type ForgeApiOptions } from './api';
import { assertCompatible } from './compat';
import { ForgeError, ForgeErrorCode } from './errors';
import { buildManifest } from './manifest';
import { estimateBuild, type BuildEstimate } from './estimate';
import { validateRvf, type ValidateResult } from './validate';
import { resolveTargets, type BuildManifest, type ForgeConfig } from './types';

export interface SubmitOptions {
  config: ForgeConfig;
  rvfPath?: string;
  targets?: readonly string[];
  /** Required for the upload to proceed; the CLI sets it from `--yes`. */
  confirmed?: boolean;
  /** Assume warm dependency caches when estimating. */
  cached?: boolean;
  allowUnsigned?: boolean;
  api?: ForgeApiOptions;
}

export interface SubmitPlan {
  manifest: BuildManifest;
  estimate: BuildEstimate;
  validation: ValidateResult;
  endpoint: string;
}

export interface SubmitResult extends SubmitPlan {
  build: BuildRecord;
}

/** Validate, build the manifest, and estimate — without contacting the service. */
export async function planSubmit(options: SubmitOptions): Promise<SubmitPlan> {
  const { config } = options;
  const rvfPath = resolve(options.rvfPath ?? config.rvf);

  // Reject unsupported combinations before the RVF is read, let alone
  // uploaded, and before a worker is allocated (ADR-291 §2).
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

  const estimate = estimateBuild({
    targets: manifest.targets,
    packagingMode: manifest.packaging.mode,
    rvfSize: manifest.rvf.size,
    cached: options.cached,
  });

  return { manifest, estimate, validation, endpoint: new ForgeApiClient(options.api).endpoint };
}

/**
 * Submit the build.
 *
 * @throws {ForgeError} `FORGE_E_USAGE` without confirmation, `FORGE_E_AUTH`
 * without credentials, `FORGE_E_NETWORK` when the service is unreachable.
 */
export async function runSubmit(options: SubmitOptions): Promise<SubmitResult> {
  const plan = await planSubmit(options);

  if (!options.confirmed) {
    throw new ForgeError(
      ForgeErrorCode.USAGE,
      'Submission uploads the RVF to the hosted build service. Re-run with --yes to confirm.',
      {
        targets: plan.manifest.targets,
        estimatedCostUsd: plan.estimate.totalCostUsd,
        estimatedMinutes: plan.estimate.wallClockMinutes,
      },
    );
  }

  const client = new ForgeApiClient(options.api);
  if (!client.hasToken) {
    throw new ForgeError(
      ForgeErrorCode.AUTH,
      'No build-service credentials. Set FORGE_API_TOKEN in the environment — never pass a token as an argument.',
      { endpoint: client.endpoint },
    );
  }

  const build = await client.submitBuild({
    manifest: plan.manifest,
    rvfSha256: plan.manifest.rvf.sha256,
    rvfSizeBytes: plan.manifest.rvf.size,
  });

  return { ...plan, build };
}
