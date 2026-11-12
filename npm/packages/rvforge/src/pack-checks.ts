/**
 * The ADR-294 §4 `pack` validation list — one function per line of it.
 *
 * Each check is total and independent: it returns a verdict rather than
 * throwing, so `pack` can run all ten and report every problem at once. Only
 * {@link assertNoFailures} raises, once, with the aggregate.
 *
 * A check has three verdicts, and the difference between two of them is the
 * point. `fail` means the package is not publishable. `warn` means it is
 * publishable but a human has to look at it — ADR-294 §8 review triggers land
 * here, and treating them as failures would push publishers toward describing
 * capabilities less precisely to avoid the flag.
 */

import { basename } from 'node:path';

import { checkCompatibility, compatibilityMatrixRevision } from './compat';
import { ForgeError, ForgeErrorCode } from './errors';
import type { RvforgeProject } from './project';
import { RUNTIME_PROFILES, resolveTargets, type RuntimeProfile } from './types';
import type { ValidateResult } from './validate';

export type CheckStatus = 'pass' | 'fail' | 'warn';

export interface PackCheck {
  id: string;
  title: string;
  status: CheckStatus;
  detail: string;
  /** Error code the aggregate failure carries when this check fails. */
  code?: ForgeErrorCode;
}

export function structureCheck(validation: ValidateResult): PackCheck {
  return {
    id: 'rvf-structure',
    title: 'RVF structure',
    status: 'pass',
    detail:
      `format v${validation.root.formatVersion}, ${validation.segments.count} segment(s), ` +
      `sha256 ${validation.identity.sha256}`,
  };
}

export function signatureCheck(validation: ValidateResult, allowUnsigned: boolean): PackCheck {
  if (validation.root.signaturePresent) {
    return {
      id: 'publisher-signature',
      title: 'Publisher signature',
      status: 'pass',
      detail: `root manifest signed (algo ${validation.root.signatureAlgo})`,
    };
  }
  // Reaching here with executable segments means --allow-unsigned was passed;
  // validation would already have refused otherwise.
  return {
    id: 'publisher-signature',
    title: 'Publisher signature',
    status: allowUnsigned ? 'warn' : 'fail',
    detail: allowUnsigned
      ? 'root manifest is unsigned; --allow-unsigned makes this a development pack, not a release'
      : 'root manifest carries no signature; sign the RVF before packing',
    code: ForgeErrorCode.UNSIGNED_SEGMENT,
  };
}

export function executableSegmentCheck(validation: ValidateResult): PackCheck {
  const { executableCount } = validation.segments;
  return {
    id: 'executable-segments',
    title: 'Executable segments',
    status: 'pass',
    detail:
      executableCount === 0
        ? 'none present'
        : `${executableCount} present, covered by the root-manifest signature`,
  };
}

export function modelProvenanceCheck(project: RvforgeProject): PackCheck {
  const { location, digests } = project.modelManifest;
  const missing = location !== 'byo' && digests.length === 0;
  return {
    id: 'model-provenance',
    title: 'Model provenance',
    status: missing ? 'fail' : 'pass',
    detail: missing
      ? `modelManifest.location is "${location}" but no digests are declared; a model that ships ` +
        'with the package must be identified by digest'
      : `location "${location}", ${digests.length} digest(s)`,
    code: ForgeErrorCode.MANIFEST,
  };
}

export function capabilityPolicyCheck(project: RvforgeProject, triggers: readonly string[]): PackCheck {
  const { requests, denials } = project.capabilities;
  if (requests.length === 0 && denials.length === 0) {
    return {
      id: 'capability-policy',
      title: 'Capability policy',
      status: 'fail',
      detail: 'no capability contract declared; default-deny still has to be written down to be shown',
      code: ForgeErrorCode.POLICY,
    };
  }
  const overlap = requests.filter((request) => denials.includes(request.class));
  if (overlap.length > 0) {
    return {
      id: 'capability-policy',
      title: 'Capability policy',
      status: 'fail',
      detail: `class(es) both requested and denied: ${overlap.map((r) => r.class).join(', ')}`,
      code: ForgeErrorCode.POLICY,
    };
  }
  return {
    id: 'capability-policy',
    title: 'Capability policy',
    status: triggers.length === 0 ? 'pass' : 'warn',
    detail:
      `default-deny, ${requests.length} request(s), ${denials.length} denial(s)` +
      (triggers.length === 0 ? '' : `; manual review required: ${triggers.join(', ')}`),
  };
}

export function runtimeCompatibilityCheck(project: RvforgeProject): PackCheck {
  const profiles = project.runtimeRequirements.profiles;
  const unknown = profiles.filter((p) => !(RUNTIME_PROFILES as readonly string[]).includes(p));
  if (profiles.length === 0 || unknown.length > 0) {
    return {
      id: 'runtime-compatibility',
      title: 'Runtime compatibility',
      status: 'fail',
      detail:
        profiles.length === 0
          ? 'runtimeRequirements.profiles is empty'
          : `unknown runtime profile(s): ${unknown.join(', ')}`,
      code: ForgeErrorCode.UNSUPPORTED_TARGET,
    };
  }

  const targets = resolveTargets(project.runtimeRequirements.systems);
  const refused: string[] = [];
  for (const profile of profiles as RuntimeProfile[]) {
    for (const target of targets) {
      const check = checkCompatibility({ mode: 'embedded', target, runtimeProfile: profile });
      if (!check.supported) refused.push(`${profile}/${target}: ${check.reason}`);
    }
  }
  return {
    id: 'runtime-compatibility',
    title: 'Runtime compatibility',
    status: refused.length === 0 ? 'pass' : 'fail',
    detail:
      refused.length === 0
        ? `${profiles.join(', ')} on ${targets.join(', ')} (matrix ${compatibilityMatrixRevision()})`
        : `absent from the compatibility matrix — ${refused.join('; ')}`,
    code: ForgeErrorCode.UNSUPPORTED_TARGET,
  };
}

export function memoryRequirementsCheck(project: RvforgeProject): PackCheck {
  const declared = project.runtimeRequirements.memoryMiB;
  const fromCapability = project.capabilities.requests.find((r) => r.class === 'memory');
  if (typeof declared === 'number' && Number.isFinite(declared) && declared > 0) {
    return {
      id: 'memory-requirements',
      title: 'Memory requirements',
      status: 'pass',
      detail: `${declared} MiB declared${fromCapability ? ` (capability scope "${fromCapability.scope}")` : ''}`,
    };
  }
  return {
    id: 'memory-requirements',
    title: 'Memory requirements',
    status: 'fail',
    detail:
      'runtimeRequirements.memoryMiB is not a positive number; a host cannot admit an agent whose ' +
      'working set is undeclared',
    code: ForgeErrorCode.MANIFEST,
  };
}

export function externalServicesCheck(project: RvforgeProject): PackCheck {
  const services = project.externalServices;
  const networkDenied = project.capabilities.denials.includes('network');
  if (services.length > 0 && networkDenied) {
    return {
      id: 'external-services',
      title: 'External services',
      status: 'fail',
      detail: `${services.length} external service(s) declared while the network class is denied: ${services.join(', ')}`,
      code: ForgeErrorCode.POLICY,
    };
  }
  return {
    id: 'external-services',
    title: 'External services',
    status: 'pass',
    detail: services.length === 0 ? 'none' : services.join(', '),
  };
}

export function licenseCheck(project: RvforgeProject): PackCheck {
  const license = project.license;
  if (typeof license !== 'string' || license.trim() === '') {
    return {
      id: 'license-compatibility',
      title: 'License compatibility',
      status: 'fail',
      detail: 'no license declared; a listing without a license grants a user no rights to run it',
      code: ForgeErrorCode.LICENSE,
    };
  }
  return {
    id: 'license-compatibility',
    title: 'License compatibility',
    status: 'pass',
    detail: license.trim(),
  };
}

export function assertNoFailures(checks: readonly PackCheck[], rvfPath: string): void {
  const failures = checks.filter((check) => check.status === 'fail');
  if (failures.length === 0) return;
  throw new ForgeError(
    failures[0].code ?? ForgeErrorCode.MANIFEST,
    `pack failed ${failures.length} of ${checks.length} checks for ${basename(rvfPath)}: ` +
      failures.map((f) => `${f.id} (${f.detail})`).join('; '),
    { failures: failures.map((f) => ({ id: f.id, detail: f.detail })) },
  );
}
