/**
 * Canonical build manifest generation (ADR-283 §7, RVM integration contract).
 *
 * The manifest is the thing a build is traceable to, so it is deterministic:
 * sorted keys, sorted target and capability lists, and no timestamps, paths,
 * or hostnames. The same logical build description always serialises to the
 * same bytes and therefore the same `manifestSha256`. Anything that varies per
 * run — the clock, the builder, the machine — belongs in the provenance
 * record instead.
 *
 * The manifest carries signing *references* only. Key material never enters
 * it, is never logged, and is never transmitted.
 */

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJson, canonicalJsonFile, sha256Canonical } from './hash';
import {
  CAPABILITY_CLASSES,
  PACKAGING_MODES,
  RUNTIME_PROFILES,
  resolveTargets,
  type AppMetadata,
  type BuildManifest,
  type CapabilityClass,
  type CapabilityPolicy,
  type PackagingMode,
  type RuntimeProfile,
  type RvfIdentityRecord,
  type SigningRefs,
} from './types';
import { FORGE_PACKAGE_NAME, forgeVersion } from './version';

/**
 * Runtime profiles present in the published RVM compatibility matrix.
 *
 * The MVP ships WASM execution only (ADR-283 §8); native acceleration, Linux
 * microVM, and bare-metal RVM are deferred. Forge rejects a combination that
 * is absent from the matrix rather than emitting a package whose runtime
 * contract nothing can honour (RVM §9).
 */
export const RVM_COMPATIBILITY_MATRIX: Readonly<Record<RuntimeProfile, boolean>> = Object.freeze({
  wasm: true,
  native: false,
  microvm: false,
  rvm: false,
});

const SEMVER = /^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)*$/;
const IDENTIFIER = /^[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?(\.[A-Za-z0-9]([A-Za-z0-9-]*[A-Za-z0-9])?)+$/;

export interface BuildManifestInput {
  app: AppMetadata;
  identity: RvfIdentityRecord;
  targets: readonly string[];
  packaging: { mode: PackagingMode; updateChannel?: string; distributionUrl?: string };
  runtime: { profile: RuntimeProfile; rvmVersion: string; rvmCommit: string };
  capabilityPolicy?: Partial<CapabilityPolicy>;
  signing?: SigningRefs;
  /** Overrides the detected generator version; used by tests for determinism. */
  generatorVersion?: string;
}

/** Default-deny: every capability class present, every allowlist empty. */
export function defaultCapabilityPolicy(): CapabilityPolicy {
  const allow: Partial<Record<CapabilityClass, string[]>> = {};
  for (const cls of CAPABILITY_CLASSES) allow[cls] = [];
  return { defaultDeny: true, allow };
}

/**
 * Normalise a policy into canonical form: `defaultDeny` is forced true, every
 * class is present, and each allowlist is de-duplicated and sorted.
 *
 * Forcing `defaultDeny` is deliberate. A policy that opts out of default-deny
 * is not a policy forge knows how to express, and silently accepting one would
 * put an undeclared-capability grant into a signed manifest.
 */
export function normalizeCapabilityPolicy(policy?: Partial<CapabilityPolicy>): CapabilityPolicy {
  const normalized = defaultCapabilityPolicy();
  if (!policy?.allow) return normalized;

  for (const [rawClass, rawGrants] of Object.entries(policy.allow)) {
    if (!(CAPABILITY_CLASSES as readonly string[]).includes(rawClass)) {
      throw new ForgeError(
        ForgeErrorCode.MANIFEST,
        `Unknown capability class "${rawClass}".`,
        { capabilityClass: rawClass, known: [...CAPABILITY_CLASSES] },
      );
    }
    if (!Array.isArray(rawGrants)) {
      throw new ForgeError(
        ForgeErrorCode.MANIFEST,
        `Capability class "${rawClass}" must map to an array of grants.`,
        { capabilityClass: rawClass },
      );
    }
    for (const grant of rawGrants) {
      if (typeof grant !== 'string' || grant.trim() === '') {
        throw new ForgeError(
          ForgeErrorCode.MANIFEST,
          `Capability class "${rawClass}" contains a non-string or empty grant.`,
          { capabilityClass: rawClass },
        );
      }
    }
    normalized.allow[rawClass as CapabilityClass] = [...new Set(rawGrants.map((g) => g.trim()))].sort();
  }
  return normalized;
}

/** SHA256 over the canonical JSON of a normalised capability policy. */
export const capabilityPolicyHash = (policy: CapabilityPolicy): string => sha256Canonical(policy);

/**
 * Build the canonical manifest.
 *
 * @throws {ForgeError} `FORGE_E_MANIFEST` for invalid metadata, an unsupported
 * runtime combination, or a malformed capability policy;
 * `FORGE_E_UNSUPPORTED_TARGET` for an unknown target.
 */
export function buildManifest(input: BuildManifestInput): BuildManifest {
  const app = validateApp(input.app);
  const targets = resolveTargets(input.targets);
  if (targets.length === 0) {
    throw new ForgeError(ForgeErrorCode.MANIFEST, 'At least one build target is required.');
  }

  if (!PACKAGING_MODES.includes(input.packaging.mode)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `Unknown packaging mode "${input.packaging.mode}".`,
      { mode: input.packaging.mode, known: [...PACKAGING_MODES] },
    );
  }
  if (input.packaging.mode === 'thin' && !input.packaging.distributionUrl) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      'Thin packaging requires packaging.distributionUrl — the reader has to know where to fetch the signed RVF.',
    );
  }

  validateRuntime(input.runtime);
  validateIdentity(input.identity);

  const capabilityPolicy = normalizeCapabilityPolicy(input.capabilityPolicy);

  return {
    schemaVersion: 1,
    generator: { name: FORGE_PACKAGE_NAME, version: input.generatorVersion ?? forgeVersion() },
    app,
    rvf: {
      sha256: input.identity.sha256,
      size: input.identity.size,
      fileId: input.identity.fileId,
      formatVersion: input.identity.formatVersion,
      signatureAlgo: input.identity.signatureAlgo,
      signaturePresent: input.identity.signaturePresent,
    },
    targets,
    packaging: {
      mode: input.packaging.mode,
      ...(input.packaging.updateChannel ? { updateChannel: input.packaging.updateChannel } : {}),
      ...(input.packaging.distributionUrl ? { distributionUrl: input.packaging.distributionUrl } : {}),
    },
    runtime: {
      rvmVersion: input.runtime.rvmVersion,
      rvmCommit: input.runtime.rvmCommit,
      runtimeProfile: input.runtime.profile,
      stateSchemaVersion: 1,
      witnessSchemaVersion: 1,
    },
    capabilityPolicy,
    capabilityPolicyHash: capabilityPolicyHash(capabilityPolicy),
    signing: normalizeSigning(input.signing),
  };
}

function validateApp(app: AppMetadata): AppMetadata {
  requireText(app.name, 'app.name');
  requireText(app.publisher, 'app.publisher');
  if (!SEMVER.test(app.version)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `app.version "${app.version}" is not a semantic version.`,
      { version: app.version },
    );
  }
  if (!IDENTIFIER.test(app.identifier)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `app.identifier "${app.identifier}" must be reverse-DNS, e.g. "io.ruv.myagent".`,
      { identifier: app.identifier },
    );
  }
  return {
    name: app.name.trim(),
    version: app.version.trim(),
    publisher: app.publisher.trim(),
    identifier: app.identifier.trim(),
    ...(app.description ? { description: app.description.trim() } : {}),
    ...(app.icon ? { icon: app.icon.trim() } : {}),
  };
}

function validateRuntime(runtime: BuildManifestInput['runtime']): void {
  if (!RUNTIME_PROFILES.includes(runtime.profile)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `Unknown runtime profile "${runtime.profile}".`,
      { profile: runtime.profile, known: [...RUNTIME_PROFILES] },
    );
  }
  if (!RVM_COMPATIBILITY_MATRIX[runtime.profile]) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `Runtime profile "${runtime.profile}" is absent from the published RVM compatibility matrix; this release packages "wasm" only.`,
      { profile: runtime.profile, supported: ['wasm'] },
    );
  }
  if (!SEMVER.test(runtime.rvmVersion)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `runtime.rvmVersion "${runtime.rvmVersion}" is not a semantic version.`,
      { rvmVersion: runtime.rvmVersion },
    );
  }
  requireText(runtime.rvmCommit, 'runtime.rvmCommit');
}

function validateIdentity(identity: RvfIdentityRecord): void {
  if (!/^[0-9a-f]{64}$/.test(identity.sha256)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      'rvf.sha256 must be a 64-character lowercase hex digest. Run validation with --deep to compute it.',
    );
  }
  if (!Number.isInteger(identity.size) || identity.size <= 0) {
    throw new ForgeError(ForgeErrorCode.MANIFEST, 'rvf.size must be a positive integer.');
  }
  if (!/^[0-9a-f]{32}$/.test(identity.fileId)) {
    throw new ForgeError(ForgeErrorCode.MANIFEST, 'rvf.fileId must be a 32-character lowercase hex value.');
  }
}

function normalizeSigning(signing?: SigningRefs): SigningRefs {
  if (!signing) return {};
  const out: SigningRefs = {};
  for (const platform of ['linux', 'macos', 'windows'] as const) {
    const ref = signing[platform];
    if (!ref) continue;
    requireText(ref.provider, `signing.${platform}.provider`);
    requireText(ref.keyRef, `signing.${platform}.keyRef`);
    out[platform] = {
      provider: ref.provider.trim(),
      keyRef: ref.keyRef.trim(),
      ...(ref.identity ? { identity: ref.identity.trim() } : {}),
    };
  }
  return out;
}

function requireText(value: unknown, field: string): void {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new ForgeError(ForgeErrorCode.MANIFEST, `${field} is required and must be a non-empty string.`, {
      field,
    });
  }
}

/** Canonical JSON for a manifest (no trailing newline). */
export const serializeManifest = (manifest: BuildManifest): string => canonicalJson(manifest);

/** Exact bytes forge writes to `build-manifest.json`. */
export const manifestFileContents = (manifest: BuildManifest): string => canonicalJsonFile(manifest);

/** SHA256 over the canonical manifest encoding — the build's manifest identity. */
export const manifestHash = (manifest: BuildManifest): string => sha256Canonical(manifest);
