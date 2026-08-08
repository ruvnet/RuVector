/**
 * Compatibility enforcement against the published RVM compatibility matrix
 * (ADR-291 §2).
 *
 * Forge refuses to build a packaging-mode / target / runtime-profile
 * combination that is absent from the matrix, before a byte is uploaded and
 * before a worker is allocated. It never approximates, downgrades, or
 * substitutes a runtime to make an unsupported request succeed — instead the
 * rejection names the offending field and the nearest supported combination,
 * so the caller can decide rather than being silently retargeted.
 *
 * The matrix lives in `src/compatibility-matrix.json`, vendored from
 * `docs/research/rvf-forge/compatibility-matrix.json`. **The docs copy is the
 * single source of truth**: change it there first, then re-copy. `tests/
 * compat.test.ts` fails when the two diverge.
 */

import { ForgeError, ForgeErrorCode } from './errors';
import { sha256Canonical } from './hash';
import matrixJson from './compatibility-matrix.json';
import {
  SUPPORTED_TARGETS,
  type BuildTarget,
  type PackagingMode,
  type RuntimeProfile,
} from './types';

/** Runtime-profile keys as spelled in the matrix. */
export type MatrixRuntimeProfile = 'wasm' | 'os-isolation+wasm' | 'linux-microvm' | 'rvm-native';

/** Support state a matrix entry can carry. */
export type MatrixStatus = 'supported' | 'planned';

export interface MatrixPlatform {
  os: string;
  arch: string[];
  min: string;
}

export interface MatrixRuntimeProfileEntry {
  status: string;
  isolationClaim: string;
  platforms: MatrixPlatform[];
}

export interface CompatibilityMatrix {
  schemaVersion: number;
  updated: string;
  runtimeProfiles: Record<string, MatrixRuntimeProfileEntry>;
  packagingModes: Record<string, { status: string; targets: string[] }>;
  rvmOutputs: Record<string, { status: string }>;
  selectionOrder: string[];
}

export const COMPATIBILITY_MATRIX = matrixJson as unknown as CompatibilityMatrix;

/**
 * Hash-address of the matrix (ADR-291 §2: "the matrix is versioned and
 * hash-addressed... the provenance record names the matrix revision
 * consulted"), so a past admission decision can be reconstructed.
 */
export const compatibilityMatrixRevision = (): string =>
  `sha256:${sha256Canonical(COMPATIBILITY_MATRIX)}`;

/**
 * Forge's runtime-profile names to the matrix's.
 *
 * Forge speaks the four names in requirements §2.5 (WASM, native, Linux
 * microVM, RVM); the matrix spells the same four families by the isolation
 * they actually provide. `native` maps to `os-isolation+wasm` because that is
 * what hosted native acceleration is — OS isolation plus WASM, never
 * bare-metal isolation (ADR-285).
 */
export const RUNTIME_PROFILE_MATRIX_KEY: Readonly<Record<RuntimeProfile, MatrixRuntimeProfile>> =
  Object.freeze({
    wasm: 'wasm',
    native: 'os-isolation+wasm',
    microvm: 'linux-microvm',
    rvm: 'rvm-native',
  });

const MATRIX_KEY_RUNTIME_PROFILE: Readonly<Record<string, RuntimeProfile>> = Object.freeze(
  Object.fromEntries(
    Object.entries(RUNTIME_PROFILE_MATRIX_KEY).map(([forge, key]) => [key, forge as RuntimeProfile]),
  ),
);

/** OS and architectures each build target requires the matrix to list. */
export const TARGET_PLATFORM: Readonly<Record<BuildTarget, { os: string; arch: string[] }>> =
  Object.freeze({
    'windows-x64': { os: 'windows', arch: ['x64'] },
    'windows-arm64': { os: 'windows', arch: ['arm64'] },
    'macos-x64': { os: 'macos', arch: ['x64'] },
    'macos-arm64': { os: 'macos', arch: ['aarch64'] },
    'macos-universal': { os: 'macos', arch: ['x64', 'aarch64'] },
    'linux-x64': { os: 'linux', arch: ['x64'] },
    'linux-arm64': { os: 'linux', arch: ['aarch64'] },
    rvm: { os: 'rvm', arch: ['x64'] },
  });

/**
 * Installer formats a target produces, in the matrix's spelling. `rvm` has
 * none: it is served by `rvmOutputs`, not by a platform installer.
 */
export const TARGET_INSTALLER_FORMATS: Readonly<Record<BuildTarget, readonly string[]>> =
  Object.freeze({
    'windows-x64': ['exe', 'msi'],
    'windows-arm64': ['exe', 'msi'],
    'macos-x64': ['dmg'],
    'macos-arm64': ['dmg'],
    'macos-universal': ['dmg'],
    'linux-x64': ['deb', 'rpm', 'appimage'],
    'linux-arm64': ['deb', 'rpm', 'appimage'],
    rvm: [],
  });

/** The RVM output a `rvm` target produces; gated separately from installers. */
const RVM_TARGET_OUTPUT = 'Agent.rvf';

export interface CompatRequest {
  mode: PackagingMode;
  target: BuildTarget;
  runtimeProfile: RuntimeProfile;
}

export interface CompatSuggestion extends CompatRequest {}

export interface CompatCheck {
  request: CompatRequest;
  supported: boolean;
  /** Why the combination was refused; absent when supported. */
  reason?: string;
  /** Nearest combination the matrix does admit; absent when supported. */
  suggestion?: CompatSuggestion;
}

const isSupported = (entry: { status: string } | undefined): boolean =>
  entry?.status === 'supported';

const supportedModes = (): PackagingMode[] =>
  Object.entries(COMPATIBILITY_MATRIX.packagingModes)
    .filter(([, entry]) => isSupported(entry))
    .map(([name]) => name as PackagingMode)
    .sort();

/** Supported runtime profiles, strongest first per the matrix selection order. */
const supportedProfilesStrongestFirst = (): RuntimeProfile[] =>
  COMPATIBILITY_MATRIX.selectionOrder
    .filter((key) => isSupported(COMPATIBILITY_MATRIX.runtimeProfiles[key]))
    .map((key) => MATRIX_KEY_RUNTIME_PROFILE[key])
    .filter((profile): profile is RuntimeProfile => profile !== undefined);

function platformFor(profile: RuntimeProfile, target: BuildTarget): MatrixPlatform | undefined {
  const entry = COMPATIBILITY_MATRIX.runtimeProfiles[RUNTIME_PROFILE_MATRIX_KEY[profile]];
  if (!entry) return undefined;
  const wanted = TARGET_PLATFORM[target];
  return entry.platforms.find(
    (p) => p.os === wanted.os && wanted.arch.every((arch) => p.arch.includes(arch)),
  );
}

/** Targets a supported profile can actually build for. */
function targetsFor(profile: RuntimeProfile): BuildTarget[] {
  return SUPPORTED_TARGETS.filter((target) => platformFor(profile, target) !== undefined);
}

function modeAdmitsTarget(mode: PackagingMode, target: BuildTarget): boolean {
  const entry = COMPATIBILITY_MATRIX.packagingModes[mode];
  if (!entry) return false;
  const formats = TARGET_INSTALLER_FORMATS[target];
  if (formats.length === 0) return isSupported(COMPATIBILITY_MATRIX.rvmOutputs[RVM_TARGET_OUTPUT]);
  return formats.some((format) => entry.targets.includes(format));
}

/**
 * The nearest combination the matrix admits.
 *
 * Fields the matrix already accepts are preserved; only the unsupported ones
 * move, and each moves to the closest supported value — the same OS with a
 * different architecture before a different OS, the strongest runtime profile
 * that covers the requested target before a weaker one.
 */
export function closestSupported(request: CompatRequest): CompatSuggestion | undefined {
  const profiles = supportedProfilesStrongestFirst();
  if (profiles.length === 0) return undefined;

  const profile =
    profiles.find((candidate) => candidate === request.runtimeProfile) ??
    profiles.find((candidate) => platformFor(candidate, request.target) !== undefined) ??
    profiles[0];

  const candidates = targetsFor(profile);
  if (candidates.length === 0) return undefined;
  const wantedOs = TARGET_PLATFORM[request.target].os;
  const target =
    candidates.find((candidate) => candidate === request.target) ??
    candidates.find((candidate) => TARGET_PLATFORM[candidate].os === wantedOs) ??
    candidates[0];

  const modes = supportedModes().filter((candidate) => modeAdmitsTarget(candidate, target));
  if (modes.length === 0) return undefined;
  const mode = modes.find((candidate) => candidate === request.mode) ?? modes[0];

  return { mode, target, runtimeProfile: profile };
}

/** Check one combination against the matrix without throwing. */
export function checkCompatibility(request: CompatRequest): CompatCheck {
  const reason = refusalReason(request);
  if (!reason) return { request, supported: true };
  const suggestion = closestSupported(request);
  return { request, supported: false, reason, ...(suggestion ? { suggestion } : {}) };
}

function refusalReason(request: CompatRequest): string | undefined {
  const { mode, target, runtimeProfile } = request;

  const modeEntry = COMPATIBILITY_MATRIX.packagingModes[mode];
  if (!modeEntry) return `packaging mode "${mode}" is not in the compatibility matrix`;
  if (!isSupported(modeEntry)) {
    return `packaging mode "${mode}" is "${modeEntry.status}", not supported`;
  }

  const profileKey = RUNTIME_PROFILE_MATRIX_KEY[runtimeProfile];
  const profileEntry = COMPATIBILITY_MATRIX.runtimeProfiles[profileKey];
  if (!profileEntry) return `runtime profile "${runtimeProfile}" is not in the compatibility matrix`;
  if (!isSupported(profileEntry)) {
    return `runtime profile "${runtimeProfile}" (matrix "${profileKey}") is "${profileEntry.status}", not supported`;
  }

  if (!platformFor(runtimeProfile, target)) {
    const platform = TARGET_PLATFORM[target];
    return (
      `runtime profile "${runtimeProfile}" has no platform entry for target "${target}" ` +
      `(os "${platform.os}", arch ${platform.arch.join('+')})`
    );
  }

  if (!modeAdmitsTarget(mode, target)) {
    return `packaging mode "${mode}" produces none of the outputs target "${target}" needs`;
  }
  return undefined;
}

export interface AssertCompatibleInput {
  mode: PackagingMode;
  targets: readonly BuildTarget[];
  runtimeProfile: RuntimeProfile;
}

/**
 * Reject every target whose combination is absent from the matrix.
 *
 * @throws {ForgeError} `FORGE_E_UNSUPPORTED_TARGET`, carrying the reason and
 * the closest supported combination (ADR-291 §2).
 */
export function assertCompatible(input: AssertCompatibleInput): CompatCheck[] {
  const checks = input.targets.map((target) =>
    checkCompatibility({ mode: input.mode, target, runtimeProfile: input.runtimeProfile }),
  );
  const refused = checks.filter((check) => !check.supported);
  if (refused.length === 0) return checks;

  const first = refused[0];
  const suggestion = first.suggestion
    ? ` Closest supported combination: mode=${first.suggestion.mode} target=${first.suggestion.target} runtime=${first.suggestion.runtimeProfile}.`
    : ' No supported combination is available in this matrix revision.';

  throw new ForgeError(
    ForgeErrorCode.UNSUPPORTED_TARGET,
    `mode=${first.request.mode} target=${first.request.target} runtime=${first.request.runtimeProfile} ` +
      `is absent from the RVM compatibility matrix — ${first.reason}.${suggestion}`,
    {
      matrixRevision: compatibilityMatrixRevision(),
      refused: refused.map((check) => ({
        mode: check.request.mode,
        target: check.request.target,
        runtimeProfile: check.request.runtimeProfile,
        reason: check.reason,
      })),
      ...(first.suggestion ? { suggestion: first.suggestion } : {}),
    },
  );
}
