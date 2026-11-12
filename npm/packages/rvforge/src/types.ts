/**
 * Shared types for the forge CLI: build targets, packaging modes, capability
 * policy, and the canonical build manifest shape (ADR-283 §7, ADR-291).
 */

import { ForgeError, ForgeErrorCode } from './errors';

/** A concrete OS + architecture forge can package for. */
export type BuildTarget =
  | 'windows-x64'
  | 'windows-arm64'
  | 'macos-x64'
  | 'macos-arm64'
  | 'macos-universal'
  | 'linux-x64'
  | 'linux-arm64'
  | 'rvm';

export const SUPPORTED_TARGETS: readonly BuildTarget[] = Object.freeze([
  'linux-arm64',
  'linux-x64',
  'macos-arm64',
  'macos-universal',
  'macos-x64',
  'rvm',
  'windows-arm64',
  'windows-x64',
] as const);

/** Friendly aliases accepted on the command line. */
const TARGET_ALIASES: Readonly<Record<string, BuildTarget>> = Object.freeze({
  win: 'windows-x64',
  windows: 'windows-x64',
  win32: 'windows-x64',
  mac: 'macos-universal',
  macos: 'macos-universal',
  darwin: 'macos-universal',
  linux: 'linux-x64',
  rvm: 'rvm',
});

/** The build host family a target must be packaged on. */
export type WorkerFamily = 'linux' | 'windows' | 'macos';

export const TARGET_WORKER: Readonly<Record<BuildTarget, WorkerFamily>> = Object.freeze({
  'windows-x64': 'windows',
  'windows-arm64': 'windows',
  'macos-x64': 'macos',
  'macos-arm64': 'macos',
  'macos-universal': 'macos',
  'linux-x64': 'linux',
  'linux-arm64': 'linux',
  rvm: 'linux',
});

/** Installer formats each target produces (requirements §3). */
export const TARGET_ARTIFACTS: Readonly<Record<BuildTarget, readonly string[]>> = Object.freeze({
  'windows-x64': ['exe', 'msi'],
  'windows-arm64': ['exe', 'msi'],
  'macos-x64': ['app', 'dmg'],
  'macos-arm64': ['app', 'dmg'],
  'macos-universal': ['app', 'dmg'],
  'linux-x64': ['deb', 'rpm', 'AppImage'],
  'linux-arm64': ['deb', 'rpm', 'AppImage'],
  rvm: ['rvf', 'rvm.img', 'rvm.efi', 'qemu.img', 'appliance.bundle'],
});

/**
 * Resolve a user-supplied target string to a canonical target.
 *
 * @throws {ForgeError} `FORGE_E_UNSUPPORTED_TARGET` when unrecognised.
 */
export function resolveTarget(raw: string): BuildTarget {
  const key = raw.trim().toLowerCase();
  const alias = TARGET_ALIASES[key];
  if (alias) return alias;
  if ((SUPPORTED_TARGETS as readonly string[]).includes(key)) return key as BuildTarget;
  throw new ForgeError(
    ForgeErrorCode.UNSUPPORTED_TARGET,
    `Unsupported build target "${raw}".`,
    { target: raw, supported: [...SUPPORTED_TARGETS] },
  );
}

/** Resolve, de-duplicate, and sort a target list so manifests stay canonical. */
export function resolveTargets(raw: readonly string[]): BuildTarget[] {
  const resolved = new Set(raw.map(resolveTarget));
  return [...resolved].sort();
}

/** Packaging mode (requirements §6, FR001–FR003). */
export type PackagingMode = 'embedded' | 'thin' | 'shared-reader';
export const PACKAGING_MODES: readonly PackagingMode[] = Object.freeze([
  'embedded',
  'shared-reader',
  'thin',
] as const);

/** Runtime preference (requirements §2.5, FR004). */
export type RuntimeProfile = 'wasm' | 'native' | 'microvm' | 'rvm';
export const RUNTIME_PROFILES: readonly RuntimeProfile[] = Object.freeze([
  'microvm',
  'native',
  'rvm',
  'wasm',
] as const);

/** Capability classes a policy may grant (requirements §2.6, RVM §6). */
export const CAPABILITY_CLASSES = Object.freeze([
  'devices',
  'filesystem',
  'memory',
  'models',
  'network',
  'state',
  'tools',
] as const);

export type CapabilityClass = (typeof CAPABILITY_CLASSES)[number];

/**
 * Default-deny capability policy: a class absent from `allow` grants nothing,
 * and an empty allowlist is the default for every class.
 */
export interface CapabilityPolicy {
  defaultDeny: true;
  allow: Partial<Record<CapabilityClass, string[]>>;
}

/** Application branding and identity (requirements §2.2, FR007). */
export interface AppMetadata {
  name: string;
  version: string;
  publisher: string;
  identifier: string;
  description?: string;
  icon?: string;
}

/**
 * A *reference* to a signing identity. Forge never reads, stores, or
 * transmits private key material — only the name of the key and the service
 * that holds it.
 */
export interface SigningRef {
  /** Where the key lives: `azure-kv`, `apple-notary`, `gpg`, `kms`, `none`. */
  provider: string;
  /** Opaque identifier the provider understands (key name, team ID, key ID). */
  keyRef: string;
  /** Optional certificate/identity name shown to the OS. */
  identity?: string;
}

export interface SigningRefs {
  windows?: SigningRef;
  macos?: SigningRef;
  linux?: SigningRef;
}

/** RVM compatibility contract embedded in every package (RVM §9). */
export interface RuntimeContract {
  rvmVersion: string;
  rvmCommit: string;
  runtimeProfile: RuntimeProfile;
  stateSchemaVersion: 1;
  witnessSchemaVersion: 1;
}

/** Identity of the input RVF, as read from its root manifest. */
export interface RvfIdentityRecord {
  /** SHA256 of the whole file — the invariant that must match everywhere. */
  sha256: string;
  /** Byte length of the `.rvf`. */
  size: number;
  /** 16-byte `FileIdentity.file_id` from the root manifest, lowercase hex. */
  fileId: string;
  /** Root manifest format version. */
  formatVersion: number;
  /** Signature algorithm id from the root manifest (0 = unsigned). */
  signatureAlgo: number;
  signaturePresent: boolean;
}

/** The canonical build manifest. Deterministic: no timestamps, no paths. */
export interface BuildManifest {
  schemaVersion: 1;
  generator: { name: string; version: string };
  app: AppMetadata;
  rvf: RvfIdentityRecord;
  targets: BuildTarget[];
  packaging: {
    mode: PackagingMode;
    updateChannel?: string;
    distributionUrl?: string;
  };
  runtime: RuntimeContract;
  capabilityPolicy: CapabilityPolicy;
  /** SHA256 over the canonical JSON of `capabilityPolicy`. */
  capabilityPolicyHash: string;
  signing: SigningRefs;
}

/** On-disk `forge.config.json` produced by `forge init`. */
export interface ForgeConfig {
  $schema?: string;
  app: AppMetadata;
  rvf: string;
  targets: BuildTarget[];
  packaging: { mode: PackagingMode; updateChannel?: string; distributionUrl?: string };
  runtime: { profile: RuntimeProfile; rvmVersion: string; rvmCommit: string };
  capabilityPolicy: CapabilityPolicy;
  signing: SigningRefs;
}

/** One file recorded in a provenance record. */
export interface ProvenanceFile {
  path: string;
  sha256: string;
  size: number;
}

/** Build provenance: what was built, from what, by whom (invariant 5). */
export interface ProvenanceRecord {
  schemaVersion: 1;
  /** RFC 3339 UTC. Lives here rather than in the manifest so the manifest stays byte-stable. */
  createdAt: string;
  rvfIdentity: string;
  /** SHA256 over the canonical JSON of the build manifest. */
  manifestSha256: string;
  /**
   * `sha256:<hex>` of the compatibility matrix that admitted this build, so a
   * past admission decision can be reconstructed (ADR-291 §2).
   */
  compatibilityMatrixRevision: string;
  builder: {
    tool: string;
    version: string;
    node: string;
    platform: string;
    sourceRevision: string | null;
  };
  /** `staged` means no installer was produced — see `packaging.status`. */
  status: 'staged' | 'built';
  packaging: {
    mode: PackagingMode;
    status: 'staged' | 'packaged';
    reason?: string;
  };
  signing: { mode: 'unsigned-development' | 'signed'; refs: SigningRefs };
  targets: BuildTarget[];
  files: ProvenanceFile[];
}
