/**
 * `rvforge publish agent.rvf` — append a signed release to a local registry
 * (requirements P4 "Publish", ADR-294 §4/§9, registry data model v0.1).
 *
 * Publishing is four appends and a signature:
 *
 * 1. the `CapabilityManifest`, `Release`, software inventory, and pack
 *    provenance land at their content addresses under `objects/sha256/`;
 * 2. the release is signed with the publisher's Ed25519 key over its
 *    `releaseId`, and the key must already be registered to the publisher —
 *    an unregistered key is refused rather than silently enrolled;
 * 3. the release is appended to `packages/<publisher>/<name>/releases.jsonl`
 *    with its predecessor resolved from the current head, so lineage is a
 *    chain rather than a set;
 * 4. a `TransparencyLogEntry` extends the log and its tree head, and a
 *    `WitnessReceipt` chains onto the registry's receipt file.
 *
 * The target is a directory, not a service: the local registry is the MVP
 * implementation target named by the data model, and `crates/rvforge-registry`
 * owns the hosted one.
 *
 * Key material is read only by `src/keys.ts` and never reaches a log line, an
 * error message, or a registry object — only the key's derived `keyId` and its
 * public half do.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJsonFile } from './hash';
import { loadPublisherKey, signObjectId, verifyObjectId, type PublisherKey } from './keys';
import { runPack, type PackResult, type SecurityProfile } from './pack';
import { publisherIdFor, type RvforgeProject } from './project';
import {
  DEFAULT_REGISTRY_DIR,
  appendLogEntry,
  appendRelease,
  appendWitnessReceipt,
  digestOf,
  getObject,
  initRegistry,
  objectId,
  putObject,
  readReleaseIndex,
  publisherPath,
  type CapabilityManifest,
  type PublisherRecord,
  type Release,
  type TransparencyLogEntry,
  type TreeHead,
} from './registry';
import { resolveTargets, type BuildTarget } from './types';
import { FORGE_PACKAGE_NAME, forgeVersion } from './version';
import { type WitnessReceipt } from './witness';

export interface PublishOptions {
  rvfPath: string;
  project: RvforgeProject;
  /** Local registry directory; defaults to `~/.rvforge/registry`. */
  registryDir?: string;
  /** Path to the Ed25519 key file; falls back to `publisher.keyFile`. */
  keyFile?: string;
  allowUnsigned?: boolean;
  /** Overrides the clock; used by tests for deterministic ids. */
  publishedAt?: string;
}

export interface PublishResult {
  registryDir: string;
  pack: PackResult;
  /**
   * The content address of the publisher record — what `release.package`
   * names, and the directory the package index lives under.
   */
  publisherId: string;
  /**
   * The publisher's stable identity, derived from its display name and
   * evidence. Survives key rotation, and is where the record pointer is filed;
   * see {@link publisherIdFor}.
   */
  publisherIdentity: string;
  publisherRecord: { id: string; created: boolean };
  keyId: string;
  capabilityManifest: CapabilityManifest;
  release: Release;
  predecessor: string | null;
  objects: Array<{ id: string; path: string; type: string }>;
  releasesPath: string;
  logEntry: TransparencyLogEntry;
  treeHead: TreeHead;
  receipt: WitnessReceipt;
  summary: PublishSummary;
}

/** The block P4 says a publisher sees before the release goes out. */
export interface PublishSummary {
  validation: 'passed';
  securityProfile: SecurityProfile;
  /** Null, never a number: an evaluation forge did not run has no score. */
  evaluationScore: null;
  evaluationNote: string;
  supportedRuntimes: string[];
  supportedSystems: string[];
  requestedCapabilities: string[];
  networkAccess: string;
  manualReviewTriggers: string[];
  readyToPublish: boolean;
}

/**
 * Publish a release into a local registry.
 *
 * @throws {ForgeError} whatever `pack` raises for a validation failure;
 * `FORGE_E_KEY` for a missing, over-permissioned, or unregistered key;
 * `FORGE_E_LINEAGE` when the predecessor chain or the version conflicts.
 */
export async function runPublish(options: PublishOptions): Promise<PublishResult> {
  const registryDir = options.registryDir ?? DEFAULT_REGISTRY_DIR;
  const keyFile = options.keyFile ?? options.project.publisher.keyFile;
  if (!keyFile) {
    throw new ForgeError(
      ForgeErrorCode.KEY,
      'No signing key. Pass --key-file <path>, or set publisher.keyFile in rvforge.json. ' +
        'Generate one with "rvforge init --keygen".',
    );
  }

  const pack = await runPack({
    rvfPath: options.rvfPath,
    project: options.project,
    allowUnsigned: options.allowUnsigned,
    publishedAt: options.publishedAt,
  });

  const key = await loadPublisherKey(keyFile);
  await initRegistry(registryDir);

  // Two different names for the publisher, and the difference matters. The
  // *identity* is stable across key rotation and is what the pointer file is
  // filed under; the *id* is the content address of the record that identity
  // currently resolves to, and is what a release names — because the model's
  // identity rule makes `publisherId` the hash of the record's own content.
  const publisherIdentity = publisherIdFor(options.project.publisher);
  const publishedAt = pack.release.publishedAt;
  const publisher = await resolvePublisher(
    registryDir,
    options.project,
    publisherIdentity,
    key,
    publishedAt,
  );
  const publisherId = publisher.id;

  const objects: PublishResult['objects'] = [];
  if (publisher.created) {
    objects.push({ id: publisher.id, path: publisher.path, type: 'publisher' });
    // The publisher record is logged too: ADR-294 §9 makes the transparency
    // log cover releases, revocations, *and* publisher records, so a key that
    // appears out of nowhere is visible rather than inferred.
    await appendLogEntry(registryDir, {
      object: publisher.id,
      objectType: 'publisher',
      timestamp: publishedAt,
    });
  }

  const capabilityManifest = sign(pack.capabilityManifest, key, 'manifestId');
  objects.push({
    id: capabilityManifest.manifestId,
    path: (await putObject(registryDir, capabilityManifest.manifestId, capabilityManifest)).path,
    type: 'capability-manifest',
  });

  const inventoryId = pack.release.softwareInventory;
  objects.push({
    id: inventoryId,
    path: (await putObject(registryDir, inventoryId, pack.softwareInventory)).path,
    type: 'software-inventory',
  });
  objects.push({
    id: pack.provenance.id,
    path: (await putObject(registryDir, pack.provenance.id, pack.provenance.record)).path,
    type: 'pack-provenance',
  });

  // The draft release carries `predecessor: null` and the publisher's stable
  // identity, because `pack` has no registry to resolve either against.
  // Binding both changes the content, so the id is recomputed before signing.
  const index = await readReleaseIndex(registryDir, publisherId, options.project.listing.name);
  const predecessor = index.length === 0 ? null : index[index.length - 1].releaseId;
  const rebound = {
    ...pack.release,
    package: { ...pack.release.package, publisherId },
    predecessor,
  };
  const release = sign({ ...rebound, releaseId: objectId(rebound) }, key, 'releaseId');

  if (!verifyObjectId(key.publicKey, release.releaseId, release.signatures[0].sig)) {
    throw new ForgeError(
      ForgeErrorCode.KEY,
      'The release signature does not verify against the key that produced it.',
      { keyId: key.keyId },
    );
  }

  objects.push({
    id: release.releaseId,
    path: (await putObject(registryDir, release.releaseId, release)).path,
    type: 'release',
  });

  const appended = await appendRelease(registryDir, release);
  const { entry, head } = await appendLogEntry(registryDir, {
    object: release.releaseId,
    objectType: 'release',
    timestamp: publishedAt,
  });

  // `evidence` is a single `details` string, not a bag of fields: that is the
  // shape registry-model.md gives a receipt, and the shape the Reader parses.
  const receipt = await appendWitnessReceipt(registryDir, {
    subject: release.rvfIdentity,
    event: 'publish',
    outcome: 'pass',
    actor: { kind: 'registry', id: `${FORGE_PACKAGE_NAME}@${forgeVersion()}` },
    evidence: {
      details:
        `release ${release.releaseId} (${release.version} of ${release.package.name}) accepted ` +
        `at log index ${entry.index}, tree head ${head.rootHash}; manifest ` +
        `${capabilityManifest.manifestId}, predecessor ${predecessor ?? 'none'}, ` +
        `signed by ${key.keyId}, security profile ${pack.securityProfile}`,
    },
    timestamp: options.publishedAt,
  });

  return {
    registryDir,
    pack,
    publisherId,
    publisherIdentity,
    publisherRecord: { id: publisher.id, created: publisher.created },
    keyId: key.keyId,
    capabilityManifest,
    release,
    predecessor,
    objects,
    releasesPath: appended.path,
    logEntry: entry,
    treeHead: head,
    receipt,
    summary: publishSummary(options.project, pack),
  };
}

/** Attach the publisher signature over an object's content id. */
function sign<T extends { signatures: Array<{ keyId: string; role: 'publisher' | 'registry'; sig: string }> }>(
  object: T,
  key: PublisherKey,
  idField: 'manifestId' | 'releaseId',
): T {
  const id = (object as unknown as Record<string, string>)[idField];
  return {
    ...object,
    signatures: [{ keyId: key.keyId, role: 'publisher' as const, sig: signObjectId(key, id) }],
  };
}

interface ResolvedPublisher {
  id: string;
  path: string;
  created: boolean;
  record: PublisherRecord;
}

/**
 * Find the publisher record, or register it on first publish.
 *
 * `identity` is the stable, key-independent name from {@link publisherIdFor};
 * the returned `id` is the record's content address, which changes whenever the
 * record does. The pointer file at `packages/<identity>/publisher.json` is what
 * bridges the two — without it, a second publish would have no way to find the
 * record it wrote the first time.
 *
 * A key that the record does not already list is **refused**, not enrolled.
 * Silently accepting a new key would make the publisher identity meaningless:
 * anyone able to write a release could also introduce the key that vouches for
 * it. Key rotation is a registry operation, not a side effect of publishing.
 */
async function resolvePublisher(
  registryDir: string,
  project: RvforgeProject,
  identity: string,
  key: PublisherKey,
  now: string,
): Promise<ResolvedPublisher> {
  const pointerPath = publisherPath(registryDir, identity);
  const pointer = await readPointer(pointerPath);

  if (pointer) {
    const record = await getObject<PublisherRecord>(registryDir, pointer);
    if (!record) {
      throw new ForgeError(
        ForgeErrorCode.REGISTRY,
        `Publisher pointer at ${pointerPath} names ${pointer}, which is not in the object store.`,
        { identity, pointer },
      );
    }
    const registered = record.publicKeys.find((k) => k.keyId === key.keyId && k.revokedAt === null);
    if (!registered) {
      throw new ForgeError(
        ForgeErrorCode.KEY,
        `Key ${key.keyId} is not a registered, unrevoked key for publisher "${record.displayName}". ` +
          'Rotate keys through the registry rather than by publishing with a new one.',
        { keyId: key.keyId, identity, registeredKeys: record.publicKeys.map((k) => k.keyId) },
      );
    }
    return { id: pointer, path: pointerPath, created: false, record };
  }

  const unidentified = {
    schemaVersion: 1 as const,
    type: 'publisher' as const,
    displayName: project.publisher.displayName.trim(),
    publicKeys: [
      {
        keyId: key.keyId,
        alg: 'ed25519' as const,
        publicKey: key.publicKey,
        validFrom: now,
        revokedAt: null,
      },
    ],
    identityEvidence: {
      method: project.publisher.identityEvidence.method,
      reference: project.publisher.identityEvidence.reference.trim(),
    },
    contact: {
      support: project.publisher.contact.support.trim(),
      privacyPolicy: project.publisher.contact.privacyPolicy.trim(),
    },
    signatures: [],
  };
  const id = objectId(unidentified);
  const record: PublisherRecord = { ...unidentified, publisherId: id };
  const stored = await putObject(registryDir, id, record);

  await mkdir(dirname(pointerPath), { recursive: true });
  await writeFile(
    pointerPath,
    canonicalJsonFile({ identity, publisherRecord: id, digest: digestOf(id) }),
    'utf8',
  );
  return { id, path: stored.path, created: true, record };
}

async function readPointer(path: string): Promise<string | null> {
  try {
    const parsed = JSON.parse(await readFile(path, 'utf8')) as { publisherRecord?: string };
    return typeof parsed.publisherRecord === 'string' ? parsed.publisherRecord : null;
  } catch {
    return null;
  }
}

/** Human-facing OS names for the P4 "Supported systems" line. */
const SYSTEM_LABEL: Readonly<Record<string, string>> = Object.freeze({
  windows: 'Windows',
  macos: 'macOS',
  linux: 'Linux',
  rvm: 'RVM appliance',
});

const osOf = (target: BuildTarget): string => target.split('-')[0];

/** Assemble the P4 publish summary from what pack actually established. */
export function publishSummary(project: RvforgeProject, pack: PackResult): PublishSummary {
  const systems = [
    ...new Set(resolveTargets(project.runtimeRequirements.systems).map((t) => SYSTEM_LABEL[osOf(t)] ?? osOf(t))),
  ].sort();
  const network = project.capabilities.requests.find((r) => r.class === 'network');

  return {
    validation: 'passed',
    securityProfile: pack.securityProfile,
    evaluationScore: null,
    evaluationNote: 'not evaluated: requires quarantined runtime (Reader/RVM)',
    supportedRuntimes: [...project.runtimeRequirements.profiles].sort().map((p) => p.toUpperCase()),
    supportedSystems: systems,
    requestedCapabilities: pack.capabilityManifest.requests.map((r) => `${r.class}: ${r.scope}`),
    networkAccess: network ? network.scope : 'none',
    manualReviewTriggers: pack.manualReviewTriggers,
    readyToPublish: true,
  };
}

/** Render the summary as the P4 block. */
export function renderPublishSummary(summary: PublishSummary): string[] {
  return [
    'Validation passed',
    `Security profile: ${summary.securityProfile}`,
    `Evaluation score: ${summary.evaluationNote}`,
    `Supported runtimes: ${summary.supportedRuntimes.join(' and ') || 'none declared'}`,
    `Supported systems: ${summary.supportedSystems.join(', ') || 'none declared'}`,
    `Requested capabilities: ${summary.requestedCapabilities.join(' · ') || 'none'}`,
    `Network access: ${summary.networkAccess}`,
    ...(summary.manualReviewTriggers.length > 0
      ? [`Manual review required: ${summary.manualReviewTriggers.join(', ')}`]
      : []),
    summary.readyToPublish ? 'Ready to publish' : 'Not ready to publish',
  ];
}
