/**
 * Local, file-backed registry (registry data model v0.1, "Storage layout").
 *
 * ```text
 * registry/
 *   objects/sha256/<2-hex>/<digest>.json   all objects, content-addressed
 *   packages/<publisher>/publisher.json    publisher record pointer
 *   packages/<publisher>/<name>/releases.jsonl   append-only release index
 *   log/entries.jsonl                      transparency log
 *   log/tree-head.json                     current tree head
 *   receipts.jsonl                         witness chain over publish events
 * ```
 *
 * Identity follows the model's rules: an object's id is `sha256:<hex>` over
 * its canonical JSON, excluding `signatures` — and excluding the id field
 * itself, since a field cannot contain its own hash. That exclusion is the
 * convention `src/witness.ts` already established for `receiptId`.
 *
 * The transparency log is an RFC 6962 Merkle tree (see `./merkle`), so a
 * Reader can be handed an inclusion proof for one release instead of the whole
 * log, and `log/tree-head.json` is the head that proof reconstructs.
 *
 * **Additions to registry-model.md**, shared with `crates/rvforge-registry`:
 *
 * - `packages/<publisher>/publisher.json` is a pointer from the publisher's
 *   *stable* identity — see `publisherIdFor` — to the content address of its
 *   current record. The documented layout names no such file, and the model
 *   gives no other way to find a record whose address changes whenever its key
 *   set does.
 * - `witness/<sha256(subject)>.jsonl` indexes each subject's receipt chain, as
 *   the Rust registry does. The model defines the receipt and its `prevReceipt`
 *   link but no index, and without one the previous receipt for a subject could
 *   only be found by rescanning the object store.
 * - The tree head is unsigned. The model calls it "the current signed tree
 *   head", but signing it is the registry's act, and this registry has no
 *   registry key — only the publisher's. The field is present and empty.
 */

import { appendFile, mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, join } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { canonicalJson, canonicalJsonFile, sha256String } from './hash';
import { inclusionPath, leafHash, root as merkleRoot, verifyInclusion, type Hash } from './merkle';
import { createWitnessReceipt, type CreateReceiptInput, type WitnessReceipt } from './witness';

export const DEFAULT_REGISTRY_DIR = join(homedir(), '.rvforge', 'registry');
export const LOG_ENTRIES_FILE = join('log', 'entries.jsonl');
export const TREE_HEAD_FILE = join('log', 'tree-head.json');
export const RELEASES_FILENAME = 'releases.jsonl';
export const PUBLISHER_FILENAME = 'publisher.json';
export const WITNESS_DIRNAME = 'witness';

export interface ObjectSignature {
  keyId: string;
  role: 'publisher' | 'registry';
  sig: string;
}

export interface PublisherRecord {
  schemaVersion: 1;
  type: 'publisher';
  publisherId: string;
  displayName: string;
  publicKeys: Array<{
    keyId: string;
    alg: 'ed25519';
    publicKey: string;
    validFrom: string;
    revokedAt: string | null;
  }>;
  identityEvidence: { method: string; reference: string };
  contact: { support: string; privacyPolicy: string };
  signatures: ObjectSignature[];
}

export interface CapabilityManifest {
  schemaVersion: 1;
  type: 'capability-manifest';
  manifestId: string;
  defaultPolicy: 'deny';
  requests: Array<{ class: string; scope: string; rationale: string }>;
  denials: string[];
  manualReviewTriggers: string[];
  signatures: ObjectSignature[];
}

export interface Release {
  schemaVersion: 1;
  type: 'release';
  releaseId: string;
  package: { name: string; publisherId: string };
  version: string;
  predecessor: string | null;
  rvfIdentity: string;
  rvfSize: number;
  capabilityManifest: string;
  runtimeProfiles: string[];
  compatibility: { rvmVersionMin: string; stateSchemaVersion: number; witnessSchemaVersion: number };
  modelManifest: { location: string; digests: string[] };
  softwareInventory: string;
  evaluationReport: string | null;
  securityReport: string | null;
  provenance: string;
  trustLevel: 'published' | 'tested' | 'reviewed' | 'enterprise-approved';
  rollbackSafeUntilStateSchema: number;
  listing: { category: string; description: string; priceModel: string };
  publishedAt: string;
  signatures: ObjectSignature[];
}

export interface TransparencyLogEntry {
  schemaVersion: 1;
  type: 'log-entry';
  index: number;
  entryHash: string;
  treeHead: string;
  object: string;
  objectType: 'release' | 'revocation' | 'publisher';
  timestamp: string;
}

export interface TreeHead {
  schemaVersion: 1;
  type: 'tree-head';
  /** Number of leaves the root covers. */
  treeSize: number;
  /** The Merkle tree head, as `sha256:<hex>`. */
  rootHash: string;
  /** RFC 3339 instant of the append that produced this head. */
  timestamp: string;
  /** Registry signatures over {@link TreeHead.rootHash}; this registry has none. */
  signatures: ObjectSignature[];
}

/** An inclusion proof for one log entry against a tree head. */
export interface InclusionProof {
  leafIndex: number;
  treeSize: number;
  leafHash: string;
  /** The audit path, root-ward, each as `sha256:<hex>`. */
  path: string[];
  rootHash: string;
}

/** Any object the registry content-addresses. */
export type RegistryObject = PublisherRecord | CapabilityManifest | Release;

/** Fields excluded from an object's content id: its own id, and signatures. */
const ID_FIELDS = new Set(['publisherId', 'manifestId', 'releaseId', 'receiptId']);

/**
 * `sha256:<hex>` over an object's canonical JSON, excluding `signatures` and
 * the object's own id field.
 */
export function objectId(value: Record<string, unknown>): string {
  const content: Record<string, unknown> = {};
  for (const key of Object.keys(value)) {
    if (key === 'signatures' || ID_FIELDS.has(key)) continue;
    content[key] = value[key];
  }
  return `sha256:${sha256String(canonicalJson(content))}`;
}

/** Strip the `sha256:` prefix; ids are stored and pathed by their hex digest. */
export const digestOf = (id: string): string => (id.startsWith('sha256:') ? id.slice(7) : id);

/** Path of a content-addressed object within a registry directory. */
export const objectPath = (registryDir: string, id: string): string => {
  const digest = digestOf(id);
  if (!/^[0-9a-f]{64}$/.test(digest)) {
    throw new ForgeError(ForgeErrorCode.REGISTRY, `"${id}" is not a sha256 object id.`, { id });
  }
  return join(registryDir, 'objects', 'sha256', digest.slice(0, 2), `${digest}.json`);
};

export const packagePath = (registryDir: string, publisherId: string, name: string): string =>
  join(registryDir, 'packages', digestOf(publisherId), name);

export const publisherPath = (registryDir: string, publisherId: string): string =>
  join(registryDir, 'packages', digestOf(publisherId), PUBLISHER_FILENAME);

/** Create the directory skeleton. Idempotent. */
export async function initRegistry(registryDir: string): Promise<void> {
  for (const dir of [
    join(registryDir, 'objects', 'sha256'),
    join(registryDir, 'packages'),
    join(registryDir, 'log'),
    join(registryDir, WITNESS_DIRNAME),
  ]) {
    await mkdir(dir, { recursive: true });
  }
}

/**
 * Write an object at its content address.
 *
 * Content addressing makes the write idempotent: identical bytes land on the
 * same path, so re-publishing an unchanged manifest is a no-op rather than a
 * conflict. `existed` reports which happened.
 */
export async function putObject(
  registryDir: string,
  id: string,
  object: unknown,
): Promise<{ path: string; existed: boolean }> {
  const path = objectPath(registryDir, id);
  await mkdir(dirname(path), { recursive: true });
  const body = canonicalJsonFile(object);
  try {
    await writeFile(path, body, { encoding: 'utf8', flag: 'wx' });
    return { path, existed: false };
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== 'EEXIST') {
      throw new ForgeError(ForgeErrorCode.IO, `Cannot write registry object ${id}.`, { id, path });
    }
    // A signature added later changes the stored bytes but not the id, so an
    // existing object is refreshed rather than left at its unsigned version.
    await writeFile(path, body, 'utf8');
    return { path, existed: true };
  }
}

/** Read a content-addressed object, or null when it is absent. */
export async function getObject<T>(registryDir: string, id: string): Promise<T | null> {
  try {
    return JSON.parse(await readFile(objectPath(registryDir, id), 'utf8')) as T;
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'ENOENT') return null;
    if (err instanceof ForgeError) throw err;
    throw new ForgeError(ForgeErrorCode.REGISTRY, `Registry object ${id} is unreadable or malformed.`, { id });
  }
}

/**
 * One line of `releases.jsonl`: the index, not the release itself.
 *
 * Exactly these four fields — the Rust registry reads the same file with
 * `deny_unknown_fields`, so an extra convenience column here would make the
 * index unreadable to the other half of the pipeline.
 */
export interface ReleaseIndexEntry {
  releaseId: string;
  version: string;
  predecessor: string | null;
  publishedAt: string;
}

/** Read a package's release index; an absent file means no releases yet. */
export async function readReleaseIndex(
  registryDir: string,
  publisherId: string,
  name: string,
): Promise<ReleaseIndexEntry[]> {
  let raw: string;
  try {
    raw = await readFile(join(packagePath(registryDir, publisherId, name), RELEASES_FILENAME), 'utf8');
  } catch {
    return [];
  }
  return raw
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as ReleaseIndexEntry);
}

/**
 * Append a release to its package index after checking the lineage.
 *
 * A release must name the current head as its predecessor: the first release
 * names null, and every later one names the release before it. Rejecting a
 * mismatch here is what keeps `releases.jsonl` a chain rather than a set.
 *
 * @throws {ForgeError} `FORGE_E_LINEAGE` when the predecessor is not the head.
 */
export async function appendRelease(
  registryDir: string,
  release: Release,
): Promise<{ path: string; index: number }> {
  const dir = packagePath(registryDir, release.package.publisherId, release.package.name);
  const existing = await readReleaseIndex(registryDir, release.package.publisherId, release.package.name);
  const head = existing.length === 0 ? null : existing[existing.length - 1].releaseId;

  if (release.predecessor !== head) {
    throw new ForgeError(
      ForgeErrorCode.LINEAGE,
      `Release names predecessor ${String(release.predecessor)}, but the head of ` +
        `${release.package.name} is ${String(head)}.`,
      { predecessor: release.predecessor, head, package: release.package.name },
    );
  }
  if (existing.some((entry) => entry.version === release.version)) {
    throw new ForgeError(
      ForgeErrorCode.LINEAGE,
      `Version ${release.version} of ${release.package.name} is already published; releases are immutable.`,
      { version: release.version, package: release.package.name },
    );
  }

  const entry: ReleaseIndexEntry = {
    releaseId: release.releaseId,
    version: release.version,
    predecessor: release.predecessor,
    publishedAt: release.publishedAt,
  };
  await mkdir(dir, { recursive: true });
  await appendFile(join(dir, RELEASES_FILENAME), `${canonicalJson(entry)}\n`, 'utf8');
  return { path: join(dir, RELEASES_FILENAME), index: existing.length };
}

/** Read the transparency log. An absent file means an empty log. */
export async function readLog(registryDir: string): Promise<TransparencyLogEntry[]> {
  let raw: string;
  try {
    raw = await readFile(join(registryDir, LOG_ENTRIES_FILE), 'utf8');
  } catch {
    return [];
  }
  return raw
    .split('\n')
    .filter((line) => line.trim() !== '')
    .map((line) => JSON.parse(line) as TransparencyLogEntry);
}

/** Read the current tree head, or null when the log is empty. */
export async function readTreeHead(registryDir: string): Promise<TreeHead | null> {
  try {
    return JSON.parse(await readFile(join(registryDir, TREE_HEAD_FILE), 'utf8')) as TreeHead;
  } catch {
    return null;
  }
}

/**
 * The Merkle leaf hash for a log entry's position and payload.
 *
 * `entryHash` and `treeHead` are excluded because both are derived from the
 * leaf; including them would make the hash self-referential.
 */
const entryLeafHash = (entry: {
  index: number;
  object: string;
  objectType: TransparencyLogEntry['objectType'];
  timestamp: string;
}): Hash =>
  leafHash(
    Buffer.from(
      canonicalJson({
        index: entry.index,
        object: entry.object,
        objectType: entry.objectType,
        timestamp: entry.timestamp,
      }),
      'utf8',
    ),
  );

const asDigest = (hash: Hash): string => `sha256:${hash.toString('hex')}`;
const fromDigest = (digest: string): Hash => Buffer.from(digestOf(digest), 'hex');

/**
 * Append one entry to the transparency log and advance the tree head.
 *
 * The head is the Merkle root over every leaf so far, so removing, reordering,
 * or editing any entry changes it — and any single entry can still be proved
 * against it without the rest of the log ({@link inclusionProof}).
 */
export async function appendLogEntry(
  registryDir: string,
  input: { object: string; objectType: TransparencyLogEntry['objectType']; timestamp: string },
): Promise<{ entry: TransparencyLogEntry; head: TreeHead }> {
  const existing = await readLog(registryDir);
  const index = existing.length;

  const leaves = existing.map((e) => fromDigest(e.entryHash));
  const leaf = entryLeafHash({ index, ...input });
  leaves.push(leaf);
  const rootHash = asDigest(merkleRoot(leaves));

  const entry: TransparencyLogEntry = {
    schemaVersion: 1,
    type: 'log-entry',
    index,
    entryHash: asDigest(leaf),
    treeHead: rootHash,
    object: input.object,
    objectType: input.objectType,
    timestamp: input.timestamp,
  };
  const head: TreeHead = {
    schemaVersion: 1,
    type: 'tree-head',
    treeSize: leaves.length,
    rootHash,
    timestamp: input.timestamp,
    signatures: [],
  };

  await mkdir(join(registryDir, 'log'), { recursive: true });
  await appendFile(join(registryDir, LOG_ENTRIES_FILE), `${canonicalJson(entry)}\n`, 'utf8');
  await writeFile(join(registryDir, TREE_HEAD_FILE), canonicalJsonFile(head), 'utf8');
  return { entry, head };
}

/** Recompute every leaf and head, and report the first entry that disagrees. */
export function verifyLog(entries: readonly TransparencyLogEntry[]): {
  ok: boolean;
  size: number;
  treeHead: string | null;
  problems: Array<{ index: number; reason: 'entry-hash' | 'tree-head' | 'index' }>;
} {
  const problems: Array<{ index: number; reason: 'entry-hash' | 'tree-head' | 'index' }> = [];
  const leaves: Hash[] = [];

  entries.forEach((entry, position) => {
    if (entry.index !== position) problems.push({ index: position, reason: 'index' });
    const leaf = entryLeafHash(entry);
    if (asDigest(leaf) !== entry.entryHash) problems.push({ index: position, reason: 'entry-hash' });
    leaves.push(leaf);
    if (asDigest(merkleRoot(leaves)) !== entry.treeHead) {
      problems.push({ index: position, reason: 'tree-head' });
    }
  });

  return {
    ok: problems.length === 0,
    size: entries.length,
    treeHead: entries.length === 0 ? null : entries[entries.length - 1].treeHead,
    problems,
  };
}

/**
 * An inclusion proof for `object` against the head the log currently produces.
 *
 * Returns null when the object was never logged — a release absent from the log
 * is unpublished, and there is nothing to prove.
 */
export function inclusionProof(
  entries: readonly TransparencyLogEntry[],
  object: string,
): InclusionProof | null {
  const entry = entries.find((e) => e.object === object);
  if (!entry) return null;

  const leaves = entries.map(entryLeafHash);
  const path = inclusionPath(entry.index, leaves);
  return {
    leafIndex: entry.index,
    treeSize: leaves.length,
    leafHash: asDigest(leaves[entry.index]),
    path: path.map(asDigest),
    rootHash: asDigest(merkleRoot(leaves)),
  };
}

/** Recompute the root from a proof's leaf and path, against a trusted head. */
export function verifyInclusionProof(proof: InclusionProof, expectedRoot: string): boolean {
  if (proof.rootHash !== expectedRoot) return false;
  try {
    return verifyInclusion(
      fromDigest(proof.leafHash),
      proof.leafIndex,
      proof.treeSize,
      proof.path.map(fromDigest),
      fromDigest(expectedRoot),
    );
  } catch {
    // A proof that cannot be parsed is a proof that does not verify.
    return false;
  }
}

/**
 * The receipt-chain index for one subject: `witness/<sha256(subject)>.jsonl`.
 *
 * A subject may be a content address or a `keyId`, so it is hashed rather than
 * used directly — one naming rule, no escaping, no collision with a path
 * separator.
 */
export const witnessIndexPath = (registryDir: string, subject: string): string =>
  join(registryDir, WITNESS_DIRNAME, `${sha256String(subject)}.jsonl`);

/** Receipt ids recorded for `subject`, oldest first. */
export async function readWitnessChain(registryDir: string, subject: string): Promise<string[]> {
  try {
    return (await readFile(witnessIndexPath(registryDir, subject), 'utf8'))
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => line !== '');
  } catch {
    return [];
  }
}

/**
 * Store a receipt at its content address and chain it onto its subject's index.
 *
 * The previous receipt for the subject comes from the index rather than from a
 * scan of the object store, which is the whole reason the index exists.
 */
export async function appendWitnessReceipt(
  registryDir: string,
  input: CreateReceiptInput,
): Promise<WitnessReceipt> {
  const subject = input.subject.startsWith('sha256:') ? input.subject : `sha256:${input.subject}`;
  const chain = await readWitnessChain(registryDir, subject);
  const receipt = createWitnessReceipt({
    ...input,
    prevReceipt: input.prevReceipt ?? (chain.length === 0 ? null : chain[chain.length - 1]),
  });

  await putObject(registryDir, receipt.receiptId, receipt);
  const path = witnessIndexPath(registryDir, subject);
  await mkdir(dirname(path), { recursive: true });
  await appendFile(path, `${receipt.receiptId}\n`, 'utf8');
  return receipt;
}
