/**
 * Ed25519 publisher keys (registry data model v0.1, ADR-294 §9.1).
 *
 * The publisher signs; the registry countersigns. This module owns the
 * publisher half: generating a keypair, reading one back, and producing an
 * Ed25519 signature over a registry object's content id.
 *
 * Three rules the rest of the package depends on:
 *
 * - **Key material never leaves this module.** A {@link PublisherKey} carries
 *   Node `KeyObject`s, not encoded bytes; nothing here returns, logs, or puts
 *   a private key into an error's detail bag. Callers get the *path* to a key
 *   file and its public half, never its contents.
 * - **A world- or group-readable key file is refused on POSIX.** A signing key
 *   readable by every account on the machine is not a signing key, and
 *   discovering that at publish time is better than discovering it in an
 *   incident report. Windows has no comparable mode bits, so the check is
 *   skipped there rather than faked.
 * - **`keyId` is derived, not chosen.** It is `ed25519:<sha256 of the raw
 *   32-byte public key>`, so the same key always names itself the same way and
 *   a key cannot claim another key's identifier.
 */

import {
  createPrivateKey,
  createPublicKey,
  generateKeyPairSync,
  sign as cryptoSign,
  verify as cryptoVerify,
  type KeyObject,
} from 'node:crypto';
import { readFile, stat, writeFile } from 'node:fs/promises';

import { ForgeError, ForgeErrorCode } from './errors';
import { sha256Buffer } from './hash';

/** DER prefix of an Ed25519 SPKI public key; the raw 32 bytes follow it. */
const SPKI_ED25519_PREFIX_LENGTH = 12;

/** File mode bits that must be clear on a POSIX key file: group + other. */
const OVER_PERMISSIVE_MODE = 0o077;

export interface PublisherKeyFile {
  schemaVersion: 1;
  alg: 'ed25519';
  keyId: string;
  /** Base64 of the raw 32-byte public key. */
  publicKey: string;
  /** Base64 of the PKCS#8 private key. Never read outside this module. */
  privateKey: string;
  createdAt: string;
}

/** A loaded keypair. The private half is a `KeyObject`, never bytes. */
export interface PublisherKey {
  keyId: string;
  /** Base64 of the raw 32-byte public key — safe to publish and log. */
  publicKey: string;
  privateKeyObject: KeyObject;
  publicKeyObject: KeyObject;
}

/** Raw 32-byte public key from a `KeyObject`, base64-encoded. */
function rawPublicKey(key: KeyObject): Buffer {
  const der = key.export({ type: 'spki', format: 'der' });
  return Buffer.from(der.subarray(SPKI_ED25519_PREFIX_LENGTH));
}

/** `ed25519:<sha256 hex of the raw public key>` — derived, never chosen. */
export const keyIdFor = (rawPublic: Uint8Array): string => `ed25519:${sha256Buffer(rawPublic)}`;

/** Generate a keypair and its on-disk representation. */
export function generatePublisherKey(now = new Date()): PublisherKeyFile {
  const { publicKey, privateKey } = generateKeyPairSync('ed25519');
  const raw = rawPublicKey(publicKey);
  return {
    schemaVersion: 1,
    alg: 'ed25519',
    keyId: keyIdFor(raw),
    publicKey: raw.toString('base64'),
    privateKey: privateKey.export({ type: 'pkcs8', format: 'der' }).toString('base64'),
    createdAt: now.toISOString(),
  };
}

/**
 * Write a key file with owner-only permissions.
 *
 * The mode is passed to `open(2)` rather than applied afterwards, so the file
 * is never briefly world-readable between creation and `chmod`.
 *
 * @throws {ForgeError} `FORGE_E_KEY` when the file already exists.
 */
export async function writePublisherKey(path: string, key: PublisherKeyFile): Promise<void> {
  try {
    await writeFile(path, `${JSON.stringify(key, null, 2)}\n`, {
      encoding: 'utf8',
      flag: 'wx',
      mode: 0o600,
    });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code === 'EEXIST') {
      throw new ForgeError(
        ForgeErrorCode.KEY,
        `A key file already exists at ${path}. Refusing to overwrite signing key material.`,
        { path },
      );
    }
    throw new ForgeError(ForgeErrorCode.IO, `Cannot write the key file at ${path}.`, { path });
  }
}

/**
 * Refuse a key file any account other than its owner can read.
 *
 * Windows reports a synthetic mode through `fs.stat`, so the check runs on
 * POSIX only. Skipping it is stated rather than silent: the publish summary
 * records that permissions were not checked on this platform.
 */
export async function assertKeyFilePermissions(path: string): Promise<{ checked: boolean }> {
  if (process.platform === 'win32') return { checked: false };

  let mode: number;
  try {
    mode = (await stat(path)).mode;
  } catch {
    throw new ForgeError(ForgeErrorCode.NOT_FOUND, `No key file at ${path}.`, { path });
  }

  const offending = mode & OVER_PERMISSIVE_MODE;
  if (offending !== 0) {
    throw new ForgeError(
      ForgeErrorCode.KEY,
      `Key file ${path} is readable beyond its owner (mode ${(mode & 0o777).toString(8)}). ` +
        'Run "chmod 600" on it before publishing.',
      { path, mode: (mode & 0o777).toString(8) },
    );
  }
  return { checked: true };
}

/**
 * Load a keypair from disk, after checking its permissions.
 *
 * @throws {ForgeError} `FORGE_E_KEY` for a malformed or over-permissioned key
 * file, `FORGE_E_NOT_FOUND` when it is absent.
 */
export async function loadPublisherKey(path: string): Promise<PublisherKey> {
  await assertKeyFilePermissions(path);

  let raw: string;
  try {
    raw = await readFile(path, 'utf8');
  } catch {
    throw new ForgeError(ForgeErrorCode.NOT_FOUND, `No key file at ${path}.`, { path });
  }

  let parsed: Partial<PublisherKeyFile>;
  try {
    parsed = JSON.parse(raw) as Partial<PublisherKeyFile>;
  } catch {
    // The parser's message can quote the file's bytes; those bytes are a
    // private key, so the message is dropped rather than forwarded.
    throw new ForgeError(ForgeErrorCode.KEY, `The key file at ${path} is not valid JSON.`, { path });
  }

  if (parsed.alg !== 'ed25519' || typeof parsed.privateKey !== 'string') {
    throw new ForgeError(
      ForgeErrorCode.KEY,
      `The key file at ${path} is not an Ed25519 key generated by "rvforge init --keygen".`,
      { path },
    );
  }

  let privateKeyObject: KeyObject;
  try {
    privateKeyObject = createPrivateKey({
      key: Buffer.from(parsed.privateKey, 'base64'),
      format: 'der',
      type: 'pkcs8',
    });
  } catch {
    throw new ForgeError(ForgeErrorCode.KEY, `The key file at ${path} does not decode to an Ed25519 key.`, {
      path,
    });
  }

  const publicKeyObject = createPublicKey(privateKeyObject);
  const rawPublic = rawPublicKey(publicKeyObject);
  return {
    keyId: keyIdFor(rawPublic),
    publicKey: rawPublic.toString('base64'),
    privateKeyObject,
    publicKeyObject,
  };
}

/**
 * Sign a registry object's content id.
 *
 * The id, not the object, is what is signed: it already commits to every field
 * outside `signatures`, and signing a fixed-length ASCII string keeps the
 * operation independent of how the object was serialised on the way in.
 */
export const signObjectId = (key: PublisherKey, objectId: string): string =>
  cryptoSign(null, Buffer.from(objectId, 'utf8'), key.privateKeyObject).toString('base64');

/** Verify a signature over an object id against a base64 raw public key. */
export function verifyObjectId(publicKeyBase64: string, objectId: string, signature: string): boolean {
  let publicKeyObject: KeyObject;
  try {
    publicKeyObject = createPublicKey({
      key: Buffer.concat([
        Buffer.from('302a300506032b6570032100', 'hex'),
        Buffer.from(publicKeyBase64, 'base64'),
      ]),
      format: 'der',
      type: 'spki',
    });
  } catch {
    return false;
  }
  try {
    return cryptoVerify(null, Buffer.from(objectId, 'utf8'), publicKeyObject, Buffer.from(signature, 'base64'));
  } catch {
    return false;
  }
}
