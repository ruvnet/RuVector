/**
 * Minimal writer for the RVF v1 wire format — the mirror of {@link ./rvf}.
 *
 * `rvf.ts` parses segment headers and the Level-0 root manifest; this module
 * writes them, sharing that module's constants so the two cannot drift. It
 * knows about bytes and signatures and nothing else: no project model, no
 * filesystem, no CLI. {@link ./create} owns the policy of what to put in a
 * container; this owns how a container is spelled.
 *
 * The layout mirrors `crates/rvf-forge-core/src/author/` field for field, which
 * is what lets the Rust crate read what this writes. See
 * `scripts/rvforge-parity-check.sh`.
 *
 * Writing is not running: payload bytes are hashed and copied, never parsed as
 * code, linked, or executed (ADR-283 invariant 2).
 */

import { createHash, sign as cryptoSign } from 'node:crypto';

import type { PublisherKey } from './keys';
import {
  ROOT_MANIFEST_MAGIC_BYTES,
  ROOT_MANIFEST_SIZE,
  SEGMENT_FLAG_SIGNED,
  SEGMENT_HEADER_SIZE,
  SEGMENT_MAGIC_BYTES,
  alignUp,
  crc32c,
} from './rvf';

/**
 * SHAKE-256 truncated to 128 bits, as `checksum_algo` names it.
 *
 * The format's default is XXH3-128, which is faster. SHAKE-256 is used here
 * because the Rust crate has to recompute these hashes byte for byte, and
 * SHAKE-256 is reachable from any standard crypto library where XXH3-128 would
 * mean hand-porting a hash. It is also the cryptographic option, which is the
 * right default for a signing-adjacent tool.
 */
export const CHECKSUM_ALGO_SHAKE256 = 2;

/**
 * Ed25519 in a *segment signature footer* (`rvf-crypto/src/sign.rs`).
 *
 * Not a typo for the root manifest's `1` below: the two structures carry
 * different algorithm enumerations, and each writer has to use its own.
 */
const SEGMENT_SIG_ALGO_ED25519 = 0;
/** Ed25519 in the Level-0 root manifest's `sig_algo` field. */
const ROOT_SIG_ALGO_ED25519 = 1;
const ED25519_SIGNATURE_LENGTH = 64;
/** `sig_algo` + `sig_length` + signature + `footer_length`. */
export const SIGNATURE_FOOTER_LENGTH = 2 + 2 + ED25519_SIGNATURE_LENGTH + 4;
/** Domain separator a segment signature covers (`rvf-crypto/src/sign.rs`). */
const SEGMENT_SIGNING_CONTEXT = Buffer.from('RVF-v1-segment', 'ascii');

/** The segment types `create` emits. */
export const SEGMENT_TYPE = { VEC: 0x01, MANIFEST: 0x05, META: 0x07, WASM: 0x10 } as const;

const SEGMENT_TYPE_NAMES: Readonly<Record<number, string>> = Object.freeze({
  [SEGMENT_TYPE.VEC]: 'vec',
  [SEGMENT_TYPE.MANIFEST]: 'manifest',
  [SEGMENT_TYPE.META]: 'meta',
  [SEGMENT_TYPE.WASM]: 'wasm',
});

export const segmentTypeName = (type: number): string =>
  SEGMENT_TYPE_NAMES[type] ?? `unknown(0x${type.toString(16)})`;

/** Level-0 root manifest field offsets (`rvf-manifest/src/level0.rs`). */
const L0 = {
  VERSION: 0x004,
  L1_OFFSET: 0x008,
  L1_LENGTH: 0x010,
  DTYPE: 0x022,
  EPOCH: 0x024,
  SIG_ALGO: 0x098,
  SIG_LEN: 0x09a,
  SIGNATURE: 0x09c,
  FILE_ID: 0xf00,
  CHECKSUM: 0xffc,
} as const;

export const shake256_128 = (data: Uint8Array): Buffer =>
  createHash('shake256', { outputLength: 16 }).update(data).digest();

/**
 * Serialise one segment: 64-byte header, payload, optional signature footer,
 * zero padding to the next 64-byte boundary.
 */
export function writeSegment(
  segType: number,
  payload: Buffer,
  segmentId: number,
  key?: PublisherKey,
): Buffer {
  const footerLength = key ? SIGNATURE_FOOTER_LENGTH : 0;
  const unpadded = SEGMENT_HEADER_SIZE + payload.length + footerLength;
  const padded = alignUp(unpadded);

  const header = Buffer.alloc(SEGMENT_HEADER_SIZE);
  SEGMENT_MAGIC_BYTES.copy(header, 0x00);
  header[0x04] = 1; // version
  header[0x05] = segType;
  header.writeUInt16LE(key ? SEGMENT_FLAG_SIGNED : 0, 0x06);
  header.writeBigUInt64LE(BigInt(segmentId), 0x08);
  header.writeBigUInt64LE(BigInt(payload.length), 0x10);
  // timestamp_ns stays zero: output must be byte-reproducible.
  header[0x20] = CHECKSUM_ALGO_SHAKE256;
  shake256_128(payload).copy(header, 0x28);
  header.writeUInt32LE(padded - unpadded, 0x3c); // alignment_pad

  const out = Buffer.alloc(padded);
  header.copy(out, 0);
  payload.copy(out, SEGMENT_HEADER_SIZE);
  if (key) {
    signatureFooter(header, payload, segmentId, key).copy(
      out,
      SEGMENT_HEADER_SIZE + payload.length,
    );
  }
  return out;
}

/**
 * The Ed25519 signature footer for one segment.
 *
 * The signed message is the canonical form from `rvf-crypto/src/sign.rs`:
 * `header[0..40] || content_hash || "RVF-v1-segment" || segment_id || shake256_128(payload)`.
 */
function signatureFooter(
  header: Buffer,
  payload: Buffer,
  segmentId: number,
  key: PublisherKey,
): Buffer {
  const segmentIdBytes = Buffer.alloc(8);
  segmentIdBytes.writeBigUInt64LE(BigInt(segmentId));
  const message = Buffer.concat([
    header.subarray(0, 40),
    header.subarray(0x28, 0x38), // content_hash
    SEGMENT_SIGNING_CONTEXT,
    segmentIdBytes,
    shake256_128(payload),
  ]);
  // Ed25519 in Node takes a null algorithm: the scheme fixes its own hash.
  const signature = cryptoSign(null, message, key.privateKeyObject);

  const footer = Buffer.alloc(SIGNATURE_FOOTER_LENGTH);
  footer.writeUInt16LE(SEGMENT_SIG_ALGO_ED25519, 0);
  footer.writeUInt16LE(ED25519_SIGNATURE_LENGTH, 2);
  signature.copy(footer, 4);
  footer.writeUInt32LE(SIGNATURE_FOOTER_LENGTH, 4 + ED25519_SIGNATURE_LENGTH);
  return footer;
}

/** Build the 4096-byte Level-0 root manifest page that closes the file. */
export function rootManifestPage(
  fileId: Buffer,
  l1Offset: number,
  l1Length: number,
  key?: PublisherKey,
): Buffer {
  const page = Buffer.alloc(ROOT_MANIFEST_SIZE);
  ROOT_MANIFEST_MAGIC_BYTES.copy(page, 0x000);
  page.writeUInt16LE(1, L0.VERSION);
  page.writeBigUInt64LE(BigInt(l1Offset), L0.L1_OFFSET);
  page.writeBigUInt64LE(BigInt(l1Length), L0.L1_LENGTH);
  // total_vector_count and dimension stay zero: an agent container declares no
  // vectors, and a zero dimension is only invalid alongside a vector count.
  page[L0.DTYPE] = 1; // f32
  page.writeUInt32LE(1, L0.EPOCH);
  // created_ns and modified_ns stay zero, for reproducibility.
  fileId.copy(page, L0.FILE_ID);

  if (key) {
    // sig_algo and sig_length are written before signing so the signature
    // covers them; otherwise `sig_algo` could be rewritten to name a weaker
    // scheme without invalidating anything.
    page.writeUInt16LE(ROOT_SIG_ALGO_ED25519, L0.SIG_ALGO);
    page.writeUInt16LE(ED25519_SIGNATURE_LENGTH, L0.SIG_LEN);
    const signature = cryptoSign(null, rootManifestSigningBytes(page), key.privateKeyObject);
    signature.copy(page, L0.SIGNATURE);
  }

  page.writeUInt32LE(crc32c(page.subarray(0, L0.CHECKSUM)), L0.CHECKSUM);
  return page;
}

/**
 * The bytes a root manifest page's signature covers: the whole page with the
 * signature region and the trailing CRC32C zeroed, since both are written
 * after signing.
 */
export function rootManifestSigningBytes(page: Buffer): Buffer {
  const signable = Buffer.from(page);
  signable.fill(0, L0.SIGNATURE, L0.SIGNATURE + ED25519_SIGNATURE_LENGTH);
  signable.fill(0, L0.CHECKSUM);
  return signable;
}
