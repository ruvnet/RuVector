/**
 * Minimal, inspection-only reader for the RVF v1 wire format.
 *
 * This module parses bytes and nothing else. It never evaluates, links,
 * instantiates, or otherwise executes any part of an RVF — packaging and
 * validation are inspection operations (ADR-283 invariant 2).
 *
 * `@ruvector/rvf` is the SDK for *querying* an RVF through a native or WASM
 * backend; it does not expose a standalone header/manifest parser and pulling
 * a backend in would mean loading an RVF into a runtime. So the layout below
 * mirrors `crates/rvf/rvf-types/src/constants.rs` and
 * `crates/rvf/rvf-manifest/src/level0.rs` directly.
 *
 * Wire note: RVF v1 serialises every multi-byte integer little-endian, so the
 * magic bytes on disk are the reverse of their "RVM0"/"RVFS" mnemonics.
 */

/** The Level-0 root manifest is exactly one page, at EOF. */
export const ROOT_MANIFEST_SIZE = 4096;
/** `ROOT_MANIFEST_MAGIC` (0x52564D30, "RVM0") as it appears on the wire. */
export const ROOT_MANIFEST_MAGIC_BYTES = Buffer.from([0x30, 0x4d, 0x56, 0x52]);
/** `SEGMENT_MAGIC` (0x52564653, "RVFS") as it appears on the wire. */
export const SEGMENT_MAGIC_BYTES = Buffer.from([0x53, 0x46, 0x56, 0x52]);
export const SEGMENT_HEADER_SIZE = 64;
export const SEGMENT_ALIGNMENT = 64;
export const MAX_SEGMENT_PAYLOAD = 4 * 1024 * 1024 * 1024;

/**
 * `SegmentFlags::SIGNED` (`crates/rvf/rvf-types/src/flags.rs`) — a signature
 * footer follows the payload, and the next segment starts after *it*.
 */
export const SEGMENT_FLAG_SIGNED = 0x0004;
/** `SegmentFlags::ENCRYPTED`. */
export const SEGMENT_FLAG_ENCRYPTED = 0x0002;
/** `sig_algo` + `sig_length` + `footer_length`, with no signature bytes. */
export const SIGNATURE_FOOTER_MIN_SIZE = 8;
/** Longest signature the footer codec accepts, from `SignatureFooter`. */
export const MAX_SIGNATURE_LENGTH = 7856;

/** Root manifest field offsets (level0.rs). */
const OFF = {
  MAGIC: 0x000,
  VERSION: 0x004,
  FLAGS: 0x006,
  L1_OFFSET: 0x008,
  L1_LENGTH: 0x010,
  TOTAL_VEC: 0x018,
  DIM: 0x020,
  DTYPE: 0x022,
  PROFILE: 0x023,
  EPOCH: 0x024,
  CREATED: 0x028,
  MODIFIED: 0x030,
  SIG_ALGO: 0x098,
  SIG_LEN: 0x09a,
  SIGNATURE: 0x09c,
  FILE_ID: 0xf00,
  PARENT_ID: 0xf10,
  PARENT_HASH: 0xf20,
  LINEAGE_DEPTH: 0xf40,
  CHECKSUM: 0xffc,
} as const;

/** Segment types that carry executable content and therefore must be signed. */
export const EXECUTABLE_SEGMENT_TYPES: Readonly<Record<number, string>> = Object.freeze({
  0x0e: 'kernel',
  0x0f: 'ebpf',
  0x10: 'wasm',
});

export const SEGMENT_TYPE_NAMES: Readonly<Record<number, string>> = Object.freeze({
  0x00: 'invalid',
  0x01: 'vec',
  0x02: 'index',
  0x03: 'overlay',
  0x04: 'journal',
  0x05: 'manifest',
  0x06: 'quant',
  0x07: 'meta',
  0x08: 'hot',
  0x09: 'sketch',
  0x0a: 'witness',
  0x0b: 'profile',
  0x0c: 'crypto',
  0x0d: 'metaidx',
  0x0e: 'kernel',
  0x0f: 'ebpf',
  0x10: 'wasm',
  0x11: 'dashboard',
  0x20: 'cowmap',
  0x21: 'refcount',
  0x22: 'membership',
  0x23: 'delta',
  0x30: 'transfer-prior',
  0x31: 'policy-kernel',
});

// ---------------------------------------------------------------- CRC32C ---

const CRC32C_POLY = 0x82f63b78;
let crcTable: Int32Array | undefined;

function crc32cTable(): Int32Array {
  if (crcTable) return crcTable;
  const table = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = c & 1 ? CRC32C_POLY ^ (c >>> 1) : c >>> 1;
    }
    table[n] = c;
  }
  crcTable = table;
  return table;
}

/** CRC32C (Castagnoli), matching the `crc32c` crate used by rvf-manifest. */
export function crc32c(data: Uint8Array): number {
  const table = crc32cTable();
  let crc = -1;
  for (let i = 0; i < data.length; i++) {
    crc = table[(crc ^ data[i]) & 0xff] ^ (crc >>> 8);
  }
  return (crc ^ -1) >>> 0;
}

// ------------------------------------------------------------ structures ---

export interface RvfRootManifest {
  magic: number;
  version: number;
  flags: number;
  l1ManifestOffset: bigint;
  l1ManifestLength: bigint;
  totalVectorCount: bigint;
  dimension: number;
  baseDtype: number;
  profileId: number;
  epoch: number;
  createdNs: bigint;
  modifiedNs: bigint;
  sigAlgo: number;
  sigLength: number;
  fileId: string;
  parentId: string;
  parentHash: string;
  lineageDepth: number;
  storedChecksum: number;
  computedChecksum: number;
  checksumValid: boolean;
  magicValid: boolean;
  signaturePresent: boolean;
}

export interface RvfSegmentHeader {
  offset: number;
  magicValid: boolean;
  version: number;
  segType: number;
  segTypeName: string;
  flags: number;
  segmentId: bigint;
  payloadLength: bigint;
  timestampNs: bigint;
  checksumAlgo: number;
  compression: number;
  contentHash: string;
  executable: boolean;
  /** The `SIGNED` flag is set, so a signature footer follows the payload. */
  signed: boolean;
  /** The `ENCRYPTED` flag is set. */
  encrypted: boolean;
}

/** A segment's trailing signature footer (`rvf-crypto/src/footer.rs`). */
export interface RvfSignatureFooter {
  sigAlgo: number;
  sigLength: number;
  /** Total footer size on the wire, as the footer itself declares it. */
  footerLength: number;
}

/** The footer length a `sigLength`-byte signature implies. */
export const footerLengthFor = (sigLength: number): number =>
  SIGNATURE_FOOTER_MIN_SIZE + sigLength;

/**
 * Parse a signature footer at `offset`.
 *
 * Returns `undefined` when the bytes are not a well-formed footer: too short,
 * an implausible signature length, or a declared `footerLength` inconsistent
 * with the signature it claims to carry. A segment's zero padding decodes as a
 * structurally valid but empty footer, so accepting one would let any segment
 * claim `SIGNED` by flipping a single flag bit and changing nothing else.
 */
export function parseSignatureFooter(buf: Buffer, offset: number): RvfSignatureFooter | undefined {
  if (offset + SIGNATURE_FOOTER_MIN_SIZE > buf.length) return undefined;
  const sigAlgo = buf.readUInt16LE(offset);
  const sigLength = buf.readUInt16LE(offset + 2);
  if (sigLength === 0 || sigLength > MAX_SIGNATURE_LENGTH) return undefined;
  if (offset + 4 + sigLength + 4 > buf.length) return undefined;

  const footerLength = buf.readUInt32LE(offset + 4 + sigLength);
  if (footerLength !== footerLengthFor(sigLength)) return undefined;
  return { sigAlgo, sigLength, footerLength };
}

const hex = (buf: Buffer): string => buf.toString('hex');

/**
 * Parse a 4096-byte Level-0 root manifest.
 *
 * Returns the parsed view even when magic or checksum are wrong so callers can
 * report precisely what failed; check `magicValid` and `checksumValid`.
 */
export function parseRootManifest(page: Buffer): RvfRootManifest {
  if (page.length !== ROOT_MANIFEST_SIZE) {
    throw new RangeError(`root manifest must be ${ROOT_MANIFEST_SIZE} bytes, got ${page.length}`);
  }
  const storedChecksum = page.readUInt32LE(OFF.CHECKSUM);
  const computedChecksum = crc32c(page.subarray(0, OFF.CHECKSUM));
  const sigAlgo = page.readUInt16LE(OFF.SIG_ALGO);
  const sigLength = page.readUInt16LE(OFF.SIG_LEN);

  return {
    magic: page.readUInt32LE(OFF.MAGIC),
    version: page.readUInt16LE(OFF.VERSION),
    flags: page.readUInt16LE(OFF.FLAGS),
    l1ManifestOffset: page.readBigUInt64LE(OFF.L1_OFFSET),
    l1ManifestLength: page.readBigUInt64LE(OFF.L1_LENGTH),
    totalVectorCount: page.readBigUInt64LE(OFF.TOTAL_VEC),
    dimension: page.readUInt16LE(OFF.DIM),
    baseDtype: page[OFF.DTYPE],
    profileId: page[OFF.PROFILE],
    epoch: page.readUInt32LE(OFF.EPOCH),
    createdNs: page.readBigUInt64LE(OFF.CREATED),
    modifiedNs: page.readBigUInt64LE(OFF.MODIFIED),
    sigAlgo,
    sigLength,
    fileId: hex(page.subarray(OFF.FILE_ID, OFF.FILE_ID + 16)),
    parentId: hex(page.subarray(OFF.PARENT_ID, OFF.PARENT_ID + 16)),
    parentHash: hex(page.subarray(OFF.PARENT_HASH, OFF.PARENT_HASH + 32)),
    lineageDepth: page.readUInt32LE(OFF.LINEAGE_DEPTH),
    storedChecksum,
    computedChecksum,
    checksumValid: storedChecksum === computedChecksum,
    magicValid: page.subarray(0, 4).equals(ROOT_MANIFEST_MAGIC_BYTES),
    signaturePresent: sigAlgo !== 0 && sigLength > 0,
  };
}

/** Parse one 64-byte segment header at `offset` within `buf`. */
export function parseSegmentHeader(buf: Buffer, offset: number): RvfSegmentHeader {
  if (offset + SEGMENT_HEADER_SIZE > buf.length) {
    throw new RangeError(`segment header at ${offset} extends past end of buffer`);
  }
  const segType = buf[offset + 0x05];
  const flags = buf.readUInt16LE(offset + 0x06);
  return {
    signed: (flags & SEGMENT_FLAG_SIGNED) !== 0,
    encrypted: (flags & SEGMENT_FLAG_ENCRYPTED) !== 0,
    offset,
    magicValid: buf.subarray(offset, offset + 4).equals(SEGMENT_MAGIC_BYTES),
    version: buf[offset + 0x04],
    segType,
    segTypeName: SEGMENT_TYPE_NAMES[segType] ?? `unknown(0x${segType.toString(16)})`,
    flags,
    segmentId: buf.readBigUInt64LE(offset + 0x08),
    payloadLength: buf.readBigUInt64LE(offset + 0x10),
    timestampNs: buf.readBigUInt64LE(offset + 0x18),
    checksumAlgo: buf[offset + 0x20],
    compression: buf[offset + 0x21],
    contentHash: hex(buf.subarray(offset + 0x28, offset + 0x38)),
    executable: segType in EXECUTABLE_SEGMENT_TYPES,
  };
}

/** Round `n` up to the next 64-byte segment boundary. */
export const alignUp = (n: number): number =>
  n % SEGMENT_ALIGNMENT === 0 ? n : n + (SEGMENT_ALIGNMENT - (n % SEGMENT_ALIGNMENT));
