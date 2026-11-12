/**
 * Synthetic RVF builder for tests.
 *
 * Emits the minimal structure `validateRvf` inspects — 64-byte segment headers
 * followed by a 4096-byte Level-0 root manifest at EOF — using the layout in
 * `crates/rvf/rvf-types/src/{constants,segment}.rs` and
 * `crates/rvf/rvf-manifest/src/level0.rs`.
 *
 * The payloads are zero bytes. Nothing here is a real model, and nothing here
 * is executable: the fixtures exist to exercise the parser's accept and reject
 * paths, not a runtime.
 */

import {
  ROOT_MANIFEST_MAGIC_BYTES,
  ROOT_MANIFEST_SIZE,
  SEGMENT_HEADER_SIZE,
  SEGMENT_MAGIC_BYTES,
  alignUp,
  crc32c,
} from '../../src/rvf';

export interface FixtureSegment {
  /** `SegmentType` discriminator, e.g. 0x01 vec, 0x05 manifest, 0x10 wasm. */
  type: number;
  /** Payload length in bytes; the payload itself is zero-filled. */
  payloadLength: number;
}

export interface FixtureSpec {
  segments: FixtureSegment[];
  /** Non-zero signature algorithm plus a signature blob when true. */
  signed: boolean;
  formatVersion: number;
  dimension: number;
  vectorCount: number;
  /** 16 bytes, lowercase hex. */
  fileId: string;
}

/** The fixture `tests/fixtures/minimal.rvf` is generated from. */
export const MINIMAL_FIXTURE: FixtureSpec = {
  segments: [
    { type: 0x01, payloadLength: 128 },
    { type: 0x02, payloadLength: 64 },
    { type: 0x05, payloadLength: 32 },
  ],
  signed: true,
  formatVersion: 1,
  dimension: 8,
  vectorCount: 4,
  fileId: '0123456789abcdef0123456789abcdef',
};

/** A fixture whose WASM segment makes it an executable-bearing RVF. */
export const EXECUTABLE_FIXTURE: FixtureSpec = {
  ...MINIMAL_FIXTURE,
  segments: [...MINIMAL_FIXTURE.segments, { type: 0x10, payloadLength: 96 }],
};

export function buildRvfFixture(spec: FixtureSpec = MINIMAL_FIXTURE): Buffer {
  const body = buildSegments(spec.segments);
  const root = buildRootManifest(spec, body.length);
  return Buffer.concat([body, root]);
}

function buildSegments(segments: readonly FixtureSegment[]): Buffer {
  const chunks: Buffer[] = [];
  segments.forEach((segment, index) => {
    const header = Buffer.alloc(SEGMENT_HEADER_SIZE);
    SEGMENT_MAGIC_BYTES.copy(header, 0);
    header[0x04] = 1; // version
    header[0x05] = segment.type;
    header.writeUInt16LE(0, 0x06); // flags
    header.writeBigUInt64LE(BigInt(index), 0x08); // segment_id
    header.writeBigUInt64LE(BigInt(segment.payloadLength), 0x10);
    header.writeBigUInt64LE(0n, 0x18); // timestamp_ns — zero keeps fixtures byte-stable
    header[0x20] = 0; // checksum_algo: CRC32C
    header[0x21] = 0; // compression: none

    const unpadded = SEGMENT_HEADER_SIZE + segment.payloadLength;
    chunks.push(header, Buffer.alloc(alignUp(unpadded) - SEGMENT_HEADER_SIZE));
  });
  return Buffer.concat(chunks);
}

function buildRootManifest(spec: FixtureSpec, bodyLength: number): Buffer {
  const page = Buffer.alloc(ROOT_MANIFEST_SIZE);
  ROOT_MANIFEST_MAGIC_BYTES.copy(page, 0x000);
  page.writeUInt16LE(spec.formatVersion, 0x004);
  page.writeUInt16LE(0, 0x006); // flags
  page.writeBigUInt64LE(0n, 0x008); // l1_manifest_offset
  page.writeBigUInt64LE(BigInt(Math.min(SEGMENT_HEADER_SIZE, bodyLength)), 0x010); // l1_manifest_length
  page.writeBigUInt64LE(BigInt(spec.vectorCount), 0x018);
  page.writeUInt16LE(spec.dimension, 0x020);
  page[0x022] = 1; // base_dtype: f32
  page[0x023] = 0; // profile_id
  page.writeUInt32LE(1, 0x024); // epoch
  page.writeBigUInt64LE(0n, 0x028); // created_ns
  page.writeBigUInt64LE(0n, 0x030); // modified_ns

  if (spec.signed) {
    page.writeUInt16LE(1, 0x098); // sig_algo: Ed25519
    page.writeUInt16LE(64, 0x09a); // sig_length
    Buffer.alloc(64, 0xab).copy(page, 0x09c); // placeholder signature bytes
  }

  Buffer.from(spec.fileId, 'hex').copy(page, 0xf00);
  page.writeUInt32LE(crc32c(page.subarray(0, 0xffc)), 0xffc);
  return page;
}

/** Recompute the root-manifest CRC32C in place after mutating a fixture. */
export function resealRootManifest(file: Buffer): Buffer {
  const rootStart = file.length - ROOT_MANIFEST_SIZE;
  const page = file.subarray(rootStart);
  page.writeUInt32LE(crc32c(page.subarray(0, 0xffc)), 0xffc);
  return file;
}
