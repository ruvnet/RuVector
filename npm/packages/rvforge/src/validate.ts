/**
 * Local RVF validation.
 *
 * Reads structure only: file size, root-manifest magic and CRC32C, manifest
 * field sanity, a segment-header walk, and signature presence. It never
 * executes, links, or interprets RVF content (ADR-283 invariant 2).
 *
 * Budget: under two seconds for RVFs below 1 GB (requirements §10.1). That
 * holds because the default pass touches one 4 KiB page at EOF plus one
 * 64-byte header per segment; whole-file hashing happens only under `--deep`.
 */

import { open, stat, type FileHandle } from 'node:fs/promises';

import { ForgeError, ForgeErrorCode } from './errors';
import { sha256File } from './hash';
import {
  ROOT_MANIFEST_SIZE,
  SEGMENT_HEADER_SIZE,
  SIGNATURE_FOOTER_MIN_SIZE,
  MAX_SEGMENT_PAYLOAD,
  alignUp,
  parseRootManifest,
  parseSegmentHeader,
  parseSignatureFooter,
  type RvfRootManifest,
  type RvfSegmentHeader,
} from './rvf';
import type { RvfIdentityRecord } from './types';

/** Refuse to walk more than this many segments; a malformed file can loop. */
const MAX_SEGMENTS = 200_000;

export interface ValidateOptions {
  /** Hash the whole file to produce the canonical RVF identity. */
  deep?: boolean;
  /**
   * Accept executable segments that carry no root-manifest signature.
   * Development builds only — hosted submission always requires a signature.
   */
  allowUnsigned?: boolean;
}

export interface SegmentSummary {
  count: number;
  executableCount: number;
  /** Segments carrying an Ed25519 signature footer. */
  signedCount: number;
  /** Executable segments with no signature footer of their own. */
  unsignedExecutableCount: number;
  payloadBytes: number;
  byType: Record<string, number>;
  truncated: boolean;
}

export interface ValidateResult {
  path: string;
  size: number;
  ok: true;
  deep: boolean;
  /** `sha256` is the empty string unless `deep` was requested. */
  identity: RvfIdentityRecord;
  root: {
    formatVersion: number;
    flags: number;
    dimension: number;
    totalVectorCount: string;
    epoch: number;
    fileId: string;
    parentId: string;
    lineageDepth: number;
    signatureAlgo: number;
    signatureLength: number;
    signaturePresent: boolean;
  };
  segments: SegmentSummary;
  warnings: string[];
  durationMs: number;
}

/**
 * Validate the RVF at `path`.
 *
 * @throws {ForgeError} `FORGE_E_IO` if unreadable, `FORGE_E_INVALID_RVF` for a
 * structural failure, `FORGE_E_UNSIGNED_SEGMENT` when an executable segment is
 * present without a root-manifest signature.
 */
export async function validateRvf(path: string, options: ValidateOptions = {}): Promise<ValidateResult> {
  const startedAt = Date.now();
  const size = await statSize(path);

  if (size < ROOT_MANIFEST_SIZE) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `File is ${size} bytes; an RVF is at least ${ROOT_MANIFEST_SIZE} (the root manifest alone).`,
      { path, size, minimum: ROOT_MANIFEST_SIZE },
    );
  }

  let handle: FileHandle;
  try {
    handle = await open(path, 'r');
  } catch (err) {
    throw new ForgeError(ForgeErrorCode.IO, `Cannot open ${path}: ${errText(err)}`, { path });
  }

  try {
    const root = await readRootManifest(handle, path, size);
    const warnings = checkManifestSanity(root, size, path);
    const segments = await walkSegments(handle, size - ROOT_MANIFEST_SIZE, path);

    // An executable segment is covered either by its own signature footer or
    // by a signed root manifest. Only a segment with neither is unsigned.
    const uncovered = root.signaturePresent ? 0 : segments.unsignedExecutableCount;
    if (uncovered > 0 && !options.allowUnsigned) {
      throw new ForgeError(
        ForgeErrorCode.UNSIGNED_SEGMENT,
        `RVF contains ${uncovered} executable segment(s) with no signature, and its root manifest carries none either. ` +
          'Sign the RVF, or pass --allow-unsigned for a development build.',
        { path, executableSegments: segments.executableCount, unsignedExecutableSegments: uncovered },
      );
    }
    if (uncovered > 0) {
      warnings.push(
        `${uncovered} executable segment(s) are unsigned; the resulting build is development-only.`,
      );
    }

    const sha256 = options.deep ? await sha256File(path) : '';

    return {
      path,
      size,
      ok: true,
      deep: Boolean(options.deep),
      identity: {
        sha256,
        size,
        fileId: root.fileId,
        formatVersion: root.version,
        signatureAlgo: root.sigAlgo,
        signaturePresent: root.signaturePresent,
      },
      root: {
        formatVersion: root.version,
        flags: root.flags,
        dimension: root.dimension,
        totalVectorCount: root.totalVectorCount.toString(),
        epoch: root.epoch,
        fileId: root.fileId,
        parentId: root.parentId,
        lineageDepth: root.lineageDepth,
        signatureAlgo: root.sigAlgo,
        signatureLength: root.sigLength,
        signaturePresent: root.signaturePresent,
      },
      segments,
      warnings,
      durationMs: Date.now() - startedAt,
    };
  } finally {
    await handle.close().catch(() => undefined);
  }
}

async function statSize(path: string): Promise<number> {
  try {
    const info = await stat(path);
    if (!info.isFile()) {
      throw new ForgeError(ForgeErrorCode.INVALID_RVF, `${path} is not a regular file.`, { path });
    }
    return info.size;
  } catch (err) {
    if (err instanceof ForgeError) throw err;
    throw new ForgeError(ForgeErrorCode.IO, `Cannot stat ${path}: ${errText(err)}`, { path });
  }
}

async function readRootManifest(handle: FileHandle, path: string, size: number): Promise<RvfRootManifest> {
  const page = Buffer.alloc(ROOT_MANIFEST_SIZE);
  const { bytesRead } = await handle.read(page, 0, ROOT_MANIFEST_SIZE, size - ROOT_MANIFEST_SIZE);
  if (bytesRead !== ROOT_MANIFEST_SIZE) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `Truncated root manifest: read ${bytesRead} of ${ROOT_MANIFEST_SIZE} bytes at EOF.`,
      { path, bytesRead },
    );
  }

  const root = parseRootManifest(page);
  if (!root.magicValid) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `Bad root manifest magic 0x${root.magic.toString(16).padStart(8, '0')}; expected 0x52564d30 ("RVM0").`,
      { path, magic: root.magic },
    );
  }
  if (!root.checksumValid) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      'Root manifest CRC32C mismatch — the file is corrupt or has been modified.',
      { path, stored: root.storedChecksum, computed: root.computedChecksum },
    );
  }
  if (root.version === 0) {
    throw new ForgeError(ForgeErrorCode.INVALID_RVF, 'Root manifest declares format version 0.', { path });
  }
  return root;
}

function checkManifestSanity(root: RvfRootManifest, size: number, path: string): string[] {
  const warnings: string[] = [];
  const bodyLimit = BigInt(size - ROOT_MANIFEST_SIZE);

  if (root.l1ManifestOffset + root.l1ManifestLength > BigInt(size)) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      'Level-1 manifest pointer runs past the end of the file.',
      {
        path,
        l1Offset: root.l1ManifestOffset.toString(),
        l1Length: root.l1ManifestLength.toString(),
        size,
      },
    );
  }
  if (root.l1ManifestLength === 0n) {
    warnings.push('Root manifest declares an empty Level-1 manifest.');
  }
  if (root.totalVectorCount > 0n && root.dimension === 0) {
    throw new ForgeError(
      ForgeErrorCode.MANIFEST,
      `Root manifest declares ${root.totalVectorCount} vectors with dimension 0.`,
      { path, totalVectorCount: root.totalVectorCount.toString() },
    );
  }
  if (bodyLimit === 0n) {
    warnings.push('File contains a root manifest and no segments.');
  }
  if (!root.signaturePresent) {
    warnings.push('Root manifest is unsigned (sig_algo=0 or sig_length=0).');
  }
  return warnings;
}

async function walkSegments(handle: FileHandle, bodySize: number, path: string): Promise<SegmentSummary> {
  const summary: SegmentSummary = {
    count: 0,
    executableCount: 0,
    signedCount: 0,
    unsignedExecutableCount: 0,
    payloadBytes: 0,
    byType: {},
    truncated: false,
  };

  const header = Buffer.alloc(SEGMENT_HEADER_SIZE);
  let offset = 0;

  while (offset + SEGMENT_HEADER_SIZE <= bodySize) {
    if (summary.count >= MAX_SEGMENTS) {
      summary.truncated = true;
      break;
    }
    const { bytesRead } = await handle.read(header, 0, SEGMENT_HEADER_SIZE, offset);
    if (bytesRead !== SEGMENT_HEADER_SIZE) {
      throw new ForgeError(
        ForgeErrorCode.INVALID_RVF,
        `Truncated segment header at byte ${offset}.`,
        { path, offset },
      );
    }

    const seg = parseSegmentHeader(header, 0);
    if (!seg.magicValid) {
      throw new ForgeError(
        ForgeErrorCode.INVALID_RVF,
        `Bad segment magic at byte ${offset}; expected 0x52564653 ("RVFS").`,
        { path, offset, segmentIndex: summary.count },
      );
    }
    assertPayloadFits(seg, offset, bodySize, path, summary.count);

    const payloadEnd = offset + SEGMENT_HEADER_SIZE + Number(seg.payloadLength);
    const footerLength = seg.signed
      ? await readFooterLength(handle, payloadEnd, bodySize, path, summary.count)
      : 0;

    summary.count += 1;
    summary.payloadBytes += Number(seg.payloadLength);
    summary.byType[seg.segTypeName] = (summary.byType[seg.segTypeName] ?? 0) + 1;
    if (seg.signed) summary.signedCount += 1;
    if (seg.executable) {
      summary.executableCount += 1;
      if (!seg.signed) summary.unsignedExecutableCount += 1;
    }

    // The next segment starts after the footer, not after the payload. Missing
    // this lands the walk in the middle of a signature and reports bad magic.
    offset = alignUp(payloadEnd + footerLength);
  }

  return summary;
}

/**
 * Size of the signature footer at `payloadEnd`.
 *
 * The footer is read rather than assumed: its length depends on the signature
 * algorithm, and a footer whose declared length disagrees with the signature
 * it carries is a malformed container, not a differently-sized one.
 */
async function readFooterLength(
  handle: FileHandle,
  payloadEnd: number,
  bodySize: number,
  path: string,
  index: number,
): Promise<number> {
  const bad = (detail: string): ForgeError =>
    new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `Segment ${index} sets the SIGNED flag but ${detail}.`,
      { path, offset: payloadEnd, segmentIndex: index },
    );

  const head = Buffer.alloc(SIGNATURE_FOOTER_MIN_SIZE);
  const { bytesRead } = await handle.read(head, 0, head.length, payloadEnd);
  if (bytesRead !== head.length) throw bad('its footer is truncated');

  const sigLength = head.readUInt16LE(2);
  const full = Buffer.alloc(4 + sigLength + 4);
  if (payloadEnd + full.length > bodySize) throw bad('its footer runs past the segment area');
  const read = await handle.read(full, 0, full.length, payloadEnd);
  if (read.bytesRead !== full.length) throw bad('its footer is truncated');

  const footer = parseSignatureFooter(full, 0);
  if (!footer) throw bad('its footer is malformed or declares a zero-length signature');
  return footer.footerLength;
}

function assertPayloadFits(
  seg: RvfSegmentHeader,
  offset: number,
  bodySize: number,
  path: string,
  index: number,
): void {
  if (seg.payloadLength > BigInt(MAX_SEGMENT_PAYLOAD)) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `Segment ${index} declares a ${seg.payloadLength}-byte payload, above the 4 GiB per-segment limit.`,
      { path, offset, payloadLength: seg.payloadLength.toString() },
    );
  }
  const end = BigInt(offset + SEGMENT_HEADER_SIZE) + seg.payloadLength;
  if (end > BigInt(bodySize)) {
    throw new ForgeError(
      ForgeErrorCode.INVALID_RVF,
      `Segment ${index} at byte ${offset} declares a payload running ${end - BigInt(bodySize)} bytes past the segment area.`,
      { path, offset, payloadLength: seg.payloadLength.toString(), bodySize },
    );
  }
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err));
