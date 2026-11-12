/**
 * `rvforge create` — write a valid, signed `.rvf`.
 *
 * Every other command in this package reads an RVF. This one writes the first
 * one, which is what makes the documented sequence runnable: `init --keygen`
 * produces a key and a project file, and until now nothing produced the
 * artifact they describe.
 *
 * # Writing is not running
 *
 * Payload bytes are read, length-checked, and hashed. They are never parsed as
 * code, instantiated, linked, or executed, whatever segment type they are
 * labelled with (ADR-283 invariant 2). `--from` copies files into segments; it
 * does not build, compile, or evaluate them.
 *
 * # The format
 *
 * The bytes are spelled by {@link ./rvf-writer}, which mirrors
 * `crates/rvf-forge-core/src/author/` field for field. This module decides
 * *what* goes in a container; that one decides how it is encoded:
 *
 * ```text
 * [ META segment: capability declaration   ]  (when the project requests any)
 * [ payload segments from --from, sorted   ]
 * [ MANIFEST segment: the root manifest    ]
 * [ 4096-byte Level-0 root manifest page   ]  <- at EOF, always
 * ```
 *
 * Content hashes are SHAKE-256 truncated to 128 bits (`checksum_algo` 2)
 * because that is the one 128-bit option both Node and the Rust crate compute
 * natively; the format's XXH3-128 default would mean hand-porting a hash.
 *
 * # Determinism
 *
 * The same project, key, and inputs produce byte-identical output. Nothing
 * here reads the clock: segment timestamps and the root page's created and
 * modified fields are zero, segment ids are assigned by position, payload
 * files are visited in sorted order, and the file id is derived from the
 * project's identity rather than randomly generated.
 */

import { readFile, readdir, stat, writeFile } from 'node:fs/promises';
import { join, relative, resolve, sep } from 'node:path';

import { ForgeError, ForgeErrorCode } from './errors';
import { formatBytes } from './estimate';
import { canonicalJson, sha256Buffer } from './hash';
import { loadPublisherKey, type PublisherKey } from './keys';
import { MANIFEST_CAPABILITY_CLASSES, PROJECT_FILENAME, type RvforgeProject } from './project';
import {
  SEGMENT_TYPE,
  rootManifestPage,
  segmentTypeName,
  writeSegment,
} from './rvf-writer';

/** Default output path, and the name every other command's examples use. */
export const DEFAULT_RVF_FILENAME = 'agent.rvf';

/** The `META` key under which an RVF declares its capability classes. */
const CAPABILITY_DECLARATION_KEY = 'rvf.capabilities';

/** Cap on `--from` inputs, so a mistyped directory cannot be read forever. */
const MAX_PAYLOAD_FILES = 4096;

export interface CreateOptions {
  /** Where to write. Defaults to `agent.rvf` in the working directory. */
  outPath: string;
  project: RvforgeProject;
  /** Loaded publisher key, or `undefined` for an unsigned container. */
  key?: PublisherKey;
  /** Directory whose files become payload segments. */
  fromDir?: string;
  /** Overwrite an existing output file. */
  force?: boolean;
}

export interface CreatedSegment {
  type: string;
  /** Source file for a `--from` payload; absent for generated segments. */
  source?: string;
  payloadBytes: number;
  signed: boolean;
}

export interface CreateResult {
  path: string;
  size: number;
  fileId: string;
  sha256: string;
  signed: boolean;
  /** Key id of the signer. Never the key itself. */
  keyId?: string;
  segments: CreatedSegment[];
  declaredCapabilities: string[];
}

/**
 * Build and write an RVF.
 *
 * @throws {ForgeError} `FORGE_E_IO` if the output exists without `--force` or
 * cannot be written, `FORGE_E_NOT_FOUND` if `--from` names a directory with no
 * readable files, `FORGE_E_INVALID_RVF` if a payload is too large to encode.
 */
export async function createRvf(options: CreateOptions): Promise<CreateResult> {
  const outPath = resolve(options.outPath);
  await assertWritable(outPath, Boolean(options.force));

  const capabilities = declaredCapabilities(options.project);
  const payloads = options.fromDir ? await collectPayloads(resolve(options.fromDir)) : [];

  const planned: PlannedSegment[] = [];
  if (capabilities.length > 0) {
    planned.push({
      type: SEGMENT_TYPE.META,
      payload: Buffer.from(`${CAPABILITY_DECLARATION_KEY}=${capabilities.join(',')}\n`, 'utf8'),
    });
  }
  for (const payload of payloads) {
    planned.push({
      type: payload.path.toLowerCase().endsWith('.wasm') ? SEGMENT_TYPE.WASM : SEGMENT_TYPE.VEC,
      payload: payload.bytes,
      source: payload.path,
    });
  }
  planned.push({
    type: SEGMENT_TYPE.MANIFEST,
    payload: Buffer.from(
      `${canonicalJson(rootManifestBody(options.project, capabilities, planned))}\n`,
      'utf8',
    ),
  });

  const fileId = fileIdFor(options.project);
  const { body, manifestOffset, manifestLength, encoded } = encodeSegments(planned, options.key);
  const page = rootManifestPage(fileId, manifestOffset, manifestLength, options.key);
  const bytes = Buffer.concat([body, page]);

  try {
    await writeFile(outPath, bytes, { flag: options.force ? 'w' : 'wx' });
  } catch (err) {
    throw new ForgeError(ForgeErrorCode.IO, `Cannot write ${outPath}: ${errText(err)}`, {
      path: outPath,
    });
  }

  return {
    path: outPath,
    size: bytes.length,
    fileId: fileId.toString('hex'),
    sha256: sha256Buffer(bytes),
    signed: Boolean(options.key),
    ...(options.key ? { keyId: options.key.keyId } : {}),
    segments: encoded,
    declaredCapabilities: capabilities,
  };
}

/**
 * Resolve which key signs a new container.
 *
 * `--key-file` wins over the path recorded in the project so a publisher can
 * sign with a release key without editing `rvforge.json`. Only the path is
 * resolved here; the key itself never leaves {@link loadPublisherKey}.
 *
 * @throws {ForgeError} `FORGE_E_KEY` when neither source names a key.
 */
export async function resolveSigningKey(
  project: RvforgeProject,
  keyFile?: string,
): Promise<PublisherKey> {
  const path = keyFile ?? project.publisher.keyFile ?? undefined;
  if (!path) {
    throw new ForgeError(
      ForgeErrorCode.KEY,
      `No signing key: ${PROJECT_FILENAME} records no publisher.keyFile and --key-file was not given. ` +
        'Run "rvforge init --keygen", pass --key-file <path>, or pass --unsigned for a development build.',
    );
  }
  return loadPublisherKey(resolve(path));
}

/** The human-readable summary the `create` command prints. */
export function renderCreateSummary(result: CreateResult, operand: string): string[] {
  return [
    `Wrote ${result.path} (${formatBytes(result.size)})`,
    `  file id      ${result.fileId}`,
    `  sha256       ${result.sha256}`,
    `  segments     ${result.segments.length} (${result.segments.map((s) => s.type).join(', ')})`,
    `  capabilities ${result.declaredCapabilities.join(', ') || 'none declared (default-deny)'}`,
    // The key id, never the key: nothing in forge prints key material.
    result.signed ? `  signed by    ${result.keyId}` : '  signed       no (development build)',
    `Next: rvforge validate ${operand} --deep`,
  ];
}

interface PlannedSegment {
  type: number;
  payload: Buffer;
  source?: string;
}

/** Segment types that carry a signature footer when a key is supplied. */
const isSignable = (type: number): boolean =>
  type === SEGMENT_TYPE.WASM || type === SEGMENT_TYPE.MANIFEST;

function encodeSegments(
  planned: readonly PlannedSegment[],
  key: PublisherKey | undefined,
): { body: Buffer; manifestOffset: number; manifestLength: number; encoded: CreatedSegment[] } {
  const chunks: Buffer[] = [];
  const encoded: CreatedSegment[] = [];
  let offset = 0;
  let manifestOffset = 0;
  let manifestLength = 0;

  planned.forEach((segment, index) => {
    const signer = key && isSignable(segment.type) ? key : undefined;
    // Segment ids are 1-based and assigned by position, matching author/mod.rs.
    const bytes = writeSegment(segment.type, segment.payload, index + 1, signer);
    if (segment.type === SEGMENT_TYPE.MANIFEST) {
      manifestOffset = offset;
      manifestLength = bytes.length;
    }
    chunks.push(bytes);
    offset += bytes.length;
    encoded.push({
      type: segmentTypeName(segment.type),
      ...(segment.source ? { source: segment.source } : {}),
      payloadBytes: segment.payload.length,
      signed: Boolean(signer),
    });
  });

  return { body: Buffer.concat(chunks), manifestOffset, manifestLength, encoded };
}

/**
 * The capability classes to declare, deduplicated and in the canonical class
 * order so the output does not depend on how the project file was written.
 */
function declaredCapabilities(project: RvforgeProject): string[] {
  const requested = new Set((project.capabilities?.requests ?? []).map((r) => r.class));
  return MANIFEST_CAPABILITY_CLASSES.filter((cls) => requested.has(cls));
}

/**
 * The MANIFEST segment's payload: what the artifact is, what it asks for, and
 * what it carries. Inert metadata — nothing reads it to decide behaviour.
 */
function rootManifestBody(
  project: RvforgeProject,
  capabilities: readonly string[],
  planned: readonly PlannedSegment[],
): Record<string, unknown> {
  return {
    schema: 'rvforge.agent-manifest/1',
    name: project.listing.name,
    displayName: project.listing.displayName,
    version: project.version,
    publisher: project.publisher.displayName,
    license: project.license,
    capabilities: [...capabilities],
    runtimeProfiles: [...project.runtimeRequirements.profiles],
    payloads: planned
      .filter((s) => s.source !== undefined)
      .map((s) => ({ source: s.source, bytes: s.payload.length })),
  };
}

/**
 * A deterministic 16-byte file id derived from the project's identity.
 *
 * Derived rather than random so that rebuilding the same agent produces the
 * same artifact; two different agents collide only if they share a name,
 * version, and publisher, at which point they are not distinguishable anyway.
 */
function fileIdFor(project: RvforgeProject): Buffer {
  const identity = `${project.listing.name}\n${project.version}\n${project.publisher.displayName}`;
  return Buffer.from(sha256Buffer(Buffer.from(identity, 'utf8')), 'hex').subarray(0, 16);
}

interface PayloadFile {
  /** Path relative to the `--from` directory, with `/` separators. */
  path: string;
  bytes: Buffer;
}

/**
 * Read every file under `dir`, in sorted order.
 *
 * Sorting is what makes `--from` deterministic: directory iteration order is
 * a filesystem detail, and letting it through would mean the same inputs
 * produced different artifacts on different machines.
 */
async function collectPayloads(dir: string): Promise<PayloadFile[]> {
  let entries: string[];
  try {
    entries = await walk(dir);
  } catch (err) {
    throw new ForgeError(
      ForgeErrorCode.NOT_FOUND,
      `Cannot read the payload directory ${dir}: ${errText(err)}`,
      { path: dir },
    );
  }
  if (entries.length === 0) {
    throw new ForgeError(
      ForgeErrorCode.NOT_FOUND,
      `The payload directory ${dir} holds no files. Omit --from to write a minimal agent.`,
      { path: dir },
    );
  }
  entries.sort();

  const files: PayloadFile[] = [];
  for (const absolute of entries) {
    files.push({
      path: relative(dir, absolute).split(sep).join('/'),
      bytes: await readFile(absolute),
    });
  }
  return files;
}

async function walk(dir: string): Promise<string[]> {
  const found: string[] = [];
  const queue = [dir];
  while (queue.length > 0) {
    const current = queue.shift() as string;
    for (const entry of await readdir(current, { withFileTypes: true })) {
      const absolute = join(current, entry.name);
      if (entry.isDirectory()) {
        queue.push(absolute);
      } else if (entry.isFile()) {
        if (found.length >= MAX_PAYLOAD_FILES) {
          throw new ForgeError(
            ForgeErrorCode.IO,
            `The payload directory holds more than ${MAX_PAYLOAD_FILES} files.`,
            { path: dir, limit: MAX_PAYLOAD_FILES },
          );
        }
        found.push(absolute);
      }
    }
  }
  return found;
}

async function assertWritable(path: string, force: boolean): Promise<void> {
  if (force) return;
  try {
    await stat(path);
  } catch {
    return; // Absent, which is what we want.
  }
  throw new ForgeError(
    ForgeErrorCode.IO,
    `${path} already exists. Pass --force to overwrite it.`,
    { path },
  );
}

const errText = (err: unknown): string => (err instanceof Error ? err.message : String(err));
