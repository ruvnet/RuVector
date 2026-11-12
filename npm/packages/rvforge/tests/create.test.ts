import { verify as cryptoVerify } from 'node:crypto';
import { mkdir, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { main } from '../src/cli';
import { DEFAULT_RVF_FILENAME, createRvf, type CreateResult } from '../src/create';
import { rootManifestSigningBytes } from '../src/rvf-writer';
import { EXIT_CODES, ForgeError, ForgeErrorCode } from '../src/errors';
import { generatePublisherKey, loadPublisherKey, writePublisherKey, type PublisherKey } from '../src/keys';
import { writeProject, type RvforgeProject } from '../src/project';
import {
  ROOT_MANIFEST_SIZE,
  SEGMENT_FLAG_SIGNED,
  SEGMENT_HEADER_SIZE,
  parseRootManifest,
  parseSegmentHeader,
  parseSignatureFooter,
} from '../src/rvf';
import { validateRvf } from '../src/validate';
import { fixtureProject } from './fixtures/project-fixture';

let dir: string;
let key: PublisherKey;
let project: RvforgeProject;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'forge-create-'));
  const keyPath = join(dir, 'publisher.key');
  await writePublisherKey(keyPath, generatePublisherKey());
  key = await loadPublisherKey(keyPath);
  project = fixtureProject();
});

afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

let counter = 0;
const outPath = (name = 'agent'): string => join(dir, `${name}-${counter++}.rvf`);

const create = (overrides: Partial<Parameters<typeof createRvf>[0]> = {}): Promise<CreateResult> =>
  createRvf({ outPath: outPath(), project, key, ...overrides });

async function expectCode(promise: Promise<unknown>, code: ForgeErrorCode): Promise<ForgeError> {
  try {
    await promise;
  } catch (err) {
    expect(err).toBeInstanceOf(ForgeError);
    expect((err as ForgeError).code).toBe(code);
    return err as ForgeError;
  }
  throw new Error(`expected ${code}`);
}

/** Walk the segment area the way `validate` does, footers included. */
function segmentsOf(file: Buffer): { type: string; payload: Buffer; signed: boolean }[] {
  const bodySize = file.length - ROOT_MANIFEST_SIZE;
  const found: { type: string; payload: Buffer; signed: boolean }[] = [];
  let offset = 0;
  while (offset + SEGMENT_HEADER_SIZE <= bodySize) {
    const header = parseSegmentHeader(file, offset);
    expect(header.magicValid).toBe(true);
    const payloadStart = offset + SEGMENT_HEADER_SIZE;
    const payloadEnd = payloadStart + Number(header.payloadLength);
    found.push({
      type: header.segTypeName,
      payload: file.subarray(payloadStart, payloadEnd),
      signed: header.signed,
    });
    let end = payloadEnd;
    if (header.signed) {
      const footer = parseSignatureFooter(file, payloadEnd);
      expect(footer).toBeDefined();
      end += (footer as { footerLength: number }).footerLength;
    }
    offset = end % 64 === 0 ? end : end + (64 - (end % 64));
  }
  return found;
}

describe('createRvf', () => {
  it('writes a container the local validator accepts', async () => {
    const result = await create();
    const validated = await validateRvf(result.path, { deep: true });

    expect(validated.ok).toBe(true);
    expect(validated.identity.sha256).toBe(result.sha256);
    expect(validated.root.fileId).toBe(result.fileId);
    expect(validated.root.signaturePresent).toBe(true);
  });

  it('closes the file with a 4096-byte root manifest page at EOF', async () => {
    const result = await create();
    const file = await readFile(result.path);

    expect(file.length % 64).toBe(0);
    const page = parseRootManifest(file.subarray(file.length - ROOT_MANIFEST_SIZE));
    expect(page.magicValid).toBe(true);
    expect(page.checksumValid).toBe(true);
    expect(page.version).toBe(1);
  });

  it("points the root page's L1 pointer at the manifest segment", async () => {
    const result = await create();
    const file = await readFile(result.path);
    const page = parseRootManifest(file.subarray(file.length - ROOT_MANIFEST_SIZE));

    expect(page.l1ManifestLength).toBeGreaterThan(0n);
    expect(page.l1ManifestOffset + page.l1ManifestLength).toBeLessThanOrEqual(BigInt(file.length));
    const header = parseSegmentHeader(file, Number(page.l1ManifestOffset));
    expect(header.segTypeName).toBe('manifest');
  });

  it('ends the segment stream with a signed MANIFEST segment', async () => {
    const result = await create();
    const segments = segmentsOf(await readFile(result.path));

    expect(segments[segments.length - 1].type).toBe('manifest');
    expect(segments[segments.length - 1].signed).toBe(true);
  });

  it('declares the project capabilities in a META segment', async () => {
    const result = await create();
    const meta = segmentsOf(await readFile(result.path)).find((s) => s.type === 'meta');

    expect(meta).toBeDefined();
    expect((meta as { payload: Buffer }).payload.toString('utf8')).toBe(
      `rvf.capabilities=${result.declaredCapabilities.join(',')}\n`,
    );
    expect(result.declaredCapabilities).toEqual([...result.declaredCapabilities].sort());
  });

  it('is deterministic: the same inputs produce identical bytes', async () => {
    const one = await create();
    const two = await create();

    expect(one.sha256).toBe(two.sha256);
    expect(await readFile(one.path)).toEqual(await readFile(two.path));
  });

  it('derives the file id from the project identity, not the clock', async () => {
    const first = await create();
    const renamed = await create({
      project: { ...project, version: '9.9.9' },
    });

    expect(first.fileId).toHaveLength(32);
    expect(renamed.fileId).not.toBe(first.fileId);
  });
});

describe('signing', () => {
  it("signs the root manifest page so the publisher's key verifies it", async () => {
    const result = await create();
    const file = await readFile(result.path);
    const page = file.subarray(file.length - ROOT_MANIFEST_SIZE);

    // sig_algo 1 / sig_length 64 is Ed25519 in the Level-0 manifest.
    expect(page.readUInt16LE(0x098)).toBe(1);
    expect(page.readUInt16LE(0x09a)).toBe(64);

    const signature = page.subarray(0x09c, 0x09c + 64);
    const ok = cryptoVerify(null, rootManifestSigningBytes(page), key.publicKeyObject, signature);
    expect(ok).toBe(true);
  });

  it('refuses a root manifest signature over tampered bytes', async () => {
    const result = await create();
    const file = await readFile(result.path);
    const page = Buffer.from(file.subarray(file.length - ROOT_MANIFEST_SIZE));

    page[0x004] = 7; // rewrite the format version
    const signature = page.subarray(0x09c, 0x09c + 64);
    expect(cryptoVerify(null, rootManifestSigningBytes(page), key.publicKeyObject, signature)).toBe(
      false,
    );
  });

  it('writes an unsigned container when no key is supplied', async () => {
    const result = await createRvf({ outPath: outPath('dev'), project });
    const file = await readFile(result.path);

    expect(result.signed).toBe(false);
    expect(result.keyId).toBeUndefined();
    expect(parseRootManifest(file.subarray(file.length - ROOT_MANIFEST_SIZE)).signaturePresent).toBe(
      false,
    );
    expect(segmentsOf(file).every((s) => !s.signed)).toBe(true);
  });

  it('never puts key material in the result', async () => {
    const result = await create();
    const serialised = JSON.stringify(result);

    expect(serialised).not.toContain('PRIVATE');
    expect(serialised).toContain(key.keyId);
  });

  it('sets the SIGNED flag and a well-formed footer on signed segments', async () => {
    const result = await create();
    const file = await readFile(result.path);
    const manifestOffset = Number(
      parseRootManifest(file.subarray(file.length - ROOT_MANIFEST_SIZE)).l1ManifestOffset,
    );
    const header = parseSegmentHeader(file, manifestOffset);

    expect(header.flags & SEGMENT_FLAG_SIGNED).toBe(SEGMENT_FLAG_SIGNED);
    const footer = parseSignatureFooter(
      file,
      manifestOffset + SEGMENT_HEADER_SIZE + Number(header.payloadLength),
    );
    expect(footer).toEqual({ sigAlgo: 0, sigLength: 64, footerLength: 72 });
  });
});

describe('--from', () => {
  let payloadDir: string;

  beforeAll(async () => {
    payloadDir = join(dir, 'payload');
    await mkdir(join(payloadDir, 'nested'), { recursive: true });
    await writeFile(join(payloadDir, 'agent.wasm'), Buffer.from('\0asm\x01\0\0\0', 'binary'));
    await writeFile(join(payloadDir, 'nested', 'weights.bin'), Buffer.alloc(32, 7));
  });

  it('turns .wasm into an executable segment and other files into opaque payload', async () => {
    const result = await create({ fromDir: payloadDir });
    const types = segmentsOf(await readFile(result.path)).map((s) => s.type);

    expect(types).toEqual(['meta', 'wasm', 'vec', 'manifest']);
    expect(result.segments.map((s) => s.source)).toEqual([
      undefined,
      'agent.wasm',
      'nested/weights.bin',
      undefined,
    ]);
  });

  it('signs the executable segment, so validation accepts it', async () => {
    const result = await create({ fromDir: payloadDir });
    const wasm = result.segments.find((s) => s.type === 'wasm');

    expect(wasm?.signed).toBe(true);
    const validated = await validateRvf(result.path);
    expect(validated.segments.executableCount).toBe(1);
    expect(validated.segments.unsignedExecutableCount).toBe(0);
  });

  it('produces an unsigned executable that validation refuses', async () => {
    const result = await createRvf({ outPath: outPath('dev'), project, fromDir: payloadDir });

    await expectCode(validateRvf(result.path), ForgeErrorCode.UNSIGNED_SEGMENT);
    // The same file is accepted once the caller opts into a development build.
    await expect(validateRvf(result.path, { allowUnsigned: true })).resolves.toMatchObject({
      ok: true,
    });
  });

  it('reports a missing payload directory as not found', async () => {
    await expectCode(create({ fromDir: join(dir, 'no-such-dir') }), ForgeErrorCode.NOT_FOUND);
  });

  it('reports an empty payload directory rather than writing nothing', async () => {
    const empty = join(dir, 'empty');
    await mkdir(empty, { recursive: true });
    const err = await expectCode(create({ fromDir: empty }), ForgeErrorCode.NOT_FOUND);
    expect(err.message).toContain('Omit --from');
  });
});

describe('overwrite protection', () => {
  it('refuses to overwrite an existing file', async () => {
    const path = outPath();
    await createRvf({ outPath: path, project, key });
    const err = await expectCode(
      createRvf({ outPath: path, project, key }),
      ForgeErrorCode.IO,
    );
    expect(err.message).toContain('--force');
  });

  it('overwrites when --force is given', async () => {
    const path = outPath();
    await createRvf({ outPath: path, project, key });
    await expect(createRvf({ outPath: path, project, key, force: true })).resolves.toMatchObject({
      path,
    });
  });
});

/**
 * Command-level coverage.
 *
 * Every path is passed explicitly rather than relying on the process working
 * directory: `main` resolves relative paths against `process.cwd()`, and a
 * test that changed the working directory would write its scaffold into the
 * package under test if the change did not take.
 */
describe('the create command', () => {
  let workdir: string;
  let out: string[];

  const run = async (argv: string[]): Promise<Record<string, never> & { ok: boolean; data?: never }> => {
    out = [];
    const code = await main(argv);
    return { ...JSON.parse(out[out.length - 1]), exit: code };
  };

  beforeEach(async () => {
    workdir = await mkdtemp(join(tmpdir(), 'forge-create-cmd-'));
    jest.spyOn(process.stdout, 'write').mockImplementation((text) => {
      out.push(String(text));
      return true;
    });
    jest.spyOn(process.stderr, 'write').mockImplementation(() => true);
  });

  afterEach(async () => {
    jest.restoreAllMocks();
    await rm(workdir, { recursive: true, force: true });
  });

  it('writes a signed container from what init --keygen scaffolded', async () => {
    const init = (await run([
      'init',
      '--json',
      '--keygen',
      '--config',
      join(workdir, 'forge.config.json'),
      '--project',
      join(workdir, 'rvforge.json'),
    ])) as unknown as { ok: boolean };
    expect(init.ok).toBe(true);

    const created = (await run([
      'create',
      join(workdir, DEFAULT_RVF_FILENAME),
      '--json',
      '--project',
      join(workdir, 'rvforge.json'),
    ])) as unknown as { ok: boolean; data: { path: string; signed: boolean } };

    expect(created.ok).toBe(true);
    expect(created.data.path).toBe(join(workdir, DEFAULT_RVF_FILENAME));
    expect(created.data.signed).toBe(true);
  });

  it('reports a missing signing key rather than writing an unsigned file', async () => {
    await writeProject(join(workdir, 'rvforge.json'), fixtureProject());
    const result = (await run([
      'create',
      join(workdir, 'agent.rvf'),
      '--json',
      '--project',
      join(workdir, 'rvforge.json'),
    ])) as unknown as { exit: number; error: { code: string; message: string } };

    expect(result.exit).toBe(EXIT_CODES[ForgeErrorCode.KEY]);
    expect(result.error.code).toBe(ForgeErrorCode.KEY);
    expect(result.error.message).toContain('--unsigned');
  });

  it('writes an unsigned container when asked, with no key present', async () => {
    await writeProject(join(workdir, 'rvforge.json'), fixtureProject());
    const result = (await run([
      'create',
      join(workdir, 'agent.rvf'),
      '--unsigned',
      '--json',
      '--project',
      join(workdir, 'rvforge.json'),
    ])) as unknown as { ok: boolean; data: { signed: boolean } };

    expect(result.ok).toBe(true);
    expect(result.data.signed).toBe(false);
  });
});
