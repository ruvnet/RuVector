import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ForgeError, ForgeErrorCode } from '../src/errors';
import { sha256Buffer } from '../src/hash';
import { ROOT_MANIFEST_SIZE } from '../src/rvf';
import { validateRvf } from '../src/validate';
import {
  EXECUTABLE_FIXTURE,
  MINIMAL_FIXTURE,
  buildRvfFixture,
  resealRootManifest,
} from './fixtures/rvf-fixture';

let dir: string;

beforeAll(async () => {
  dir = await mkdtemp(join(tmpdir(), 'forge-validate-'));
});

afterAll(async () => {
  await rm(dir, { recursive: true, force: true });
});

async function write(name: string, bytes: Buffer): Promise<string> {
  const path = join(dir, name);
  await writeFile(path, bytes);
  return path;
}

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

describe('the committed fixture', () => {
  it('matches what the builder produces, so the two cannot drift', async () => {
    const committed = await readFile(join(__dirname, 'fixtures', 'minimal.rvf'));
    expect(committed.equals(buildRvfFixture(MINIMAL_FIXTURE))).toBe(true);
  });

  it('is small enough to be obviously not a model', async () => {
    const committed = await readFile(join(__dirname, 'fixtures', 'minimal.rvf'));
    expect(committed.length).toBeLessThan(16 * 1024);
  });
});

describe('valid RVF', () => {
  it('accepts the minimal fixture and reports its structure', async () => {
    const path = await write('valid.rvf', buildRvfFixture());
    const result = await validateRvf(path);

    expect(result.ok).toBe(true);
    expect(result.root.formatVersion).toBe(1);
    expect(result.root.fileId).toBe(MINIMAL_FIXTURE.fileId);
    expect(result.root.signaturePresent).toBe(true);
    expect(result.segments.count).toBe(MINIMAL_FIXTURE.segments.length);
    expect(result.segments.executableCount).toBe(0);
    expect(result.segments.byType).toEqual({ vec: 1, index: 1, manifest: 1 });
  });

  it('skips whole-file hashing unless --deep is requested', async () => {
    const bytes = buildRvfFixture();
    const path = await write('deep.rvf', bytes);

    expect((await validateRvf(path)).identity.sha256).toBe('');
    expect((await validateRvf(path, { deep: true })).identity.sha256).toBe(sha256Buffer(bytes));
  });

  it('validates well inside the two-second budget', async () => {
    const path = await write('fast.rvf', buildRvfFixture());
    expect((await validateRvf(path)).durationMs).toBeLessThan(2000);
  });
});

describe('rejects malformed files', () => {
  it('reports a missing file', async () => {
    await expectCode(validateRvf(join(dir, 'nope.rvf')), ForgeErrorCode.IO);
  });

  it('reports a file smaller than the root manifest', async () => {
    const path = await write('tiny.rvf', Buffer.alloc(128));
    await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
  });

  it('reports a truncated file whose root manifest was cut off', async () => {
    const bytes = buildRvfFixture();
    const path = await write('truncated.rvf', bytes.subarray(0, bytes.length - 512));
    await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
  });

  it('reports bad root-manifest magic', async () => {
    const bytes = buildRvfFixture();
    bytes.write('XXXX', bytes.length - ROOT_MANIFEST_SIZE, 'ascii');
    const path = await write('bad-magic.rvf', resealRootManifest(bytes));
    const err = await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
    expect(err.message).toMatch(/magic/i);
  });

  it('reports a tampered root manifest through its CRC32C', async () => {
    const bytes = buildRvfFixture();
    // Flip a byte in the manifest without resealing: the checksum still
    // covers the original content, so the change is detectable.
    bytes[bytes.length - ROOT_MANIFEST_SIZE + 0x20] ^= 0xff;
    const path = await write('tampered.rvf', bytes);
    const err = await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
    expect(err.message).toMatch(/CRC32C/);
  });

  it('reports bad segment magic', async () => {
    const bytes = buildRvfFixture();
    bytes.write('ZZZZ', 0, 'ascii');
    const path = await write('bad-segment.rvf', bytes);
    const err = await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
    expect(err.message).toMatch(/segment magic/i);
  });

  it('reports a segment whose payload runs past the segment area', async () => {
    const bytes = buildRvfFixture();
    bytes.writeBigUInt64LE(1n << 20n, 0x10); // first segment claims 1 MiB
    const path = await write('overlong-segment.rvf', bytes);
    await expectCode(validateRvf(path), ForgeErrorCode.INVALID_RVF);
  });

  it('reports a Level-1 pointer that runs past the end of the file', async () => {
    const bytes = buildRvfFixture();
    const root = bytes.length - ROOT_MANIFEST_SIZE;
    bytes.writeBigUInt64LE(BigInt(bytes.length), root + 0x008);
    bytes.writeBigUInt64LE(4096n, root + 0x010);
    const path = await write('bad-l1.rvf', resealRootManifest(bytes));
    await expectCode(validateRvf(path), ForgeErrorCode.MANIFEST);
  });
});

describe('signature policy', () => {
  it('rejects unsigned executable segments by default', async () => {
    const bytes = buildRvfFixture({ ...EXECUTABLE_FIXTURE, signed: false });
    const path = await write('unsigned-exec.rvf', bytes);
    const err = await expectCode(validateRvf(path), ForgeErrorCode.UNSIGNED_SEGMENT);
    expect(err.details.executableSegments).toBe(1);
  });

  it('accepts them under --allow-unsigned but warns', async () => {
    const bytes = buildRvfFixture({ ...EXECUTABLE_FIXTURE, signed: false });
    const path = await write('unsigned-exec-ok.rvf', bytes);
    const result = await validateRvf(path, { allowUnsigned: true });
    expect(result.segments.executableCount).toBe(1);
    expect(result.warnings.join(' ')).toMatch(/development-only/);
  });

  it('accepts signed executable segments without a warning about signing', async () => {
    const path = await write('signed-exec.rvf', buildRvfFixture(EXECUTABLE_FIXTURE));
    const result = await validateRvf(path);
    expect(result.root.signaturePresent).toBe(true);
    expect(result.warnings.join(' ')).not.toMatch(/unsigned/);
  });
});
