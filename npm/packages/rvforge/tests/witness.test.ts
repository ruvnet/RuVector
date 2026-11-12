import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runBuild } from '../src/build';
import { defaultConfig } from '../src/config';
import { ForgeError, ForgeErrorCode } from '../src/errors';
import { runVerify } from '../src/verify';
import {
  RECEIPTS_FILENAME,
  createWitnessReceipt,
  lastReceiptFor,
  readReceipts,
  receiptContentId,
  receiptLine,
  verifyReceiptChain,
  type WitnessReceipt,
} from '../src/witness';
import type { ForgeConfig } from '../src/types';
import { buildRvfFixture } from './fixtures/rvf-fixture';

jest.setTimeout(30_000);

let root: string;
let rvfPath: string;

const config = (): ForgeConfig =>
  defaultConfig({
    app: {
      name: 'Fixture Agent',
      version: '0.1.0',
      publisher: 'RuVector Tests',
      identifier: 'dev.ruvector.fixture',
    },
    targets: ['linux-x64'],
  });

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'forge-witness-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

const build = (name: string) =>
  runBuild({ config: config(), rvfPath, outDir: join(root, name), force: true });

const receiptsPath = (dir: string): string => join(dir, RECEIPTS_FILENAME);

const rewrite = async (dir: string, receipts: WitnessReceipt[]): Promise<void> =>
  writeFile(receiptsPath(dir), receipts.map(receiptLine).join(''), 'utf8');

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

const receipt = (over: Partial<Parameters<typeof createWitnessReceipt>[0]> = {}): WitnessReceipt =>
  createWitnessReceipt({
    subject: 'a'.repeat(64),
    event: 'build',
    outcome: 'pass',
    actor: { kind: 'builder', id: 'test' },
    timestamp: '2026-08-03T00:00:00.000Z',
    ...over,
  });

describe('receipt identity', () => {
  it('ids a receipt by sha256 over its canonical JSON', () => {
    expect(receipt().receiptId).toMatch(/^sha256:[0-9a-f]{64}$/);
  });

  it('gives the same id to logically identical receipts', () => {
    expect(receipt().receiptId).toBe(receipt().receiptId);
  });

  it('changes the id when any content field changes', () => {
    const base = receipt();
    expect(receipt({ outcome: 'fail' }).receiptId).not.toBe(base.receiptId);
    expect(receipt({ event: 'verify' }).receiptId).not.toBe(base.receiptId);
    expect(receipt({ subject: 'b'.repeat(64) }).receiptId).not.toBe(base.receiptId);
    expect(receipt({ prevReceipt: base.receiptId }).receiptId).not.toBe(base.receiptId);
  });

  it('excludes signatures from the id so a countersignature cannot change it', () => {
    const base = receipt();
    const signed = { ...base, signatures: [{ keyId: 'ed25519:k', role: 'registry' as const, sig: 'b64' }] };
    const { receiptId: _id, ...content } = signed;
    expect(receiptContentId(content)).toBe(base.receiptId);
  });

  it('normalises a bare digest into sha256: form', () => {
    expect(receipt().subject).toBe(`sha256:${'a'.repeat(64)}`);
    expect(receipt({ subject: `sha256:${'a'.repeat(64)}` }).subject).toBe(`sha256:${'a'.repeat(64)}`);
  });

  it('emits an empty signature array — forge never signs', () => {
    expect(receipt().signatures).toEqual([]);
  });
});

describe('chain verification', () => {
  it('accepts a well-formed chain', () => {
    const first = receipt();
    const second = receipt({ event: 'verify', prevReceipt: first.receiptId });
    const check = verifyReceiptChain([first, second]);

    expect(check.ok).toBe(true);
    expect(check.count).toBe(2);
    expect(check.subjects).toEqual([{ subject: `sha256:${'a'.repeat(64)}`, length: 2 }]);
  });

  it('chains each subject independently', () => {
    const a1 = receipt();
    const b1 = receipt({ subject: 'b'.repeat(64) });
    const a2 = receipt({ event: 'verify', prevReceipt: a1.receiptId });
    expect(verifyReceiptChain([a1, b1, a2]).ok).toBe(true);
  });

  it('detects a link pointing at the wrong predecessor', () => {
    const first = receipt();
    const forged = receipt({ event: 'verify', prevReceipt: `sha256:${'9'.repeat(64)}` });
    const check = verifyReceiptChain([first, forged]);

    expect(check.ok).toBe(false);
    expect(check.problems.map((p) => p.reason)).toContain('broken-link');
  });

  it('detects a removed link', () => {
    const first = receipt();
    const second = receipt({ event: 'verify', prevReceipt: first.receiptId });
    const third = receipt({ event: 'verify', outcome: 'fail', prevReceipt: second.receiptId });
    expect(verifyReceiptChain([first, third]).problems[0].reason).toBe('broken-link');
  });

  it('detects a non-null prevReceipt on the first receipt for a subject', () => {
    expect(verifyReceiptChain([receipt({ prevReceipt: `sha256:${'1'.repeat(64)}` })]).problems[0].reason).toBe(
      'orphan-root',
    );
  });

  it('detects an edited receipt whose id no longer matches its content', () => {
    const original = receipt();
    const edited = { ...original, outcome: 'pass' as const, evidence: { tampered: true } };
    const check = verifyReceiptChain([edited]);

    expect(check.ok).toBe(false);
    expect(check.problems.map((p) => p.reason)).toContain('id-mismatch');
  });

  it('reports a malformed entry rather than throwing', () => {
    const check = verifyReceiptChain([{ nonsense: true } as unknown as WitnessReceipt]);
    expect(check.problems[0].reason).toBe('malformed');
  });

  it('treats an empty chain as intact', () => {
    expect(verifyReceiptChain([])).toEqual({ ok: true, count: 0, subjects: [], problems: [] });
  });
});

describe('build to verify continuity', () => {
  it('starts the chain at build and extends it at verify', async () => {
    const built = await build('witness-continuity');

    const afterBuild = await readReceipts(built.outDir);
    expect(afterBuild).toHaveLength(1);
    expect(afterBuild[0]).toMatchObject({
      type: 'witness-receipt',
      event: 'build',
      outcome: 'pass',
      prevReceipt: null,
      subject: `sha256:${built.manifest.rvf.sha256}`,
    });
    expect(afterBuild[0].receiptId).toBe(built.receipt.receiptId);

    const verified = await runVerify(built.outDir);
    expect(verified.witnessChain.count).toBe(1);
    expect(verified.receipt?.event).toBe('verify');

    const afterVerify = await readReceipts(built.outDir);
    expect(afterVerify).toHaveLength(2);
    expect(afterVerify[1].prevReceipt).toBe(afterBuild[0].receiptId);
    expect(verifyReceiptChain(afterVerify).ok).toBe(true);

    // A second verification keeps extending the same chain.
    await runVerify(built.outDir);
    const afterSecond = await readReceipts(built.outDir);
    expect(afterSecond).toHaveLength(3);
    expect(verifyReceiptChain(afterSecond).ok).toBe(true);
    expect(lastReceiptFor(afterSecond, built.manifest.rvf.sha256)).toBe(afterSecond[2].receiptId);
  });

  it('records a failed verification in the chain', async () => {
    const built = await build('witness-fail');
    await rm(join(built.outDir, 'inventory.json'));

    await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);

    const receipts = await readReceipts(built.outDir);
    expect(receipts).toHaveLength(2);
    expect(receipts[1]).toMatchObject({ event: 'verify', outcome: 'fail' });
    expect(verifyReceiptChain(receipts).ok).toBe(true);
  });

  it('keeps receipts.jsonl out of the provenance record', async () => {
    const built = await build('witness-not-in-provenance');
    expect(built.provenance.files.map((f) => f.path)).not.toContain(RECEIPTS_FILENAME);
    // Otherwise the first verify would invalidate the build it just checked.
    await expect(runVerify(built.outDir)).resolves.toMatchObject({ verified: true });
  });
});

describe('broken chain detection at verify', () => {
  it('refuses a chain whose link was rewritten', async () => {
    const built = await build('witness-broken-link');
    await runVerify(built.outDir);

    const receipts = await readReceipts(built.outDir);
    receipts[1].prevReceipt = `sha256:${'0'.repeat(64)}`;
    await rewrite(built.outDir, receipts);

    const err = await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(err.message).toMatch(/witness chain/);
    // Rewriting the link also changes the content, so both checks fire: the
    // recorded id no longer matches, and the link no longer resolves.
    expect(err.details.witnessProblems).toEqual([
      { index: 1, reason: 'id-mismatch' },
      { index: 1, reason: 'broken-link' },
    ]);
  });

  it('refuses a chain whose receipt content was edited', async () => {
    const built = await build('witness-edited');
    const receipts = await readReceipts(built.outDir);
    receipts[0].outcome = 'fail';
    await rewrite(built.outDir, receipts);

    const err = await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(err.details.witnessProblems).toEqual([{ index: 0, reason: 'id-mismatch' }]);
  });

  it('refuses a chain with a receipt removed from the middle', async () => {
    const built = await build('witness-removed');
    await runVerify(built.outDir);
    await runVerify(built.outDir);

    const receipts = await readReceipts(built.outDir);
    await rewrite(built.outDir, [receipts[0], receipts[2]]);

    await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
  });

  it('does not append onto a chain it just refused', async () => {
    const built = await build('witness-no-append-on-break');
    const receipts = await readReceipts(built.outDir);
    receipts[0].evidence = { tampered: true };
    await rewrite(built.outDir, receipts);

    await expectCode(runVerify(built.outDir), ForgeErrorCode.VERIFY_FAILED);
    expect(await readReceipts(built.outDir)).toHaveLength(1);
  });

  it('reads an absent receipts file as an empty chain', async () => {
    const built = await build('witness-absent');
    await rm(receiptsPath(built.outDir));
    const result = await runVerify(built.outDir, { noReceipt: true });
    expect(result.witnessChain).toEqual({ ok: true, count: 0, subjects: [], problems: [] });
    await expect(readFile(receiptsPath(built.outDir))).rejects.toThrow();
  });
});
