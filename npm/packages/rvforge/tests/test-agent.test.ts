import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { defaultDenials, type RvforgeProject } from '../src/project';
import { ROOT_MANIFEST_SIZE } from '../src/rvf';
import { QUARANTINE_REASON, runTestAgent, type TestAgentResult } from '../src/test-agent';
import { appendReceipt, receiptLine, readReceipts } from '../src/witness';
import { buildRvfFixture } from './fixtures/rvf-fixture';
import { fixtureProject, withRequests } from './fixtures/project-fixture';

jest.setTimeout(30_000);

let root: string;
let rvfPath: string;

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'rvforge-test-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

const run = (project: RvforgeProject = fixtureProject(), receiptsDir?: string) =>
  runTestAgent({ rvfPath, project, ...(receiptsDir ? { receiptsDir } : {}) });

const categoryOf = (result: TestAgentResult, id: string) => {
  const category = result.categories.find((c) => c.id === id);
  if (!category) throw new Error(`no test category "${id}"`);
  return category;
};

describe('test — category coverage', () => {
  it('reports all ten P4 categories', async () => {
    const result = await run();
    expect(result.categories.map((c) => c.id)).toEqual([
      'capability-denials',
      'clean-installation',
      'deterministic-evaluations',
      'filesystem-escape',
      'malformed-inputs',
      'network-monitoring',
      'resource-exhaustion',
      'state-checkpoint-recovery',
      'update-rollback',
      'witness-verification',
    ]);
  });

  it('skips — never passes — the four categories that need a quarantined runtime', async () => {
    const result = await run();
    for (const id of [
      'clean-installation',
      'deterministic-evaluations',
      'network-monitoring',
      'resource-exhaustion',
    ]) {
      const category = categoryOf(result, id);
      expect(category.status).toBe('skipped');
      expect(category.detail).toBe(QUARANTINE_REASON);
      expect(category.detail).toMatch(/Reader\/RVM/);
    }
  });

  it('passes the inspection-only categories on a clean project', async () => {
    const result = await run();
    for (const id of [
      'capability-denials',
      'filesystem-escape',
      'malformed-inputs',
      'state-checkpoint-recovery',
      'update-rollback',
    ]) {
      expect(categoryOf(result, id).status).toBe('pass');
    }
    expect(result.failed).toBe(0);
    expect(result.ok).toBe(true);
  });

  it('counts pass, fail, and skipped separately', async () => {
    const result = await run();
    expect(result.passed + result.failed + result.skipped).toBe(result.categories.length);
    expect(result.skipped).toBeGreaterThanOrEqual(4);
  });

  it('says plainly that filesystem escape probing was not attempted at runtime', async () => {
    expect(categoryOf(await run(), 'filesystem-escape').detail).toMatch(
      /runtime escape probing requires quarantined runtime/,
    );
  });
});

describe('test — malformed inputs', () => {
  it('rejects truncated and bit-flipped variants of the file', async () => {
    const category = categoryOf(await run(), 'malformed-inputs');
    expect(category.status).toBe('pass');
    expect(category.detail).toMatch(/truncated-half/);
    expect(category.detail).toMatch(/truncated-below-root-manifest/);
    expect(category.detail).toMatch(/bit-flipped-root-manifest/);
    expect(category.detail).toMatch(/bit-flipped-segment-magic/);
  });

  it('fails when a tampered file is presented as the agent itself', async () => {
    const tamperedPath = join(root, 'tampered.rvf');
    const original = await readFile(rvfPath);
    const tampered = Buffer.from(original);
    tampered[original.length - ROOT_MANIFEST_SIZE + 0x24] ^= 0x01;
    await writeFile(tamperedPath, tampered);

    await expect(runTestAgent({ rvfPath: tamperedPath, project: fixtureProject() })).rejects.toThrow(
      /CRC32C mismatch/,
    );
  });
});

describe('test — declared contract', () => {
  it('fails when a capability class is neither requested nor denied', async () => {
    const project = fixtureProject({
      capabilities: {
        defaultPolicy: 'deny',
        requests: [{ class: 'memory', scope: '512MiB', rationale: 'working set' }],
        denials: ['network'],
      },
    });
    const category = categoryOf(await run(project), 'capability-denials');
    expect(category.status).toBe('fail');
    expect(category.detail).toMatch(/neither requested nor denied/);
    expect(category.detail).toMatch(/filesystem/);
  });

  it('passes when every class is accounted for', async () => {
    const requests = [{ class: 'memory' as const, scope: '512MiB', rationale: 'working set' }];
    const project = fixtureProject({
      capabilities: { defaultPolicy: 'deny', requests, denials: defaultDenials(requests) },
    });
    expect(categoryOf(await run(project), 'capability-denials').status).toBe('pass');
  });

  it.each([['../etc'], ['/'], ['~'], ['C:\\'], ['docs/**']])(
    'fails a filesystem scope that can reach outside its root: %s',
    async (scope) => {
      const project = withRequests([
        { class: 'filesystem', scope, rationale: 'r' },
        { class: 'memory', scope: '512MiB', rationale: 'r' },
      ]);
      const category = categoryOf(await run(project), 'filesystem-escape');
      expect(category.status).toBe('fail');
      expect(category.detail).toMatch(/reach outside their root/);
    },
  );

  it('fails when checkpointing or recovery is not declared', async () => {
    const project = fixtureProject({
      state: { checkpointing: false, recovery: true, rollbackSafeUntilStateSchema: 2 },
    });
    const category = categoryOf(await run(project), 'state-checkpoint-recovery');
    expect(category.status).toBe('fail');
    expect(category.detail).toMatch(/state\.checkpointing/);
  });

  it('fails when rollback is already unsafe at the shipping state schema', async () => {
    const project = fixtureProject({
      state: { checkpointing: true, recovery: true, rollbackSafeUntilStateSchema: 0 },
    });
    const category = categoryOf(await run(project), 'update-rollback');
    expect(category.status).toBe('fail');
    expect(category.detail).toMatch(/rollback is declared unsafe/);
  });

  it('fails a version that gives updates no order', async () => {
    const category = categoryOf(await run(fixtureProject({ version: 'latest' })), 'update-rollback');
    expect(category.status).toBe('fail');
    expect(category.detail).toMatch(/not a semantic version/);
  });
});

describe('test — witness verification', () => {
  it('skips with a reason when no receipt chain exists', async () => {
    const empty = await mkdtemp(join(tmpdir(), 'rvforge-no-receipts-'));
    try {
      const category = categoryOf(await run(fixtureProject(), empty), 'witness-verification');
      expect(category.status).toBe('skipped');
      expect(category.detail).toMatch(/no receipts\.jsonl found/);
    } finally {
      await rm(empty, { recursive: true, force: true });
    }
  });

  it('passes on an intact chain', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rvforge-receipts-'));
    try {
      const subject = 'a'.repeat(64);
      await appendReceipt(dir, {
        subject,
        event: 'build',
        outcome: 'pass',
        actor: { kind: 'builder', id: 'test' },
      });
      await appendReceipt(dir, {
        subject,
        event: 'verify',
        outcome: 'pass',
        actor: { kind: 'builder', id: 'test' },
      });

      const category = categoryOf(await run(fixtureProject(), dir), 'witness-verification');
      expect(category.status).toBe('pass');
      expect(category.detail).toMatch(/2 receipt\(s\)/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('fails when a receipt has been removed from the middle of the chain', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'rvforge-broken-chain-'));
    try {
      const subject = 'b'.repeat(64);
      for (const event of ['build', 'verify', 'verify'] as const) {
        await appendReceipt(dir, {
          subject,
          event,
          outcome: 'pass',
          actor: { kind: 'builder', id: 'test' },
        });
      }
      const receipts = await readReceipts(dir);
      await writeFile(
        join(dir, 'receipts.jsonl'),
        [receipts[0], receipts[2]].map(receiptLine).join(''),
        'utf8',
      );

      const category = categoryOf(await run(fixtureProject(), dir), 'witness-verification');
      expect(category.status).toBe('fail');
      expect(category.detail).toMatch(/broken-link/);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
