import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { ForgeError, ForgeErrorCode } from '../src/errors';
import {
  buildCapabilityManifest,
  capabilityContractHash,
  manualReviewTriggers,
  runPack,
  securityProfileFor,
} from '../src/pack';
import { objectId } from '../src/registry';
import { defaultDenials, type RvforgeProject } from '../src/project';
import { buildRvfFixture } from './fixtures/rvf-fixture';
import { fixtureProject, withRequests } from './fixtures/project-fixture';

jest.setTimeout(30_000);

let root: string;
let rvfPath: string;

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'rvforge-pack-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

const pack = (project: RvforgeProject = fixtureProject()) =>
  runPack({ rvfPath, project, publishedAt: '2026-08-03T00:00:00.000Z' });

async function expectCode(promise: Promise<unknown>, code: ForgeErrorCode): Promise<ForgeError> {
  try {
    await promise;
  } catch (err) {
    expect(err).toBeInstanceOf(ForgeError);
    expect((err as ForgeError).code).toBe(code);
    return err as ForgeError;
  }
  throw new Error(`expected a ForgeError with code ${code}`);
}

describe('pack — happy path', () => {
  it('passes every check and drafts both registry objects', async () => {
    const result = await pack();

    expect(result.checks.map((c) => c.id).sort()).toEqual([
      'capability-policy',
      'executable-segments',
      'external-services',
      'license-compatibility',
      'memory-requirements',
      'model-provenance',
      'publisher-signature',
      'runtime-compatibility',
      'rvf-structure',
      'software-inventory',
    ]);
    expect(result.checks.every((c) => c.status === 'pass')).toBe(true);
    expect(result.manualReviewTriggers).toEqual([]);
    expect(result.securityProfile).toBe('restricted');

    expect(result.capabilityManifest.type).toBe('capability-manifest');
    expect(result.capabilityManifest.defaultPolicy).toBe('deny');
    expect(result.capabilityManifest.manifestId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(result.capabilityManifest.signatures).toEqual([]);

    expect(result.release.type).toBe('release');
    expect(result.release.releaseId).toMatch(/^sha256:[0-9a-f]{64}$/);
    expect(result.release.signatures).toEqual([]);
    expect(result.release.trustLevel).toBe('published');
    expect(result.release.predecessor).toBeNull();
    expect(result.release.rvfIdentity).toBe(`sha256:${result.validation.identity.sha256}`);
    expect(result.release.capabilityManifest).toBe(result.capabilityManifest.manifestId);
  });

  it('never claims an evaluation or security report it did not produce', async () => {
    const result = await pack();
    expect(result.release.evaluationReport).toBeNull();
    expect(result.release.securityReport).toBeNull();
  });

  it('addresses the release by content, excluding its own id and signatures', async () => {
    const result = await pack();
    const { releaseId, ...content } = result.release;
    expect(objectId({ ...content, releaseId: 'tampered', signatures: [{ keyId: 'x', role: 'publisher', sig: 'y' }] })).toBe(
      releaseId,
    );
  });

  it('is deterministic for the same inputs', async () => {
    const [first, second] = [await pack(), await pack()];
    expect(second.release.releaseId).toBe(first.release.releaseId);
    expect(second.capabilityManifest.manifestId).toBe(first.capabilityManifest.manifestId);
    expect(second.provenance.id).toBe(first.provenance.id);
  });

  it('generates a software inventory naming the payload and the segments', async () => {
    const { softwareInventory, validation } = await pack();
    expect(softwareInventory.files).toEqual([
      { path: 'agent.rvf', sha256: validation.identity.sha256, size: validation.size, role: 'rvf' },
    ]);
    expect(softwareInventory.bundles).toEqual([]);
    expect(softwareInventory.rvfSegments.count).toBe(validation.segments.count);
    expect(softwareInventory.capabilityPolicyHash).toBe(capabilityContractHash(fixtureProject()));
  });
});

describe('pack — manual review triggers', () => {
  it('flags a vague filesystem scope without failing the pack', async () => {
    const project = withRequests([
      { class: 'filesystem', scope: 'all-files', rationale: 'reads everything' },
      { class: 'memory', scope: '512MiB', rationale: 'working set' },
    ]);
    const result = await pack(project);

    expect(result.manualReviewTriggers).toEqual(['vague-scope:filesystem:all-files']);
    expect(result.securityProfile).toBe('review-required');
    expect(result.capabilityManifest.manualReviewTriggers).toEqual(['vague-scope:filesystem:all-files']);

    const policy = result.checks.find((c) => c.id === 'capability-policy');
    expect(policy?.status).toBe('warn');
    expect(policy?.detail).toMatch(/manual review required/);
  });

  it('flags every scope the install UX could not render concretely', () => {
    for (const scope of ['*', 'any', 'unrestricted', 'all-folders', '**']) {
      expect(
        manualReviewTriggers([{ class: 'filesystem', scope, rationale: 'r' }]),
      ).toEqual([`vague-scope:filesystem:${scope}`]);
    }
  });

  it('flags a class whose request alone requires review', () => {
    expect(manualReviewTriggers([{ class: 'process', scope: 'spawn-helper', rationale: 'r' }])).toEqual([
      'class:process',
    ]);
  });

  it('leaves a concrete scope alone', () => {
    expect(
      manualReviewTriggers([{ class: 'filesystem', scope: 'user-selected', rationale: 'r' }]),
    ).toEqual([]);
  });

  it('reports "standard" when network is requested with a concrete scope', () => {
    const project = withRequests([
      { class: 'network', scope: 'api.example.invalid:443', rationale: 'inference' },
      { class: 'memory', scope: '256MiB', rationale: 'working set' },
    ]);
    expect(securityProfileFor(project, [])).toBe('standard');
  });
});

describe('pack — failures', () => {
  it('refuses a listing with no license', async () => {
    const err = await expectCode(pack(fixtureProject({ license: null })), ForgeErrorCode.LICENSE);
    expect(err.message).toMatch(/license-compatibility/);
  });

  it('refuses an undeclared memory requirement', async () => {
    const project = fixtureProject({
      runtimeRequirements: { ...fixtureProject().runtimeRequirements, memoryMiB: null },
    });
    await expectCode(pack(project), ForgeErrorCode.MANIFEST);
  });

  it('refuses a class that is both requested and denied', async () => {
    const base = fixtureProject();
    const project = fixtureProject({
      capabilities: {
        defaultPolicy: 'deny',
        requests: base.capabilities.requests,
        denials: ['filesystem', ...defaultDenials(base.capabilities.requests)],
      },
    });
    const err = await expectCode(pack(project), ForgeErrorCode.POLICY);
    expect(err.message).toMatch(/filesystem/);
  });

  it('refuses external services declared alongside a network denial', async () => {
    await expectCode(pack(fixtureProject({ externalServices: ['openai'] })), ForgeErrorCode.POLICY);
  });

  it('refuses a runtime profile absent from the compatibility matrix', async () => {
    const project = fixtureProject({
      runtimeRequirements: { ...fixtureProject().runtimeRequirements, profiles: ['rvm'] },
    });
    await expectCode(pack(project), ForgeErrorCode.UNSUPPORTED_TARGET);
  });

  it('refuses an embedded model with no declared digest', async () => {
    const project = fixtureProject({ modelManifest: { location: 'embedded', digests: [] } });
    await expectCode(pack(project), ForgeErrorCode.MANIFEST);
  });

  it('names every failure, not only the first', async () => {
    const err = await expectCode(
      pack(
        fixtureProject({
          license: null,
          runtimeRequirements: { ...fixtureProject().runtimeRequirements, memoryMiB: 0 },
        }),
      ),
      ForgeErrorCode.MANIFEST,
    );
    expect(err.message).toMatch(/memory-requirements/);
    expect(err.message).toMatch(/license-compatibility/);
    expect((err.details.failures as unknown[]).length).toBe(2);
  });

  it('refuses an RVF that is not an RVF at all', async () => {
    const junk = join(root, 'junk.rvf');
    await writeFile(junk, Buffer.alloc(64, 0x7f));
    await expectCode(
      runPack({ rvfPath: junk, project: fixtureProject() }),
      ForgeErrorCode.INVALID_RVF,
    );
  });
});

describe('buildCapabilityManifest', () => {
  it('sorts requests and denials so the manifest id is order-independent', () => {
    const forward = withRequests([
      { class: 'memory', scope: '512MiB', rationale: 'a' },
      { class: 'filesystem', scope: 'user-selected', rationale: 'b' },
    ]);
    const reverse = withRequests([
      { class: 'filesystem', scope: 'user-selected', rationale: 'b' },
      { class: 'memory', scope: '512MiB', rationale: 'a' },
    ]);
    expect(buildCapabilityManifest(forward, []).manifestId).toBe(
      buildCapabilityManifest(reverse, []).manifestId,
    );
  });
});
