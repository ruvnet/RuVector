import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { runBuild } from '../src/build';
import { compatibilityMatrixRevision } from '../src/compat';
import { defaultConfig } from '../src/config';
import { canonicalJson } from '../src/hash';
import { roleFor, type SoftwareInventory } from '../src/inventory';
import type { ForgeConfig, PackagingMode } from '../src/types';
import { buildRvfFixture } from './fixtures/rvf-fixture';

jest.setTimeout(30_000);

const DISTRIBUTION_URL = 'https://dist.example/agent.rvf';

let root: string;
let rvfPath: string;

const config = (mode: PackagingMode = 'embedded'): ForgeConfig =>
  defaultConfig({
    app: {
      name: 'Fixture Agent',
      version: '0.1.0',
      publisher: 'RuVector Tests',
      identifier: 'dev.ruvector.fixture',
    },
    targets: ['linux-x64', 'macos-universal'] as ForgeConfig['targets'],
    packaging: mode === 'embedded' ? { mode } : { mode, distributionUrl: DISTRIBUTION_URL },
  });

beforeAll(async () => {
  root = await mkdtemp(join(tmpdir(), 'forge-inventory-'));
  rvfPath = join(root, 'agent.rvf');
  await writeFile(rvfPath, buildRvfFixture());
});

afterAll(async () => {
  await rm(root, { recursive: true, force: true });
});

const build = (name: string, mode: PackagingMode = 'embedded') =>
  runBuild({ config: config(mode), rvfPath, outDir: join(root, name), force: true });

const readInventory = async (outDir: string): Promise<SoftwareInventory> =>
  JSON.parse(await readFile(join(outDir, 'inventory.json'), 'utf8')) as SoftwareInventory;

describe('software inventory (§3.9)', () => {
  it('lists every file placed in a bundle with a path, digest, size, and role', async () => {
    const built = await build('inventory-roles');
    const inventory = await readInventory(built.outDir);

    expect(inventory.files.map((f) => f.path)).toEqual([
      'build-manifest.json',
      'bundles/linux-x64/reader/reader-slot.json',
      'bundles/linux-x64/rvf/agent.rvf',
      'bundles/macos-universal/reader/reader-slot.json',
      'bundles/macos-universal/rvf/agent.rvf',
    ]);

    for (const entry of inventory.files) {
      expect(entry.sha256).toMatch(/^[0-9a-f]{64}$/);
      expect(entry.size).toBeGreaterThan(0);
    }
    expect(inventory.files.filter((f) => f.role === 'rvf')).toHaveLength(2);
    expect(inventory.files.filter((f) => f.role === 'reader')).toHaveLength(2);
    expect(inventory.files.filter((f) => f.role === 'metadata')).toHaveLength(1);
    expect(inventory.files.filter((f) => f.role === 'locator')).toHaveLength(0);
  });

  it('labels locators in thin mode', async () => {
    const built = await build('inventory-thin', 'thin');
    const inventory = await readInventory(built.outDir);

    expect(inventory.files.filter((f) => f.role === 'locator').map((f) => f.path)).toEqual([
      'bundles/linux-x64/rvf/locator.json',
      'bundles/macos-universal/rvf/locator.json',
    ]);
    expect(inventory.files.filter((f) => f.role === 'rvf')).toHaveLength(0);
  });

  it('is byte-identical across two builds of the same input', async () => {
    const [first, second] = [await build('inventory-det-a'), await build('inventory-det-b')];
    expect(canonicalJson(first.inventory)).toBe(canonicalJson(second.inventory));

    const [a, b] = await Promise.all([
      readFile(join(first.outDir, 'inventory.json'), 'utf8'),
      readFile(join(second.outDir, 'inventory.json'), 'utf8'),
    ]);
    expect(a).toBe(b);
  });

  it('carries no timestamp, hostname, or absolute path', async () => {
    const built = await build('inventory-no-clock');
    const raw = await readFile(join(built.outDir, 'inventory.json'), 'utf8');
    for (const banned of ['createdAt', 'timestamp', 'builtAt', 'hostname', 'cwd', built.outDir]) {
      expect(raw).not.toContain(banned);
    }
  });

  it('names the compatibility-matrix revision that admitted the build', async () => {
    const built = await build('inventory-matrix');
    const inventory = await readInventory(built.outDir);
    expect(inventory.compatibilityMatrixRevision).toBe(compatibilityMatrixRevision());
    expect(built.provenance.compatibilityMatrixRevision).toBe(compatibilityMatrixRevision());
  });

  it('records the components and segment counts a reviewer needs', async () => {
    const built = await build('inventory-components');
    const inventory = await readInventory(built.outDir);

    expect(inventory.components.map((c) => c.role)).toEqual(['builder', 'payload', 'runtime']);
    expect(inventory.components[1].sha256).toBe(built.manifest.rvf.sha256);
    expect(inventory.rvfSegments.count).toBe(built.validation.segments.count);
    expect(inventory.capabilityPolicyHash).toBe(built.manifest.capabilityPolicyHash);
  });

  it('mirrors the bundle layout per target', async () => {
    const built = await build('inventory-bundles');
    const inventory = await readInventory(built.outDir);
    expect(inventory.bundles).toEqual([
      {
        target: 'linux-x64',
        mode: 'embedded',
        rvfPath: 'bundles/linux-x64/rvf/agent.rvf',
        locatorPath: null,
        readerSlotPath: 'bundles/linux-x64/reader/reader-slot.json',
      },
      {
        target: 'macos-universal',
        mode: 'embedded',
        rvfPath: 'bundles/macos-universal/rvf/agent.rvf',
        locatorPath: null,
        readerSlotPath: 'bundles/macos-universal/reader/reader-slot.json',
      },
    ]);
  });
});

describe('roleFor', () => {
  it('classifies a path by where the layout puts it', () => {
    const layout = {
      schemaVersion: 1 as const,
      mode: 'embedded' as const,
      rvfSha256: 'a'.repeat(64),
      bundles: [
        {
          target: 'linux-x64' as const,
          mode: 'embedded' as const,
          rvfPath: 'bundles/linux-x64/rvf/agent.rvf',
          readerSlotPath: 'bundles/linux-x64/reader/reader-slot.json',
        },
      ],
    };
    expect(roleFor('bundles/linux-x64/rvf/agent.rvf', layout)).toBe('rvf');
    expect(roleFor('bundles/linux-x64/reader/reader-slot.json', layout)).toBe('reader');
    expect(roleFor('build-manifest.json', layout)).toBe('metadata');
  });
});
