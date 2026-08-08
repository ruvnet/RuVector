/**
 * Software inventory — required output §3.9.
 *
 * A deterministic listing of every file placed in a bundle: its path, digest,
 * size, and role. Deterministic means the same build description always
 * produces byte-identical inventory JSON: entries are sorted by path, keys are
 * canonical, and nothing time-, host-, or run-dependent appears.
 *
 * The inventory covers *bundle contents* — the payload, locators, reader
 * slots, and the build manifest they are described by. It deliberately does
 * not list `inventory.json`, `provenance.json`, `checksums.txt`, or
 * `receipts.jsonl`: those are the build record rather than things shipped
 * inside a package, and an inventory that listed itself could not be hashed.
 */

import { stat } from 'node:fs/promises';
import { join } from 'node:path';

import { sha256File } from './hash';
import { FORGE_PACKAGE_NAME } from './version';
import type { BundleEntry, BundleLayout } from './package-modes';
import type { ValidateResult } from './validate';
import type { BuildManifest, BuildTarget } from './types';

/** What a file is doing in the bundle. */
export type InventoryRole = 'rvf' | 'reader' | 'locator' | 'metadata';

export interface InventoryEntry {
  path: string;
  sha256: string;
  size: number;
  role: InventoryRole;
}

export interface InventoryComponent {
  name: string;
  version: string;
  role: string;
  [key: string]: string | number;
}

export interface InventoryBundle {
  target: BuildTarget;
  mode: string;
  rvfPath: string | null;
  locatorPath: string | null;
  readerSlotPath: string;
}

export interface SoftwareInventory {
  schemaVersion: 1;
  /** Named components: builder, payload, runtime. */
  components: InventoryComponent[];
  /** Per-target bundle layout. */
  bundles: InventoryBundle[];
  /** Every file placed in a bundle, sorted by path. */
  files: InventoryEntry[];
  rvfSegments: { count: number; executableCount: number; byType: Record<string, number> };
  capabilityPolicyHash: string;
  /** Matrix revision that admitted this build (ADR-291 §2). */
  compatibilityMatrixRevision: string;
}

export interface InventoryInput {
  stagingDir: string;
  manifest: BuildManifest;
  validation: ValidateResult;
  layout: BundleLayout;
  /** Relative paths of bundle files, plus any metadata files to include. */
  files: readonly string[];
  compatibilityMatrixRevision: string;
}

/** Classify a staged path by where it sits in the bundle layout. */
export function roleFor(path: string, layout: BundleLayout): InventoryRole {
  for (const bundle of layout.bundles) {
    if (bundle.rvfPath === path) return 'rvf';
    if (bundle.locatorPath === path) return 'locator';
    if (bundle.readerSlotPath === path) return 'reader';
  }
  return 'metadata';
}

/** Build the inventory by hashing each listed file. */
export async function buildInventory(input: InventoryInput): Promise<SoftwareInventory> {
  const { manifest, validation, layout } = input;

  const files: InventoryEntry[] = [];
  for (const path of [...new Set(input.files)].sort()) {
    const abs = join(input.stagingDir, path);
    const info = await stat(abs);
    files.push({
      path,
      sha256: await sha256File(abs),
      size: info.size,
      role: roleFor(path, layout),
    });
  }

  return {
    schemaVersion: 1,
    components: [
      { name: FORGE_PACKAGE_NAME, version: manifest.generator.version, role: 'builder' },
      {
        name: 'rvf',
        version: `format-v${validation.root.formatVersion}`,
        role: 'payload',
        sha256: manifest.rvf.sha256,
        sizeBytes: manifest.rvf.size,
      },
      {
        name: 'rvm',
        version: manifest.runtime.rvmVersion,
        role: 'runtime',
        commit: manifest.runtime.rvmCommit,
        profile: manifest.runtime.runtimeProfile,
      },
    ],
    bundles: layout.bundles.map(toInventoryBundle),
    files,
    rvfSegments: {
      count: validation.segments.count,
      executableCount: validation.segments.executableCount,
      byType: validation.segments.byType,
    },
    capabilityPolicyHash: manifest.capabilityPolicyHash,
    compatibilityMatrixRevision: input.compatibilityMatrixRevision,
  };
}

function toInventoryBundle(bundle: BundleEntry): InventoryBundle {
  return {
    target: bundle.target,
    mode: bundle.mode,
    rvfPath: bundle.rvfPath ?? null,
    locatorPath: bundle.locatorPath ?? null,
    readerSlotPath: bundle.readerSlotPath,
  };
}
