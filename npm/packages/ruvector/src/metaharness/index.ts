/**
 * Lazy CommonJS-to-ESM bridge for the pinned MetaHarness capability set.
 *
 * `ruvector` remains CommonJS for compatibility, while MetaHarness is ESM-only.
 * Keeping native import() intact also means importing `ruvector` does not pay the
 * startup or side-effect cost of loading research infrastructure.
 */
import { readFile, stat } from 'node:fs/promises';
import { join } from 'node:path';

const nativeImport = new Function(
  'specifier',
  'return import(specifier)',
) as (specifier: string) => Promise<Record<string, any>>;

/** Compatibility baselines, not a record of the versions installed at runtime. */
export const METAHARNESS_VERSIONS = Object.freeze({
  metaharness: '0.4.7',
  darwin: '0.9.2',
  flywheel: '0.1.10',
  harness: '0.2.0',
  router: '0.4.0',
  redblue: '0.1.6',
  weightEft: '0.1.1',
  workspaceLens: '0.1.2',
  workspaceProbe: '0.1.1',
});

const PACKAGE_NAMES = Object.freeze({
  metaharness: 'metaharness',
  darwin: '@metaharness/darwin',
  flywheel: '@metaharness/flywheel',
  harness: '@metaharness/harness',
  router: '@metaharness/router',
  redblue: '@metaharness/redblue',
  weightEft: '@metaharness/weight-eft',
  workspaceLens: '@metaharness/workspace-lens',
  workspaceProbe: '@metaharness/workspace-probe',
});

export type MetaHarnessCapabilityName = keyof typeof METAHARNESS_VERSIONS;

export interface MetaHarnessCapabilityStatus {
  name: MetaHarnessCapabilityName;
  package: string;
  /** Installed package metadata, or "unknown" when absent or unreadable. */
  version: string;
  /** The SDK's compatibility baseline; does not imply this version is installed. */
  baselineVersion: string;
  available: boolean;
  detail: string;
}

export interface MetaHarnessRouteInput {
  rows: Array<{ embedding: number[]; scores: Record<string, number> }>;
  prices: Record<string, number>;
  queryEmbedding: number[];
  qualityBar?: number;
  k?: number;
}

export interface MetaHarnessPromotionEvidence {
  baseline: {
    primary: number;
    noopRate: number;
    costPerWin: number;
    regressed?: boolean;
  };
  candidate: {
    primary: number;
    noopRate: number;
    costPerWin: number;
    regressed?: boolean;
  };
  anchor?: { baseline: number; candidate: number };
}

const moduleCache = new Map<string, Promise<Record<string, any>>>();

function loadPackage(name: string): Promise<Record<string, any>> {
  let pending = moduleCache.get(name);
  if (!pending) {
    pending = nativeImport(name).catch((error: any) => {
      if (error?.code === 'ERR_MODULE_NOT_FOUND' || error?.code === 'MODULE_NOT_FOUND') {
        // Per ruflo ADR-150 "MetaHarness Integration Surfaces"
        // (ruflo/v3/docs/adr/ADR-150-metaharness-integration-surfaces.md, Implemented
        // 2026-06-16; summarized locally in METAHARNESS-README.md): "@metaharness/*
        // packages MUST appear in optionalDependencies or peerDependencies (optional),
        // never in dependencies". ruvector must keep working without the MetaHarness
        // stack, and callers of a MetaHarness-backed capability get a clear error
        // naming the missing package.
        moduleCache.delete(name);
        const missing = new Error(
          `Optional MetaHarness package "${name}" is not installed. ` +
            `Run \`npm install ${name}\` to enable this capability; ` +
            `ruvector itself works without it (ruflo ADR-150, see METAHARNESS-README.md).`,
        );
        (missing as any).code = 'ERR_METAHARNESS_OPTIONAL_MISSING';
        (missing as any).cause = error;
        throw missing;
      }
      throw error;
    });
    moduleCache.set(name, pending);
  }
  return pending;
}

function assertFiniteVector(name: string, vector: number[]): void {
  if (!Array.isArray(vector) || vector.length === 0 || vector.some((v) => !Number.isFinite(v))) {
    throw new TypeError(`${name} must be a non-empty array of finite numbers`);
  }
}

async function installedPackageVersion(packageName: string): Promise<string> {
  // MetaHarness exposes import-only entrypoints and blocks /package.json in
  // exports. Inspect metadata at Node's package lookup locations without
  // requiring its ESM entrypoint or guessing a version from loaded symbols.
  for (const directory of require.resolve.paths(packageName) ?? []) {
    const packageDirectory = join(directory, packageName);
    try {
      await stat(packageDirectory);
    } catch (error: any) {
      if (error?.code === 'ENOENT' || error?.code === 'ENOTDIR') continue;
      return 'unknown';
    }
    try {
      const metadata = JSON.parse(await readFile(join(packageDirectory, 'package.json'), 'utf8'));
      return metadata.name === packageName && typeof metadata.version === 'string' && metadata.version.trim()
        ? metadata.version.trim()
        : 'unknown';
    } catch {
      // A nearer installation with unreadable metadata shadows outer copies.
      // Do not attribute another installation's version to the loaded package.
      return 'unknown';
    }
  }
  return 'unknown';
}

export async function getMetaHarnessCapabilities(): Promise<MetaHarnessCapabilityStatus[]> {
  const checks: Record<MetaHarnessCapabilityName, [string, string]> = {
    metaharness: ['scaffold', 'core scaffold and orchestration exports loaded'],
    darwin: ['evolve', 'statistical evolution engine loaded'],
    flywheel: ['runFlywheelGenerations', 'signed replay and promotion gate loaded'],
    harness: ['HarnessKernel', 'algorithmic four-gate control plane loaded'],
    router: ['Router', 'cost-aware embedding router loaded'],
    redblue: ['enforceSafetyLimits', 'capability-contained Red/Blue suite loaded'],
    weightEft: ['detectRewardHack', 'reward-hack detector loaded'],
    workspaceLens: ['', 'workspace receipt contract loaded'],
    workspaceProbe: ['workspaceProbeScore', 'workspace mutation evidence loaded'],
  };
  return Promise.all((Object.keys(METAHARNESS_VERSIONS) as MetaHarnessCapabilityName[]).map(async (name) => {
    const packageName = PACKAGE_NAMES[name];
    const [symbol, detail] = checks[name];
    const version = await installedPackageVersion(packageName);
    try {
      const loaded = await loadPackage(packageName);
      return {
        name,
        package: packageName,
        version,
        baselineVersion: METAHARNESS_VERSIONS[name],
        available: symbol ? typeof loaded[symbol] !== 'undefined' : true,
        detail,
      };
    } catch (error: any) {
      return {
        name,
        package: packageName,
        version,
        baselineVersion: METAHARNESS_VERSIONS[name],
        available: false,
        detail: error?.message || String(error),
      };
    }
  }));
}

/** Route to the cheapest model predicted to meet the requested quality bar. */
export async function routeWithMetaHarness(input: MetaHarnessRouteInput): Promise<{
  id: string;
  predictedQuality: number;
  costPerMTok: number;
  metBar: boolean;
}> {
  assertFiniteVector('queryEmbedding', input.queryEmbedding);
  if (!Array.isArray(input.rows) || input.rows.length === 0) {
    throw new TypeError('rows must contain at least one labelled routing example');
  }
  for (const [index, row] of input.rows.entries()) {
    assertFiniteVector(`rows[${index}].embedding`, row.embedding);
    if (row.embedding.length !== input.queryEmbedding.length) {
      throw new RangeError(`rows[${index}].embedding dimension does not match queryEmbedding`);
    }
  }
  const { Router } = await loadPackage(PACKAGE_NAMES.router);
  return Router.fromExamples(input.rows, input.prices, {
    qualityBar: input.qualityBar ?? 0.9,
    ...(input.k === undefined ? {} : { k: input.k }),
  }).route(input.queryEmbedding);
}

/** Evaluate the frozen upstream conjunctive promotion rule without executing a candidate. */
export async function evaluateMetaHarnessPromotion(
  evidence: MetaHarnessPromotionEvidence,
): Promise<{ promote: boolean; reasons: string[] }> {
  const { meetsPromotionRule } = await loadPackage(PACKAGE_NAMES.flywheel);
  return meetsPromotionRule(evidence);
}

/** Verify signed lineage, parent continuity, and the pinned promotion gate. */
export async function verifyMetaHarnessReplay(
  bundle: unknown,
  options: { pinnedGateFingerprint?: string } = {},
): Promise<Record<string, any>> {
  const { verifyReplayBundle } = await loadPackage(PACKAGE_NAMES.flywheel);
  return verifyReplayBundle(bundle, options);
}

/** Run a caller-supplied Flywheel configuration. This is intentionally SDK-only. */
export async function runMetaHarnessFlywheel(config: Record<string, unknown>): Promise<unknown> {
  const { runFlywheelGenerations } = await loadPackage(PACKAGE_NAMES.flywheel);
  return runFlywheelGenerations(config);
}

/** Run Darwin only after the caller explicitly opts into code execution. */
export async function runMetaHarnessDarwin(
  config: Record<string, unknown>,
  options: { execute?: boolean } = {},
): Promise<unknown> {
  if (options.execute !== true) {
    throw new Error('Darwin execution requires { execute: true }');
  }
  const { evolve } = await loadPackage(PACKAGE_NAMES.darwin);
  return evolve(config);
}

export async function createMetaHarnessKernel(options: Record<string, unknown>): Promise<any> {
  const { HarnessKernel } = await loadPackage(PACKAGE_NAMES.harness);
  return new HarnessKernel(options);
}

export async function scoreMetaHarnessWorkspace(
  receipts: unknown[],
  options: Record<string, unknown> = {},
): Promise<Record<string, any>> {
  const { workspaceProbeScore } = await loadPackage(PACKAGE_NAMES.workspaceProbe);
  return workspaceProbeScore(receipts, options);
}

export async function scanMetaHarnessRewardHacks(
  trajectory: Record<string, unknown>,
): Promise<{ clean: boolean; findings: unknown[] }> {
  const { detectRewardHack } = await loadPackage(PACKAGE_NAMES.weightEft);
  const findings = detectRewardHack(trajectory);
  return { clean: findings.length === 0, findings };
}

export async function assertMetaHarnessSafePayload(payload: unknown): Promise<void> {
  const { assertNoLiveCredential } = await loadPackage(PACKAGE_NAMES.redblue);
  const serialized = typeof payload === 'string' ? payload : JSON.stringify(payload);
  assertNoLiveCredential(serialized);
}
