/**
 * `rvforge test agent.rvf` — the inspection-only subset of the ten P4 test
 * categories (requirements P4 "Test", ADR-294 §4).
 *
 * Six categories are decidable by inspecting the artifact and the declared
 * contract. Four are not: clean installation, deterministic evaluation,
 * network monitoring, and resource exhaustion all require *running* the agent
 * under a quarantined runtime, which is the Reader's and RVM's job, not this
 * CLI's (ADR-283 invariant 2: forge never executes an RVF).
 *
 * Those four report `skipped` with the reason. **They are never reported as
 * passing.** A green row that means "we did not look" is worse than a missing
 * row, because a publisher reads it as evidence and ADR-294 §7 says a trust
 * level names the evidence that was actually gathered.
 *
 * `filesystem-escape` is the one category that runs in a reduced form: the
 * declared filesystem scopes are checked for traversal and root escapes, and
 * the detail says plainly that runtime probing was not attempted.
 */

import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join } from 'node:path';

import { ForgeError } from './errors';
import {
  MANIFEST_CAPABILITY_CLASSES,
  type ManifestCapabilityClass,
  type RvforgeProject,
} from './project';
import { ROOT_MANIFEST_SIZE } from './rvf';
import { validateRvf, type ValidateResult } from './validate';
import { RECEIPTS_FILENAME, readReceipts, verifyReceiptChain } from './witness';

export type TestStatus = 'pass' | 'fail' | 'skipped';

export interface TestCategory {
  id: string;
  title: string;
  status: TestStatus;
  detail: string;
}

export interface TestAgentOptions {
  rvfPath: string;
  project: RvforgeProject;
  /** Directory holding `receipts.jsonl`; defaults to the RVF's directory. */
  receiptsDir?: string;
  allowUnsigned?: boolean;
}

export interface TestAgentResult {
  rvfPath: string;
  categories: TestCategory[];
  passed: number;
  failed: number;
  skipped: number;
  ok: boolean;
}

/** The single reason a category is skipped, spelled the same way everywhere. */
export const QUARANTINE_REASON = 'skipped: requires quarantined runtime (Reader/RVM)';

/** Categories forge cannot decide without executing the agent. */
const QUARANTINE_ONLY: ReadonlyArray<{ id: string; title: string }> = Object.freeze([
  { id: 'clean-installation', title: 'Clean installation' },
  { id: 'deterministic-evaluations', title: 'Deterministic evaluations' },
  { id: 'network-monitoring', title: 'Network monitoring' },
  { id: 'resource-exhaustion', title: 'Resource exhaustion' },
]);

/** Run the inspection-only test suite. Never throws for a failing category. */
export async function runTestAgent(options: TestAgentOptions): Promise<TestAgentResult> {
  const validation = await validateRvf(options.rvfPath, {
    deep: true,
    allowUnsigned: options.allowUnsigned,
  });

  const categories: TestCategory[] = [
    ...QUARANTINE_ONLY.map(({ id, title }) => ({
      id,
      title,
      status: 'skipped' as const,
      detail: QUARANTINE_REASON,
    })),
    await malformedInputs(options.rvfPath, validation),
    capabilityDenials(options.project),
    filesystemEscape(options.project),
    stateCheckpointRecovery(options.project),
    updateRollback(options.project),
    await witnessVerification(options.receiptsDir ?? dirname(options.rvfPath)),
  ].sort((a, b) => a.id.localeCompare(b.id));

  const passed = categories.filter((c) => c.status === 'pass').length;
  const failed = categories.filter((c) => c.status === 'fail').length;
  const skipped = categories.filter((c) => c.status === 'skipped').length;
  return { rvfPath: options.rvfPath, categories, passed, failed, skipped, ok: failed === 0 };
}

/**
 * Malformed-input robustness: damaged copies of the RVF must be *rejected*,
 * not merely mishandled.
 *
 * Each variant is written to a temporary directory, validated, and expected to
 * raise a `ForgeError`. A variant that validates clean is the failure this
 * category exists to catch: it means the parser accepted something it should
 * have refused. A variant that throws something other than a `ForgeError` is
 * also a failure — an unhandled crash is not a clean rejection.
 */
async function malformedInputs(rvfPath: string, validation: ValidateResult): Promise<TestCategory> {
  const id = 'malformed-inputs';
  const title = 'Malformed inputs';
  const original = await readFile(rvfPath);
  const variants = malformedVariants(original, validation);

  const scratch = await mkdtemp(join(tmpdir(), 'rvforge-malformed-'));
  const accepted: string[] = [];
  const crashed: string[] = [];
  try {
    for (const [name, bytes] of variants) {
      const path = join(scratch, `${name}.rvf`);
      await writeFile(path, bytes);
      try {
        await validateRvf(path, { allowUnsigned: true });
        accepted.push(name);
      } catch (err) {
        if (!(err instanceof ForgeError)) crashed.push(name);
      }
    }
  } finally {
    await rm(scratch, { recursive: true, force: true });
  }

  const problems = [
    ...accepted.map((name) => `${name} validated clean`),
    ...crashed.map((name) => `${name} raised a non-ForgeError`),
  ];
  return {
    id,
    title,
    status: problems.length === 0 ? 'pass' : 'fail',
    detail:
      problems.length === 0
        ? `${variants.length} damaged variant(s) rejected cleanly: ${variants.map(([n]) => n).join(', ')}`
        : problems.join('; '),
  };
}

/** Damaged copies: truncations at two depths, and bit flips in two places. */
function malformedVariants(original: Buffer, validation: ValidateResult): Array<[string, Buffer]> {
  const variants: Array<[string, Buffer]> = [
    ['truncated-half', Buffer.from(original.subarray(0, Math.floor(original.length / 2)))],
    ['truncated-below-root-manifest', Buffer.from(original.subarray(0, ROOT_MANIFEST_SIZE - 1))],
  ];

  // A flipped bit inside the root manifest breaks its CRC32C. Offset 0x24 is
  // the epoch field: covered by the checksum, and not a length the walker
  // would trip over first, so the CRC is what does the rejecting.
  const rootFlip = Buffer.from(original);
  rootFlip[original.length - ROOT_MANIFEST_SIZE + 0x24] ^= 0x01;
  variants.push(['bit-flipped-root-manifest', rootFlip]);

  if (validation.segments.count > 0) {
    const segmentFlip = Buffer.from(original);
    segmentFlip[0] ^= 0x01;
    variants.push(['bit-flipped-segment-magic', segmentFlip]);
  }
  return variants;
}

/**
 * Capability denials: the declared contract must account for all fifteen
 * classes (ADR-286).
 *
 * Default-deny already denies an unmentioned class at runtime, so this is not
 * a runtime hole — it is a disclosure hole. Requirements P6 renders "this
 * agent cannot" from the denial list, and a class in neither list is one the
 * user is never told about.
 */
function capabilityDenials(project: RvforgeProject): TestCategory {
  const requested = new Set(project.capabilities.requests.map((r) => r.class));
  const denied = new Set(project.capabilities.denials);
  const undeclared = MANIFEST_CAPABILITY_CLASSES.filter(
    (cls: ManifestCapabilityClass) => !requested.has(cls) && !denied.has(cls),
  );

  return {
    id: 'capability-denials',
    title: 'Capability denials',
    status: undeclared.length === 0 ? 'pass' : 'fail',
    detail:
      undeclared.length === 0
        ? `all ${MANIFEST_CAPABILITY_CLASSES.length} classes accounted for: ` +
          `${requested.size} requested, ${denied.size} explicitly denied`
        : `class(es) neither requested nor denied, so the install UX cannot state them: ${undeclared.join(', ')}`,
  };
}

/** Path fragments that would let a declared scope reach outside its root. */
const ESCAPING_SCOPE = /(^|[/\\])\.\.([/\\]|$)|^[/\\]|^~|^[A-Za-z]:[/\\]|\*\*/;

/**
 * Filesystem escape: the *declared* scopes cannot express an escape.
 *
 * This is the reduced form of the P4 category. It proves the contract does not
 * ask for traversal; it does not prove the agent will not attempt one, which
 * needs the quarantined runtime. The detail says so rather than letting a
 * `pass` imply the stronger claim.
 */
function filesystemEscape(project: RvforgeProject): TestCategory {
  const scopes = project.capabilities.requests
    .filter((r) => r.class === 'filesystem')
    .map((r) => r.scope.trim());
  const escaping = scopes.filter((scope) => ESCAPING_SCOPE.test(scope));

  return {
    id: 'filesystem-escape',
    title: 'Filesystem escape attempts',
    status: escaping.length === 0 ? 'pass' : 'fail',
    detail:
      escaping.length === 0
        ? `${scopes.length} declared filesystem scope(s) contain no traversal or root escape; ` +
          'runtime escape probing requires quarantined runtime (Reader/RVM) and was not attempted'
        : `declared scope(s) can reach outside their root: ${escaping.join(', ')}`,
  };
}

/** State checkpoint and recovery: the contract fields the RVM needs. */
function stateCheckpointRecovery(project: RvforgeProject): TestCategory {
  const { checkpointing, recovery } = project.state;
  const { stateSchemaVersion } = project.runtimeRequirements;
  const missing: string[] = [];
  if (checkpointing !== true) missing.push('state.checkpointing');
  if (recovery !== true) missing.push('state.recovery');
  if (!Number.isInteger(stateSchemaVersion) || stateSchemaVersion < 1) {
    missing.push('runtimeRequirements.stateSchemaVersion');
  }

  return {
    id: 'state-checkpoint-recovery',
    title: 'State checkpoint and recovery',
    status: missing.length === 0 ? 'pass' : 'fail',
    detail:
      missing.length === 0
        ? `checkpointing and recovery declared at state schema v${stateSchemaVersion}`
        : `contract field(s) missing or false: ${missing.join(', ')}`,
  };
}

/**
 * Update and rollback: the lineage fields have to be coherent *before*
 * installation (ADR-294 §6).
 *
 * `rollbackSafeUntilStateSchema` below the current schema version means
 * rollback is already unsafe at install time — which the user must be told
 * before installing, never discovered after a migration has run.
 */
function updateRollback(project: RvforgeProject): TestCategory {
  const { rollbackSafeUntilStateSchema } = project.state;
  const { stateSchemaVersion } = project.runtimeRequirements;
  const problems: string[] = [];

  if (!/^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)*$/.test(project.version)) {
    problems.push(`version "${project.version}" is not a semantic version, so no update order exists`);
  }
  if (!Number.isInteger(rollbackSafeUntilStateSchema)) {
    problems.push('state.rollbackSafeUntilStateSchema is not an integer');
  } else if (rollbackSafeUntilStateSchema < stateSchemaVersion) {
    problems.push(
      `rollback is declared unsafe from state schema v${rollbackSafeUntilStateSchema}, which is ` +
        `already below the shipping schema v${stateSchemaVersion}`,
    );
  }

  return {
    id: 'update-rollback',
    title: 'Update and rollback',
    status: problems.length === 0 ? 'pass' : 'fail',
    detail:
      problems.length === 0
        ? `v${project.version}, rollback safe until state schema v${rollbackSafeUntilStateSchema}`
        : problems.join('; '),
  };
}

/** Witness verification: the receipt chain, when one exists, must verify. */
async function witnessVerification(receiptsDir: string): Promise<TestCategory> {
  const id = 'witness-verification';
  const title = 'Witness verification';
  const receipts = await readReceipts(receiptsDir);

  if (receipts.length === 0) {
    return {
      id,
      title,
      status: 'skipped',
      detail: `skipped: no ${RECEIPTS_FILENAME} found in ${receiptsDir}`,
    };
  }

  const chain = verifyReceiptChain(receipts);
  return {
    id,
    title,
    status: chain.ok ? 'pass' : 'fail',
    detail: chain.ok
      ? `${chain.count} receipt(s) over ${chain.subjects.length} subject(s), chain intact`
      : chain.problems.map((p) => `#${p.index} ${p.reason}: ${p.detail}`).join('; '),
  };
}
