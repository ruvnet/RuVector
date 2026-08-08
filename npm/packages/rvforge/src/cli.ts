#!/usr/bin/env node
/**
 * `forge` command router.
 *
 * Hand-rolled argument parsing rather than a CLI framework: the surface is
 * seven commands, the package ships as a signing-adjacent tool where every
 * runtime dependency is attack surface, and zero dependencies keeps the
 * install reproducible.
 *
 * Every command supports `--json` for unattended use and exits with the stable
 * code in `EXIT_CODES`.
 */

import { resolve } from 'node:path';

import { CONFIG_FILENAME, loadConfig, runInit } from './config';
import { runBuild, splitBuildOperands } from './build';
import {
  DEFAULT_RVF_FILENAME,
  createRvf,
  renderCreateSummary,
  resolveSigningKey,
} from './create';
import { runDownload } from './download';
import { ForgeError, ForgeErrorCode, toForgeError } from './errors';
import { formatBytes } from './estimate';
import { createReporter, renderEstimate, USAGE, type Reporter } from './output';
import { runPack } from './pack';
import { PROJECT_FILENAME, loadProject } from './project';
import { renderPublishSummary, runPublish } from './publish';
import { planSubmit, runSubmit } from './submit';
import { runStatus } from './status';
import { runTestAgent } from './test-agent';
import { runVerify } from './verify';
import { validateRvf } from './validate';
import { PACKAGING_MODES, type PackagingMode } from './types';
import { forgeVersion } from './version';

// Operand splitting belongs to `build`, but it is part of the router's parsing
// contract, so it stays reachable from here.
export { splitBuildOperands } from './build';

interface ParsedArgs {
  command: string;
  positionals: string[];
  flags: Map<string, string | boolean>;
  repeated: Map<string, string[]>;
}

const VALUE_FLAGS = new Set([
  'config',
  'out',
  'provenance',
  'mode',
  'project',
  'key-file',
  'registry',
  'receipts',
  'from',
]);
const REPEATED_FLAGS = new Set(['only']);

export function parseArgs(argv: readonly string[]): ParsedArgs {
  const positionals: string[] = [];
  const flags = new Map<string, string | boolean>();
  const repeated = new Map<string, string[]>();

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--') {
      positionals.push(...argv.slice(i + 1));
      break;
    }
    if (!arg.startsWith('-')) {
      positionals.push(arg);
      continue;
    }

    const short = arg === '-h' ? '--help' : arg === '-v' ? '--version' : arg;
    if (!short.startsWith('--')) {
      throw new ForgeError(ForgeErrorCode.USAGE, `Unknown flag "${arg}".`, { flag: arg });
    }

    const eq = short.indexOf('=');
    const name = (eq === -1 ? short.slice(2) : short.slice(2, eq)).toLowerCase();
    let value: string | undefined = eq === -1 ? undefined : short.slice(eq + 1);

    if (VALUE_FLAGS.has(name) || REPEATED_FLAGS.has(name)) {
      if (value === undefined) {
        value = argv[++i];
        if (value === undefined || value.startsWith('-')) {
          throw new ForgeError(ForgeErrorCode.USAGE, `Flag --${name} requires a value.`, { flag: name });
        }
      }
      if (REPEATED_FLAGS.has(name)) {
        repeated.set(name, [...(repeated.get(name) ?? []), value]);
      } else {
        flags.set(name, value);
      }
      continue;
    }
    if (value !== undefined) {
      throw new ForgeError(ForgeErrorCode.USAGE, `Flag --${name} does not take a value.`, { flag: name });
    }
    flags.set(name, true);
  }

  const command = positionals.shift() ?? '';
  return { command, positionals, flags, repeated };
}

const bool = (args: ParsedArgs, name: string): boolean => args.flags.get(name) === true;
const str = (args: ParsedArgs, name: string): string | undefined => {
  const value = args.flags.get(name);
  return typeof value === 'string' ? value : undefined;
};

/** Run the CLI. Returns the process exit code; never throws. */
export async function main(argv: readonly string[]): Promise<number> {
  let args: ParsedArgs;
  let reporter: Reporter = createReporter();

  try {
    args = parseArgs(argv);
    reporter = createReporter({ json: bool(args, 'json'), quiet: bool(args, 'quiet') });
  } catch (err) {
    return reporter.failure('forge', toForgeError(err));
  }

  if (bool(args, 'version')) {
    reporter.success('version', { version: forgeVersion() }, () => [forgeVersion()]);
    return 0;
  }
  if (bool(args, 'help') || args.command === '' || args.command === 'help') {
    reporter.success('help', { usage: USAGE }, () => [USAGE]);
    return 0;
  }

  try {
    return await dispatch(args, reporter);
  } catch (err) {
    return reporter.failure(args.command, toForgeError(err));
  }
}

async function dispatch(args: ParsedArgs, reporter: Reporter): Promise<number> {
  switch (args.command) {
    case 'init':
      return cmdInit(args, reporter);
    case 'create':
      return cmdCreate(args, reporter);
    case 'validate':
      return cmdValidate(args, reporter);
    case 'pack':
      return cmdPack(args, reporter);
    case 'test':
      return cmdTest(args, reporter);
    case 'publish':
      return cmdPublish(args, reporter);
    case 'build':
      return cmdBuild(args, reporter);
    case 'submit':
      return cmdSubmit(args, reporter);
    case 'status':
      return cmdStatus(args, reporter);
    case 'download':
      return cmdDownload(args, reporter);
    case 'verify':
      return cmdVerify(args, reporter);
    default:
      throw new ForgeError(
        ForgeErrorCode.USAGE,
        `Unknown command "${args.command}". Run "forge --help" for the command list.`,
        { command: args.command },
      );
  }
}

const configPath = (args: ParsedArgs): string => resolve(str(args, 'config') ?? CONFIG_FILENAME);
const projectPath = (args: ParsedArgs): string => resolve(str(args, 'project') ?? PROJECT_FILENAME);

function requireOperand(args: ParsedArgs, index: number, label: string): string {
  const value = args.positionals[index];
  if (!value) {
    throw new ForgeError(ForgeErrorCode.USAGE, `Missing required argument <${label}>.`, { expected: label });
  }
  return value;
}

async function cmdInit(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const result = await runInit(configPath(args), {
    force: bool(args, 'force'),
    project: bool(args, 'with-project'),
    keygen: bool(args, 'keygen'),
    ...(str(args, 'project') ? { projectPath: projectPath(args) } : {}),
    ...(str(args, 'key-file') ? { keyFile: resolve(str(args, 'key-file') as string) } : {}),
  });
  reporter.success('init', result, () => [
    `Wrote ${result.path}`,
    ...(result.projectPath ? [`Wrote ${result.projectPath}`] : []),
    // The path, never the key: nothing in forge prints key material.
    ...(result.keyFile ? [`Wrote ${result.keyFile} (mode 600) — key id ${result.keyId}`] : []),
    'Capability policy is default-deny: add explicit grants before building.',
    // `create` is the only next step that works from here: nothing else in
    // forge writes an .rvf, so pointing at pack/validate would name a file
    // that does not exist yet.
    result.projectPath
      ? 'Next: rvforge create  (writes agent.rvf, signed with the key above)'
      : 'Next: rvforge init --keygen, then rvforge create',
  ]);
  return 0;
}

async function cmdCreate(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const operand = args.positionals[0] ?? DEFAULT_RVF_FILENAME;
  const project = await loadProject(projectPath(args));
  const key = bool(args, 'unsigned')
    ? undefined
    : await resolveSigningKey(project, str(args, 'key-file'));

  const result = await createRvf({
    outPath: resolve(operand),
    project,
    force: bool(args, 'force'),
    ...(key ? { key } : {}),
    ...(str(args, 'from') ? { fromDir: resolve(str(args, 'from') as string) } : {}),
  });

  reporter.success('create', result, () => renderCreateSummary(result, operand));
  return 0;
}

async function cmdPack(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const rvfPath = resolve(requireOperand(args, 0, 'agent.rvf'));
  const project = await loadProject(projectPath(args));
  const result = await runPack({ rvfPath, project, allowUnsigned: bool(args, 'allow-unsigned') });

  for (const warning of result.warnings) reporter.warn(warning);
  reporter.success(
    'pack',
    {
      rvfPath: result.rvfPath,
      checks: result.checks,
      securityProfile: result.securityProfile,
      manualReviewTriggers: result.manualReviewTriggers,
      capabilityManifest: result.capabilityManifest,
      softwareInventory: result.softwareInventory,
      release: result.release,
    },
    () => [
      `Packed ${result.rvfPath}`,
      ...result.checks.map((c) => `  ${statusMark(c.status)} ${c.title.padEnd(26)} ${c.detail}`),
      `  security profile   ${result.securityProfile}`,
      `  capability manifest ${result.capabilityManifest.manifestId}`,
      `  draft release       ${result.release.releaseId} (unsigned, trust level ${result.release.trustLevel})`,
      ...(result.manualReviewTriggers.length > 0
        ? [`  manual review       ${result.manualReviewTriggers.join(', ')}`]
        : []),
    ],
  );
  return 0;
}

const statusMark = (status: string): string =>
  status === 'pass' ? 'ok  ' : status === 'warn' ? 'warn' : status === 'skipped' ? 'skip' : 'FAIL';

/**
 * `test` reports; it does not throw for a failing category. The exit code is
 * what an unattended caller branches on, so a failure exits non-zero — with
 * the per-category report already printed.
 */
async function cmdTest(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const rvfPath = resolve(requireOperand(args, 0, 'agent.rvf'));
  const project = await loadProject(projectPath(args));
  const result = await runTestAgent({
    rvfPath,
    project,
    allowUnsigned: bool(args, 'allow-unsigned'),
    ...(str(args, 'receipts') ? { receiptsDir: resolve(str(args, 'receipts') as string) } : {}),
  });

  reporter.success('test', result, () => [
    `Tested ${result.rvfPath} (inspection only — nothing was executed)`,
    ...result.categories.map((c) => `  ${statusMark(c.status)} ${c.title.padEnd(30)} ${c.detail}`),
    `  ${result.passed} passed, ${result.failed} failed, ${result.skipped} skipped`,
  ]);
  if (result.ok) return 0;

  return reporter.failure(
    'test',
    new ForgeError(
      ForgeErrorCode.VERIFY_FAILED,
      `${result.failed} of ${result.categories.length} test categories failed: ` +
        result.categories
          .filter((c) => c.status === 'fail')
          .map((c) => `${c.id} (${c.detail})`)
          .join('; '),
    ),
  );
}

async function cmdPublish(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const rvfPath = resolve(requireOperand(args, 0, 'agent.rvf'));
  const project = await loadProject(projectPath(args));
  const result = await runPublish({
    rvfPath,
    project,
    allowUnsigned: bool(args, 'allow-unsigned'),
    ...(str(args, 'registry') ? { registryDir: resolve(str(args, 'registry') as string) } : {}),
    ...(str(args, 'key-file') ? { keyFile: resolve(str(args, 'key-file') as string) } : {}),
  });

  reporter.success(
    'publish',
    {
      registryDir: result.registryDir,
      publisherId: result.publisherId,
      keyId: result.keyId,
      releaseId: result.release.releaseId,
      predecessor: result.predecessor,
      capabilityManifest: result.capabilityManifest.manifestId,
      objects: result.objects,
      logEntry: result.logEntry,
      treeHead: result.treeHead,
      receiptId: result.receipt.receiptId,
      summary: result.summary,
    },
    () => [
      ...renderPublishSummary(result.summary),
      '',
      `Published to ${result.registryDir}`,
      `  release      ${result.release.releaseId}`,
      `  predecessor  ${result.predecessor ?? 'none (first release)'}`,
      `  signed by    ${result.keyId}`,
      `  objects      ${result.objects.length} written`,
      `  log entry    #${result.logEntry.index}, tree head ${result.treeHead.rootHash}`,
      `  witness      ${result.receipt.receiptId}`,
    ],
  );
  return 0;
}

async function cmdValidate(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const path = requireOperand(args, 0, 'agent.rvf');
  const result = await validateRvf(resolve(path), {
    deep: bool(args, 'deep'),
    allowUnsigned: bool(args, 'allow-unsigned'),
  });

  for (const warning of result.warnings) reporter.warn(warning);
  reporter.success('validate', result, () => [
    `${result.path}: valid RVF (format v${result.root.formatVersion}, ${formatBytes(result.size)})`,
    `  file id     ${result.root.fileId}`,
    `  segments    ${result.segments.count} (${result.segments.executableCount} executable)`,
    `  signature   ${result.root.signaturePresent ? `present (algo ${result.root.signatureAlgo})` : 'absent'}`,
    result.deep ? `  sha256      ${result.identity.sha256}` : '  sha256      not computed (pass --deep)',
    `  checked in  ${result.durationMs} ms`,
  ]);
  return 0;
}

/** Resolve `--mode`, which selects the packaging mode for one build. */
export function resolveMode(raw: string | undefined): PackagingMode | undefined {
  if (raw === undefined) return undefined;
  const mode = raw.trim().toLowerCase();
  if ((PACKAGING_MODES as readonly string[]).includes(mode)) return mode as PackagingMode;
  throw new ForgeError(
    ForgeErrorCode.USAGE,
    `Unknown packaging mode "${raw}". Expected one of ${PACKAGING_MODES.join(', ')}.`,
    { mode: raw, known: [...PACKAGING_MODES] },
  );
}

async function cmdBuild(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const config = await loadConfig(configPath(args));
  const result = await runBuild({
    config,
    ...splitBuildOperands(args.positionals),
    ...(resolveMode(str(args, 'mode')) ? { mode: resolveMode(str(args, 'mode')) } : {}),
    outDir: str(args, 'out') ?? 'forge-out',
    force: bool(args, 'force'),
    allowUnsigned: bool(args, 'allow-unsigned'),
  });

  for (const warning of result.warnings) reporter.warn(warning);
  reporter.success(
    'build',
    {
      outDir: result.outDir,
      manifestSha256: result.manifestSha256,
      rvfIdentity: result.manifest.rvf.sha256,
      targets: result.manifest.targets,
      packaging: result.manifest.packaging.mode,
      bundles: result.inventory.bundles,
      compatibilityMatrixRevision: result.provenance.compatibilityMatrixRevision,
      receiptId: result.receipt.receiptId,
      packaged: result.packaged,
      status: result.provenance.status,
      toolchain: result.toolchain,
      files: result.provenance.files,
    },
    () => [
      `Staged build in ${result.outDir}`,
      `  rvf identity   ${result.manifest.rvf.sha256}`,
      `  manifest hash  ${result.manifestSha256}`,
      `  targets        ${result.manifest.targets.join(', ')}`,
      `  packaging      ${result.manifest.packaging.mode} (${result.provenance.packaging.status})`,
      `  bundles        ${result.inventory.bundles.length}`,
      `  files          ${result.provenance.files.length}`,
      `  witness        ${result.receipt.receiptId}`,
      result.packaged ? '' : '  No installers were produced — this is a staged bundle, not a release.',
    ].filter(Boolean),
  );
  return 0;
}

async function cmdSubmit(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const config = await loadConfig(configPath(args));
  const options = {
    config,
    ...splitBuildOperands(args.positionals),
    cached: bool(args, 'cached'),
    allowUnsigned: bool(args, 'allow-unsigned'),
    confirmed: bool(args, 'yes'),
  };

  // Show the estimate before the upload, whether or not it is confirmed.
  if (!options.confirmed) {
    const plan = await planSubmit(options);
    for (const line of renderEstimate(plan.estimate)) reporter.warn(line);
  }

  const result = await runSubmit(options);
  reporter.success(
    'submit',
    { build: result.build, estimate: result.estimate, endpoint: result.endpoint },
    () => [
      `Submitted build ${result.build.id} to ${result.endpoint}`,
      `  state    ${result.build.state}`,
      `  targets  ${result.build.targets.join(', ')}`,
      ...renderEstimate(result.estimate),
      `Next: forge status ${result.build.id}`,
    ],
  );
  return 0;
}

async function cmdStatus(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const id = requireOperand(args, 0, 'BUILD_ID');
  const result = await runStatus(id);
  reporter.success('status', result, () => [
    `Build ${result.build.id}: ${result.build.state}${result.terminal ? ' (terminal)' : ''}`,
    `  targets   ${result.build.targets.join(', ')}`,
    `  rvf       ${result.build.rvfIdentity}`,
    `  updated   ${result.build.updatedAt}`,
    ...(result.build.message ? [`  message   ${result.build.message}`] : []),
  ]);
  return 0;
}

async function cmdDownload(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const id = requireOperand(args, 0, 'BUILD_ID');
  const result = await runDownload(id, {
    outDir: str(args, 'out'),
    only: args.repeated.get('only'),
  });
  reporter.success('download', result, () => [
    `Downloaded ${result.artifacts.length} artifact(s) to ${result.outDir}`,
    ...result.artifacts.map((a) => `  ${a.name}  ${formatBytes(a.sizeBytes)}  sha256 verified`),
  ]);
  return 0;
}

async function cmdVerify(args: ParsedArgs, reporter: Reporter): Promise<number> {
  const target = requireOperand(args, 0, 'artifact');
  const result = await runVerify(resolve(target), { provenancePath: str(args, 'provenance') });
  reporter.success('verify', result, () => [
    `Verified against ${result.provenancePath}`,
    `  rvf identity   ${result.rvfIdentity}`,
    `  manifest hash  ${result.manifestSha256.expected}`,
    `  files checked  ${result.files.length}, all matching`,
    `  witness chain  ${result.witnessChain.count} receipt(s), intact`,
    ...(result.receipt ? [`  receipt        ${result.receipt.receiptId}`] : []),
    ...(result.target ? [`  target         ${result.target.path} ok`] : []),
  ]);
  return 0;
}

/* istanbul ignore next -- entrypoint wiring */
if (require.main === module) {
  main(process.argv.slice(2))
    .then((code) => {
      process.exitCode = code;
    })
    .catch((err: unknown) => {
      process.stderr.write(`internal error: ${toForgeError(err).message}\n`);
      process.exitCode = 20;
    });
}
