/**
 * CLI output. Two shapes, one source of truth: a human-readable rendering and
 * a `--json` envelope with the same payload, so an unattended caller never has
 * to scrape prose.
 */

import { ForgeError, type ForgeErrorJson } from './errors';
import { formatBytes, type BuildEstimate } from './estimate';

export interface JsonEnvelope {
  ok: boolean;
  command: string;
  data?: unknown;
  error?: ForgeErrorJson;
  exitCode: number;
}

export interface Reporter {
  readonly json: boolean;
  /** Emit the successful result of a command. */
  success(command: string, data: unknown, lines: () => string[]): void;
  /** Emit a failure. Returns the process exit code. */
  failure(command: string, error: ForgeError): number;
  /** Emit a non-fatal note. Suppressed by `--quiet` and in JSON mode. */
  warn(message: string): void;
}

export interface ReporterOptions {
  json?: boolean;
  quiet?: boolean;
  stdout?: (text: string) => void;
  stderr?: (text: string) => void;
}

export function createReporter(options: ReporterOptions = {}): Reporter {
  const json = Boolean(options.json);
  const quiet = Boolean(options.quiet);
  const out = options.stdout ?? ((text: string): void => void process.stdout.write(`${text}\n`));
  const err = options.stderr ?? ((text: string): void => void process.stderr.write(`${text}\n`));

  return {
    json,
    success(command, data, lines): void {
      if (json) {
        out(JSON.stringify({ ok: true, command, data, exitCode: 0 } satisfies JsonEnvelope));
        return;
      }
      if (quiet) return;
      for (const line of lines()) out(line);
    },
    failure(command, error): number {
      const exitCode = error.exitCode;
      if (json) {
        out(JSON.stringify({ ok: false, command, error: error.toJSON(), exitCode } satisfies JsonEnvelope));
      } else {
        err(`error: ${error.message}`);
        err(`code:  ${error.code}`);
      }
      return exitCode;
    },
    warn(message): void {
      if (json || quiet) return;
      err(`warning: ${message}`);
    },
  };
}

/** Render a build estimate as aligned lines. */
export function renderEstimate(estimate: BuildEstimate): string[] {
  const lines = [
    `Estimate (${estimate.cached ? 'warm caches' : 'cold caches'}):`,
    ...estimate.targets.map(
      (t) =>
        `  ${t.target.padEnd(16)} ${String(t.minutes).padStart(2)} min  ` +
        `$${t.costUsd.toFixed(4)}  ${formatBytes(t.outputBytes)}  [${t.artifacts.join(', ')}]`,
    ),
    `  ${'total'.padEnd(16)} ${String(estimate.wallClockMinutes).padStart(2)} min wall-clock, ` +
      `${estimate.totalRunnerMinutes} runner-min, $${estimate.totalCostUsd.toFixed(4)}, ` +
      `${formatBytes(estimate.totalOutputBytes)} of output`,
  ];
  return lines;
}

export const USAGE = `RVForge — one canonical RVF to signed platform installers

Usage:
  rvforge init [--force] [--with-project] [--keygen] [--key-file <path>]
  rvforge create [agent.rvf] [--from <dir>] [--unsigned] [--force]
  rvforge validate <agent.rvf> [--deep] [--allow-unsigned]
  rvforge build [agent.rvf] [targets...] [--mode <mode>] [--out <dir>] [--force]
  rvforge submit [agent.rvf] [targets...] [--yes] [--cached]
  rvforge status <BUILD_ID>
  rvforge download <BUILD_ID> [--out <dir>] [--only <name>]
  rvforge verify <artifact|dir> [--provenance <path>]

Publisher (marketplace) verbs — read rvforge.json:
  rvforge pack <agent.rvf> [--project <path>] [--allow-unsigned]
  rvforge test <agent.rvf> [--project <path>] [--receipts <dir>]
  rvforge publish <agent.rvf> [--registry <dir>] [--key-file <path>]

Getting started — the full sequence from an empty directory:
  rvforge init --keygen            scaffold forge.config.json, rvforge.json, a key
  rvforge create                   write a signed agent.rvf
  rvforge validate agent.rvf --deep
  rvforge test agent.rvf
  rvforge build --mode embedded

Targets:
  windows-x64  windows-arm64  macos-x64  macos-arm64  macos-universal
  linux-x64    linux-arm64    rvm
  Aliases: windows, macos, linux

Packaging modes (--mode):
  embedded          The bundle carries the full RVF; runs offline
  thin              The bundle carries a signed RVF locator; fetched on install
  shared-reader     Register .rvf with the OS (planned; rejected by the matrix)

Global flags:
  --json            Machine-readable output on stdout
  --quiet           Suppress human-readable output
  --config <path>   Build config file (default: forge.config.json)
  --project <path>  Marketplace project file (default: rvforge.json)
  --registry <dir>  Local registry directory (default: ~/.rvforge/registry)
  --key-file <path> Ed25519 publisher key; never printed, path only
  --help, -h        Show this help
  --version, -v     Print the forge version

Environment:
  FORGE_API_URL     Hosted build service base URL
  FORGE_API_TOKEN   Bearer token for the hosted service (never pass as a flag)

Exit codes:
  0 success   2 usage           3 invalid rvf   4 unsigned segment
  5 target    6 manifest        7 network       8 auth
  9 verify    10 io             11 not found    12 config
  13 toolchain 14 policy        15 license      16 key
  17 registry  18 lineage       20 internal`;
