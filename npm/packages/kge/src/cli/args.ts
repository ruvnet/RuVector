/**
 * A dependency-free argv parser and the file readers/writers the CLI needs. No
 * `child_process`, no network — only `node:fs`.
 */

import * as fs from 'node:fs';
import type { Binding } from '../binding';

export interface ParsedArgs {
  /** Positionals after the subcommand. */
  _: string[];
  flags: Record<string, string | boolean>;
}

/** Parse `--key value`, `--key=value`, `--flag`, `-k <n>`, and positionals. */
export function parseArgs(argv: readonly string[]): ParsedArgs {
  const out: ParsedArgs = { _: [], flags: {} };
  for (let i = 0; i < argv.length; i++) {
    const token = argv[i];
    if (token.startsWith('--')) {
      const body = token.slice(2);
      const eq = body.indexOf('=');
      if (eq >= 0) {
        out.flags[body.slice(0, eq)] = body.slice(eq + 1);
      } else if (i + 1 < argv.length && !argv[i + 1].startsWith('-')) {
        out.flags[body] = argv[++i];
      } else {
        out.flags[body] = true;
      }
    } else if (token === '-h') {
      out.flags.help = true;
    } else if (token === '-v') {
      out.flags.version = true;
    } else if (token === '-k') {
      if (i + 1 < argv.length) out.flags.k = argv[++i];
    } else {
      out._.push(token);
    }
  }
  return out;
}

export function flagString(flags: ParsedArgs['flags'], key: string): string | undefined {
  const v = flags[key];
  return typeof v === 'string' ? v : undefined;
}

export function flagNumber(flags: ParsedArgs['flags'], key: string): number | undefined {
  const v = flagString(flags, key);
  return v !== undefined ? Number(v) : undefined;
}

export function requireFlag(flags: ParsedArgs['flags'], key: string): string {
  const v = flagString(flags, key);
  if (v === undefined) throw new Error(`missing required --${key}`);
  return v;
}

export interface CliContext {
  binding?: Binding;
  out: (s: string) => void;
  err: (s: string) => void;
}

export interface MainDeps {
  binding?: Binding;
  stdout?: (s: string) => void;
  stderr?: (s: string) => void;
}

export function readJsonFile<T = unknown>(path: string): T {
  return JSON.parse(fs.readFileSync(path, 'utf8')) as T;
}

/** Parse a JSONL file: one JSON value per non-empty line. */
export function readJsonl<T = unknown>(path: string): T[] {
  return fs
    .readFileSync(path, 'utf8')
    .split('\n')
    .map((line) => line.trim())
    .filter((line) => line.length > 0)
    .map((line) => JSON.parse(line) as T);
}

export function writeFile(path: string, contents: string): void {
  fs.writeFileSync(path, contents);
}
