/**
 * A dependency-free argv parser and the file/stdin readers the CLI needs. No
 * `child_process`, no network — only `node:fs` reads and stdin.
 */

import * as fs from 'node:fs';
import type { Binding } from '../binding';

export interface ParsedArgs {
  /** Positionals after the subcommand. */
  _: string[];
  flags: Record<string, string | boolean>;
}

/** Parse `--key value`, `--key=value`, `--flag`, and positionals. */
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

export function flagBool(flags: ParsedArgs['flags'], key: string): boolean {
  return flags[key] === true || flags[key] === 'true';
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

/** Resolve `state` from `--state`, `--state-file`, or stdin (in that order). */
export function readState(flags: ParsedArgs['flags']): string {
  const inline = flagString(flags, 'state');
  if (inline !== undefined) return inline;
  const file = flagString(flags, 'state-file');
  if (file !== undefined) return fs.readFileSync(file, 'utf8');
  return fs.readFileSync(0, 'utf8');
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
