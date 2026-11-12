#!/usr/bin/env node
/**
 * workspace-check.mjs - Workspace-membership guard (PIR #859).
 *
 * Fails (exit 1) if any crates/⋆⋆/Cargo.toml is in the implicit-orphan state:
 * neither a root workspace member, nor covered by the root `exclude` list,
 * nor a workspace root of its own (or nested under one). Orphan crates are
 * invisible to `cargo test --workspace` — their tests silently never run,
 * which is how 12 crates accumulated before #859.
 *
 * The root workspace deliberately uses literal `members` entries (no
 * `crates/*` glob): membership stays an explicit, reviewable decision, and
 * this script is the backstop that makes silent orphaning impossible.
 *
 * Zero dependencies. Usage:
 *   node scripts/workspace-check.mjs
 */

import { readFileSync, readdirSync, existsSync } from 'fs';
import { join, resolve, relative, dirname, sep } from 'path';

const ROOT = resolve(import.meta.dirname, '..');

// --- Parse the root manifest's members/exclude arrays (tolerant, no TOML dep) ---

function parseStringArray(toml, key) {
  const m = toml.match(new RegExp(`^${key}\\s*=\\s*\\[`, 'm'));
  if (!m) return [];
  let i = m.index + m[0].length;
  const out = [];
  while (i < toml.length && toml[i] !== ']') {
    const ch = toml[i];
    if (ch === '#') {
      // comment: skip to end of line
      while (i < toml.length && toml[i] !== '\n') i++;
    } else if (ch === '"') {
      const end = toml.indexOf('"', i + 1);
      if (end === -1) throw new Error(`unterminated string in ${key}`);
      out.push(toml.slice(i + 1, end));
      i = end + 1;
    } else {
      i++;
    }
  }
  return out;
}

const rootToml = readFileSync(join(ROOT, 'Cargo.toml'), 'utf8');
const members = parseStringArray(rootToml, 'members');
const excludes = parseStringArray(rootToml, 'exclude');

// --- Matching helpers ---

// Cargo path globs: `*` matches a single path segment.
function globToRegex(pattern) {
  const escaped = pattern
    .split('*')
    .map((part) => part.replace(/[.+^${}()|[\]\\?]/g, '\\$&'))
    .join('[^/]+');
  return new RegExp(`^${escaped}$`);
}

const memberMatchers = members.map(globToRegex);
const excludeMatchers = excludes.map(globToRegex);

function isMember(rel) {
  return memberMatchers.some((re) => re.test(rel));
}

// An excluded directory covers everything beneath it (cargo semantics:
// excluding a path removes it and its contents from workspace discovery).
function isExcluded(rel) {
  return excludeMatchers.some((re) => {
    if (re.test(rel)) return true;
    const parts = rel.split('/');
    for (let n = 1; n < parts.length; n++) {
      if (re.test(parts.slice(0, n).join('/'))) return true;
    }
    return false;
  });
}

function hasOwnWorkspace(manifestPath) {
  const toml = readFileSync(manifestPath, 'utf8');
  return /^\s*\[workspace([\].])/m.test(toml);
}

// --- Walk crates/ for every Cargo.toml ---

const SKIP_DIRS = new Set(['target', 'node_modules', '.git', 'pkg']);

function findManifests(dir, acc) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) findManifests(join(dir, entry.name), acc);
    } else if (entry.name === 'Cargo.toml') {
      acc.push(join(dir, entry.name));
    }
  }
  return acc;
}

const manifests = findManifests(join(ROOT, 'crates'), []);

// Directories (relative, /-separated) whose Cargo.toml declares [workspace].
const workspaceRoots = new Set(
  manifests
    .filter(hasOwnWorkspace)
    .map((p) => relative(ROOT, dirname(p)).split(sep).join('/')),
);

function underOwnWorkspace(relDir) {
  const parts = relDir.split('/');
  for (let n = 1; n <= parts.length; n++) {
    if (workspaceRoots.has(parts.slice(0, n).join('/'))) return true;
  }
  return false;
}

// --- Classify ---

const orphans = [];
for (const manifest of manifests) {
  const relDir = relative(ROOT, dirname(manifest)).split(sep).join('/');
  if (isMember(relDir)) continue;
  if (isExcluded(relDir)) continue;
  if (underOwnWorkspace(relDir)) continue;
  orphans.push(relDir);
}

// Sanity: stale members pointing at nothing are cargo's problem, but flag
// them here too since we already parsed the list.
const staleMembers = members.filter(
  (m) => !m.includes('*') && !existsSync(join(ROOT, m, 'Cargo.toml')),
);

if (orphans.length === 0 && staleMembers.length === 0) {
  console.log(
    `workspace-check: OK — ${manifests.length} manifests under crates/, ` +
      `${members.length} members, ${excludes.length} excludes, 0 orphans.`,
  );
  process.exit(0);
}

if (orphans.length > 0) {
  console.error(
    `workspace-check: FAIL — ${orphans.length} orphan crate(s) under crates/ ` +
      `(Cargo.toml present, but not a workspace member, not excluded, and not ` +
      `its own workspace root):\n`,
  );
  for (const o of orphans.sort()) console.error(`  ${o}`);
  console.error(
    `\nFix: add each path to \`members\` in the root Cargo.toml (preferred — ` +
      `its tests then run in CI), or to \`exclude\` with a comment stating why ` +
      `(see #859).`,
  );
}

if (staleMembers.length > 0) {
  console.error(
    `workspace-check: FAIL — ${staleMembers.length} member entr${
      staleMembers.length === 1 ? 'y' : 'ies'
    } with no Cargo.toml on disk:\n`,
  );
  for (const s of staleMembers.sort()) console.error(`  ${s}`);
}

process.exit(1);
