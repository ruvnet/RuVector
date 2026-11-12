#!/usr/bin/env node
/**
 * adr-index.mjs — canonical ADR index generator + duplicate-number gate.
 *
 * Usage:
 *   node scripts/adr-index.mjs             # regenerate docs/adr/INDEX.md
 *   node scripts/adr-index.mjs --check     # exit 1 if a NEW duplicate ADR number
 *                                          # exists (one not in the frozen
 *                                          # historical list below); exit 0 otherwise
 *   node scripts/adr-index.mjs --self-test # build an adversarial fixture tree in a
 *                                          # temp dir (symlinks, hostile filenames)
 *                                          # and assert the script handles it safely
 *
 * Policy: see docs/adr/ADR-316-adr-numbering-hygiene.md.
 * Duplicate numbers that existed before ADR-316 are FROZEN historical
 * artifacts — they are never renamed, and are disambiguated in prose as
 * "ADR-NNN (slug)". This script's --check mode is the CI gate that keeps
 * the frozen list from growing.
 *
 * Security posture (Phase-4 review, PR #857):
 *   - Symlinks in docs/adr/ are NEVER followed (neither file nor directory
 *     symlinks) — they are skipped with a warning, so a committed symlink
 *     cannot exfiltrate content from outside the tree into INDEX.md.
 *   - Every visited entry is belt-and-braces asserted (via realpath) to
 *     resolve inside docs/adr/; anything that escapes is skipped.
 *   - Traversal is wrapped per-entry: broken symlinks, permission errors,
 *     and cycles produce a warning + skip, never a stack-trace abort, so
 *     the --check gate fails (or passes) cleanly on adversarial trees.
 *   - All filename-derived text (slug, rel) is escaped before being
 *     interpolated into INDEX.md, and link targets are URI-encoded.
 *
 * No dependencies beyond node >= 18 and git.
 */

import { execFileSync, spawnSync } from 'node:child_process';
import {
  readdirSync, readFileSync, writeFileSync, realpathSync,
  mkdirSync, mkdtempSync, rmSync, symlinkSync, copyFileSync,
} from 'node:fs';
import { join, relative, dirname, basename, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { fileURLToPath } from 'node:url';

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const REPO_ROOT = join(dirname(SCRIPT_PATH), '..');
const ADR_DIR = join(REPO_ROOT, 'docs', 'adr');
const INDEX_PATH = join(ADR_DIR, 'INDEX.md');

/**
 * FROZEN historical duplicate list (verified against origin/main, 2026-08-19,
 * at the time ADR-316 was written). Key = plain ADR number (no namespace,
 * no a/b sub-ADR suffix), value = number of files historically sharing it.
 * Do NOT add entries here — new collisions must be renumbered before merge.
 */
const FROZEN_DUPLICATES = {
  272: 5,
  264: 3, 252: 3, 194: 3, 144: 3,
  268: 2, 266: 2, 260: 2, 258: 2, 256: 2, 254: 2,
  143: 2, 139: 2, 138: 2, 137: 2, 136: 2, 135: 2,
  134: 2, 133: 2, 132: 2, 117: 2,
  96: 2, 95: 2, 94: 2, 93: 2, 91: 2, 90: 2,
};

// Filename shapes handled:
//   ADR-016-slug.md                (plain, canonical counter)
//   ADR-040a-slug.md               (sub-ADR convention — distinct from ADR-040)
//   ADR-DB-001-slug.md             (namespaced family: DB, CE, QE, TTS, ...)
const FILE_RE = /^ADR-(?:([A-Z]{2,4})-)?(\d+)([a-z])?(?:[-_](.+?))?\.md$/;

function warn(msg) {
  console.error(`adr-index: warning: ${msg}`);
}

/**
 * Walk `dir` collecting .md files. Symlinks (file or directory) are NEVER
 * followed — skipped with a warning. Every kept entry must realpath-resolve
 * under `rootReal` (belt-and-braces against anything that still escapes).
 * Unreadable/broken entries are skipped with a warning, never a throw.
 */
function walk(dir, rootReal, visited = new Set()) {
  const out = [];
  let dirReal;
  try {
    dirReal = realpathSync(dir);
  } catch (err) {
    warn(`cannot resolve directory ${dir}: ${err.message}`);
    return out;
  }
  if (visited.has(dirReal)) {
    warn(`directory cycle detected at ${dir} — skipping`);
    return out;
  }
  visited.add(dirReal);

  let dirents;
  try {
    dirents = readdirSync(dir, { withFileTypes: true });
  } catch (err) {
    warn(`cannot read directory ${dir}: ${err.message}`);
    return out;
  }
  for (const dirent of dirents) {
    const p = join(dir, dirent.name);
    if (dirent.isSymbolicLink()) {
      warn(`skipping symlink (not followed): ${p}`);
      continue;
    }
    if (dirent.isDirectory()) {
      out.push(...walk(p, rootReal, visited));
      continue;
    }
    if (!dirent.isFile() || !dirent.name.endsWith('.md')) continue;
    try {
      const real = realpathSync(p);
      if (real !== rootReal && !real.startsWith(rootReal + sep)) {
        warn(`skipping entry that resolves outside the ADR tree: ${p} -> ${real}`);
        continue;
      }
    } catch (err) {
      warn(`skipping unresolvable entry ${p}: ${err.message}`);
      continue;
    }
    out.push(p);
  }
  return out;
}

function parseTitle(content, fallback) {
  const m = content.match(/^#\s+(.+?)\s*$/m);
  return m ? m[1].trim() : fallback;
}

function parseStatus(content) {
  // Frontmatter: status: Accepted
  let m = content.match(/^status:\s*(.+?)\s*$/im);
  if (m && content.slice(0, 4) === '---\n') return m[1];
  // Inline: **Status**: Proposed  /  **Status:** Proposed  /  - Status: Proposed
  m = content.match(/^[-*\s]*\*\*Status:?\*\*:?\s*(.+?)\s*$/im) ||
      content.match(/^[-*]\s*Status:\s*(.+?)\s*$/im);
  if (m) return m[1].replace(/\*+/g, '').trim();
  // Section: ## Status \n <first non-empty line>
  m = content.match(/^##\s*Status\s*\n+([^\n#]+)/im);
  if (m) return m[1].trim();
  return '';
}

function gitDate(relPath, repoRoot) {
  try {
    const out = execFileSync(
      'git', ['log', '--follow', '-1', '--format=%ci', '--', relPath],
      { cwd: repoRoot, encoding: 'utf8' },
    ).trim();
    return out ? out.slice(0, 10) : '';
  } catch {
    return '';
  }
}

function collect({ withDates, adrDir = ADR_DIR, repoRoot = REPO_ROOT }) {
  let rootReal;
  try {
    rootReal = realpathSync(adrDir);
  } catch (err) {
    throw new Error(`ADR directory not found or unreadable: ${adrDir} (${err.message})`);
  }
  const entries = [];
  for (const abs of walk(adrDir, rootReal)) {
    // path.sep-safe (Windows included): basename + relative, normalized to '/'.
    const rel = relative(repoRoot, abs).split(sep).join('/');
    const base = basename(abs);
    if (base === 'README.md' || base === 'INDEX.md') continue;
    const m = base.match(FILE_RE);
    if (!m) continue; // non-ADR markdown (design notes etc.)
    const [, ns, numStr, subSuffix, slug] = m;
    const num = parseInt(numStr, 10);
    let title = '', status = '';
    try {
      const content = readFileSync(abs, 'utf8');
      title = parseTitle(content, slug ?? base);
      status = parseStatus(content);
    } catch { /* unreadable file — keep filename-derived fields */ }
    entries.push({
      ns: ns ?? '',
      num,
      subSuffix: subSuffix ?? '',
      slug: slug ?? '',
      rel,
      title,
      status,
      date: withDates ? gitDate(rel, repoRoot) : '',
      // Duplicate-detection key. The a/b sub-ADR suffix makes a DISTINCT
      // key on purpose: ADR-040 / ADR-040a / ADR-040b never collide.
      key: `${ns ? ns + ':' : ''}${num}${subSuffix ?? ''}`,
    });
  }
  entries.sort((a, b) =>
    a.ns.localeCompare(b.ns) || a.num - b.num ||
    a.subSuffix.localeCompare(b.subSuffix) || a.rel.localeCompare(b.rel));
  return entries;
}

function groupDuplicates(entries) {
  const byKey = new Map();
  for (const e of entries) {
    if (!byKey.has(e.key)) byKey.set(e.key, []);
    byKey.get(e.key).push(e);
  }
  return byKey;
}

function maxPlainNum(entries) {
  // reduce, not Math.max(...): an empty set yields 0, not -Infinity.
  return entries.filter(e => !e.ns).reduce((m, e) => Math.max(m, e.num), 0);
}

function check() {
  const entries = collect({ withDates: false });
  const byKey = groupDuplicates(entries);
  const violations = [];
  for (const [key, group] of byKey) {
    if (group.length < 2) continue;
    const e = group[0];
    const frozen = e.ns === '' && e.subSuffix === '' ? FROZEN_DUPLICATES[e.num] : undefined;
    if (frozen === undefined) {
      violations.push(`NEW duplicate ${key}: ${group.map(g => g.rel).join(', ')}`);
    } else if (group.length > frozen) {
      violations.push(
        `Duplicate count for ADR-${e.num} grew: ${group.length} files ` +
        `(frozen historical allowance is ${frozen}): ${group.map(g => g.rel).join(', ')}`);
    }
  }
  if (violations.length > 0) {
    console.error('ADR duplicate-number check FAILED:\n');
    for (const v of violations) console.error(`  - ${v}`);
    console.error('\nPick the next available number from docs/adr/INDEX.md ' +
      '(regenerate with `node scripts/adr-index.mjs`). Never reuse a number.');
    process.exit(1);
  }
  const maxNum = maxPlainNum(entries);
  console.log(`ADR duplicate-number check OK: ${entries.length} ADR files, ` +
    `no duplicates outside the frozen historical list. Next available number: ${maxNum + 1}.`);
}

// Escape text destined for a markdown table cell.
function mdCell(s) {
  return String(s ?? '').replace(/\|/g, '\\|').replace(/\s+/g, ' ').trim();
}

// Escape text destined for markdown link text inside a table cell.
function mdLinkText(s) {
  return mdCell(s).replace(/\[/g, '\\[').replace(/\]/g, '\\]');
}

// Encode a markdown link target: encodeURI, plus the characters that would
// terminate the link or the table cell but that encodeURI leaves alone.
function mdLinkTarget(s) {
  return encodeURI(String(s ?? ''))
    .replace(/\(/g, '%28')
    .replace(/\)/g, '%29');
}

function formatRow(e, group) {
  const numLabel = `ADR-${e.ns ? e.ns + '-' : ''}${String(e.num).padStart(3, '0')}${e.subSuffix}`;
  const dup = group.length > 1
    ? `DUPLICATE ×${group.length} — cite as \`ADR-${e.num} (${mdCell(e.slug) || 'untitled'})\``
    : '';
  const shortRel = e.rel.replace('docs/adr/', '');
  const link = `[\`${mdLinkText(shortRel)}\`](${mdLinkTarget('./' + shortRel)})`;
  return `| ${numLabel} | ${mdCell(e.title)} | ${link} | ${e.date} | ${mdCell(e.status).slice(0, 80)} | ${dup} |`;
}

function generate() {
  const entries = collect({ withDates: true });
  const byKey = groupDuplicates(entries);
  const plain = entries.filter(e => !e.ns);
  const maxNum = maxPlainNum(entries);
  const next = maxNum + 1;
  const dupKeys = [...byKey.values()].filter(g => g.length > 1);
  const dupFileCount = dupKeys.reduce((n, g) => n + g.length, 0);

  const lines = [];
  lines.push('# ADR Index');
  lines.push('');
  lines.push(`**Next available ADR number: ${next}**`);
  lines.push('');
  lines.push('> Generated by `node scripts/adr-index.mjs` — do not edit by hand.');
  lines.push('> This file is the canonical allocation counter for new ADR numbers');
  lines.push('> (policy: `ADR-316`). Duplicate numbers listed below are frozen');
  lines.push('> historical artifacts and are cited as `ADR-NNN (slug)`.');
  lines.push('> CI gate: `node scripts/adr-index.mjs --check`.');
  lines.push('');
  lines.push(`- ADR files indexed: **${entries.length}** ` +
    `(${plain.length} on the canonical counter, ${entries.length - plain.length} in namespaced families)`);
  lines.push(`- Highest allocated number: **ADR-${maxNum}**`);
  lines.push(`- Frozen duplicate numbers: **${dupKeys.length}** ` +
    `(spanning ${dupFileCount} files)`);
  lines.push('');
  lines.push('| Number | Title | File | Last commit | Status | Duplicate |');
  lines.push('|---|---|---|---|---|---|');
  for (const e of entries) {
    lines.push(formatRow(e, byKey.get(e.key)));
  }
  lines.push('');
  writeFileSync(INDEX_PATH, lines.join('\n'));
  console.log(`Wrote ${relative(REPO_ROOT, INDEX_PATH).split(sep).join('/')}: ${entries.length} entries, ` +
    `${dupKeys.length} frozen duplicate numbers, next available ADR-${next}.`);
}

/**
 * --self-test: build an adversarial fixture tree in a temp dir, copy this
 * script into it (REPO_ROOT derives from the script location, so the copy
 * operates on the fixture), and assert:
 *   1. a symlink pointing outside the tree is skipped — its content never
 *      reaches INDEX.md;
 *   2. a hostile filename (|, [, ], (, )) is escaped in the generated table;
 *   3. a tree containing broken symlinks and a symlink loop is traversed
 *      cleanly — --check exits 0 with warnings, no stack trace.
 */
function selfTest() {
  const failures = [];
  const ok = (cond, label) => {
    if (cond) console.log(`  PASS  ${label}`);
    else { console.error(`  FAIL  ${label}`); failures.push(label); }
  };

  const root = mkdtempSync(join(tmpdir(), 'adr-index-selftest-'));
  try {
    const adr = join(root, 'docs', 'adr');
    mkdirSync(adr, { recursive: true });
    mkdirSync(join(root, 'scripts'), { recursive: true });
    copyFileSync(SCRIPT_PATH, join(root, 'scripts', 'adr-index.mjs'));

    // Legit ADR.
    writeFileSync(join(adr, 'ADR-001-real.md'),
      '# Real decision\n\n**Status**: Accepted\n');
    // Hostile filename: pipes, brackets, parens in slug.
    const hostile = 'ADR-090-beta|BROKEN](httpX) [pwn](httpY.md';
    writeFileSync(join(adr, hostile), '# Hostile | title\n\n**Status**: Evil|Status\n');
    // Secret outside the ADR tree + file symlink to it.
    writeFileSync(join(root, 'outside-secret.txt'),
      '# TOP SECRET DEPLOY KEY abc123\nstatus: leaked\n');
    symlinkSync(join(root, 'outside-secret.txt'), join(adr, 'ADR-500-exfil.md'));
    // Directory symlink escaping the tree.
    mkdirSync(join(root, 'outside-dir'));
    writeFileSync(join(root, 'outside-dir', 'ADR-600-outside.md'),
      '# OUTSIDE REPO SECRET\n');
    symlinkSync(join(root, 'outside-dir'), join(adr, 'vendor'));
    // Broken symlink.
    symlinkSync('/nonexistent/target.md', join(adr, 'ADR-502-broken.md'));
    // Symlink loop.
    symlinkSync('.', join(adr, 'loop'));

    const script = join(root, 'scripts', 'adr-index.mjs');
    const run = (args) => {
      const r = spawnSync(process.execPath, [script, ...args],
        { encoding: 'utf8', timeout: 30_000 });
      return { code: r.status ?? 1, stdout: r.stdout ?? '', stderr: r.stderr ?? '' };
    };

    // --- 3. adversarial tree: --check exits cleanly, no stack trace ---
    const chk = run(['--check']);
    ok(chk.code === 0, '--check exits 0 on adversarial tree (no dupes present)');
    ok(chk.stdout.includes('check OK'), '--check prints its normal summary');
    ok(!(chk.stdout + chk.stderr).includes('at walk'),
      'no stack trace on broken symlink / symlink loop');
    ok(chk.stderr.includes('skipping symlink'),
      'symlinks produce a skip warning');

    // --- 1 + 2. generate: symlinks skipped, hostile filename escaped ---
    const gen = run([]);
    ok(gen.code === 0, 'generate exits 0 on adversarial tree');
    const index = readFileSync(join(adr, 'INDEX.md'), 'utf8');
    ok(!index.includes('TOP SECRET'),
      'file symlink target content is NOT inlined into INDEX.md');
    ok(!index.includes('OUTSIDE REPO SECRET') && !index.includes('ADR-600'),
      'directory symlink is NOT followed');
    ok(!index.includes('ADR-500') && !index.includes('ADR-502'),
      'symlinked/broken entries are absent from the index');
    ok(index.includes('ADR-090'), 'hostile-named ADR is still indexed');
    const hostileRow = index.split('\n').find(l => l.includes('ADR-090'));
    const unescapedPipes = hostileRow.replace(/\\\|/g, '').split('|').length - 1;
    ok(unescapedPipes === 7,
      'hostile filename: table row has exactly 7 unescaped pipes (cells intact)');
    // The link target is the one that starts with './' (escaped `\](`
    // sequences inside the link text would otherwise match first).
    const target = hostileRow.match(/\]\((\.\/[^)]*)\)/)?.[1] ?? '';
    ok(target.startsWith('./ADR-090') && !/[ |[\]]/.test(target)
      && !target.includes('('),
      'hostile filename: link target is URI-encoded (no raw space | [ ] ( )');
    ok(index.includes('Next available ADR number: 91'),
      'counter unaffected by symlinked ADR-500/600');
  } finally {
    rmSync(root, { recursive: true, force: true });
  }

  if (failures.length > 0) {
    console.error(`\nself-test FAILED: ${failures.length} assertion(s)`);
    process.exit(1);
  }
  console.log('\nself-test OK: all assertions passed');
}

try {
  if (process.argv.includes('--self-test')) selfTest();
  else if (process.argv.includes('--check')) check();
  else generate();
} catch (err) {
  console.error(`adr-index: fatal: ${err.message}`);
  process.exit(1);
}
