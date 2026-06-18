#!/usr/bin/env node
/**
 * Static no-op detector for the SONA learn-from-feedback seams.
 *
 * The #519 and #553 regressions were both "the function exists but its body
 * does nothing" stubs that shipped because nothing checked the seam. This
 * tripwire extracts each named seam function body, strips comments and
 * logging-only statements, and fails if no state-mutating statement remains
 * (assignment / compound assignment / increment / non-logging call).
 *
 * It is deliberately simple and lenient — a tripwire, not a verifier: a
 * `let x = ...;` binding counts as evidence of real work. The failure mode
 * it guards is "body is empty, or only comments + console/log lines".
 *
 * Usage:    node scripts/sona-drift/stub-tripwire.mjs
 * Exit:     non-zero if any required seam is missing or effectively empty.
 * Exports:  scanFile(absPath, fnNames) for the demo in README.md.
 */

import { readFileSync, existsSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO = path.resolve(__dirname, '..', '..');

/**
 * Seam registry. `required: false` functions may be absent (reported, not
 * fatal) — recordRouteOutcome only exists once the #517 fix lands on main.
 */
const TARGETS = [
  {
    file: 'crates/sona/src/wasm.rs',
    fns: [
      { name: 'learn_from_feedback', required: true },
      { name: 'end_trajectory', required: true },
    ],
  },
  {
    file: 'crates/sona/src/napi_simple.rs',
    fns: [
      { name: 'learn_from_feedback', required: false },
      { name: 'end_trajectory', required: true },
    ],
  },
  {
    file: 'npm/packages/ruvllm/src/sona.ts',
    fns: [{ name: 'processInstantLearning', required: true }],
  },
  {
    file: 'npm/packages/ruvector/src/core/intelligence-engine.ts',
    fns: [{ name: 'recordRouteOutcome', required: false, absentNote: '#517 fix not on main' }],
  },
];

/** Statements that are pure logging — removed before the mutation check. */
const LOGGING_TOKENS = [
  /console\.\w+\s*\(/y,
  /web_sys::console::\w+\s*\(/y,
  /log::\w+!\s*\(/y,
  /tracing::\w+!\s*\(/y,
  /println!\s*\(/y,
  /eprintln!\s*\(/y,
  /dbg!\s*\(/y,
];

/**
 * Char-scan that understands line/block comments and string literals
 * ("..." '...' `...`), so braces inside comments/strings don't break
 * matching. Returns source with comments replaced by spaces.
 */
function stripComments(src) {
  let out = '';
  let i = 0;
  const n = src.length;
  while (i < n) {
    const c = src[i];
    const c2 = src[i + 1];
    if (c === '/' && c2 === '/') {
      while (i < n && src[i] !== '\n') { out += ' '; i++; }
    } else if (c === '/' && c2 === '*') {
      let depth = 1; // Rust block comments nest
      out += '  '; i += 2;
      while (i < n && depth > 0) {
        if (src[i] === '/' && src[i + 1] === '*') { depth++; out += '  '; i += 2; continue; }
        if (src[i] === '*' && src[i + 1] === '/') { depth--; out += '  '; i += 2; continue; }
        out += src[i] === '\n' ? '\n' : ' ';
        i++;
      }
    } else if (c === '"' || c === "'" || c === '`') {
      // Keep string contents verbatim except we neutralize braces so brace
      // matching stays balanced. Rust lifetimes ('a) are handled by bailing
      // out of single-quote mode when no closing quote appears nearby.
      const quote = c;
      let j = i + 1;
      let closed = -1;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === quote) { closed = j; break; }
        if (quote === "'" && j - i > 2 && !src.slice(i + 1, j + 1).includes("'")) break; // lifetime, not char
        j++;
      }
      if (quote === "'" && (closed === -1 || closed - i > 4)) {
        out += c; i++; // treat as lifetime tick, not a string
      } else {
        const end = closed === -1 ? n - 1 : closed;
        out += src.slice(i, end + 1).replace(/[{}]/g, '.');
        i = end + 1;
      }
    } else {
      out += c;
      i++;
    }
  }
  return out;
}

/** Extract `{...}` body of the named function from comment-stripped source. */
function extractBody(stripped, fnName) {
  // Anchored to DEFINITIONS only (a call site like `this.fn(x)` must not
  // match): Rust `fn name(`, or a TS class method at the start of a line
  // (optionally private/public/protected/async).
  let re = new RegExp(`\\bfn\\s+${fnName}\\s*\\(`, 'g');
  let m = re.exec(stripped);
  if (!m) {
    re = new RegExp(`^\\s*(?:private\\s+|public\\s+|protected\\s+)?(?:async\\s+)?${fnName}\\s*\\(`, 'gm');
    m = re.exec(stripped);
  }
  if (!m) return null;
  const open = stripped.indexOf('{', re.lastIndex);
  if (open === -1) return null;
  let depth = 0;
  for (let i = open; i < stripped.length; i++) {
    if (stripped[i] === '{') depth++;
    else if (stripped[i] === '}') {
      depth--;
      if (depth === 0) return stripped.slice(open + 1, i);
    }
  }
  return null;
}

/** Remove a logging call starting at `start` through its closing paren + `;`. */
function removeLoggingCalls(body) {
  let out = body;
  for (const token of LOGGING_TOKENS) {
    const re = new RegExp(token.source, 'g');
    let m;
    while ((m = re.exec(out)) !== null) {
      // walk to matching close paren
      let depth = 0;
      let end = -1;
      for (let i = m.index; i < out.length; i++) {
        if (out[i] === '(') depth++;
        else if (out[i] === ')') {
          depth--;
          if (depth === 0) { end = i; break; }
        }
      }
      if (end === -1) break;
      while (end + 1 < out.length && /[\s;?]/.test(out[end + 1])) end++;
      out = out.slice(0, m.index) + out.slice(end + 1);
      re.lastIndex = 0;
    }
  }
  return out;
}

/** True if the cleaned body contains at least one state-mutating statement. */
function hasMutation(cleaned) {
  // assignment (not ==, =>, <=, >=, !=), compound assignment, inc/dec
  if (/(\+=|-=|\*=|\/=|\+\+|--)/.test(cleaned)) return true;
  if (/[^=!<>+\-*/]=[^=>]/.test(cleaned)) return true;
  // any remaining (non-logging) call expression
  if (/[A-Za-z_][\w:.]*\s*\(/.test(cleaned)) return true;
  return false;
}

export function scanFile(absPath, fnNames) {
  const findings = [];
  if (!existsSync(absPath)) {
    return fnNames.map((f) => ({ fn: f.name, status: 'file-missing', ok: false }));
  }
  const stripped = stripComments(readFileSync(absPath, 'utf8'));
  for (const f of fnNames) {
    const body = extractBody(stripped, f.name);
    if (body === null) {
      findings.push({
        fn: f.name,
        status: f.required ? 'missing' : `absent (ok: ${f.absentNote ?? 'optional seam'})`,
        ok: !f.required,
      });
      continue;
    }
    const cleaned = removeLoggingCalls(body).replace(/\breturn\b\s*(Ok\(\(\)\)|;)?/g, ' ');
    const ok = hasMutation(cleaned);
    findings.push({ fn: f.name, status: ok ? 'ok' : 'EFFECTIVELY EMPTY (no-op stub)', ok });
  }
  return findings;
}

const isMain = process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) {
  let failed = 0;
  for (const target of TARGETS) {
    const abs = path.join(REPO, target.file);
    for (const finding of scanFile(abs, target.fns)) {
      const tag = finding.ok ? 'OK  ' : 'FAIL';
      console.log(`${tag} ${target.file} :: ${finding.fn} — ${finding.status}`);
      if (!finding.ok) failed++;
    }
  }
  if (failed > 0) {
    console.log(`TRIPWIRE FIRED: ${failed} seam(s) look like no-op stubs or are missing.`);
    console.log('A learn-from-feedback seam must mutate state — see scripts/sona-drift/README.md.');
    process.exit(1);
  }
  console.log('TRIPWIRE PASS: all learn-from-feedback seams contain real work.');
}
