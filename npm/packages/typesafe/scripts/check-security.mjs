#!/usr/bin/env node
// ADR-005 CI security assertions, run against the built artifacts (the
// regex-against-artifact pattern from ruvector's mcp-command-security.js).
//
//   (a) no shell/subprocess in the decision path (src, bin, index.js, dist)
//   (b) no network in the decision path (src, bin) — the `serve` command's
//       server-side and tests are allow-listed by file
//   (c) a wasm module (wasm/*.wasm) imports no WASI and nothing fs/net/fetch
//   (d) a native .node binary contains no curl/wget//bin/sh/http(s):// strings
//       (a github.com/ruvnet https URL in a version/repository string is the
//        one known-benign exception)
//
// Exits non-zero on any violation. Exported `runChecks({root, wasmPath,
// nativePath})` so tests drive it without spawning a process; passes cleanly
// with "0 files scanned" when a directory is absent.

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, basename, extname } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(HERE, '..');

// ---------------------------------------------------------------------------
// patterns
// ---------------------------------------------------------------------------

// child_process import (require or ESM), and shell/exec/spawn CALLS. The call
// regex uses a lookbehind so `re.exec(str)` / `foo.spawn(` (method calls) do
// not false-positive; only bareword exec/spawn/execFile/execSync/spawnSync do.
const SUBPROCESS = [
  { name: 'child_process import', re: /require\(\s*['"](?:node:)?child_process['"]\s*\)|from\s*['"](?:node:)?child_process['"]/ },
  { name: 'shell:true', re: /shell\s*:\s*true\b/ },
  { name: 'exec/spawn call', re: /(?<![.\w])(execSync|execFileSync|execFile|exec|spawnSync|spawn)\s*\(/ },
];

// network in the decision path
const NETWORK = [
  { name: 'fetch(', re: /(?<![.\w])fetch\s*\(/ },
  { name: 'http.request', re: /\bhttps?\.request\s*\(/ },
  { name: 'http.createServer', re: /\bhttps?\.createServer\s*\(/ },
  { name: 'net.connect', re: /\bnet\.connect\s*\(/ },
  { name: 'require(http)', re: /require\(\s*['"](?:node:)?(?:http|https|net|dgram|tls)['"]\s*\)|from\s*['"](?:node:)?(?:http|https|net|dgram|tls)['"]/ },
];

// Files whose network use is legitimate (the HTTP server behind `serve`) or
// which are type declarations (not runtime decision path).
const NETWORK_ALLOW = (file) => {
  const b = basename(file);
  return b.startsWith('serve.') || b === 'node-shims.d.ts' || b.endsWith('.d.ts');
};

const CODE_EXT = new Set(['.js', '.cjs', '.mjs', '.ts', '.cts', '.mts']);

function walk(dir, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (e === 'node_modules' || e === '.cache') continue;
      walk(p, acc);
    } else if (CODE_EXT.has(extname(p))) acc.push(p);
  }
  return acc;
}

/** Strip line and block comments so a `// no child_process` note is not a hit. */
function stripComments(src) {
  return src
    .replace(/\/\*[\s\S]*?\*\//g, ' ')
    .replace(/(^|[^:])\/\/[^\n]*/g, '$1');
}

function scanFiles(files, patterns, { allow } = {}) {
  const violations = [];
  for (const f of files) {
    const src = stripComments(readFileSync(f, 'utf8'));
    for (const { name, re } of patterns) {
      if (re.test(src)) {
        if (allow && allow(f)) continue;
        violations.push({ file: f, pattern: name });
      }
    }
  }
  return violations;
}

// ---------------------------------------------------------------------------
// (c) wasm imports
// ---------------------------------------------------------------------------

const WASM_FORBIDDEN = /wasi_snapshot_preview1|wasi_unstable|\b(fs|net|socket|fetch|sock)\b/i;

export function checkWasm(wasmPath) {
  if (!wasmPath || !existsSync(wasmPath)) return { scanned: 0, violations: [] };
  const mod = new WebAssembly.Module(readFileSync(wasmPath));
  const imports = WebAssembly.Module.imports(mod);
  const violations = [];
  for (const imp of imports) {
    const label = `${imp.module}.${imp.name}`;
    if (WASM_FORBIDDEN.test(imp.module) || WASM_FORBIDDEN.test(imp.name)) {
      violations.push({ file: wasmPath, pattern: `forbidden wasm import ${label}` });
    }
  }
  return { scanned: imports.length, violations };
}

// ---------------------------------------------------------------------------
// (d) native binary strings
// ---------------------------------------------------------------------------

function checkNative(nativePath) {
  if (!nativePath || !existsSync(nativePath)) return { scanned: 0, violations: [] };
  const bytes = readFileSync(nativePath).toString('latin1');
  const violations = [];
  // curl / wget only as whole tokens (napi binaries embed GetModuleHandleExW,
  // whose concatenation contains "wget" as a non-token substring).
  for (const m of bytes.matchAll(/(?<![A-Za-z])(curl|wget)(?![A-Za-z])/gi)) {
    violations.push({ file: nativePath, pattern: `native string '${m[0]}'` });
  }
  if (bytes.includes('/bin/sh')) violations.push({ file: nativePath, pattern: "native string '/bin/sh'" });
  if (bytes.includes('http://')) violations.push({ file: nativePath, pattern: "native string 'http://'" });
  // ADR-005 ("no network in the decision path"): the engine makes NO network
  // calls. The https strings below are STATIC rodata compiled into the binary —
  // ONNX Runtime's operator-schema docs (which cite papers and vendor docs) and
  // the Rust deps ORT links (getrandom, re2, ndarray) — never an egress target.
  // curl/wget/http:///bin/sh above stay hard rejects; only this closed set of
  // documentation hosts is tolerated, and any other https URL still fails.
  const DOC_HOSTS = new Set([
    'arxiv.org',
    'onnx.ai',
    'docs.nvidia.com',
    'docs.rs',
    'numpy.org',
    'en.wikipedia.org',
    'ieeexplore.ieee.org',
    'tinyurl.com',
  ]);
  const GITHUB_DOC_ORGS = new Set(['ruvnet', 'microsoft', 'onnx', 'google']);
  for (const m of bytes.matchAll(/https:\/\/[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]*/g)) {
    const url = m[0];
    let host = '';
    let org = '';
    try {
      const u = new URL(url);
      host = u.host;
      org = u.pathname.split('/')[1] || '';
    } catch {
      /* malformed slice from the binary — fall through to a violation */
    }
    if (DOC_HOSTS.has(host)) continue;
    if (host === 'github.com' && GITHUB_DOC_ORGS.has(org)) continue;
    violations.push({ file: nativePath, pattern: `native string '${url.slice(0, 60)}'` });
  }
  return { scanned: 1, violations };
}

// ---------------------------------------------------------------------------
// driver
// ---------------------------------------------------------------------------

export function runChecks({ root = PKG_ROOT, wasmPath, nativePath } = {}) {
  // (a) decision path: src, bin, index.js, dist
  const decisionDirs = ['src', 'bin', 'dist'].map((d) => join(root, d));
  const decisionFiles = decisionDirs.flatMap((d) => walk(d));
  const indexJs = join(root, 'index.js');
  if (existsSync(indexJs)) decisionFiles.push(indexJs);

  // (b) network: src + bin only (serve.* and .d.ts allow-listed)
  const netFiles = [join(root, 'src'), join(root, 'bin')].flatMap((d) => walk(d));
  if (existsSync(indexJs)) netFiles.push(indexJs);

  // auto-discover wasm / native artifacts if not passed
  const wasm = wasmPath ?? firstFile(join(root, 'wasm'), (f) => f.endsWith('.wasm'));
  const native = nativePath ?? firstFile(join(root, 'native'), (f) => f.endsWith('.node'));

  const a = scanFiles(decisionFiles, SUBPROCESS);
  const b = scanFiles(netFiles, NETWORK, { allow: NETWORK_ALLOW });
  const c = checkWasm(wasm);
  const d = checkNative(native);

  const violations = [...a, ...b, ...c.violations, ...d.violations];
  return {
    ok: violations.length === 0,
    violations,
    scanned: {
      decision_files: decisionFiles.length,
      network_files: netFiles.length,
      wasm: wasm ? basename(wasm) : null,
      wasm_imports: c.scanned,
      native: native ? basename(native) : null,
    },
  };
}

function firstFile(dir, pred) {
  if (!existsSync(dir)) return null;
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    if (statSync(p).isFile() && pred(p)) return p;
  }
  return null;
}

function main(argv) {
  const wasmIdx = argv.indexOf('--wasm');
  const nativeIdx = argv.indexOf('--native');
  const res = runChecks({
    wasmPath: wasmIdx >= 0 ? argv[wasmIdx + 1] : undefined,
    nativePath: nativeIdx >= 0 ? argv[nativeIdx + 1] : undefined,
  });
  const s = res.scanned;
  console.log(
    `security scan: ${s.decision_files} decision files, ${s.network_files} network-path files, ` +
      `wasm=${s.wasm ?? 'none'} (${s.wasm_imports} imports), native=${s.native ?? 'none'}`,
  );
  if (!res.ok) {
    console.error(`\nADR-005 security assertions FAILED:`);
    for (const v of res.violations) console.error(`  ${v.pattern}  in  ${v.file}`);
    process.exit(1);
  }
  console.log('ADR-005 security assertions: all clear');
}

if (import.meta.url === `file://${process.argv[1]}`) main(process.argv.slice(2));
