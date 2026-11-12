#!/usr/bin/env node
// ADR-005 CI security assertions, run against the source + built artifacts.
//
//   (a) no shell/subprocess anywhere in the package's own JS (src, bin, index.js,
//       dist, bench, scripts) — the scorer AND the harness must not shell out
//   (b) no network in the SCORING path (src, bin, index.js) or the harness
//       (bench, scripts) EXCEPT the dataset fetchers under bench/datasets/, which
//       download canonical datasets at bench time by design (allow-listed)
//   (c) a wasm module (wasm/*.wasm) imports no WASI and nothing fs/net/fetch
//   (d) a native .node binary contains no curl/wget//bin/sh/http(s):// strings
//       (a github.com/ruvnet https URL in a version/repository string is benign)
//
// Exits non-zero on any violation. Exported `runChecks({root, wasmPath,
// nativePath})` so tests drive it without spawning; passes cleanly with
// "0 files scanned" when a directory is absent.

import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, basename, extname } from 'node:path';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(HERE, '..');

const SUBPROCESS = [
  { name: 'child_process import', re: /require\(\s*['"](?:node:)?child_process['"]\s*\)|from\s*['"](?:node:)?child_process['"]/ },
  { name: 'shell:true', re: /shell\s*:\s*true\b/ },
  { name: 'exec/spawn call', re: /(?<![.\w])(execSync|execFileSync|execFile|exec|spawnSync|spawn)\s*\(/ },
];

const NETWORK = [
  { name: 'fetch(', re: /(?<![.\w])fetch\s*\(/ },
  { name: 'http.request', re: /\bhttps?\.request\s*\(/ },
  { name: 'http.createServer', re: /\bhttps?\.createServer\s*\(/ },
  { name: 'net.connect', re: /\bnet\.connect\s*\(/ },
  { name: 'require(http)', re: /require\(\s*['"](?:node:)?(?:http|https|net|dgram|tls)['"]\s*\)|from\s*['"](?:node:)?(?:http|https|net|dgram|tls)['"]/ },
];

// Legitimate network: the dataset fetchers (bench time only, never shipped) and
// type declarations. `serve.*` reserved for a future server-side entry.
const NETWORK_ALLOW = (file) => {
  const b = basename(file);
  return file.split(/[\\/]/).includes('datasets') || b.startsWith('serve.') || b.endsWith('.d.ts');
};

const CODE_EXT = new Set(['.js', '.cjs', '.mjs', '.ts', '.cts', '.mts']);

function walk(dir, acc = []) {
  if (!existsSync(dir)) return acc;
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    const st = statSync(p);
    if (st.isDirectory()) {
      if (e === 'node_modules' || e === '.cache' || e === 'results') continue;
      walk(p, acc);
    } else if (CODE_EXT.has(extname(p))) acc.push(p);
  }
  return acc;
}

/** Strip line and block comments so a `// no child_process` note is not a hit. */
function stripComments(src) {
  return src.replace(/\/\*[\s\S]*?\*\//g, ' ').replace(/(^|[^:])\/\/[^\n]*/g, '$1');
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

function checkNative(nativePath) {
  if (!nativePath || !existsSync(nativePath)) return { scanned: 0, violations: [] };
  const bytes = readFileSync(nativePath).toString('latin1');
  const violations = [];
  for (const m of bytes.matchAll(/(?<![A-Za-z])(curl|wget)(?![A-Za-z])/gi)) {
    violations.push({ file: nativePath, pattern: `native string '${m[0]}'` });
  }
  if (bytes.includes('/bin/sh')) violations.push({ file: nativePath, pattern: "native string '/bin/sh'" });
  if (bytes.includes('http://')) violations.push({ file: nativePath, pattern: "native string 'http://'" });
  for (const m of bytes.matchAll(/https:\/\/[A-Za-z0-9._~:/?#@!$&'()*+,;=%-]*/g)) {
    if (/^https:\/\/github\.com\/ruvnet(\/|$)/.test(m[0])) continue;
    violations.push({ file: nativePath, pattern: `native string '${m[0].slice(0, 60)}'` });
  }
  return { scanned: 1, violations };
}

export function runChecks({ root = PKG_ROOT, wasmPath, nativePath } = {}) {
  // (a) subprocess: the whole package's own JS — scorer AND harness
  const jsDirs = ['src', 'bin', 'dist', 'bench', 'scripts'].map((d) => join(root, d));
  const jsFiles = jsDirs.flatMap((d) => walk(d));
  const indexJs = join(root, 'index.js');
  if (existsSync(indexJs)) jsFiles.push(indexJs);

  // (b) network: same set, minus the allow-listed dataset fetchers
  const a = scanFiles(jsFiles, SUBPROCESS);
  const b = scanFiles(jsFiles, NETWORK, { allow: NETWORK_ALLOW });
  const wasm = wasmPath ?? firstFile(join(root, 'wasm'), (f) => f.endsWith('.wasm'));
  const native = nativePath ?? firstFile(join(root, 'native'), (f) => f.endsWith('.node'));
  const c = checkWasm(wasm);
  const d = checkNative(native);

  const violations = [...a, ...b, ...c.violations, ...d.violations];
  return {
    ok: violations.length === 0,
    violations,
    scanned: {
      js_files: jsFiles.length,
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
  console.log(`security scan: ${s.js_files} JS files, wasm=${s.wasm ?? 'none'} (${s.wasm_imports} imports), native=${s.native ?? 'none'}`);
  if (!res.ok) {
    console.error('\nADR-005 security assertions FAILED:');
    for (const v of res.violations) console.error(`  ${v.pattern}  in  ${v.file}`);
    process.exit(1);
  }
  console.log('ADR-005 security assertions: all clear');
}

if (import.meta.url === `file://${process.argv[1]}`) main(process.argv.slice(2));
