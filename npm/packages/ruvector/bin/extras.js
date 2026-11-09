'use strict';
/**
 * Optional @ruvector extras — the allowlist behind `ruvector add <extra>`.
 *
 * Each extra is a package that ruvector already integrates with through a lazy
 * require/import: nothing here is installed unless the user asks for it.
 *
 * Security: names are resolved only against this frozen, prototype-free table,
 * and installs are spawned with an argv array and no shell, so a user-supplied
 * name can never become a package spec, a flag, or shell syntax.
 */

const fs = require('fs');
const path = require('path');

const EXTRAS = Object.freeze(Object.assign(Object.create(null), {
  rabitq: {
    pkg: '@ruvector/rabitq-wasm',
    spec: '@ruvector/rabitq-wasm@^0.1.1',
    why: 'RaBitQ 1-bit quantized index: ~32x smaller vector codes with exact-L2 rerank, for memory-bound search',
    usage: "const { RabitqIndex } = require('ruvector'); RabitqIndex.build(ids, vectors, { dim })",
  },
  diskann: {
    pkg: '@ruvector/diskann',
    spec: '@ruvector/diskann@^0.1.2',
    why: 'DiskANN/Vamana graph index with PQ, saved to disk, for indexes larger than RAM',
    usage: 'CLI: ruvector diskann build|search; MCP: diskann_build, diskann_search; library: DiskAnnIndex',
  },
  router: {
    pkg: '@ruvector/router',
    spec: '@ruvector/router@^0.1.32',
    why: 'Native HNSW semantic intent router behind `ruvector route` and SemanticRouter',
    usage: 'npx ruvector route --help',
  },
  graph: {
    pkg: '@ruvector/graph-node',
    spec: '@ruvector/graph-node@^2.1.1',
    why: 'Native hypergraph database with Cypher queries, used by the CodeGraph API',
    usage: "const { CodeGraph } = require('ruvector')",
  },
  rvlite: {
    pkg: 'rvlite',
    spec: 'rvlite@^0.2.7',
    why: 'SQL / Cypher / SPARQL over a vector store, used by the rvlite_* MCP tools',
    usage: 'MCP tools: rvlite_sql, rvlite_cypher, rvlite_sparql',
  },
  rvf: {
    pkg: '@ruvector/rvf',
    spec: '@ruvector/rvf@^0.4.2',
    why: 'Persistent RVF vector stores behind `ruvector rvf` and the rvf_* MCP tools (optional dependency)',
    usage: 'npx ruvector rvf --help',
  },
  'tiny-dancer': {
    pkg: '@ruvector/tiny-dancer',
    spec: '@ruvector/tiny-dancer@^0.1.22',
    why: 'Cost-optimal FastGRNN model router behind `ruvector tiny-dancer` (optional dependency)',
    usage: 'npx ruvector tiny-dancer info',
  },
  quantum: {
    pkg: '@ruvector/ruqu-wasm',
    spec: '@ruvector/ruqu-wasm@^2.1.0',
    why: 'ruqu state-vector quantum circuit simulator (Rust/WASM) behind `ruvector quantum` and the quantum_* MCP tools (optional dependency)',
    usage: 'npx ruvector quantum simulate bell',
  },
  sona: {
    pkg: '@ruvector/sona',
    spec: '@ruvector/sona@^0.1.4',
    why: 'SONA adaptive learning (LoRA + EWC++) — already a ruvector dependency',
    usage: 'npx ruvector sona info',
    bundled: true,
  },
}));

const PACKAGE_MANAGERS = Object.freeze(['npm', 'pnpm', 'yarn', 'bun']);

// Ordered: the first lockfile found wins.
const LOCKFILES = Object.freeze([
  ['pnpm-lock.yaml', 'pnpm'],
  ['yarn.lock', 'yarn'],
  ['bun.lockb', 'bun'],
  ['bun.lock', 'bun'],
  ['package-lock.json', 'npm'],
  ['npm-shrinkwrap.json', 'npm'],
]);

function toEntry(key) {
  return { key, ...EXTRAS[key] };
}

function listExtras() {
  return Object.keys(EXTRAS).map(toEntry);
}

/** Resolve a user-supplied name to an allowlisted extra, or null. */
function resolveExtra(name) {
  if (typeof name !== 'string') return null;
  const wanted = name.trim().toLowerCase();
  if (!wanted) return null;
  if (Object.prototype.hasOwnProperty.call(EXTRAS, wanted)) return toEntry(wanted);
  const byPkg = Object.keys(EXTRAS).find((key) => EXTRAS[key].pkg === wanted);
  return byPkg ? toEntry(byPkg) : null;
}

function detectPackageManager(cwd) {
  for (const [file, pm] of LOCKFILES) {
    if (fs.existsSync(path.join(cwd, file))) return pm;
  }
  try {
    const manifest = JSON.parse(fs.readFileSync(path.join(cwd, 'package.json'), 'utf8'));
    const m = /^(npm|pnpm|yarn|bun)@/.exec(String(manifest.packageManager || ''));
    if (m) return m[1];
  } catch {
    // no package.json or unreadable — default below
  }
  return 'npm';
}

function buildInstallCommand(pm, specs, platform = process.platform) {
  if (!PACKAGE_MANAGERS.includes(pm)) {
    throw new Error(`Unsupported package manager: ${pm}`);
  }
  // Same convention as NPX_COMMAND in mcp-server.js: name the .cmd shim on
  // Windows instead of falling back to a shell.
  const cmd = platform === 'win32' && pm !== 'bun' ? `${pm}.cmd` : pm;
  const verb = pm === 'npm' ? 'install' : 'add';
  return { cmd, args: [verb, ...specs] };
}

/** True when `pkg` resolves from the project (cwd) or from ruvector itself. */
function isPackageInstalled(pkg, cwd = process.cwd()) {
  const paths = [cwd, __dirname];
  try {
    require.resolve(`${pkg}/package.json`, { paths });
    return true;
  } catch {
    try {
      require.resolve(pkg, { paths });
      return true;
    } catch {
      return false;
    }
  }
}

function extraHint(key) {
  const extra = EXTRAS[key];
  if (!extra) return '';
  return `Install with: npx ruvector add ${key}  (or: npm install ${extra.pkg})`;
}

/**
 * Plan and (unless dryRun) run the install for the named extras.
 * `spawn` and `isInstalled` are injectable for tests.
 */
function runAdd(names, options = {}) {
  const cwd = options.cwd || process.cwd();
  const isInstalled = options.isInstalled || ((pkg) => isPackageInstalled(pkg, cwd));
  const spawn = options.spawn || require('child_process').spawnSync;
  const platform = options.platform || process.platform;

  const unknown = [];
  const selected = [];
  for (const name of names) {
    const extra = resolveExtra(name);
    if (!extra) unknown.push(name);
    else if (!selected.some((s) => s.key === extra.key)) selected.push(extra);
  }
  if (unknown.length) return { ok: false, unknown, toInstall: [], skipped: [] };

  const skipped = [];
  const toInstall = [];
  for (const extra of selected) {
    if (extra.bundled) skipped.push({ key: extra.key, reason: 'bundled with ruvector' });
    else if (isInstalled(extra.pkg)) skipped.push({ key: extra.key, reason: 'already installed' });
    else toInstall.push(extra);
  }

  const pm = detectPackageManager(cwd);
  const command = toInstall.length
    ? buildInstallCommand(pm, toInstall.map((e) => e.spec), platform)
    : null;
  const result = { ok: true, pm, cwd, command, toInstall, skipped, dryRun: !!options.dryRun };
  if (!command || options.dryRun) return result;

  const proc = spawn(command.cmd, command.args, { cwd, stdio: options.stdio || 'inherit', shell: false });
  result.status = proc.status;
  result.ok = !proc.error && proc.status === 0;
  if (proc.error) result.error = proc.error.message;
  return result;
}

module.exports = {
  EXTRAS,
  listExtras,
  resolveExtra,
  detectPackageManager,
  buildInstallCommand,
  isPackageInstalled,
  extraHint,
  runAdd,
};
