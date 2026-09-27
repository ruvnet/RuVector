// Receipt assembly (ADR-006 §Where it lives: "one signed receipt per run";
// ADR-005: triples' text is never logged — receipts store hashes and counts,
// never entity/relation ids or descriptions). A receipt records WHAT was
// measured and against WHAT: scorer, config, dataset/split hashes, filtered
// metrics per split, the tie-break mode, ANN recall, the adversarial report,
// predict latency, train throughput, host info and the binding's statsJson.
//
// Provenance (ADR-007 M0): every receipt carries the git SHA of HEAD, a
// sha256 over the engine crates' source files (so uncommitted edits are
// visible, not just the commit), a sha256 of the canonical config, a sha256
// over the dataset file hashes, the sha256 of the loaded binding module and of
// every native/wasm binary next to it, and finally `receipt_sha256` over the
// whole body — so a published number can be tied to the exact code, inputs and
// binary. No absolute host paths are recorded. Git is read from the .git files
// directly: the package's ADR-005 policy forbids child_process.

import os from 'node:os';
import { createHash } from 'node:crypto';
import { readFileSync, readdirSync, existsSync, statSync } from 'node:fs';
import { join, relative, isAbsolute, basename, dirname, resolve, sep } from 'node:path';

const sha256hex = (s) => createHash('sha256').update(typeof s === 'string' || Buffer.isBuffer(s) ? s : String(s)).digest('hex');

/** Canonical JSON: object keys sorted recursively, so a hash is order-independent. */
export function canonicalJson(v) {
  if (Array.isArray(v)) return `[${v.map(canonicalJson).join(',')}]`;
  if (v && typeof v === 'object') {
    return `{${Object.keys(v)
      .filter((k) => v[k] !== undefined)
      .sort()
      .map((k) => `${JSON.stringify(k)}:${canonicalJson(v[k])}`)
      .join(',')}}`;
  }
  return JSON.stringify(v ?? null);
}

/** Host + runtime provenance for reproducibility (no secrets, no paths). */
export function hostInfo() {
  return {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus()?.length ?? null,
    cpu_model: os.cpus()?.[0]?.model ?? null,
    os_release: os.release(),
    total_mem_mb: Math.round(os.totalmem() / 1024 / 1024),
  };
}

/** Repo-relative directories whose source defines the measured engine. */
const ENGINE_SRC = ['crates/ruvector-kge', 'crates/ruvector-kge-ffi', 'crates/ruvector-kge-wasm'];

/** Walk up from `dir` to the directory holding `.git` (a dir, or a worktree's file). */
function findRepoRoot(dir) {
  let d = resolve(dir);
  for (;;) {
    if (existsSync(join(d, '.git'))) return d;
    const up = dirname(d);
    if (up === d) return null;
    d = up;
  }
}

/**
 * HEAD commit SHA read from the .git files (handles worktrees' `gitdir:`
 * indirection, loose refs and packed-refs). Null when unresolvable — never throws.
 */
export function gitHeadSha(cwd) {
  try {
    const root = findRepoRoot(cwd);
    if (!root) return null;
    let gitDir = join(root, '.git');
    if (statSync(gitDir).isFile()) {
      const m = /^gitdir:\s*(.+)$/m.exec(readFileSync(gitDir, 'utf8'));
      if (!m) return null;
      gitDir = resolve(root, m[1].trim());
    }
    const common = existsSync(join(gitDir, 'commondir'))
      ? resolve(gitDir, readFileSync(join(gitDir, 'commondir'), 'utf8').trim())
      : gitDir;
    const head = readFileSync(join(gitDir, 'HEAD'), 'utf8').trim();
    if (/^[0-9a-f]{40}$/.test(head)) return head;
    const ref = /^ref:\s*(.+)$/.exec(head)?.[1];
    if (!ref) return null;
    for (const base of [gitDir, common]) {
      const p = join(base, ref);
      if (existsSync(p)) return readFileSync(p, 'utf8').trim();
    }
    const packed = join(common, 'packed-refs');
    if (existsSync(packed)) {
      for (const line of readFileSync(packed, 'utf8').split('\n')) {
        const [sha, name] = line.trim().split(' ');
        if (name === ref && /^[0-9a-f]{40}$/.test(sha)) return sha;
      }
    }
    return null;
  } catch {
    return null;
  }
}

/**
 * sha256 over (repo-relative path, file sha256) of every file under the engine
 * crates' `src/` plus their Cargo.toml, sorted — changes with any uncommitted
 * engine edit, unlike the commit SHA. Null when the crates are not present.
 */
export function engineSourceSha(cwd) {
  const root = findRepoRoot(cwd);
  if (!root) return null;
  const entries = [];
  const walk = (dir) => {
    for (const f of readdirSync(dir, { withFileTypes: true })) {
      const p = join(dir, f.name);
      if (f.isDirectory()) walk(p);
      else if (f.isFile()) entries.push([relative(root, p).split(sep).join('/'), fileSha(p)]);
    }
  };
  for (const c of ENGINE_SRC) {
    const src = join(root, c, 'src');
    if (existsSync(src)) walk(src);
    const toml = join(root, c, 'Cargo.toml');
    if (existsSync(toml)) entries.push([`${c}/Cargo.toml`, fileSha(toml)]);
  }
  if (!entries.length) return null;
  entries.sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
  return sha256hex(entries.map(([p, h]) => `${p} ${h}`).join('\n'));
}

const fileSha = (p) => {
  try {
    return statSync(p).isFile() ? sha256hex(readFileSync(p)) : null;
  } catch {
    return null;
  }
};

/** `{ basename: sha256 }` for every `*.node` in `pkgRoot` and `*.wasm` in `pkgRoot/wasm`. */
export function nativeBinaries(pkgRoot) {
  const out = {};
  const scan = (dir, ext, prefix) => {
    if (!existsSync(dir)) return;
    for (const f of readdirSync(dir).sort()) {
      if (f.endsWith(ext)) {
        const h = fileSha(join(dir, f));
        if (h) out[prefix + f] = h;
      }
    }
  };
  scan(pkgRoot, '.node', '');
  scan(join(pkgRoot, 'wasm'), '.wasm', 'wasm/');
  return out;
}

/**
 * Receipt-safe binding source: `injected`, `env:KGE_BENCH_BINDING(<rel>)` or a
 * package-relative path. An absolute path outside the package is reduced to
 * its basename, so no host directory layout leaks into a published receipt.
 */
export function sanitizeBindingSource(source, pkgRoot) {
  if (typeof source !== 'string') return source ?? null;
  const rel = (p) => {
    if (!isAbsolute(p)) return p;
    const r = relative(pkgRoot, p);
    return r && !r.startsWith('..') && !isAbsolute(r) ? r.split(sep).join('/') : `<external>/${basename(p)}`;
  };
  const env = /^env:KGE_BENCH_BINDING\((.*)\)$/.exec(source);
  if (env) return `env:KGE_BENCH_BINDING(${rel(env[1])})`;
  return rel(source);
}

/**
 * Provenance block. `bindingPath` is the absolute path of the loaded binding
 * module (hashed, never recorded). `git` = { sha, engineSrc } may be injected (tests).
 */
export function buildProvenance({ pkgRoot, config, datasetHashes, bindingPath, git }) {
  const g = git ?? { sha: gitHeadSha(pkgRoot), engineSrc: engineSourceSha(pkgRoot) };
  return {
    git_sha: g.sha ?? null,
    engine_src_sha256: g.engineSrc ?? null,
    config_sha256: sha256hex(canonicalJson(config ?? null)),
    dataset_sha256: datasetHashes ? sha256hex(canonicalJson(datasetHashes)) : null,
    binding_file_sha256: bindingPath ? fileSha(bindingPath) : null,
    native_binaries: pkgRoot ? nativeBinaries(pkgRoot) : {},
  };
}

/** sha256 over the canonical receipt body, excluding `receipt_sha256` itself. */
export function receiptDigest(receipt) {
  const { receipt_sha256: _omit, ...body } = receipt;
  return sha256hex(canonicalJson(body));
}

/** True when `receipt.receipt_sha256` matches its body (tamper check). */
export function verifyReceipt(receipt) {
  return typeof receipt?.receipt_sha256 === 'string' && receipt.receipt_sha256 === receiptDigest(receipt);
}

/**
 * Assemble the receipt. `metrics` is { split: { mrr, mr, hits, perSide } } for
 * whichever splits were scored. Optional blocks (ann, adversarial, tieCheck,
 * hardNegatives, training) are included only when present. No labels ever.
 * The returned receipt is sealed with `receipt_sha256`.
 */
export function buildReceipt({
  suite,
  scorer,
  config,
  tieBreak,
  datasetHashes,
  splitsHash,
  counts,
  metrics,
  ann,
  adversarial,
  hardNegatives,
  tieCheck,
  latency,
  training,
  gates,
  binding,
  bindingSource,
  stats,
  provenance,
  extra,
}) {
  const receipt = {
    schema: 'ruvector-kge-bench/receipt@1',
    generated_at: new Date().toISOString(),
    suite,
    scorer: scorer ?? null,
    config: config ?? null, // {scorer, dims, seed, epochs, ...} — no data, only hyperparams
    tie_break: tieBreak ?? null,
    binding: binding ?? null, // { version, backend } or { unavailable, error }
    binding_source: bindingSource ?? null, // injected | package-relative path | env:KGE_BENCH_BINDING(...)
    provenance: provenance ?? null, // { git_sha, engine_src_sha256, config_sha256, dataset_sha256, binding_file_sha256, native_binaries }
    dataset: {
      hashes: datasetHashes ?? null, // { file: sha256 } — the frozen inputs, no data
      splits_hash: splitsHash ?? null,
      counts: counts ?? null,
    },
    metrics: metrics ?? {}, // { split: { mrr, mr, hits:{1,3,10}, perSide } }
    ann: ann ?? null, // { recall_at_10, queries, withIndex/withoutIndex latency }
    adversarial: adversarial ?? null, // { targets, decoys, mean_before, mean_after, drop, targets_hash }
    hard_negatives: hardNegatives ?? null,
    tie_check: tieCheck ?? null, // { available, method, topMr, randomMr, bottomMr, nEntities }
    latency_ms: latency ?? null, // predict p50/p95
    training: training ?? null, // { epochs, triple_epochs_per_sec, wall_ms }
    gates: gates ?? null,
    stats_json: stats ?? null,
    host: hostInfo(),
    ...(extra ? { extra } : {}),
  };
  receipt.receipt_sha256 = receiptDigest(receipt);
  return receipt;
}

export { sha256hex };
