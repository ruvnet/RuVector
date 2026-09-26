#!/usr/bin/env node
// Fetch the ONNX embedder weights + tokenizers for @ruvector/typesafe and
// verify them against models/manifest.json (ADR-002 §6, ADR-005 supply chain).
//
// No dependencies (Node >= 20 built-in fetch + node:crypto). Weights are NOT
// committed (see models/.gitignore); the manifest IS. Run this once locally or
// in CI to populate models/<name>/ before the `#[ignore]`d integration tests
// and the embedder spike can run.
//
// Bootstrap: if a manifest entry's `sha256` is empty, the script downloads the
// file, computes the hash and writes it back into the manifest (so you fill
// real hashes exactly once). If a `sha256` is present, the download is verified
// against it and the script FAILS CLOSED on any mismatch — same contract the
// Rust loader enforces.
//
// Usage:
//   node scripts/fetch-models.mjs            # download + verify (or bootstrap)
//   node scripts/fetch-models.mjs --check    # verify only; do not download
//   node scripts/fetch-models.mjs --model bge-small-en-v1.5  # one pinned model
//   node scripts/fetch-models.mjs --add-openjev <40-hex HF commit>
//       # append the openjev-small-v0 entry (ADR-007 §6) with EMPTY pins and
//       # bootstrap them from that immutable revision (publish step, plan Step 7)
//
// OpenJev (ADR-007 §6): files come from HF `ruvnet/openjev-small-v0` at an
// immutable commit (`/resolve/<40-hex sha>/…`, never `main`). The revision is
// carried by the manifest entry (`hf_revision`), so the download map below is
// derived from the entry — a clean checkout reproduces the pinned hashes. No
// entry is committed until the model is published: every committed entry must
// carry real pins (scripts/verify-release-bench.mjs --preflight-models).

import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile, stat } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG = resolve(HERE, '..');
const MODELS_DIR = join(PKG, 'models');
const MANIFEST = join(MODELS_DIR, 'manifest.json');
const HF = 'https://huggingface.co';

// The download plan: which HF repo + path backs each manifest `file`. The
// manifest is the source of truth for hashes; this map is only how to fetch.
const PLAN = {
  'bge-small-en-v1.5/model.onnx': 'Xenova/bge-small-en-v1.5/resolve/main/onnx/model.onnx',
  'bge-small-en-v1.5/model_quantized.onnx': 'Xenova/bge-small-en-v1.5/resolve/main/onnx/model_quantized.onnx',
  'bge-small-en-v1.5/tokenizer.json': 'Xenova/bge-small-en-v1.5/resolve/main/tokenizer.json',
  'all-MiniLM-L6-v2/model.onnx': 'Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx',
  'all-MiniLM-L6-v2/model_quantized.onnx': 'Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model_quantized.onnx',
  'all-MiniLM-L6-v2/tokenizer.json': 'Xenova/all-MiniLM-L6-v2/resolve/main/tokenizer.json',
};

// OpenJev download map: manifest-relative file → path inside the HF repo.
export const OPENJEV = {
  name: 'openjev-small-v0',
  repo: 'ruvnet/openjev-small-v0',
  files: {
    'openjev-small-v0/model.onnx': 'onnx/model.onnx',
    'openjev-small-v0/tokenizer.json': 'tokenizer.json',
    'openjev-small-v0/train-text-hashes.txt': 'train-text-hashes.txt',
  },
};
const HF_REVISION = /^[0-9a-f]{40}$/;

/** Throw unless `rev` is an immutable 40-hex HF commit sha (never a branch). */
export function assertHfRevision(rev) {
  if (typeof rev !== 'string' || !HF_REVISION.test(rev)) {
    throw new Error(`HF revision must be a 40-hex commit sha (immutable), got ${JSON.stringify(rev)}`);
  }
  return rev;
}

/**
 * The ADR-007 §6 manifest entry shape for OpenJev. Empty pins are the
 * bootstrap state (filled by this script from the immutable revision).
 */
export function openjevManifestEntry({ revision, sha256 = '', tokenizerSha256 = '', trainHashesSha256 = '', added, reviewBy } = {}) {
  assertHfRevision(revision);
  const today = added ?? new Date().toISOString().slice(0, 10);
  return {
    name: OPENJEV.name,
    file: 'openjev-small-v0/model.onnx',
    sha256,
    dims: 384,
    license: 'MIT',
    source_url: `${HF}/${OPENJEV.repo}/resolve/${revision}/onnx/model.onnx`,
    added: today,
    review_by: reviewBy ?? addMonths(today, 6),
    pooling: 'cls',
    tokenizer_file: 'openjev-small-v0/tokenizer.json',
    tokenizer_sha256: tokenizerSha256,
    max_tokens: 256,
    hf_repo: OPENJEV.repo,
    hf_revision: revision,
    train_hashes_file: 'openjev-small-v0/train-text-hashes.txt',
    train_hashes_sha256: trainHashesSha256,
  };
}

function addMonths(isoDate, months) {
  const d = new Date(`${isoDate}T00:00:00Z`);
  d.setUTCMonth(d.getUTCMonth() + months);
  return d.toISOString().slice(0, 10);
}

/** HF path (`repo/resolve/rev/path`) for a manifest file, or null if unknown. */
export function planPath(relPath, entry) {
  if (relPath in PLAN) return PLAN[relPath];
  if (entry && entry.hf_repo === OPENJEV.repo && relPath in OPENJEV.files) {
    return `${OPENJEV.repo}/resolve/${assertHfRevision(entry.hf_revision)}/${OPENJEV.files[relPath]}`;
  }
  return null;
}

export function parseArgs(argv) {
  let checkOnly = false;
  let modelName = null;
  let addOpenjev = null;
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--add-openjev') {
      if (addOpenjev !== null) throw new Error('--add-openjev supplied twice');
      addOpenjev = assertHfRevision(argv[++i]);
    } else if (argv[i] === '--check') {
      if (checkOnly) throw new Error('--check supplied twice');
      checkOnly = true;
    } else if (argv[i] === '--model') {
      if (modelName !== null) throw new Error('--model supplied twice');
      modelName = argv[++i];
      if (!modelName || modelName.startsWith('--')) throw new Error('--model requires a model name');
    } else {
      throw new Error(`unknown argument: ${argv[i]}`);
    }
  }
  if (addOpenjev !== null && checkOnly) throw new Error('--add-openjev downloads; it cannot be combined with --check');
  if (addOpenjev !== null && modelName !== null && modelName !== OPENJEV.name) {
    throw new Error(`--add-openjev fetches only ${OPENJEV.name}`);
  }
  return { checkOnly, modelName: addOpenjev !== null ? OPENJEV.name : modelName, addOpenjev };
}

function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

async function exists(p) {
  try { await stat(p); return true; } catch { return false; }
}

async function fetchTo(relPath, dest, entry) {
  const url = `${HF}/${planPath(relPath, entry)}`;
  const res = await fetch(url, { redirect: 'follow' });
  if (!res.ok) throw new Error(`GET ${url} -> HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  await mkdir(dirname(dest), { recursive: true });
  await writeFile(dest, buf);
  return buf;
}

// Collect every distinct file referenced by the manifest (model + tokenizer).
function filesFromManifest(manifest) {
  const seen = new Map(); // relPath -> { hashField, entryName }
  for (const m of manifest.models) {
    seen.set(m.file, { entry: m, kind: 'model' });
    if (m.tokenizer_file && !seen.has(m.tokenizer_file)) {
      seen.set(m.tokenizer_file, { entry: m, kind: 'tokenizer' });
    }
    // OpenJev ships the Assertion B input (ADR-007 §2); pinned like the weights.
    if (m.train_hashes_file && !seen.has(m.train_hashes_file)) {
      seen.set(m.train_hashes_file, { entry: m, kind: 'train_hashes' });
    }
  }
  return seen;
}

const HASH_FIELD = { model: 'sha256', tokenizer: 'tokenizer_sha256', train_hashes: 'train_hashes_sha256' };
const FILE_FIELD = { model: 'file', tokenizer: 'tokenizer_file', train_hashes: 'train_hashes_file' };

async function main() {
  const { checkOnly, modelName, addOpenjev } = parseArgs(process.argv.slice(2));
  const manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
  let bootstrapping = false;
  if (addOpenjev !== null) {
    if (manifest.models.some((m) => m.name === OPENJEV.name)) {
      throw new Error(`manifest already has ${OPENJEV.name}; edit its hf_revision/pins by review, not by --add-openjev`);
    }
    manifest.models.push(openjevManifestEntry({ revision: addOpenjev }));
    bootstrapping = true; // the new entry's empty pins are filled below
  }
  const matching = modelName === null ? manifest.models : manifest.models.filter((m) => m.name === modelName);
  if (modelName !== null) {
    if (matching.length !== 1) {
      throw new Error(`--model ${modelName}: expected one manifest entry, found ${matching.length}`);
    }
    const [model] = matching;
    if (!bootstrapping && (!/^[0-9a-f]{64}$/.test(model.sha256) ||
        !/^[0-9a-f]{64}$/.test(model.tokenizer_sha256) ||
        (model.train_hashes_file && !/^[0-9a-f]{64}$/.test(model.train_hashes_sha256 ?? '')))) {
      throw new Error(`--model ${modelName}: model and tokenizer SHA-256 pins are required`);
    }
  }
  const files = filesFromManifest({ models: matching });
  let bootstrapped = false;
  let failures = 0;

  for (const [relPath, { entry, kind }] of files) {
    if (planPath(relPath, entry) === null) {
      console.error(`✗ ${relPath}: no download source in PLAN`);
      failures++;
      continue;
    }
    const dest = join(MODELS_DIR, relPath);
    let buf;
    if (await exists(dest)) {
      buf = await readFile(dest);
    } else if (checkOnly) {
      console.error(`✗ ${relPath}: missing (run without --check to download)`);
      failures++;
      continue;
    } else {
      process.stdout.write(`↓ ${relPath} ... `);
      buf = await fetchTo(relPath, dest, entry);
      console.log(`${(buf.length / 1e6).toFixed(1)} MB`);
    }

    const actual = sha256(buf);
    const field = HASH_FIELD[kind];
    const expected = entry[field];

    if (!expected) {
      // Bootstrap: fill every manifest entry that references this file.
      for (const m of manifest.models) {
        if (m[FILE_FIELD[kind]] === relPath) m[field] = actual;
      }
      bootstrapped = true;
      console.log(`  bootstrap ${field}=${actual}`);
    } else if (expected.toLowerCase() !== actual.toLowerCase()) {
      console.error(`✗ ${relPath}: hash mismatch\n    manifest=${expected}\n    actual  =${actual}`);
      failures++;
    } else {
      console.log(`✓ ${relPath}`);
    }
  }

  if (bootstrapped && !checkOnly) {
    await writeFile(MANIFEST, JSON.stringify(manifest, null, 2) + '\n');
    console.log(`\nWrote bootstrapped hashes to ${MANIFEST}`);
  }
  if (failures > 0) {
    console.error(`\n${failures} file(s) failed verification.`);
    process.exit(1);
  }
  console.log('\nAll model files present and verified.');
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch((e) => {
    console.error(e);
    process.exit(1);
  });
}
