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

const checkOnly = process.argv.includes('--check');

function sha256(buf) {
  return createHash('sha256').update(buf).digest('hex');
}

async function exists(p) {
  try { await stat(p); return true; } catch { return false; }
}

async function fetchTo(relPath, dest) {
  const url = `${HF}/${PLAN[relPath]}`;
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
  }
  return seen;
}

async function main() {
  const manifest = JSON.parse(await readFile(MANIFEST, 'utf8'));
  const files = filesFromManifest(manifest);
  let bootstrapped = false;
  let failures = 0;

  for (const [relPath, { entry, kind }] of files) {
    if (!(relPath in PLAN)) {
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
      buf = await fetchTo(relPath, dest);
      console.log(`${(buf.length / 1e6).toFixed(1)} MB`);
    }

    const actual = sha256(buf);
    const field = kind === 'model' ? 'sha256' : 'tokenizer_sha256';
    const expected = entry[field];

    if (!expected) {
      // Bootstrap: fill every manifest entry that references this file.
      for (const m of manifest.models) {
        if (kind === 'model' && m.file === relPath) m.sha256 = actual;
        if (kind === 'tokenizer' && m.tokenizer_file === relPath) m.tokenizer_sha256 = actual;
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

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
