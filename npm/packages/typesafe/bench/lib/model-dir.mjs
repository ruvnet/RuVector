// Resolve an ONNX embedder arm to ONE sha256-verified manifest entry
// (ADR-005 supply chain; ADR-007 §6, plan Step 4).
//
// Why in JS: the binding accepts the manifest either as a path or as inline
// JSON, and when handed a multi-entry manifest it silently falls back to the
// FIRST entry if the name is missing (ffi `select_manifest`). So the bench
// selects the entry itself, hashes the model + tokenizer bytes, fails closed on
// a missing or mismatched pin, records the result in the receipt, and passes
// the single verified entry inline. The Rust loader then re-verifies the same
// pins (belt and braces).
//
// Model dir layouts accepted by `--model-dir DIR`:
//   1. DIR/manifest.json (or --manifest PATH) — the committed `models/` root or
//      a staged candidate (plan Step 4: DIR/openjev-small-v0/{model.onnx,
//      tokenizer.json,train-text-hashes.txt} + DIR/manifest.json with pins).
//      Entry `file` / `tokenizer_file` / `train_hashes_file` are relative to DIR.
//   2. A bare, unpublished model dir with no manifest: DIR/model.onnx (or
//      DIR/onnx/model.onnx) + DIR/tokenizer.json [+ DIR/train-text-hashes.txt].
//      An entry is synthesized with the ADR-007 frozen shape (dims 384, CLS,
//      max_tokens 256); its pins are the hashes computed now, and the receipt
//      marks it `synthesized: true`.
//
// Train-hash discovery (Assertion B): --train-hashes PATH > entry
// `train_hashes_file` (verified against `train_hashes_sha256` when pinned) >
// `<dir of entry.file>/train-text-hashes.txt` when present. A model whose name
// starts with `openjev` MUST resolve one.

import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { basename, dirname, isAbsolute, join, normalize, resolve, sep } from 'node:path';
import { readTrainHashes } from './leakage.mjs';

const HEX64 = /^[0-9a-f]{64}$/;
export const DEFAULT_MODEL = 'bge-small-en-v1.5';
export const TRAIN_HASHES_NAME = 'train-text-hashes.txt';
/** ADR-007 §3/§6 frozen shape used when synthesizing an entry for a bare dir. */
export const OPENJEV_SHAPE = { dims: 384, pooling: 'cls', max_tokens: 256 };

const sha256File = (p) => createHash('sha256').update(readFileSync(p)).digest('hex');

function modelError(message, code) {
  const e = new Error(message);
  e.code = code;
  return e;
}

/** Join a manifest-relative path under `dir`, refusing absolute paths and `..` escapes. */
export function safeJoin(dir, rel, what) {
  if (typeof rel !== 'string' || !rel.length) throw modelError(`${what}: missing path`, 'MODEL_INVALID');
  if (isAbsolute(rel) || normalize(rel).split(/[\\/]/).includes('..')) {
    throw modelError(`${what}: '${rel}' must be a relative path inside the model dir`, 'MODEL_INVALID');
  }
  const abs = resolve(dir, rel);
  if (abs !== resolve(dir) && !abs.startsWith(resolve(dir) + sep)) {
    throw modelError(`${what}: '${rel}' escapes the model dir`, 'MODEL_INVALID');
  }
  return abs;
}

function readManifestSource(src) {
  // Same contract as the binding: a path, else inline JSON.
  if (existsSync(src)) return { json: JSON.parse(readFileSync(src, 'utf8')), source: resolve(src), sha256: sha256File(src) };
  return { json: JSON.parse(src), source: 'inline', sha256: createHash('sha256').update(src).digest('hex') };
}

function selectEntry(json, name) {
  const entries = Array.isArray(json?.models) ? json.models : json && typeof json.name === 'string' ? [json] : null;
  if (!entries) throw modelError('manifest: neither {models:[…]} nor a single entry', 'MODEL_INVALID');
  const want = name ?? (entries.length === 1 ? entries[0].name : DEFAULT_MODEL);
  const hits = entries.filter((m) => m.name === want);
  if (hits.length !== 1) {
    throw modelError(`manifest: expected exactly one entry named '${want}', found ${hits.length} (have: ${entries.map((m) => m.name).join(', ')})`, 'MODEL_INVALID');
  }
  return hits[0];
}

function synthesizeEntry(dir, name) {
  const file = ['model.onnx', join('onnx', 'model.onnx')].find((f) => existsSync(join(dir, f)));
  if (!file || !existsSync(join(dir, 'tokenizer.json'))) {
    throw modelError(`model dir ${dir}: no manifest.json and no model.onnx|onnx/model.onnx + tokenizer.json to synthesize one from`, 'MODEL_MISSING');
  }
  const today = new Date().toISOString().slice(0, 10);
  return {
    name: name ?? basename(resolve(dir)),
    file: file.split(sep).join('/'),
    sha256: sha256File(join(dir, file)),
    tokenizer_file: 'tokenizer.json',
    tokenizer_sha256: sha256File(join(dir, 'tokenizer.json')),
    ...OPENJEV_SHAPE,
    license: 'UNRELEASED',
    source_url: 'local:unpublished',
    added: today,
    review_by: today,
    ...(existsSync(join(dir, TRAIN_HASHES_NAME)) ? { train_hashes_file: TRAIN_HASHES_NAME } : {}),
  };
}

function verifyPinned(dir, entry, fileKey, hashKey) {
  const pin = entry[hashKey];
  if (typeof pin !== 'string' || !HEX64.test(pin)) {
    throw modelError(`${entry.name}: ${hashKey} must be a pinned lowercase 64-hex sha256 (got ${JSON.stringify(pin ?? null)})`, 'MODEL_UNPINNED');
  }
  const abs = safeJoin(dir, entry[fileKey], `${entry.name}.${fileKey}`);
  if (!existsSync(abs)) throw modelError(`${entry.name}: ${entry[fileKey]} missing under ${dir} (run scripts/fetch-models.mjs)`, 'MODEL_MISSING');
  const actual = sha256File(abs);
  if (actual !== pin) {
    throw modelError(`${entry.name}: ${entry[fileKey]} sha256 mismatch\n  pinned ${pin}\n  actual ${actual}`, 'MODEL_HASH_MISMATCH');
  }
  return actual;
}

function resolveTrainHashes(dir, entry, explicit) {
  if (explicit) {
    if (!existsSync(explicit)) throw modelError(`--train-hashes ${explicit}: not found`, 'MODEL_INVALID');
    return { path: resolve(explicit), via: 'flag' };
  }
  if (entry.train_hashes_file) {
    const abs = safeJoin(dir, entry.train_hashes_file, `${entry.name}.train_hashes_file`);
    if (!existsSync(abs)) throw modelError(`${entry.name}: train_hashes_file ${entry.train_hashes_file} missing`, 'MODEL_INVALID');
    if (entry.train_hashes_sha256 !== undefined && entry.train_hashes_sha256 !== '') {
      const actual = sha256File(abs);
      if (actual !== entry.train_hashes_sha256) {
        throw modelError(`${entry.name}: train_hashes_file sha256 mismatch\n  pinned ${entry.train_hashes_sha256}\n  actual ${actual}`, 'MODEL_HASH_MISMATCH');
      }
    }
    return { path: abs, via: 'manifest' };
  }
  const sibling = join(dirname(safeJoin(dir, entry.file, `${entry.name}.file`)), TRAIN_HASHES_NAME);
  return existsSync(sibling) ? { path: sibling, via: 'model-dir' } : null;
}

/**
 * @param opts { modelDir?, manifest?, model?, trainHashes? } (CLI flag values)
 * @returns { spec, record, trainHashesPath } — spec is the binding's embedder
 *   object with the single verified entry inlined; record goes in the receipt.
 * Throws with e.code ∈ MODEL_MISSING | MODEL_UNPINNED | MODEL_HASH_MISMATCH | MODEL_INVALID.
 */
export function resolveOnnxModel({ modelDir, manifest, model, trainHashes } = {}) {
  const dir = resolve(modelDir || 'models');
  const manifestSrc = manifest || (existsSync(join(dir, 'manifest.json')) ? join(dir, 'manifest.json') : null);
  let entry;
  let manifestInfo;
  if (manifestSrc) {
    const m = readManifestSource(manifestSrc);
    entry = selectEntry(m.json, model);
    manifestInfo = { source: m.source, sha256: m.sha256, synthesized: false };
  } else {
    entry = synthesizeEntry(dir, model);
    manifestInfo = { source: 'synthesized', sha256: null, synthesized: true };
  }
  const sha256 = verifyPinned(dir, entry, 'file', 'sha256');
  const tokenizerSha256 = verifyPinned(dir, entry, 'tokenizer_file', 'tokenizer_sha256');
  const th = resolveTrainHashes(dir, entry, trainHashes);
  if (!th && /^openjev/i.test(entry.name)) {
    throw modelError(`${entry.name}: OpenJev models must ship ${TRAIN_HASHES_NAME} (or pass --train-hashes) — ADR-007 §2 Assertion B`, 'MODEL_INVALID');
  }
  return {
    spec: { kind: 'onnx', modelDir: dir, manifest: JSON.stringify(entry), model: entry.name },
    trainHashesPath: th?.path ?? null,
    record: {
      name: entry.name,
      id: `${entry.name}@${sha256.slice(0, 12)}`,
      model_dir: dir,
      file: entry.file,
      sha256,
      tokenizer_file: entry.tokenizer_file,
      tokenizer_sha256: tokenizerSha256,
      dims: entry.dims,
      pooling: entry.pooling ?? null,
      max_tokens: entry.max_tokens ?? null,
      source_url: entry.source_url ?? null,
      manifest: manifestInfo,
      train_hashes: th ? { path: th.path, via: th.via } : null,
    },
  };
}

/**
 * Per-run local-arm context for bench/run.mjs: the verified ONNX entry (when
 * `--embedder onnx`) and the parsed train hashes (Assertion B input). A missing
 * model file keeps the historical "engine unavailable" behaviour; an unpinned,
 * mismatched or ambiguous model is a hard error (fail closed). The jev replay
 * arm loads no model and gets an empty context.
 */
export function resolveLocalContext(args) {
  const ctx = { onnx: null, trainHashes: null };
  if (args.arm === 'jev') return ctx;
  let trainHashesPath = args.trainHashes ? resolve(args.trainHashes) : null;
  if (args.embedder === 'onnx') {
    try {
      ctx.onnx = resolveOnnxModel(args);
      trainHashesPath = ctx.onnx.trainHashesPath;
    } catch (e) {
      if (e.code !== 'MODEL_MISSING') throw e;
      ctx.onnx = { unavailable: e.message };
    }
  }
  if (trainHashesPath) ctx.trainHashes = readTrainHashes(trainHashesPath);
  return ctx;
}
