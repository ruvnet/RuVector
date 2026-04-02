#!/usr/bin/env node
/**
 * rvf-builder.mjs - Create binary RVF containers from extracted source modules.
 *
 * Uses the @ruvector/rvf-node native backend to produce real binary .rvf files
 * with HNSW-indexed vector embeddings and witness chains.
 *
 * Each source fragment is embedded as a deterministic vector derived from its
 * content hash (a lightweight "fingerprint" embedding). This allows similarity
 * search across versions without requiring a full ML embedding model.
 *
 * Usage:
 *   node scripts/lib/rvf-builder.mjs <source-dir> <output.rvf> [--meta key=val ...]
 *
 * source-dir  : directory with .js module files + metrics.json
 * output.rvf  : path for the binary RVF container
 * --meta      : optional key=value metadata pairs
 */

import { readFileSync, readdirSync, existsSync, writeFileSync } from 'fs';
import { join, basename, resolve } from 'path';
import { createHash } from 'crypto';

// Vector dimension for fingerprint embeddings
const DIMENSIONS = 128;

/**
 * Generate a deterministic fingerprint vector from text content.
 *
 * Uses SHA-256 → expand to DIMENSIONS floats in [-1, 1].
 * This is NOT a semantic embedding but a content fingerprint that
 * allows exact-match deduplication and change detection across versions.
 */
function fingerprintVector(text) {
  const hash = createHash('sha256').update(text).digest();
  const vec = new Float32Array(DIMENSIONS);

  // Expand 32 bytes of hash into DIMENSIONS floats using a simple
  // deterministic expansion: for each float, mix two hash bytes.
  for (let i = 0; i < DIMENSIONS; i++) {
    const byteA = hash[i % 32];
    const byteB = hash[(i * 7 + 13) % 32];
    // Map to [-1, 1]
    vec[i] = ((byteA * 256 + byteB) / 65535) * 2 - 1;
  }

  // Normalize to unit length for cosine distance
  let norm = 0;
  for (let i = 0; i < DIMENSIONS; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) {
    for (let i = 0; i < DIMENSIONS; i++) vec[i] /= norm;
  }

  return vec;
}

/**
 * Load the native rvf-node backend.
 */
async function loadRvfNode() {
  // Try several possible paths for the native module
  const candidates = [
    resolve(process.cwd(), 'npm/packages/rvf-node/index.js'),
    resolve(process.cwd(), 'node_modules/@ruvector/rvf-node/index.js'),
  ];

  for (const p of candidates) {
    if (existsSync(p)) {
      const mod = await import(p);
      return mod.RvfDatabase ?? mod.default?.RvfDatabase ?? mod;
    }
  }
  throw new Error(
    'Could not find @ruvector/rvf-node. Tried:\n  ' + candidates.join('\n  ')
  );
}

/**
 * Parse --meta key=value arguments from argv.
 */
function parseMeta(argv) {
  const meta = {};
  for (let i = 0; i < argv.length; i++) {
    if (argv[i] === '--meta' && argv[i + 1]) {
      const [k, ...rest] = argv[i + 1].split('=');
      meta[k] = rest.join('=');
      i++;
    }
  }
  return meta;
}

/**
 * Main entry point.
 */
async function main() {
  const args = process.argv.slice(2);
  const sourceDir = args[0];
  const outputRvf = args[1];

  if (!sourceDir || !outputRvf) {
    console.error(
      'Usage: node rvf-builder.mjs <source-dir> <output.rvf> [--meta key=val ...]'
    );
    process.exit(1);
  }

  const meta = parseMeta(args.slice(2));

  // Load native RVF module
  let RvfDatabase;
  try {
    RvfDatabase = await loadRvfNode();
  } catch (err) {
    console.error('Failed to load @ruvector/rvf-node:', err.message);
    process.exit(1);
  }

  // Read metrics if available
  const metricsPath = join(sourceDir, 'metrics.json');
  let metrics = {};
  if (existsSync(metricsPath)) {
    metrics = JSON.parse(readFileSync(metricsPath, 'utf-8'));
  }

  // Collect all .js module files
  const moduleFiles = readdirSync(sourceDir)
    .filter((f) => f.endsWith('.js'))
    .sort();

  if (moduleFiles.length === 0) {
    console.error(`No .js module files found in ${sourceDir}`);
    process.exit(1);
  }

  console.log(
    `Building RVF container: ${basename(outputRvf)} (${moduleFiles.length} modules, ${DIMENSIONS}d vectors)`
  );

  // Create the RVF database
  const db = RvfDatabase.create(outputRvf, {
    dimension: DIMENSIONS,
    metric: 'Cosine',
    profile: 0,
    compression: 'None',
    signing: false,
    m: 16,
    ef_construction: 200,
  });

  // Ingest vectors for each module fragment
  let totalFragments = 0;
  let vectorId = 1;
  const idMap = {};

  for (const modFile of moduleFiles) {
    const modName = basename(modFile, '.js');
    const content = readFileSync(join(sourceDir, modFile), 'utf-8');
    const fragments = content.split('\n\n').filter((f) => f.trim().length > 10);

    if (fragments.length === 0) continue;

    // Build a flat vector array and IDs for batch ingest
    const vectors = new Float32Array(fragments.length * DIMENSIONS);
    const ids = [];

    for (let i = 0; i < fragments.length; i++) {
      const vec = fingerprintVector(fragments[i]);
      vectors.set(vec, i * DIMENSIONS);
      ids.push(vectorId);
      idMap[vectorId] = {
        module: modName,
        fragmentIndex: i,
        sizeBytes: Buffer.byteLength(fragments[i]),
        hash: createHash('sha256').update(fragments[i]).digest('hex').slice(0, 16),
      };
      vectorId++;
    }

    const result = db.ingestBatch(vectors, ids);
    totalFragments += result.accepted;
    console.log(
      `  ${modName}: ${result.accepted} vectors ingested (${fragments.length} fragments)`
    );
  }

  // Get final status
  const status = db.status();
  const fileId = db.fileId();
  const segments = db.segments();

  // Write the ID mapping sidecar (extends the default .idmap.json)
  const sidecarPath = outputRvf + '.manifest.json';
  const manifest = {
    format: 'rvf-binary',
    version: '1.0',
    fileId,
    dimensions: DIMENSIONS,
    metric: 'cosine',
    totalVectors: status.totalVectors,
    totalSegments: status.totalSegments,
    fileSizeBytes: status.fileSize,
    epoch: status.currentEpoch,
    segments: segments.map((s) => ({
      id: s.id,
      type: s.segType,
      offset: s.offset,
      payloadLength: s.payloadLength,
    })),
    source: {
      package: meta.package || '@anthropic-ai/claude-code',
      version: meta.version || metrics.version || 'unknown',
      extractedAt: metrics.extractedAt || new Date().toISOString(),
      metrics: {
        bundleSizeBytes: metrics.sizeBytes || 0,
        classes: metrics.classes || 0,
        functions: metrics.functions || 0,
        asyncFunctions: metrics.asyncFunctions || 0,
        arrowFunctions: metrics.arrowFunctions || 0,
      },
    },
    modules: Object.entries(metrics.modules || {}).map(([name, info]) => ({
      name,
      ...info,
    })),
    idMap,
    meta,
    createdAt: new Date().toISOString(),
  };

  writeFileSync(sidecarPath, JSON.stringify(manifest, null, 2));

  db.close();

  console.log(`\nRVF container created successfully:`);
  console.log(`  File: ${outputRvf}`);
  console.log(`  File ID: ${fileId}`);
  console.log(`  Vectors: ${totalFragments}`);
  console.log(`  Segments: ${status.totalSegments}`);
  console.log(`  Size: ${(status.fileSize / 1024).toFixed(1)} KB`);
  console.log(`  Manifest: ${sidecarPath}`);

  // Output JSON for caller
  const result = {
    success: true,
    path: outputRvf,
    fileId,
    vectors: totalFragments,
    segments: status.totalSegments,
    sizeBytes: status.fileSize,
  };
  console.log(JSON.stringify(result));
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
