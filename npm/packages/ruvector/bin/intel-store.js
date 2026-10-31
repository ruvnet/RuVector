'use strict';

const fs = require('fs');
const os = require('os');
const path = require('path');

/**
 * Where the intelligence store lives — shared by the CLI hooks and the MCP
 * server so both always read and write the same file.
 *
 * Memory belongs to the project: a directory with .ruvector/ or .claude/, or a
 * repository root, gets its own store even when a ~/.ruvector store exists from
 * other work. The home store is only the fallback for ad-hoc directories.
 */
const PROJECT_MARKERS = ['.git', 'package.json', 'Cargo.toml', 'pyproject.toml', 'go.mod'];

function resolveIntelStorePath(cwd = process.cwd()) {
  const projectPath = path.join(cwd, '.ruvector', 'intelligence.json');
  const homePath = path.join(os.homedir(), '.ruvector', 'intelligence.json');
  if (fs.existsSync(path.dirname(projectPath))) return projectPath;
  if (fs.existsSync(path.join(cwd, '.claude'))) return projectPath;
  if (PROJECT_MARKERS.some(m => fs.existsSync(path.join(cwd, m)))) return projectPath;
  if (fs.existsSync(homePath)) return homePath;
  return projectPath;
}

/**
 * Serialize the store compactly. Embeddings dominate the file and every hook
 * parses it, so they are written with float32-level precision (7 significant
 * digits, max error ~5e-8) and without indentation: a 5k-memory store goes from
 * ~56 MB to ~21 MB and parses in half the time. The format is unchanged — plain
 * JSON number arrays — so the Rust CLI and @ruvector/cli still read it.
 */
function roundEmbedding(embedding) {
  if (!Array.isArray(embedding)) return embedding;
  return embedding.map(v => (typeof v === 'number' && Number.isFinite(v) ? Number(v.toPrecision(7)) : v));
}

function serializeStore(data) {
  const out = Array.isArray(data && data.memories)
    ? { ...data, memories: data.memories.map(m => (m && Array.isArray(m.embedding) ? { ...m, embedding: roundEmbedding(m.embedding) } : m)) }
    : data;
  return JSON.stringify(out);
}

module.exports = { resolveIntelStorePath, serializeStore, roundEmbedding };
