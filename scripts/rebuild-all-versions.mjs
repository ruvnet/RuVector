#!/usr/bin/env node
/**
 * rebuild-all-versions.mjs - Delete and rebuild ALL Claude Code version
 * decompilations from scratch.
 *
 * For each major.minor series:
 *   1. Download the latest patch from npm
 *   2. Extract cli.js
 *   3. Beautify the entire file
 *   4. Split into modules (100% coverage with uncategorized)
 *   5. Compute metrics per module
 *   6. Generate witness chain
 *   7. Build RVF container
 *   8. Write README with version info and metrics
 *
 * Also rebuilds the extracted/ directory for the latest version.
 *
 * Usage:
 *   node scripts/rebuild-all-versions.mjs [--series 0.2,1.0,2.0,2.1]
 */

import { execSync } from 'child_process';
import {
  existsSync, mkdirSync, readFileSync, writeFileSync,
  readdirSync, rmSync, statSync,
} from 'fs';
import { join, resolve, basename } from 'path';
import { createHash } from 'crypto';

const ROOT = resolve(import.meta.dirname, '..');
const VERSIONS_DIR = join(ROOT, 'docs/research/claude-code-rvsource/versions');
const EXTRACTED_DIR = join(ROOT, 'docs/research/claude-code-rvsource/extracted');
const DECOMPILER_DIR = join(ROOT, 'npm/packages/ruvector/src/decompiler');
const RVF_NODE_DIR = join(ROOT, 'npm/packages/rvf-node');
const TMP_BASE = '/tmp/cc-rebuild-' + process.pid;

// Load decompiler modules
const { splitModules } = await import(join(DECOMPILER_DIR, 'module-splitter.js'));
const { computeMetrics, computeModuleMetrics } = await import(
  join(DECOMPILER_DIR, 'metrics.js')
);
const { buildWitnessChain } = await import(join(DECOMPILER_DIR, 'witness.js'));

// Try to load js-beautify
let beautify;
try {
  const jsBeautify = (await import('js-beautify')).default;
  beautify = (source) =>
    (jsBeautify.js || jsBeautify)(source, {
      indent_size: 2,
      space_in_empty_paren: false,
      preserve_newlines: true,
      max_preserve_newlines: 2,
      end_with_newline: true,
    });
  console.log('[+] js-beautify loaded');
} catch {
  beautify = (s) => s;
  console.log('[!] js-beautify not available, using raw source');
}

// Try to load RVF native backend
let RvfDatabase = null;
try {
  const mod = await import(join(RVF_NODE_DIR, 'index.js'));
  RvfDatabase = mod.RvfDatabase ?? mod.default?.RvfDatabase ?? null;
  if (RvfDatabase) console.log('[+] @ruvector/rvf-node loaded');
} catch {
  console.log('[!] @ruvector/rvf-node not available, will skip RVF creation');
}

// Parse CLI args
const args = process.argv.slice(2);
let filterSeries = null;
for (let i = 0; i < args.length; i++) {
  if (args[i] === '--series' && args[i + 1]) {
    filterSeries = args[i + 1].split(',');
    i++;
  }
}

// Module-to-subdirectory mapping for organized output
const MODULE_DIR_MAP = {
  'agent-loop': 'core',
  'context-manager': 'core',
  'streaming-handler': 'core',
  'tool-dispatch': 'tools',
  'mcp-client': 'tools/mcp',
  'permission-system': 'permissions',
  'commands': 'ui',
  'command-defs': 'ui',
  'config': 'config',
  'env-vars': 'config',
  'model-provider': 'config',
  'session': 'core',
  'telemetry': 'telemetry',
  'telemetry-events': 'telemetry',
  'class-hierarchy': 'types',
  'api-endpoints': 'types',
  'uncategorized': 'uncategorized',
};

/**
 * Get the subdirectory path for a module name.
 */
function getModuleDir(moduleName) {
  return MODULE_DIR_MAP[moduleName] || 'uncategorized';
}

/**
 * Get the file path (within source/) for a module.
 */
function getModulePath(baseDir, moduleName) {
  const subDir = getModuleDir(moduleName);
  const dir = join(baseDir, subDir);
  mkdirSync(dir, { recursive: true });
  return join(dir, `${moduleName}.js`);
}

// Vector fingerprint for RVF
const DIMENSIONS = 128;
function fingerprintVector(text) {
  const hash = createHash('sha256').update(text).digest();
  const vec = new Float32Array(DIMENSIONS);
  for (let i = 0; i < DIMENSIONS; i++) {
    const a = hash[i % 32];
    const b = hash[(i * 7 + 13) % 32];
    vec[i] = ((a * 256 + b) / 65535) * 2 - 1;
  }
  let norm = 0;
  for (let i = 0; i < DIMENSIONS; i++) norm += vec[i] * vec[i];
  norm = Math.sqrt(norm);
  if (norm > 0) for (let i = 0; i < DIMENSIONS; i++) vec[i] /= norm;
  return vec;
}

/**
 * Get all Claude Code versions from npm, grouped by major.minor.
 * Returns array of { series, version } sorted by semver.
 */
function getVersionGroups() {
  console.log('[+] Fetching Claude Code versions from npm...');
  const raw = execSync(
    'npm view @anthropic-ai/claude-code versions --json 2>/dev/null',
    { encoding: 'utf-8', maxBuffer: 10 * 1024 * 1024 },
  );
  const versions = JSON.parse(raw);
  const groups = {};

  for (const v of versions) {
    const parts = v.split('.');
    const key = parts[0] + '.' + parts[1];
    const patch = parseInt(parts[2], 10);
    if (!groups[key] || patch > groups[key].patch) {
      groups[key] = { version: v, patch, series: key };
    }
  }

  return Object.values(groups).sort((a, b) => {
    const [aMaj, aMin] = a.series.split('.').map(Number);
    const [bMaj, bMin] = b.series.split('.').map(Number);
    return aMaj !== bMaj ? aMaj - bMaj : aMin - bMin;
  });
}

/**
 * Download a specific version and extract cli.js.
 * Returns the path to cli.js or null on failure.
 */
function downloadVersion(version) {
  const dir = join(TMP_BASE, `extract-${version}`);
  mkdirSync(dir, { recursive: true });

  console.log(`  Downloading @anthropic-ai/claude-code@${version}...`);
  try {
    const tgzDir = join(TMP_BASE, 'tarballs');
    mkdirSync(tgzDir, { recursive: true });

    execSync(
      `npm pack "@anthropic-ai/claude-code@${version}" --pack-destination "${tgzDir}" 2>/dev/null`,
      { stdio: 'pipe' },
    );

    const tgzFiles = readdirSync(tgzDir).filter((f) =>
      f.startsWith('anthropic-ai-claude-code-') && f.endsWith('.tgz'),
    );
    if (tgzFiles.length === 0) return null;

    const tgz = join(tgzDir, tgzFiles[0]);

    // Try cli.js then cli.mjs
    try {
      execSync(`tar xf "${tgz}" -C "${dir}" --strip-components=1 package/cli.js 2>/dev/null`, {
        stdio: 'pipe',
      });
    } catch {}
    try {
      execSync(`tar xf "${tgz}" -C "${dir}" --strip-components=1 package/cli.mjs 2>/dev/null`, {
        stdio: 'pipe',
      });
    } catch {}
    try {
      execSync(
        `tar xf "${tgz}" -C "${dir}" --strip-components=1 package/package.json 2>/dev/null`,
        { stdio: 'pipe' },
      );
    } catch {}

    // Clean up tarball
    try { rmSync(tgz); } catch {}

    // Rename cli.mjs -> cli.js if needed
    if (existsSync(join(dir, 'cli.mjs')) && !existsSync(join(dir, 'cli.js'))) {
      execSync(`mv "${join(dir, 'cli.mjs')}" "${join(dir, 'cli.js')}"`);
    }

    const cliPath = join(dir, 'cli.js');
    if (!existsSync(cliPath)) return null;

    const size = statSync(cliPath).size;
    console.log(`  Extracted cli.js (${(size / 1024 / 1024).toFixed(1)} MB)`);
    return cliPath;
  } catch (err) {
    console.log(`  [!] Download failed: ${err.message}`);
    return null;
  }
}

/**
 * Full decompilation pipeline for a single version.
 */
function decompileVersion(cliPath, outputDir, series, version) {
  const sourceDir = join(outputDir, 'source');
  mkdirSync(sourceDir, { recursive: true });

  // Read raw source
  const raw = readFileSync(cliPath, 'utf-8');
  const rawSize = Buffer.byteLength(raw);
  console.log(`  Raw source: ${raw.split('\n').length} lines, ${(rawSize / 1024 / 1024).toFixed(1)} MB`);

  // Beautify
  console.log('  Beautifying...');
  const beautified = beautify(raw);
  console.log(`  Beautified: ${beautified.split('\n').length} lines`);

  // Split into modules (using beautified source for readability)
  console.log('  Splitting into modules...');
  const { modules, unclassified } = splitModules(beautified, { minConfidence: 0 });
  console.log(`  Found ${modules.length} modules`);

  // Write each module to source/<subdir>/
  let totalCapturedBytes = 0;
  const moduleResults = {};

  for (const mod of modules) {
    const filePath = getModulePath(sourceDir, mod.name);
    const header = `// Module: ${mod.name}\n// Confidence: ${mod.confidence}\n// Fragments: ${mod.fragments}\n// Version: ${version}\n\n`;
    const content = header + mod.content;

    writeFileSync(filePath, content);
    const sizeBytes = Buffer.byteLength(content);
    totalCapturedBytes += sizeBytes;

    const subDir = getModuleDir(mod.name);
    moduleResults[mod.name] = {
      fragments: mod.fragments,
      sizeBytes,
      confidence: mod.confidence,
      directory: subDir,
    };
    console.log(`    ${subDir}/${mod.name}: ${mod.fragments} fragments (${(sizeBytes / 1024).toFixed(1)} KB, confidence=${mod.confidence})`);
  }

  // Compute source-level metrics
  const sourceMetrics = computeMetrics(beautified);
  const moduleMetrics = computeModuleMetrics(modules);

  // Build witness chain
  console.log('  Building witness chain...');
  const witness = buildWitnessChain(raw, modules);

  // Write metrics at sourceDir root (not in any subdirectory)
  const metricsData = {
    version,
    series,
    sizeBytes: rawSize,
    beautifiedSizeBytes: Buffer.byteLength(beautified),
    capturedBytes: totalCapturedBytes,
    coveragePercent: parseFloat(
      ((totalCapturedBytes / Buffer.byteLength(beautified)) * 100).toFixed(1),
    ),
    lines: sourceMetrics.lines,
    functions: sourceMetrics.functions,
    asyncFunctions: sourceMetrics.asyncFunctions,
    arrowFunctions: sourceMetrics.arrowFunctions,
    classes: sourceMetrics.classes,
    classExtensions: sourceMetrics.classExtensions,
    constDeclarations: sourceMetrics.constDeclarations,
    letDeclarations: sourceMetrics.letDeclarations,
    varDeclarations: sourceMetrics.varDeclarations,
    imports: sourceMetrics.imports,
    exports: sourceMetrics.exports,
    requires: sourceMetrics.requires,
    awaitExpressions: sourceMetrics.awaitExpressions,
    tryBlocks: sourceMetrics.tryBlocks,
    sourceFile: basename(cliPath),
    extractedAt: new Date().toISOString(),
    modules: moduleResults,
    moduleMetrics: moduleMetrics,
  };

  // Write metrics at version dir root (parent of source/)
  writeFileSync(join(outputDir, 'metrics.json'), JSON.stringify(metricsData, null, 2));
  // Write witness in source/ root
  writeFileSync(join(sourceDir, 'witness.json'), JSON.stringify(witness, null, 2));

  return { metricsData, witness, modules };
}

/**
 * Recursively collect all .js files from a directory tree.
 */
function collectJsFiles(dir, prefix = '') {
  const results = [];
  if (!existsSync(dir)) return results;
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const relPath = prefix ? `${prefix}/${entry.name}` : entry.name;
    if (entry.isDirectory()) {
      results.push(...collectJsFiles(join(dir, entry.name), relPath));
    } else if (entry.name.endsWith('.js')) {
      results.push({ path: join(dir, entry.name), relPath, name: entry.name });
    }
  }
  return results;
}

/**
 * Create a single RVF container from a list of JS files.
 */
function createRvfFromFiles(jsFiles, rvfPath, metricsJson, version, series) {
  if (!RvfDatabase || jsFiles.length === 0) return null;

  try {
    const db = RvfDatabase.create(rvfPath, {
      dimension: DIMENSIONS,
      metric: 'Cosine',
      profile: 0,
      compression: 'None',
      signing: false,
      m: 16,
      ef_construction: 200,
    });

    let vectorId = 1;
    let totalFragments = 0;
    const idMap = {};

    for (const file of jsFiles) {
      const modName = basename(file.name, '.js');
      const content = readFileSync(file.path, 'utf-8');
      const fragments = content.split('\n\n').filter((f) => f.trim().length > 10);
      if (fragments.length === 0) continue;

      const vectors = new Float32Array(fragments.length * DIMENSIONS);
      const ids = [];

      for (let i = 0; i < fragments.length; i++) {
        const vec = fingerprintVector(fragments[i]);
        vectors.set(vec, i * DIMENSIONS);
        ids.push(vectorId);
        idMap[vectorId] = {
          module: modName,
          file: file.relPath,
          fragmentIndex: i,
          sizeBytes: Buffer.byteLength(fragments[i]),
          hash: createHash('sha256').update(fragments[i]).digest('hex').slice(0, 16),
        };
        vectorId++;
      }

      const result = db.ingestBatch(vectors, ids);
      totalFragments += result.accepted;
    }

    const status = db.status();
    const fileId = db.fileId();
    const segments = db.segments();

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
        package: '@anthropic-ai/claude-code',
        version: version || metricsJson?.version || 'unknown',
        extractedAt: metricsJson?.extractedAt || new Date().toISOString(),
        metrics: {
          bundleSizeBytes: metricsJson?.sizeBytes || 0,
          classes: metricsJson?.classes || 0,
          functions: metricsJson?.functions || 0,
          asyncFunctions: metricsJson?.asyncFunctions || 0,
          arrowFunctions: metricsJson?.arrowFunctions || 0,
        },
      },
      files: jsFiles.map((f) => f.relPath),
      idMap,
      meta: { version, series, package: '@anthropic-ai/claude-code' },
      createdAt: new Date().toISOString(),
    };

    writeFileSync(rvfPath + '.manifest.json', JSON.stringify(manifest, null, 2));
    db.close();

    return { manifest, totalFragments, fileSize: status.fileSize, fileId };
  } catch (err) {
    console.log(`  [!] RVF creation failed: ${err.message}`);
    return null;
  }
}

/**
 * Build RVF containers in a separate rvf/ directory: master + per-category.
 * Source and RVF are kept completely separate.
 */
function buildRvf(sourceDir, rvfDir, version, series) {
  if (!RvfDatabase) {
    console.log('  [!] Skipping RVF (no native backend)');
    return null;
  }

  console.log('  Building RVF containers...');
  mkdirSync(rvfDir, { recursive: true });

  // Load metrics
  let metricsJson = {};
  // Check parent dir for metrics (version dir root)
  const metricsPath = join(sourceDir, 'metrics.json');
  const parentMetricsPath = join(resolve(sourceDir, '..'), 'metrics.json');
  if (existsSync(metricsPath)) {
    metricsJson = JSON.parse(readFileSync(metricsPath, 'utf-8'));
  } else if (existsSync(parentMetricsPath)) {
    metricsJson = JSON.parse(readFileSync(parentMetricsPath, 'utf-8'));
  }

  // Collect all JS files recursively from source
  const allFiles = collectJsFiles(sourceDir);
  if (allFiles.length === 0) return null;

  // Build master RVF
  const masterRvfPath = join(rvfDir, 'master.rvf');
  const masterResult = createRvfFromFiles(allFiles, masterRvfPath, metricsJson, version, series);
  if (masterResult) {
    console.log(`    master.rvf: ${masterResult.totalFragments} vectors, ${(masterResult.fileSize / 1024).toFixed(1)} KB`);
  }

  // Build per-category RVFs
  const subDirs = new Set();
  for (const f of allFiles) {
    const parts = f.relPath.split('/');
    if (parts.length > 1) subDirs.add(parts[0]);
  }

  for (const subDir of subDirs) {
    const subDirPath = join(sourceDir, subDir);
    const subFiles = collectJsFiles(subDirPath, subDir);
    if (subFiles.length === 0) continue;

    const subRvfPath = join(rvfDir, `${subDir}.rvf`);
    const subResult = createRvfFromFiles(subFiles, subRvfPath, metricsJson, version, series);
    if (subResult) {
      console.log(`    ${subDir}.rvf: ${subResult.totalFragments} vectors, ${(subResult.fileSize / 1024).toFixed(1)} KB`);
    }
  }

  return masterResult?.manifest || null;
}

/**
 * Generate README for a version directory.
 */
function generateVersionReadme(verDir, series, version, metrics, manifest) {
  const bundleSize = metrics
    ? `${(metrics.sizeBytes / 1024 / 1024).toFixed(1)}MB`
    : 'unknown';
  const classes = metrics?.classes ?? '?';
  const functions = metrics?.functions ?? '?';
  const modulesCount = metrics?.modules
    ? Object.keys(metrics.modules).length
    : '?';
  const coverage = metrics?.coveragePercent ?? '?';
  const rvfSize = manifest
    ? `${(manifest.fileSizeBytes / 1024).toFixed(1)}KB`
    : 'N/A';
  const rvfVectors = manifest?.totalVectors ?? 'N/A';
  const rvfId = manifest?.fileId
    ? '`' + manifest.fileId.slice(0, 12) + '...`'
    : 'N/A';

  const moduleTable = metrics?.modules
    ? Object.entries(metrics.modules)
        .map(
          ([name, info]) =>
            `| ${name} | ${info.fragments} | ${(info.sizeBytes / 1024).toFixed(1)}KB | ${info.confidence} |`,
        )
        .join('\n')
    : '';

  const readme = `# Claude Code v${version} (${series} series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | ${version} |
| Series | ${series} |
| Bundle size | ${bundleSize} |
| RVF size | ${rvfSize} |
| Vectors | ${rvfVectors} |
| RVF File ID | ${rvfId} |
| Classes | ${classes} |
| Functions | ${functions} |
| Modules | ${modulesCount} |
| Coverage | ${coverage}% |
| Extracted | ${new Date().toISOString()} |

## Source Metrics

| Metric | Value |
|--------|-------|
| Lines | ${metrics?.lines ?? '?'} |
| Async functions | ${metrics?.asyncFunctions ?? '?'} |
| Arrow functions | ${metrics?.arrowFunctions ?? '?'} |
| Class extensions | ${metrics?.classExtensions ?? '?'} |
| const declarations | ${metrics?.constDeclarations ?? '?'} |
| let declarations | ${metrics?.letDeclarations ?? '?'} |
| var declarations | ${metrics?.varDeclarations ?? '?'} |
| imports | ${metrics?.imports ?? '?'} |
| exports | ${metrics?.exports ?? '?'} |
| requires | ${metrics?.requires ?? '?'} |
| await expressions | ${metrics?.awaitExpressions ?? '?'} |
| try blocks | ${metrics?.tryBlocks ?? '?'} |

## Modules

| Module | Fragments | Size | Confidence |
|--------|-----------|------|------------|
${moduleTable}

## Directory Structure

\`\`\`
v${series}.x/
  source/               # Source code only (no .rvf files)
    core/               # agent-loop, context-manager, streaming-handler, session
    tools/              # tool-dispatch
    tools/mcp/          # mcp-client
    permissions/        # permission-system
    ui/                 # commands, command-defs
    config/             # config, env-vars, model-provider
    telemetry/          # telemetry, telemetry-events
    types/              # class-hierarchy, api-endpoints
    uncategorized/      # remaining bundle code
    witness.json        # SHA-256 witness chain

  rvf/                  # RVF containers only (no .js files)
    master.rvf          # All vectors combined
    core.rvf            # Core modules only
    tools.rvf           # Tool modules only
    permissions.rvf     # Permission modules only
    config.rvf          # Configuration modules only
    telemetry.rvf       # Telemetry modules only
    ...

  metrics.json          # Overall metrics
\`\`\`

## RVF Container Details

Each \`.rvf\` file is a binary container with:

- **128-dimensional fingerprint vectors** for each code fragment
- **HNSW index** (M=16, ef_construction=200) for fast similarity search
- **Cosine distance** metric
- **Witness chain** for provenance verification

\`\`\`typescript
import { RvfDatabase } from '@ruvector/rvf';

const db = await RvfDatabase.openReadonly('./rvf/master.rvf');
const results = await db.query(queryVector, 10);
await db.close();
\`\`\`
`;

  writeFileSync(join(verDir, 'README.md'), readme);
}

/**
 * Generate the top-level versions/README.md index.
 */
function generateVersionsIndex(versionsDir, allResults) {
  const rows = allResults
    .map((r) => {
      const bundleSize = r.metrics
        ? `${(r.metrics.sizeBytes / 1024 / 1024).toFixed(1)}MB`
        : '?';
      const rvfSize = r.manifest
        ? `${(r.manifest.fileSizeBytes / 1024).toFixed(1)}KB`
        : 'N/A';
      const vectors = r.manifest?.totalVectors ?? 'N/A';
      const fileId = r.manifest?.fileId
        ? '`' + r.manifest.fileId.slice(0, 12) + '...`'
        : 'N/A';
      const classes = r.metrics?.classes ?? '?';
      const funcs = r.metrics?.functions ?? '?';
      const modules = r.metrics?.modules
        ? Object.keys(r.metrics.modules).length
        : '?';
      return `| ${r.series} | ${r.version} | ${bundleSize} | ${classes} | ${funcs} | ${modules} | ${rvfSize} | ${vectors} | ${fileId} |`;
    })
    .join('\n');

  const readme = `# Claude Code RVF Corpus

Binary RVF containers for every major Claude Code CLI release, with
HNSW-indexed vector embeddings and witness chains for provenance.

## Versions

| Series | Version | Bundle | Classes | Functions | Modules | RVF Size | Vectors | File ID |
|--------|---------|--------|---------|-----------|---------|----------|---------|---------|
${rows}

## Cross-Version Growth

The Claude Code CLI has grown significantly across releases:

${allResults
  .map(
    (r) =>
      `- **v${r.series}.x** (${r.version}): ${r.metrics ? (r.metrics.sizeBytes / 1024 / 1024).toFixed(1) + 'MB' : '?'} bundle, ${r.metrics?.classes ?? '?'} classes, ${r.metrics?.functions ?? '?'} functions`,
  )
  .join('\n')}

## How to Use

\`\`\`bash
# Rebuild all versions from scratch
node scripts/rebuild-all-versions.mjs

# Rebuild only specific series
node scripts/rebuild-all-versions.mjs --series 2.0,2.1

# Or use the shell wrapper
./scripts/claude-code-rvf-corpus.sh
\`\`\`

## Format

Each version directory contains:
- A binary \`.rvf\` container (128-dim cosine-distance HNSW index)
- A \`.manifest.json\` sidecar with vector-to-fragment mapping
- Extracted JavaScript modules in \`source/\`
- \`metrics.json\` with code metrics
- \`witness.json\` with SHA-256 witness chain

Generated by \`scripts/rebuild-all-versions.mjs\` using the decompiler library
at \`npm/packages/ruvector/src/decompiler/\`.
`;

  writeFileSync(join(versionsDir, 'README.md'), readme);
}

/**
 * Rebuild the extracted/ directory for the latest version.
 */
function rebuildExtracted(cliPath, version) {
  console.log('\n[+] Rebuilding extracted/ directory for latest version...');

  // Clean entire extracted/ directory and recreate
  if (existsSync(EXTRACTED_DIR)) {
    rmSync(EXTRACTED_DIR, { recursive: true, force: true });
  }
  const extractedSourceDir = join(EXTRACTED_DIR, 'source');
  const extractedRvfDir = join(EXTRACTED_DIR, 'rvf');
  mkdirSync(extractedSourceDir, { recursive: true });
  mkdirSync(extractedRvfDir, { recursive: true });

  // Read and beautify
  const raw = readFileSync(cliPath, 'utf-8');
  console.log(`  Source: ${(Buffer.byteLength(raw) / 1024 / 1024).toFixed(1)} MB`);

  const beautified = beautify(raw);

  // Split into modules
  const { modules } = splitModules(beautified, { minConfidence: 0 });

  // Write each module into source/<subdir>/
  let totalBytes = 0;
  const moduleResults = {};

  for (const mod of modules) {
    const filePath = getModulePath(extractedSourceDir, mod.name);
    const header = `// ===================================================================\n` +
      `// Module: ${mod.name}\n` +
      `// Source: @anthropic-ai/claude-code@${version}\n` +
      `// Confidence: ${mod.confidence}\n` +
      `// Fragments: ${mod.fragments}\n` +
      `// Extracted: ${new Date().toISOString()}\n` +
      `// ===================================================================\n\n`;
    const content = header + mod.content;

    writeFileSync(filePath, content);
    const sizeBytes = Buffer.byteLength(content);
    totalBytes += sizeBytes;

    const subDir = getModuleDir(mod.name);
    moduleResults[mod.name] = {
      fragments: mod.fragments,
      sizeBytes,
      confidence: mod.confidence,
      directory: subDir,
    };

    console.log(`  source/${subDir}/${mod.name}.js: ${mod.fragments} fragments (${(sizeBytes / 1024).toFixed(1)} KB)`);
  }

  // Write metrics at extracted/ root
  const sourceMetrics = computeMetrics(beautified);
  const metricsData = {
    version,
    package: '@anthropic-ai/claude-code',
    extractedAt: new Date().toISOString(),
    bundleSizeBytes: Buffer.byteLength(raw),
    beautifiedSizeBytes: Buffer.byteLength(beautified),
    capturedBytes: totalBytes,
    coveragePercent: parseFloat(
      ((totalBytes / Buffer.byteLength(beautified)) * 100).toFixed(1),
    ),
    ...sourceMetrics,
    modules: moduleResults,
  };

  writeFileSync(join(EXTRACTED_DIR, 'metrics.json'), JSON.stringify(metricsData, null, 2));

  // Write witness at source/ root
  const witness = buildWitnessChain(raw, modules);
  writeFileSync(join(extractedSourceDir, 'witness.json'), JSON.stringify(witness, null, 2));

  // Build RVF containers in rvf/ directory
  buildRvf(extractedSourceDir, extractedRvfDir, version, '');

  // Generate extracted README
  const readmeContent = `# Extracted Source - Claude Code v${version}

Decompiled source modules from \`@anthropic-ai/claude-code@${version}\`.

## Directory Structure

\`\`\`
extracted/
  source/                   # Source code only (no .rvf files)
    core/                   # Core execution engine
      agent-loop.js         # Main async generator
      context-manager.js    # Token counting and compaction
      streaming-handler.js  # SSE event processing
      session.js            # Session management
    tools/                  # Tool system
      tool-dispatch.js      # Tool registry and routing
      mcp/
        mcp-client.js       # MCP protocol client
    permissions/            # Permission system
      permission-system.js  # Permission checker and sandbox
    ui/                     # User interface
      commands.js           # Slash commands
      command-defs.js       # Command definitions
    config/                 # Configuration
      config.js             # Settings schema
      env-vars.js           # Environment variables
      model-provider.js     # Model selection/routing
    telemetry/              # Observability
      telemetry.js          # OpenTelemetry integration
      telemetry-events.js   # Event definitions
    types/                  # Type info
      class-hierarchy.js    # Class declarations
      api-endpoints.js      # API endpoints
    uncategorized/          # Remaining bundle code
      uncategorized.js

  rvf/                      # RVF containers only (no .js files)
    master.rvf              # All vectors combined
    core.rvf                # Core modules only
    tools.rvf               # Tool modules only
    permissions.rvf         # Permission modules only
    config.rvf              # Configuration modules only
    telemetry.rvf           # Telemetry modules only
    ui.rvf                  # UI modules only
    types.rvf               # Type modules only
    uncategorized.rvf       # Uncategorized modules

  metrics.json              # Overall metrics
\`\`\`

## Metrics

| Metric | Value |
|--------|-------|
| Version | ${version} |
| Bundle size | ${(metricsData.bundleSizeBytes / 1024 / 1024).toFixed(1)} MB |
| Classes | ${metricsData.classes} |
| Functions | ${metricsData.functions} |
| Modules | ${Object.keys(moduleResults).length} |
| Coverage | ${metricsData.coveragePercent}% |
| Extracted | ${metricsData.extractedAt} |

## RVF Containers

Source and RVF files are cleanly separated:
- \`rvf/master.rvf\` - Master RVF (all modules, all vectors)
- \`rvf/core.rvf\` - Core execution modules only
- \`rvf/tools.rvf\` - Tool system modules only
- \`rvf/permissions.rvf\` - Permission modules only
- \`rvf/config.rvf\` - Configuration modules only
- \`rvf/telemetry.rvf\` - Telemetry modules only

Each RVF container has an accompanying \`.manifest.json\` sidecar.
`;

  writeFileSync(join(EXTRACTED_DIR, 'README.md'), readmeContent);

  console.log(`  Total captured: ${(totalBytes / 1024).toFixed(1)} KB (${metricsData.coveragePercent}% coverage)`);
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

async function main() {
  console.log('===========================================');
  console.log('Claude Code Version Corpus - Full Rebuild');
  console.log('===========================================\n');

  mkdirSync(TMP_BASE, { recursive: true });

  // Step 1: Clean existing version directories
  console.log('[+] Cleaning existing version directories...');
  if (existsSync(VERSIONS_DIR)) {
    for (const d of readdirSync(VERSIONS_DIR)) {
      if (d.startsWith('v') && d.endsWith('.x')) {
        const p = join(VERSIONS_DIR, d);
        console.log(`  Removing ${d}/`);
        rmSync(p, { recursive: true, force: true });
      }
    }
  }
  mkdirSync(VERSIONS_DIR, { recursive: true });

  // Step 2: Get version groups
  const groups = getVersionGroups();
  console.log(`[+] Found ${groups.length} major.minor series\n`);

  // Apply filter
  const filtered = filterSeries
    ? groups.filter((g) => filterSeries.includes(g.series))
    : groups;

  if (filterSeries) {
    console.log(`[+] Filtered to ${filtered.length} series: ${filterSeries.join(', ')}\n`);
  }

  // Step 3: Process each version
  const allResults = [];
  let processed = 0;
  let failed = 0;
  let latestCliPath = null;
  let latestVersion = null;

  for (const { series, version } of filtered) {
    console.log(`\n[+] Processing v${series}.x (latest: ${version})`);
    console.log('─'.repeat(50));

    const verDir = join(VERSIONS_DIR, `v${series}.x`);
    mkdirSync(verDir, { recursive: true });

    // Download
    const cliPath = downloadVersion(version);
    if (!cliPath) {
      console.log(`  [!] Skipping ${version} (download failed)`);
      failed++;
      continue;
    }

    // Track the latest version for extracted/ rebuild
    latestCliPath = cliPath;
    latestVersion = version;

    // Decompile
    const { metricsData, witness, modules } = decompileVersion(
      cliPath, verDir, series, version,
    );

    // Build RVF (separate rvf/ directory)
    const rvfDir = join(verDir, 'rvf');
    const manifest = buildRvf(
      join(verDir, 'source'), rvfDir, version, series,
    );

    // Generate README
    generateVersionReadme(verDir, series, version, metricsData, manifest);

    allResults.push({ series, version, metrics: metricsData, manifest });
    processed++;
    console.log(`  Done (${processed}/${filtered.length})`);
  }

  // Step 4: Generate versions index
  console.log('\n[+] Generating versions index...');
  generateVersionsIndex(VERSIONS_DIR, allResults);

  // Step 5: Rebuild extracted/ from latest version
  if (latestCliPath && existsSync(latestCliPath)) {
    rebuildExtracted(latestCliPath, latestVersion);
  } else {
    console.log('\n[!] No latest CLI path available, skipping extracted/ rebuild');
  }

  // Step 6: Clean up tmp
  try {
    rmSync(TMP_BASE, { recursive: true, force: true });
  } catch {}

  // Summary
  console.log('\n===========================================');
  console.log('Rebuild complete');
  console.log('===========================================');
  console.log(`  Versions processed: ${processed}`);
  console.log(`  Versions failed: ${failed}`);
  console.log(`  Output: ${VERSIONS_DIR}/`);
  if (latestVersion) {
    console.log(`  Latest extracted: ${EXTRACTED_DIR}/ (v${latestVersion})`);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
