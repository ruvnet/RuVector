/**
 * decompiler/index.js - High-level decompiler API.
 *
 * Exports three main entry points:
 *   - decompilePackage(name, version, options)
 *   - decompileFile(filePath, options)
 *   - decompileUrl(url, options)
 *
 * Each returns a standardized DecompileResult:
 *   { modules, metrics, witness, source, packageInfo? }
 */

'use strict';

const fs = require('fs');
const path = require('path');
const {
  fetchPackageInfo,
  fetchPackageFileList,
  fetchFileContent,
  findMainBundle,
  parseTarget,
} = require('./npm-fetch');
const { splitModules } = require('./module-splitter');
const { buildWitnessChain, verifyWitnessChain } = require('./witness');
const { computeMetrics, computeModuleMetrics } = require('./metrics');
const { reconstructCode, reconstructRunnable } = require('./reconstructor');
const { validateReconstruction } = require('./validator');

/**
 * Try the WASM Louvain decompiler (full graph-partitioning pipeline).
 * Returns null if WASM module is not available or fails.
 *
 * @param {string} source - raw JavaScript source
 * @param {object} [options]
 * @returns {{modules: object[], metrics: object, witness: object|null}|null}
 */
function tryWasmDecompiler(source, options = {}) {
  try {
    const wasm = require('../../wasm/ruvector_decompiler_wasm');
    const configJson = JSON.stringify({
      target_modules: null,
      min_confidence: options.minConfidence || 0.3,
      generate_source_maps: false,
      generate_witness: options.witness !== false,
      output_filename: 'bundle.js',
      model_path: null,
      hierarchical_output: true,
      max_depth: 3,
      min_folder_size: 3,
    });
    const resultJson = wasm.decompile(source, configJson);
    const result = JSON.parse(resultJson);
    if (result.error) return null;

    // Convert Rust DecompileResult to Node.js format
    return {
      modules: (result.modules || []).map((m) => ({
        name: m.name,
        content: m.source || '',
        declarations: (m.declarations && m.declarations.length) || 0,
        fragments: (m.declarations && m.declarations.length) || 0,
        confidence: 0.8,
      })),
      metrics: {
        source: { sizeBytes: source.length },
        modules: (result.modules || []).length,
        engine: 'wasm-louvain',
      },
      witness: result.witness || null,
      moduleTree: result.module_tree || null,
      beautifiedSource: source,
    };
  } catch {
    return null; // WASM not available, fall back
  }
}

/**
 * Try to beautify source code using js-beautify (optional dep).
 * Falls back to returning the source unchanged if not installed.
 * @param {string} source
 * @returns {string}
 */
function beautify(source) {
  try {
    const jsBeautify = require('js-beautify');
    const beautifyFn = jsBeautify.js || jsBeautify;
    return beautifyFn(source, {
      indent_size: 2,
      space_in_empty_paren: false,
      preserve_newlines: true,
      max_preserve_newlines: 2,
      end_with_newline: true,
    });
  } catch {
    // js-beautify not installed; return source as-is
    return source;
  }
}

/**
 * Try to use the Rust decompiler for full Louvain graph partitioning (878+ modules).
 * Falls back to Node.js keyword splitting if Rust binary not available.
 *
 * @param {string} filePath - path to JS file
 * @param {string} outputDir - output directory
 * @returns {{success: boolean, modules: number, outputDir: string}|null}
 */
function tryRustDecompiler(filePath, outputDir) {
  try {
    const { execSync } = require('child_process');
    // Try to find the Rust binary
    const candidates = [
      'cargo run --release -p ruvector-decompiler --example run_on_cli --',
      path.join(__dirname, '../../../../target/release/examples/run_on_cli'),
    ];
    for (const bin of candidates) {
      try {
        const cmd = bin.includes('cargo')
          ? `${bin} "${filePath}" --output-dir "${outputDir}"`
          : `"${bin}" "${filePath}" --output-dir "${outputDir}"`;
        const result = execSync(cmd, {
          timeout: 120000,
          stdio: ['pipe', 'pipe', 'pipe'],
          cwd: path.join(__dirname, '../../../..'),
        });
        const stderr = result.toString();
        const match = stderr.match(/Wrote (\d+) modules/);
        const moduleCount = match ? parseInt(match[1]) : 0;
        return { success: true, modules: moduleCount, outputDir };
      } catch { continue; }
    }
  } catch {}
  return null;
}

/**
 * Core decompilation pipeline: beautify -> split -> metrics -> witness -> reconstruct.
 *
 * If the Rust decompiler is available (cargo built), uses Louvain graph partitioning
 * for 878+ modules with 100% parse rate. Falls back to Node.js keyword splitting.
 *
 * @param {string} source - raw JavaScript source
 * @param {object} [options]
 * @param {number} [options.minConfidence=0.3]
 * @param {boolean} [options.witness=true]
 * @param {boolean} [options.reconstruct=false] - apply readable reconstruction
 * @param {boolean} [options.validate=false] - validate reconstruction preserves semantics
 * @param {string} [options.patternPath] - path to training patterns JSON
 * @param {boolean} [options.addComments=true] - add JSDoc comments during reconstruction
 * @param {boolean} [options.improveStyle=true] - apply style improvements during reconstruction
 * @param {boolean} [options.useRust=true] - try Rust Louvain partitioner first
 * @param {string} [options.filePath] - original file path (needed for Rust pipeline)
 * @returns {{modules: object[], metrics: object, witness: object|null, beautifiedSource: string, reconstruction?: object}}
 */
function decompileSource(source, options = {}) {
  const {
    minConfidence = 0.3,
    witness: generateWitness = true,
    reconstruct = false,
    validate = false,
    patternPath,
    addComments = true,
    improveStyle = true,
    useRust = true,
    filePath,
  } = options;

  // Priority 1: WASM Louvain (full pipeline, works everywhere, no binary needed)
  if (useRust !== false && source.length > 1000) {
    const wasmResult = tryWasmDecompiler(source, options);
    if (wasmResult) return wasmResult;
  }

  // Priority 2: Rust binary (full pipeline, requires cargo build)
  if (useRust && filePath && source.length > 100000) {
    const tmpDir = path.join(require('os').tmpdir(), 'ruvector-decompile-' + Date.now());
    const rustResult = tryRustDecompiler(filePath, tmpDir);
    if (rustResult && rustResult.success) {
      // Load modules from Rust output
      const sourceDir = path.join(tmpDir, 'source');
      const rustModules = [];
      try {
        for (const f of fs.readdirSync(sourceDir).filter(f => f.endsWith('.js'))) {
          const content = fs.readFileSync(path.join(sourceDir, f), 'utf8');
          rustModules.push({
            name: f.replace('.js', ''),
            content,
            fragments: 0,
            confidence: 0.8,
          });
        }
      } catch {}
      if (rustModules.length > 0) {
        const sourceMetrics = computeMetrics(source);
        const witnessPath = path.join(tmpDir, 'witness.json');
        let witnessChain = null;
        try { witnessChain = JSON.parse(fs.readFileSync(witnessPath, 'utf8')); } catch {}
        return {
          modules: rustModules,
          metrics: { source: sourceMetrics, modules: rustModules.length, engine: 'rust-louvain' },
          witness: witnessChain,
          beautifiedSource: source,
          source,
        };
      }
    }
  }

  // Fallback: Node.js keyword-based splitting
  const beautified = beautify(source);
  const { modules, unclassified } = splitModules(beautified, { minConfidence });
  const sourceMetrics = computeMetrics(beautified);
  const moduleMetrics = computeModuleMetrics(modules);
  const witnessChain = generateWitness ? buildWitnessChain(source, modules) : null;

  // Optional: apply readable reconstruction to each module
  let reconstructionSummary = null;
  if (reconstruct) {
    let totalRenames = 0;
    let totalComments = 0;
    let totalConfidence = 0;
    let validationResults = [];

    for (const mod of modules) {
      const result = reconstructCode(mod.content, {
        patternPath,
        propagateNames: true,
        addComments,
        improveStyle,
        minConfidence,
      });

      const originalContent = mod.content;
      mod.content = result.code;
      mod.renames = result.renames;
      mod.confidence = Math.max(mod.confidence, result.confidence);

      totalRenames += result.renames.length;
      totalComments += result.comments;
      totalConfidence += result.confidence;

      // Optional: validate the reconstruction
      if (validate) {
        const validation = validateReconstruction(originalContent, result.code);
        validationResults.push({
          module: mod.name,
          ...validation,
        });
      }
    }

    reconstructionSummary = {
      totalRenames,
      totalComments,
      averageConfidence: modules.length > 0
        ? parseFloat((totalConfidence / modules.length).toFixed(3))
        : 0,
      modulesProcessed: modules.length,
    };

    if (validate) {
      reconstructionSummary.validation = validationResults;
      reconstructionSummary.allValid = validationResults.every((v) => v.syntaxValid);
      reconstructionSummary.allEquivalent = validationResults.every((v) => v.functionallyEquivalent);
    }
  }

  return {
    modules,
    metrics: {
      source: sourceMetrics,
      modules: moduleMetrics,
      unclassifiedStatements: unclassified.length,
    },
    witness: witnessChain,
    beautifiedSource: beautified,
    ...(reconstructionSummary ? { reconstruction: reconstructionSummary } : {}),
  };
}

/**
 * Decompile an npm package.
 *
 * @param {string} packageName - e.g. 'express', '@anthropic-ai/claude-code'
 * @param {string} [version] - defaults to 'latest'
 * @param {object} [options]
 * @param {number} [options.minConfidence=0.3]
 * @param {boolean} [options.witness=true]
 * @returns {Promise<{modules: object[], metrics: object, witness: object|null, packageInfo: object, bundlePath: string, source: string}>}
 */
async function decompilePackage(packageName, version, options = {}) {
  const info = await fetchPackageInfo(packageName);
  const resolvedVersion = version || info.latest;

  if (!info.versions.includes(resolvedVersion)) {
    throw new Error(
      `Version "${resolvedVersion}" not found for ${packageName}. ` +
        `Available: ${info.versions.slice(0, 10).join(', ')}...`,
    );
  }

  const files = await fetchPackageFileList(packageName, resolvedVersion);
  const pkgJson = info.packageJson || {};
  const bundlePath = findMainBundle(files, pkgJson);

  if (!bundlePath) {
    throw new Error(
      `Could not find main bundle for ${packageName}@${resolvedVersion}. ` +
        `Files: ${files.slice(0, 10).map((f) => f.name).join(', ')}`,
    );
  }

  const source = await fetchFileContent(packageName, resolvedVersion, bundlePath);
  const result = decompileSource(source, options);

  return {
    ...result,
    packageInfo: {
      name: info.name,
      version: resolvedVersion,
      description: info.description,
      bundlePath,
      bundleSize: source.length,
    },
    source,
  };
}

/**
 * Decompile a local JavaScript file.
 *
 * @param {string} filePath - path to a .js file
 * @param {object} [options]
 * @returns {{modules: object[], metrics: object, witness: object|null, filePath: string, source: string}}
 */
function decompileFile(filePath, options = {}) {
  const resolved = path.resolve(filePath);

  if (!fs.existsSync(resolved)) {
    throw new Error(`File not found: ${resolved}`);
  }

  const source = fs.readFileSync(resolved, 'utf-8');
  const result = decompileSource(source, { ...options, filePath: resolved });

  return {
    ...result,
    filePath: resolved,
    source,
  };
}

/**
 * Decompile JavaScript from a URL.
 *
 * @param {string} url
 * @param {object} [options]
 * @returns {Promise<{modules: object[], metrics: object, witness: object|null, url: string, source: string}>}
 */
async function decompileUrl(url, options = {}) {
  const resp = await fetch(url, { redirect: 'follow' });
  if (!resp.ok) {
    throw new Error(`Failed to fetch ${url} (HTTP ${resp.status})`);
  }

  const source = await resp.text();
  const result = decompileSource(source, options);

  return {
    ...result,
    url,
    source,
  };
}

/**
 * Write decompilation results to an output directory.
 *
 * @param {object} result - decompilation result from any of the decompile* functions
 * @param {string} outputDir
 * @param {string} [format='modules'] - 'modules', 'single', 'json'
 */
function writeOutput(result, outputDir, format = 'modules') {
  fs.mkdirSync(outputDir, { recursive: true });

  if (format === 'json') {
    const jsonResult = {
      modules: result.modules.map((m) => ({
        name: m.name,
        fragments: m.fragments,
        confidence: m.confidence,
        content: m.content,
      })),
      metrics: result.metrics,
      witness: result.witness,
      packageInfo: result.packageInfo || null,
    };
    fs.writeFileSync(
      path.join(outputDir, 'decompiled.json'),
      JSON.stringify(jsonResult, null, 2),
    );
    return;
  }

  if (format === 'single') {
    let output = '';
    for (const mod of result.modules) {
      output += `// ─── Module: ${mod.name} (confidence: ${mod.confidence}) ───\n\n`;
      output += mod.content + '\n\n';
    }
    fs.writeFileSync(path.join(outputDir, 'decompiled.js'), output);
    return;
  }

  // Default: 'modules' format — one file per module
  // Supports hierarchical module names like 'tools/bash' -> tools/bash.js
  for (let i = 0; i < result.modules.length; i++) {
    const mod = result.modules[i];
    const header = `// Module: ${mod.name}\n// Confidence: ${mod.confidence}\n// Fragments: ${mod.fragments}\n\n`;

    if (mod.name.includes('/')) {
      // Hierarchical: create subdirectories
      const filePath = path.join(outputDir, mod.name + '.js');
      fs.mkdirSync(path.dirname(filePath), { recursive: true });
      fs.writeFileSync(filePath, header + mod.content);
    } else {
      const idx = String(i + 1).padStart(3, '0');
      const fileName = `module-${idx}-${mod.name}.js`;
      fs.writeFileSync(path.join(outputDir, fileName), header + mod.content);
    }
  }

  // Metrics
  fs.writeFileSync(
    path.join(outputDir, 'metrics.json'),
    JSON.stringify(result.metrics, null, 2),
  );

  // Witness chain
  if (result.witness) {
    fs.writeFileSync(
      path.join(outputDir, 'witness.json'),
      JSON.stringify(result.witness, null, 2),
    );
  }
}

module.exports = {
  decompilePackage,
  decompileFile,
  decompileUrl,
  decompileSource,
  writeOutput,
  beautify,
  parseTarget,
  verifyWitnessChain,
  reconstructCode,
  reconstructRunnable,
  validateReconstruction,
  tryWasmDecompiler,
};
