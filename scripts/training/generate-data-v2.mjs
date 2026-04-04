#!/usr/bin/env node
/**
 * Generate expanded training data for JS deobfuscation model (v2).
 *
 * Sources:
 *   1. Existing training-data.jsonl (merge)
 *   2. Real JS files from node_modules (identifier extraction)
 *   3. Synthetic augmentation with context diversity
 *
 * Targets 15,000+ unique pairs for SOTA training.
 *
 * Usage:
 *   node scripts/training/generate-data-v2.mjs [--output training-data-v2.jsonl]
 */

import { readFileSync, writeFileSync, readdirSync, statSync, existsSync } from "fs";
import { join, resolve, basename } from "path";
import { parseArgs } from "util";

const { values: args } = parseArgs({
  options: {
    output: { type: "string", default: "training-data-v2.jsonl" },
    help: { type: "boolean", short: "h", default: false },
  },
});

if (args.help) {
  console.log("Usage: generate-data-v2.mjs [--output FILE]");
  process.exit(0);
}

const OUTPUT_PATH = resolve(args.output);
const ROOT = resolve(import.meta.dirname, "../..");

/** @type {Map<string, object>} key -> pair object, for dedup */
const pairMap = new Map();

function addPair(minified, original, contextStrings, properties, kind) {
  if (!minified || !original || original.length <= 1) return;
  // Skip if original looks minified itself
  if (original.length <= 2 && !/^[A-Z]/.test(original)) return;
  const key = `${minified}|${original}`;
  if (pairMap.has(key)) return;
  pairMap.set(key, {
    minified,
    original,
    context_strings: contextStrings.slice(0, 8),
    properties: properties.slice(0, 8),
    kind,
  });
}

// ---------------------------------------------------------------------------
// Source 1: Merge existing training data
// ---------------------------------------------------------------------------

function mergeExisting() {
  const existingPath = join(ROOT, "training-data.jsonl");
  if (!existsSync(existingPath)) {
    console.log("  [existing] no training-data.jsonl found, skipping");
    return 0;
  }
  const lines = readFileSync(existingPath, "utf8").trim().split("\n");
  let count = 0;
  for (const line of lines) {
    if (!line.trim()) continue;
    try {
      const obj = JSON.parse(line);
      addPair(
        obj.minified,
        obj.original,
        obj.context_strings || [],
        obj.properties || [],
        obj.kind || "var"
      );
      count++;
    } catch { /* skip bad lines */ }
  }
  console.log(`  [existing] merged ${count} pairs`);
  return count;
}

// ---------------------------------------------------------------------------
// Source 2: Extract identifiers from real JS files in node_modules
// ---------------------------------------------------------------------------

/** Walk directory tree, collect .js files up to maxDepth */
function collectJsFiles(dir, maxDepth = 3, depth = 0) {
  const files = [];
  if (depth > maxDepth) return files;
  let entries;
  try { entries = readdirSync(dir); } catch { return files; }
  for (const entry of entries) {
    if (entry === "node_modules" && depth > 0) continue;
    if (entry.startsWith(".")) continue;
    const full = join(dir, entry);
    let stat;
    try { stat = statSync(full); } catch { continue; }
    if (stat.isDirectory()) {
      files.push(...collectJsFiles(full, maxDepth, depth + 1));
    } else if (entry.endsWith(".js") && stat.size > 1000 && stat.size < 200000) {
      files.push(full);
    }
  }
  return files;
}

/**
 * Extract identifiers from a JS source file using regex patterns.
 * Returns array of { name, kind, nearbyTokens }
 */
function extractIdentifiers(source) {
  const results = [];
  const seen = new Set();

  // Pattern: function declarations
  const funcDeclRe = /\bfunction\s+([a-zA-Z_$][a-zA-Z0-9_$]{2,})\s*\(/g;
  let m;
  while ((m = funcDeclRe.exec(source)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name: m[1], kind: "function", ctx });
    }
  }

  // Pattern: const/let/var declarations with meaningful names
  const varDeclRe = /\b(?:const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]{2,})\s*=/g;
  while ((m = varDeclRe.exec(source)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name: m[1], kind: "var", ctx });
    }
  }

  // Pattern: class declarations
  const classDeclRe = /\bclass\s+([a-zA-Z_$][a-zA-Z0-9_$]{2,})\b/g;
  while ((m = classDeclRe.exec(source)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name: m[1], kind: "class", ctx });
    }
  }

  // Pattern: method definitions (object/class methods)
  const methodRe = /\b([a-zA-Z_$][a-zA-Z0-9_$]{2,})\s*\([^)]*\)\s*\{/g;
  while ((m = methodRe.exec(source)) !== null) {
    const name = m[1];
    if (!seen.has(name) && !SKIP_NAMES.has(name)) {
      seen.add(name);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name, kind: "function", ctx });
    }
  }

  // Pattern: exports.X = or module.exports.X =
  const exportsRe = /(?:exports|module\.exports)\.([a-zA-Z_$][a-zA-Z0-9_$]{2,})\s*=/g;
  while ((m = exportsRe.exec(source)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name: m[1], kind: "var", ctx });
    }
  }

  // Pattern: prototype methods
  const protoRe = /\.prototype\.([a-zA-Z_$][a-zA-Z0-9_$]{2,})\s*=/g;
  while ((m = protoRe.exec(source)) !== null) {
    if (!seen.has(m[1])) {
      seen.add(m[1]);
      const ctx = extractNearbyContext(source, m.index, 200);
      results.push({ name: m[1], kind: "function", ctx });
    }
  }

  return results;
}

const SKIP_NAMES = new Set([
  "if", "else", "for", "while", "do", "switch", "case", "break",
  "continue", "return", "try", "catch", "finally", "throw", "new",
  "delete", "typeof", "void", "instanceof", "in", "of", "with",
  "this", "super", "true", "false", "null", "undefined", "NaN",
  "Infinity", "arguments", "eval", "constructor", "prototype",
  "use", "strict", "exports", "module", "require",
]);

/**
 * Extract nearby context tokens around a match position.
 */
function extractNearbyContext(source, pos, window) {
  const start = Math.max(0, pos - window);
  const end = Math.min(source.length, pos + window);
  const snippet = source.slice(start, end);

  // Extract string literals as context
  const strings = [];
  const strRe = /["']([a-zA-Z][a-zA-Z0-9_.-]{2,})["']/g;
  let m;
  while ((m = strRe.exec(snippet)) !== null) {
    if (!SKIP_NAMES.has(m[1]) && m[1].length < 30) {
      strings.push(m[1]);
    }
  }

  // Extract property accesses as context
  const propRe = /\.([a-zA-Z_$][a-zA-Z0-9_$]{2,})/g;
  while ((m = propRe.exec(snippet)) !== null) {
    if (!SKIP_NAMES.has(m[1]) && m[1].length < 25) {
      strings.push(m[1]);
    }
  }

  // Deduplicate and limit
  return [...new Set(strings)].slice(0, 10);
}

/**
 * Extract property accesses for a given identifier from source.
 */
function extractProperties(source, name) {
  const props = new Set();
  // Look for name.property patterns
  const re = new RegExp(`\\b${escapeRegex(name)}\\.([a-zA-Z_$][a-zA-Z0-9_$]{1,})`, "g");
  let m;
  while ((m = re.exec(source)) !== null) {
    if (m[1].length < 25) props.add(m[1]);
  }
  return [...props].slice(0, 8);
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// Minifier name generators
const MINIFIER_STYLES = [
  (i) => String.fromCharCode(97 + (i % 26)),
  (i) => String.fromCharCode(97 + (i % 26)) + "$",
  (i) => "_" + String.fromCharCode(97 + (i % 26)),
  (i) => "_0x" + (0x1a2b + i).toString(16),
  (i) => String.fromCharCode(97 + (i % 26)) + (i % 10).toString(),
  (i) => "__" + String.fromCharCode(97 + (i % 26)),
  (i) => "$" + String.fromCharCode(97 + (i % 26)),
  (i) => String.fromCharCode(65 + (i % 26)),
  (i) => {
    const a = String.fromCharCode(97 + (i % 26));
    const b = String.fromCharCode(97 + ((i + 1) % 26));
    return a + b;
  },
  (i) => "$" + (i % 100).toString(),
  (i) => "_" + (i % 100).toString(),
  (i) => "t" + i,
  (i) => "e$" + String.fromCharCode(97 + (i % 26)),
  (i) => "n" + (i % 100),
  (i) => "r" + String.fromCharCode(97 + (i % 26)),
  (i) => String.fromCharCode(97 + (i % 26)) + String.fromCharCode(97 + ((i * 7) % 26)),
];

function extractFromNodeModules() {
  const nmDir = join(ROOT, "node_modules");
  if (!existsSync(nmDir)) {
    console.log("  [node_modules] directory not found");
    return 0;
  }

  const jsFiles = collectJsFiles(nmDir, 4);
  console.log(`  [node_modules] found ${jsFiles.length} JS files to scan`);

  let totalExtracted = 0;
  let fileIdx = 0;

  for (const file of jsFiles) {
    let source;
    try { source = readFileSync(file, "utf8"); } catch { continue; }

    // Skip minified files (low ratio of newlines to content)
    const lineCount = source.split("\n").length;
    if (lineCount < 10 && source.length > 5000) continue;

    const identifiers = extractIdentifiers(source);
    if (identifiers.length === 0) continue;

    for (let i = 0; i < identifiers.length; i++) {
      const { name, kind, ctx } = identifiers[i];
      if (name.length < 3 || SKIP_NAMES.has(name)) continue;

      const properties = extractProperties(source, name);

      // Generate multiple minified variants per identifier
      const numVariants = Math.min(4, MINIFIER_STYLES.length);
      for (let v = 0; v < numVariants; v++) {
        const styleIdx = (fileIdx + i + v) % MINIFIER_STYLES.length;
        const minified = MINIFIER_STYLES[styleIdx](fileIdx + i);

        // Vary context slightly for each variant
        const contextVariant = varySyntheticContext(ctx, v);
        addPair(minified, name, contextVariant, properties, kind);
        totalExtracted++;
      }
    }
    fileIdx++;
  }

  console.log(`  [node_modules] extracted ${totalExtracted} pairs`);
  return totalExtracted;
}

// ---------------------------------------------------------------------------
// Source 3: Augmentation -- camelCase splitting + semantic context
// ---------------------------------------------------------------------------

/** Split camelCase/PascalCase into tokens */
function splitCamelCase(name) {
  return name
    .replace(/([A-Z])/g, " $1")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 1);
}

/** Generate semantic context from the name itself */
function generateSemanticContext(name) {
  const tokens = splitCamelCase(name);
  const semantic = [];

  // Add the camelCase tokens as context hints
  semantic.push(...tokens.slice(0, 4));

  // Add type hints based on common prefixes/suffixes
  if (/^is[A-Z]/.test(name)) semantic.push("boolean", "check");
  if (/^has[A-Z]/.test(name)) semantic.push("boolean", "exists");
  if (/^get[A-Z]/.test(name)) semantic.push("getter", "return");
  if (/^set[A-Z]/.test(name)) semantic.push("setter", "assign");
  if (/^on[A-Z]/.test(name)) semantic.push("event", "handler");
  if (/^handle[A-Z]/.test(name)) semantic.push("event", "callback");
  if (/^create[A-Z]/.test(name)) semantic.push("factory", "new");
  if (/^parse[A-Z]/.test(name)) semantic.push("parse", "input");
  if (/^format[A-Z]/.test(name)) semantic.push("format", "output");
  if (/^validate[A-Z]/.test(name)) semantic.push("validate", "check");
  if (/^render[A-Z]/.test(name)) semantic.push("render", "display");
  if (/^fetch[A-Z]/.test(name)) semantic.push("async", "request");
  if (/^load[A-Z]/.test(name)) semantic.push("async", "data");
  if (/^save[A-Z]/.test(name)) semantic.push("persist", "store");
  if (/^delete[A-Z]/.test(name)) semantic.push("remove", "destroy");
  if (/^update[A-Z]/.test(name)) semantic.push("modify", "change");
  if (/^init/.test(name)) semantic.push("initialize", "setup");
  if (/^process/.test(name)) semantic.push("transform", "pipeline");

  // Suffix-based hints
  if (/Error$/.test(name)) semantic.push("error", "exception");
  if (/Handler$/.test(name)) semantic.push("handler", "callback");
  if (/Manager$/.test(name)) semantic.push("manager", "lifecycle");
  if (/Service$/.test(name)) semantic.push("service", "business");
  if (/Controller$/.test(name)) semantic.push("controller", "http");
  if (/Factory$/.test(name)) semantic.push("factory", "create");
  if (/Builder$/.test(name)) semantic.push("builder", "construct");
  if (/Adapter$/.test(name)) semantic.push("adapter", "convert");
  if (/Provider$/.test(name)) semantic.push("provider", "inject");
  if (/Listener$/.test(name)) semantic.push("listener", "event");
  if (/Config$/.test(name)) semantic.push("config", "settings");
  if (/Options$/.test(name)) semantic.push("options", "settings");
  if (/Result$/.test(name)) semantic.push("result", "output");
  if (/Callback$/.test(name)) semantic.push("callback", "async");

  return [...new Set(semantic)].slice(0, 8);
}

/**
 * Vary context slightly for training diversity.
 */
function varySyntheticContext(ctx, variant) {
  if (!ctx || ctx.length === 0) return ["unknown"];
  switch (variant % 5) {
    case 0: return ctx;
    case 1: return ctx.length > 2 ? [...ctx.slice(1), ctx[0]] : ctx;
    case 2: return ctx.slice(0, Math.max(2, Math.ceil(ctx.length / 2)));
    case 3: return [...ctx, "prototype", "constructor"].slice(0, 8);
    case 4: return [...ctx.slice(0, 3), "undefined", "null"].slice(0, 8);
    default: return ctx;
  }
}

/**
 * Generate augmented pairs by cross-version simulation.
 */
function generateCrossVersionAugmentation() {
  const originals = new Map();
  for (const [, pair] of pairMap) {
    if (!originals.has(pair.original)) {
      originals.set(pair.original, pair);
    }
  }

  let augmented = 0;
  const allOriginals = [...originals.entries()];

  for (const [originalName, basePair] of allOriginals) {
    // Generate 2-3 extra "version" variants
    const versions = 2 + Math.floor(Math.random() * 2);
    for (let v = 0; v < versions; v++) {
      const minified = randomMinifiedName();
      const key = `${minified}|${originalName}`;
      if (pairMap.has(key)) continue;

      // Vary context
      const ctx = varySyntheticContext(basePair.context_strings, v);
      addPair(minified, originalName, ctx, basePair.properties, basePair.kind);
      augmented++;
    }
  }

  console.log(`  [cross-version] augmented ${augmented} pairs`);
  return augmented;
}

function randomMinifiedName() {
  const styles = [
    () => String.fromCharCode(97 + rand(26)) + rand(100),
    () => "_0x" + rand(0xffff).toString(16),
    () => String.fromCharCode(97 + rand(26)) + String.fromCharCode(97 + rand(26)),
    () => "$" + String.fromCharCode(97 + rand(26)),
    () => "t" + rand(200),
    () => "n" + rand(100),
    () => "_" + rand(200),
    () => String.fromCharCode(97 + rand(26)) + String.fromCharCode(97 + rand(26)) + rand(10),
  ];
  return styles[rand(styles.length)]();
}

function rand(max) { return Math.floor(Math.random() * max); }

// ---------------------------------------------------------------------------
// Source 4: Additional synthetic names for coverage
// ---------------------------------------------------------------------------

function generateAdditionalSynthetic() {
  // Common web/Node.js identifiers not likely in node_modules source
  const EXTRA_NAMES = {
    function: [
      // Webpack/bundler internals
      "__webpack_require__", "__webpack_modules__", "__webpack_exports__",
      // React internals
      "createElement", "cloneElement", "createRef", "forwardRef",
      "memo", "lazy", "Suspense", "Fragment",
      "useId", "useSyncExternalStore", "useInsertionEffect",
      // Next.js patterns
      "getServerSideProps", "getStaticProps", "getStaticPaths",
      "generateMetadata", "generateStaticParams",
      // Express patterns
      "createApplication", "createMiddleware", "createRoute",
      "useRouter", "useParams", "useSearchParams",
      // Testing
      "beforeEach", "afterEach", "beforeAll", "afterAll",
      "spyOn", "mockImplementation", "mockReturnValue",
      // Utilities
      "cloneDeep", "mergeWith", "assignIn", "defaultsDeep",
      "flattenDeep", "uniqBy", "groupBy", "sortBy", "orderBy",
      "pickBy", "omitBy", "mapKeys", "mapValues",
      // Crypto/Security
      "createHash", "createCipher", "createDecipher", "createSign",
      "randomBytes", "scrypt", "pbkdf2",
      // Stream
      "createReadStream", "createWriteStream", "pipeline", "finished",
      "Transform", "Readable", "Writable", "Duplex", "PassThrough",
    ],
    class: [
      "AbortController", "AbortSignal", "TextEncoder", "TextDecoder",
      "URLSearchParams", "FormData", "Headers", "ReadableStream",
      "WritableStream", "TransformStream", "BroadcastChannel",
      "IntersectionObserver", "MutationObserver", "ResizeObserver",
      "PerformanceObserver", "MessageChannel", "MessagePort",
      "WeakRef", "FinalizationRegistry", "SharedArrayBuffer",
      // Framework classes
      "EventTarget", "CustomEvent", "DOMParser", "XMLSerializer",
      "WebSocket", "Worker", "ServiceWorker", "SharedWorker",
    ],
    var: [
      // Common config keys
      "baseURL", "timeout", "maxRedirects", "maxContentLength",
      "validateStatus", "transformRequest", "transformResponse",
      "paramsSerializer", "withCredentials", "responseEncoding",
      // State patterns
      "initialState", "rootReducer", "rootSaga", "rootEpic",
      "storeEnhancers", "middlewares", "devTools",
      // Build tools
      "webpackConfig", "rollupConfig", "viteConfig", "babelConfig",
      "tsConfig", "eslintConfig", "prettierConfig",
      // Environment
      "NODE_ENV", "API_URL", "BASE_PATH", "PUBLIC_URL",
    ],
  };

  let count = 0;
  for (const [kind, names] of Object.entries(EXTRA_NAMES)) {
    for (let i = 0; i < names.length; i++) {
      const original = names[i];
      const semanticCtx = generateSemanticContext(original);
      const props = kind === "function"
        ? ["length", "name", "call", "apply", "bind"]
        : kind === "class"
          ? ["prototype", "constructor", "name"]
          : ["toString", "valueOf"];

      // 4 minified variants per name
      for (let v = 0; v < 4; v++) {
        const styleIdx = (i + v) % MINIFIER_STYLES.length;
        const minified = MINIFIER_STYLES[styleIdx](i);
        const ctx = varySyntheticContext(semanticCtx, v);
        addPair(minified, original, ctx, props, kind);
        count++;
      }
    }
  }

  console.log(`  [extra-synthetic] generated ${count} pairs`);
  return count;
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("=== Generating expanded training data (v2) ===\n");

console.log("Step 1: Merging existing training data");
mergeExisting();

console.log("\nStep 2: Extracting identifiers from node_modules");
extractFromNodeModules();

console.log("\nStep 3: Additional synthetic identifiers");
generateAdditionalSynthetic();

console.log("\nStep 4: Cross-version augmentation");
generateCrossVersionAugmentation();

// Convert to array and shuffle
const allPairs = [...pairMap.values()];

// Fisher-Yates shuffle
for (let i = allPairs.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [allPairs[i], allPairs[j]] = [allPairs[j], allPairs[i]];
}

console.log(`\n=== Total unique pairs: ${allPairs.length} ===`);

// Write JSONL
const lines = allPairs.map((p) => JSON.stringify(p)).join("\n");
writeFileSync(OUTPUT_PATH, lines + "\n", "utf8");
console.log(`Wrote ${allPairs.length} pairs to ${OUTPUT_PATH}`);

// Print stats
const kindCounts = {};
for (const p of allPairs) {
  kindCounts[p.kind] = (kindCounts[p.kind] || 0) + 1;
}
console.log("\nBreakdown by kind:");
for (const [kind, count] of Object.entries(kindCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${kind}: ${count}`);
}

// Print average context length
const avgCtx = allPairs.reduce((s, p) => s + p.context_strings.length, 0) / allPairs.length;
const avgProps = allPairs.reduce((s, p) => s + p.properties.length, 0) / allPairs.length;
console.log(`\nAverage context strings per pair: ${avgCtx.toFixed(1)}`);
console.log(`Average properties per pair: ${avgProps.toFixed(1)}`);
