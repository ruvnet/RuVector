#!/usr/bin/env node
/**
 * Extract training pairs from source maps in node_modules.
 *
 * Source maps contain `names` arrays with original identifiers and VLQ-encoded
 * mappings that tell us exactly which minified name maps to which original.
 *
 * For each source map:
 *   1. Parse the .js.map JSON
 *   2. Decode VLQ mappings to get (line, col, nameIdx) tuples
 *   3. Read the corresponding .js file
 *   4. Extract the minified identifier at each (line, col) position
 *   5. Extract surrounding context (string literals, property accesses)
 *   6. Output (minified, original, context, properties, kind) pairs
 *
 * Usage:
 *   node scripts/training/extract-sourcemap-pairs.mjs [--output training-data-sourcemaps.jsonl]
 */

import { readFileSync, writeFileSync, existsSync } from "fs";
import { resolve, dirname, basename, join } from "path";
import { execSync } from "child_process";
import { parseArgs } from "util";

const { values: args } = parseArgs({
  options: {
    output: { type: "string", default: "training-data-sourcemaps.jsonl" },
    help: { type: "boolean", short: "h", default: false },
  },
});

if (args.help) {
  console.log("Usage: extract-sourcemap-pairs.mjs [--output FILE]");
  process.exit(0);
}

const OUTPUT_PATH = resolve(args.output);
const ROOT = resolve(import.meta.dirname, "../..");

// ---------------------------------------------------------------------------
// VLQ Decoder
// ---------------------------------------------------------------------------

const VLQ_BASE_SHIFT = 5;
const VLQ_BASE = 1 << VLQ_BASE_SHIFT;
const VLQ_BASE_MASK = VLQ_BASE - 1;
const VLQ_CONTINUATION_BIT = VLQ_BASE;

const BASE64_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const BASE64_MAP = new Map();
for (let i = 0; i < BASE64_CHARS.length; i++) {
  BASE64_MAP.set(BASE64_CHARS[i], i);
}

function decodeVLQ(str, pos) {
  let result = 0;
  let shift = 0;
  let continuation = true;

  while (continuation && pos < str.length) {
    const digit = BASE64_MAP.get(str[pos++]);
    if (digit === undefined) break;
    continuation = !!(digit & VLQ_CONTINUATION_BIT);
    result += (digit & VLQ_BASE_MASK) << shift;
    shift += VLQ_BASE_SHIFT;
  }

  // Sign is in the least significant bit
  const negate = result & 1;
  result >>= 1;
  return { value: negate ? -result : result, pos };
}

/**
 * Decode source map mappings string into an array of segments.
 * Each segment: [genCol, sourceIdx, sourceLine, sourceCol, nameIdx?]
 */
function decodeMappings(mappingsStr) {
  const lines = [];
  let currentLine = [];

  let generatedColumn = 0;
  let sourceIndex = 0;
  let sourceLine = 0;
  let sourceColumn = 0;
  let nameIndex = 0;

  let pos = 0;
  while (pos < mappingsStr.length) {
    const ch = mappingsStr[pos];

    if (ch === ";") {
      lines.push(currentLine);
      currentLine = [];
      generatedColumn = 0;
      pos++;
      continue;
    }

    if (ch === ",") {
      pos++;
      continue;
    }

    // Decode segment
    const segment = [];

    // Field 1: generated column (relative)
    let decoded = decodeVLQ(mappingsStr, pos);
    generatedColumn += decoded.value;
    segment.push(generatedColumn);
    pos = decoded.pos;

    if (pos < mappingsStr.length && mappingsStr[pos] !== "," && mappingsStr[pos] !== ";") {
      // Field 2: source index (relative)
      decoded = decodeVLQ(mappingsStr, pos);
      sourceIndex += decoded.value;
      segment.push(sourceIndex);
      pos = decoded.pos;

      // Field 3: source line (relative)
      decoded = decodeVLQ(mappingsStr, pos);
      sourceLine += decoded.value;
      segment.push(sourceLine);
      pos = decoded.pos;

      // Field 4: source column (relative)
      decoded = decodeVLQ(mappingsStr, pos);
      sourceColumn += decoded.value;
      segment.push(sourceColumn);
      pos = decoded.pos;

      // Field 5: name index (optional, relative)
      if (pos < mappingsStr.length && mappingsStr[pos] !== "," && mappingsStr[pos] !== ";") {
        decoded = decodeVLQ(mappingsStr, pos);
        nameIndex += decoded.value;
        segment.push(nameIndex);
        pos = decoded.pos;
      }
    }

    currentLine.push(segment);
  }

  if (currentLine.length > 0) {
    lines.push(currentLine);
  }

  return lines;
}

// ---------------------------------------------------------------------------
// Identifier extraction from minified JS at a given position
// ---------------------------------------------------------------------------

const IDENT_RE = /^[a-zA-Z_$][a-zA-Z0-9_$]*/;

function extractIdentifierAt(lines, lineIdx, colIdx) {
  if (lineIdx >= lines.length) return null;
  const line = lines[lineIdx];
  if (colIdx >= line.length) return null;
  const rest = line.substring(colIdx);
  const m = rest.match(IDENT_RE);
  return m ? m[0] : null;
}

// ---------------------------------------------------------------------------
// Context extraction
// ---------------------------------------------------------------------------

function extractContext(lines, lineIdx, colIdx, windowLines = 3) {
  const context = [];
  const startLine = Math.max(0, lineIdx - windowLines);
  const endLine = Math.min(lines.length, lineIdx + windowLines + 1);

  for (let i = startLine; i < endLine; i++) {
    const line = lines[i];

    // Extract string literals
    const strRe = /["']([a-zA-Z][a-zA-Z0-9_./-]{2,})["']/g;
    let m;
    while ((m = strRe.exec(line)) !== null) {
      if (m[1].length < 30) context.push(m[1]);
    }

    // Extract property accesses
    const propRe = /\.([a-zA-Z_$][a-zA-Z0-9_$]{2,})/g;
    while ((m = propRe.exec(line)) !== null) {
      if (m[1].length < 25) context.push(m[1]);
    }
  }

  return [...new Set(context)].slice(0, 10);
}

function extractProperties(lines, lineIdx, identifier, windowLines = 5) {
  const props = new Set();
  const startLine = Math.max(0, lineIdx - windowLines);
  const endLine = Math.min(lines.length, lineIdx + windowLines + 1);

  for (let i = startLine; i < endLine; i++) {
    const re = new RegExp(`\\b${escapeRegex(identifier)}\\.([a-zA-Z_$][a-zA-Z0-9_$]{1,})`, "g");
    let m;
    while ((m = re.exec(lines[i])) !== null) {
      if (m[1].length < 25) props.add(m[1]);
    }
  }

  return [...props].slice(0, 8);
}

function escapeRegex(s) {
  return s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

// ---------------------------------------------------------------------------
// Determine kind from context
// ---------------------------------------------------------------------------

function inferKind(lines, lineIdx, colIdx, identifier) {
  if (lineIdx >= lines.length) return "var";
  const line = lines[lineIdx];

  // Check what precedes the identifier
  const before = line.substring(0, colIdx).trimEnd();
  if (/\bfunction\s*$/.test(before)) return "function";
  if (/\bclass\s*$/.test(before)) return "class";
  if (/\b(?:const|let|var)\s*$/.test(before)) return "var";

  // Check if identifier starts with uppercase (likely class)
  if (/^[A-Z][a-z]/.test(identifier)) return "class";

  // Check if followed by ( → likely function
  const after = line.substring(colIdx + identifier.length).trimStart();
  if (after.startsWith("(") || after.startsWith("=function") || after.startsWith("=async")) {
    return "function";
  }

  return "var";
}

// ---------------------------------------------------------------------------
// Process a single source map
// ---------------------------------------------------------------------------

function processSourceMap(mapPath) {
  const pairs = [];

  let mapJson;
  try {
    mapJson = JSON.parse(readFileSync(mapPath, "utf8"));
  } catch {
    return pairs;
  }

  const names = mapJson.names || [];
  const mappings = mapJson.mappings;
  if (!names.length || !mappings) return pairs;

  // Find the corresponding JS file
  const jsPath = mapPath.replace(/\.map$/, "");
  if (!existsSync(jsPath)) return pairs;

  let jsContent;
  try {
    jsContent = readFileSync(jsPath, "utf8");
  } catch {
    return pairs;
  }

  const jsLines = jsContent.split("\n");

  // Decode mappings
  let decodedLines;
  try {
    decodedLines = decodeMappings(mappings);
  } catch {
    return pairs;
  }

  // Process each segment that has a name index
  const seen = new Set();

  for (let lineIdx = 0; lineIdx < decodedLines.length; lineIdx++) {
    const segments = decodedLines[lineIdx];
    for (const seg of segments) {
      if (seg.length < 5) continue; // No name index

      const genCol = seg[0];
      const nameIdx = seg[4];

      if (nameIdx < 0 || nameIdx >= names.length) continue;

      const originalName = names[nameIdx];
      if (!originalName || originalName.length < 3) continue;

      // Skip common keywords
      if (SKIP_NAMES.has(originalName)) continue;

      // Extract the minified identifier at this position
      const minified = extractIdentifierAt(jsLines, lineIdx, genCol);
      if (!minified) continue;

      // Skip if minified === original (no renaming happened)
      if (minified === originalName) continue;

      // Skip if minified is too long (probably not actually minified)
      if (minified.length > 6) continue;

      // Deduplicate
      const key = `${minified}|${originalName}`;
      if (seen.has(key)) continue;
      seen.add(key);

      // Extract context and properties
      const context = extractContext(jsLines, lineIdx, genCol);
      const properties = extractProperties(jsLines, lineIdx, minified);
      const kind = inferKind(jsLines, lineIdx, genCol, originalName);

      pairs.push({
        minified,
        original: originalName,
        context_strings: context,
        properties,
        kind,
      });
    }
  }

  return pairs;
}

const SKIP_NAMES = new Set([
  "if", "else", "for", "while", "do", "switch", "case", "break",
  "continue", "return", "try", "catch", "finally", "throw", "new",
  "delete", "typeof", "void", "instanceof", "in", "of", "with",
  "this", "super", "true", "false", "null", "undefined", "NaN",
  "Infinity", "arguments", "eval", "constructor", "prototype",
  "use", "strict", "exports", "module", "require",
  "Object", "Array", "String", "Number", "Boolean", "Function",
  "Symbol", "BigInt", "Map", "Set", "WeakMap", "WeakSet",
  "Promise", "Error", "TypeError", "RangeError", "SyntaxError",
  "Math", "Date", "JSON", "RegExp", "Proxy", "Reflect",
  "console", "document", "window", "global", "globalThis",
  "process", "Buffer", "setTimeout", "setInterval", "clearTimeout",
  "length", "push", "pop", "shift", "unshift",
  "call", "apply", "bind", "toString", "valueOf",
  "hasOwnProperty", "propertyIsEnumerable", "isPrototypeOf",
  "__proto__", "__defineGetter__", "__defineSetter__",
]);

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("=== Extracting training pairs from source maps ===\n");

// Find all source map files
const findCmd = `find ${join(ROOT, "node_modules")} -name "*.js.map" -not -path "*/node_modules/*/node_modules/*" -size +1k 2>/dev/null`;
const mapFiles = execSync(findCmd, { encoding: "utf8" }).trim().split("\n").filter(Boolean);

console.log(`Found ${mapFiles.length} source map files\n`);

let totalPairs = 0;
let filesWithPairs = 0;
const allPairs = [];

for (let i = 0; i < mapFiles.length; i++) {
  const mapFile = mapFiles[i];
  const pairs = processSourceMap(mapFile);
  if (pairs.length > 0) {
    allPairs.push(...pairs);
    totalPairs += pairs.length;
    filesWithPairs++;

    if (pairs.length >= 10) {
      const rel = mapFile.replace(ROOT + "/node_modules/", "");
      console.log(`  [${pairs.length} pairs] ${rel}`);
    }
  }

  // Progress every 500 files
  if ((i + 1) % 500 === 0) {
    console.log(`  ... processed ${i + 1}/${mapFiles.length} files, ${totalPairs} pairs so far`);
  }
}

console.log(`\nProcessed ${mapFiles.length} files`);
console.log(`Files with pairs: ${filesWithPairs}`);
console.log(`Total pairs: ${totalPairs}`);

// Deduplicate globally
const globalSeen = new Set();
const deduped = allPairs.filter((p) => {
  const key = `${p.minified}|${p.original}`;
  if (globalSeen.has(key)) return false;
  globalSeen.add(key);
  return true;
});

console.log(`After dedup: ${deduped.length} unique pairs`);

// Shuffle
for (let i = deduped.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [deduped[i], deduped[j]] = [deduped[j], deduped[i]];
}

// Write
const lines = deduped.map((p) => JSON.stringify(p)).join("\n");
writeFileSync(OUTPUT_PATH, lines + "\n", "utf8");
console.log(`\nWrote ${deduped.length} pairs to ${OUTPUT_PATH}`);

// Stats
const kindCounts = {};
for (const p of deduped) {
  kindCounts[p.kind] = (kindCounts[p.kind] || 0) + 1;
}
console.log("\nBreakdown by kind:");
for (const [kind, count] of Object.entries(kindCounts).sort((a, b) => b[1] - a[1])) {
  console.log(`  ${kind}: ${count}`);
}

const avgCtx = deduped.reduce((s, p) => s + p.context_strings.length, 0) / Math.max(deduped.length, 1);
console.log(`\nAvg context strings: ${avgCtx.toFixed(1)}`);
