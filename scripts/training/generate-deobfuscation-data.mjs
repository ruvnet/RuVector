#!/usr/bin/env node
/**
 * Generate training data for the JS deobfuscation model.
 *
 * Sources:
 *   1. Ground-truth fixtures from ruvector-decompiler tests
 *   2. Synthetic minification of open-source npm packages
 *   3. Cross-version analysis patterns
 *
 * Output: JSONL where each line is:
 *   {"minified":"a$","original":"createRouter","context_strings":[...],"properties":[...],"kind":"function"}
 *
 * Usage:
 *   node scripts/training/generate-deobfuscation-data.mjs [--output training-data.jsonl] [--min-pairs 10000]
 */

import { readFileSync, writeFileSync, readdirSync, statSync } from "fs";
import { join, resolve, extname } from "path";
import { execSync } from "child_process";
import { parseArgs } from "util";
import { COMMON_NAMES, CONTEXT_MAP, PROPERTY_MAP } from "./data/identifier-dictionaries.mjs";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

const { values: args } = parseArgs({
  options: {
    output: { type: "string", default: "training-data.jsonl" },
    "min-pairs": { type: "string", default: "10000" },
    "skip-npm": { type: "boolean", default: false },
    help: { type: "boolean", short: "h", default: false },
  },
});

if (args.help) {
  console.log("Usage: generate-deobfuscation-data.mjs [--output FILE] [--min-pairs N] [--skip-npm]");
  process.exit(0);
}

const OUTPUT_PATH = resolve(args.output);
const MIN_PAIRS = parseInt(args["min-pairs"], 10);

/** @type {Array<{minified: string, original: string, context_strings: string[], properties: string[], kind: string}>} */
const pairs = [];

// ---------------------------------------------------------------------------
// Source 1: Ground-truth fixtures
// ---------------------------------------------------------------------------

function extractGroundTruthFixtures() {
  const ROOT = resolve(import.meta.dirname, "../../crates/ruvector-decompiler/tests");
  const files = ["ground_truth.rs", "real_world.rs"];

  for (const file of files) {
    const path = join(ROOT, file);
    let content;
    try {
      content = readFileSync(path, "utf8");
    } catch {
      console.warn(`  [skip] ${path} not found`);
      continue;
    }

    // Extract (&str, &str) pairs from ORIGINAL_NAMES arrays.
    // Pattern: ("minified", "original")
    const tupleRe = /\("([^"]+)",\s*"([^"]+)"\)/g;
    let match;
    while ((match = tupleRe.exec(content)) !== null) {
      const [, minified, original] = match;
      if (minified.length <= 3 && original.length > 3) {
        pairs.push({
          minified,
          original,
          context_strings: [],
          properties: [],
          kind: "var",
        });
      }
    }

    // Extract standalone name arrays: &["Router", "Request", ...]
    const nameArrayRe = /ORIGINAL_NAMES:\s*&\[&str\]\s*=\s*&\[([\s\S]*?)\];/g;
    while ((match = nameArrayRe.exec(content)) !== null) {
      const names = match[1].match(/"([^"]+)"/g);
      if (names) {
        names.forEach((n, i) => {
          const original = n.replace(/"/g, "");
          const minified = String.fromCharCode(97 + (i % 26));
          if (!pairs.some((p) => p.original === original && p.minified === minified)) {
            pairs.push({
              minified,
              original,
              context_strings: [],
              properties: [],
              kind: "function",
            });
          }
        });
      }
    }

    // Extract string literals from minified source constants for context.
    const strLitRe = /"([a-zA-Z_][a-zA-Z0-9_]{2,})"/g;
    const contextStrings = new Set();
    while ((match = strLitRe.exec(content)) !== null) {
      const s = match[1];
      if (!["var", "let", "const", "function", "class", "return"].includes(s)) {
        contextStrings.add(s);
      }
    }

    // Enrich pairs from this file with context strings.
    const ctxArray = [...contextStrings].slice(0, 20);
    for (const pair of pairs) {
      if (pair.context_strings.length === 0) {
        pair.context_strings = ctxArray.slice(0, 5);
      }
    }
  }

  console.log(`  [ground-truth] extracted ${pairs.length} pairs`);
}

// ---------------------------------------------------------------------------
// Source 2: Synthetic minification from common identifier patterns
// ---------------------------------------------------------------------------

/**
 * Generate synthetic training pairs from common JS identifier patterns.
 * This simulates what real minifiers produce.
 */
function generateSyntheticPairs() {
  // Dictionaries imported from ./data/identifier-dictionaries.mjs

  // Minifier name generators -- expanded with more strategies.
  const minifierStyles = [
    // Single letter: a, b, c ... z
    (i) => String.fromCharCode(97 + (i % 26)),
    // With dollar suffix: a$, b$...
    (i) => String.fromCharCode(97 + (i % 26)) + "$",
    // Underscore prefix: _a, _b...
    (i) => "_" + String.fromCharCode(97 + (i % 26)),
    // Hex obfuscation: _0x1a2b...
    (i) => "_0x" + (0x1a2b + i).toString(16),
    // Letter + digit: a0, b1...
    (i) => String.fromCharCode(97 + (i % 26)) + (i % 10).toString(),
    // Double underscore: __a, __b...
    (i) => "__" + String.fromCharCode(97 + (i % 26)),
    // Dollar prefix: $a, $b...
    (i) => "$" + String.fromCharCode(97 + (i % 26)),
    // Uppercase single: A, B, C...
    (i) => String.fromCharCode(65 + (i % 26)),
    // Double letter: aa, ab, ac...
    (i) => String.fromCharCode(97 + (i % 26)) + String.fromCharCode(97 + ((i + 1) % 26)),
    // Mixed case: aA, bB, cC...
    (i) => String.fromCharCode(97 + (i % 26)) + String.fromCharCode(65 + (i % 26)),
    // Dollar + digit: $0, $1...
    (i) => "$" + (i % 100).toString(),
    // Underscore + digit: _0, _1...
    (i) => "_" + (i % 100).toString(),
    // Two letters + digit: aa1, ab2...
    (i) => String.fromCharCode(97 + (i % 26)) + String.fromCharCode(97 + ((i * 7) % 26)) + (i % 10),
    // Webpack style: __WEBPACK_MODULE_a__
    (i) => "__W" + String.fromCharCode(97 + (i % 26)) + "__",
    // Terser numbered: t0, t1, t2...
    (i) => "t" + i,
    // esbuild style: e$a, e$b...
    (i) => "e$" + String.fromCharCode(97 + (i % 26)),
  ];

  // Context variation templates for richer training signal.
  const CONTEXT_TEMPLATES = [
    (ctx) => ctx,  // original
    (ctx) => ctx.length > 2 ? [...ctx.slice(1), ctx[0]] : ctx,  // rotated
    (ctx) => ctx.slice(0, 3),  // truncated
    (ctx) => [...ctx, "prototype", "constructor"],  // with prototype hints
    (ctx) => [...ctx, "undefined", "null", "true", "false"],  // with literals
  ];

  let syntheticCount = 0;
  let globalIdx = 0;

  for (const [kind, names] of Object.entries(COMMON_NAMES)) {
    for (let i = 0; i < names.length; i++) {
      const original = names[i];
      const baseCtx = CONTEXT_MAP[original] || generateGenericContext(original);
      const baseProps = PROPERTY_MAP[original] || generateGenericProperties(kind);

      // Generate 8 minified variants per original name using a global
      // counter so names from different kinds do not collide.
      const numVariants = 8;
      for (let v = 0; v < numVariants; v++) {
        const styleIdx = (globalIdx + v) % minifierStyles.length;
        const minified = minifierStyles[styleIdx](globalIdx);

        const ctxVariant = CONTEXT_TEMPLATES[v % CONTEXT_TEMPLATES.length];
        const ctx = ctxVariant(baseCtx.length > 0 ? baseCtx : ["unknown"]);

        pairs.push({
          minified,
          original,
          context_strings: ctx,
          properties: baseProps,
          kind,
        });
        syntheticCount++;
      }
      globalIdx++;
    }
  }

  console.log(`  [synthetic] generated ${syntheticCount} pairs`);
}

/**
 * Generate generic context strings from an identifier name.
 * Splits camelCase into tokens and uses them as context hints.
 */
function generateGenericContext(name) {
  const tokens = name
    .replace(/([A-Z])/g, " $1")
    .trim()
    .toLowerCase()
    .split(/\s+/)
    .filter((t) => t.length > 2);
  return tokens.slice(0, 5);
}

/**
 * Generate generic property names based on declaration kind.
 */
function generateGenericProperties(kind) {
  switch (kind) {
    case "function":
      return ["length", "name", "call", "apply"];
    case "class":
      return ["prototype", "constructor", "name"];
    case "var":
      return ["toString", "valueOf"];
    default:
      return [];
  }
}

// ---------------------------------------------------------------------------
// Source 3: Cross-version augmentation
// ---------------------------------------------------------------------------

/**
 * Generate augmented pairs by simulating cross-version name changes.
 * Same original name gets different minified names across "versions".
 */
function generateCrossVersionPairs() {
  const existingOriginals = [...new Set(pairs.map((p) => p.original))];
  let augmented = 0;

  for (const original of existingOriginals) {
    const existing = pairs.find((p) => p.original === original);
    if (!existing) continue;

    // Simulate 3-5 additional "versions" with different minified names.
    const versions = 3 + Math.floor(Math.random() * 3);
    for (let v = 0; v < versions; v++) {
      const minified = generateRandomMinifiedName();
      if (pairs.some((p) => p.minified === minified && p.original === original)) continue;

      pairs.push({
        minified,
        original,
        context_strings: existing.context_strings,
        properties: existing.properties,
        kind: existing.kind,
      });
      augmented++;
    }
  }

  console.log(`  [cross-version] augmented ${augmented} pairs`);
}

/**
 * Generate a random minified-style variable name.
 */
function generateRandomMinifiedName() {
  const letter = () => String.fromCharCode(97 + Math.floor(Math.random() * 26));
  const LETTER = () => String.fromCharCode(65 + Math.floor(Math.random() * 26));
  const digit = () => Math.floor(Math.random() * 10).toString();
  const styles = [
    () => letter() + Math.floor(Math.random() * 100),       // a42
    () => "_0x" + Math.floor(Math.random() * 0xffff).toString(16), // _0x3f1a
    () => letter() + letter(),                                // ab
    () => "$" + letter(),                                     // $a
    () => "_" + letter(),                                     // _a
    () => letter() + LETTER(),                                // aB
    () => letter() + letter() + digit(),                      // ab3
    () => "__" + letter() + letter(),                          // __ab
    () => "$" + digit() + digit(),                            // $42
    () => letter() + "$" + digit(),                           // a$3
    () => "_" + digit() + letter(),                           // _3a
    () => "t" + Math.floor(Math.random() * 1000),             // t523
  ];
  return styles[Math.floor(Math.random() * styles.length)]();
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

console.log("Generating deobfuscation training data...\n");

console.log("Source 1: Ground-truth fixtures");
extractGroundTruthFixtures();

console.log("\nSource 2: Synthetic minification patterns");
generateSyntheticPairs();

console.log("\nSource 3: Cross-version augmentation");
generateCrossVersionPairs();

// Deduplicate.
const seen = new Set();
const deduplicated = pairs.filter((p) => {
  const key = `${p.minified}|${p.original}`;
  if (seen.has(key)) return false;
  seen.add(key);
  return true;
});

console.log(`\nTotal: ${deduplicated.length} unique pairs (target: ${MIN_PAIRS})`);

if (deduplicated.length < MIN_PAIRS) {
  console.warn(`WARNING: Only ${deduplicated.length} pairs generated, below target of ${MIN_PAIRS}.`);
  console.warn("Consider adding more npm packages or expanding COMMON_NAMES.");
}

// Shuffle for training.
for (let i = deduplicated.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [deduplicated[i], deduplicated[j]] = [deduplicated[j], deduplicated[i]];
}

// Write JSONL.
const lines = deduplicated.map((p) => JSON.stringify(p)).join("\n");
writeFileSync(OUTPUT_PATH, lines + "\n", "utf8");
console.log(`\nWrote ${deduplicated.length} training pairs to ${OUTPUT_PATH}`);
