// Text normalization shared by the bench (JS) and the OpenJev trainer (Rust),
// ADR-007 §2. Every leakage hash on both sides is sha256Norm(text), so the two
// implementations MUST agree byte for byte. The contract is the golden file
// bench/test/norm-golden.json (input → norm → sha256); the Rust `norm` runs the
// same file.
//
// norm(text), exactly:
//   1. Unicode NFKC normalization.
//   2. Full Unicode lowercase (default case mapping incl. SpecialCasing and the
//      context-sensitive final sigma: JS String#toLowerCase / Rust
//      str::to_lowercase).
//   3. Every maximal run of code points whose General_Category is NOT a Letter
//      (L*: Lu Ll Lt Lm Lo) or a Number (N*: Nd Nl No) becomes ONE space
//      (U+0020). Marks (Mn/Mc/Me — combining accents left after NFKC, Indic
//      vowel signs, U+0307 from lowercasing "İ"), punctuation, symbols, emoji,
//      whitespace, controls and format characters (ZWJ) are all separators.
//      NB: this is the General_Category test `[^\p{L}\p{N}]`, NOT Rust's
//      char::is_alphanumeric (which uses the Alphabetic derived property and
//      keeps Other_Alphabetic marks). Rust: regex `[^\p{L}\p{N}]+`.
//   4. Trim leading/trailing spaces (after step 3 only U+0020 can remain).
//
// sha256Norm(text) = lowercase hex SHA-256 of the UTF-8 bytes of norm(text).
// Unicode-version skew between ICU (Node) and Rust's tables can only affect
// code points assigned after both versions' common base; the golden file is the
// arbiter.

import { createHash } from 'node:crypto';

const NON_ALNUM_RUN = /[^\p{L}\p{N}]+/gu;

/** ADR-007 §2 normalization. Throws on a non-string input (fail closed). */
export function norm(text) {
  if (typeof text !== 'string') throw new TypeError(`norm: expected a string, got ${typeof text}`);
  return text.normalize('NFKC').toLowerCase().replace(NON_ALNUM_RUN, ' ').trim();
}

/** Lowercase hex sha256 of the UTF-8 bytes of norm(text). */
export function sha256Norm(text) {
  return createHash('sha256').update(norm(text), 'utf8').digest('hex');
}
