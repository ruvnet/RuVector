//! Integration tests for the Unicode security module (ADR-103 C7).
//!
//! Tests cover BiDi override detection, zero-width character detection,
//! dangerous character stripping, ASCII identifier validation, and
//! Cyrillic confusable detection.

use rvagent_backends::unicode_security::{
    detect_confusables, detect_dangerous_unicode, detect_script, strip_dangerous_unicode,
    validate_ascii_identifier, ScriptCategory,
};

/// BiDi directional override characters must be detected.
#[test]
fn test_detect_bidi_override() {
    // RIGHT-TO-LEFT OVERRIDE (U+202E) — the classic attack vector.
    let text = "normal\u{202E}reversed";
    let issues = detect_dangerous_unicode(text);

    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].character, '\u{202E}');
    assert_eq!(issues[0].codepoint, "U+202E");
    assert_eq!(issues[0].description, "RIGHT-TO-LEFT OVERRIDE");

    // Multiple BiDi controls.
    let multi = "\u{202A}LRE\u{202B}RLE\u{202C}PDF\u{202D}LRO\u{202E}RLO";
    let issues2 = detect_dangerous_unicode(multi);
    assert_eq!(issues2.len(), 5);

    // BiDi isolate controls (U+2066-U+2069).
    let isolates = "\u{2066}\u{2067}\u{2068}\u{2069}";
    let issues3 = detect_dangerous_unicode(isolates);
    assert_eq!(issues3.len(), 4);
}

/// Zero-width characters must be detected.
#[test]
fn test_detect_zero_width_chars() {
    // ZERO WIDTH SPACE (U+200B)
    let text = "hello\u{200B}world";
    let issues = detect_dangerous_unicode(text);
    assert_eq!(issues.len(), 1);
    assert_eq!(issues[0].character, '\u{200B}');
    assert_eq!(issues[0].codepoint, "U+200B");

    // ZERO WIDTH JOINER (U+200D) — used to construct invisible differences.
    let zwj = "a\u{200D}b";
    let issues2 = detect_dangerous_unicode(zwj);
    assert_eq!(issues2.len(), 1);
    assert_eq!(issues2[0].character, '\u{200D}');

    // BOM (U+FEFF)
    let bom = "\u{FEFF}file content";
    let issues3 = detect_dangerous_unicode(bom);
    assert_eq!(issues3.len(), 1);
    assert_eq!(issues3[0].character, '\u{FEFF}');

    // Clean text should produce no issues.
    let clean = "perfectly normal text 123 !@#";
    assert!(detect_dangerous_unicode(clean).is_empty());
}

/// strip_dangerous_unicode should remove all dangerous codepoints
/// while preserving safe text (including non-ASCII like accented chars).
#[test]
fn test_strip_dangerous_unicode() {
    // Strip zero-width space and BiDi override.
    let dirty = "he\u{200B}llo\u{202E} world";
    let clean = strip_dangerous_unicode(dirty);
    assert_eq!(clean, "hello world");

    // Preserve safe non-ASCII.
    let accented = "caf\u{00E9}";
    assert_eq!(strip_dangerous_unicode(accented), "caf\u{00E9}");

    // Strip multiple dangerous characters.
    let multi = "\u{FEFF}\u{200B}abc\u{200D}def\u{202E}ghi";
    let result = strip_dangerous_unicode(multi);
    assert_eq!(result, "abcdefghi");

    // Empty string stays empty.
    assert_eq!(strip_dangerous_unicode(""), "");

    // Already-clean text is unchanged.
    let safe = "fn main() { println!(\"hello\"); }";
    assert_eq!(strip_dangerous_unicode(safe), safe);
}

/// ASCII identifier validation (ADR-103 C10) should accept only
/// lowercase ASCII letters, digits, hyphens, and underscores,
/// starting with a letter.
#[test]
fn test_ascii_identifier_validation() {
    // Valid identifiers.
    assert!(validate_ascii_identifier("hello"));
    assert!(validate_ascii_identifier("my-skill"));
    assert!(validate_ascii_identifier("test_123"));
    assert!(validate_ascii_identifier("a"));
    assert!(validate_ascii_identifier("skill-name-v2"));
    assert!(validate_ascii_identifier("x0"));

    // Invalid: empty.
    assert!(!validate_ascii_identifier(""));

    // Invalid: starts with digit.
    assert!(!validate_ascii_identifier("123abc"));

    // Invalid: starts with hyphen.
    assert!(!validate_ascii_identifier("-start"));

    // Invalid: starts with underscore.
    assert!(!validate_ascii_identifier("_start"));

    // Invalid: uppercase letters.
    assert!(!validate_ascii_identifier("Hello"));
    assert!(!validate_ascii_identifier("ALLCAPS"));

    // Invalid: contains Cyrillic (confusable with Latin).
    assert!(!validate_ascii_identifier("na\u{0441}me")); // Cyrillic 'с' looks like 'c'

    // Invalid: contains accented characters.
    assert!(!validate_ascii_identifier("caf\u{00E9}"));

    // Invalid: contains spaces.
    assert!(!validate_ascii_identifier("has space"));

    // Invalid: contains dots.
    assert!(!validate_ascii_identifier("has.dot"));
}

/// Cyrillic/Greek/Armenian homoglyphs confusable with Latin characters
/// must be detected (ADR-103 C7).
#[test]
fn test_cyrillic_confusable_detection() {
    // Cyrillic 'А' (U+0410) looks like Latin 'A'.
    let cyrillic_a = "\u{0410}";
    let results = detect_confusables(cyrillic_a);
    assert_eq!(results.len(), 1);
    assert_eq!(results[0].1, '\u{0410}'); // the confusable char
    assert_eq!(results[0].2, 'A'); // the Latin lookalike

    // Cyrillic 'с' (U+0441) looks like Latin 'c'.
    let cyrillic_c = "\u{0441}";
    let results2 = detect_confusables(cyrillic_c);
    assert_eq!(results2.len(), 1);
    assert_eq!(results2[0].2, 'c');

    // Mixed text with some confusables embedded.
    let mixed = "hell\u{043E}"; // Cyrillic 'о' instead of Latin 'o'
    let results3 = detect_confusables(mixed);
    assert_eq!(results3.len(), 1);
    assert_eq!(results3[0].2, 'o');

    // Greek confusables.
    let greek_alpha = "\u{0391}"; // Greek 'Α' looks like Latin 'A'
    let results4 = detect_confusables(greek_alpha);
    assert_eq!(results4.len(), 1);
    assert_eq!(results4[0].2, 'A');

    // Pure Latin text should have zero confusables.
    let latin = "Hello World";
    assert!(detect_confusables(latin).is_empty());

    // Script detection sanity.
    assert_eq!(detect_script('A'), ScriptCategory::Latin);
    assert_eq!(detect_script('\u{0410}'), ScriptCategory::Cyrillic);
    assert_eq!(detect_script('\u{0391}'), ScriptCategory::Greek);
    assert_eq!(detect_script('\u{0531}'), ScriptCategory::Armenian);
}
