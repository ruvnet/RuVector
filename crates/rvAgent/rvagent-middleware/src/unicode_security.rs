//! Unicode security module for detecting dangerous characters and confusables.
//!
//! Provides detection and mitigation for:
//! - BiDi directional formatting controls (can reverse text display)
//! - Zero-width and invisible characters
//! - Confusable characters (Cyrillic/Latin homoglyphs)
//! - Mixed-script attacks
//! - Homoglyph attacks
//!
//! ## Security Considerations (C7)
//!
//! This module implements defenses against Unicode-based attacks:
//! - **BiDi Override**: U+202E can reverse displayed text (e.g., "evil.exe" → "exe.live")
//! - **Zero-Width Injection**: Invisible characters that can hide malicious content
//! - **Homoglyph Attacks**: Using visually similar characters (Cyrillic 'а' vs Latin 'a')
//! - **Mixed Scripts**: Detect suspicious mixing of Unicode scripts
//!
//! ## Usage
//!
//! ```rust
//! use rvagent_middleware::unicode_security::UnicodeSecurityChecker;
//!
//! let checker = UnicodeSecurityChecker::strict();
//!
//! // Check for issues
//! let text = "Hello\u{202E}world"; // Contains BiDi override
//! let issues = checker.check(text);
//! assert!(!issues.is_empty());
//!
//! // Sanitize dangerous characters
//! let safe = checker.sanitize(text);
//! assert_eq!(safe, "Helloworld");
//!
//! // Check if ASCII-safe
//! assert!(UnicodeSecurityChecker::is_ascii_safe("safe text"));
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// BiDi directional formatting controls (can reverse text display).
///
/// **Most dangerous**: U+202E (RLO) can completely reverse displayed text.
/// Example attack: "evil\u{202E}txt.exe" displays as "evilexe.txt"
const BIDI_CONTROLS: &[char] = &[
    '\u{202A}', // LRE - Left-to-Right Embedding
    '\u{202B}', // RLE - Right-to-Left Embedding
    '\u{202C}', // PDF - Pop Directional Format
    '\u{202D}', // LRO - Left-to-Right Override
    '\u{202E}', // RLO - Right-to-Left Override (MOST DANGEROUS)
    '\u{2066}', // LRI - Left-to-Right Isolate
    '\u{2067}', // RLI - Right-to-Left Isolate
    '\u{2068}', // FSI - First Strong Isolate
    '\u{2069}', // PDI - Pop Directional Isolate
];

/// Zero-width and invisible characters.
///
/// Can be used to hide malicious content or bypass filters.
const ZERO_WIDTH: &[char] = &[
    '\u{200B}', // ZWSP - Zero Width Space
    '\u{200C}', // ZWNJ - Zero Width Non-Joiner
    '\u{200D}', // ZWJ - Zero Width Joiner
    '\u{200E}', // LRM - Left-to-Right Mark
    '\u{200F}', // RLM - Right-to-Left Mark
    '\u{2060}', // WJ - Word Joiner
    '\u{FEFF}', // BOM/ZWNBSP - Zero Width No-Break Space
];

/// Common Cyrillic confusables for Latin letters.
///
/// These Cyrillic characters look identical to Latin letters in many fonts.
/// Example attack: "pаypal.com" (Cyrillic 'а') vs "paypal.com" (Latin 'a')
const CYRILLIC_CONFUSABLES: &[(char, char)] = &[
    ('а', 'a'), // Cyrillic small letter a
    ('е', 'e'), // Cyrillic small letter ie
    ('о', 'o'), // Cyrillic small letter o
    ('р', 'p'), // Cyrillic small letter er
    ('с', 'c'), // Cyrillic small letter es
    ('у', 'y'), // Cyrillic small letter u
    ('х', 'x'), // Cyrillic small letter ha
    ('А', 'A'), // Cyrillic capital letter A
    ('В', 'B'), // Cyrillic capital letter Ve
    ('Е', 'E'), // Cyrillic capital letter Ie
    ('К', 'K'), // Cyrillic capital letter Ka
    ('М', 'M'), // Cyrillic capital letter Em
    ('Н', 'H'), // Cyrillic capital letter En
    ('О', 'O'), // Cyrillic capital letter O
    ('Р', 'P'), // Cyrillic capital letter Er
    ('С', 'C'), // Cyrillic capital letter Es
    ('Т', 'T'), // Cyrillic capital letter Te
    ('Х', 'X'), // Cyrillic capital letter Ha
];

/// Type of Unicode security issue detected.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnicodeIssue {
    /// BiDi control character detected.
    BidiControl {
        char: char,
        position: usize,
        unicode: String,
    },
    /// Zero-width or invisible character detected.
    ZeroWidth {
        char: char,
        position: usize,
        unicode: String,
    },
    /// Confusable character detected (looks like another character).
    Confusable {
        char: char,
        looks_like: char,
        position: usize,
    },
    /// Mixed scripts detected (potential homoglyph attack).
    MixedScript { scripts: Vec<String> },
    /// Homoglyph attack detected (normalized form differs significantly).
    HomoglyphAttack {
        original: String,
        normalized: String,
    },
}

impl std::fmt::Display for UnicodeIssue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnicodeIssue::BidiControl {
                char,
                position,
                unicode,
            } => write!(
                f,
                "BiDi control '{}' ({}) at position {}",
                char.escape_unicode(),
                unicode,
                position
            ),
            UnicodeIssue::ZeroWidth {
                char,
                position,
                unicode,
            } => write!(
                f,
                "Zero-width '{}' ({}) at position {}",
                char.escape_unicode(),
                unicode,
                position
            ),
            UnicodeIssue::Confusable {
                char,
                looks_like,
                position,
            } => write!(
                f,
                "Confusable '{}' looks like '{}' at position {}",
                char, looks_like, position
            ),
            UnicodeIssue::MixedScript { scripts } => {
                write!(f, "Mixed scripts: {}", scripts.join(", "))
            }
            UnicodeIssue::HomoglyphAttack {
                original,
                normalized,
            } => write!(
                f,
                "Homoglyph attack: '{}' normalizes to '{}'",
                original, normalized
            ),
        }
    }
}

/// Configuration for Unicode security checks.
#[derive(Debug, Clone)]
pub struct UnicodeSecurityConfig {
    /// Allow BiDi control characters (default: false).
    pub allow_bidi: bool,
    /// Allow zero-width characters (default: false).
    pub allow_zero_width: bool,
    /// Check for confusable characters (default: true).
    pub check_confusables: bool,
    /// Check for mixed scripts (default: true).
    pub check_mixed_scripts: bool,
    /// Check for homoglyph attacks (default: true).
    pub check_homoglyphs: bool,
}

impl Default for UnicodeSecurityConfig {
    fn default() -> Self {
        Self::strict()
    }
}

impl UnicodeSecurityConfig {
    /// Strict security (all checks enabled, nothing allowed).
    pub fn strict() -> Self {
        Self {
            allow_bidi: false,
            allow_zero_width: false,
            check_confusables: true,
            check_mixed_scripts: true,
            check_homoglyphs: true,
        }
    }

    /// Permissive security (only BiDi and zero-width checks).
    pub fn permissive() -> Self {
        Self {
            allow_bidi: false, // Never allow BiDi
            allow_zero_width: false,
            check_confusables: false,
            check_mixed_scripts: false,
            check_homoglyphs: false,
        }
    }
}

/// Unicode security checker.
///
/// Detects and mitigates Unicode-based security issues.
pub struct UnicodeSecurityChecker {
    config: UnicodeSecurityConfig,
}

impl UnicodeSecurityChecker {
    /// Create a new checker with strict security.
    pub fn strict() -> Self {
        Self {
            config: UnicodeSecurityConfig::strict(),
        }
    }

    /// Create a new checker with custom configuration.
    pub fn new(config: UnicodeSecurityConfig) -> Self {
        Self { config }
    }

    /// Check text for Unicode security issues.
    ///
    /// Returns a list of all detected issues.
    pub fn check(&self, text: &str) -> Vec<UnicodeIssue> {
        let mut issues = Vec::new();

        // Check each character
        for (i, c) in text.char_indices() {
            // BiDi control check
            if !self.config.allow_bidi && BIDI_CONTROLS.contains(&c) {
                issues.push(UnicodeIssue::BidiControl {
                    char: c,
                    position: i,
                    unicode: format!("U+{:04X}", c as u32),
                });
            }

            // Zero-width check
            if !self.config.allow_zero_width && ZERO_WIDTH.contains(&c) {
                issues.push(UnicodeIssue::ZeroWidth {
                    char: c,
                    position: i,
                    unicode: format!("U+{:04X}", c as u32),
                });
            }

            // Confusable check
            if self.config.check_confusables {
                for &(cyrillic, latin) in CYRILLIC_CONFUSABLES {
                    if c == cyrillic {
                        issues.push(UnicodeIssue::Confusable {
                            char: c,
                            looks_like: latin,
                            position: i,
                        });
                    }
                }
            }
        }

        // Mixed script check
        if self.config.check_mixed_scripts {
            if let Some(scripts) = self.detect_mixed_scripts(text) {
                issues.push(UnicodeIssue::MixedScript { scripts });
            }
        }

        // Homoglyph attack check
        if self.config.check_homoglyphs {
            if let Some((original, normalized)) = self.detect_homoglyph_attack(text) {
                issues.push(UnicodeIssue::HomoglyphAttack {
                    original,
                    normalized,
                });
            }
        }

        issues
    }

    /// Strip dangerous characters from text.
    ///
    /// Removes BiDi controls and zero-width characters.
    pub fn sanitize(&self, text: &str) -> String {
        text.chars()
            .filter(|c| !BIDI_CONTROLS.contains(c) && !ZERO_WIDTH.contains(c))
            .collect()
    }

    /// Check if string is pure ASCII (safe).
    pub fn is_ascii_safe(text: &str) -> bool {
        text.chars().all(|c| c.is_ascii())
    }

    /// Detect mixed scripts in text.
    ///
    /// Returns Some(scripts) if multiple scripts are detected, None if uniform.
    fn detect_mixed_scripts(&self, text: &str) -> Option<Vec<String>> {
        let mut scripts = HashSet::new();

        for c in text.chars() {
            if c.is_ascii_alphabetic() {
                scripts.insert("Latin");
            } else if ('\u{0400}'..='\u{04FF}').contains(&c) {
                scripts.insert("Cyrillic");
            } else if ('\u{0370}'..='\u{03FF}').contains(&c) {
                scripts.insert("Greek");
            } else if ('\u{0590}'..='\u{05FF}').contains(&c) {
                scripts.insert("Hebrew");
            } else if ('\u{0600}'..='\u{06FF}').contains(&c) {
                scripts.insert("Arabic");
            } else if ('\u{4E00}'..='\u{9FFF}').contains(&c) {
                scripts.insert("CJK");
            }
        }

        if scripts.len() > 1 {
            Some(scripts.into_iter().map(|s| s.to_string()).collect())
        } else {
            None
        }
    }

    /// Detect homoglyph attacks by comparing original and normalized text.
    ///
    /// Returns Some((original, normalized)) if they differ significantly.
    fn detect_homoglyph_attack(&self, text: &str) -> Option<(String, String)> {
        // Normalize by replacing confusables with their Latin equivalents
        let normalized: String = text
            .chars()
            .map(|c| {
                for &(cyrillic, latin) in CYRILLIC_CONFUSABLES {
                    if c == cyrillic {
                        return latin;
                    }
                }
                c
            })
            .collect();

        // If normalized differs, it's a potential homoglyph attack
        if text != normalized && text.chars().any(|c| !c.is_ascii()) {
            Some((text.to_string(), normalized))
        } else {
            None
        }
    }

    /// Check if text is safe (no issues detected).
    pub fn is_safe(&self, text: &str) -> bool {
        self.check(text).is_empty()
    }

    /// Get the configuration.
    pub fn config(&self) -> &UnicodeSecurityConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bidi_control_detection() {
        let checker = UnicodeSecurityChecker::strict();

        // RLO (most dangerous)
        let text = "evil\u{202E}txt.exe";
        let issues = checker.check(text);
        assert_eq!(issues.len(), 1);
        match &issues[0] {
            UnicodeIssue::BidiControl { char, position, .. } => {
                assert_eq!(*char, '\u{202E}');
                assert_eq!(*position, 4);
            }
            _ => panic!("Expected BidiControl issue"),
        }

        // LRE
        let text2 = "Hello\u{202A}world";
        let issues2 = checker.check(text2);
        assert_eq!(issues2.len(), 1);
    }

    #[test]
    fn test_zero_width_detection() {
        let checker = UnicodeSecurityChecker::strict();

        // ZWSP
        let text = "Hello\u{200B}world";
        let issues = checker.check(text);
        assert_eq!(issues.len(), 1);
        match &issues[0] {
            UnicodeIssue::ZeroWidth { char, position, .. } => {
                assert_eq!(*char, '\u{200B}');
                assert_eq!(*position, 5);
            }
            _ => panic!("Expected ZeroWidth issue"),
        }

        // BOM
        let text2 = "\u{FEFF}secret";
        let issues2 = checker.check(text2);
        assert_eq!(issues2.len(), 1);
    }

    #[test]
    fn test_confusable_detection() {
        let checker = UnicodeSecurityChecker::strict();

        // Cyrillic 'а' (U+0430) looks like Latin 'a' (U+0061)
        let text = "pаypal.com"; // Contains Cyrillic 'а'
        let issues = checker.check(text);

        // Should detect at least one confusable
        let confusables: Vec<_> = issues
            .iter()
            .filter_map(|issue| match issue {
                UnicodeIssue::Confusable {
                    char,
                    looks_like,
                    position,
                } => Some((*char, *looks_like, *position)),
                _ => None,
            })
            .collect();

        assert!(!confusables.is_empty());
        assert_eq!(confusables[0].1, 'a'); // Should look like 'a'
    }

    #[test]
    fn test_mixed_script_detection() {
        let checker = UnicodeSecurityChecker::strict();

        // Latin + Cyrillic
        let text = "Helloмир"; // "Hello" in Latin + "world" in Cyrillic
        let issues = checker.check(text);

        let mixed = issues.iter().find_map(|issue| match issue {
            UnicodeIssue::MixedScript { scripts } => Some(scripts),
            _ => None,
        });

        assert!(mixed.is_some());
        let scripts = mixed.unwrap();
        assert!(scripts.contains(&"Latin".to_string()));
        assert!(scripts.contains(&"Cyrillic".to_string()));
    }

    #[test]
    fn test_homoglyph_attack_detection() {
        let checker = UnicodeSecurityChecker::strict();

        // Mix of Latin and Cyrillic that looks like pure Latin
        let text = "pаypal"; // 'а' is Cyrillic
        let issues = checker.check(text);

        let homoglyph = issues.iter().find_map(|issue| match issue {
            UnicodeIssue::HomoglyphAttack {
                original,
                normalized,
            } => Some((original, normalized)),
            _ => None,
        });

        assert!(homoglyph.is_some());
        let (original, normalized) = homoglyph.unwrap();
        assert_eq!(normalized, "paypal"); // All Latin
        assert_ne!(original, normalized);
    }

    #[test]
    fn test_sanitize() {
        let checker = UnicodeSecurityChecker::strict();

        let text = "Hello\u{202E}world\u{200B}!";
        let safe = checker.sanitize(text);
        assert_eq!(safe, "Helloworld!");
        assert!(UnicodeSecurityChecker::is_ascii_safe(&safe));
    }

    #[test]
    fn test_is_ascii_safe() {
        assert!(UnicodeSecurityChecker::is_ascii_safe("Hello world"));
        assert!(UnicodeSecurityChecker::is_ascii_safe("test123"));
        assert!(!UnicodeSecurityChecker::is_ascii_safe("Привет"));
        assert!(!UnicodeSecurityChecker::is_ascii_safe("Hello\u{202E}"));
    }

    #[test]
    fn test_is_safe() {
        let checker = UnicodeSecurityChecker::strict();

        assert!(checker.is_safe("Hello world"));
        assert!(checker.is_safe("test123"));
        assert!(!checker.is_safe("Hello\u{202E}world"));
        assert!(!checker.is_safe("pаypal")); // Cyrillic 'а'
    }

    #[test]
    fn test_permissive_config() {
        let checker = UnicodeSecurityChecker::new(UnicodeSecurityConfig::permissive());

        // Should still detect BiDi
        let text = "evil\u{202E}txt.exe";
        let issues = checker.check(text);
        assert!(!issues.is_empty());

        // Should not detect confusables
        let text2 = "pаypal"; // Cyrillic 'а'
        let issues2 = checker.check(text2);
        // Should have BiDi/zero-width checks but not confusable checks
        let has_confusable = issues2.iter().any(|issue| {
            matches!(
                issue,
                UnicodeIssue::Confusable { .. } | UnicodeIssue::MixedScript { .. }
            )
        });
        assert!(!has_confusable);
    }

    #[test]
    fn test_all_bidi_controls() {
        let checker = UnicodeSecurityChecker::strict();

        for &bidi in BIDI_CONTROLS {
            let text = format!("test{}text", bidi);
            let issues = checker.check(&text);
            assert!(
                !issues.is_empty(),
                "Should detect BiDi control U+{:04X}",
                bidi as u32
            );
        }
    }

    #[test]
    fn test_all_zero_width() {
        let checker = UnicodeSecurityChecker::strict();

        for &zw in ZERO_WIDTH {
            let text = format!("test{}text", zw);
            let issues = checker.check(&text);
            assert!(
                !issues.is_empty(),
                "Should detect zero-width U+{:04X}",
                zw as u32
            );
        }
    }

    #[test]
    fn test_multiple_issues() {
        let checker = UnicodeSecurityChecker::strict();

        // BiDi + zero-width + confusable
        let text = "test\u{202E}\u{200B}pаypal";
        let issues = checker.check(text);

        assert!(issues.len() >= 3);

        let has_bidi = issues
            .iter()
            .any(|issue| matches!(issue, UnicodeIssue::BidiControl { .. }));
        let has_zw = issues
            .iter()
            .any(|issue| matches!(issue, UnicodeIssue::ZeroWidth { .. }));
        let has_confusable = issues
            .iter()
            .any(|issue| matches!(issue, UnicodeIssue::Confusable { .. }));

        assert!(has_bidi);
        assert!(has_zw);
        assert!(has_confusable);
    }

    #[test]
    fn test_display_formatting() {
        let issue1 = UnicodeIssue::BidiControl {
            char: '\u{202E}',
            position: 10,
            unicode: "U+202E".to_string(),
        };
        let display = issue1.to_string();
        assert!(display.contains("202E"));
        assert!(display.contains("10"));

        let issue2 = UnicodeIssue::Confusable {
            char: 'а',
            looks_like: 'a',
            position: 5,
        };
        let display2 = issue2.to_string();
        assert!(display2.contains("looks like"));
    }
}
