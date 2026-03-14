//! Unicode security module (ADR-103 C7).
//!
//! Provides detection and stripping of dangerous Unicode characters
//! including BiDi controls, zero-width characters, and script confusable
//! homoglyphs. Full parity with Python's `unicode_security.py`.

use std::fmt;

/// Dangerous codepoints: BiDi directional formatting controls and zero-width characters.
pub const DANGEROUS_CODEPOINTS: &[char] = &[
    // BiDi directional formatting controls (U+202A-U+202E)
    '\u{202A}', // LEFT-TO-RIGHT EMBEDDING
    '\u{202B}', // RIGHT-TO-LEFT EMBEDDING
    '\u{202C}', // POP DIRECTIONAL FORMATTING
    '\u{202D}', // LEFT-TO-RIGHT OVERRIDE
    '\u{202E}', // RIGHT-TO-LEFT OVERRIDE
    // BiDi isolate controls (U+2066-U+2069)
    '\u{2066}', // LEFT-TO-RIGHT ISOLATE
    '\u{2067}', // RIGHT-TO-LEFT ISOLATE
    '\u{2068}', // FIRST STRONG ISOLATE
    '\u{2069}', // POP DIRECTIONAL ISOLATE
    // Zero-width characters
    '\u{200B}', // ZERO WIDTH SPACE
    '\u{200C}', // ZERO WIDTH NON-JOINER
    '\u{200D}', // ZERO WIDTH JOINER
    '\u{200E}', // LEFT-TO-RIGHT MARK
    '\u{200F}', // RIGHT-TO-LEFT MARK
    '\u{2060}', // WORD JOINER
    '\u{FEFF}', // ZERO WIDTH NO-BREAK SPACE (BOM)
];

/// A single Unicode security issue found in text.
#[derive(Debug, Clone, PartialEq)]
pub struct UnicodeIssue {
    /// Character position (byte offset) in the text.
    pub position: usize,
    /// The dangerous character found.
    pub character: char,
    /// Unicode codepoint as a string (e.g., "U+202E").
    pub codepoint: String,
    /// Human-readable description of the issue.
    pub description: String,
}

impl fmt::Display for UnicodeIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unicode issue at position {}: {} ({}) - {}",
            self.position, self.codepoint, self.character as u32, self.description
        )
    }
}

/// Result of URL safety checking.
#[derive(Debug, Clone, PartialEq)]
pub enum UrlSafetyResult {
    /// The URL is safe.
    Safe,
    /// The URL contains dangerous Unicode characters.
    DangerousChars(Vec<UnicodeIssue>),
    /// The URL contains mixed scripts (potential homoglyph attack).
    MixedScripts(String),
    /// The URL is invalid.
    Invalid(String),
}

/// Script category for confusable detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScriptCategory {
    Latin,
    Cyrillic,
    Greek,
    Armenian,
    Other,
}

/// Detect dangerous Unicode characters in the given text.
///
/// Returns a list of `UnicodeIssue` for each dangerous codepoint found.
pub fn detect_dangerous_unicode(text: &str) -> Vec<UnicodeIssue> {
    let mut issues = Vec::new();

    for (pos, ch) in text.char_indices() {
        if DANGEROUS_CODEPOINTS.contains(&ch) {
            let description = describe_dangerous_char(ch);
            issues.push(UnicodeIssue {
                position: pos,
                character: ch,
                codepoint: format!("U+{:04X}", ch as u32),
                description,
            });
        }
    }

    issues
}

/// Strip all dangerous Unicode characters from the given text.
pub fn strip_dangerous_unicode(text: &str) -> String {
    text.chars()
        .filter(|ch| !DANGEROUS_CODEPOINTS.contains(ch))
        .collect()
}

/// Check if a URL is safe from Unicode-based attacks.
///
/// Checks for:
/// - Dangerous Unicode codepoints
/// - Mixed-script content (potential homoglyph attacks)
pub fn check_url_safety(url: &str) -> UrlSafetyResult {
    if url.is_empty() {
        return UrlSafetyResult::Invalid("empty URL".to_string());
    }

    // Check for dangerous characters
    let issues = detect_dangerous_unicode(url);
    if !issues.is_empty() {
        return UrlSafetyResult::DangerousChars(issues);
    }

    // Check for mixed scripts in the domain part
    let domain = extract_domain(url);
    if let Some(domain) = domain {
        if has_mixed_scripts(domain) {
            return UrlSafetyResult::MixedScripts(format!(
                "Mixed scripts detected in domain: {}",
                domain
            ));
        }
    }

    UrlSafetyResult::Safe
}

/// Detect script category for a character (for confusable detection).
pub fn detect_script(ch: char) -> ScriptCategory {
    let cp = ch as u32;
    match cp {
        // Basic Latin + Latin Extended
        0x0041..=0x024F => ScriptCategory::Latin,
        // Latin Extended Additional
        0x1E00..=0x1EFF => ScriptCategory::Latin,
        // Cyrillic
        0x0400..=0x04FF => ScriptCategory::Cyrillic,
        // Cyrillic Supplement
        0x0500..=0x052F => ScriptCategory::Cyrillic,
        // Greek and Coptic
        0x0370..=0x03FF => ScriptCategory::Greek,
        // Armenian
        0x0530..=0x058F => ScriptCategory::Armenian,
        _ => ScriptCategory::Other,
    }
}

/// Known Cyrillic/Greek/Armenian characters that are confusable with Latin.
const CONFUSABLE_CHARS: &[(char, char, &str)] = &[
    // (confusable, latin_lookalike, description)
    ('\u{0410}', 'A', "Cyrillic A"),
    ('\u{0412}', 'B', "Cyrillic Ve"),
    ('\u{0421}', 'C', "Cyrillic Es"),
    ('\u{0415}', 'E', "Cyrillic Ie"),
    ('\u{041D}', 'H', "Cyrillic En"),
    ('\u{041A}', 'K', "Cyrillic Ka"),
    ('\u{041C}', 'M', "Cyrillic Em"),
    ('\u{041E}', 'O', "Cyrillic O"),
    ('\u{0420}', 'P', "Cyrillic Er"),
    ('\u{0422}', 'T', "Cyrillic Te"),
    ('\u{0425}', 'X', "Cyrillic Kha"),
    ('\u{0430}', 'a', "Cyrillic a"),
    ('\u{0435}', 'e', "Cyrillic ie"),
    ('\u{043E}', 'o', "Cyrillic o"),
    ('\u{0440}', 'p', "Cyrillic er"),
    ('\u{0441}', 'c', "Cyrillic es"),
    ('\u{0443}', 'y', "Cyrillic u"),
    ('\u{0445}', 'x', "Cyrillic kha"),
    // Greek
    ('\u{0391}', 'A', "Greek Alpha"),
    ('\u{0392}', 'B', "Greek Beta"),
    ('\u{0395}', 'E', "Greek Epsilon"),
    ('\u{0397}', 'H', "Greek Eta"),
    ('\u{0399}', 'I', "Greek Iota"),
    ('\u{039A}', 'K', "Greek Kappa"),
    ('\u{039C}', 'M', "Greek Mu"),
    ('\u{039D}', 'N', "Greek Nu"),
    ('\u{039F}', 'O', "Greek Omicron"),
    ('\u{03A1}', 'P', "Greek Rho"),
    ('\u{03A4}', 'T', "Greek Tau"),
    ('\u{03A5}', 'Y', "Greek Upsilon"),
    ('\u{03A7}', 'X', "Greek Chi"),
    ('\u{03B1}', 'a', "Greek alpha"),
    ('\u{03BF}', 'o', "Greek omicron"),
    // Armenian
    ('\u{0555}', 'O', "Armenian Oh"),
    ('\u{0585}', 'o', "Armenian oh"),
];

/// Check if a character is a known confusable homoglyph.
pub fn is_confusable(ch: char) -> Option<(char, &'static str)> {
    for &(confusable, latin, desc) in CONFUSABLE_CHARS {
        if ch == confusable {
            return Some((latin, desc));
        }
    }
    None
}

/// Detect confusable characters in text and return descriptions.
pub fn detect_confusables(text: &str) -> Vec<(usize, char, char, &'static str)> {
    let mut results = Vec::new();
    for (pos, ch) in text.char_indices() {
        if let Some((latin, desc)) = is_confusable(ch) {
            results.push((pos, ch, latin, desc));
        }
    }
    results
}

/// Validate that a string contains only ASCII identifier characters.
///
/// Valid identifiers: lowercase ASCII letters, digits, hyphens, underscores.
/// Must start with a letter. (ADR-103 C10)
pub fn validate_ascii_identifier(name: &str) -> bool {
    if name.is_empty() {
        return false;
    }

    let mut chars = name.chars();

    // First character must be an ASCII lowercase letter
    match chars.next() {
        Some(c) if c.is_ascii_lowercase() => {}
        _ => return false,
    }

    // Remaining characters: lowercase ASCII, digits, hyphens, underscores
    for c in chars {
        if c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_' {
            continue;
        }
        return false;
    }

    true
}

// --- Internal helpers ---

fn describe_dangerous_char(ch: char) -> String {
    match ch {
        '\u{202A}' => "LEFT-TO-RIGHT EMBEDDING".to_string(),
        '\u{202B}' => "RIGHT-TO-LEFT EMBEDDING".to_string(),
        '\u{202C}' => "POP DIRECTIONAL FORMATTING".to_string(),
        '\u{202D}' => "LEFT-TO-RIGHT OVERRIDE".to_string(),
        '\u{202E}' => "RIGHT-TO-LEFT OVERRIDE".to_string(),
        '\u{2066}' => "LEFT-TO-RIGHT ISOLATE".to_string(),
        '\u{2067}' => "RIGHT-TO-LEFT ISOLATE".to_string(),
        '\u{2068}' => "FIRST STRONG ISOLATE".to_string(),
        '\u{2069}' => "POP DIRECTIONAL ISOLATE".to_string(),
        '\u{200B}' => "ZERO WIDTH SPACE".to_string(),
        '\u{200C}' => "ZERO WIDTH NON-JOINER".to_string(),
        '\u{200D}' => "ZERO WIDTH JOINER".to_string(),
        '\u{200E}' => "LEFT-TO-RIGHT MARK".to_string(),
        '\u{200F}' => "RIGHT-TO-LEFT MARK".to_string(),
        '\u{2060}' => "WORD JOINER".to_string(),
        '\u{FEFF}' => "ZERO WIDTH NO-BREAK SPACE (BOM)".to_string(),
        _ => format!("dangerous codepoint U+{:04X}", ch as u32),
    }
}

fn extract_domain(url: &str) -> Option<&str> {
    let url = url.strip_prefix("https://").or_else(|| url.strip_prefix("http://"))?;
    let domain = url.split('/').next()?;
    // Strip port
    let domain = domain.split(':').next()?;
    // Strip userinfo
    let domain = if let Some(pos) = domain.rfind('@') {
        &domain[pos + 1..]
    } else {
        domain
    };
    if domain.is_empty() {
        None
    } else {
        Some(domain)
    }
}

fn has_mixed_scripts(domain: &str) -> bool {
    let mut has_latin = false;
    let mut has_cyrillic = false;
    let mut has_greek = false;
    let mut has_armenian = false;

    for ch in domain.chars() {
        if ch == '.' || ch == '-' || ch.is_ascii_digit() {
            continue;
        }
        match detect_script(ch) {
            ScriptCategory::Latin => has_latin = true,
            ScriptCategory::Cyrillic => has_cyrillic = true,
            ScriptCategory::Greek => has_greek = true,
            ScriptCategory::Armenian => has_armenian = true,
            ScriptCategory::Other => {}
        }
    }

    let script_count =
        has_latin as u8 + has_cyrillic as u8 + has_greek as u8 + has_armenian as u8;
    script_count > 1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_bidi_override() {
        let text = "normal\u{202E}reversed";
        let issues = detect_dangerous_unicode(text);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].character, '\u{202E}');
        assert_eq!(issues[0].codepoint, "U+202E");
        assert_eq!(issues[0].description, "RIGHT-TO-LEFT OVERRIDE");
    }

    #[test]
    fn test_detect_zero_width() {
        let text = "hello\u{200B}world";
        let issues = detect_dangerous_unicode(text);
        assert_eq!(issues.len(), 1);
        assert_eq!(issues[0].character, '\u{200B}');
    }

    #[test]
    fn test_detect_multiple_dangerous() {
        let text = "\u{202A}test\u{200D}\u{FEFF}";
        let issues = detect_dangerous_unicode(text);
        assert_eq!(issues.len(), 3);
    }

    #[test]
    fn test_detect_clean_text() {
        let text = "This is perfectly safe ASCII text with numbers 123.";
        let issues = detect_dangerous_unicode(text);
        assert!(issues.is_empty());
    }

    #[test]
    fn test_strip_dangerous() {
        let text = "he\u{200B}llo\u{202E} world";
        let stripped = strip_dangerous_unicode(text);
        assert_eq!(stripped, "hello world");
    }

    #[test]
    fn test_strip_preserves_safe_unicode() {
        let text = "caf\u{00E9}"; // cafe with accent
        let stripped = strip_dangerous_unicode(text);
        assert_eq!(stripped, "caf\u{00E9}");
    }

    #[test]
    fn test_url_safety_clean() {
        assert_eq!(check_url_safety("https://example.com"), UrlSafetyResult::Safe);
    }

    #[test]
    fn test_url_safety_empty() {
        assert!(matches!(check_url_safety(""), UrlSafetyResult::Invalid(_)));
    }

    #[test]
    fn test_url_safety_dangerous_chars() {
        let url = "https://exam\u{202E}ple.com";
        match check_url_safety(url) {
            UrlSafetyResult::DangerousChars(issues) => {
                assert_eq!(issues.len(), 1);
            }
            other => panic!("Expected DangerousChars, got {:?}", other),
        }
    }

    #[test]
    fn test_url_safety_mixed_scripts() {
        // Mix Latin and Cyrillic in domain
        let url = "https://exam\u{0440}le.com"; // Cyrillic 'р' looks like Latin 'p'
        match check_url_safety(url) {
            UrlSafetyResult::MixedScripts(_) => {}
            other => panic!("Expected MixedScripts, got {:?}", other),
        }
    }

    #[test]
    fn test_confusable_detection() {
        let text = "\u{0410}"; // Cyrillic A
        let confusables = detect_confusables(text);
        assert_eq!(confusables.len(), 1);
        assert_eq!(confusables[0].2, 'A'); // Latin lookalike
    }

    #[test]
    fn test_validate_ascii_identifier_valid() {
        assert!(validate_ascii_identifier("hello"));
        assert!(validate_ascii_identifier("my-skill"));
        assert!(validate_ascii_identifier("test_123"));
        assert!(validate_ascii_identifier("a"));
    }

    #[test]
    fn test_validate_ascii_identifier_invalid() {
        assert!(!validate_ascii_identifier(""));
        assert!(!validate_ascii_identifier("123abc")); // starts with digit
        assert!(!validate_ascii_identifier("Hello")); // uppercase
        assert!(!validate_ascii_identifier("-start")); // starts with hyphen
        assert!(!validate_ascii_identifier("na\u{0441}me")); // Cyrillic с
        assert!(!validate_ascii_identifier("café")); // non-ASCII
    }

    #[test]
    fn test_script_detection() {
        assert_eq!(detect_script('A'), ScriptCategory::Latin);
        assert_eq!(detect_script('z'), ScriptCategory::Latin);
        assert_eq!(detect_script('\u{0410}'), ScriptCategory::Cyrillic);
        assert_eq!(detect_script('\u{0391}'), ScriptCategory::Greek);
        assert_eq!(detect_script('\u{0531}'), ScriptCategory::Armenian);
        assert_eq!(detect_script('1'), ScriptCategory::Other);
    }

    #[test]
    fn test_all_dangerous_codepoints_detected() {
        let text: String = DANGEROUS_CODEPOINTS.iter().collect();
        let issues = detect_dangerous_unicode(&text);
        assert_eq!(issues.len(), DANGEROUS_CODEPOINTS.len());
    }

    #[test]
    fn test_extract_domain() {
        assert_eq!(extract_domain("https://example.com/path"), Some("example.com"));
        assert_eq!(extract_domain("http://user@host.com:8080/"), Some("host.com"));
        assert_eq!(extract_domain("ftp://nope"), None);
    }
}
