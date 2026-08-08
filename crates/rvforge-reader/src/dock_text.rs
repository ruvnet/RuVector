//! Sanitizing agent-authored text (ADR-295 §7).
//!
//! Agents supply exactly one string to the dock — their current task — and it
//! lands next to controls that carry a security meaning. Two things have to be
//! true of it before it is rendered:
//!
//! 1. It is inert. No escape sequence, control character, bidi override, or
//!    line break survives, and its length is capped, so it cannot repaint the
//!    terminal, reorder what is around it, or push chrome out of the pill.
//! 2. It is not pretending to be chrome. A task string reading "Paused" or
//!    "✓ Network disabled" is indistinguishable from the dock's own label once
//!    it sits beside the dock's own label, so it is withheld and flagged rather
//!    than rendered.
//!
//! Screening runs *after* cleaning, on the cleaned string, so an escape
//! sequence cannot be used to smuggle "Paused" past the check.
//!
//! [`sanitize_system_text`] is the same cleaning without the mimicry screen:
//! "Running → Paused" written by the runtime *is* the system label, and
//! screening the dock's own chrome would make the state indicator
//! unrenderable.

use serde::Serialize;

/// Agent task text is capped so a long string cannot push system chrome out of
/// the pill.
pub const MAX_TASK_CHARS: usize = 120;

/// Rendered in place of text that imitates system UI.
pub const WITHHELD_TEXT: &str = "(withheld: this text imitates a system status label)";

/// What the sanitizer had to do to a piece of agent text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case", tag = "kind", content = "detail")]
pub enum SanitizerFinding {
    ControlCharactersRemoved,
    AnsiEscapesRemoved,
    LineBreaksFlattened,
    Truncated,
    /// The text claims a system state. Rendering it would let the agent draw
    /// chrome (ADR-295 §7).
    SystemLabelMimicry(String),
}

/// Agent-authored text after sanitization.
///
/// [`Self::display`] is the only string the dock renders. When the text is
/// suspicious it is the withheld placeholder, and the cleaned original is
/// retained under [`Self::withheld`] for the expanded panel's audit list.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SanitizedText {
    display: String,
    withheld: Option<String>,
    suspicious: bool,
    findings: Vec<SanitizerFinding>,
}

impl SanitizedText {
    pub fn display(&self) -> &str {
        &self.display
    }
    pub fn withheld(&self) -> Option<&str> {
        self.withheld.as_deref()
    }
    pub fn is_suspicious(&self) -> bool {
        self.suspicious
    }
    pub fn findings(&self) -> &[SanitizerFinding] {
        &self.findings
    }
}

/// Words and glyphs that assert a system state. A task string containing one
/// of these is indistinguishable from chrome once it lands next to chrome, so
/// it is flagged rather than rendered.
const SYSTEM_LABEL_TOKENS: [&str; 18] = [
    "approved",
    "approve",
    "approval granted",
    "denied",
    "paused",
    "stopped",
    "terminated",
    "quarantined",
    "verified publisher",
    "verified",
    "network disabled",
    "witness valid",
    "witness chain valid",
    "sandboxed by rvforge",
    "rvforge",
    "system:",
    "permission granted",
    "capability granted",
];

/// State-claim glyphs. A leading "✓" reads as a system verdict whatever the
/// words after it say.
const SYSTEM_LABEL_GLYPHS: [char; 10] = ['✓', '✔', '✅', '☑', '⏸', '⏹', '⛔', '🛑', '❌', '🔒'];

/// Strip, flatten, cap, then screen for chrome mimicry.
///
/// Screening runs last, on the cleaned string, so an escape sequence cannot be
/// used to smuggle "Paused" past the check.
pub fn sanitize_agent_text(raw: &str) -> SanitizedText {
    clean(raw, true)
}

/// The same cleaning for text RVForge wrote itself.
///
/// System lines are *not* screened for mimicry: "Running → Paused" is the
/// system label, not an imitation of one, and screening chrome would make the
/// state indicator unrenderable. Cleaning still applies, because a policy rule
/// or an error message can carry a path or a payload the runtime did not
/// author.
pub fn sanitize_system_text(raw: &str) -> SanitizedText {
    clean(raw, false)
}

fn clean(raw: &str, screen_for_mimicry: bool) -> SanitizedText {
    let mut findings = Vec::new();

    let stripped = strip_ansi(raw, &mut findings);
    let flattened = flatten_and_strip_controls(&stripped, &mut findings);
    let mut text = collapse_whitespace(&flattened);

    if text.chars().count() > MAX_TASK_CHARS {
        text = text.chars().take(MAX_TASK_CHARS - 1).collect::<String>() + "…";
        findings.push(SanitizerFinding::Truncated);
    }

    if screen_for_mimicry {
        if let Some(token) = system_label_hit(&text) {
            findings.push(SanitizerFinding::SystemLabelMimicry(token));
            return SanitizedText {
                display: WITHHELD_TEXT.to_string(),
                withheld: Some(text),
                suspicious: true,
                findings,
            };
        }
    }

    SanitizedText {
        display: text,
        withheld: None,
        suspicious: false,
        findings,
    }
}

/// Remove CSI and OSC escape sequences, plus any stray ESC.
fn strip_ansi(raw: &str, findings: &mut Vec<SanitizerFinding>) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut removed = false;

    while let Some(c) = chars.next() {
        if c != '\u{1b}' {
            out.push(c);
            continue;
        }
        removed = true;
        match chars.peek() {
            // CSI: ESC [ params final-byte in 0x40..=0x7e.
            Some('[') => {
                chars.next();
                for c in chars.by_ref() {
                    if ('\u{40}'..='\u{7e}').contains(&c) {
                        break;
                    }
                }
            }
            // OSC: ESC ] ... BEL or ESC \.
            Some(']') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '\u{7}' {
                        break;
                    }
                    if c == '\u{1b}' && chars.peek() == Some(&'\\') {
                        chars.next();
                        break;
                    }
                }
            }
            // Any other two-character escape.
            Some(_) => {
                chars.next();
            }
            None => {}
        }
    }

    if removed {
        findings.push(SanitizerFinding::AnsiEscapesRemoved);
    }
    out
}

/// Newlines and tabs become spaces; every other control character and every
/// bidi/zero-width formatting character is dropped.
fn flatten_and_strip_controls(input: &str, findings: &mut Vec<SanitizerFinding>) -> String {
    let mut out = String::with_capacity(input.len());
    let mut flattened = false;
    let mut removed = false;

    for c in input.chars() {
        match c {
            '\n' | '\r' | '\t' | '\u{b}' | '\u{c}' | '\u{85}' | '\u{2028}' | '\u{2029}' => {
                flattened = true;
                out.push(' ');
            }
            // C0/C1 controls, zero-width and bidi-override formatting.
            c if c.is_control()
                || matches!(c, '\u{200b}'..='\u{200f}' | '\u{202a}'..='\u{202e}' | '\u{2066}'..='\u{2069}' | '\u{feff}') =>
            {
                removed = true;
            }
            c => out.push(c),
        }
    }

    if flattened {
        findings.push(SanitizerFinding::LineBreaksFlattened);
    }
    if removed {
        findings.push(SanitizerFinding::ControlCharactersRemoved);
    }
    out
}

fn collapse_whitespace(input: &str) -> String {
    input.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The token this text mimics, if any. Word-boundary matched so "unapproved
/// invoices" is ordinary task text while "Approved" is not.
fn system_label_hit(text: &str) -> Option<String> {
    if let Some(glyph) = text.chars().find(|c| SYSTEM_LABEL_GLYPHS.contains(c)) {
        return Some(glyph.to_string());
    }
    let lowered = text.to_ascii_lowercase();
    SYSTEM_LABEL_TOKENS
        .iter()
        .find(|token| contains_word(&lowered, token))
        .map(|token| (*token).to_string())
}

fn contains_word(haystack: &str, needle: &str) -> bool {
    let mut from = 0;
    while let Some(idx) = haystack[from..].find(needle) {
        let start = from + idx;
        let end = start + needle.len();
        let before_ok = start == 0
            || !haystack[..start]
                .chars()
                .next_back()
                .is_some_and(|c| c.is_alphanumeric());
        let after_ok = end == haystack.len()
            || !haystack[end..]
                .chars()
                .next()
                .is_some_and(|c| c.is_alphanumeric());
        if before_ok && after_ok {
            return true;
        }
        from = start + 1;
        if from >= haystack.len() {
            break;
        }
    }
    false
}
