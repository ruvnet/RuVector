//! Diagnostics for failed string edits (ADR-273 §3.1, §3.3).
//!
//! "Error: old_string not found" is the single highest-frequency tool failure
//! in an editing agent, and it is nearly useless on its own: the model already
//! believed the string was there, so restating that it isn't gives it nothing
//! to change. It then retries a near-identical call, which is the input
//! condition for loop detection.
//!
//! Edit-tool ergonomics are load-bearing rather than incidental — in the
//! published reproductions, only the flavor with real failure diagnostics moved
//! the benchmark number; a plain `edit`/`write_file` pair gave no improvement
//! at all. So this module works out *why* a match failed and says so.

/// Maximum characters of a candidate line echoed back in a diagnostic.
const SNIPPET_LEN: usize = 160;

/// Explain why `old_string` did not match anything in `content`.
///
/// Returns an actionable message naming the likely cause. Ordered by how
/// common the cause is in practice, so the first plausible explanation is the
/// most likely one.
pub fn diagnose_edit_failure(content: &str, old_string: &str, path: &str) -> String {
    let base = format!("Error: old_string not found in {path}.");

    // 1. Line endings. Invisible, and it defeats an otherwise exact match.
    if content.contains("\r\n") && !old_string.contains("\r\n") {
        let normalized = old_string.replace('\n', "\r\n");
        if content.contains(&normalized) {
            return format!(
                "{base} The file uses CRLF line endings but old_string uses LF. \
                 The text is present — re-read the file and copy the exact bytes."
            );
        }
    }

    // 2. Whitespace. Indentation drift is the classic cause: the model
    //    reconstructs the line from memory and gets the leading spaces wrong.
    let squeeze = |s: &str| -> String { s.split_whitespace().collect::<Vec<_>>().join(" ") };
    let squeezed_old = squeeze(old_string);
    if !squeezed_old.is_empty() && squeeze(content).contains(&squeezed_old) {
        return format!(
            "{base} A match exists when whitespace is ignored, so the difference is \
             indentation, trailing spaces, or tabs-vs-spaces. Re-read the file and \
             copy the exact leading whitespace."
        );
    }

    // 3. Case.
    if content.to_lowercase().contains(&old_string.to_lowercase()) {
        return format!(
            "{base} A match exists ignoring case — the difference is capitalization only."
        );
    }

    // 4. Partial match: locate the anchor line and show what is actually there.
    //    This is the most useful case, because it hands the model the real text.
    if let Some(hint) = nearest_line_hint(content, old_string) {
        return format!("{base} {hint}");
    }

    format!(
        "{base} No similar text was found. The file may not contain this code at all — \
         read the file before editing it, rather than assuming its contents."
    )
}

/// Find the line in `content` most similar to the first line of `old_string`,
/// and describe the mismatch.
fn nearest_line_hint(content: &str, old_string: &str) -> Option<String> {
    let needle = old_string.lines().next()?.trim();
    if needle.len() < 4 {
        // Too short to attribute a near-match to anything meaningful.
        return None;
    }

    let mut best: Option<(usize, usize, &str)> = None; // (score, line_no, text)
    for (i, line) in content.lines().enumerate() {
        let score = shared_prefix_len(line.trim(), needle);
        if score >= 4 && best.map(|(b, _, _)| score > b).unwrap_or(true) {
            best = Some((score, i + 1, line));
        }
    }

    let (_, line_no, text) = best?;
    Some(format!(
        "The closest line in the file is line {line_no}: {:?}. \
         Copy it exactly, including whitespace.",
        truncate(text.trim_end(), SNIPPET_LEN)
    ))
}

/// Length of the common prefix of two strings, in characters.
fn shared_prefix_len(a: &str, b: &str) -> usize {
    a.chars().zip(b.chars()).take_while(|(x, y)| x == y).count()
}

/// Truncate on a character boundary, marking the cut.
fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let cut: String = s.chars().take(max).collect();
    format!("{cut}…")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_indentation_mismatch() {
        let content = "fn main() {\n    let x = 1;\n}\n";
        // Wrong indentation, not missing: matching is substring-based, so an
        // omitted indent still matches and never reaches this diagnostic.
        let msg = diagnose_edit_failure(content, "        let x = 1;", "a.rs");
        assert!(msg.contains("whitespace is ignored"), "got: {msg}");
        assert!(msg.contains("exact leading whitespace"));
    }

    #[test]
    fn detects_tabs_versus_spaces() {
        let content = "fn main() {\n\tlet x = 1;\n}\n";
        let msg = diagnose_edit_failure(content, "    let x = 1;", "a.rs");
        assert!(msg.contains("tabs-vs-spaces"), "got: {msg}");
    }

    #[test]
    fn detects_crlf_mismatch() {
        let content = "line one\r\nline two\r\n";
        let msg = diagnose_edit_failure(content, "line one\nline two", "a.txt");
        assert!(msg.contains("CRLF"), "got: {msg}");
        assert!(msg.contains("The text is present"));
    }

    #[test]
    fn detects_case_mismatch() {
        let content = "let Value = 1;\n";
        let msg = diagnose_edit_failure(content, "let value = 1;", "a.rs");
        assert!(msg.contains("capitalization"), "got: {msg}");
    }

    #[test]
    fn shows_the_nearest_line_when_content_drifted() {
        let content = "fn compute(a: u32, b: u32) -> u32 {\n    a + b\n}\n";
        // Same opening but a different signature — the most common real case.
        let msg = diagnose_edit_failure(content, "fn compute(a: u32) -> u32 {", "a.rs");
        assert!(
            msg.contains("closest line in the file is line 1"),
            "got: {msg}"
        );
        assert!(msg.contains("fn compute(a: u32, b: u32)"), "got: {msg}");
    }

    #[test]
    fn says_so_plainly_when_nothing_is_close() {
        let content = "completely unrelated file contents\n";
        let msg = diagnose_edit_failure(content, "fn transmogrify() {", "a.rs");
        assert!(msg.contains("No similar text was found"), "got: {msg}");
        assert!(msg.contains("read the file before editing"));
    }

    #[test]
    fn always_names_the_path() {
        let msg = diagnose_edit_failure("x", "y", "src/lib.rs");
        assert!(msg.contains("src/lib.rs"));
    }

    #[test]
    fn handles_multibyte_content_without_panicking() {
        let content = "let s = \"héllo 🙂 wörld\";\n";
        let msg = diagnose_edit_failure(content, "let s = \"hello world\";", "a.rs");
        assert!(!msg.is_empty());
    }

    #[test]
    fn does_not_guess_from_a_trivially_short_needle() {
        let content = "aaaa bbbb\ncccc dddd\n";
        // A 2-char needle would "nearly match" almost any line; it must not
        // produce a confident and wrong nearest-line claim.
        let msg = diagnose_edit_failure(content, "xy", "a.txt");
        assert!(msg.contains("No similar text was found"), "got: {msg}");
    }

    #[test]
    fn truncates_a_very_long_candidate_line() {
        let long = "x".repeat(500);
        let content = format!("prefix_{long}\n");
        let msg = diagnose_edit_failure(&content, &format!("prefix_{}", "y".repeat(20)), "a.txt");
        assert!(
            msg.len() < 400,
            "diagnostic should stay compact: {}",
            msg.len()
        );
    }
}
