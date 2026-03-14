//! Utility functions for backend operations.
//!
//! Contains optimized helpers used across backend implementations,
//! including the line-number formatting function (ADR-103 A7).

use std::fmt::Write;

/// Format file content with line numbers in `cat -n` style.
///
/// Pre-calculates total output size and uses a single `String::with_capacity`
/// allocation to avoid intermediate allocations (ADR-103 A7).
///
/// Each line is formatted as: `{line_number:>6}\t{content}`
/// where line content is truncated to `max_line_len` characters.
pub fn format_content_with_line_numbers(
    content: &str,
    start_line: usize,
    max_line_len: usize,
) -> String {
    let lines: Vec<&str> = content.lines().collect();
    // Estimate: each line gets up to max_line_len chars + ~8 chars for line number + tab + newline
    let total_est: usize = lines.iter().map(|l| l.len().min(max_line_len) + 8).sum();
    let mut out = String::with_capacity(total_est);
    for (i, line) in lines.iter().enumerate() {
        if i > 0 {
            out.push('\n');
        }
        let truncated = &line[..line.len().min(max_line_len)];
        write!(out, "{:>6}\t{}", start_line + i, truncated).unwrap();
    }
    out
}

/// Sanitize a file path component, rejecting dangerous patterns.
///
/// Returns `true` if the path is safe, `false` if it contains
/// path traversal or other dangerous sequences.
pub fn is_safe_path_component(component: &str) -> bool {
    if component.is_empty() {
        return false;
    }
    if component == "." || component == ".." {
        return false;
    }
    if component.contains('\0') {
        return false;
    }
    true
}

/// Check if a path string contains traversal sequences.
pub fn contains_traversal(path: &str) -> bool {
    // Check for ".." components
    for component in path.split('/') {
        if component == ".." {
            return true;
        }
    }
    // Also check backslash-separated (Windows-style)
    for component in path.split('\\') {
        if component == ".." {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_empty_content() {
        let result = format_content_with_line_numbers("", 1, 2000);
        assert_eq!(result, "     1\t");
    }

    #[test]
    fn test_format_single_line() {
        let result = format_content_with_line_numbers("hello world", 1, 2000);
        assert_eq!(result, "     1\thello world");
    }

    #[test]
    fn test_format_multiple_lines() {
        let content = "line one\nline two\nline three";
        let result = format_content_with_line_numbers(content, 1, 2000);
        let expected = "     1\tline one\n     2\tline two\n     3\tline three";
        assert_eq!(result, expected);
    }

    #[test]
    fn test_format_with_offset() {
        let content = "first\nsecond";
        let result = format_content_with_line_numbers(content, 10, 2000);
        assert_eq!(result, "    10\tfirst\n    11\tsecond");
    }

    #[test]
    fn test_format_line_truncation() {
        let content = "abcdefghij";
        let result = format_content_with_line_numbers(content, 1, 5);
        assert_eq!(result, "     1\tabcde");
    }

    #[test]
    fn test_format_preserves_short_lines() {
        let content = "ab";
        let result = format_content_with_line_numbers(content, 1, 2000);
        assert_eq!(result, "     1\tab");
    }

    #[test]
    fn test_format_large_line_numbers() {
        let content = "data";
        let result = format_content_with_line_numbers(content, 999999, 2000);
        assert_eq!(result, "999999\tdata");
    }

    #[test]
    fn test_is_safe_path_component() {
        assert!(is_safe_path_component("file.rs"));
        assert!(is_safe_path_component("src"));
        assert!(!is_safe_path_component(""));
        assert!(!is_safe_path_component("."));
        assert!(!is_safe_path_component(".."));
        assert!(!is_safe_path_component("file\0.rs"));
    }

    #[test]
    fn test_contains_traversal() {
        assert!(contains_traversal("../etc/passwd"));
        assert!(contains_traversal("foo/../../bar"));
        assert!(contains_traversal("foo\\..\\bar"));
        assert!(!contains_traversal("foo/bar/baz"));
        assert!(!contains_traversal("foo/bar..baz"));
        assert!(!contains_traversal("..."));
    }

    #[test]
    fn test_format_correctness_many_lines() {
        let lines: Vec<String> = (0..100).map(|i| format!("line {}", i)).collect();
        let content = lines.join("\n");
        let result = format_content_with_line_numbers(&content, 1, 2000);
        let output_lines: Vec<&str> = result.lines().collect();
        assert_eq!(output_lines.len(), 100);
        assert!(output_lines[0].starts_with("     1\t"));
        assert!(output_lines[99].starts_with("   100\t"));
        assert!(output_lines[99].ends_with("line 99"));
    }
}
