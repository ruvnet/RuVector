//! UTF-8-safe text truncation helpers.
//!
//! `str` indexing in Rust is by **byte** offset, and slicing at an offset that
//! is not a `char` boundary panics. Every `&s[..N]` in this crate that bounds a
//! user-supplied string by a byte budget was therefore a latent panic: a title
//! whose 120th byte lands mid-emoji (or mid-CJK, or mid-accented-letter) takes
//! the whole request thread down with it.
//!
//! `str::floor_char_boundary` would do exactly this job, but it is still
//! unstable, so this module provides a stable equivalent.

/// Truncate `s` to at most `max_bytes` bytes, rounding **down** to the nearest
/// `char` boundary so the result is always valid UTF-8.
///
/// This preserves the byte-budget semantics of the `&s[..N]` calls it replaces
/// — the result is never longer than `max_bytes` — and differs from the old
/// code only in the cases where the old code would have panicked.
///
/// ```
/// use mcp_brain_server::text::truncate_at_char_boundary;
/// assert_eq!(truncate_at_char_boundary("hello", 3), "hel");
/// assert_eq!(truncate_at_char_boundary("héllo", 2), "h"); // 'é' is 2 bytes
/// assert_eq!(truncate_at_char_boundary("hi", 99), "hi");
/// ```
#[must_use]
pub fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    // Walk back from max_bytes to the nearest char boundary. At most 3 steps,
    // since the longest UTF-8 encoding is 4 bytes.
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

#[cfg(test)]
mod tests {
    use super::truncate_at_char_boundary;

    /// Each of these inputs is constructed so that byte index `max` falls
    /// strictly *inside* a multi-byte character. The pre-fix code
    /// (`&s[..max]`) panics with "byte index N is not a char boundary" on
    /// every one of them.
    #[test]
    fn does_not_panic_when_budget_splits_a_multibyte_char() {
        // 2-byte char (Latin-1 supplement): boundary at 120 splits 'é'.
        let accented = format!("{}é", "a".repeat(119));
        assert_eq!(truncate_at_char_boundary(&accented, 120).len(), 119);

        // 3-byte char (CJK): boundary at 120 splits '中' (starts at 119).
        let cjk = format!("{}中", "a".repeat(119));
        assert_eq!(truncate_at_char_boundary(&cjk, 120).len(), 119);

        // 4-byte char (emoji): boundary at 120 splits '😀' (starts at 118).
        let emoji = format!("{}😀", "a".repeat(118));
        assert_eq!(truncate_at_char_boundary(&emoji, 120).len(), 118);

        // Split at every interior byte of a 4-byte char.
        for max in 119..=121 {
            let s = format!("{}😀", "a".repeat(118));
            let out = truncate_at_char_boundary(&s, max);
            assert_eq!(out, "a".repeat(118), "max={max}");
        }
    }

    #[test]
    fn returns_whole_string_when_under_budget() {
        assert_eq!(truncate_at_char_boundary("hello", 120), "hello");
        assert_eq!(truncate_at_char_boundary("😀😀", 8), "😀😀");
        assert_eq!(truncate_at_char_boundary("", 0), "");
    }

    #[test]
    fn truncates_exactly_at_a_boundary_when_budget_lands_on_one() {
        assert_eq!(truncate_at_char_boundary("abcdef", 3), "abc");
        // 4 bytes == exactly one emoji.
        assert_eq!(truncate_at_char_boundary("😀😀", 4), "😀");
    }

    #[test]
    fn zero_budget_yields_empty() {
        assert_eq!(truncate_at_char_boundary("😀", 0), "");
        assert_eq!(truncate_at_char_boundary("abc", 0), "");
    }

    /// The result is always valid UTF-8 and never exceeds the budget, for
    /// every budget across a mixed-width string.
    #[test]
    fn never_exceeds_budget_and_always_valid_for_all_offsets() {
        let s = "aé中😀b£€𝄞z";
        for max in 0..=s.len() + 4 {
            let out = truncate_at_char_boundary(s, max);
            assert!(out.len() <= max.min(s.len()), "max={max}");
            assert!(s.starts_with(out), "max={max}");
        }
    }
}
