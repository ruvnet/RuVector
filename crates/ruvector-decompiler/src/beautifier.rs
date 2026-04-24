//! Code beautification for decompiled modules.
//!
//! Transforms minified code into readable, indented output with one
//! declaration per logical block.
//!
//! Memory optimization: Works on `&str` slices from the original source
//! instead of copying strings. Only materializes the final beautified
//! output once per module.

use crate::types::{Declaration, InferredName, Module};

/// Beautify a module's source code from the original bundle.
///
/// Extracts each declaration's source from the original bundle and formats
/// it with proper indentation and spacing. Applies inferred name replacements
/// where confidence exceeds the threshold.
pub fn beautify_module(
    module: &mut Module,
    original_source: &str,
    inferred_names: &[InferredName],
    min_confidence: f64,
) {
    // Pre-compute estimated output size to avoid repeated reallocations.
    let estimated_size = module
        .declarations
        .iter()
        .map(|d| d.byte_range.1.saturating_sub(d.byte_range.0) + 64)
        .sum::<usize>()
        + 128;

    let mut output = String::with_capacity(estimated_size);

    // Module header comment.
    output.push_str("// Module: ");
    output.push_str(&module.name);
    output.push('\n');
    output.push('\n');

    for decl in &module.declarations {
        let (start, end) = decl.byte_range;
        let end = end.min(original_source.len());

        let raw = if start < end {
            &original_source[start..end]
        } else {
            ""
        };

        // Format the declaration directly into the output buffer.
        format_declaration_into(&mut output, decl, raw, inferred_names, min_confidence);
        output.push('\n');
        output.push('\n');
    }

    module.source = output;
}

/// Format a single declaration with indentation and name replacement,
/// writing directly into the output buffer to avoid intermediate allocations.
fn format_declaration_into(
    out: &mut String,
    decl: &Declaration,
    raw: &str,
    inferred_names: &[InferredName],
    min_confidence: f64,
) {
    let trimmed = raw.trim();

    // Strip leading separator characters.
    let code = if trimmed.starts_with(';') || trimmed.starts_with('}') {
        trimmed[1..].trim_start()
    } else {
        trimmed
    };

    // Find the inferred name for this declaration (if any).
    let inf_name = inferred_names
        .iter()
        .find(|n| n.original == decl.name && n.confidence >= min_confidence);

    // Add leading comment with original minified name if it's short.
    if decl.name.len() <= 3 {
        out.push_str("/* original: ");
        out.push_str(&decl.name);
        out.push_str(" */ ");
    }

    // Apply name replacement and indentation.
    if let Some(inf) = inf_name {
        let replaced = replace_identifier(code, &decl.name, &inf.inferred);
        indent_braces_into(out, &replaced);
        out.push_str(&format!(
            " /* confidence: {:.0}% */",
            inf.confidence * 100.0
        ));
    } else {
        indent_braces_into(out, code);
    }
}

/// Replace all standalone occurrences of `old` with `new_name` in code.
fn replace_identifier(code: &str, old: &str, new_name: &str) -> String {
    let mut result = String::with_capacity(code.len());
    let bytes = code.as_bytes();
    let old_bytes = old.as_bytes();
    let old_len = old_bytes.len();
    let mut i = 0;

    while i < bytes.len() {
        if i + old_len <= bytes.len() && &bytes[i..i + old_len] == old_bytes {
            let before_ok = i == 0 || !is_ident_char(bytes[i - 1]);
            let after_ok = i + old_len >= bytes.len() || !is_ident_char(bytes[i + old_len]);

            if before_ok && after_ok {
                result.push_str(new_name);
                i += old_len;
                continue;
            }
        }
        result.push(bytes[i] as char);
        i += 1;
    }

    result
}

/// Check if a byte is a valid JS identifier character.
#[inline]
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Add basic indentation for code inside braces, writing directly
/// into the output buffer.
fn indent_braces_into(out: &mut String, code: &str) {
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut prev_was_escape = false;

    for ch in code.chars() {
        if in_string {
            out.push(ch);
            if prev_was_escape {
                prev_was_escape = false;
                continue;
            }
            if ch == '\\' {
                prev_was_escape = true;
                continue;
            }
            if ch == string_char {
                in_string = false;
            }
            continue;
        }

        match ch {
            '"' | '\'' | '`' => {
                in_string = true;
                string_char = ch;
                out.push(ch);
            }
            '{' => {
                out.push(ch);
                out.push('\n');
                depth += 1;
                push_indent(out, depth);
            }
            '}' => {
                out.push('\n');
                depth = depth.saturating_sub(1);
                push_indent(out, depth);
                out.push(ch);
            }
            ';' => {
                out.push(ch);
                if depth > 0 {
                    out.push('\n');
                    push_indent(out, depth);
                }
            }
            _ => {
                out.push(ch);
            }
        }
    }
}

/// Push indentation spaces.
#[inline]
fn push_indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

/// Beautify all modules in place.
pub fn beautify_all(
    modules: &mut [Module],
    original_source: &str,
    inferred_names: &[InferredName],
    min_confidence: f64,
) {
    for module in modules.iter_mut() {
        beautify_module(module, original_source, inferred_names, min_confidence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replace_identifier() {
        assert_eq!(
            replace_identifier("var a = a + 1", "a", "counter"),
            "var counter = counter + 1"
        );
    }

    #[test]
    fn test_replace_no_substring() {
        assert_eq!(replace_identifier("var bar = 1", "a", "x"), "var bar = 1");
    }

    #[test]
    fn test_indent_braces() {
        let input = "function(){return 1}";
        let mut output = String::new();
        indent_braces_into(&mut output, input);
        assert!(output.contains('\n'));
    }
}
