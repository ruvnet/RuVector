//! Code beautification for decompiled modules.
//!
//! Transforms minified code into readable, indented output with one
//! declaration per logical block.

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
    let mut lines = Vec::new();

    // Module header comment.
    lines.push(format!("// Module: {}", module.name));
    lines.push(String::new());

    for decl in &module.declarations {
        let (start, end) = decl.byte_range;
        let end = end.min(original_source.len());

        let raw = if start < end {
            &original_source[start..end]
        } else {
            ""
        };

        // Clean up and format the declaration.
        let formatted = format_declaration(decl, raw, inferred_names, min_confidence);
        lines.push(formatted);
        lines.push(String::new());
    }

    module.source = lines.join("\n");
}

/// Format a single declaration with indentation and name replacement.
fn format_declaration(
    decl: &Declaration,
    raw: &str,
    inferred_names: &[InferredName],
    min_confidence: f64,
) -> String {
    let mut code = raw.trim().to_string();

    // Strip leading separator characters.
    if code.starts_with(';') || code.starts_with('}') {
        code = code[1..].trim_start().to_string();
    }

    // Apply inferred name replacement for this declaration.
    if let Some(inf) = inferred_names
        .iter()
        .find(|n| n.original == decl.name && n.confidence >= min_confidence)
    {
        code = replace_identifier(&code, &decl.name, &inf.inferred);
        code = format!(
            "{} /* confidence: {:.0}% */",
            code,
            inf.confidence * 100.0
        );
    }

    // Add basic indentation for braces.
    code = indent_braces(&code);

    // Add a leading comment with the original minified name.
    if decl.name.len() <= 3 {
        format!("/* original: {} */ {}", decl.name, code)
    } else {
        code
    }
}

/// Replace all standalone occurrences of `old` with `new_name` in code.
fn replace_identifier(code: &str, old: &str, new_name: &str) -> String {
    // Simple word-boundary replacement. For short identifiers, be careful
    // not to replace substrings of longer identifiers.
    let mut result = String::with_capacity(code.len());
    let bytes = code.as_bytes();
    let old_bytes = old.as_bytes();
    let old_len = old_bytes.len();
    let mut i = 0;

    while i < bytes.len() {
        if i + old_len <= bytes.len() && &bytes[i..i + old_len] == old_bytes {
            // Check word boundaries.
            let before_ok =
                i == 0 || !is_ident_char(bytes[i - 1]);
            let after_ok =
                i + old_len >= bytes.len() || !is_ident_char(bytes[i + old_len]);

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
fn is_ident_char(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'$'
}

/// Add basic indentation for code inside braces.
fn indent_braces(code: &str) -> String {
    let mut result = String::with_capacity(code.len() + 64);
    let mut depth: usize = 0;
    let mut in_string = false;
    let mut string_char = '"';
    let mut prev_was_escape = false;

    for ch in code.chars() {
        if in_string {
            result.push(ch);
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
                result.push(ch);
            }
            '{' => {
                result.push(ch);
                result.push('\n');
                depth += 1;
                push_indent(&mut result, depth);
            }
            '}' => {
                result.push('\n');
                depth = depth.saturating_sub(1);
                push_indent(&mut result, depth);
                result.push(ch);
            }
            ';' => {
                result.push(ch);
                // Only add newline if we're inside braces.
                if depth > 0 {
                    result.push('\n');
                    push_indent(&mut result, depth);
                }
            }
            _ => {
                result.push(ch);
            }
        }
    }

    result
}

/// Push indentation spaces.
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
        // Should not replace "a" inside "bar".
        assert_eq!(
            replace_identifier("var bar = 1", "a", "x"),
            "var bar = 1"
        );
    }

    #[test]
    fn test_indent_braces() {
        let input = "function(){return 1}";
        let output = indent_braces(input);
        assert!(output.contains('\n'));
    }
}
