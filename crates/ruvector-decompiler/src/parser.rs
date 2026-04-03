//! Regex-based JavaScript bundle parser.
//!
//! Extracts top-level declarations, string literals, property accesses,
//! and cross-references from minified JS without a full AST.

use std::collections::HashSet;

use once_cell::sync::Lazy;
use regex::Regex;

use crate::error::{DecompilerError, Result};
use crate::types::{DeclKind, Declaration};

// Cached compiled regexes -- compiled once, reused across all calls.
static VAR_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:^|[;}\s])(var|let|const)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=")
        .expect("valid regex")
});

static FN_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:^|[;}\s])function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(")
        .expect("valid regex")
});

static CLASS_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:^|[;}\s])class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*[{\(]")
        .expect("valid regex")
});

static EXPORT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?:^|[;}\s])export\s+(?:default\s+)?(?:function|class|const|let|var)\s+([a-zA-Z_$][a-zA-Z0-9_$]*)")
        .expect("valid regex")
});

static STRING_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r#""([^"\\]*(?:\\.[^"\\]*)*)"|'([^'\\]*(?:\\.[^'\\]*)*)'"#)
        .expect("valid regex")
});

static PROP_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\.([a-zA-Z_$][a-zA-Z0-9_$]*)").expect("valid regex")
});

static IDENT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"\b([a-zA-Z_$][a-zA-Z0-9_$]*)\b").expect("valid regex")
});

/// Parse a minified JavaScript bundle and extract declarations.
pub fn parse_bundle(source: &str) -> Result<Vec<Declaration>> {
    if source.trim().is_empty() {
        return Err(DecompilerError::EmptyBundle(
            "source is empty".to_string(),
        ));
    }

    let decls = extract_declarations(source);
    if decls.is_empty() {
        return Err(DecompilerError::NoDeclarations);
    }

    Ok(decls)
}

/// Extract top-level declarations from source using regex heuristics.
fn extract_declarations(source: &str) -> Vec<Declaration> {
    let mut declarations = Vec::new();

    // Use HashSet for O(1) name lookups during cross-reference detection.
    let mut all_names: HashSet<String> = HashSet::new();

    // --- var/let/const ---
    for cap in VAR_RE.captures_iter(source) {
        let kind = match &cap[1] {
            "var" => DeclKind::Var,
            "let" => DeclKind::Let,
            "const" => DeclKind::Const,
            _ => continue,
        };
        let name = cap[2].to_string();
        let match_start = cap.get(0).map_or(0, |m| m.start());
        let body_end = find_declaration_end(source, match_start);

        all_names.insert(name.clone());
        declarations.push(Declaration {
            name,
            kind,
            byte_range: (match_start, body_end),
            string_literals: Vec::new(),
            property_accesses: Vec::new(),
            references: Vec::new(),
        });
    }

    // --- function ---
    for cap in FN_RE.captures_iter(source) {
        let name = cap[1].to_string();
        let match_start = cap.get(0).map_or(0, |m| m.start());
        let body_end = find_declaration_end(source, match_start);

        all_names.insert(name.clone());
        declarations.push(Declaration {
            name,
            kind: DeclKind::Function,
            byte_range: (match_start, body_end),
            string_literals: Vec::new(),
            property_accesses: Vec::new(),
            references: Vec::new(),
        });
    }

    // --- class ---
    for cap in CLASS_RE.captures_iter(source) {
        let name = cap[1].to_string();
        let match_start = cap.get(0).map_or(0, |m| m.start());
        let body_end = find_declaration_end(source, match_start);

        all_names.insert(name.clone());
        declarations.push(Declaration {
            name,
            kind: DeclKind::Class,
            byte_range: (match_start, body_end),
            string_literals: Vec::new(),
            property_accesses: Vec::new(),
            references: Vec::new(),
        });
    }

    // --- export declarations (ES modules) ---
    for cap in EXPORT_RE.captures_iter(source) {
        let name = cap[1].to_string();
        // Skip if already captured by var/fn/class regex.
        if all_names.contains(&name) {
            continue;
        }
        let match_start = cap.get(0).map_or(0, |m| m.start());
        let body_end = find_declaration_end(source, match_start);

        all_names.insert(name.clone());
        declarations.push(Declaration {
            name,
            kind: DeclKind::Const, // Treat exports as const by default.
            byte_range: (match_start, body_end),
            string_literals: Vec::new(),
            property_accesses: Vec::new(),
            references: Vec::new(),
        });
    }

    // Second pass: extract metadata for each declaration.
    for decl in &mut declarations {
        let (start, end) = decl.byte_range;
        let end = end.min(source.len());
        let body = &source[start..end];

        // Extract string literals.
        for cap in STRING_RE.captures_iter(body) {
            let s = cap
                .get(1)
                .or_else(|| cap.get(2))
                .map(|m| m.as_str().to_string())
                .unwrap_or_default();
            if !s.is_empty() {
                decl.string_literals.push(s);
            }
        }

        // Extract property accesses (use HashSet for dedup).
        let mut seen_props: HashSet<String> = HashSet::new();
        for cap in PROP_RE.captures_iter(body) {
            let prop = cap[1].to_string();
            if seen_props.insert(prop.clone()) {
                decl.property_accesses.push(prop);
            }
        }

        // Extract cross-references to other declarations (use HashSet for dedup).
        let mut seen_refs: HashSet<String> = HashSet::new();
        for cap in IDENT_RE.captures_iter(body) {
            let ident = &cap[1];
            if ident != decl.name
                && all_names.contains(ident)
                && seen_refs.insert(ident.to_string())
            {
                decl.references.push(ident.to_string());
            }
        }
    }

    declarations
}

/// Find the end of a declaration body by tracking brace depth,
/// or falling back to the next semicolon at depth 0.
fn find_declaration_end(source: &str, start: usize) -> usize {
    let bytes = source.as_bytes();
    let mut brace_depth = 0i32;
    let mut paren_depth = 0i32;
    let mut in_string = false;
    let mut string_char = b'"';
    let mut found_brace = false;

    let mut i = start;
    while i < bytes.len() {
        let ch = bytes[i];

        // Handle string escapes.
        if in_string {
            if ch == b'\\' {
                i += 2;
                continue;
            }
            if ch == string_char {
                in_string = false;
            }
            i += 1;
            continue;
        }

        match ch {
            b'"' | b'\'' | b'`' => {
                in_string = true;
                string_char = ch;
            }
            b'{' => {
                brace_depth += 1;
                found_brace = true;
            }
            b'}' => {
                brace_depth -= 1;
                if found_brace && brace_depth <= 0 {
                    // Consume trailing semicolon if present.
                    if i + 1 < bytes.len() && bytes[i + 1] == b';' {
                        return i + 2;
                    }
                    return i + 1;
                }
            }
            b'(' => {
                paren_depth += 1;
            }
            b')' => {
                paren_depth -= 1;
            }
            b';' if brace_depth <= 0 && paren_depth <= 0 && i > start + 2 => {
                return i + 1;
            }
            _ => {}
        }

        i += 1;
    }

    source.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_var_declarations() {
        let src = r#"var a=function(){return"hello"};var b=42;"#;
        let decls = parse_bundle(src).unwrap();
        assert!(decls.len() >= 2);
        assert_eq!(decls[0].name, "a");
        assert_eq!(decls[0].kind, DeclKind::Var);
        assert!(decls[0].string_literals.contains(&"hello".to_string()));
    }

    #[test]
    fn test_parse_class() {
        let src = r#"var x=1;class Foo{constructor(){this.name="test"}}"#;
        let decls = parse_bundle(src).unwrap();
        let class_decl = decls.iter().find(|d| d.kind == DeclKind::Class);
        assert!(class_decl.is_some());
        assert_eq!(class_decl.unwrap().name, "Foo");
    }

    #[test]
    fn test_cross_references() {
        let src = r#"var a=function(){return 1};var b=function(){return a()}"#;
        let decls = parse_bundle(src).unwrap();
        let b_decl = decls.iter().find(|d| d.name == "b").unwrap();
        assert!(b_decl.references.contains(&"a".to_string()));
    }

    #[test]
    fn test_empty_bundle() {
        let result = parse_bundle("");
        assert!(result.is_err());
    }
}
