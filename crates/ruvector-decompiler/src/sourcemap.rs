//! V3 source map generation with VLQ encoding.
//!
//! Generates standard source maps that can be loaded in browser DevTools.

use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::types::{InferredName, Module};

/// A V3 source map.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMap {
    /// Always 3.
    pub version: u32,
    /// Output filename.
    pub file: String,
    /// Reconstructed source filenames.
    pub sources: Vec<String>,
    /// Optional source contents (for inline source maps).
    #[serde(rename = "sourcesContent", skip_serializing_if = "Option::is_none")]
    pub sources_content: Option<Vec<String>>,
    /// Inferred original names.
    pub names: Vec<String>,
    /// VLQ-encoded mapping segments.
    pub mappings: String,
}

/// Generate a source map for a single module.
pub fn generate_source_map(
    module: &Module,
    inferred_names: &[InferredName],
    output_filename: &str,
) -> Result<String> {
    let source_filename = format!("{}.js", module.name);

    // Collect names relevant to this module.
    let module_decl_names: Vec<&str> = module
        .declarations
        .iter()
        .map(|d| d.name.as_str())
        .collect();

    let names: Vec<String> = inferred_names
        .iter()
        .filter(|n| module_decl_names.contains(&n.original.as_str()))
        .map(|n| n.inferred.clone())
        .collect();

    // Build VLQ mappings.
    // We map each line/column in the beautified output back to positions
    // in the original source.
    let mappings = build_vlq_mappings(module);

    let sm = SourceMap {
        version: 3,
        file: output_filename.to_string(),
        sources: vec![source_filename],
        sources_content: Some(vec![module.source.clone()]),
        names,
        mappings,
    };

    let json = serde_json::to_string(&sm)?;
    Ok(json)
}

/// Build VLQ-encoded mappings for a module's beautified source.
///
/// Each segment maps (generated_col) -> (source_idx, source_line, source_col, name_idx).
/// We generate a simple 1:1 line mapping since our beautifier produces one
/// declaration per line.
fn build_vlq_mappings(module: &Module) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut prev_gen_col: i64 = 0;
    let mut prev_src_line: i64 = 0;
    let mut prev_src_col: i64 = 0;
    let mut prev_name_idx: i64 = 0;

    for (i, decl) in module.declarations.iter().enumerate() {
        let gen_col: i64 = 0; // Each declaration starts at column 0 after beautification.
        let src_line: i64 = 0; // Original is typically one line (minified).
        let src_col: i64 = decl.byte_range.0 as i64;
        let name_idx: i64 = i as i64;

        let mut segment = String::new();
        // Field 1: generated column (relative).
        encode_vlq(gen_col - prev_gen_col, &mut segment);
        // Field 2: source index (always 0, relative).
        encode_vlq(0, &mut segment);
        // Field 3: original line (relative).
        encode_vlq(src_line - prev_src_line, &mut segment);
        // Field 4: original column (relative).
        encode_vlq(src_col - prev_src_col, &mut segment);
        // Field 5: name index (relative).
        encode_vlq(name_idx - prev_name_idx, &mut segment);

        prev_src_line = src_line;
        prev_src_col = src_col;
        prev_name_idx = name_idx;

        lines.push(segment);

        // Reset generated column tracking per line.
        prev_gen_col = 0;
    }

    lines.join(";")
}

/// VLQ Base64 encoding for source map segments.
///
/// Encodes a signed integer into a VLQ Base64 string.
fn encode_vlq(value: i64, out: &mut String) {
    const VLQ_BASE64: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const VLQ_BASE_SHIFT: u32 = 5;
    const VLQ_BASE: i64 = 1 << VLQ_BASE_SHIFT;
    const VLQ_BASE_MASK: i64 = VLQ_BASE - 1;
    const VLQ_CONTINUATION_BIT: i64 = VLQ_BASE;

    // Convert to VLQ signed representation.
    let mut vlq = if value < 0 {
        ((-value) << 1) + 1
    } else {
        value << 1
    };

    loop {
        let mut digit = vlq & VLQ_BASE_MASK;
        vlq >>= VLQ_BASE_SHIFT;

        if vlq > 0 {
            digit |= VLQ_CONTINUATION_BIT;
        }

        out.push(VLQ_BASE64[digit as usize] as char);

        if vlq == 0 {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{DeclKind, Declaration, Module};

    #[test]
    fn test_vlq_encode_zero() {
        let mut out = String::new();
        encode_vlq(0, &mut out);
        assert_eq!(out, "A");
    }

    #[test]
    fn test_vlq_encode_positive() {
        let mut out = String::new();
        encode_vlq(1, &mut out);
        assert_eq!(out, "C");
    }

    #[test]
    fn test_vlq_encode_negative() {
        let mut out = String::new();
        encode_vlq(-1, &mut out);
        assert_eq!(out, "D");
    }

    #[test]
    fn test_generate_source_map() {
        let module = Module {
            name: "test_module".to_string(),
            index: 0,
            declarations: vec![Declaration {
                name: "a".to_string(),
                kind: DeclKind::Var,
                byte_range: (0, 20),
                string_literals: vec![],
                property_accesses: vec![],
                references: vec![],
            }],
            source: "var a = function() { return 'hello'; };".to_string(),
            byte_range: (0, 20),
        };

        let names = vec![InferredName {
            original: "a".to_string(),
            inferred: "greeter".to_string(),
            confidence: 0.9,
            evidence: vec![],
        }];

        let json = generate_source_map(&module, &names, "output.js").unwrap();
        let parsed: SourceMap = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.version, 3);
        assert_eq!(parsed.sources, vec!["test_module.js"]);
        assert_eq!(parsed.names, vec!["greeter"]);
    }
}
