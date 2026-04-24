//! Integration tests for the ruvector-decompiler crate.
//!
//! Tests the full pipeline with a small minified bundle sample.

use ruvector_decompiler::{decompile, DecompileConfig};

/// A small minified bundle with 3 declarations and cross-references.
const SAMPLE_BUNDLE: &str = r#"var a=function(){return"hello"};var b=class{constructor(){this.name="test"}};var c=function(x){return a()+b.name};"#;

#[test]
fn test_parser_finds_declarations() {
    let result = decompile(SAMPLE_BUNDLE, &DecompileConfig::default()).unwrap();

    // Should find at least 3 declarations across all modules.
    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    assert!(
        total_decls >= 3,
        "expected at least 3 declarations, found {}",
        total_decls
    );
}

#[test]
fn test_reference_graph_edges() {
    let decls = ruvector_decompiler::parser::parse_bundle(SAMPLE_BUNDLE).unwrap();
    let graph = ruvector_decompiler::graph::build_reference_graph(decls);

    // c references a and b, so at least 2 edges.
    assert!(
        graph.edge_count() >= 2,
        "expected at least 2 edges, found {}",
        graph.edge_count()
    );
}

#[test]
fn test_mincut_partitions() {
    let config = DecompileConfig {
        target_modules: Some(2),
        ..DecompileConfig::default()
    };
    let result = decompile(SAMPLE_BUNDLE, &config).unwrap();

    // Should produce at least 1 module (partitioning may merge small groups).
    assert!(!result.modules.is_empty(), "expected at least 1 module");

    // Total declarations should equal what we parsed.
    let total: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    assert!(
        total >= 3,
        "expected at least 3 total declarations, got {}",
        total
    );
}

#[test]
fn test_name_inference_confidence() {
    let result = decompile(SAMPLE_BUNDLE, &DecompileConfig::default()).unwrap();

    // At least some names should be inferred.
    assert!(
        !result.inferred_names.is_empty(),
        "expected at least one inferred name"
    );

    // All confidence scores should be in [0, 1].
    for name in &result.inferred_names {
        assert!(
            (0.0..=1.0).contains(&name.confidence),
            "confidence out of range: {}",
            name.confidence
        );
    }
}

#[test]
fn test_source_map_v3_format() {
    let result = decompile(SAMPLE_BUNDLE, &DecompileConfig::default()).unwrap();

    assert!(
        !result.source_maps.is_empty(),
        "expected at least one source map"
    );

    for sm_json in &result.source_maps {
        let parsed: serde_json::Value = serde_json::from_str(sm_json).unwrap();
        assert_eq!(parsed["version"], 3, "source map version should be 3");
        assert!(
            parsed["mappings"].is_string(),
            "mappings should be a string"
        );
        assert!(parsed["sources"].is_array(), "sources should be an array");
    }
}

#[test]
fn test_witness_chain_valid() {
    let result = decompile(SAMPLE_BUNDLE, &DecompileConfig::default()).unwrap();

    // Witness chain should have a non-empty source hash.
    assert!(
        !result.witness.source_hash.is_empty(),
        "witness source hash should not be empty"
    );

    // Chain root should be non-empty.
    assert!(
        !result.witness.chain_root.is_empty(),
        "chain root should not be empty"
    );

    // Module witnesses should match module count.
    assert_eq!(
        result.witness.module_witnesses.len(),
        result.modules.len(),
        "witness count should match module count"
    );
}

#[test]
fn test_witness_chain_deterministic() {
    let config = DecompileConfig::default();
    let r1 = decompile(SAMPLE_BUNDLE, &config).unwrap();
    let r2 = decompile(SAMPLE_BUNDLE, &config).unwrap();

    assert_eq!(
        r1.witness.source_hash, r2.witness.source_hash,
        "source hash should be deterministic"
    );
    assert_eq!(
        r1.witness.chain_root, r2.witness.chain_root,
        "chain root should be deterministic"
    );
}

#[test]
fn test_full_pipeline_end_to_end() {
    let config = DecompileConfig {
        target_modules: Some(2),
        min_confidence: 0.3,
        generate_source_maps: true,
        generate_witness: true,
        output_filename: "test_output.js".to_string(),
        model_path: None,
        hierarchical_output: Some(true),
        max_depth: Some(3),
        min_folder_size: Some(3),
    };

    let result = decompile(SAMPLE_BUNDLE, &config).unwrap();

    // Print summary for manual inspection.
    println!("--- Decompilation Summary ---");
    println!("Modules: {}", result.modules.len());
    for module in &result.modules {
        println!(
            "  [{}] {} ({} declarations, bytes {}-{})",
            module.index,
            module.name,
            module.declarations.len(),
            module.byte_range.0,
            module.byte_range.1,
        );
    }
    println!("Inferred names: {}", result.inferred_names.len());
    for name in &result.inferred_names {
        println!(
            "  {} -> {} (confidence: {:.0}%, evidence: {:?})",
            name.original,
            name.inferred,
            name.confidence * 100.0,
            name.evidence,
        );
    }
    println!("Source maps: {}", result.source_maps.len());
    println!("Witness chain root: {}", result.witness.chain_root);
}
