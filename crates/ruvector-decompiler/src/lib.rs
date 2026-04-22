//! # RuVector Decompiler
//!
//! SOTA JavaScript bundle decompiler using MinCut graph partitioning,
//! self-learning name inference, and RVF witness chains.
//!
//! ## Pipeline
//!
//! 1. **Parse**: Regex-based extraction of declarations from minified JS.
//! 2. **Graph**: Build a weighted reference graph of cross-declaration links.
//! 3. **Partition**: Use MinCut to detect natural module boundaries.
//! 4. **Infer**: Score and assign human-readable names to minified identifiers.
//! 5. **Witness**: Generate a SHA3-256 Merkle chain for cryptographic provenance.
//!
//! ## Quick Start
//!
//! ```rust,no_run
//! use ruvector_decompiler::{decompile, DecompileConfig};
//!
//! let minified = r#"var a=function(){return"hello"};var b=42;"#;
//! let config = DecompileConfig::default();
//! let result = decompile(minified, &config).unwrap();
//!
//! for module in &result.modules {
//!     println!("Module: {}", module.name);
//!     println!("{}", module.source);
//! }
//! ```

pub mod beautifier;
pub mod error;
pub mod graph;
pub mod inferrer;
#[cfg(feature = "neural")]
pub mod neural;
pub mod parser;
pub mod partitioner;
pub mod sourcemap;
pub mod training;
pub mod transformer;
pub mod tree;
pub mod types;
pub mod witness;

// LLM model weight decompilation (feature-gated).
#[cfg(feature = "model")]
pub mod model_decompiler;
#[cfg(feature = "model")]
pub mod model_gguf;
#[cfg(feature = "model")]
pub mod model_safetensors;
#[cfg(feature = "model")]
pub mod model_types;

pub use error::{DecompilerError, Result};
pub use types::{
    Declaration, DecompileConfig, DecompileResult, InferredName, Module, ModuleTree,
    WitnessChainData,
};

#[cfg(feature = "model")]
pub use model_decompiler::{decompile_gguf, decompile_model, decompile_safetensors};
#[cfg(feature = "model")]
pub use model_types::{
    LayerInfo, LayerType, ModelArchitecture, ModelDecompileResult, ModelFormat, QuantizationInfo,
    TokenizerInfo,
};

/// Decompile a minified JavaScript bundle.
///
/// Runs the full five-phase pipeline: parse, graph, partition, infer, witness.
///
/// # Arguments
///
/// * `source` - The minified JavaScript bundle source code.
/// * `config` - Configuration for the decompilation pipeline.
///
/// # Returns
///
/// A `DecompileResult` containing reconstructed modules, inferred names,
/// source maps, and a witness chain.
pub fn decompile(source: &str, config: &DecompileConfig) -> Result<DecompileResult> {
    // Phase 1: Parse.
    let declarations = parser::parse_bundle(source)?;

    // Phase 2: Build reference graph.
    let ref_graph = graph::build_reference_graph(declarations);

    // Phase 3: Partition into modules.
    let mut modules = partitioner::partition_modules(&ref_graph, config.target_modules)?;

    // Phase 4: Infer names.
    let inferred_names = inferrer::infer_names(&modules);

    // Filter names by confidence.
    let filtered_names: Vec<InferredName> = inferred_names
        .iter()
        .filter(|n| n.confidence >= config.min_confidence)
        .cloned()
        .collect();

    // Beautify module source code.
    beautifier::beautify_all(&mut modules, source, &filtered_names, config.min_confidence);

    // Phase 4b: Build hierarchical module tree from graph structure.
    let module_tree = if config.hierarchical_output.unwrap_or(true) {
        Some(tree::build_module_tree(&ref_graph, &modules, config))
    } else {
        None
    };

    // Phase 4c: Generate source maps.
    let source_maps = if config.generate_source_maps {
        modules
            .iter()
            .map(|m| sourcemap::generate_source_map(m, &filtered_names, &config.output_filename))
            .collect::<Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    // Phase 5: Witness chain.
    let witness_data = if config.generate_witness {
        let chain = witness::build_witness_chain(source, &modules, &filtered_names);
        if !witness::verify_witness_chain(&chain) {
            return Err(DecompilerError::WitnessError(
                "witness chain failed self-verification".to_string(),
            ));
        }
        witness::witness_to_data(&chain)
    } else {
        WitnessChainData {
            source_hash: String::new(),
            module_witnesses: Vec::new(),
            chain_root: String::new(),
        }
    };

    Ok(DecompileResult {
        modules,
        module_tree,
        inferred_names: filtered_names,
        source_maps,
        witness: witness_data,
    })
}

/// Convenience function: decompile with default configuration.
pub fn decompile_default(source: &str) -> Result<DecompileResult> {
    decompile(source, &DecompileConfig::default())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decompile_simple() {
        let src = r#"var a=function(){return"hello"};var b=42;"#;
        let result = decompile_default(src).unwrap();
        assert!(!result.modules.is_empty());
        assert!(!result.witness.source_hash.is_empty());
    }

    #[test]
    fn test_decompile_with_refs() {
        let src = r#"var a=function(){return"hello"};var b=function(){return a()};var c=42;"#;
        let config = DecompileConfig {
            target_modules: Some(2),
            ..DecompileConfig::default()
        };
        let result = decompile(src, &config).unwrap();
        assert!(!result.modules.is_empty());
        // Verify witness chain.
        assert!(!result.witness.chain_root.is_empty());
    }
}
