//! Core domain types for the decompiler.

use serde::{Deserialize, Serialize};

/// The kind of a top-level declaration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeclKind {
    /// `var x = ...`
    Var,
    /// `let x = ...`
    Let,
    /// `const x = ...`
    Const,
    /// `function x(...) { ... }`
    Function,
    /// `class x { ... }`
    Class,
}

impl std::fmt::Display for DeclKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Var => write!(f, "var"),
            Self::Let => write!(f, "let"),
            Self::Const => write!(f, "const"),
            Self::Function => write!(f, "function"),
            Self::Class => write!(f, "class"),
        }
    }
}

/// A top-level declaration extracted from the minified bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Declaration {
    /// The minified name (e.g., `a`, `b`, `_0x1a2b`).
    pub name: String,
    /// Declaration kind.
    pub kind: DeclKind,
    /// Byte range `(start, end)` in the original bundle.
    pub byte_range: (usize, usize),
    /// String literals found within this declaration's body.
    pub string_literals: Vec<String>,
    /// Property names accessed (e.g., `.name`, `.permission`).
    pub property_accesses: Vec<String>,
    /// Names of other declarations referenced by this one.
    pub references: Vec<String>,
}

/// A reconstructed module extracted from the bundle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Module {
    /// Inferred module name.
    pub name: String,
    /// Index of this module in the decompilation output.
    pub index: usize,
    /// Declarations belonging to this module.
    pub declarations: Vec<Declaration>,
    /// Beautified source code for this module.
    pub source: String,
    /// Byte range in the original bundle that this module covers.
    pub byte_range: (usize, usize),
}

/// An inferred name mapping from minified to reconstructed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferredName {
    /// The original minified name.
    pub original: String,
    /// The inferred human-readable name.
    pub inferred: String,
    /// Confidence score from 0.0 (guess) to 1.0 (certain).
    pub confidence: f64,
    /// Evidence strings explaining why this name was inferred.
    pub evidence: Vec<String>,
}

/// Confidence thresholds for name inference.
#[derive(Debug, Clone, Copy)]
pub enum Confidence {
    /// Direct string match, confidence > 0.9.
    High,
    /// Contextual inference, confidence 0.6 -- 0.9.
    Medium,
    /// Structural guess only, confidence < 0.6.
    Low,
}

impl Confidence {
    /// Returns the minimum confidence value for this level.
    pub fn min_value(self) -> f64 {
        match self {
            Self::High => 0.9,
            Self::Medium => 0.6,
            Self::Low => 0.0,
        }
    }
}

/// The full result of a decompilation run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompileResult {
    /// Reconstructed modules.
    pub modules: Vec<Module>,
    /// All inferred name mappings.
    pub inferred_names: Vec<InferredName>,
    /// Source maps (one JSON string per module).
    pub source_maps: Vec<String>,
    /// Witness chain for cryptographic provenance.
    pub witness: WitnessChainData,
}

/// Serializable witness chain data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessChainData {
    /// Hex-encoded SHA3-256 hash of the original bundle.
    pub source_hash: String,
    /// Per-module witness entries.
    pub module_witnesses: Vec<ModuleWitnessData>,
    /// Hex-encoded Merkle root of all module hashes.
    pub chain_root: String,
}

/// Per-module witness data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleWitnessData {
    /// Module name.
    pub module_name: String,
    /// Byte range in the original bundle.
    pub byte_range: (usize, usize),
    /// Hex-encoded SHA3-256 hash of the module content.
    pub content_hash: String,
    /// Hex-encoded SHA3-256 hash of the inferred names.
    pub inferred_names_hash: String,
}

/// Configuration for the decompiler pipeline.
#[derive(Debug, Clone)]
pub struct DecompileConfig {
    /// Target number of modules to reconstruct. If `None`, auto-detect.
    pub target_modules: Option<usize>,
    /// Minimum confidence threshold for including inferred names.
    pub min_confidence: f64,
    /// Whether to generate source maps.
    pub generate_source_maps: bool,
    /// Whether to generate witness chains.
    pub generate_witness: bool,
    /// The filename to use in source map output.
    pub output_filename: String,
}

impl Default for DecompileConfig {
    fn default() -> Self {
        Self {
            target_modules: None,
            min_confidence: 0.0,
            generate_source_maps: true,
            generate_witness: true,
            output_filename: "bundle.js".to_string(),
        }
    }
}
