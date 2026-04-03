//! Run the decompiler on a real JS bundle and report timing and metrics.
//!
//! Usage: cargo run --release --example run_on_cli -- <path-to-js-file>

use std::time::Instant;

use ruvector_decompiler::{decompile, DecompileConfig, ModuleTree};

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "cli.js".to_string());

    eprintln!("Reading file: {}", path);
    let source = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Failed to read file: {}", e);
            std::process::exit(1);
        }
    };
    eprintln!(
        "File size: {} bytes ({:.2} MB)",
        source.len(),
        source.len() as f64 / 1_048_576.0
    );

    // Phase 1: Parse
    let t0 = Instant::now();
    let decls = ruvector_decompiler::parser::parse_bundle(&source).unwrap();
    let t_parse = t0.elapsed();
    eprintln!(
        "Phase 1 (Parse): {:?} -- {} declarations found",
        t_parse,
        decls.len()
    );

    // Phase 2: Graph
    let t1 = Instant::now();
    let graph = ruvector_decompiler::graph::build_reference_graph(decls);
    let t_graph = t1.elapsed();
    eprintln!(
        "Phase 2 (Graph): {:?} -- {} nodes, {} edges",
        t_graph,
        graph.node_count(),
        graph.edge_count()
    );

    // Phase 3: Partition -- uses Louvain for large graphs automatically.
    let large_graph = graph.node_count() > 5000;
    if large_graph {
        eprintln!(
            "Phase 3 (Partition): Using Louvain community detection ({} nodes, {} edges)",
            graph.node_count(),
            graph.edge_count()
        );
    }
    let t2 = Instant::now();
    let modules =
        ruvector_decompiler::partitioner::partition_modules(&graph, None).unwrap();
    let t_partition = t2.elapsed();
    eprintln!(
        "Phase 3 (Partition): {:?} -- {} modules detected{}",
        t_partition,
        modules.len(),
        if large_graph { " (Louvain)" } else { " (MinCut)" }
    );

    // Phase 4: Infer names
    let t3 = Instant::now();
    let inferred = ruvector_decompiler::inferrer::infer_names(&modules);
    let t_infer = t3.elapsed();

    let high = inferred.iter().filter(|n| n.confidence > 0.9).count();
    let medium = inferred
        .iter()
        .filter(|n| n.confidence >= 0.6 && n.confidence <= 0.9)
        .count();
    let low = inferred.iter().filter(|n| n.confidence < 0.6).count();
    eprintln!(
        "Phase 4 (Infer): {:?} -- {} names (HIGH={}, MEDIUM={}, LOW={})",
        t_infer,
        inferred.len(),
        high,
        medium,
        low
    );

    // Full pipeline
    let t_full_start = Instant::now();
    let config = DecompileConfig {
        target_modules: None, // Auto-detect, Louvain handles large graphs.
        min_confidence: 0.3,
        generate_source_maps: false, // Skip for speed on large files.
        generate_witness: true,
        output_filename: path.clone(),
        model_path: None,
        hierarchical_output: Some(true),
        max_depth: Some(3),
        min_folder_size: Some(3),
    };
    let result = decompile(&source, &config).unwrap();
    let t_full = t_full_start.elapsed();

    eprintln!("\n=== Summary ===");
    eprintln!(
        "File: {} ({:.2} MB)",
        path,
        source.len() as f64 / 1_048_576.0
    );
    eprintln!("Total pipeline time: {:?}", t_full);
    eprintln!("  Parse:     {:?}", t_parse);
    eprintln!("  Graph:     {:?}", t_graph);
    eprintln!("  Partition: {:?}", t_partition);
    eprintln!("  Infer:     {:?}", t_infer);
    eprintln!(
        "Declarations: {}",
        result
            .modules
            .iter()
            .map(|m| m.declarations.len())
            .sum::<usize>()
    );
    eprintln!("Modules: {}", result.modules.len());
    eprintln!(
        "Inferred names: {} (filtered by confidence >= 0.3)",
        result.inferred_names.len()
    );
    eprintln!(
        "  HIGH confidence (>0.9): {}",
        result
            .inferred_names
            .iter()
            .filter(|n| n.confidence > 0.9)
            .count()
    );
    eprintln!(
        "  MEDIUM confidence (0.6-0.9): {}",
        result
            .inferred_names
            .iter()
            .filter(|n| n.confidence >= 0.6 && n.confidence <= 0.9)
            .count()
    );
    eprintln!(
        "  LOW confidence (<0.6): {}",
        result
            .inferred_names
            .iter()
            .filter(|n| n.confidence < 0.6)
            .count()
    );
    if !result.witness.chain_root.is_empty() {
        eprintln!(
            "Witness chain root: {}",
            &result.witness.chain_root[..16.min(result.witness.chain_root.len())]
        );
    }

    // Print hierarchical module tree.
    if let Some(ref tree) = result.module_tree {
        eprintln!("\n=== Module Tree (graph-derived) ===");
        print_tree(tree, "");
    }

    // Print top-10 highest confidence names.
    let mut sorted_names = result.inferred_names.clone();
    sorted_names.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
    eprintln!("\nTop 10 inferred names:");
    for name in sorted_names.iter().take(10) {
        eprintln!(
            "  {} -> {} ({:.0}% confidence)",
            name.original,
            name.inferred,
            name.confidence * 100.0
        );
    }

    // Rough memory estimate.
    let decl_mem = result
        .modules
        .iter()
        .flat_map(|m| m.declarations.iter())
        .map(|d| {
            d.name.len()
                + d.string_literals.iter().map(|s| s.len()).sum::<usize>()
                + d.property_accesses.iter().map(|s| s.len()).sum::<usize>()
                + d.references.iter().map(|s| s.len()).sum::<usize>()
                + 64
        })
        .sum::<usize>();
    let module_mem = result
        .modules
        .iter()
        .map(|m| m.source.len() + m.name.len() + 64)
        .sum::<usize>();
    eprintln!("\nEstimated memory usage:");
    eprintln!("  Declarations: {:.2} MB", decl_mem as f64 / 1_048_576.0);
    eprintln!("  Module sources: {:.2} MB", module_mem as f64 / 1_048_576.0);
    eprintln!(
        "  Total estimate: {:.2} MB",
        (decl_mem + module_mem) as f64 / 1_048_576.0
    );

    // Write tree output if --output-dir is provided.
    if let Some(out_dir) = std::env::args().nth(2) {
        if let Some(ref tree) = result.module_tree {
            let base = std::path::Path::new(&out_dir);
            eprintln!("\nWriting tree output to: {}", out_dir);
            write_tree_output(tree, base, &source);
            eprintln!("Done.");
        }
    }
}

/// Print the module tree to stderr with indentation.
fn print_tree(tree: &ModuleTree, indent: &str) {
    let module_count = tree.modules.len();
    let child_count = tree.children.len();
    if module_count > 0 {
        eprintln!(
            "{}{}/  ({} modules)",
            indent, tree.name, module_count
        );
        for m in &tree.modules {
            eprintln!(
                "{}  {} ({} decls)",
                indent, m.name, m.declarations.len()
            );
        }
    } else {
        eprintln!(
            "{}{}/  ({} subfolders)",
            indent, tree.name, child_count
        );
    }
    for child in &tree.children {
        print_tree(child, &format!("{}  ", indent));
    }
}

/// Write tree structure to disk as a folder hierarchy.
fn write_tree_output(tree: &ModuleTree, base_dir: &std::path::Path, source: &str) {
    let dir = base_dir.join(&tree.path);
    std::fs::create_dir_all(&dir).ok();

    // Write leaf modules in this folder.
    for module in &tree.modules {
        let filename = format!("{}.js", module.name);
        let content = if module.source.is_empty() {
            // Fall back to extracting from source by byte range.
            let (start, end) = module.byte_range;
            let end = end.min(source.len());
            let start = start.min(end);
            &source[start..end]
        } else {
            &module.source
        };
        std::fs::write(dir.join(filename), content).ok();
    }

    // Recurse into children.
    for child in &tree.children {
        write_tree_output(child, base_dir, source);
    }
}
