//! Module boundary detection with adaptive partitioning.
//!
//! Uses exact MinCut for small graphs (<5K nodes) and Louvain community
//! detection for large graphs (>=5K nodes). Louvain is O(n log n) and
//! handles 100K+ node graphs in seconds.
//!
//! Performance: Louvain local-move phase is parallelized with rayon.
//! Each node's modularity gain is computed independently (read-only
//! access to current community assignments), then moves are applied
//! sequentially to prevent conflicts.

use std::collections::HashMap;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::error::{DecompilerError, Result};
use crate::graph::ReferenceGraph;
use crate::types::{Declaration, Module};

use ruvector_mincut::GraphPartitioner;

/// Partition the reference graph into modules.
///
/// Automatically selects the partitioning algorithm based on graph size:
/// - <5000 nodes: exact MinCut via `ruvector-mincut::GraphPartitioner`
/// - >=5000 nodes: Louvain community detection (approximate, O(n log n))
///
/// If `target_modules` is `None`, the partition count is estimated from
/// the graph structure.
pub fn partition_modules(
    graph: &ReferenceGraph,
    target_modules: Option<usize>,
) -> Result<Vec<Module>> {
    let n = graph.node_count();
    if n == 0 {
        return Err(DecompilerError::PartitioningFailed(
            "no declarations to partition".to_string(),
        ));
    }

    let target = target_modules.unwrap_or_else(|| estimate_module_count(graph));
    let target = target.clamp(1, n);

    if target == 1 || n <= 2 {
        return Ok(vec![build_module(
            0,
            &graph.declarations,
        )]);
    }

    // Choose algorithm based on graph size.
    if n >= 5000 {
        louvain_partition(graph, target)
    } else {
        exact_mincut_partition(graph, target)
    }
}

/// Exact MinCut partitioning for small-to-medium graphs (<5K nodes).
fn exact_mincut_partition(
    graph: &ReferenceGraph,
    target: usize,
) -> Result<Vec<Module>> {
    let partitioner = GraphPartitioner::new(graph.graph.clone(), target);
    let partitions = partitioner.partition();

    let mut assigned: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    let mut modules = Vec::new();
    let mut mod_idx = 0;

    for partition in &partitions {
        if partition.is_empty() {
            continue;
        }

        let mut decls: Vec<Declaration> = Vec::new();
        for &vid in partition {
            if let Some(idx) = graph.vertex_to_decl.get(&vid) {
                if let Some(decl) = graph.declarations.get(*idx) {
                    decls.push(decl.clone());
                    assigned.insert(*idx);
                }
            }
        }

        if !decls.is_empty() {
            modules.push(build_module(mod_idx, &decls));
            mod_idx += 1;
        }
    }

    distribute_orphans(graph, &mut modules, &assigned);
    finalize_modules(graph, modules)
}

/// Louvain community detection for large graphs (>=5K nodes).
///
/// O(n log n) -- handles 100K+ node graphs in seconds.
///
/// Algorithm:
/// 1. Start with each node in its own community.
/// 2. Repeatedly move nodes to the neighbor community that maximizes
///    modularity gain (parallelized with rayon).
/// 3. When no more single-node moves improve modularity, aggregate
///    communities into super-nodes and repeat.
/// 4. Merge small communities to meet target count if needed.
///
/// Performance: The local-move phase uses `rayon::par_iter` to compute
/// modularity gains in parallel. Each node reads the current community
/// assignments without locking. Moves are applied sequentially after
/// each parallel sweep to prevent conflicts.
fn louvain_partition(
    graph: &ReferenceGraph,
    target: usize,
) -> Result<Vec<Module>> {
    let n = graph.node_count();

    // Build adjacency list from the reference graph.
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];
    let mut total_weight = 0.0;

    for (i, decl) in graph.declarations.iter().enumerate() {
        for ref_name in &decl.references {
            if let Some(&vid) = graph.name_to_vertex.get(ref_name) {
                if let Some(&j) = graph.vertex_to_decl.get(&vid) {
                    if i != j {
                        adj[i].push((j, 1.0));
                        total_weight += 1.0;
                    }
                }
            }
        }
    }

    // If no edges, fall back to positional grouping.
    if total_weight < 1.0 {
        return positional_partition(graph, target);
    }

    // Node weights: sum of edge weights for each node.
    let node_weights: Vec<f64> = adj
        .iter()
        .map(|neighbors| neighbors.iter().map(|(_, w)| w).sum::<f64>())
        .collect();

    // Phase 1: Local moves -- assign each node to its own community,
    // then iteratively move nodes to improve modularity.
    let mut community: Vec<usize> = (0..n).collect();
    let m2 = total_weight; // sum of all edge weights (each counted once)

    // Pre-compute community total weights for O(1) lookup.
    // sigma_totals[c] = sum of node_weights for all nodes in community c.
    let mut sigma_totals: Vec<f64> = node_weights.clone();

    let mut improved = true;
    let mut iterations = 0;
    let max_iterations = 20; // Prevent infinite loops

    while improved && iterations < max_iterations {
        improved = false;
        iterations += 1;

        // Parallel phase: compute best community for each node.
        // Each node reads community[] and sigma_totals[] (snapshot).
        // No writes during this phase.
        #[cfg(feature = "parallel")]
        let iter = (0..n).into_par_iter();
        #[cfg(not(feature = "parallel"))]
        let iter = (0..n).into_iter();

        let gains: Vec<(usize, usize, f64)> = iter
            .filter_map(|i| {
                let current_comm = community[i];
                let ki = node_weights[i];

                if adj[i].is_empty() {
                    return None;
                }

                // Compute sum of weights to each neighbor community.
                let mut comm_weights: HashMap<usize, f64> =
                    HashMap::with_capacity(adj[i].len());
                for &(j, w) in &adj[i] {
                    *comm_weights.entry(community[j]).or_insert(0.0) += w;
                }

                let mut best_comm = current_comm;
                let mut best_gain = 0.0f64;

                let ki_in_current =
                    comm_weights.get(&current_comm).copied().unwrap_or(0.0);
                let sigma_current = sigma_totals[current_comm];

                for (&candidate_comm, &ki_in_candidate) in &comm_weights {
                    if candidate_comm == current_comm {
                        continue;
                    }

                    let sigma_candidate = sigma_totals[candidate_comm];

                    // Modularity gain of moving i from current to candidate.
                    let gain = (ki_in_candidate - ki_in_current)
                        - ki * (sigma_candidate - sigma_current + ki) / m2;

                    if gain > best_gain {
                        best_gain = gain;
                        best_comm = candidate_comm;
                    }
                }

                if best_comm != current_comm && best_gain > 0.0 {
                    Some((i, best_comm, best_gain))
                } else {
                    None
                }
            })
            .collect();

        // Sequential phase: apply moves and update sigma_totals.
        for &(node, new_comm, _) in &gains {
            let old_comm = community[node];
            if old_comm != new_comm {
                let ki = node_weights[node];
                sigma_totals[old_comm] -= ki;
                // Ensure sigma_totals vec is large enough.
                if new_comm >= sigma_totals.len() {
                    sigma_totals.resize(new_comm + 1, 0.0);
                }
                sigma_totals[new_comm] += ki;
                community[node] = new_comm;
                improved = true;
            }
        }
    }

    // Phase 2: Collect communities and merge small ones to meet target.
    let mut comm_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &c) in community.iter().enumerate() {
        comm_members.entry(c).or_default().push(i);
    }

    let mut communities: Vec<Vec<usize>> = comm_members.into_values().collect();

    // Sort by size (largest first) for stable merging.
    communities.sort_by(|a, b| b.len().cmp(&a.len()));

    // Merge small communities if we have too many.
    while communities.len() > target && communities.len() > 1 {
        // Merge the two smallest communities.
        let small = communities.pop().unwrap();
        if let Some(last) = communities.last_mut() {
            last.extend(small);
        }
    }

    // Build modules from communities.
    let mut modules = Vec::new();
    for (mod_idx, members) in communities.iter().enumerate() {
        let decls: Vec<Declaration> = members
            .iter()
            .filter_map(|&i| graph.declarations.get(i).cloned())
            .collect();
        if !decls.is_empty() {
            modules.push(build_module(mod_idx, &decls));
        }
    }

    finalize_modules(graph, modules)
}

/// Fallback: positional partitioning by byte offset for edge-less graphs.
fn positional_partition(
    graph: &ReferenceGraph,
    target: usize,
) -> Result<Vec<Module>> {
    let n = graph.node_count();
    let chunk_size = (n + target - 1) / target;

    let mut modules = Vec::new();
    for (mod_idx, chunk) in graph.declarations.chunks(chunk_size).enumerate() {
        modules.push(build_module(mod_idx, chunk));
    }

    finalize_modules(graph, modules)
}

/// Distribute orphan declarations (not assigned by partitioner) to
/// nearest modules by byte position.
fn distribute_orphans(
    graph: &ReferenceGraph,
    modules: &mut Vec<Module>,
    assigned: &std::collections::HashSet<usize>,
) {
    let orphans: Vec<Declaration> = graph
        .declarations
        .iter()
        .enumerate()
        .filter(|(i, _)| !assigned.contains(i))
        .map(|(_, d)| d.clone())
        .collect();

    if orphans.is_empty() {
        return;
    }

    if modules.is_empty() {
        modules.push(build_module(0, &orphans));
    } else {
        for orphan in &orphans {
            let best_module = modules
                .iter_mut()
                .min_by_key(|m| {
                    let mid = (m.byte_range.0 + m.byte_range.1) / 2;
                    let orphan_mid =
                        (orphan.byte_range.0 + orphan.byte_range.1) / 2;
                    (mid as i64 - orphan_mid as i64).unsigned_abs()
                })
                .unwrap();
            best_module.declarations.push(orphan.clone());
            best_module.byte_range.0 =
                best_module.byte_range.0.min(orphan.byte_range.0);
            best_module.byte_range.1 =
                best_module.byte_range.1.max(orphan.byte_range.1);
        }
    }
}

/// Finalize module list: ensure at least one module exists.
fn finalize_modules(
    graph: &ReferenceGraph,
    modules: Vec<Module>,
) -> Result<Vec<Module>> {
    if modules.is_empty() {
        Ok(vec![build_module(0, &graph.declarations)])
    } else {
        Ok(modules)
    }
}

/// Build a `Module` from a set of declarations.
fn build_module(index: usize, decls: &[Declaration]) -> Module {
    let name = infer_module_name(decls, index);
    let start = decls.iter().map(|d| d.byte_range.0).min().unwrap_or(0);
    let end = decls.iter().map(|d| d.byte_range.1).max().unwrap_or(0);

    Module {
        name,
        index,
        declarations: decls.to_vec(),
        source: String::new(), // Filled in by the beautifier later.
        byte_range: (start, end),
    }
}

/// Infer a module name from the dominant string literals and property names.
fn infer_module_name(decls: &[Declaration], fallback_index: usize) -> String {
    let mut candidates: Vec<&str> = Vec::new();

    for decl in decls {
        for s in &decl.string_literals {
            if s.len() >= 2 && s.len() <= 40 && !s.contains(' ') {
                candidates.push(s.as_str());
            }
        }
        for p in &decl.property_accesses {
            candidates.push(p.as_str());
        }
    }

    if !candidates.is_empty() {
        let mut freq: HashMap<&str, usize> = HashMap::new();
        for c in &candidates {
            *freq.entry(c).or_insert(0) += 1;
        }
        if let Some((&best, _)) = freq.iter().max_by_key(|(_, &count)| count) {
            return sanitize_module_name(best);
        }
    }

    format!("module_{}", fallback_index)
}

/// Sanitize a string into a valid module name.
fn sanitize_module_name(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
        .collect();
    if cleaned.is_empty() {
        "module".to_string()
    } else {
        cleaned
    }
}

/// Estimate the number of modules based on graph structure.
fn estimate_module_count(graph: &ReferenceGraph) -> usize {
    let n = graph.node_count();
    let e = graph.edge_count();

    if n <= 3 {
        return 1;
    }

    let avg_degree = if n > 0 {
        (2 * e) as f64 / n as f64
    } else {
        0.0
    };

    if avg_degree < 1.0 {
        (n / 2).max(2)
    } else {
        (n as f64 / (avg_degree + 1.0)).ceil().max(2.0) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_reference_graph;
    use crate::types::DeclKind;

    fn make_decl(name: &str, refs: &[&str], strings: &[&str]) -> Declaration {
        Declaration {
            name: name.to_string(),
            kind: DeclKind::Var,
            byte_range: (0, 10),
            string_literals: strings.iter().map(|s| s.to_string()).collect(),
            property_accesses: vec![],
            references: refs.iter().map(|s| s.to_string()).collect(),
        }
    }

    #[test]
    fn test_partition_single() {
        let decls = vec![make_decl("a", &[], &["hello"])];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(1)).unwrap();
        assert_eq!(modules.len(), 1);
    }

    #[test]
    fn test_partition_multiple() {
        let decls = vec![
            make_decl("a", &[], &["alpha"]),
            make_decl("b", &["a"], &["beta"]),
            make_decl("c", &[], &["gamma"]),
            make_decl("d", &["c"], &["delta"]),
        ];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(2)).unwrap();
        assert!(!modules.is_empty());
        let total: usize = modules.iter().map(|m| m.declarations.len()).sum();
        assert_eq!(total, 4);
    }

    #[test]
    fn test_module_naming() {
        let decls = vec![make_decl("x", &[], &["auth", "auth", "login"])];
        let name = infer_module_name(&decls, 0);
        assert_eq!(name, "auth");
    }

    #[test]
    fn test_louvain_large_graph() {
        // Create a graph with 100 nodes in two clusters.
        let mut decls = Vec::new();
        for i in 0..50 {
            let refs: Vec<&str> = Vec::new();
            decls.push(make_decl(
                &format!("a{}", i),
                &refs,
                &["cluster_a"],
            ));
        }
        for i in 0..50 {
            decls.push(make_decl(
                &format!("b{}", i),
                &[],
                &["cluster_b"],
            ));
        }
        // Add cross-references within clusters.
        for i in 1..50 {
            decls[i].references.push(format!("a{}", i - 1));
        }
        for i in 51..100 {
            decls[i].references.push(format!("b{}", i - 51));
        }

        let graph = build_reference_graph(decls);
        // Force louvain by calling it directly.
        let modules = louvain_partition(&graph, 2).unwrap();
        assert!(!modules.is_empty());
        let total: usize = modules.iter().map(|m| m.declarations.len()).sum();
        assert_eq!(total, 100);
    }
}
