//! Hierarchical module tree construction from graph partitions.
//!
//! The folder structure IS the graph structure. Tightly-coupled code
//! (high edge weight) stays in the same folder. Loosely-coupled code
//! (low edge weight / MinCut boundary) goes in different folders.
//! Folder names are inferred from discriminative string context.

use std::collections::HashMap;

use crate::graph::ReferenceGraph;
use crate::inferrer::infer_folder_name;
use crate::types::{DecompileConfig, Module, ModuleTree};

/// Build a hierarchical module tree from Louvain community hierarchy.
///
/// Algorithm:
/// 1. Start with flat module list (already partitioned by Louvain).
/// 2. Build inter-module adjacency from shared references.
/// 3. Cluster modules using agglomerative grouping by edge density.
/// 4. Recursively split large groups if depth < max_depth.
/// 5. Name each folder from discriminative string context.
pub fn build_module_tree(
    graph: &ReferenceGraph,
    modules: &[Module],
    config: &DecompileConfig,
) -> ModuleTree {
    let max_depth = config.max_depth.unwrap_or(3);
    let min_folder_size = config.min_folder_size.unwrap_or(3);

    let inter_module = build_inter_module_edges(graph, modules);

    let module_indices: Vec<usize> = (0..modules.len()).collect();
    build_tree_recursive(
        modules,
        &module_indices,
        &inter_module,
        0,
        max_depth,
        min_folder_size,
        "",
    )
}

/// Build inter-module edge weights.
///
/// Returns a map: (module_i, module_j) -> weight, where weight is the
/// number of cross-references between declarations in the two modules.
fn build_inter_module_edges(
    _graph: &ReferenceGraph,
    modules: &[Module],
) -> HashMap<(usize, usize), f64> {
    let mut decl_to_module: HashMap<&str, usize> = HashMap::new();
    for (mi, module) in modules.iter().enumerate() {
        for decl in &module.declarations {
            decl_to_module.insert(&decl.name, mi);
        }
    }

    let mut edges: HashMap<(usize, usize), f64> = HashMap::new();
    for (mi, module) in modules.iter().enumerate() {
        for decl in &module.declarations {
            for ref_name in &decl.references {
                if let Some(&mj) = decl_to_module.get(ref_name.as_str()) {
                    if mi != mj {
                        let key = if mi < mj { (mi, mj) } else { (mj, mi) };
                        *edges.entry(key).or_insert(0.0) += 1.0;
                    }
                }
            }
        }
    }

    edges
}

/// Recursively build the module tree by clustering modules.
fn build_tree_recursive(
    all_modules: &[Module],
    indices: &[usize],
    inter_module: &HashMap<(usize, usize), f64>,
    depth: usize,
    max_depth: usize,
    min_folder_size: usize,
    parent_path: &str,
) -> ModuleTree {
    // Base case: max depth reached, or folder is small enough to be a leaf.
    // At depth 0: only leaf if <=min_folder_size modules
    // At depth 1+: leaf if <=20 modules (enough granularity)
    let leaf_threshold = if depth == 0 {
        min_folder_size
    } else {
        20.min(min_folder_size * 5)
    };
    if indices.len() <= leaf_threshold || depth >= max_depth {
        return make_leaf(all_modules, indices, depth, parent_path);
    }

    let groups = agglomerative_cluster(indices, inter_module, min_folder_size);

    // Single group means this is a leaf.
    if groups.len() <= 1 {
        return make_leaf(all_modules, indices, depth, parent_path);
    }

    // Name the internal node.
    let all_in_group: Vec<Module> = indices
        .iter()
        .filter_map(|&i| all_modules.get(i).cloned())
        .collect();
    let folder_name = if depth == 0 {
        "src".to_string()
    } else {
        infer_folder_name(&all_in_group, all_modules)
    };
    let folder_path = if parent_path.is_empty() {
        folder_name.clone()
    } else {
        format!("{}/{}", parent_path, folder_name)
    };

    let children: Vec<ModuleTree> = groups
        .iter()
        .map(|group| {
            build_tree_recursive(
                all_modules,
                group,
                inter_module,
                depth + 1,
                max_depth,
                min_folder_size,
                &folder_path,
            )
        })
        .collect();

    ModuleTree {
        name: folder_name,
        path: folder_path,
        modules: Vec::new(),
        children,
        depth,
    }
}

/// Create a leaf node from a set of module indices.
fn make_leaf(
    all_modules: &[Module],
    indices: &[usize],
    depth: usize,
    parent_path: &str,
) -> ModuleTree {
    let leaf_modules: Vec<Module> = indices
        .iter()
        .filter_map(|&i| all_modules.get(i).cloned())
        .collect();
    let name = infer_folder_name(&leaf_modules, all_modules);
    let path = if parent_path.is_empty() {
        name.clone()
    } else {
        format!("{}/{}", parent_path, name)
    };
    ModuleTree {
        name,
        path,
        modules: leaf_modules,
        children: Vec::new(),
        depth,
    }
}

/// Agglomerative clustering of module indices by inter-module edge density.
fn agglomerative_cluster(
    indices: &[usize],
    inter_module: &HashMap<(usize, usize), f64>,
    min_size: usize,
) -> Vec<Vec<usize>> {
    let n = indices.len();
    if n <= 1 {
        return vec![indices.to_vec()];
    }

    let mut clusters: Vec<Vec<usize>> = indices.iter().map(|&i| vec![i]).collect();

    // Target: 5-15 top-level folders for large codebases, 3-5 for small
    let target_clusters = if n > 500 {
        10
    } else if n > 100 {
        7
    } else if n > 20 {
        5
    } else {
        3
    };

    loop {
        if clusters.len() <= target_clusters {
            break;
        }

        let mut best_pair = (0, 1);
        let mut best_weight = -1.0f64;

        for i in 0..clusters.len() {
            for j in (i + 1)..clusters.len() {
                let w = cluster_edge_weight(&clusters[i], &clusters[j], inter_module);
                // Normalize by geometric mean of sizes (prevents giant clusters from absorbing everything)
                let size_factor = ((clusters[i].len() * clusters[j].len()) as f64)
                    .sqrt()
                    .max(1.0);
                let normalized = w / size_factor;
                if normalized > best_weight {
                    best_weight = normalized;
                    best_pair = (i, j);
                }
            }
        }

        // Stop if no connections remain
        if best_weight <= 0.0 {
            break;
        }

        // Don't merge if result would be too large (max 20% of total)
        let max_cluster = (n / 5).max(min_size * 3);
        if clusters[best_pair.0].len() + clusters[best_pair.1].len() > max_cluster {
            // Try next best pair that doesn't exceed max
            let mut found = false;
            for i in 0..clusters.len() {
                for j in (i + 1)..clusters.len() {
                    if clusters[i].len() + clusters[j].len() <= max_cluster {
                        let w = cluster_edge_weight(&clusters[i], &clusters[j], inter_module);
                        if w > 0.0 {
                            let merged = {
                                let mut m = clusters[i].clone();
                                m.extend_from_slice(&clusters[j]);
                                m
                            };
                            clusters.remove(j);
                            clusters.remove(i);
                            clusters.push(merged);
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
            }
            if !found {
                break;
            }
            continue;
        }

        let (i, j) = best_pair;
        let merged = {
            let mut m = clusters[i].clone();
            m.extend_from_slice(&clusters[j]);
            m
        };
        clusters.remove(j);
        clusters.remove(i);
        clusters.push(merged);
    }

    // Absorb tiny clusters into nearest large cluster.
    let mut final_clusters: Vec<Vec<usize>> = Vec::new();
    let mut tiny: Vec<Vec<usize>> = Vec::new();

    for c in clusters {
        if c.len() >= min_size {
            final_clusters.push(c);
        } else {
            tiny.push(c);
        }
    }

    for small in tiny {
        if final_clusters.is_empty() {
            final_clusters.push(small);
        } else {
            let best = final_clusters
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| {
                    let wa = cluster_edge_weight(&small, a, inter_module);
                    let wb = cluster_edge_weight(&small, b, inter_module);
                    wa.partial_cmp(&wb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            final_clusters[best].extend_from_slice(&small);
        }
    }

    if final_clusters.is_empty() {
        vec![indices.to_vec()]
    } else {
        final_clusters
    }
}

/// Compute total edge weight between two clusters.
fn cluster_edge_weight(
    a: &[usize],
    b: &[usize],
    inter_module: &HashMap<(usize, usize), f64>,
) -> f64 {
    let mut total = 0.0;
    for &ai in a {
        for &bi in b {
            let key = if ai < bi { (ai, bi) } else { (bi, ai) };
            if let Some(&w) = inter_module.get(&key) {
                total += w;
            }
        }
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_reference_graph;
    use crate::partitioner::partition_modules;
    use crate::types::{DeclKind, Declaration};

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
    fn test_build_module_tree_basic() {
        let mut decls = Vec::new();
        for i in 0..5 {
            let refs = if i > 0 {
                vec![format!("a{}", i - 1)]
            } else {
                vec![]
            };
            decls.push(Declaration {
                name: format!("a{}", i),
                kind: DeclKind::Var,
                byte_range: (i * 10, (i + 1) * 10),
                string_literals: vec!["auth".to_string(), "token".to_string()],
                property_accesses: vec![],
                references: refs,
            });
        }
        for i in 0..5 {
            let refs = if i > 0 {
                vec![format!("b{}", i - 1)]
            } else {
                vec![]
            };
            decls.push(Declaration {
                name: format!("b{}", i),
                kind: DeclKind::Var,
                byte_range: ((i + 5) * 10, (i + 6) * 10),
                string_literals: vec!["stream".to_string(), "delta".to_string()],
                property_accesses: vec![],
                references: refs,
            });
        }

        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(2)).unwrap();
        let config = DecompileConfig::default();
        let tree = build_module_tree(&graph, &modules, &config);

        assert_eq!(tree.depth, 0);
        let total = count_tree_modules(&tree);
        assert!(total > 0, "Tree should contain modules");
    }

    #[test]
    fn test_build_module_tree_single() {
        let decls = vec![make_decl("x", &[], &["hello"])];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(1)).unwrap();
        let config = DecompileConfig::default();
        let tree = build_module_tree(&graph, &modules, &config);
        assert_eq!(tree.depth, 0);
        assert!(!tree.modules.is_empty() || !tree.children.is_empty());
    }

    #[test]
    fn test_tree_names_from_graph_not_hardcoded() {
        // The folder names should come from string literals in the graph.
        let decls = vec![
            make_decl("a", &[], &["streaming", "delta"]),
            make_decl("b", &["a"], &["streaming", "buffer"]),
        ];
        let graph = build_reference_graph(decls);
        let modules = partition_modules(&graph, Some(1)).unwrap();
        let config = DecompileConfig::default();
        let tree = build_module_tree(&graph, &modules, &config);
        // Name should be derived from the string context, not "module_0".
        let all_names = collect_tree_names(&tree);
        assert!(
            all_names.iter().any(|n| n != "module_0" && n != "group_1"),
            "Expected graph-derived name, got: {:?}",
            all_names
        );
    }

    fn count_tree_modules(tree: &ModuleTree) -> usize {
        let mut count = tree.modules.len();
        for child in &tree.children {
            count += count_tree_modules(child);
        }
        count
    }

    fn collect_tree_names(tree: &ModuleTree) -> Vec<String> {
        let mut names = vec![tree.name.clone()];
        for child in &tree.children {
            names.extend(collect_tree_names(child));
        }
        names
    }
}
