//! `invalid_resolution`: install a top package fresh, run `npm ls --all`,
//! and report every dependency whose resolved version breaks a declared
//! range, naming the consumer whose range won hoisting. Origin: a fresh
//! `ruvector@0.3.13` install resolved `@ruvector/ruvllm@2.7.0` (invalid
//! against ruvector's `^4.1.0`) because metaharness 0.4.20's `^2.5.6` won.

use crate::model::{Graph, Signal, SignalKind};
use crate::util::clip;
use serde_json::Value;
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
pub struct Invalid {
    pub dep: String,
    pub version: String,
    /// `npm ls`'s reason, e.g. `"^4.1.0" from node_modules/ruvector`.
    pub reason: String,
    pub path: String,
}

/// Every node `npm ls --all --json` marks `invalid`.
pub fn invalid_nodes(tree: &Value) -> Vec<Invalid> {
    fn walk(
        node: &Value,
        path: &mut Vec<String>,
        seen: &mut BTreeSet<(String, String)>,
        out: &mut Vec<Invalid>,
    ) {
        let Some(deps) = node["dependencies"].as_object() else {
            return;
        };
        for (name, child) in deps {
            path.push(name.clone());
            let invalid = match &child["invalid"] {
                Value::String(s) => Some(s.clone()),
                Value::Bool(true) => Some("invalid".into()),
                _ => None,
            };
            if let (Some(reason), Some(v)) = (invalid, child["version"].as_str()) {
                if seen.insert((name.clone(), reason.clone())) {
                    out.push(Invalid {
                        dep: name.clone(),
                        version: v.to_string(),
                        reason,
                        path: path.join(" > "),
                    });
                }
            }
            walk(child, path, seen, out);
            path.pop();
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut Vec::new(), &mut BTreeSet::new(), &mut out);
    out
}

/// Package names present anywhere in the tree.
pub fn tree_names(tree: &Value) -> BTreeSet<String> {
    fn walk(node: &Value, out: &mut BTreeSet<String>) {
        if let Some(deps) = node["dependencies"].as_object() {
            for (name, child) in deps {
                out.insert(name.clone());
                walk(child, out);
            }
        }
    }
    let mut out = BTreeSet::new();
    walk(tree, &mut out);
    out
}

/// Consumers in the tree whose declared range (per the graph) accepts the
/// version that won: the packages to fix.
pub fn winners(graph: &Graph, tree: &Value, inv: &Invalid) -> Vec<String> {
    tree_names(tree)
        .into_iter()
        .filter(|n| {
            graph
                .get(n)
                .and_then(|p| p.dep_ranges.get(&inv.dep))
                .and_then(|r| crate::semver::satisfies(r, &inv.version))
                .unwrap_or(false)
        })
        .collect()
}

/// Signals for one fresh install of `root`. The lane belongs to the winning
/// consumer's repo (that range must move); falls back to the root's repo.
pub fn signals(graph: &Graph, root: &str, tree: &Value) -> Vec<Signal> {
    let root_repo = graph
        .get(root)
        .and_then(|p| p.repo_slug.clone())
        .unwrap_or_else(|| "ruvnet/ruvector".into());
    invalid_nodes(tree)
        .into_iter()
        .map(|inv| {
            // The root is the consumer whose range is unmet, never the winner.
            let win: Vec<String> = winners(graph, tree, &inv)
                .into_iter()
                .filter(|w| w != root)
                .collect();
            let repo = win
                .iter()
                .find_map(|w| graph.get(w).and_then(|p| p.repo_slug.clone()))
                .unwrap_or_else(|| root_repo.clone());
            let won = if win.is_empty() {
                "no consumer in the graph accepts it".to_string()
            } else {
                format!("won by {}", win.join(", "))
            };
            Signal {
                kind: SignalKind::ReleaseRegression,
                repo,
                file: format!("npm ls --all (fresh install of {root})"),
                line: 0,
                evidence: clip(
                    &format!(
                        "{}@{} invalid: {} at {}; {won}",
                        inv.dep, inv.version, inv.reason, inv.path
                    ),
                    240,
                ),
                subject: Some(crate::release::subject(
                    "invalid_resolution",
                    &format!("{root}->{}", inv.dep),
                )),
                keyword: Some("invalid_resolution".into()),
                candidate: Some(inv.dep.clone()),
                confidence: 0.95,
                severity: Some("high".into()),
                occurrences: 1,
            }
        })
        .collect()
}
