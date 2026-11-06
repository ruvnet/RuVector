//! Evidence gate for vulnerability reports: before a lane files "X@latest
//! is vulnerable", `weaver vuln-evidence` resolves what the consumer
//! actually installs (`npm ls <pkg> --all --json`) and checks every copy
//! against the affected range. Origin: a report against @ruvector/sona
//! "latest" came from a nested stale copy (< 0.1.9) while the top-level
//! resolution (0.1.10) was already fixed.

use serde_json::Value;

#[derive(Debug, Clone, PartialEq)]
pub struct Resolved {
    /// Dependency path from the consumer root, e.g. `agentdb > @ruvector/sona`.
    pub path: String,
    pub version: String,
    /// Direct dependency of the consumer (depth 1) or nested.
    pub top_level: bool,
    pub affected: bool,
}

/// Every resolved copy of `pkg` in an `npm ls --all --json` tree.
pub fn resolved_copies(tree: &Value, pkg: &str, affected_range: &str) -> Vec<Resolved> {
    let mut out = Vec::new();
    fn walk(node: &Value, pkg: &str, range: &str, path: &mut Vec<String>, out: &mut Vec<Resolved>) {
        let Some(deps) = node["dependencies"].as_object() else {
            return;
        };
        for (name, child) in deps {
            path.push(name.clone());
            if name == pkg {
                if let Some(v) = child["version"].as_str() {
                    out.push(Resolved {
                        path: path.join(" > "),
                        version: v.to_string(),
                        top_level: path.len() == 1,
                        affected: crate::semver::satisfies(range, v).unwrap_or(false),
                    });
                }
            }
            walk(child, pkg, range, path, out);
            path.pop();
        }
    }
    walk(tree, pkg, affected_range, &mut Vec::new(), &mut out);
    out
}

/// One-line verdict for the brief / ledger reason.
pub fn verdict(pkg: &str, copies: &[Resolved]) -> (bool, String) {
    let bad: Vec<&Resolved> = copies.iter().filter(|c| c.affected).collect();
    if copies.is_empty() {
        return (
            false,
            format!("{pkg} is not installed in this consumer: report not substantiated"),
        );
    }
    if bad.is_empty() {
        let vs: Vec<String> = copies
            .iter()
            .map(|c| format!("{}@{}", c.path, c.version))
            .collect();
        return (
            false,
            format!(
                "every resolved copy is outside the affected range: {}",
                vs.join(", ")
            ),
        );
    }
    let where_: Vec<String> = bad
        .iter()
        .map(|c| {
            format!(
                "{} ({}{})",
                c.path,
                c.version,
                if c.top_level {
                    ", top-level"
                } else {
                    ", nested stale copy"
                }
            )
        })
        .collect();
    (
        true,
        format!("affected copies resolved: {}", where_.join(", ")),
    )
}
