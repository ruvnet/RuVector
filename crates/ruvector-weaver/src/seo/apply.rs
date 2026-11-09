//! `weaver seo apply`: repo-level metadata goes out immediately through
//! `gh repo edit` (reversible; old values recorded in the ledger first);
//! package-level changes become a branch + PR with a patch bump, published
//! later through the repo's own release path (never from `apply`).

use super::jsontext;
use super::propose::{PkgProposal, RepoProposal};
use anyhow::Result;
use serde_json::json;

/// `gh` arguments for one repo edit (topics replaced, not appended).
pub fn repo_edit_args(p: &RepoProposal) -> Vec<String> {
    let mut a = vec![
        "repo".to_string(),
        "edit".into(),
        p.slug.clone(),
        "--homepage".into(),
        p.homepage_new.clone(),
    ];
    if !p.remove.is_empty() {
        a.push("--remove-topic".into());
        a.push(p.remove.join(","));
    }
    if !p.add.is_empty() {
        a.push("--add-topic".into());
        a.push(p.add.join(","));
    }
    a
}

/// Does the repo already match the proposal?
pub fn repo_noop(p: &RepoProposal) -> bool {
    p.add.is_empty()
        && p.remove.is_empty()
        && p.homepage_old
            .as_deref()
            .is_some_and(|h| super::audit::same_url(h, &p.homepage_new))
}

/// Ledger reason carrying the OLD values so the change can be reverted.
pub fn repo_reason(p: &RepoProposal) -> String {
    format!(
        "repo metadata: homepage {} -> {}; topics [{}] -> [{}]; revert: gh repo edit {} --homepage '{}' --remove-topic '{}' --add-topic '{}'",
        p.homepage_old.as_deref().unwrap_or("(none)"),
        p.homepage_new,
        p.topics_old.join(","),
        p.topics_new.join(","),
        p.slug,
        p.homepage_old.as_deref().unwrap_or(""),
        p.add.join(","),
        p.remove.join(",")
    )
}

/// Package proposal edits plus a patch bump of `version` in package.json.
pub fn with_version(prop: &PkgProposal, version: &str) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    for (i, e) in prop.edits.iter().enumerate() {
        let text = if i == 0 && e.path.ends_with("package.json") {
            jsontext::set(&e.after, "version", &json!(version), &["name"])?
        } else {
            e.after.clone()
        };
        out.push((e.path.clone(), text));
    }
    Ok(out)
}

/// Branch name for a repo's metadata PR.
pub fn branch_for(slug: &str, date: &str) -> String {
    let name = slug.rsplit('/').next().unwrap_or(slug);
    format!("weaver/seo-{name}-{date}")
}
