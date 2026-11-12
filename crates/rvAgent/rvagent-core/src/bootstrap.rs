//! Environment bootstrap — a workspace snapshot injected before the loop
//! starts (ADR-273 §3.5).
//!
//! Without it the agent spends its first turns discovering what it is looking
//! at: listing the directory, finding the test command, checking whether the
//! build is already broken. Those turns cost tokens, fill context, and produce
//! nothing the harness could not have supplied for free.
//!
//! The snapshot is deliberately small and factual. It is *not* a repo map or a
//! summary — those are lossy and expensive. It reports only what is cheap to
//! observe and expensive for the agent to discover.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// A cheap, factual snapshot of the workspace.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct EnvironmentSnapshot {
    /// Absolute working directory.
    pub cwd: PathBuf,
    /// Top-level entries, directories marked with a trailing `/`.
    pub entries: Vec<String>,
    /// Detected project kind, e.g. "Rust (cargo)".
    pub project_kind: Option<String>,
    /// Toolchain versions, as `(tool, version)`.
    pub toolchain: Vec<(String, String)>,
    /// The command to run tests, when it can be determined.
    pub test_command: Option<String>,
    /// Whether the tree currently builds, when cheaply checkable.
    ///
    /// `None` means "not checked" — never guessed. Reporting a guess here
    /// would be worse than reporting nothing, since the agent would trust it.
    pub build_ok: Option<bool>,
    /// Current VCS branch, when the workspace is a repository.
    pub vcs_branch: Option<String>,
}

/// How much work the snapshot is allowed to do.
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Maximum top-level entries to list.
    pub max_entries: usize,
    /// Whether to include hidden entries.
    pub include_hidden: bool,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            max_entries: 50,
            include_hidden: false,
        }
    }
}

impl EnvironmentSnapshot {
    /// Collect a snapshot of `root`.
    ///
    /// Filesystem-only: nothing here spawns a process, so it is fast and safe
    /// to run unconditionally. `build_ok` is left `None` — a build check is a
    /// caller decision, since it costs real time.
    pub fn collect(root: &Path, config: &BootstrapConfig) -> Self {
        let entries = list_entries(root, config);
        let project_kind = detect_project_kind(&entries);
        let test_command = detect_test_command(&entries);
        let vcs_branch = detect_branch(root);

        Self {
            cwd: root.to_path_buf(),
            entries,
            project_kind,
            toolchain: Vec::new(),
            test_command,
            build_ok: None,
            vcs_branch,
        }
    }

    /// Render as a prompt section.
    ///
    /// Returns `None` when there is nothing worth saying, so an empty or
    /// unreadable workspace does not inject a misleading stub.
    pub fn to_prompt_section(&self) -> Option<String> {
        if self.entries.is_empty() && self.project_kind.is_none() {
            return None;
        }

        let mut out = String::from("<environment>\n");
        let _ = writeln!(out, "cwd: {}", self.cwd.display());

        if let Some(kind) = &self.project_kind {
            let _ = writeln!(out, "project: {kind}");
        }
        if let Some(branch) = &self.vcs_branch {
            let _ = writeln!(out, "branch: {branch}");
        }
        if let Some(cmd) = &self.test_command {
            let _ = writeln!(out, "tests: {cmd}");
        }
        match self.build_ok {
            Some(true) => {
                let _ = writeln!(out, "build: passing");
            }
            Some(false) => {
                let _ = writeln!(out, "build: FAILING before any of your changes");
            }
            // Silence is correct here: an unchecked build must not read as passing.
            None => {}
        }
        for (tool, version) in &self.toolchain {
            let _ = writeln!(out, "{tool}: {version}");
        }
        if !self.entries.is_empty() {
            let _ = writeln!(out, "contents: {}", self.entries.join(", "));
        }
        out.push_str("</environment>");
        Some(out)
    }

    /// Append the environment section to a base system prompt.
    ///
    /// Returns `base` unchanged when there is nothing to report, so callers
    /// need no conditional of their own. Kept here rather than at the call site
    /// so the composition is covered by tests — the CLI is a binary crate and
    /// anything assembled there is unreachable from a test.
    pub fn augment_prompt(&self, base: &str) -> String {
        match self.to_prompt_section() {
            Some(section) => format!("{base}\n\n{section}"),
            None => base.to_string(),
        }
    }
}

fn list_entries(root: &Path, config: &BootstrapConfig) -> Vec<String> {
    let Ok(read) = std::fs::read_dir(root) else {
        return Vec::new();
    };
    let mut names: Vec<String> = read
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            if !config.include_hidden && name.starts_with('.') {
                return None;
            }
            let is_dir = e.file_type().map(|t| t.is_dir()).unwrap_or(false);
            Some(if is_dir { format!("{name}/") } else { name })
        })
        .collect();
    names.sort();
    // Say so rather than silently showing a partial list: a truncated listing
    // that looks complete invites the agent to conclude a file is absent.
    if names.len() > config.max_entries {
        let hidden = names.len() - config.max_entries;
        names.truncate(config.max_entries);
        names.push(format!("… and {hidden} more"));
    }
    names
}

fn has(entries: &[String], name: &str) -> bool {
    entries.iter().any(|e| e == name)
}

fn detect_project_kind(entries: &[String]) -> Option<String> {
    if has(entries, "Cargo.toml") {
        Some("Rust (cargo)".into())
    } else if has(entries, "package.json") {
        Some("Node (npm)".into())
    } else if has(entries, "pyproject.toml") || has(entries, "setup.py") {
        Some("Python".into())
    } else if has(entries, "go.mod") {
        Some("Go".into())
    } else {
        None
    }
}

fn detect_test_command(entries: &[String]) -> Option<String> {
    if has(entries, "Cargo.toml") {
        Some("cargo test".into())
    } else if has(entries, "package.json") {
        Some("npm test".into())
    } else if has(entries, "pyproject.toml") || has(entries, "setup.py") {
        Some("pytest".into())
    } else if has(entries, "go.mod") {
        Some("go test ./...".into())
    } else {
        None
    }
}

/// Read the current branch from `.git/HEAD` without shelling out.
fn detect_branch(root: &Path) -> Option<String> {
    let head = std::fs::read_to_string(root.join(".git").join("HEAD")).ok()?;
    let head = head.trim();
    head.strip_prefix("ref: refs/heads/")
        .map(str::to_string)
        // A detached HEAD is a raw sha; report a short form rather than nothing.
        .or_else(|| Some(head.chars().take(12).collect()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn detects_a_rust_project_and_its_test_command() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Cargo.toml", "[package]");
        std::fs::create_dir(dir.path().join("src")).unwrap();

        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        assert_eq!(snap.project_kind.as_deref(), Some("Rust (cargo)"));
        assert_eq!(snap.test_command.as_deref(), Some("cargo test"));
        assert!(snap.entries.contains(&"src/".to_string()));
        assert!(snap.entries.contains(&"Cargo.toml".to_string()));
    }

    #[test]
    fn renders_a_prompt_section() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "package.json", "{}");

        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        let section = snap.to_prompt_section().unwrap();
        assert!(section.starts_with("<environment>"));
        assert!(section.ends_with("</environment>"));
        assert!(section.contains("project: Node (npm)"));
        assert!(section.contains("tests: npm test"));
    }

    #[test]
    fn unchecked_build_is_silent_not_passing() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Cargo.toml", "[package]");
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        assert_eq!(snap.build_ok, None);
        let section = snap.to_prompt_section().unwrap();
        assert!(
            !section.contains("build:"),
            "an unchecked build must not be reported at all: {section}"
        );
    }

    #[test]
    fn failing_build_is_stated_plainly() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Cargo.toml", "[package]");
        let mut snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        snap.build_ok = Some(false);
        let section = snap.to_prompt_section().unwrap();
        assert!(section.contains("build: FAILING before any of your changes"));
    }

    #[test]
    fn empty_workspace_yields_no_section() {
        let dir = tempfile::tempdir().unwrap();
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        assert!(snap.to_prompt_section().is_none());
    }

    #[test]
    fn truncation_is_announced() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..30 {
            write(dir.path(), &format!("f{i:02}.txt"), "");
        }
        let config = BootstrapConfig {
            max_entries: 10,
            ..BootstrapConfig::default()
        };
        let snap = EnvironmentSnapshot::collect(dir.path(), &config);
        assert_eq!(snap.entries.len(), 11);
        assert!(snap.entries.last().unwrap().contains("and 20 more"));
    }

    #[test]
    fn hidden_entries_are_excluded_by_default() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), ".secret", "");
        write(dir.path(), "visible.txt", "");
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        assert!(!snap.entries.iter().any(|e| e.starts_with('.')));
        assert!(snap.entries.contains(&"visible.txt".to_string()));
    }

    #[test]
    fn reads_the_branch_from_git_head() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir(dir.path().join(".git")).unwrap();
        write(
            &dir.path().join(".git"),
            "HEAD",
            "ref: refs/heads/feature/x\n",
        );
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        assert_eq!(snap.vcs_branch.as_deref(), Some("feature/x"));
    }

    #[test]
    fn augment_prompt_appends_the_section() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "Cargo.toml", "[package]");
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());

        let prompt = snap.augment_prompt("BASE PROMPT");
        assert!(prompt.starts_with("BASE PROMPT"));
        assert!(prompt.contains("<environment>"));
        assert!(prompt.contains("tests: cargo test"));
    }

    #[test]
    fn augment_prompt_is_a_noop_when_there_is_nothing_to_say() {
        let dir = tempfile::tempdir().unwrap();
        let snap = EnvironmentSnapshot::collect(dir.path(), &BootstrapConfig::default());
        // No trailing whitespace, no empty stub — byte-identical to the input.
        assert_eq!(snap.augment_prompt("BASE PROMPT"), "BASE PROMPT");
    }

    #[test]
    fn unreadable_workspace_does_not_panic() {
        let snap = EnvironmentSnapshot::collect(
            Path::new("/nonexistent/path/xyz"),
            &BootstrapConfig::default(),
        );
        assert!(snap.entries.is_empty());
        assert!(snap.to_prompt_section().is_none());
    }
}
