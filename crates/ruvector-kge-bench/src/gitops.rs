//! [`FinalGit`] over the `git` CLI. Reads origin only: the tag via
//! `ls-remote`, the ledger and selection via `fetch` + `show <commit>:<path>`,
//! and appends by committing in a throw-away detached worktree and pushing
//! `HEAD:refs/heads/kge-final-ledger` **without** `--force` (origin rejects a
//! non-fast-forward, which is the concurrency guard). The caller's working
//! tree and branches are never modified.

use crate::final_mode::{FinalGit, OriginLedger, TagInfo};
use crate::ledger::{LEDGER_BRANCH, LEDGER_PATH, SELECTION_PATH, SIGNERS_PATH};
use anyhow::{bail, Context, Result};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The real implementation, rooted at a repository and a remote name.
pub struct GitCli {
    pub repo: PathBuf,
    pub remote: String,
}

impl GitCli {
    pub fn new(repo: impl Into<PathBuf>) -> Self {
        Self {
            repo: repo.into(),
            remote: "origin".into(),
        }
    }

    fn raw(&self, dir: &Path, args: &[&str]) -> Result<Output> {
        Command::new("git")
            .arg("-C")
            .arg(dir)
            .args(args)
            .output()
            .with_context(|| format!("spawn git {}", args.join(" ")))
    }

    fn git(&self, args: &[&str]) -> Result<String> {
        self.git_in(&self.repo, args)
    }

    fn git_in(&self, dir: &Path, args: &[&str]) -> Result<String> {
        let o = self.raw(dir, args)?;
        if !o.status.success() {
            bail!(
                "git {} failed: {}",
                args.join(" "),
                String::from_utf8_lossy(&o.stderr).trim()
            );
        }
        Ok(String::from_utf8_lossy(&o.stdout).into_owned())
    }

    /// `git show <commit>:<path>`; `None` when the path does not exist there.
    fn show(&self, commit: &str, path: &str) -> Result<Option<String>> {
        let o = self.raw(&self.repo, &["show", &format!("{commit}:{path}")])?;
        Ok(o.status
            .success()
            .then(|| String::from_utf8_lossy(&o.stdout).into_owned()))
    }
}

/// Fingerprints listed in a `SIGNERS` file (one per line, `#` comments).
pub fn parse_signers(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| l.replace(' ', "").to_uppercase())
        .collect()
}

/// Signing-key fingerprints reported by `git verify-tag --raw` (gpg
/// `VALIDSIG <fpr>` and its primary-key fingerprint) or by an SSH signature
/// (`... with <TYPE> key SHA256:<fp>`), upper-cased.
pub fn parse_verify_output(stderr: &str) -> Vec<String> {
    let mut out = Vec::new();
    for l in stderr.lines() {
        if let Some(rest) = l.strip_prefix("[GNUPG:] VALIDSIG ") {
            let f: Vec<&str> = rest.split_whitespace().collect();
            if let Some(fpr) = f.first() {
                out.push(fpr.to_uppercase());
            }
            if let Some(primary) = f.get(9) {
                out.push(primary.to_uppercase());
            }
        } else if let Some(i) = l.find(" key SHA256:") {
            let fp = l[i + 5..].split_whitespace().next().unwrap_or("");
            if l.starts_with("Good ") {
                out.push(fp.to_uppercase());
            }
        }
    }
    out
}

impl FinalGit for GitCli {
    fn origin_tag(&self, tag: &str) -> Result<Option<TagInfo>> {
        let r = format!("refs/tags/{tag}");
        let out = self.git(&["ls-remote", &self.remote, &r, &format!("{r}^{{}}")])?;
        let (mut obj, mut peeled) = (None, None);
        for l in out.lines() {
            let mut it = l.split_whitespace();
            let (sha, name) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
            if name == r {
                obj = Some(sha.to_string());
            } else if name == format!("{r}^{{}}") {
                peeled = Some(sha.to_string());
            }
        }
        Ok(obj.map(|o| TagInfo {
            commit: peeled.unwrap_or_else(|| o.clone()),
            tag_object: o,
        }))
    }

    fn verify_tag_signer(&self, tag: &TagInfo) -> Result<()> {
        // Fetch the tag object itself (by id, into no ref) and verify it.
        self.git(&[
            "fetch",
            "--no-tags",
            &self.remote,
            &format!("refs/tags/{}", crate::ledger::PREREG_TAG),
        ])?;
        let fetched = self.git(&["rev-parse", "FETCH_HEAD"])?;
        if fetched.trim() != tag.tag_object {
            bail!(
                "fetched tag object {} != ls-remote {}",
                fetched.trim(),
                tag.tag_object
            );
        }
        let o = self.raw(&self.repo, &["verify-tag", "--raw", &tag.tag_object])?;
        if !o.status.success() {
            bail!(
                "git verify-tag failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            );
        }
        let seen = parse_verify_output(&String::from_utf8_lossy(&o.stderr));
        let signers = parse_signers(
            &self
                .show(&tag.commit, SIGNERS_PATH)?
                .with_context(|| format!("no {SIGNERS_PATH} at the tag"))?,
        );
        if !seen.iter().any(|f| signers.contains(f)) {
            bail!("tag signer {seen:?} is not listed in {SIGNERS_PATH} at the tag");
        }
        Ok(())
    }

    fn is_ancestor_of_head(&self, commit: &str) -> Result<bool> {
        let o = self.raw(&self.repo, &["merge-base", "--is-ancestor", commit, "HEAD"])?;
        match o.status.code() {
            Some(0) => Ok(true),
            Some(1) => Ok(false),
            _ => bail!(
                "git merge-base failed: {}",
                String::from_utf8_lossy(&o.stderr).trim()
            ),
        }
    }

    fn fetch_origin_ledger(&self) -> Result<OriginLedger> {
        self.git(&[
            "fetch",
            "--no-tags",
            &self.remote,
            &format!("refs/heads/{LEDGER_BRANCH}"),
        ])?;
        let head = self
            .git(&["rev-parse", "FETCH_HEAD^{commit}"])?
            .trim()
            .to_string();
        let ledger = self
            .show(&head, LEDGER_PATH)?
            .with_context(|| format!("origin/{LEDGER_BRANCH} has no {LEDGER_PATH}"))?;
        Ok(OriginLedger {
            selection: self.show(&head, SELECTION_PATH)?,
            ledger,
            head,
        })
    }

    fn push_ledger_line(&self, base: &OriginLedger, line: &str) -> Result<()> {
        let wt = std::env::temp_dir().join(format!(
            "kge-ledger-{}-{}",
            std::process::id(),
            &base.head[..12.min(base.head.len())]
        ));
        self.git(&[
            "worktree",
            "add",
            "--detach",
            &wt.to_string_lossy(),
            &base.head,
        ])?;
        let res = (|| -> Result<()> {
            let path = wt.join(LEDGER_PATH);
            let cur = std::fs::read_to_string(&path)?;
            if cur != base.ledger {
                bail!(
                    "ledger at {} differs from the one authorised against",
                    base.head
                );
            }
            let mut next = cur;
            if !next.is_empty() && !next.ends_with('\n') {
                next.push('\n');
            }
            next.push_str(line);
            next.push('\n');
            std::fs::write(&path, next)?;
            self.git_in(&wt, &["add", LEDGER_PATH])?;
            self.git_in(
                &wt,
                &[
                    "commit",
                    "-q",
                    "-m",
                    "kge ledger: append line (ruvector-kge-bench --final)",
                ],
            )?;
            self.git_in(
                &wt,
                &[
                    "push",
                    &self.remote,
                    &format!("HEAD:refs/heads/{LEDGER_BRANCH}"),
                ],
            )?;
            Ok(())
        })();
        let _ = self.git(&["worktree", "remove", "--force", &wt.to_string_lossy()]);
        res
    }
}
