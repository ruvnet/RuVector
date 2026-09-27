//! `--final` (one-time test scoring) and `--verify-final` (ADR-007 §2.6, §2.8;
//! plan M3).
//!
//! `--final` refuses unless **all** hold, checked against **origin**, never
//! the working tree:
//! - the seed is a pre-registered final seed {0..4} and the config is a grid
//!   config (C1–C8);
//! - tag `kge-prereg-v1` resolves on origin, `git verify-tag` passes with a
//!   signer listed in `SIGNERS` at the tag, and the tag is an ancestor of HEAD;
//! - origin's `kge-final-ledger` has `selection.json` naming this config hash
//!   for the dataset, its ledger passes the CI rules, and it has no line for
//!   `(dataset, seed)` — except an `intent` without a `result` carrying the
//!   same checkpoint sha256 (a retry: no new intent, result only).
//!
//! Then it pushes an `intent` (fast-forward; a rejected push scores nothing),
//! scores test once, writes the receipt with the per-query Bottom/RANDOM rank
//! vectors and their sha256, and pushes a `result` line.
//!
//! Every git interaction goes through [`FinalGit`], so the whole protocol is
//! unit-tested on synthetic data with a fake (`tests/final_mode.rs`); the real
//! implementation is `gitops::GitCli`.

use crate::ledger::{self, key_state, KeyState, Line, Selection, FINAL_SEEDS};
use anyhow::{bail, Context, Result};

/// A tag as seen on origin: the tag object and the commit it points at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TagInfo {
    pub tag_object: String,
    pub commit: String,
}

/// origin's ledger branch at one commit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OriginLedger {
    /// Commit of `origin/kge-final-ledger` these files were read from.
    pub head: String,
    pub ledger: String,
    pub selection: Option<String>,
}

/// The git operations `--final` needs. Implementations must read origin, not
/// the working tree.
pub trait FinalGit {
    /// `git ls-remote origin refs/tags/<tag>`; `None` when absent.
    fn origin_tag(&self, tag: &str) -> Result<Option<TagInfo>>;
    /// `git verify-tag` passes and the signing key's fingerprint is listed in
    /// `SIGNERS` as committed at the tag. `Err` otherwise.
    fn verify_tag_signer(&self, tag: &TagInfo) -> Result<()>;
    /// Is `commit` an ancestor of (or equal to) HEAD?
    fn is_ancestor_of_head(&self, commit: &str) -> Result<bool>;
    /// Fetch origin's ledger branch and read the ledger + selection from it.
    fn fetch_origin_ledger(&self) -> Result<OriginLedger>;
    /// Append `line` to the ledger on top of `base.head` and push it as a
    /// fast-forward. `Err` if the push is rejected (origin moved) or fails.
    fn push_ledger_line(&self, base: &OriginLedger, line: &str) -> Result<()>;
}

/// What is being scored.
#[derive(Debug, Clone)]
pub struct FinalRequest {
    pub dataset: String,
    pub seed: u64,
    pub config_hash: String,
    pub config_is_grid: bool,
    /// sha256 of the tables to be scored (the best-on-valid export's weights).
    pub checkpoint_sha256: String,
}

/// A request that passed every refusal check.
#[derive(Debug, Clone)]
pub struct Authorized {
    pub tag: TagInfo,
    pub origin: OriginLedger,
    /// True when origin already holds our `intent` (no new intent is pushed).
    pub retry: bool,
}

/// What the scoring step reports back (for the `result` line).
#[derive(Debug, Clone)]
pub struct Scored {
    pub receipt_sha256: String,
    pub bottom_ranks_sha256: String,
    pub random_ranks_sha256: String,
}

/// Run every refusal check; no side effects.
pub fn authorize(git: &dyn FinalGit, req: &FinalRequest) -> Result<Authorized> {
    if !FINAL_SEEDS.contains(&req.seed) {
        bail!(
            "refusing --final: seed {} is not a pre-registered final seed {{0..4}}",
            req.seed
        );
    }
    if !req.config_is_grid {
        bail!("refusing --final: only a pre-registered grid config (C1..C8) can be scored");
    }
    let tag = git
        .origin_tag(ledger::PREREG_TAG)
        .context("refusing --final: cannot query origin for the pre-registration tag")?
        .with_context(|| {
            format!(
                "refusing --final: tag {} does not exist on origin",
                ledger::PREREG_TAG
            )
        })?;
    git.verify_tag_signer(&tag).with_context(|| {
        format!(
            "refusing --final: tag {} failed signature/signer verification",
            ledger::PREREG_TAG
        )
    })?;
    if !git.is_ancestor_of_head(&tag.commit)? {
        bail!(
            "refusing --final: tag {} ({}) is not an ancestor of HEAD",
            ledger::PREREG_TAG,
            tag.commit
        );
    }
    let origin = git
        .fetch_origin_ledger()
        .context("refusing --final: cannot fetch origin's ledger branch")?;
    let sel_text = origin
        .selection
        .as_deref()
        .context("refusing --final: origin has no selection.json")?;
    let sel = Selection::parse(sel_text)?;
    let want = sel.datasets.get(&req.dataset).with_context(|| {
        format!(
            "refusing --final: selection.json has no entry for '{}'",
            req.dataset
        )
    })?;
    if want.config_hash != req.config_hash {
        bail!(
            "refusing --final: config {} is not the selected config for '{}' ({})",
            req.config_hash,
            req.dataset,
            want.config_hash
        );
    }
    ledger::check(&origin.ledger, Some(&sel), None, Some(&tag.tag_object))
        .context("refusing --final: origin's ledger fails the ledger rules")?;
    let lines = ledger::parse(&origin.ledger)?;
    let retry = match key_state(&lines, &req.dataset, req.seed) {
        KeyState::Free => false,
        KeyState::Done => bail!(
            "refusing --final: ({}, seed {}) is already scored on origin's ledger",
            req.dataset,
            req.seed
        ),
        KeyState::IntentOnly(Line::Intent {
            checkpoint_sha256,
            config_hash,
            tag_sha,
            ..
        }) => {
            if *checkpoint_sha256 != req.checkpoint_sha256
                || *config_hash != req.config_hash
                || *tag_sha != tag.tag_object
            {
                bail!("refusing --final: origin holds an intent for ({}, seed {}) with a different checkpoint/config/tag", req.dataset, req.seed);
            }
            true
        }
        KeyState::IntentOnly(_) => unreachable!("key_state returns intents only"),
    };
    Ok(Authorized { tag, origin, retry })
}

/// The full protocol: authorize, push `intent` (unless retrying), score via
/// `score(&tag)`, push `result`. `score` runs at most once and only after the
/// intent is on origin.
pub fn run_final(
    git: &dyn FinalGit,
    req: &FinalRequest,
    utc: &str,
    score: impl FnOnce(&Authorized) -> Result<Scored>,
) -> Result<(Authorized, Scored)> {
    let auth = authorize(git, req)?;
    if !auth.retry {
        let intent = Line::Intent {
            dataset: req.dataset.clone(),
            seed: req.seed,
            config_hash: req.config_hash.clone(),
            tag_sha: auth.tag.tag_object.clone(),
            checkpoint_sha256: req.checkpoint_sha256.clone(),
            utc: utc.into(),
        };
        git.push_ledger_line(&auth.origin, &intent.to_json_line())
            .context("intent push rejected or failed — nothing was scored")?;
    }
    let scored = score(&auth)?;
    // Re-read origin: the result goes on top of whatever is there now, and
    // our intent must be the open one for this key.
    let now = git.fetch_origin_ledger()?;
    let lines = ledger::parse(&now.ledger)?;
    match key_state(&lines, &req.dataset, req.seed) {
        KeyState::IntentOnly(Line::Intent {
            checkpoint_sha256, ..
        }) if *checkpoint_sha256 == req.checkpoint_sha256 => {}
        _ => bail!(
            "origin's ledger no longer holds our open intent for ({}, seed {}); result not pushed",
            req.dataset,
            req.seed
        ),
    }
    let result = Line::Result {
        dataset: req.dataset.clone(),
        seed: req.seed,
        config_hash: req.config_hash.clone(),
        tag_sha: auth.tag.tag_object.clone(),
        checkpoint_sha256: req.checkpoint_sha256.clone(),
        receipt_sha256: scored.receipt_sha256.clone(),
        bottom_ranks_sha256: scored.bottom_ranks_sha256.clone(),
        random_ranks_sha256: scored.random_ranks_sha256.clone(),
        utc: utc.into(),
    };
    git.push_ledger_line(&now, &result.to_json_line())
        .context("scored, but the result push failed; re-run --final to push the result (no new intent is written)")?;
    Ok((auth, scored))
}
