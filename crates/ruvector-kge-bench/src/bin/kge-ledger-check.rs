//! CI check for the append-only test-scoring ledger (ADR-007 §2.6, plan M3).
//!
//! ```text
//! kge-ledger-check <ledger.jsonl> [--selection selection.json]
//!                  [--previous old-ledger.jsonl] [--previous-selection old-selection.json]
//!                  [--tag-sha <sha>]
//! ```
//!
//! Exits 1 on: a malformed line; a seed outside {0..4}; a duplicate intent or
//! result for a (dataset, seed) key; a result without (or disagreeing with) its
//! intent; a second config hash for a dataset or one differing from
//! selection.json; mixed tag SHAs or one differing from --tag-sha; any change
//! to a pre-existing line (with --previous); selection.json changing after the
//! first intent (with --previous-selection).

use ruvector_kge_bench::ledger::{check, parse, Selection};
use std::process::ExitCode;

fn run() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut ledger = None;
    let (mut sel, mut prev, mut prev_sel, mut tag) = (None, None, None, None);
    let mut it = args.iter();
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("{a} needs a value"))
        };
        match a.as_str() {
            "--selection" => sel = Some(val()?),
            "--previous" => prev = Some(val()?),
            "--previous-selection" => prev_sel = Some(val()?),
            "--tag-sha" => tag = Some(val()?),
            "-h" | "--help" => {
                println!("usage: kge-ledger-check <ledger.jsonl> [--selection F] [--previous F] [--previous-selection F] [--tag-sha SHA]");
                return Ok(());
            }
            p if ledger.is_none() && !p.starts_with("--") => ledger = Some(p.to_string()),
            other => anyhow::bail!("unexpected argument {other}"),
        }
    }
    let path = ledger.ok_or_else(|| anyhow::anyhow!("missing <ledger.jsonl>"))?;
    let text = std::fs::read_to_string(&path)?;
    let read = |p: &Option<String>| p.as_ref().map(std::fs::read_to_string).transpose();
    let selection = read(&sel)?.map(|t| Selection::parse(&t)).transpose()?;
    let previous = read(&prev)?;
    check(
        &text,
        selection.as_ref(),
        previous.as_deref(),
        tag.as_deref(),
    )?;
    if let (Some(old_sel), Some(old)) = (read(&prev_sel)?, previous.as_deref()) {
        let had_intent = !parse(old)?.is_empty();
        let now = selection.as_ref().map(serde_json::to_value).transpose()?;
        let before = serde_json::to_value(Selection::parse(&old_sel)?)?;
        if had_intent && now.as_ref() != Some(&before) {
            anyhow::bail!("selection.json changed after the first intent");
        }
    }
    println!("ledger OK: {} line(s) in {path}", parse(&text)?.len());
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("kge-ledger-check: FAIL: {e:#}");
            ExitCode::FAILURE
        }
    }
}
