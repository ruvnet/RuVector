//! `--final` refusal rules and the intent/result protocol, on synthetic data
//! with an injected fake of the git layer (no real tag, origin or test split
//! is touched). Plus `--verify-final`: pass, pre-scoring refusals, and fail.

mod common;

use anyhow::{bail, Result};
use common::{synth_dataset, synth_run};
use ruvector_kge_bench::final_mode::{
    authorize, run_final, FinalGit, FinalRequest, OriginLedger, Scored, TagInfo,
};
use ruvector_kge_bench::ledger::{self, Line};
use ruvector_kge_bench::receipt::validate;
use ruvector_kge_bench::runner::{run, RunOptions};
use ruvector_kge_bench::scoring::{final_score, load_run_dir, verify_final};
use std::cell::{Cell, RefCell};

const TAG_OBJ: &str = "1111111111111111111111111111111111111111";
const TAG_COMMIT: &str = "2222222222222222222222222222222222222222";
const HASH: &str = "cfgcfgcfg";
const CKPT: &str = "ckptckpt";

/// In-memory origin. `pushes` counts accepted pushes; `reject_push` models a
/// concurrent writer (non-fast-forward).
struct FakeGit {
    tag: Option<TagInfo>,
    signer_ok: bool,
    ancestor: bool,
    ledger: RefCell<String>,
    selection: Option<String>,
    reject_push: bool,
    pushes: Cell<usize>,
    head: Cell<u64>,
}

impl FakeGit {
    fn good() -> Self {
        Self {
            tag: Some(TagInfo {
                tag_object: TAG_OBJ.into(),
                commit: TAG_COMMIT.into(),
            }),
            signer_ok: true,
            ancestor: true,
            ledger: RefCell::new(String::new()),
            selection: Some(format!(
                r#"{{"datasets":{{"synth":{{"config_hash":"{HASH}","hpo_receipts":["r"]}}}}}}"#
            )),
            reject_push: false,
            pushes: Cell::new(0),
            head: Cell::new(0),
        }
    }
    fn lines(&self) -> Vec<Line> {
        ledger::parse(&self.ledger.borrow()).unwrap()
    }
}

impl FinalGit for FakeGit {
    fn origin_tag(&self, tag: &str) -> Result<Option<TagInfo>> {
        assert_eq!(tag, ledger::PREREG_TAG);
        Ok(self.tag.clone())
    }
    fn verify_tag_signer(&self, _: &TagInfo) -> Result<()> {
        if self.signer_ok {
            Ok(())
        } else {
            bail!("unsigned or unknown signer")
        }
    }
    fn is_ancestor_of_head(&self, c: &str) -> Result<bool> {
        assert_eq!(c, TAG_COMMIT);
        Ok(self.ancestor)
    }
    fn fetch_origin_ledger(&self) -> Result<OriginLedger> {
        Ok(OriginLedger {
            head: self.head.get().to_string(),
            ledger: self.ledger.borrow().clone(),
            selection: self.selection.clone(),
        })
    }
    fn push_ledger_line(&self, base: &OriginLedger, line: &str) -> Result<()> {
        if self.reject_push || base.head != self.head.get().to_string() {
            bail!("non-fast-forward");
        }
        self.ledger.borrow_mut().push_str(&format!("{line}\n"));
        self.head.set(self.head.get() + 1);
        self.pushes.set(self.pushes.get() + 1);
        Ok(())
    }
}

fn req(seed: u64) -> FinalRequest {
    FinalRequest {
        dataset: "synth".into(),
        seed,
        config_hash: HASH.into(),
        config_is_grid: true,
        checkpoint_sha256: CKPT.into(),
    }
}

fn scored() -> Scored {
    Scored {
        receipt_sha256: "rcpt".into(),
        bottom_ranks_sha256: "b".into(),
        random_ranks_sha256: "r".into(),
    }
}

fn intent_line(ckpt: &str) -> String {
    Line::Intent {
        dataset: "synth".into(),
        seed: 0,
        config_hash: HASH.into(),
        tag_sha: TAG_OBJ.into(),
        checkpoint_sha256: ckpt.into(),
        utc: "t".into(),
    }
    .to_json_line()
        + "\n"
}

/// Every refusal fires before anything is pushed or scored.
#[test]
fn refusals() {
    let cases: Vec<(&str, FakeGit, FinalRequest)> = vec![
        (
            "no tag on origin",
            FakeGit {
                tag: None,
                ..FakeGit::good()
            },
            req(0),
        ),
        (
            "unsigned / unknown signer",
            FakeGit {
                signer_ok: false,
                ..FakeGit::good()
            },
            req(0),
        ),
        (
            "tag not an ancestor of HEAD",
            FakeGit {
                ancestor: false,
                ..FakeGit::good()
            },
            req(0),
        ),
        (
            "no selection.json on origin",
            FakeGit {
                selection: None,
                ..FakeGit::good()
            },
            req(0),
        ),
        (
            "non-selected config hash",
            FakeGit::good(),
            FinalRequest {
                config_hash: "other".into(),
                ..req(0)
            },
        ),
        ("seed 5", FakeGit::good(), req(5)),
        ("seed 100 (the HPO seed)", FakeGit::good(), req(100)),
        (
            "custom config",
            FakeGit::good(),
            FinalRequest {
                config_is_grid: false,
                ..req(0)
            },
        ),
        (
            "dataset missing from selection",
            FakeGit::good(),
            FinalRequest {
                dataset: "wn18rr".into(),
                ..req(0)
            },
        ),
    ];
    for (why, git, r) in cases {
        let called = Cell::new(false);
        let e = run_final(&git, &r, "t", |_| {
            called.set(true);
            Ok(scored())
        });
        assert!(e.is_err(), "{why}: must refuse");
        assert!(!called.get(), "{why}: must not score");
        assert_eq!(git.pushes.get(), 0, "{why}: must not push");
    }
}

/// A key already scored on origin is refused — even when the local working
/// tree's ledger file is empty (the working tree is never consulted).
#[test]
fn key_on_origin_is_refused_regardless_of_working_tree() {
    let git = FakeGit::good();
    run_final(&git, &req(0), "t", |_| Ok(scored())).unwrap();
    let wt = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(wt.path().join("npm/packages/kge/bench/results/final")).unwrap();
    std::fs::write(wt.path().join(ledger::LEDGER_PATH), "").unwrap(); // a "working-tree-only edit"
    let called = Cell::new(false);
    let e = run_final(&git, &req(0), "t", |_| {
        called.set(true);
        Ok(scored())
    })
    .unwrap_err();
    assert!(e.to_string().contains("already scored"), "{e}");
    assert!(!called.get());
    // Another seed is still free.
    run_final(&git, &req(1), "t", |_| Ok(scored())).unwrap();
}

#[test]
fn intent_then_result_and_ledger_stays_valid() {
    let git = FakeGit::good();
    let order = RefCell::new(Vec::new());
    run_final(&git, &req(2), "t", |a| {
        order
            .borrow_mut()
            .push(format!("score@{}", git.lines().len()));
        assert!(!a.retry);
        Ok(scored())
    })
    .unwrap();
    let lines = git.lines();
    assert_eq!(
        order.into_inner(),
        vec!["score@1"],
        "intent is on origin before scoring"
    );
    assert!(matches!(lines[0], Line::Intent { seed: 2, .. }));
    assert!(
        matches!(&lines[1], Line::Result { seed: 2, receipt_sha256, .. } if receipt_sha256 == "rcpt")
    );
    let sel = ledger::Selection::parse(git.selection.as_deref().unwrap()).unwrap();
    ledger::check(&git.ledger.borrow(), Some(&sel), None, Some(TAG_OBJ)).unwrap();
}

#[test]
fn rejected_intent_push_scores_nothing() {
    let git = FakeGit {
        reject_push: true,
        ..FakeGit::good()
    };
    let called = Cell::new(false);
    let e = run_final(&git, &req(0), "t", |_| {
        called.set(true);
        Ok(scored())
    })
    .unwrap_err();
    assert!(format!("{e:#}").contains("nothing was scored"), "{e:#}");
    assert!(!called.get());
    assert!(git.lines().is_empty());
}

#[test]
fn retry_after_intent_pushes_result_only() {
    let git = FakeGit::good();
    git.ledger.borrow_mut().push_str(&intent_line(CKPT));
    let a = authorize(&git, &req(0)).unwrap();
    assert!(a.retry);
    run_final(&git, &req(0), "t", |_| Ok(scored())).unwrap();
    let lines = git.lines();
    assert_eq!(lines.len(), 2, "no second intent");
    assert_eq!(git.pushes.get(), 1);
    // A retry with a different checkpoint is refused (no shopping).
    let git = FakeGit::good();
    git.ledger.borrow_mut().push_str(&intent_line("different"));
    assert!(run_final(&git, &req(0), "t", |_| Ok(scored())).is_err());
}

/// End to end on synthetic data: a 500-epoch no-early-stop run, `--final`
/// through the fake, then `--verify-final` passes on its own ledgered output,
/// refuses (before scoring) unledgered, forged or mismatched inputs, and a
/// failing verification reveals only booleans.
#[test]
fn final_and_verify_final_on_synthetic_data() {
    let ds = synth_dataset();
    let mut cfg = synth_run(ruvector_kge_bench::scoring::FINAL_EPOCHS);
    cfg.eval_every = 100;
    cfg.config_id = "custom".into();
    let d = tempfile::tempdir().unwrap();
    let dir = d.path().join("run");
    run(
        &cfg,
        &ds,
        &RunOptions {
            out: dir.clone(),
            resume: false,
            stop_after_epoch: None,
            repo: ".".into(),
            verbose: false,
        },
    )
    .unwrap();

    // The synthetic dataset has no C1–C8 grid, so its run is `custom`, which
    // `final_score` must refuse; the scoring path is then exercised through
    // `run_final` with the grid flag forced (a test-only request).
    let rd = load_run_dir(&dir, &ds, true).unwrap();
    // A run that is not the 500-epoch final shape is refused outright.
    let short = d.path().join("short");
    run(
        &synth_run(3),
        &ds,
        &RunOptions {
            out: short.clone(),
            resume: false,
            stop_after_epoch: None,
            repo: ".".into(),
            verbose: false,
        },
    )
    .unwrap();
    assert!(load_run_dir(&short, &ds, true).is_err());
    let git = FakeGit {
        selection: Some(format!(
            r#"{{"datasets":{{"synth":{{"config_hash":"{}"}}}}}}"#,
            rd.config.config_hash().unwrap()
        )),
        ..FakeGit::good()
    };
    // Custom -> refused before scoring.
    let out = dir.join("final-receipt.json");
    assert!(final_score(&git, &rd, &ds, &out, std::path::Path::new(".")).is_err());
    assert!(!out.exists());

    // Score through the protocol with the grid flag forced (test-only seam).
    let req = FinalRequest {
        dataset: "synth".into(),
        seed: 3,
        config_hash: rd.config.config_hash().unwrap(),
        config_is_grid: true,
        checkpoint_sha256: rd.manifest.weights_sha256.clone(),
    };
    let receipt = RefCell::new(serde_json::Value::Null);
    run_final(&git, &req, "t", |auth| {
        let r = ruvector_kge_bench::scoring::score_test_receipt(
            &rd,
            &ds,
            auth,
            std::path::Path::new("."),
        )?;
        let s = Scored {
            receipt_sha256: r["receipt_sha256"].as_str().unwrap().into(),
            bottom_ranks_sha256: r["test"]["ranks"]["bottom_sha256"].as_str().unwrap().into(),
            random_ranks_sha256: r["test"]["ranks"]["random_sha256"].as_str().unwrap().into(),
        };
        *receipt.borrow_mut() = r;
        Ok(s)
    })
    .unwrap();
    let r = receipt.into_inner();
    assert_eq!(validate(&r).unwrap(), "final");
    assert_eq!(
        r["test"]["ranks"]["bottom"].as_array().unwrap().len(),
        2 * ds.test.len()
    );
    let lines = git.lines();
    assert!(
        matches!(&lines[1], Line::Result { bottom_ranks_sha256, .. } if *bottom_ranks_sha256 == r["test"]["ranks"]["bottom_sha256"])
    );

    let best = dir.join("best");
    let here = std::path::Path::new(".");
    let v = verify_final(&git, &r, &ds, &best, Some(2), here).unwrap();
    assert_eq!(validate(&v).unwrap(), "verification");
    assert_eq!(v["pass"], true, "{}", v["checks"]);
    assert!(
        v.get("test").is_none(),
        "a verification carries no test metrics or ranks"
    );
    assert_eq!(git.pushes.get(), 2, "verification writes no ledger line");

    // Refused before scoring: the receipt is not on origin's ledger.
    let refused = |g: &FakeGit, rec: &serde_json::Value, tables: &std::path::Path| {
        let e = verify_final(g, rec, &ds, tables, Some(2), here).expect_err("must refuse");
        assert!(format!("{e:#}").contains("nothing was scored"), "{e:#}");
    };
    refused(&FakeGit::good(), &r, &best);
    // An open intent (no result) is not a scoring either.
    let intent_only = FakeGit::good();
    *intent_only.ledger.borrow_mut() =
        git.ledger.borrow().lines().next().unwrap().to_string() + "\n";
    refused(&intent_only, &r, &best);
    // A forged receipt (resealed after an edit) is not the ledgered one.
    let mut forged = r.clone();
    forged["seed"] = 100.into();
    let forged = ruvector_kge_bench::receipt::seal(forged).unwrap();
    refused(&git, &forged, &best);

    // Other tables (one weight perturbed, manifest re-sealed so the load
    // succeeds) are refused on the weights hash, even with the receipt ledgered.
    let bad = d.path().join("bad");
    std::fs::create_dir_all(&bad).unwrap();
    let mut w = std::fs::read(best.join("weights.bin")).unwrap();
    for b in w.iter_mut().take(4 * 64) {
        *b ^= 0x5a;
    }
    std::fs::write(bad.join("weights.bin"), &w).unwrap();
    let mut m: serde_json::Value =
        serde_json::from_slice(&std::fs::read(best.join("manifest.json")).unwrap()).unwrap();
    m["weights_sha256"] = ruvector_kge_bench::canon::sha256_hex(&w).into();
    std::fs::write(bad.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
    refused(&git, &r, &bad);

    // A ledgered receipt whose rank hash does not reproduce fails, and the
    // failing verification still reveals only booleans.
    let mut wrong = r.clone();
    wrong["test"]["ranks"]["bottom_sha256"] = "0".repeat(64).into();
    let wrong = ruvector_kge_bench::receipt::seal(wrong).unwrap();
    let forged_origin = FakeGit::good();
    let lines = git.lines();
    let Line::Result {
        dataset,
        seed,
        config_hash,
        tag_sha,
        checkpoint_sha256,
        random_ranks_sha256,
        utc,
        ..
    } = lines[1].clone()
    else {
        panic!("second line is the result")
    };
    let forged_line = Line::Result {
        dataset,
        seed,
        config_hash,
        tag_sha,
        checkpoint_sha256,
        random_ranks_sha256,
        utc,
        receipt_sha256: wrong["receipt_sha256"].as_str().unwrap().into(),
        bottom_ranks_sha256: "0".repeat(64),
    };
    *forged_origin.ledger.borrow_mut() = forged_line.to_json_line() + "\n";
    let v = verify_final(&forged_origin, &wrong, &ds, &best, Some(2), here).unwrap();
    assert_eq!(v["pass"], false);
    assert_eq!(v["checks"]["bottom_ranks_sha256"], false);
    assert!(v.get("test").is_none());
    let text = v.to_string();
    assert!(
        !text.contains("mrr") && !text.contains("\"ranks\""),
        "no metrics or rank vectors: {text}"
    );
}
