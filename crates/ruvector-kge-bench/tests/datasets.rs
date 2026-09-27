//! Dataset loader parity with the JS harness, pin enforcement, and the
//! identity / leakage asserts (plan M3 acceptance).

use ruvector_kge_bench::datasets::{
    self, carve_transfer, from_bytes, parse_triples, stable_split_hash, DatasetSpec, Expected,
    RawTriple, SPECS,
};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn rt(s: &str, r: &str, o: &str) -> RawTriple {
    RawTriple {
        s: s.into(),
        r: r.into(),
        o: o.into(),
    }
}

/// Every URL commit, sha256 pin and canonical count equals the `.mjs` loader's
/// (the JS side is the reference the ADR cites).
#[test]
fn datasets_match_js_loaders() {
    for spec in SPECS {
        let js = std::fs::read_to_string(
            repo_root().join(format!("npm/packages/kge/bench/datasets/{}.mjs", spec.name)),
        )
        .unwrap();
        assert!(js.contains(spec.commit), "{}: commit", spec.name);
        for pin in spec.pins {
            assert!(
                js.contains(pin),
                "{}: pin {pin} not in the .mjs loader",
                spec.name
            );
        }
        let e = spec.expected;
        let needle = format!(
            "train: {}, valid: {}, test: {}, entities: {}, relations: {}",
            e.train, e.valid, e.test, e.entities, e.relations
        );
        assert!(
            js.contains(&needle),
            "{}: canonical counts differ from the .mjs loader",
            spec.name
        );
        for url in spec.urls {
            let file = url.rsplit('/').next().unwrap();
            assert!(
                url.contains(spec.commit) && (js.contains(file)),
                "{}: url {url}",
                spec.name
            );
        }
    }
    assert_eq!(
        datasets::spec("wn18rr").unwrap().cache_file("train"),
        "wn18rr@2e440e0f-train.txt"
    );
    assert_eq!(
        datasets::spec("codexm").unwrap().cache_file("valid"),
        "codexm@3132e426-valid"
    );
}

/// Constants produced by the JS `carveTransfer` / `stableSplitHash`.
#[test]
fn carve_and_split_hash_match_js() {
    let v = vec![
        rt("a", "b", "c"),
        rt("x", "y", "z"),
        rt("00001", "_hypernym", "00002"),
        rt("q", "r", "s"),
        rt("1", "2", "3"),
    ];
    let (valid, transfer) = carve_transfer(v);
    assert_eq!(
        valid,
        vec![
            rt("a", "b", "c"),
            rt("00001", "_hypernym", "00002"),
            rt("q", "r", "s")
        ]
    );
    assert_eq!(transfer, vec![rt("x", "y", "z"), rt("1", "2", "3")]);
    let h = stable_split_hash(&[rt("b", "r", "a"), rt("a", "r", "b")]);
    assert_eq!(
        h,
        "cbf366618cee9b682b65cb7080ad06b23d22ac38c7494ea286023d9f7d6ad9a7"
    );
}

#[test]
fn parse_handles_crlf_blank_lines_and_spaces() {
    let lf = parse_triples("a\tr\tb\n\nc\tr\td\n");
    let crlf = parse_triples("a\tr\tb\r\n\r\nc\tr\td\r\n");
    assert_eq!(lf, crlf);
    assert_eq!(lf.len(), 2);
    assert_eq!(parse_triples("a r b\n"), vec![rt("a", "r", "b")]);
}

fn synth_spec(expected: Expected) -> &'static DatasetSpec {
    Box::leak(Box::new(DatasetSpec {
        name: "synth",
        commit: "0000000000000000000000000000000000000000",
        cache_ext: ".txt",
        urls: ["u/train.txt", "u/valid.txt", "u/test.txt"],
        pins: ["", "", ""],
        expected,
        licence: "test",
    }))
}

#[test]
fn identity_and_leakage_asserts_fire() {
    let train = "e1\tr1\te2\ne2\tr1\te3\ne3\tr2\te1\n";
    let valid = "e1\tr2\te3\n";
    let test = "e2\tr2\te1\n";
    let ok = Expected {
        train: 3,
        valid: 1,
        test: 1,
        entities: 3,
        relations: 2,
    };
    let ds = from_bytes(
        synth_spec(ok),
        [train.as_bytes(), valid.as_bytes(), test.as_bytes()],
    )
    .unwrap();
    assert_eq!(ds.num_entities, 3);
    // Sorted-label ids: e1=0, e2=1, e3=2; r1=0, r2=1.
    assert_eq!(ds.train[0], ruvector_kge::Triple::new(0, 0, 1));
    // Wrong canonical entity count (a doctored variant) is refused.
    let wrong = Expected { entities: 4, ..ok };
    let e = from_bytes(
        synth_spec(wrong),
        [train.as_bytes(), valid.as_bytes(), test.as_bytes()],
    )
    .unwrap_err();
    assert!(e.to_string().contains("not the canonical dataset"), "{e}");
    // A doctored test split that repeats a train triple trips the leakage assert.
    let leaky = "e1\tr1\te2\n";
    let e = from_bytes(
        synth_spec(ok),
        [train.as_bytes(), valid.as_bytes(), leaky.as_bytes()],
    )
    .unwrap_err();
    assert!(e.to_string().contains("leakage"), "{e}");
    // The filter store is the split union; test only when asked.
    assert_eq!(ds.filter_store(false).unwrap().len(), 4);
    assert_eq!(ds.filter_store(true).unwrap().len(), 5);
}

#[test]
fn tampered_cache_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let p = d.path().join("f.txt");
    std::fs::write(&p, b"a\tr\tb\n").unwrap();
    let pin = "0000000000000000000000000000000000000000000000000000000000000000";
    let e = datasets::fetch_verified("https://invalid.example/never-fetched", pin, &p).unwrap_err();
    assert!(
        e.to_string().contains("sha256 mismatch for cached file"),
        "{e}"
    );
}

/// Full-dataset parity with the JS loader on the shared cache (skipped when
/// the cache has not been warmed; `node npm/packages/kge/bench/datasets/wn18rr.mjs`).
#[test]
fn real_cache_matches_js_splits_hash() {
    let cache = repo_root().join("npm/packages/kge/bench/.cache");
    let cases = [
        (
            "wn18rr",
            "07bac99b8d2f4e30e03756d682950d738b9e068f53e4dcdfc1368d02e586850c",
            [86835, 2128, 906, 3134, 40943, 11],
        ),
        (
            "fb15k237",
            "e800ac1dbee247637d83c512125dc39fc1e8b2473cee48d4ef1a6df98ffce36b",
            [272115, 12245, 5290, 20466, 14541, 237],
        ),
        (
            "codexm",
            "878f577e2d03e78361272f0380f08cd4539ec725bd74b436a96177432c2b1e44",
            [185584, 7214, 3096, 10311, 17050, 51],
        ),
    ];
    for (name, js_hash, c) in cases {
        let spec = datasets::spec(name).unwrap();
        if !["train", "valid", "test"]
            .iter()
            .all(|k| cache.join(spec.cache_file(k)).exists())
        {
            eprintln!("skip {name}: cache not warmed");
            continue;
        }
        let ds = datasets::load(name, &cache).unwrap();
        assert_eq!(
            ds.splits_hash, js_hash,
            "{name}: splits_hash differs from the JS loader"
        );
        let k = ds.counts();
        assert_eq!(
            [
                k.train,
                k.valid,
                k.transfer,
                k.test,
                k.entities,
                k.relations
            ],
            c,
            "{name}"
        );
    }
}
