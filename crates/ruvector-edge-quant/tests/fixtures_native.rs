//! The checked-in wasm32 fixtures (`tests/fixtures/*`) must be exactly what
//! a native build writes and answers today. Regenerate after an intended
//! format or encoding change with
//! `QUANT_REGEN_FIXTURES=1 cargo test -p ruvector-edge-quant --test fixtures_native`.
#![cfg(not(target_arch = "wasm32"))]

mod fixture;

use fixture::*;
use ruvector_edge_quant::persist::load_from;
use ruvector_edge_quant::Budget;
use std::path::PathBuf;

fn path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(name)
}

#[test]
fn checked_in_fixtures_match_native() {
    let regen = std::env::var_os("QUANT_REGEN_FIXTURES").is_some();
    for (stem, kind) in KINDS {
        let s = build(kind);
        let (bytes, hits) = (snapshot(&s), answers(&s));
        let (bin, txt) = (path(&format!("{stem}.rbqx")), path(&format!("{stem}.hits")));
        if regen {
            std::fs::write(&bin, &bytes).unwrap();
            std::fs::write(&txt, &hits).unwrap();
        }
        assert_eq!(
            std::fs::read(&bin).unwrap(),
            bytes,
            "{stem}: snapshot bytes"
        );
        assert_eq!(
            std::fs::read_to_string(&txt).unwrap(),
            hits,
            "{stem}: answers"
        );
        let back = load_from(&mut bytes.as_slice(), Budget::default()).unwrap();
        assert_eq!(answers(&back), hits, "{stem}: answers after reload");
    }
}
