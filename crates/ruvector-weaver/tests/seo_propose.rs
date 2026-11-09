//! Offline tests for ADR-356 proposals: format-preserving package.json
//! edits, diffs, README cross-links, the flow SVG and apply planning.

mod seo_common;
use seo_common::*;

#[test]
fn json_text_edits_preserve_order_and_format() {
    let src = read("repo/packages/vecdb/package.json");
    let out = jsontext::set(&src, "homepage", &json!("https://ruv.io"), &[]).unwrap();
    assert_eq!(out.lines().count(), src.lines().count(), "in-place replace");
    assert!(out.contains("  \"homepage\": \"https://ruv.io\","));
    let out2 = jsontext::set(
        &out,
        "bugs",
        &json!({"url": "https://x/issues"}),
        &["homepage"],
    )
    .unwrap();
    let hp = out2.find("\"homepage\"").unwrap();
    let bugs = out2.find("\"bugs\"").unwrap();
    let lic = out2.find("\"license\"").unwrap();
    assert!(hp < bugs && bugs < lic, "inserted after the anchor");
    assert!(out2.contains("  \"bugs\": {\n    \"url\": \"https://x/issues\"\n  },"));
    assert_eq!(
        jsontext::set(&out2, "bugs", &json!({"url": "https://x/issues"}), &[]).unwrap(),
        out2
    );
    let four = "{\n    \"name\": \"a\"\n}\n";
    let o = jsontext::set(four, "keywords", &json!(["x", "y"]), &["name"]).unwrap();
    assert!(o.contains("    \"keywords\": [\n        \"x\","), "{o}");
    assert!(jsontext::set("[1]", "a", &json!(1), &[]).is_err());
}

#[test]
fn unified_diff_shows_only_the_change() {
    let a = "a\nb\nc\nd\ne\nf\ng\nh\n";
    let b = "a\nb\nc\nD\ne\nf\ng\nh\n";
    let d = diff::unified("x.txt", a, b);
    assert!(
        d.starts_with("--- a/x.txt\n+++ b/x.txt\n@@ -1,7 +1,7 @@\n"),
        "{d}"
    );
    assert!(d.contains("-d\n+D\n"));
    assert!(diff::unified("x", a, a).is_empty());
}

#[test]
fn propose_builds_links_from_real_edges_and_is_idempotent() {
    let t = tax();
    let g = graph();
    let p = g.get("@acme/vecdb").unwrap();
    let f = vecdb_facts();
    let rf = RepoFacts {
        slug: "acme/stack".into(),
        default_branch: "main".into(),
        ..RepoFacts::default()
    };
    let pp = propose::propose_package(&t, &g, p, &f, Some(&source()), Some(&rf)).unwrap();
    assert_eq!(
        pp.works_with,
        vec!["@acme/core"],
        "only fabric deps, never left-pad"
    );
    assert_eq!(pp.used_by, vec!["@acme/app"]);
    assert_eq!(
        &pp.keywords[..3],
        &["vector-database", "hnsw", "embeddings"]
    );
    assert!(pp.keywords.contains(&"acme".to_string()), "brand");
    assert!(
        !pp.keywords.contains(&"vectordb".to_string()),
        "alias of a present topic is deduped"
    );
    assert!(!pp.keywords.contains(&"blazing".to_string()));
    assert!(
        pp.keywords.len() >= 8 && pp.keywords.len() <= 16,
        "{:?}",
        pp.keywords
    );
    assert_eq!(
        pp.description,
        "Embedded vector database with HNSW search for agents"
    );
    let pj: Value = serde_json::from_str(&pp.edits[0].after).unwrap();
    assert_eq!(pj["homepage"], "https://ruv.io");
    assert_eq!(pj["repository"]["directory"], "packages/vecdb");
    assert_eq!(
        pj["repository"]["url"],
        "git+https://github.com/acme/stack.git"
    );
    assert_eq!(pj["bugs"]["url"], "https://github.com/acme/stack/issues");
    assert_eq!(
        pj["dependencies"]["left-pad"], "^1.3.0",
        "dependencies untouched"
    );
    let md = &pp.edits[1].after;
    assert!(
        md.contains("## Historic notes\n\nKept from 2024."),
        "historic notes preserved"
    );
    assert!(md.contains("npm install @acme/vecdb"));
    assert!(md.contains(LINKS_START) && md.contains(FOOTER_START));
    assert!(md.contains("[`@acme/core`](https://www.npmjs.com/package/@acme/core)"));
    assert!(md.contains("https://raw.githubusercontent.com/acme/stack/main/packages/vecdb/assets/acme-vecdb-flow.svg"));
    assert!(
        ruvector_weaver::readme::lint(md).is_empty(),
        "no unmeasured numbers added"
    );
    assert!(pp.diff.contains("+  \"homepage\": \"https://ruv.io\""));
    // Running again over the proposed output changes nothing.
    let again = SourcePkg {
        rel_dir: "packages/vecdb".into(),
        package_json: pp.edits[0].after.clone(),
        readme: Some(("README.md".into(), md.clone())),
    };
    let pp2 = propose::propose_package(&t, &g, p, &f, Some(&again), Some(&rf)).unwrap();
    assert_eq!(pp2.edits[0].after, pp.edits[0].after);
    assert_eq!(pp2.edits[1].after, *md);
}

#[test]
fn flow_svg_is_small_animated_and_self_contained() {
    let steps: Vec<String> = ["query vector", "HNSW index", "top-k results"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let s = flow::render("@acme/vecdb", &steps).unwrap();
    assert!(s.len() < flow::MAX_BYTES);
    assert!(s.contains("<animateMotion") && s.contains("<animate "));
    for bad in ["<script", "@import", "url(", "<foreignObject"] {
        assert!(!s.contains(bad), "{bad}");
    }
    assert!(
        !s.replace("http://www.w3.org", "").contains("http"),
        "no external refs"
    );
    assert!(flow::render("x", &["recall 97%".to_string(), "b".to_string()]).is_err());
    assert_eq!(flow::file_name("@ruvector/rvf"), "ruvector-rvf-flow.svg");
}

#[test]
fn apply_bumps_patch_from_the_higher_version() {
    assert_eq!(
        propose::bump_patch("1.2.3", "1.2.5").as_deref(),
        Some("1.2.6")
    );
    assert_eq!(
        propose::bump_patch("2.0.0-alpha.1", "1.9.0").as_deref(),
        Some("2.0.1")
    );
    assert_eq!(propose::bump_patch("x", "y"), None);
    let t = tax();
    let g = graph();
    let pp = propose::propose_package(
        &t,
        &g,
        g.get("@acme/vecdb").unwrap(),
        &vecdb_facts(),
        Some(&source()),
        None,
    )
    .unwrap();
    let files = apply::with_version(&pp, "1.2.4").unwrap();
    let pj: Value = serde_json::from_str(&files[0].1).unwrap();
    assert_eq!(pj["version"], "1.2.4");
    assert!(files.iter().any(|(p, _)| p == "packages/vecdb/README.md"));
    assert!(files
        .iter()
        .any(|(p, _)| p == "packages/vecdb/assets/acme-vecdb-flow.svg"));
    assert_eq!(
        apply::branch_for("ruvnet/agentdb", "20261108"),
        "weaver/seo-agentdb-20261108"
    );
}

#[test]
fn propose_never_duplicates_hand_written_sections() {
    let t = tax();
    let g = graph();
    let p = g.get("@acme/vecdb").unwrap();
    let mut s = source();
    s.readme = Some((
        "README.md".into(),
        "# x\n\nnpm install @acme/vecdb\n\n## Works with\n\nfoo\n\n## Used by\n\nbar\n\nSee [ruv.io](https://ruv.io).\n".into(),
    ));
    let pp = propose::propose_package(&t, &g, p, &vecdb_facts(), Some(&s), None).unwrap();
    let md = &pp.edits[1].after;
    assert!(
        !md.contains(LINKS_START),
        "existing Works with / Used by kept"
    );
    assert!(!md.contains(FOOTER_START), "existing ruv.io link kept");
}
