//! Offline tests for the ADR-356 SEO layer: taxonomy, audit, stale
//! recommendations, repo topics and the portfolio baseline. No network.

mod seo_common;
use seo_common::*;

#[test]
fn embedded_taxonomy_parses_and_every_topic_has_measured_demand() {
    let t = Taxonomy::parse(EMBEDDED_TOPICS).expect("embedded topics.toml");
    assert_eq!(t.meta.homepage, "https://ruv.io");
    assert!(
        t.topics.len() >= 40,
        "taxonomy too small: {}",
        t.topics.len()
    );
    for topic in &t.topics {
        if topic.kind != ruvector_weaver::seo::taxonomy::TopicKind::Brand {
            assert!(
                topic.npm_kw_total > 0 || topic.gh_topic_repos > 0,
                "{} carries no measurement",
                topic.slug
            );
        }
    }
    for r in &t.repos {
        assert!(r.topics.len() <= 20, "{} over 20 topics", r.slug);
    }
}

#[test]
fn taxonomy_rejects_unmeasured_topics_and_bad_primaries() {
    let base = read("topics.toml");
    let low = base.replace(
        "slug = \"hnsw\"\nphrase = \"HNSW\"\nnpm_kw_total = 190\ngh_topic_repos = 681",
        "slug = \"hnsw\"\nphrase = \"HNSW\"\nnpm_kw_total = 3\ngh_topic_repos = 12",
    );
    let e = Taxonomy::parse(&low).unwrap_err().to_string();
    assert!(e.contains("below the demand floor"), "{e}");
    let two = base.replace(
        "primary = [\"vector-database\", \"hnsw\", \"embeddings\"]",
        "primary = [\"vector-database\", \"hnsw\"]",
    );
    assert!(Taxonomy::parse(&two).is_err());
    let niche_primary = base.replace(
        "primary = [\"vector-database\", \"hnsw\", \"embeddings\"]",
        "primary = [\"vector-database\", \"hnsw\", \"link-prediction\"]",
    );
    assert!(
        Taxonomy::parse(&niche_primary).is_err(),
        "niche topics are never primary"
    );
}

#[test]
fn keyword_and_phrase_matching() {
    let t = tax();
    assert!(t.keyword_ok("@acme/vecdb", "VectorDB"));
    assert!(t.keyword_ok("@acme/vecdb", "vecdb"), "own name token");
    assert!(
        t.keyword_ok("@acme/vecdb", "link-prediction"),
        "niche is legal"
    );
    assert!(!t.keyword_ok("@acme/vecdb", "blazing"));
    let p = vec!["vector-database".to_string()];
    assert!(t.has_phrase("An embedded Vector Database for agents", &p));
    assert!(t.has_phrase("vector-database engine", &p));
    assert!(!t.has_phrase("a database of vectors", &p));
}

#[test]
fn audit_flags_each_rule_and_ranks_by_downloads_times_issues() {
    let t = tax();
    let g = graph();
    let f = vecdb_facts();
    assert_eq!(f.version, "1.2.3");
    assert_eq!(f.maintainers, vec!["ruvnet"]);
    let a = audit::audit_package(&t, &g, g.get("@acme/vecdb").unwrap(), &f, now());
    let codes: Vec<&str> = a.issues.iter().map(|i| i.code.as_str()).collect();
    for c in [
        "homepage",
        "keyword_count",
        "keywords_off_taxonomy",
        "description_phrase",
        "bugs_missing",
        "directory_missing",
        "readme_install",
        "readme_links",
        "readme_ruvio",
    ] {
        assert!(codes.contains(&c), "missing {c} in {codes:?}");
    }
    assert!(!codes.contains(&"stale"));
    assert_eq!(
        a.dependents,
        vec!["@acme/app"],
        "platform builds are not consumers"
    );
    assert_eq!(a.score, 5000 * a.issues.len() as u64);

    let clean = PkgFacts {
        name: "@acme/vecdb".into(),
        version: "1.2.4".into(),
        homepage: Some("https://ruv.io/".into()),
        description: Some("Embedded vector database with HNSW search".into()),
        keywords: [
            "vector-database",
            "hnsw",
            "embeddings",
            "rag",
            "rust",
            "wasm",
            "mcp",
            "vecdb",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect(),
        repository_url: Some("git+https://github.com/acme/stack.git".into()),
        repository_directory: Some("packages/vecdb".into()),
        bugs: Some("https://github.com/acme/stack/issues".into()),
        readme: Some(format!(
            "# x\n\nnpm install @acme/vecdb\n\n{LINKS_START}\n[ruv.io](https://ruv.io)\n"
        )),
        import_ok: Some(true),
        ..PkgFacts::default()
    };
    let b = audit::audit_package(&t, &g, g.get("@acme/vecdb").unwrap(), &clean, now());
    assert!(b.issues.is_empty(), "{:?}", b.issues);

    let mut r = AuditReport::default();
    let mut low = a.clone();
    low.name = "z-low".into();
    low.monthly_downloads = 1;
    low.score = low.issues.len() as u64;
    r.packages = vec![low, a];
    audit::finish(&mut r);
    assert_eq!(r.packages[0].name, "@acme/vecdb");
    assert!(audit::table(&r, 5).contains("| 1 | @acme/vecdb |"));
}

#[test]
fn stale_recommendations_never_deprecate_on_downloads_alone() {
    let t = tax();
    let g = graph();
    let old = g.get("acme-old").unwrap();
    let mut f = PkgFacts {
        name: "acme-old".into(),
        version: "0.1.0".into(),
        ..PkgFacts::default()
    };
    let a = audit::audit_package(&t, &g, old, &f, now());
    assert!(a.stale);
    assert!(
        audit::stale_rec(&a)
            .recommendation
            .starts_with("leave as-is"),
        "unverified import"
    );
    f.import_ok = Some(true);
    let a = audit::audit_package(&t, &g, old, &f, now());
    assert_eq!(audit::stale_rec(&a).recommendation, "leave as-is");
    f.import_ok = Some(false);
    let a = audit::audit_package(&t, &g, old, &f, now());
    assert!(audit::stale_rec(&a)
        .recommendation
        .starts_with("deprecate-candidate"));
    let mut used = a.clone();
    used.dependents = vec!["@acme/app".into()];
    assert_eq!(audit::stale_rec(&used).recommendation, "fix+republish");
}

#[test]
fn repo_audit_and_topic_replacement() {
    let t = tax();
    let g = graph();
    let v: Value = serde_json::from_str(&read("gh-repo.json")).unwrap();
    let rf = facts::from_gh_repo("acme/stack", &v);
    assert_eq!(rf.owner, "acme");
    let ra = audit::audit_repo(&t, &g, &rf, "acme");
    let codes: Vec<&str> = ra.issues.iter().map(|i| i.code.as_str()).collect();
    assert_eq!(codes, vec!["homepage", "topics_off_taxonomy"]);
    assert!(ra.owned);
    let rp = propose::propose_repo(&t, &g, &rf);
    assert_eq!(rp.topics_new.len(), 7);
    assert_eq!(
        rp.remove,
        vec!["database", "fast"],
        "off-proposal topics are removed"
    );
    assert!(
        !rp.add.contains(&"vector-database".to_string()),
        "kept, not re-added"
    );
    let args = apply::repo_edit_args(&rp);
    assert_eq!(
        &args[..5],
        &["repo", "edit", "acme/stack", "--homepage", "https://ruv.io"]
    );
    assert!(args
        .windows(2)
        .any(|w| w[0] == "--remove-topic" && w[1] == "database,fast"));
    let reason = apply::repo_reason(&rp);
    assert!(reason.contains("https://acme.example/stack -> https://ruv.io"));
    assert!(
        reason.contains("--homepage 'https://acme.example/stack'"),
        "revert carries the old value"
    );
    assert!(!apply::repo_noop(&rp));
}

#[test]
fn baseline_reconciles_inventory_classes_and_unknowns() {
    let t = tax();
    let g = graph();
    let inv = baseline::parse_inventory(&serde_json::from_str(&read("inventory.json")).unwrap());
    assert_eq!(inv.len(), 4, "deduped");
    let hints = baseline::parse_class_hints(&read("plan.md"));
    assert_eq!(hints["@acme/vecdb"], "pilot");
    assert_eq!(hints["@acme/core-linux-x64-gnu"], "platform");
    assert_eq!(hints["acme-old"], "alias");
    let f = vecdb_facts();
    let row = baseline::row(
        &t,
        "@acme/vecdb",
        true,
        g.get("@acme/vecdb"),
        Some(&f),
        Some("pilot"),
        None,
        "ruvnet",
    );
    let v = serde_json::to_value(&row).unwrap();
    assert_eq!(v["monthly_downloads"], "unknown", "unknowns stay explicit");
    assert_eq!(v["directory"], "unknown");
    assert_eq!(v["readme_present"], true);
    assert_eq!(v["maintainer_ruvnet"], true);
    assert_eq!(v["repo"], "acme/stack");
    assert_eq!(v["class"], "pilot");
    assert!(row.disposition.starts_with("pilot"));
    let fake = PkgFacts {
        version: "1.0.0".into(),
        ..PkgFacts::default()
    };
    let (c, check) = baseline::classify("@acme/fake-platform", Some("platform"), Some(&fake));
    assert_eq!(c, "platform");
    assert!(check.starts_with("conflict"), "{check}");
    let (c, check) = baseline::classify("acme-old", Some("alias"), Some(&fake));
    assert_eq!(c, "alias");
    assert!(
        check.starts_with("unverified"),
        "never assume an alias: {check}"
    );
    let missing = baseline::row(&t, "ghost", true, None, None, None, None, "ruvnet");
    assert!(missing.disposition.starts_with("blocker"));
    assert_eq!(missing.version, Fact::Unknown);
    let held = baseline::disposition(&t, "x", "entry", Some(&fake), Some("acme/held"));
    assert!(held.starts_with("handed off"));
}

#[test]
fn npm_demand_parser_sums_top_five() {
    let v = json!({"total": 571, "objects": [
        {"downloads": {"monthly": 10}}, {"downloads": {"monthly": 5}},
        {"downloads": {"monthly": 1}}, {"downloads": {"monthly": 1}},
        {"downloads": {"monthly": 1}}, {"downloads": {"monthly": 1000}}]});
    assert_eq!(facts::parse_npm_demand(&v), (Some(571), Some(18)));
}
