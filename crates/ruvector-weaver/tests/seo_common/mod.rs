//! Shared fixtures for the ADR-356 SEO tests.
#![allow(dead_code, unused_imports)]

pub use ruvector_weaver::config::EMBEDDED_TOPICS;
pub use ruvector_weaver::model::{ConsumerRepo, Ecosystem, Graph, Package};
pub use ruvector_weaver::seo::audit::{self, AuditReport, PkgFacts, RepoFacts};
pub use ruvector_weaver::seo::baseline::{self, Fact};
pub use ruvector_weaver::seo::propose::{self, SourcePkg};
pub use ruvector_weaver::seo::taxonomy::Taxonomy;
pub use ruvector_weaver::seo::{apply, diff, facts, flow, jsontext, FOOTER_START, LINKS_START};
pub use serde_json::{json, Value};
pub use std::collections::BTreeMap;
pub use std::path::PathBuf;

pub fn fx(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/seo")
        .join(name)
}

pub fn read(name: &str) -> String {
    std::fs::read_to_string(fx(name)).expect("fixture")
}

pub fn tax() -> Taxonomy {
    Taxonomy::parse(&read("topics.toml")).expect("fixture taxonomy")
}

pub fn pkg(
    name: &str,
    dl: u64,
    deps: &[&str],
    kws: &[&str],
    desc: &str,
    published: &str,
) -> Package {
    Package {
        name: name.into(),
        ecosystem: Ecosystem::Npm,
        version: "1.2.3".into(),
        last_publish: Some(published.into()),
        monthly_downloads: dl,
        description: desc.into(),
        keywords: kws.iter().map(|s| s.to_string()).collect(),
        dependencies: deps.iter().map(|s| s.to_string()).collect(),
        optional_dependencies: vec![],
        peer_dependencies: vec![],
        repository: Some("git+https://github.com/acme/stack.git".into()),
        repo_slug: Some("acme/stack".into()),
        platform_binary: false,
        open_ranges: vec![],
        unpacked_size: 0,
        runtime_deps: deps.len(),
        dep_ranges: BTreeMap::new(),
    }
}

pub fn graph() -> Graph {
    let recent = "2026-11-01T00:00:00Z";
    let vecdb = pkg(
        "@acme/vecdb",
        5000,
        &["@acme/core", "left-pad"],
        &["vector", "vectordb", "blazing", "hnsw"],
        "Fast thing that is 100x faster",
        recent,
    );
    let core = pkg("@acme/core", 9000, &[], &["rust"], "Core runtime", recent);
    let app = pkg("@acme/app", 100, &["@acme/vecdb"], &[], "An app", recent);
    let old = pkg(
        "acme-old",
        10,
        &[],
        &[],
        "Old thing",
        "2025-01-01T00:00:00Z",
    );
    let mut plat = pkg(
        "@acme/vecdb-linux-x64-gnu",
        0,
        &["@acme/vecdb"],
        &[],
        "",
        recent,
    );
    plat.platform_binary = true;
    Graph {
        generated_at: recent.into(),
        consumer_repos: vec![ConsumerRepo {
            slug: "acme/stack".into(),
            packages: vec![
                "@acme/vecdb".into(),
                "@acme/core".into(),
                "@acme/app".into(),
            ],
            monthly_downloads: 14100,
        }],
        packages: vec![vecdb, core, app, old, plat],
        repo_aliases: BTreeMap::new(),
    }
}

pub fn now() -> chrono::DateTime<chrono::Utc> {
    chrono::DateTime::parse_from_rfc3339("2026-11-08T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc)
}

pub fn vecdb_facts() -> PkgFacts {
    let doc: Value = serde_json::from_str(&read("packument.json")).unwrap();
    facts::from_packument("@acme/vecdb", &doc)
}

pub fn source() -> SourcePkg {
    SourcePkg {
        rel_dir: "packages/vecdb".into(),
        package_json: read("repo/packages/vecdb/package.json"),
        readme: Some(("README.md".into(), read("repo/packages/vecdb/README.md"))),
    }
}
