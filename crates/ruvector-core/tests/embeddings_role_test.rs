//! ADR-281 acceptance tests for role-aware embedding APIs.
//!
//! These exercise the retrieval-boundary contract (ingestion => passage,
//! text-search => query), asymmetric providers that lack query support, the
//! "prefix applied exactly once" rule, and role preservation across
//! `Arc<dyn EmbeddingProvider>`. They use an in-process spy provider so no
//! network access or model download is required.
//!
//! Criteria that require pinned model artifacts (ADR-281 criteria 4, 5, 8) are
//! intentionally NOT covered here: `schemas/fixtures/` carries only
//! `embedding-space-identity-v1.json`, so faithful vector fixtures for MiniLM /
//! BGE / E5 / Qwen and the cross-language conformance corpus are not available.

use ruvector_core::embeddings::{
    apply_prompt_template, build_revision, default_space_identity, legacy_embed_call_count,
    registry_identity, EmbeddingDistanceMetric, EmbeddingProvider, EmbeddingRole,
    EmbeddingRolePolicy, EmbeddingSpaceIdentity, HashEmbedding, OutputDtype, PoolingStrategy,
    PoolingStrategyName, PrefixPolicy,
};
use ruvector_core::error::{Result, RuvectorError};
use ruvector_core::{
    types::{DbOptions, VectorEntry},
    AgenticDB, VectorDB,
};
use std::sync::{Arc, Mutex};
use tempfile::tempdir;

/// A deterministic, normalized embedding whose value depends on the exact
/// (prefixed) text it is handed, so query and passage sides of an asymmetric
/// model produce different vectors.
fn embed_text(text: &str, dims: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; dims];
    for (i, b) in text.as_bytes().iter().enumerate() {
        v[i % dims] += (*b as f32) / 255.0;
    }
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}

fn spy_identity(
    model: &str,
    dims: u32,
    role_policy: EmbeddingRolePolicy,
    prefix_policy: PrefixPolicy,
) -> EmbeddingSpaceIdentity {
    EmbeddingSpaceIdentity {
        schema_version: 1,
        provider: "spy".into(),
        model_id: model.into(),
        model_artifact_sha256: "aa".repeat(32),
        model_graph_sha256: "bb".repeat(32),
        tokenizer_sha256: "cc".repeat(32),
        prompt_template_sha256: "dd".repeat(32),
        pooling_strategy: PoolingStrategy::Named(PoolingStrategyName::Mean),
        normalize: true,
        truncation_tokens: 512,
        output_dimension: dims,
        output_dtype: OutputDtype::F32,
        runtime_revision: "spy@1".into(),
        distance_metric: EmbeddingDistanceMetric::Cosine,
        role_policy,
        prefix_policy,
        prefix_policy_version: 1,
    }
}

/// Records which role each call used and the exact prepared (prefixed) text the
/// provider itself produced, so tests can assert both call routing and that the
/// model-owned prefix is applied exactly once.
struct SpyProvider {
    identity: EmbeddingSpaceIdentity,
    dims: usize,
    name: String,
    support_query: bool,
    support_passage: bool,
    query_prefix: String,
    passage_prefix: String,
    query_texts: Mutex<Vec<String>>,
    passage_texts: Mutex<Vec<String>>,
}

impl SpyProvider {
    fn query_count(&self) -> usize {
        self.query_texts.lock().unwrap().len()
    }

    fn passage_count(&self) -> usize {
        self.passage_texts.lock().unwrap().len()
    }

    fn last_query_text(&self) -> Option<String> {
        self.query_texts.lock().unwrap().last().cloned()
    }

    fn query_texts(&self) -> Vec<String> {
        self.query_texts.lock().unwrap().clone()
    }
}

impl EmbeddingProvider for SpyProvider {
    fn embed_for(&self, role: EmbeddingRole, text: &str) -> Result<Vec<f32>> {
        match role {
            EmbeddingRole::Query if !self.support_query => {
                return Err(RuvectorError::EmbeddingRoleUnsupported(
                    "spy provider has no query support".into(),
                ));
            }
            EmbeddingRole::Passage if !self.support_passage => {
                return Err(RuvectorError::EmbeddingRoleUnsupported(
                    "spy provider has no passage support".into(),
                ));
            }
            _ => {}
        }

        let prepared = match role {
            EmbeddingRole::Query => format!("{}{}", self.query_prefix, text),
            EmbeddingRole::Passage => format!("{}{}", self.passage_prefix, text),
        };
        match role {
            EmbeddingRole::Query => self.query_texts.lock().unwrap().push(prepared.clone()),
            EmbeddingRole::Passage => self.passage_texts.lock().unwrap().push(prepared.clone()),
        }
        Ok(embed_text(&prepared, self.dims))
    }

    fn embedding_space(&self) -> &EmbeddingSpaceIdentity {
        &self.identity
    }

    fn dimensions(&self) -> usize {
        self.dims
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A spy that supports both roles and applies a distinct prefix per role.
fn asymmetric_spy(dims: usize) -> SpyProvider {
    SpyProvider {
        identity: spy_identity(
            "spy/asymmetric",
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        ),
        dims,
        name: "SpyProvider(asymmetric)".into(),
        support_query: true,
        support_passage: true,
        query_prefix: "query: ".into(),
        passage_prefix: "passage: ".into(),
        query_texts: Mutex::new(Vec::new()),
        passage_texts: Mutex::new(Vec::new()),
    }
}

/// A spy that mimics an asymmetric encoder shipped without a query transform.
fn query_unsupported_spy(dims: usize) -> SpyProvider {
    SpyProvider {
        identity: spy_identity(
            "spy/passage-only",
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        ),
        dims,
        name: "SpyProvider(passage-only)".into(),
        support_query: false,
        support_passage: true,
        query_prefix: "query: ".into(),
        passage_prefix: "passage: ".into(),
        query_texts: Mutex::new(Vec::new()),
        passage_texts: Mutex::new(Vec::new()),
    }
}

fn agentic_db_with(spy: Arc<SpyProvider>, dims: usize) -> (AgenticDB, tempfile::TempDir) {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = dims;
    let db = AgenticDB::with_embedding_provider(options, spy).unwrap();
    (db, dir)
}

// ADR-281 criterion 2: the same spy proves ingestion and re-index paths invoke
// embed_passage. AgenticDB has no separate re-index API; ingestion is the
// index-building path (each ingest writes the passage vector into the index),
// so covering every ingestion entry point covers the persisted-corpus side.
#[test]
fn ingestion_paths_use_passage_role_only() {
    let dims = 32;
    let spy = Arc::new(asymmetric_spy(dims));
    let (db, _dir) = agentic_db_with(spy.clone(), dims);

    db.store_episode(
        "task".into(),
        vec!["a".into()],
        vec!["o".into()],
        "critique text".into(),
    )
    .unwrap();
    db.create_skill(
        "Parse".into(),
        "parse json".into(),
        std::collections::HashMap::new(),
        vec!["json.parse()".into()],
    )
    .unwrap();
    db.add_causal_edge(
        vec!["rain".into()],
        vec!["wet".into()],
        0.9,
        "weather".into(),
    )
    .unwrap();
    db.session_index("s1", 3600)
        .add_turn(0, "user", "hello there")
        .unwrap();
    db.witness_log()
        .append("agent-1", "edit", "changed a file")
        .unwrap();

    assert_eq!(
        spy.passage_count(),
        5,
        "each ingestion entry point must embed exactly one passage vector"
    );
    assert_eq!(
        spy.query_count(),
        0,
        "ingestion must never invoke the query role"
    );
}

// ADR-281 criterion 3 (positive half): every text-search path invokes
// embed_query. Each search is isolated on its own DB so the query count is
// attributable to exactly one search entry point.
#[test]
fn search_paths_use_query_role() {
    let dims = 32;

    // retrieve_similar_episodes
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.store_episode("t".into(), vec![], vec![], "c".into())
            .unwrap();
        let before = spy.query_count();
        db.retrieve_similar_episodes("find me", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "retrieve_similar_episodes");
    }

    // search_skills
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.create_skill(
            "n".into(),
            "d".into(),
            std::collections::HashMap::new(),
            vec![],
        )
        .unwrap();
        let before = spy.query_count();
        db.search_skills("find skill", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "search_skills");
    }

    // query_with_utility
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        db.add_causal_edge(vec!["a".into()], vec!["b".into()], 0.5, "ctx".into())
            .unwrap();
        let before = spy.query_count();
        db.query_with_utility("utility query", 3, 0.7, 0.2, 0.1)
            .unwrap();
        assert_eq!(spy.query_count(), before + 1, "query_with_utility");
    }

    // find_relevant_turns
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        let session = db.session_index("s1", 3600);
        session.add_turn(0, "user", "hi").unwrap();
        let before = spy.query_count();
        session.find_relevant_turns("earlier context", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "find_relevant_turns");
    }

    // witness_log search
    {
        let spy = Arc::new(asymmetric_spy(dims));
        let (db, _dir) = agentic_db_with(spy.clone(), dims);
        let log = db.witness_log();
        log.append("a", "act", "details").unwrap();
        let before = spy.query_count();
        log.search("audit query", 3).unwrap();
        assert_eq!(spy.query_count(), before + 1, "witness_log::search");
    }
}

// ADR-281 criterion 3 (negative half): an asymmetric provider that lacks query
// support returns EmbeddingRoleUnsupported and never silently falls back to the
// passage side (or the deprecated `embed`).
#[test]
fn query_unsupported_provider_errors_without_falling_back() {
    let dims = 32;
    let spy = Arc::new(query_unsupported_spy(dims));

    // Direct provider surface: embed_query must error, not fall back.
    let err = spy.embed_query("hello").unwrap_err();
    assert!(
        matches!(err, RuvectorError::EmbeddingRoleUnsupported(_)),
        "expected EmbeddingRoleUnsupported, got {err:?}"
    );

    // Through a high-level search path.
    let (db, _dir) = agentic_db_with(spy.clone(), dims);
    db.store_episode("t".into(), vec![], vec![], "c".into())
        .unwrap();
    let passage_before = spy.passage_count();

    let result = db.retrieve_similar_episodes("cannot embed this", 3);
    let err = result.expect_err("search on a query-unsupported provider must fail");
    assert!(
        matches!(err, RuvectorError::EmbeddingRoleUnsupported(_)),
        "expected EmbeddingRoleUnsupported from search, got {err:?}"
    );
    assert_eq!(
        spy.passage_count(),
        passage_before,
        "a failed query must not fall back to the passage role"
    );
}

// ADR-281 criterion 6: prefixes are applied exactly once in single and batch
// APIs. The spy owns prefix application; high-level callers pass raw text.
#[test]
fn prefix_applied_exactly_once_single_and_batch() {
    let dims = 32;
    let spy = asymmetric_spy(dims);

    // Single API.
    spy.embed_query("cat").unwrap();
    spy.embed_passage("dog").unwrap();
    assert_eq!(spy.last_query_text().as_deref(), Some("query: cat"));
    assert_eq!(
        spy.query_texts()[0].matches("query: ").count(),
        1,
        "single embed_query must apply the query prefix exactly once"
    );

    // Batch API must not strip the role nor double-apply the prefix.
    let batch = spy.embed_query_batch(&["one", "two"]).unwrap();
    assert_eq!(batch.len(), 2);
    for prepared in spy.query_texts().iter().skip(1) {
        assert_eq!(
            prepared.matches("query: ").count(),
            1,
            "batch embed_query must apply the query prefix exactly once per item, got {prepared:?}"
        );
    }

    // The high-level DB path must also apply the prefix exactly once (i.e. the
    // DB passes raw text and never pre-prefixes).
    let spy = Arc::new(asymmetric_spy(dims));
    let (db, _dir) = agentic_db_with(spy.clone(), dims);
    db.create_skill(
        "n".into(),
        "d".into(),
        std::collections::HashMap::new(),
        vec![],
    )
    .unwrap();
    db.search_skills("kittens", 3).unwrap();
    assert_eq!(
        spy.last_query_text().as_deref(),
        Some("query: kittens"),
        "AgenticDB must forward raw query text so the provider prefixes exactly once"
    );
}

// ADR-281 criterion 7: trait-object use preserves both roles. An asymmetric
// provider behind Arc<dyn EmbeddingProvider> keeps query and passage distinct,
// and the trait-object results match the concrete-provider results, in both the
// single and batch APIs.
#[test]
fn trait_object_preserves_both_roles_for_asymmetric_provider() {
    let dims = 32;
    let concrete = asymmetric_spy(dims);

    // Reference results computed directly (no trait object).
    let ref_query = embed_text("query: same text", dims);
    let ref_passage = embed_text("passage: same text", dims);
    assert_ne!(
        ref_query, ref_passage,
        "sanity: asymmetric fixture must produce distinct role vectors"
    );

    let provider: Arc<dyn EmbeddingProvider> = Arc::new(concrete);

    let q = provider.embed_query("same text").unwrap();
    let p = provider.embed_passage("same text").unwrap();
    assert_ne!(
        q, p,
        "trait object collapsed query and passage into the same vector"
    );
    assert_eq!(
        q, ref_query,
        "trait-object query role diverged from concrete"
    );
    assert_eq!(
        p, ref_passage,
        "trait-object passage role diverged from concrete"
    );

    // Batch surface preserves the role through the trait object as well.
    let legacy_before = legacy_embed_call_count();
    let qb = provider.embed_query_batch(&["same text"]).unwrap();
    let pb = provider.embed_passage_batch(&["same text"]).unwrap();
    assert_eq!(qb[0], ref_query);
    assert_eq!(pb[0], ref_passage);
    assert_ne!(qb[0], pb[0]);
    assert_eq!(
        legacy_embed_call_count(),
        legacy_before,
        "role-explicit trait-object calls must not route through deprecated embed"
    );
}

// ADR-281 §4: the prefix the embedding path applies comes from the registered
// identity's templates, so what is embedded matches what
// `prompt_template_sha256` attests. Covers two models whose templates differ.
#[test]
fn applied_prefix_matches_the_identitys_registered_template() {
    let e5 = registry_identity("test", "intfloat/e5-base-v2", 768).unwrap();
    assert_eq!(
        apply_prompt_template(&e5, EmbeddingRole::Query, "how tall is it").unwrap(),
        "query: how tall is it"
    );
    assert_eq!(
        apply_prompt_template(&e5, EmbeddingRole::Passage, "it is tall").unwrap(),
        "passage: it is tall"
    );

    // A different model with a different template bundle: the query side is an
    // instruction and the passage side is bare, so the hardcoded E5 pair would
    // be wrong on both sides.
    let qwen = registry_identity("test", "Qwen/Qwen3-Embedding-0.6B", 1024).unwrap();
    let query = apply_prompt_template(&qwen, EmbeddingRole::Query, "how tall is it").unwrap();
    assert!(
        query.starts_with("Instruct: ") && query.ends_with("Query:how tall is it"),
        "qwen query template not applied: {query:?}"
    );
    assert_eq!(
        apply_prompt_template(&qwen, EmbeddingRole::Passage, "it is tall").unwrap(),
        "it is tall"
    );

    assert_ne!(
        e5.prompt_template_sha256, qwen.prompt_template_sha256,
        "different templates must hash differently"
    );

    // A symmetric BGE-family control: query carries the retrieval instruction,
    // passage is raw.
    let bge = registry_identity("test", "BAAI/bge-small-en-v1.5", 384).unwrap();
    assert_eq!(
        apply_prompt_template(&bge, EmbeddingRole::Query, "cats").unwrap(),
        "Represent this sentence for searching relevant passages: cats"
    );
    assert_eq!(
        apply_prompt_template(&bge, EmbeddingRole::Passage, "cats").unwrap(),
        "cats"
    );
}

// A prefix policy that promises a transform must never silently degrade to no
// prefix: Custom has no registered template, and a template hash that does not
// match the registry means the attested identity does not describe what would
// be applied.
#[test]
fn unverifiable_prefix_policies_fail_closed() {
    let mut custom = registry_identity("test", "intfloat/e5-base-v2", 768).unwrap();
    custom.prefix_policy = PrefixPolicy::Custom;
    assert!(
        apply_prompt_template(&custom, EmbeddingRole::Query, "text").is_err(),
        "PrefixPolicy::Custom must fail closed rather than embed unprefixed text"
    );

    let mut tampered = registry_identity("test", "intfloat/e5-base-v2", 768).unwrap();
    tampered.prompt_template_sha256 = "ee".repeat(32);
    assert!(
        apply_prompt_template(&tampered, EmbeddingRole::Query, "text").is_err(),
        "a template hash that disagrees with the registry must fail closed"
    );

    // PrefixPolicy::None is the one policy that legitimately embeds raw text.
    let none = default_space_identity("test", "symmetric-model", 8);
    assert_eq!(
        apply_prompt_template(&none, EmbeddingRole::Query, "text").unwrap(),
        "text"
    );
}

// ADR-281 §5 / finding: models the providers advertise must be registered with
// the dimensions and templates from their model cards, not hard-error.
#[test]
fn registry_covers_advertised_models() {
    for (model_id, dims, prefixed) in [
        ("sentence-transformers/all-mpnet-base-v2", 768, false),
        ("intfloat/multilingual-e5-large", 1024, true),
        (
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            384,
            false,
        ),
    ] {
        let identity = registry_identity("test", model_id, dims)
            .unwrap_or_else(|e| panic!("{model_id} must be registered: {e}"));
        assert_eq!(identity.output_dimension, dims as u32);
        let query = apply_prompt_template(&identity, EmbeddingRole::Query, "text").unwrap();
        if prefixed {
            assert_eq!(query, "query: text", "{model_id}");
            assert_eq!(
                apply_prompt_template(&identity, EmbeddingRole::Passage, "text").unwrap(),
                "passage: text",
                "{model_id}"
            );
        } else {
            assert_eq!(query, "text", "{model_id} must not be prefixed");
            assert_eq!(identity.prefix_policy, PrefixPolicy::None, "{model_id}");
        }

        // The registry is keyed by exact artifact revision: the right model at
        // the wrong dimensionality is not a match.
        assert!(
            registry_identity("test", model_id, dims + 1).is_err(),
            "{model_id} must not accept a dimension it is not registered at"
        );
    }
}

// ADR-281 §7 / criterion 10: a corpus refuses a provider from a different
// embedding space even when the dimensions agree.
#[test]
fn reopening_a_corpus_with_a_different_same_dimension_model_is_rejected() {
    let dims = 32;
    let dir = tempdir().unwrap();
    let storage_path = dir.path().join("space.db").to_string_lossy().to_string();
    let options = || {
        let mut o = DbOptions::default();
        o.storage_path = storage_path.clone();
        o.dimensions = dims;
        o
    };
    let spy = |model: &str| {
        let mut provider = asymmetric_spy(dims);
        provider.identity = spy_identity(
            model,
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        );
        Arc::new(provider)
    };

    {
        let db = AgenticDB::with_embedding_provider(options(), spy("spy/model-a")).unwrap();
        db.store_episode("t".into(), vec![], vec![], "c".into())
            .unwrap();
    }

    // Same identity reopens cleanly.
    {
        AgenticDB::with_embedding_provider(options(), spy("spy/model-a"))
            .expect("reopening under the same embedding space must succeed");
    }

    // Same dimensions, different model identity: rejected.
    let err = match AgenticDB::with_embedding_provider(options(), spy("spy/model-b")) {
        Ok(_) => panic!("a different embedding space must not be able to mutate this corpus"),
        Err(e) => e,
    };
    assert!(
        matches!(err, RuvectorError::EmbeddingSpaceMismatch { .. }),
        "expected EmbeddingSpaceMismatch, got {err:?}"
    );
}

const EMBEDDING_SPACE_KEY: &str = "__ruvector_embedding_space__";

/// Delete the whole `.agentic` sidecar file, exactly as an operator (or an
/// attacker) could. The vectors and the identity copy inside the vector store
/// survive.
fn delete_sidecar_file(storage_path: &str) {
    std::fs::remove_file(format!("{storage_path}.agentic")).expect("sidecar must exist");
}

/// Read the sidecar mirror, if present.
fn sidecar_identity(storage_path: &str) -> Option<Vec<u8>> {
    let db = redb::Database::create(format!("{storage_path}.agentic")).unwrap();
    let txn = db.begin_read().unwrap();
    let table = txn
        .open_table(redb::TableDefinition::<&str, &[u8]>::new(
            "embedding_space_identity",
        ))
        .unwrap();
    let found = table
        .get(EMBEDDING_SPACE_KEY)
        .unwrap()
        .map(|v| v.value().to_vec());
    found
}

// The compatibility gate must not be removable on its own. The identity's
// authoritative copy lives in the vector store, so deleting the `.agentic`
// sidecar destroys the episodes but cannot silently re-open the corpus to a
// different embedding space.
#[test]
fn deleting_the_sidecar_does_not_downgrade_the_identity_check() {
    let dims = 32;
    let dir = tempdir().unwrap();
    let storage_path = dir.path().join("gated.db").to_string_lossy().to_string();
    let options = || {
        let mut o = DbOptions::default();
        o.storage_path = storage_path.clone();
        o.dimensions = dims;
        o
    };
    let spy = |model: &str| {
        let mut provider = asymmetric_spy(dims);
        provider.identity = spy_identity(
            model,
            dims as u32,
            EmbeddingRolePolicy::Asymmetric,
            PrefixPolicy::Required,
        );
        Arc::new(provider)
    };

    {
        let db = AgenticDB::with_embedding_provider(options(), spy("spy/model-a")).unwrap();
        db.store_episode("t".into(), vec![], vec![], "c".into())
            .unwrap();
    }

    delete_sidecar_file(&storage_path);

    let err = match AgenticDB::with_embedding_provider(options(), spy("spy/model-b")) {
        Ok(_) => panic!(
            "deleting the sidecar downgraded the embedding-space check — a different model \
             was allowed to mutate this corpus"
        ),
        Err(e) => e,
    };
    assert!(
        matches!(err, RuvectorError::EmbeddingSpaceMismatch { .. }),
        "expected EmbeddingSpaceMismatch, got {err:?}"
    );

    // The original space still opens, and that reopen rewrites the sidecar
    // mirror from the surviving primary copy.
    AgenticDB::with_embedding_provider(options(), spy("spy/model-a"))
        .expect("the original embedding space must still open");
    assert!(
        sidecar_identity(&storage_path).is_some(),
        "reopening must heal the deleted sidecar mirror"
    );
}

// Pre-ADR-281 corpora carry no stored identity. They must stay usable (adopt
// the active identity) rather than being bricked — and because the vectors are
// the thing at risk, "does this corpus already hold data?" has to look at the
// vector store, not only at the agentic sidecar tables.
#[test]
fn corpus_without_a_stored_identity_is_adopted() {
    let dims = 32;
    let dir = tempdir().unwrap();
    let storage_path = dir.path().join("legacy.db").to_string_lossy().to_string();
    let mut options = DbOptions::default();
    options.storage_path = storage_path.clone();
    options.dimensions = dims;

    // Written by a build that predates ADR-281: vectors, and no identity in
    // either location. The sidecar tables are empty, so only the vector store
    // can reveal that this corpus is populated.
    {
        let raw = VectorDB::new(options.clone()).unwrap();
        raw.insert(VectorEntry {
            id: Some("v1".into()),
            vector: vec![0.1; dims],
            metadata: None,
        })
        .unwrap();
        assert!(!raw.is_empty().unwrap());
    }

    let db = AgenticDB::with_embedding_provider(options.clone(), Arc::new(asymmetric_spy(dims)))
        .expect("a corpus with no recorded identity must remain usable");
    assert!(
        db.get("v1").unwrap().is_some(),
        "the vectors that triggered the adopt-with-warning path must still be there"
    );
    drop(db);

    // Having adopted, the corpus is gated again from here on.
    let mut other = asymmetric_spy(dims);
    other.identity = spy_identity(
        "spy/other",
        dims as u32,
        EmbeddingRolePolicy::Asymmetric,
        PrefixPolicy::Required,
    );
    assert!(
        matches!(
            AgenticDB::with_embedding_provider(options, Arc::new(other)),
            Err(RuvectorError::EmbeddingSpaceMismatch { .. })
        ),
        "the adopted identity must gate subsequent opens"
    );
}

// Smoke test for the identity-free public entry points in this crate: they
// must construct, not fail closed the way the first ADR-281 pass did.
#[test]
fn identity_free_entry_points_construct() {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("smoke.db").to_string_lossy().to_string();
    options.dimensions = 16;
    AgenticDB::new(options).expect("AgenticDB::new must construct with default embeddings");

    let provider = HashEmbedding::new(16);
    assert_eq!(provider.embed_query("x").unwrap().len(), 16);
    assert_eq!(
        provider.embedding_space().output_dimension,
        16,
        "HashEmbedding must carry a usable default identity"
    );
}

// ---------------------------------------------------------------------------
// embedding_space_id stability (ADR-281 §7)
// ---------------------------------------------------------------------------

// The id gates corpus reopen and cache reuse, so it must be a function of the
// embedding function alone. A crate release changes no vectors and must change
// no id: when 2.2.3 -> 2.3.0 last happened, an id that hashed the crate version
// would have made every persisted corpus unreadable and every cache entry cold.
#[test]
fn embedding_space_id_is_blind_to_the_crate_build_version() {
    let identity = registry_identity("ruvector-core", "BAAI/bge-small-en-v1.5", 384).unwrap();
    let canonical = identity.canonical_json().unwrap();

    assert!(
        !canonical.contains(build_revision()),
        "the crate build version {:?} reached the hashed identity: {canonical}",
        build_revision()
    );
    assert!(
        !identity.runtime_revision.contains(build_revision()),
        "runtime_revision must not carry the crate version, got {:?}",
        identity.runtime_revision
    );

    // Simulate the version bump directly: an identity is only allowed to
    // change id when an embedding-semantics input changes, so substituting any
    // plausible crate version into the struct must be a no-op by construction.
    for fake_crate_version in ["2.2.3", "2.3.0", "99.0.0-rc.1"] {
        let mut bumped = identity.clone();
        bumped.runtime_revision = bumped
            .runtime_revision
            .replace(build_revision(), fake_crate_version);
        assert_eq!(
            bumped.embedding_space_id().unwrap(),
            identity.embedding_space_id().unwrap(),
            "id moved when the crate version was rewritten to {fake_crate_version}"
        );
    }

    // Control: an input that genuinely changes the embedding function must
    // still change the id, so the test above is not passing vacuously.
    let mut retemplated = identity.clone();
    retemplated.prompt_template_sha256 = "ab".repeat(32);
    assert_ne!(
        retemplated.embedding_space_id().unwrap(),
        identity.embedding_space_id().unwrap(),
        "changing the prompt templates must change the embedding-space id"
    );
}

// Golden id for a registered model. This is the value persisted corpora carry,
// so it is a compatibility constant, not an implementation detail: if a change
// moves it, every existing AgenticDB corpus stops opening and every cluster
// cache entry is invalidated. A failure here means either a deliberate
// EMBEDDING_SPACE_FORMAT_REVISION bump (update this constant and plan the
// re-embed) or an accidental identity drift (revert it).
#[test]
fn embedding_space_id_golden_value_for_minilm() {
    let identity = registry_identity(
        "ruvector-core",
        "sentence-transformers/all-MiniLM-L6-v2",
        384,
    )
    .unwrap();
    assert_eq!(
        identity.embedding_space_id().unwrap(),
        "4ce23f439462151f8546518f8d5fce92359b0996b2431c979197db61271ee48a",
        "embedding-space id for all-MiniLM-L6-v2 drifted"
    );
}
