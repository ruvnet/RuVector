//! Integration tests for embedding providers

use ruvector_core::embeddings::{
    registry_identity, ApiEmbedding, EmbeddingDistanceMetric, EmbeddingProvider,
    EmbeddingRolePolicy, EmbeddingSpaceIdentity, HashEmbedding, OutputDtype, PoolingStrategy,
    PoolingStrategyName, PrefixPolicy,
};
use ruvector_core::error::RuvectorError;
use ruvector_core::{types::DbOptions, AgenticDB};
use std::sync::Arc;
use tempfile::tempdir;

fn remote_identity(model: &str, dimension: u32) -> EmbeddingSpaceIdentity {
    EmbeddingSpaceIdentity {
        schema_version: 1,
        provider: "test-remote".into(),
        model_id: model.into(),
        model_artifact_sha256: "11".repeat(32),
        model_graph_sha256: "22".repeat(32),
        tokenizer_sha256: "33".repeat(32),
        prompt_template_sha256: "44".repeat(32),
        pooling_strategy: PoolingStrategy::Named(PoolingStrategyName::Mean),
        normalize: true,
        truncation_tokens: 8192,
        output_dimension: dimension,
        output_dtype: OutputDtype::F32,
        runtime_revision: "test-api@1".into(),
        distance_metric: EmbeddingDistanceMetric::Cosine,
        role_policy: EmbeddingRolePolicy::Symmetric,
        prefix_policy: PrefixPolicy::None,
        prefix_policy_version: 1,
    }
}

#[test]
fn test_hash_embedding_provider() {
    let provider = HashEmbedding::new(128);

    // Test basic embedding
    let emb1 = provider.embed("hello world").unwrap();
    assert_eq!(emb1.len(), 128);

    // Test consistency
    let emb2 = provider.embed("hello world").unwrap();
    assert_eq!(emb1, emb2, "Same text should produce same embedding");

    // Test different text produces different embeddings
    let emb3 = provider.embed("goodbye world").unwrap();
    assert_ne!(
        emb1, emb3,
        "Different text should produce different embeddings"
    );

    // Test normalization
    let norm: f32 = emb1.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-5,
        "Embedding should be normalized to unit length"
    );

    // Test provider info
    assert_eq!(provider.dimensions(), 128);
    assert!(provider.name().contains("Hash"));
}

#[test]
fn test_agenticdb_with_hash_embeddings() {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = 128;

    // Create AgenticDB with default hash embeddings
    let db = AgenticDB::new(options).unwrap();

    assert_eq!(db.embedding_provider_name(), "HashEmbedding (placeholder)");

    // Test storing a reflexion episode
    let episode_id = db
        .store_episode(
            "Solve a math problem".to_string(),
            vec!["read problem".to_string(), "calculate".to_string()],
            vec!["got answer 42".to_string()],
            "Should have shown intermediate steps".to_string(),
        )
        .unwrap();

    // Test retrieving similar episodes
    let episodes = db
        .retrieve_similar_episodes("math problem solving", 5)
        .unwrap();
    assert!(!episodes.is_empty());
    assert_eq!(episodes[0].id, episode_id);
}

#[test]
fn test_agenticdb_with_custom_hash_provider() {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = 256;

    // Create custom hash provider
    let provider = Arc::new(HashEmbedding::new(256));

    // Create AgenticDB with custom provider
    let db = AgenticDB::with_embedding_provider(options, provider).unwrap();

    assert_eq!(db.embedding_provider_name(), "HashEmbedding (placeholder)");

    // Test creating a skill
    let mut params = std::collections::HashMap::new();
    params.insert("input".to_string(), "string".to_string());

    let skill_id = db
        .create_skill(
            "Parse JSON".to_string(),
            "Parse JSON from string".to_string(),
            params,
            vec!["json.parse()".to_string()],
        )
        .unwrap();

    // Search for skills
    let skills = db.search_skills("parse json data", 5).unwrap();
    assert!(!skills.is_empty());
    assert_eq!(skills[0].id, skill_id);
}

#[test]
fn test_dimension_mismatch_validation() {
    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = 128;

    // Try to create with mismatched dimensions
    let provider = Arc::new(HashEmbedding::new(256)); // Different from options

    let result = AgenticDB::with_embedding_provider(options, provider);
    assert!(result.is_err(), "Should fail when dimensions don't match");

    if let Err(err) = result {
        assert!(
            err.to_string().contains("do not match"),
            "Error should mention dimension mismatch"
        );
    }
}

#[test]
fn test_api_embedding_provider_construction() {
    // Test OpenAI provider construction
    let openai_small = ApiEmbedding::openai(
        "sk-test",
        "text-embedding-3-small",
        remote_identity("text-embedding-3-small", 1536),
    )
    .unwrap();
    assert_eq!(openai_small.dimensions(), 1536);
    assert_eq!(openai_small.name(), "ApiEmbedding");

    let openai_large = ApiEmbedding::openai(
        "sk-test",
        "text-embedding-3-large",
        remote_identity("text-embedding-3-large", 3072),
    )
    .unwrap();
    assert_eq!(openai_large.dimensions(), 3072);

    // Test Cohere provider construction
    let cohere = ApiEmbedding::cohere(
        "co-test",
        "embed-english-v3.0",
        remote_identity("embed-english-v3.0", 1024),
    )
    .unwrap();
    assert_eq!(cohere.dimensions(), 1024);

    // Test Voyage provider construction
    let voyage =
        ApiEmbedding::voyage("vo-test", "voyage-2", remote_identity("voyage-2", 1024)).unwrap();
    assert_eq!(voyage.dimensions(), 1024);

    let voyage_large = ApiEmbedding::voyage(
        "vo-test",
        "voyage-large-2",
        remote_identity("voyage-large-2", 1536),
    )
    .unwrap();
    assert_eq!(voyage_large.dimensions(), 1536);
}

#[test]
#[ignore] // Requires API key and network access
fn test_api_embedding_openai() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable required for this test");

    let provider = ApiEmbedding::openai(
        &api_key,
        "text-embedding-3-small",
        remote_identity("text-embedding-3-small", 1536),
    )
    .unwrap();

    let embedding = provider.embed("hello world").unwrap();
    assert_eq!(embedding.len(), 1536);

    // Check that embeddings are different for different texts
    let embedding2 = provider.embed("goodbye world").unwrap();
    assert_ne!(embedding, embedding2);
}

#[test]
#[ignore] // Requires API key and network access
fn test_agenticdb_with_openai_embeddings() {
    let api_key = std::env::var("OPENAI_API_KEY")
        .expect("OPENAI_API_KEY environment variable required for this test");

    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = 1536; // OpenAI text-embedding-3-small dimensions

    let provider = Arc::new(
        ApiEmbedding::openai(
            &api_key,
            "text-embedding-3-small",
            remote_identity("text-embedding-3-small", 1536),
        )
        .unwrap(),
    );
    let db = AgenticDB::with_embedding_provider(options, provider).unwrap();

    assert_eq!(db.embedding_provider_name(), "ApiEmbedding");

    // Test with real semantic embeddings
    let _episode1_id = db
        .store_episode(
            "Solve calculus problem".to_string(),
            vec![
                "identify function".to_string(),
                "take derivative".to_string(),
            ],
            vec!["computed derivative".to_string()],
            "Should explain chain rule application".to_string(),
        )
        .unwrap();

    let _episode2_id = db
        .store_episode(
            "Solve algebra problem".to_string(),
            vec!["simplify equation".to_string(), "solve for x".to_string()],
            vec!["found x = 5".to_string()],
            "Should show all steps".to_string(),
        )
        .unwrap();

    // Search with semantic query - should find calculus episode first
    let episodes = db
        .retrieve_similar_episodes("derivative calculation", 2)
        .unwrap();
    assert!(!episodes.is_empty());

    // With real embeddings, "derivative" should match calculus better than algebra
    println!(
        "Found episodes: {:?}",
        episodes.iter().map(|e| &e.task).collect::<Vec<_>>()
    );
}

#[cfg(feature = "real-embeddings")]
#[test]
#[ignore] // Requires model download
fn test_candle_embedding_provider() {
    use ruvector_core::CandleEmbedding;

    let provider =
        CandleEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", false).unwrap();

    assert_eq!(provider.dimensions(), 384);
    assert_eq!(provider.name(), "CandleEmbedding (transformer)");

    let embedding = provider.embed("hello world").unwrap();
    assert_eq!(embedding.len(), 384);

    // Check normalization
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-3, "Embedding should be normalized");

    // Test semantic similarity
    let emb_dog = provider.embed("dog").unwrap();
    let emb_cat = provider.embed("cat").unwrap();
    let emb_car = provider.embed("car").unwrap();

    // Cosine similarity
    let similarity_dog_cat: f32 = emb_dog.iter().zip(emb_cat.iter()).map(|(a, b)| a * b).sum();

    let similarity_dog_car: f32 = emb_dog.iter().zip(emb_car.iter()).map(|(a, b)| a * b).sum();

    // "dog" and "cat" should be more similar than "dog" and "car"
    assert!(
        similarity_dog_cat > similarity_dog_car,
        "Semantic embeddings should show dog-cat more similar than dog-car"
    );
}

#[cfg(feature = "real-embeddings")]
#[test]
#[ignore] // Requires model download
fn test_agenticdb_with_candle_embeddings() {
    use ruvector_core::CandleEmbedding;

    let dir = tempdir().unwrap();
    let mut options = DbOptions::default();
    options.storage_path = dir.path().join("test.db").to_string_lossy().to_string();
    options.dimensions = 384;

    let provider = Arc::new(
        CandleEmbedding::from_pretrained("sentence-transformers/all-MiniLM-L6-v2", false).unwrap(),
    );

    let db = AgenticDB::with_embedding_provider(options, provider).unwrap();

    assert_eq!(
        db.embedding_provider_name(),
        "CandleEmbedding (transformer)"
    );

    // Test with real semantic embeddings
    let skill1_id = db
        .create_skill(
            "File I/O".to_string(),
            "Read and write files to disk".to_string(),
            std::collections::HashMap::new(),
            vec![
                "open()".to_string(),
                "read()".to_string(),
                "write()".to_string(),
            ],
        )
        .unwrap();

    let skill2_id = db
        .create_skill(
            "Network I/O".to_string(),
            "Send and receive data over network".to_string(),
            std::collections::HashMap::new(),
            vec![
                "connect()".to_string(),
                "send()".to_string(),
                "recv()".to_string(),
            ],
        )
        .unwrap();

    // Search with semantic query
    let skills = db.search_skills("reading files from storage", 2).unwrap();
    assert!(!skills.is_empty());

    // With real embeddings, file I/O should match better
    println!(
        "Found skills: {:?}",
        skills.iter().map(|s| &s.name).collect::<Vec<_>>()
    );
}

// ---------------------------------------------------------------------------
// ADR-281 §4 for the remote provider (mirrors the ONNX path's contract)
// ---------------------------------------------------------------------------

/// A prefix policy the provider cannot honor must be refused at construction,
/// not discovered on the first embed, and never silently degrade to raw text.
#[test]
fn api_embedding_rejects_unhonorable_prefix_policies_at_construction() {
    let mut custom = registry_identity("test-remote", "intfloat/e5-base-v2", 768).unwrap();
    custom.prefix_policy = PrefixPolicy::Custom;
    let err = ApiEmbedding::new(
        "sk-test".into(),
        "https://example.invalid/v1/embeddings".into(),
        "intfloat/e5-base-v2".into(),
        768,
        custom,
    )
    .err()
    .expect("PrefixPolicy::Custom has no template ruvector can apply");
    assert!(
        matches!(err, RuvectorError::InvalidParameter(_)),
        "expected InvalidParameter, got {err:?}"
    );

    let mut tampered = registry_identity("test-remote", "intfloat/e5-base-v2", 768).unwrap();
    tampered.prompt_template_sha256 = "ee".repeat(32);
    assert!(
        ApiEmbedding::new(
            "sk-test".into(),
            "https://example.invalid/v1/embeddings".into(),
            "intfloat/e5-base-v2".into(),
            768,
            tampered,
        )
        .is_err(),
        "a prompt hash that disagrees with the registry must fail closed"
    );

    // A registry-consistent asymmetric identity constructs normally.
    ApiEmbedding::new(
        "sk-test".into(),
        "https://example.invalid/v1/embeddings".into(),
        "intfloat/e5-base-v2".into(),
        768,
        registry_identity("test-remote", "intfloat/e5-base-v2", 768).unwrap(),
    )
    .expect("a model whose templates ruvector knows must construct");
}

/// One-shot HTTP server: captures the request body of a single request and
/// answers with an OpenAI-shaped response. Returns `(endpoint, join handle)`.
fn capture_one_request() -> (String, std::thread::JoinHandle<String>) {
    use std::io::{Read, Write};
    let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
    let endpoint = format!("http://{}/v1/embeddings", listener.local_addr().unwrap());
    let handle = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("accept");
        stream
            .set_read_timeout(Some(std::time::Duration::from_secs(5)))
            .ok();
        let mut raw = Vec::new();
        let mut buf = [0u8; 1024];
        // Read headers, then exactly Content-Length body bytes.
        let body = loop {
            let n = match stream.read(&mut buf) {
                Ok(0) | Err(_) => break String::new(),
                Ok(n) => n,
            };
            raw.extend_from_slice(&buf[..n]);
            let text = String::from_utf8_lossy(&raw).into_owned();
            let Some(split) = text.find("\r\n\r\n") else {
                continue;
            };
            let content_length: usize = text
                .lines()
                .find_map(|line| {
                    line.strip_prefix("content-length: ")
                        .or_else(|| line.strip_prefix("Content-Length: "))
                })
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(0);
            let body = &text[split + 4..];
            if body.len() >= content_length {
                break body.to_string();
            }
        };
        let response_body = r#"{"data":[{"embedding":[0.0,1.0,0.0]}]}"#;
        let _ = write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{}",
            response_body.len(),
            response_body
        );
        let _ = stream.flush();
        body
    });
    (endpoint, handle)
}

/// The remote provider must send the text the identity attests: the registry
/// template applied exactly once, by the provider, from raw caller text.
#[test]
fn api_embedding_sends_the_identitys_template_not_raw_text() {
    let (endpoint, server) = capture_one_request();
    let provider = ApiEmbedding::new(
        "sk-test".into(),
        endpoint,
        "intfloat/e5-base-v2".into(),
        768,
        registry_identity("test-remote", "intfloat/e5-base-v2", 768).unwrap(),
    )
    .unwrap();

    provider
        .embed_query("how tall is it")
        .expect("the stub endpoint answers with a valid embedding");

    let body = server.join().expect("server thread");
    assert!(
        body.contains("query: how tall is it"),
        "the E5 query template must reach the wire, got body: {body}"
    );
    assert_eq!(
        body.matches("query: ").count(),
        1,
        "the prefix must be applied exactly once, got body: {body}"
    );
}
