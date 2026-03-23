//! REST API routes for the brain server

use crate::auth::AuthenticatedContributor;
use crate::graph::cosine_similarity;
use crate::types::{
    AddEvidenceRequest, AppState, BatchInjectRequest, BatchInjectResponse, BetaParams,
    BrainMemory, ChallengeResponse, ConsensusLoraWeights, CreatePageRequest, DriftQuery,
    DriftReport, FeedConfig, HealthResponse, InjectRequest, InjectResponse,
    ListPagesResponse, ListQuery, ListResponse, ListSort, LoraLatestResponse, LoraSubmission,
    LoraSubmitResponse, OptimizeActionResult, OptimizeRequest, OptimizeResponse,
    PageDelta, PageDetailResponse, PageResponse, PageStatus, PageSummary,
    PartitionQuery, PartitionResult, PartitionResultCompact, PipelineMetricsResponse,
    PubSubPushMessage, PublishNodeRequest, ScoredBrainMemory, SearchQuery,
    ShareRequest, ShareResponse,
    StatusResponse, SubmitDeltaRequest, TemporalResponse, TrainingCycleResult,
    TrainingPreferencesResponse,
    TrainingQuery, TransferRequest, TransferResponse, VerifyRequest, VerifyResponse,
    VoteDirection, VoteRequest, WasmNode, WasmNodeSummary,
};
use axum::{
    extract::{Path, Query, State},
    http::{HeaderMap, StatusCode},
    response::sse::{Event, KeepAlive, Sse},
    routing::{delete, get, post},
    Json, Router,
};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

/// Extract client IP from X-Forwarded-For (Cloud Run) or ConnectInfo fallback.
fn extract_client_ip(headers: &HeaderMap) -> String {
    headers
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.split(',').next())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Create the router with all routes. Returns (Router, AppState) so callers
/// can spawn background tasks with access to shared state.
pub async fn create_router() -> (Router, AppState) {
    let store = Arc::new(crate::store::FirestoreClient::new());
    // Hydrate cache from Firestore on startup (no-op if FIRESTORE_URL not set)
    store.load_from_firestore().await;
    let gcs = Arc::new(crate::gcs::GcsClient::new());
    let graph = Arc::new(parking_lot::RwLock::new(crate::graph::KnowledgeGraph::new()));
    let rate_limiter = Arc::new(crate::rate_limit::RateLimiter::default_limits());
    let ranking = Arc::new(parking_lot::RwLock::new(crate::ranking::RankingEngine::new(128)));
    let cognitive = Arc::new(parking_lot::RwLock::new(crate::cognitive::CognitiveEngine::new(128)));
    let drift = Arc::new(parking_lot::RwLock::new(crate::drift::DriftMonitor::new()));
    let aggregator = Arc::new(crate::aggregate::ByzantineAggregator::new());
    let domain_engine = Arc::new(parking_lot::RwLock::new(
        ruvector_domain_expansion::DomainExpansionEngine::new(),
    ));
    let sona = Arc::new(parking_lot::RwLock::new(sona::SonaEngine::new(128)));

    let lora_federation = Arc::new(parking_lot::RwLock::new(
        crate::types::LoraFederationStore::new(2, 128),
    ));

    // RuvLLM embedding engine — hydrate corpus from existing memories
    let mut emb_engine = crate::embeddings::EmbeddingEngine::new();
    let mut all_mems = store.all_memories();
    for mem in &all_mems {
        if mem.embedding.len() == crate::embeddings::EMBED_DIM {
            emb_engine.add_to_corpus(&mem.id.to_string(), mem.embedding.clone(), None);
        }
    }
    tracing::info!("Embedding engine: {} corpus entries, active={}", emb_engine.corpus_size(), emb_engine.engine_name());

    // If RLM is now active, re-embed all memories for embedding space consistency.
    // Stored embeddings may have been generated with HashEmbedder; re-embedding ensures
    // query (QueryConditioned) and stored (CorpusConditioned) embeddings are in the same space.
    if emb_engine.is_rlm_active() {
        tracing::info!("RLM active — re-embedding {} memories for space consistency", all_mems.len());
        // Build a fresh engine with clean corpus to avoid duplicate entries
        let mut fresh_engine = crate::embeddings::EmbeddingEngine::new();
        // First pass: seed fresh corpus with original hash embeddings
        for mem in &all_mems {
            if mem.embedding.len() == crate::embeddings::EMBED_DIM {
                fresh_engine.add_to_corpus(&mem.id.to_string(), mem.embedding.clone(), None);
            }
        }
        // Second pass: re-embed using RLM and replace in corpus
        for mem in &mut all_mems {
            let text = crate::embeddings::EmbeddingEngine::prepare_text(
                &mem.title, &mem.content, &mem.tags,
            );
            let new_emb = fresh_engine.embed_for_storage(&text);
            if new_emb.len() == crate::embeddings::EMBED_DIM {
                mem.embedding = new_emb;
            }
        }
        // Build final engine with only RLM embeddings (no duplicates)
        emb_engine = crate::embeddings::EmbeddingEngine::new();
        for mem in &all_mems {
            if mem.embedding.len() == crate::embeddings::EMBED_DIM {
                emb_engine.add_to_corpus(&mem.id.to_string(), mem.embedding.clone(), None);
            }
        }
        // Update in-memory store with re-embedded vectors
        for mem in &all_mems {
            store.update_embedding(&mem.id, &mem.embedding);
        }
        tracing::info!("Re-embedding complete: corpus={}", emb_engine.corpus_size());
    }

    let embedding_engine = Arc::new(parking_lot::RwLock::new(emb_engine));

    // Rebuild knowledge graph from (re-embedded) memories
    {
        let mut g = graph.write();
        for mem in &all_mems {
            g.add_memory(mem);
        }
        tracing::info!("Graph rebuilt: {} nodes, {} edges", g.node_count(), g.edge_count());
        // ADR-116: Build spectral sparsifier for analytics
        g.rebuild_sparsifier();
    }

    // Hydrate vote tracker from persisted quality scores (prevent re-voting)
    store.rebuild_vote_tracker().await;

    // Hydrate LoRA from Firestore
    {
        let docs = store.firestore_list_public("brain_lora").await;
        let mut lora = lora_federation.write();
        for doc in docs {
            if let Some(epoch) = doc.get("epoch").and_then(|v| v.as_u64()) {
                lora.epoch = epoch;
            }
            if let Some(consensus) = doc.get("consensus") {
                if let Ok(c) = serde_json::from_value::<ConsensusLoraWeights>(consensus.clone()) {
                    lora.consensus = Some(c);
                }
            }
        }
        if lora.epoch > 0 {
            tracing::info!("LoRA state loaded from Firestore: epoch {}", lora.epoch);
        }
    }

    let nonce_store = Arc::new(crate::types::NonceStore::new());

    // RVF feature flags — read once at startup (ADR-075)
    let rvf_flags = crate::types::RvfFeatureFlags::from_env();

    // Cached Verifier with compiled PiiStripper (15 regexes compiled once, not per request)
    let verifier = Arc::new(parking_lot::RwLock::new(crate::verify::Verifier::new()));

    // Differential privacy engine (ADR-075 Phase 3)
    let dp_engine = Arc::new(parking_lot::Mutex::new(
        rvf_federation::DiffPrivacyEngine::gaussian(rvf_flags.dp_epsilon, 1e-5, 1.0, 10.0)
            .expect("valid DP parameters"),
    ));

    // Negative cache for degenerate queries (ADR-075 Phase 6)
    let negative_cache = Arc::new(parking_lot::Mutex::new(
        rvf_runtime::NegativeCache::new(5, std::time::Duration::from_secs(3600), 10_000),
    ));

    // Global Workspace Theory attention layer (ADR-075 AGI)
    let workspace = Arc::new(parking_lot::RwLock::new(
        ruvector_nervous_system::routing::workspace::GlobalWorkspace::with_threshold(7, 0.3),
    ));

    // Temporal delta tracking for knowledge evolution (ADR-075 AGI)
    let delta_stream = Arc::new(parking_lot::RwLock::new(
        ruvector_delta_core::DeltaStream::for_vectors(crate::embeddings::EMBED_DIM),
    ));

    let sessions: Arc<dashmap::DashMap<String, tokio::sync::mpsc::Sender<String>>> =
        Arc::new(dashmap::DashMap::new());

    // Cloud Pipeline state (atomic counters + feed configs)
    let pipeline_metrics = Arc::new(crate::types::PipelineState::new());
    let feeds: Arc<dashmap::DashMap<String, crate::types::FeedConfig>> =
        Arc::new(dashmap::DashMap::new());

    // ── Common Crawl Integration (ADR-115) ──
    let web_store = Arc::new(crate::web_store::WebMemoryStore::new(store.clone()));
    let crawl_adapter = Arc::new(crate::pipeline::CommonCrawlAdapter::new());
    tracing::info!("Common Crawl adapter initialized (ADR-115)");

    // ── Midstream Platform (ADR-077) ──
    let nano_scheduler = Arc::new(crate::midstream::create_scheduler());
    let attractor_results = Arc::new(parking_lot::RwLock::new(std::collections::HashMap::new()));
    // Temporal solver: x86_64 only (uses AVX2 SIMD)
    #[cfg(feature = "x86-simd")]
    let temporal_solver = Arc::new(parking_lot::RwLock::new(
        temporal_neural_solver::TemporalSolver::new(
            crate::embeddings::EMBED_DIM,
            64, // hidden size
            crate::embeddings::EMBED_DIM,
        ),
    ));
    #[cfg(not(feature = "x86-simd"))]
    let temporal_solver = Arc::new(parking_lot::RwLock::new(
        crate::types::TemporalSolverStub::new(
            crate::embeddings::EMBED_DIM,
            64,
            crate::embeddings::EMBED_DIM,
        ),
    ));
    let strange_loop = Arc::new(parking_lot::RwLock::new(
        crate::midstream::create_strange_loop(),
    ));
    tracing::info!(
        "Midstream platform initialized: scheduler={} attractor={} solver={} strange_loop={}",
        rvf_flags.midstream_scheduler,
        rvf_flags.midstream_attractor,
        rvf_flags.midstream_solver,
        rvf_flags.midstream_strange_loop,
    );

    // ── Neural-Symbolic + Internal Voice (ADR-110) ──
    let internal_voice = Arc::new(parking_lot::RwLock::new(
        crate::voice::InternalVoice::default(),
    ));
    let neural_symbolic = Arc::new(parking_lot::RwLock::new(
        crate::symbolic::NeuralSymbolicBridge::default(),
    ));
    let optimizer = Arc::new(parking_lot::RwLock::new(
        crate::optimizer::GeminiOptimizer::default(),
    ));
    tracing::info!(
        "Cognitive layer initialized: internal_voice, neural_symbolic bridge, optimizer={}",
        optimizer.read().is_configured()
    );

    let state = AppState {
        store,
        gcs,
        graph,
        rate_limiter,
        ranking,
        cognitive,
        drift,
        aggregator,
        domain_engine,
        sona,
        lora_federation,
        embedding_engine,
        nonce_store,
        dp_engine,
        negative_cache,
        rvf_flags,
        workspace,
        delta_stream,
        verifier,
        read_only: Arc::new(AtomicBool::new(false)),
        start_time: std::time::Instant::now(),
        nano_scheduler,
        attractor_results,
        temporal_solver,
        strange_loop,
        sessions,
        internal_voice,
        neural_symbolic,
        optimizer,
        pipeline_metrics,
        feeds,
        web_store,
        crawl_adapter,
        cached_partition: Arc::new(parking_lot::RwLock::new(None)),
    };

    let router = Router::new()
        .route("/", get(landing_page))
        .route("/robots.txt", get(robots_txt))
        .route("/sitemap.xml", get(sitemap_xml))
        .route("/og-image.svg", get(og_image))
        .route("/.well-known/brain-manifest.json", get(brain_manifest))
        .route("/.well-known/agent-guide.md", get(agent_guide))
        .route("/origin", get(origin_page))
        .route("/v1/health", get(health))
        .route("/v1/challenge", get(issue_challenge))
        .route("/v1/memories", post(share_memory))
        .route("/v1/memories/search", get(search_memories))
        .route("/v1/memories/list", get(list_memories))
        .route("/v1/memories/:id", get(get_memory))
        .route("/v1/memories/:id/vote", post(vote_memory))
        .route("/v1/memories/:id", delete(delete_memory))
        .route("/v1/transfer", post(transfer))
        .route("/v1/verify", post(verify_endpoint))
        .route("/v1/drift", get(drift_report))
        .route("/v1/partition", get(partition))
        .route("/v1/status", get(status))
        .route("/v1/explore", get(explore_meta_learning))
        .route("/v1/sona/stats", get(sona_stats))
        .route("/v1/temporal", get(temporal_stats))
        .route("/v1/midstream", get(midstream_stats))
        .route("/v1/lora/latest", get(lora_latest))
        .route("/v1/lora/submit", post(lora_submit))
        .route("/v1/training/preferences", get(training_preferences))
        .route("/v1/train", post(train_endpoint))
        // Brainpedia (ADR-062)
        .route("/v1/pages", get(list_pages).post(create_page))
        .route("/v1/pages/:id", get(get_page))
        .route("/v1/pages/:id/deltas", post(submit_delta))
        .route("/v1/pages/:id/deltas", get(list_deltas))
        .route("/v1/pages/:id/evidence", post(add_evidence))
        .route("/v1/pages/:id/promote", post(promote_page))
        // WASM Executable Nodes (ADR-063)
        .route("/v1/nodes", get(list_nodes))
        .route("/v1/nodes", post(publish_node))
        .route("/v1/nodes/:id", get(get_node))
        .route("/v1/nodes/:id/wasm", get(get_node_wasm))
        .route("/v1/nodes/:id/revoke", post(revoke_node))
        // Cloud Pipeline (real-time injection + optimization)
        .route("/v1/pipeline/inject", post(pipeline_inject))
        .route("/v1/pipeline/inject/batch", post(pipeline_inject_batch))
        .route("/v1/pipeline/pubsub", post(pipeline_pubsub_push))
        .route("/v1/pipeline/metrics", get(pipeline_metrics_handler))
        .route("/v1/pipeline/optimize", post(pipeline_optimize))
        .route("/v1/pipeline/feeds", post(pipeline_add_feed).get(pipeline_list_feeds))
        .route("/v1/pipeline/scheduler/status", get(pipeline_scheduler_status))
        // Common Crawl Integration (ADR-096 §10, ADR-115)
        .route("/v1/pipeline/crawl/discover", post(pipeline_crawl_discover))
        .route("/v1/pipeline/crawl/stats", get(pipeline_crawl_stats))
        .route("/v1/pipeline/crawl/test", get(pipeline_crawl_test))
        // MCP SSE transport
        .route("/sse", get(sse_handler))
        .route("/messages", post(messages_handler))
        // ── Cognitive Layer (ADR-110) ──
        .route("/v1/cognitive/status", get(cognitive_status))
        .route("/v1/voice/working", get(voice_working_memory))
        .route("/v1/voice/history", get(voice_history))
        .route("/v1/voice/goal", post(voice_set_goal))
        .route("/v1/propositions", get(list_propositions))
        .route("/v1/reason", post(reason_endpoint))
        .route("/v1/ground", post(ground_proposition))
        .route("/v1/train/enhanced", post(train_enhanced_endpoint))
        // ── Gemini Optimizer ──
        .route("/v1/optimizer/status", get(optimizer_status))
        .route("/v1/optimize", post(optimize_endpoint))
        .layer({
            // CORS origins: configurable via CORS_ORIGINS env var (comma-separated).
            // Falls back to safe defaults if unset.
            let origins: Vec<axum::http::HeaderValue> = std::env::var("CORS_ORIGINS")
                .unwrap_or_else(|_| "https://brain.ruv.io,https://pi.ruv.io,http://localhost:8080,http://127.0.0.1:8080".to_string())
                .split(',')
                .filter_map(|s| s.trim().parse::<axum::http::HeaderValue>().ok())
                .collect();
            CorsLayer::new()
                .allow_origin(origins)
                .allow_methods([
                    axum::http::Method::GET,
                    axum::http::Method::POST,
                    axum::http::Method::DELETE,
                    axum::http::Method::OPTIONS,
                ])
                .allow_headers([
                    axum::http::header::AUTHORIZATION,
                    axum::http::header::CONTENT_TYPE,
                    axum::http::header::ACCEPT,
                ])
        })
        .layer(TraceLayer::new_for_http())
        .layer(tower_http::limit::RequestBodyLimitLayer::new(1_048_576)) // 1MB
        // Security response headers
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            axum::http::header::HeaderName::from_static("x-content-type-options"),
            axum::http::header::HeaderValue::from_static("nosniff"),
        ))
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            axum::http::header::HeaderName::from_static("x-frame-options"),
            axum::http::header::HeaderValue::from_static("DENY"),
        ))
        .with_state(state.clone());

    (router, state)
}

/// Run a training cycle: SONA force_learn + domain evolve_population.
/// Returns a summary of what happened.
pub fn run_training_cycle(state: &AppState) -> TrainingCycleResult {
    let sona_result = state.sona.write().force_learn();
    let mut domain = state.domain_engine.write();
    let pareto_before = domain.meta.pareto.len();
    domain.evolve_population();
    let pareto_after = domain.meta.pareto.len();

    let sona_stats = state.sona.read().stats();

    TrainingCycleResult {
        sona_message: sona_result,
        sona_patterns: sona_stats.patterns_stored,
        pareto_before,
        pareto_after,
        memory_count: state.store.memory_count(),
        vote_count: state.store.vote_count(),
    }
}

/// Enhanced training result (ADR-110)
#[derive(Debug, Clone, serde::Serialize)]
pub struct EnhancedTrainingResult {
    pub sona_message: String,
    pub sona_patterns: usize,
    pub pareto_before: usize,
    pub pareto_after: usize,
    pub memory_count: usize,
    pub vote_count: u64,
    /// Propositions extracted from clusters
    pub propositions_extracted: usize,
    /// Internal voice thoughts during reflection
    pub voice_thoughts: usize,
    /// Working memory utilization
    pub working_memory_load: f64,
    /// Neural-symbolic rule count
    pub rule_count: usize,
}

/// Run enhanced training cycle with neural-symbolic feedback (ADR-110).
/// Integrates: SONA → Neural-Symbolic Extraction → Internal Voice Reflection
pub fn run_enhanced_training_cycle(state: &AppState) -> EnhancedTrainingResult {
    // 1. SONA trajectory learning (existing)
    let sona_result = state.sona.write().force_learn();

    // 2. Domain evolution (existing)
    let mut domain = state.domain_engine.write();
    let pareto_before = domain.meta.pareto.len();
    domain.evolve_population();
    let pareto_after = domain.meta.pareto.len();
    drop(domain);

    // 3. Neural-symbolic rule extraction (ADR-110)
    let all_memories = state.store.all_memories();
    let clusters = build_memory_clusters(&all_memories);
    let propositions_extracted = {
        let mut ns = state.neural_symbolic.write();
        let props = ns.extract_from_clusters(&clusters);
        props.len()
    };

    // 3b. ADR-123: Record drift snapshots from cluster centroids
    {
        let mut drift = state.drift.write();
        for (centroid, _ids, category) in &clusters {
            drift.record(category, centroid);
        }
        // Also record a global centroid (average of all cluster centroids)
        if !clusters.is_empty() {
            let dim = clusters[0].0.len();
            let mut global = vec![0.0f32; dim];
            for (centroid, _, _) in &clusters {
                for (i, &v) in centroid.iter().enumerate() {
                    if i < dim {
                        global[i] += v;
                    }
                }
            }
            let n = clusters.len() as f32;
            for g in &mut global {
                *g /= n;
            }
            drift.record("global", &global);
        }
    }

    // 3c. Cache MinCut partition result (avoids recomputing on every /v1/partition request)
    {
        let graph = state.graph.read();
        if graph.node_count() > 0 {
            let (clusters, cut_value, edge_strengths) = graph.partition_full(2);
            let result = crate::types::PartitionResult {
                total_memories: graph.node_count(),
                clusters,
                cut_value,
                edge_strengths,
                cut_hash: None,
                first_separable_vertex: None,
            };
            *state.cached_partition.write() = Some(result);
        }
    }

    // 4. Internal voice reflection (ADR-110)
    let voice_thoughts = {
        let mut voice = state.internal_voice.write();
        let reflections = voice.reflect_on_learning(&sona_result);

        // Record observation about the learning
        if propositions_extracted > 0 {
            voice.observe(
                format!("extracted {} symbolic propositions", propositions_extracted),
                uuid::Uuid::nil(),
            );
        }

        reflections.len()
    };

    let sona_stats = state.sona.read().stats();
    let working_memory_load = state.internal_voice.read().working_memory_utilization();
    let rule_count = state.neural_symbolic.read().rule_count();

    EnhancedTrainingResult {
        sona_message: sona_result,
        sona_patterns: sona_stats.patterns_stored,
        pareto_before,
        pareto_after,
        memory_count: state.store.memory_count(),
        vote_count: state.store.vote_count(),
        propositions_extracted,
        voice_thoughts,
        working_memory_load,
        rule_count,
    }
}

/// Build clusters from memories for proposition extraction.
fn build_memory_clusters(memories: &[BrainMemory]) -> Vec<(Vec<f32>, Vec<uuid::Uuid>, String)> {
    use std::collections::HashMap;

    // Group memories by category
    let mut by_category: HashMap<String, Vec<&BrainMemory>> = HashMap::new();
    for mem in memories {
        let cat = mem.category.to_string();
        by_category.entry(cat).or_default().push(mem);
    }

    let mut clusters = Vec::new();
    for (category, mems) in by_category {
        if mems.len() < 3 {
            continue; // Skip small clusters
        }

        // Compute centroid
        let dim = mems[0].embedding.len();
        let mut centroid = vec![0.0f32; dim];
        for mem in &mems {
            for (i, &v) in mem.embedding.iter().enumerate() {
                if i < dim {
                    centroid[i] += v;
                }
            }
        }
        let n = mems.len() as f32;
        for c in &mut centroid {
            *c /= n;
        }

        let ids: Vec<uuid::Uuid> = mems.iter().map(|m| m.id).collect();
        clusters.push((centroid, ids, category));
    }

    clusters
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let persistence_mode = if state.store.is_persistent() {
        "firestore"
    } else {
        "local-only"
    };
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        domain: "π.ruv.io".to_string(),
        uptime_seconds: state.start_time.elapsed().as_secs(),
        persistence_mode: persistence_mode.to_string(),
    })
}

/// Issue a challenge nonce for replay protection.
/// Clients must include this nonce in write requests.
/// Nonces are single-use and expire after 5 minutes.
async fn issue_challenge(
    State(state): State<AppState>,
) -> Json<ChallengeResponse> {
    let (nonce, expires_at) = state.nonce_store.issue();
    Json(ChallengeResponse {
        nonce,
        expires_at,
    })
}

/// Validate a nonce if provided. Returns Err if nonce is present but invalid.
/// When nonce is None (backward compatibility), silently passes.
fn validate_nonce(state: &AppState, nonce: &Option<String>) -> Result<(), (StatusCode, String)> {
    if let Some(ref n) = nonce {
        if !n.is_empty() && !state.nonce_store.consume(n) {
            return Err((
                StatusCode::BAD_REQUEST,
                "Invalid or expired nonce — get a fresh one from GET /v1/challenge".into(),
            ));
        }
    }
    Ok(())
}

/// Guard: reject writes when the negative-cost fuse is tripped.
fn check_read_only(state: &AppState) -> Result<(), (StatusCode, String)> {
    if state.read_only.load(Ordering::Relaxed) {
        Err((StatusCode::SERVICE_UNAVAILABLE, "Server is in read-only mode".into()))
    } else {
        Ok(())
    }
}

async fn share_memory(
    State(state): State<AppState>,
    headers: HeaderMap,
    contributor: AuthenticatedContributor,
    Json(req): Json<ShareRequest>,
) -> Result<(StatusCode, Json<ShareResponse>), (StatusCode, String)> {
    // Negative-cost fuse
    check_read_only(&state)?;

    // Nonce validation (replay protection)
    validate_nonce(&state, &req.nonce)?;

    // Rate limit check (per-key + per-IP anti-Sybil)
    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }
    let client_ip = extract_client_ip(&headers);
    if !state.rate_limiter.check_ip_write(&client_ip) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "IP write rate limit exceeded".into()));
    }

    // ── Phase 2 (ADR-075): PII stripping ──
    let (title, content, tags, redaction_log_json) = if state.rvf_flags.pii_strip {
        let mut field_pairs: Vec<(&str, &str)> = vec![
            ("title", &req.title),
            ("content", &req.content),
        ];
        for tag in &req.tags {
            field_pairs.push(("tag", tag));
        }
        let (stripped, log) = state.verifier.write().strip_pii_fields(&field_pairs);
        let stripped_title = stripped[0].1.clone();
        let stripped_content = stripped[1].1.clone();
        let stripped_tags: Vec<String> = stripped[2..].iter().map(|(_, v)| v.clone()).collect();
        let log_json = serde_json::to_string(&log).ok();
        if log.total_redactions > 0 {
            tracing::info!("PII stripped: {} redactions in '{}'", log.total_redactions, stripped_title);
        }
        (stripped_title, stripped_content, stripped_tags, log_json)
    } else {
        (req.title, req.content, req.tags, None)
    };

    // Auto-generate embedding via ruvllm if client didn't provide one or dim mismatches
    let embedding = if req.embedding.is_empty()
        || req.embedding.len() != crate::embeddings::EMBED_DIM
    {
        let text = crate::embeddings::EmbeddingEngine::prepare_text(&title, &content, &tags);
        let emb = state.embedding_engine.read().embed_for_storage(&text);
        tracing::debug!("Auto-generated {}-dim embedding for '{}'", emb.len(), title);
        emb
    } else {
        req.embedding
    };

    // Verify input (uses final embedding — PII already stripped if enabled)
    state.verifier.read()
        .verify_share(&title, &content, &tags, &embedding)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // ── Phase 3 (ADR-075): Differential privacy noise on embedding ──
    let (embedding, dp_proof_json) = if state.rvf_flags.dp_enabled {
        let mut params: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
        let proof = state.dp_engine.lock().add_noise(&mut params);
        let noised: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        let proof_json = serde_json::to_string(&proof).ok();
        (noised, proof_json)
    } else {
        (embedding, None)
    };

    // ── Phase 4 (ADR-075): Witness chains ──
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let (witness_chain_bytes, witness_hash) = if state.rvf_flags.witness {
        // Build 3-entry witness chain: pii_strip → embed → content
        let pii_data = format!("pii_strip:{}:{}", title, content);
        let mut emb_bytes = Vec::with_capacity(embedding.len() * 4);
        for v in &embedding {
            emb_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let content_data = format!("content:{}:{}:{}", title, content, tags.join(","));

        let entries = vec![
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(pii_data.as_bytes()),
                timestamp_ns: now_ns,
                witness_type: 0x01, // PROVENANCE
            },
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(&emb_bytes),
                timestamp_ns: now_ns,
                witness_type: 0x02, // COMPUTATION
            },
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(content_data.as_bytes()),
                timestamp_ns: now_ns,
                witness_type: 0x01, // PROVENANCE
            },
        ];
        let chain = rvf_crypto::create_witness_chain(&entries);
        let hash = hex::encode(rvf_crypto::shake256_256(&chain));
        (Some(chain), hash)
    } else if req.witness_hash.is_empty() {
        // Fallback: compute witness hash from content (backward compat)
        let mut data = Vec::new();
        data.extend_from_slice(b"ruvector-witness:");
        data.extend_from_slice(title.as_bytes());
        data.extend_from_slice(b":");
        data.extend_from_slice(content.as_bytes());
        let hash = hex::encode(rvf_crypto::shake256_256(&data));
        (None, hash)
    } else {
        (None, req.witness_hash)
    };

    // ── Phase 4 (ADR-075): Adversarial embedding detection ──
    if state.rvf_flags.adversarial {
        // Use embedding values as distance proxy for degenerate detection
        if crate::verify::Verifier::verify_embedding_not_adversarial(&embedding, 10) {
            tracing::warn!(
                "Adversarial embedding detected for '{}' from contributor '{}'",
                title, contributor.pseudonym
            );
            // Phase 6: record in negative cache if enabled
            if state.rvf_flags.neg_cache {
                let sig = rvf_runtime::QuerySignature::from_query(&embedding);
                state.negative_cache.lock().record_degenerate(sig);
            }
        }
    }

    // Ensure contributor exists
    state
        .store
        .get_or_create_contributor(&contributor.pseudonym, contributor.is_system)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // ── Phase 5 (ADR-075): Build RVF container ──
    let (rvf_gcs_path, rvf_segments) = if state.rvf_flags.container {
        let input = crate::pipeline::RvfPipelineInput {
            memory_id: &id.to_string(),
            embedding: &embedding,
            title: &title,
            content: &content,
            tags: &tags,
            category: &req.category.to_string(),
            contributor_id: &contributor.pseudonym,
            witness_chain: witness_chain_bytes.as_deref(),
            dp_proof_json: dp_proof_json.as_deref(),
            redaction_log_json: redaction_log_json.as_deref(),
        };
        let container = crate::pipeline::build_rvf_container(&input)
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;
        let seg_count = crate::pipeline::count_segments(&container) as u32;
        // Upload to GCS
        let path = state
            .gcs
            .upload_rvf(&contributor.pseudonym, &id.to_string(), &container)
            .await
            .ok();
        (path, Some(seg_count))
    } else {
        // Store client-provided RVF bytes if any (backward compat)
        let path = if let Some(rvf_b64) = &req.rvf_bytes {
            if let Ok(rvf_data) = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, rvf_b64) {
                state.gcs.upload_rvf(&contributor.pseudonym, &id.to_string(), &rvf_data).await.ok()
            } else { None }
        } else { None };
        (path, None)
    };

    let memory = BrainMemory {
        id,
        category: req.category,
        title,
        content,
        tags,
        code_snippet: req.code_snippet,
        embedding: embedding.clone(),
        contributor_id: contributor.pseudonym.clone(),
        quality_score: BetaParams::new(),
        partition_id: None,
        witness_hash: witness_hash.clone(),
        rvf_gcs_path,
        redaction_log: redaction_log_json,
        dp_proof: dp_proof_json,
        witness_chain: witness_chain_bytes,
        created_at: now,
        updated_at: now,
    };

    // Add to embedding corpus for future context-aware embeddings
    state.embedding_engine.write().add_to_corpus(&id.to_string(), embedding.clone(), None);

    // Record embedding in cognitive engine and drift monitor
    {
        let mut cog = state.cognitive.write();
        cog.store_pattern(&id.to_string(), &memory.embedding);
        let mut drift = state.drift.write();
        drift.record(&memory.category.to_string(), &memory.embedding);
    }

    // ── Temporal: Record embedding delta (ADR-075 AGI) ──
    // Reuse now_ns from witness chain computation above to avoid redundant syscall
    if state.rvf_flags.temporal_enabled {
        let delta = ruvector_delta_core::VectorDelta::from_dense(embedding.clone());
        state.delta_stream.write().push_with_timestamp(delta, now_ns);
    }

    // ── Meta-learning: Record contribution as decision (ADR-075 AGI) ──
    if state.rvf_flags.meta_learning_enabled {
        let bucket = ruvector_domain_expansion::ContextBucket {
            difficulty_tier: "default".into(),
            category: memory.category.to_string(),
        };
        let arm = ruvector_domain_expansion::ArmId("contribute".into());
        state.domain_engine.write().meta.record_decision(&bucket, &arm, 0.5);
    }
    // Capture category key before memory is moved into store
    let memory_cat_key = memory.category.to_string();

    // Add to graph
    {
        let mut graph = state.graph.write();
        graph.add_memory(&memory);
    }

    // Store in Firestore
    state
        .store
        .store_memory(memory)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Update contributor reputation: record activity + increment count
    state.store.record_contribution(&contributor.pseudonym).await;

    // ── SONA: Record share as learning trajectory ──
    // Uses embedding by reference where possible; begin_trajectory needs owned vec
    if state.rvf_flags.sona_enabled {
        let sona = state.sona.read();
        let emb_for_step = embedding.clone();
        let mut builder = sona.begin_trajectory(embedding);
        builder.add_step(emb_for_step, vec![], 0.5);
        sona.end_trajectory(builder, 0.5);
    }

    // ── Midstream: Update attractor analysis for this category (ADR-077 Phase 9c) ──
    // Amortized: only recompute every 10th write (memory_count % 10 == 0) to avoid
    // O(n) scan on every write. Lyapunov estimates are stable enough to skip updates.
    if state.rvf_flags.midstream_attractor && state.store.memory_count() % 10 == 0 {
        let cat_key = memory_cat_key;
        let cat_embeddings: Vec<Vec<f32>> = state.store
            .all_memories()
            .iter()
            .filter(|m| m.category.to_string() == cat_key)
            .map(|m| m.embedding.clone())
            .collect();
        if let Some(result) = crate::midstream::analyze_category_attractor(&cat_embeddings) {
            state.attractor_results.write().insert(cat_key, result);
        }
    }

    Ok((
        StatusCode::CREATED,
        Json(ShareResponse {
            id,
            partition_id: None,
            quality_score: BetaParams::new().mean(),
            witness_hash,
            rvf_segments,
        }),
    ))
}

async fn search_memories(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Query(query): Query<SearchQuery>,
) -> Result<Json<Vec<ScoredBrainMemory>>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    let limit = query.limit.unwrap_or(10).min(100);
    let min_quality = query.min_quality.unwrap_or(0.0);

    // ── Phase 6 (ADR-075): Negative cache check ──
    // If the query embedding is blacklisted, return empty results early
    if state.rvf_flags.neg_cache {
        if let Some(ref emb) = query.embedding {
            let sig = rvf_runtime::QuerySignature::from_query(emb);
            if state.negative_cache.lock().is_blacklisted(&sig) {
                tracing::warn!("Query blocked by negative cache for contributor '{}'", contributor.pseudonym);
                return Ok(Json(Vec::new()));
            }
        }
    }

    // Generate query embedding: use ruvllm if text query, or client-provided embedding
    let query_embedding = if let Some(emb) = query.embedding {
        if emb.len() == crate::embeddings::EMBED_DIM {
            emb
        } else {
            // Dimension mismatch — re-embed from text if available
            if let Some(ref q) = query.q {
                state.embedding_engine.read().embed(q)
            } else {
                return Ok(Json(Vec::new()));
            }
        }
    } else if let Some(ref q) = query.q {
        // Text query → generate embedding via ruvllm
        state.embedding_engine.read().embed(q)
    } else {
        return Ok(Json(Vec::new()));
    };

    let tags: Option<Vec<String>> = query.tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

    // Fetch ALL memories for keyword-dominant ranking.
    let raw = state
        .store
        .search_memories(
            &query_embedding,
            query.category.as_ref(),
            tags.as_deref(),
            0, // 0 = fetch all
            min_quality,
        )
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // ── Query Expansion: synonym map for common abbreviations ──
    // Static synonym map — compiled once, reused across all search requests.
    use std::sync::LazyLock;
    static SYNONYMS: LazyLock<std::collections::HashMap<&'static str, &'static [&'static str]>> =
        LazyLock::new(|| {
            std::collections::HashMap::from([
                ("ml", &["machine", "learning"][..]),
                ("ai", &["artificial", "intelligence"][..]),
                ("gnn", &["graph", "neural", "network"][..]),
                ("rl", &["reinforcement", "learning"][..]),
                ("llm", &["large", "language", "model"][..]),
                ("dl", &["deep", "learning"][..]),
                ("nlp", &["natural", "language", "processing"][..]),
                ("cv", &["computer", "vision"][..]),
                ("nn", &["neural", "network"][..]),
                ("api", &["interface", "endpoint"][..]),
                ("auth", &["authentication", "authorization"][..]),
                ("db", &["database"][..]),
                ("ppr", &["pagerank", "personalized"][..]),
                ("snn", &["spiking", "neural"][..]),
                ("wasm", &["webassembly"][..]),
                ("rvf", &["ruvector", "format", "cognitive", "container"][..]),
                ("pii", &["personal", "identifiable", "information", "privacy"][..]),
                ("dp", &["differential", "privacy"][..]),
                ("cicd", &["continuous", "integration", "deployment"][..]),
                ("tdd", &["test", "driven", "development"][..]),
                ("ddd", &["domain", "driven", "design"][..]),
                ("sse", &["server", "sent", "events"][..]),
                ("mcp", &["model", "context", "protocol"][..]),
                ("lora", &["low", "rank", "adaptation"][..]),
                ("embed", &["embedding", "embeddings"][..]),
                ("embedding", &["embed", "embeddings"][..]),
                ("embeddings", &["embed", "embedding"][..]),
                ("neural", &["embedding", "network", "snn"][..]),
                ("mincut", &["graph", "partitioning", "cut"][..]),
                ("pagerank", &["ppr", "ranking", "graph"][..]),
                ("drift", &["anomaly", "deviation", "shift"][..]),
                ("scoring", &["reputation", "ranking", "quality"][..]),
                ("engine", &["system", "pipeline", "framework"][..]),
                ("byzantine", &["fault", "tolerant", "consensus"][..]),
                ("federated", &["federation", "aggregation"][..]),
            ])
        });
    fn expand_synonyms(tokens: &[String]) -> Vec<String> {
        let mut expanded = tokens.to_vec();
        let mut seen: std::collections::HashSet<&str> = tokens.iter().map(|s| s.as_str()).collect();
        for tok in tokens {
            if let Some(syns) = SYNONYMS.get(tok.as_str()) {
                for &s in *syns {
                    if seen.insert(s) {
                        expanded.push(s.to_string());
                    }
                }
            }
        }
        expanded
    }

    // Tokenize query for keyword matching (keep words >= 2 chars)
    let query_lower = query.q.as_deref().unwrap_or("").to_lowercase();
    let query_tokens: Vec<String> = query_lower
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() >= 2)
        .map(|s| s.to_string())
        .collect();
    // Expanded tokens include synonyms (used for matching, not phrase bonus)
    let expanded_tokens = expand_synonyms(&query_tokens);

    // ── Graph PPR scores: blend cosine+PageRank from knowledge graph ──
    // Use write lock briefly: ranked_search may lazily rebuild CSR cache
    let graph_scores: std::collections::HashMap<Uuid, f64> = {
        let mut g = state.graph.write();
        if g.node_count() >= 3 {
            g.ranked_search(&query_embedding, limit * 3)
                .into_iter()
                .collect()
        } else {
            std::collections::HashMap::new()
        }
    };

    // Helper: check if a word appears as a whole word (not just substring)
    fn word_match(haystack: &str, needle: &str) -> bool {
        haystack
            .split(|c: char| !c.is_alphanumeric())
            .any(|word| word == needle)
    }

    // Build scored list: keyword-dominant with embedding + graph + vote signals
    let mut scored: Vec<(f64, BrainMemory)> = raw
        .into_iter()
        .map(|(_, m)| {
            let rep = state
                .store
                .get_contributor_reputation(&m.contributor_id)
                .map(|r| crate::reputation::ReputationManager::contribution_weight(&r))
                .unwrap_or(0.1);

            let vec_sim = cosine_similarity(&query_embedding, &m.embedding) as f64;

            // Graph PPR score (0.0 if node not in graph results)
            let graph_sim = graph_scores.get(&m.id).copied().unwrap_or(0.0);

            // Learning-to-rank: vote quality signal (Bayesian Beta mean)
            let vote_quality = m.quality_score.mean();
            // Boost well-voted memories: scale 0-1 where 0.5 is neutral
            let vote_boost = if m.quality_score.observations() >= 2.0 {
                (vote_quality - 0.5).max(0.0) * 0.3 // up to +0.15 for high-quality
            } else {
                0.0 // not enough votes to judge
            };

            let keyword_boost = if !query_tokens.is_empty() {
                let title_lower = m.title.to_lowercase();
                let content_lower = m.content.to_lowercase();
                let cat_lower = m.category.to_string().to_lowercase();

                // Phase 1: Exact phrase match in title (strongest possible signal)
                let phrase_bonus = if query_tokens.len() >= 2 && title_lower.contains(&query_lower) {
                    2.0  // title contains the exact query phrase — dominant signal
                } else if query_tokens.len() >= 2 && content_lower.contains(&query_lower) {
                    0.5  // content contains the exact query phrase
                } else {
                    0.0
                };

                // Phase 2: Per-token word-boundary matching with field weights
                // Use expanded tokens (synonyms) for broader recall
                let mut token_hits = 0usize;
                let mut token_weight = 0.0f64;
                for tok in &expanded_tokens {
                    let mut found = false;
                    // Original query tokens get full weight; expanded synonyms get 0.5x
                    let weight_mult = if query_tokens.contains(tok) { 1.0 } else { 0.5 };
                    if word_match(&title_lower, tok) { token_weight += 6.0 * weight_mult; found = true; }
                    if m.tags.iter().any(|t| {
                        let tl = t.to_lowercase();
                        word_match(&tl, tok) || tl == *tok
                    }) { token_weight += 4.0 * weight_mult; found = true; }
                    if word_match(&cat_lower, tok) { token_weight += 3.0 * weight_mult; found = true; }
                    if word_match(&content_lower, tok) { token_weight += 1.0 * weight_mult; found = true; }
                    if found { token_hits += 1; }
                }

                // Bonus: all original query tokens appear in title
                let orig_title_hits = query_tokens.iter()
                    .filter(|tok| word_match(&title_lower, tok))
                    .count();
                let all_in_title_bonus = if query_tokens.len() >= 2 && orig_title_hits == query_tokens.len() {
                    0.6
                } else {
                    0.0
                };

                // Coverage based on expanded tokens
                let coverage = token_hits as f64 / expanded_tokens.len().max(1) as f64;
                let depth = token_weight / (expanded_tokens.len().max(1) as f64 * 14.0);

                let base = coverage * 0.55 + depth * 0.45;
                (base + phrase_bonus + all_in_title_bonus).min(3.0)
            } else {
                0.0
            };

            // Final hybrid score: keyword-dominant with graph/vote as tiebreakers.
            // A constant +1.0 floor ensures ANY keyword match always outranks
            // non-keyword results, preventing RLM's contextual gravity from
            // promoting irrelevant but embeddings-similar memories.
            let hybrid = if keyword_boost > 0.0 {
                1.0 + keyword_boost * 0.85
                    + vec_sim * 0.05
                    + graph_sim * 0.04
                    + rep.min(1.0) * 0.03
                    + vote_boost * 0.03
            } else {
                // No keyword matches: embedding + graph + vote signals
                vec_sim * 0.45
                    + graph_sim * 0.25
                    + rep.min(1.0) * 0.15
                    + vote_boost * 0.15
            };

            (hybrid, m)
        })
        .collect();

    // Apply attention-based ranking adjustments
    {
        let ranker = state.ranking.read();
        ranker.rank(&mut scored);
    }

    // ── GWT Attention Layer: broadcast candidates and let salience competition select winners ──
    // NOTE: Write lock is scoped — released before SONA/meta read locks to avoid contention.
    if state.rvf_flags.gwt_enabled && scored.len() > limit {
        use ruvector_nervous_system::routing::workspace::Representation;
        let mut ws = state.workspace.write();
        ws.compete();

        let broadcast_count = (limit * 3).min(scored.len());
        for (i, (score, _mem)) in scored.iter().enumerate().take(broadcast_count) {
            let rep = Representation::new(
                vec![*score as f32],
                *score as f32,
                i as u16,
                i as u64,
            );
            ws.broadcast(rep);
        }

        let winners = ws.retrieve_top_k(limit);
        drop(ws); // Release write lock early — SONA/meta only need read locks

        let winner_set: std::collections::HashSet<usize> = winners
            .iter()
            .map(|w| w.source_module as usize)
            .collect();

        for (i, (score, _)) in scored.iter_mut().enumerate() {
            if winner_set.contains(&i) {
                *score += 0.1;
            }
        }

        // K-WTA sparse attention (no intermediate sort needed — applied additively)
        if scored.len() > limit {
            // Sort once for K-WTA input ordering
            scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            let kwta = ruvector_nervous_system::KWTALayer::new(scored.len(), limit);
            let activations: Vec<f32> = scored.iter().map(|(s, _)| *s as f32).collect();
            let sparse = kwta.sparse_normalized(&activations);
            for (i, (score, _)) in scored.iter_mut().enumerate() {
                if sparse[i] > 0.0 {
                    *score += sparse[i] as f64 * 0.05;
                }
            }
        }
    }

    // ── SONA: Pattern-based re-ranking ──
    if state.rvf_flags.sona_enabled {
        let sona = state.sona.read();
        let patterns = sona.find_patterns(&query_embedding, 5);
        if !patterns.is_empty() {
            let inv_len = 1.0 / patterns.len() as f64;
            for (score, mem) in &mut scored {
                let pattern_boost: f64 = patterns.iter()
                    .map(|p| {
                        cosine_similarity(&mem.embedding, &p.centroid) as f64
                            * p.avg_quality as f64
                    })
                    .sum::<f64>() * inv_len;
                *score += pattern_boost * 0.15;
            }
        }
    }

    // ── Meta-learning: Curiosity bonus for under-explored categories (ADR-075 AGI) ──
    if state.rvf_flags.meta_learning_enabled {
        let de = state.domain_engine.read();
        let default_tier: String = "default".into();
        for (score, mem) in &mut scored {
            let bucket = ruvector_domain_expansion::ContextBucket {
                difficulty_tier: default_tier.clone(),
                category: mem.category.to_string(),
            };
            let novelty = de.meta.curiosity.novelty_score(&bucket);
            *score += novelty as f64 * 0.05;
        }
    }

    // ── Midstream: Attractor stability bonus (ADR-077 Phase 9c) ──
    if state.rvf_flags.midstream_attractor {
        let attractors = state.attractor_results.read();
        for (score, mem) in &mut scored {
            let cat_key = mem.category.to_string();
            if let Some(result) = attractors.get(&cat_key) {
                *score += crate::midstream::attractor_stability_score(result) as f64;
            }
        }
    }

    // ── Midstream: Strange-loop meta-cognitive bonus (ADR-077 Phase 9e) ──
    // Only apply to top candidates to keep within 5ms budget.
    // Uses select_nth_unstable (O(n)) instead of full sort (O(n log n)).
    if state.rvf_flags.midstream_strange_loop && scored.len() > limit {
        let mut sl = state.strange_loop.write();
        let pivot = limit.min(scored.len()) - 1;
        scored.select_nth_unstable_by(pivot, |a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        for (score, mem) in scored.iter_mut().take(pivot + 1) {
            let quality = mem.quality_score.mean();
            let bonus = crate::midstream::strange_loop_score(&mut sl, *score, quality);
            *score += bonus as f64;
        }
    }

    // Single final sort after all AGI + midstream scoring layers
    scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(limit);
    let results: Vec<ScoredBrainMemory> = scored.into_iter().map(|(score, memory)| ScoredBrainMemory { memory, score }).collect();

    // ── SONA: Record search trajectory for learning ──
    if state.rvf_flags.sona_enabled && !results.is_empty() {
        let sona = state.sona.read();
        let mut builder = sona.begin_trajectory(query_embedding.clone());
        builder.add_step(
            results[0].memory.embedding.clone(),
            vec![],
            results[0].memory.quality_score.mean() as f32,
        );
        sona.end_trajectory(builder, 0.5);
    }

    // ── ADR-123: Auto-populate GWT working memory from top search result ──
    if !results.is_empty() {
        let top = &results[0];
        let wm_content = format!("[search] {}", top.memory.title);
        let wm_embedding = top.memory.embedding.clone();
        let mut voice = state.internal_voice.write();
        voice.remember(
            wm_content,
            wm_embedding,
            crate::voice::ContentSource::Perception,
        );
    }

    Ok(Json(results))
}

async fn list_memories(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Query(query): Query<ListQuery>,
) -> Result<Json<ListResponse>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    let limit = query.limit.unwrap_or(20).min(100);
    let offset = query.offset.unwrap_or(0);
    let sort = query.sort.unwrap_or_default();
    let tags: Option<Vec<String>> = query.tags.map(|t| t.split(',').map(|s| s.trim().to_string()).collect());

    let (memories, total_count) = state
        .store
        .list_memories(query.category.as_ref(), tags.as_deref(), limit, offset, &sort)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(ListResponse {
        memories,
        total_count,
        offset,
        limit,
    }))
}

async fn get_memory(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(id): Path<Uuid>,
) -> Result<Json<BrainMemory>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    let memory = state
        .store
        .get_memory(&id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Memory not found".into()))?;

    Ok(Json(memory))
}

async fn vote_memory(
    State(state): State<AppState>,
    headers: HeaderMap,
    contributor: AuthenticatedContributor,
    Path(id): Path<Uuid>,
    Json(vote): Json<VoteRequest>,
) -> Result<Json<BetaParams>, (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    // Look up the content author before voting
    let content_author = state
        .store
        .get_memory(&id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .map(|m| m.contributor_id.clone());

    // Anti-Sybil: one vote per IP per memory (ADR-082)
    // Skip IP dedup for the content author — self-votes are legitimate and
    // already gated by the per-key "one vote per contributor per memory" check.
    let is_author = content_author.as_deref() == Some(&contributor.pseudonym);
    if !is_author {
        let client_ip = extract_client_ip(&headers);
        if !state.rate_limiter.check_ip_vote(&client_ip, &id.to_string()) {
            return Err((StatusCode::FORBIDDEN, "Already voted on this memory from this network".into()));
        }
    }

    let was_upvoted = matches!(vote.direction, VoteDirection::Up);

    let updated = state
        .store
        .update_quality(&id, &vote.direction, &contributor.pseudonym)
        .await
        .map_err(|e| match e {
            crate::store::StoreError::NotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
            crate::store::StoreError::Forbidden(_) => (StatusCode::FORBIDDEN, e.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })?;

    // Update content author's reputation based on vote outcome
    if let Some(author) = content_author {
        state.store.update_reputation_from_vote(&author, was_upvoted).await;

        // Check for poisoning penalty if downvoted
        if !was_upvoted {
            let down_count = (updated.beta - 1.0) as u32;
            let quality = updated.mean();
            state.store.check_poisoning(&author, down_count, quality).await;
        }
    }

    // ── Temporal: Record vote as a quality-change delta (ADR-075 AGI) ──
    if state.rvf_flags.temporal_enabled {
        // Encode vote as a small delta: +1.0 for upvote, -1.0 for downvote
        let vote_signal = if was_upvoted { 1.0f32 } else { -1.0f32 };
        let delta = ruvector_delta_core::VectorDelta::from_dense(vec![vote_signal]);
        state.delta_stream.write().push(delta);
    }

    // ── Meta-learning: Feed vote as reward signal (ADR-075 AGI) ──
    if state.rvf_flags.meta_learning_enabled {
        let reward = if was_upvoted { 1.0f32 } else { 0.0f32 };
        if let Ok(Some(memory)) = state.store.get_memory(&id).await {
            let cat_str = memory.category.to_string();
            let bucket = ruvector_domain_expansion::ContextBucket {
                difficulty_tier: "default".into(),
                category: cat_str,
            };
            let arm = ruvector_domain_expansion::ArmId("search".into());
            state.domain_engine.write().meta.record_decision(&bucket, &arm, reward);
        }
    }

    // Ensure voter exists as contributor before recording activity
    state.store.get_or_create_contributor(&contributor.pseudonym, contributor.is_system).await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    state.store.record_contribution(&contributor.pseudonym).await;

    Ok(Json(updated))
}

async fn delete_memory(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(id): Path<Uuid>,
) -> Result<StatusCode, (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    let deleted = state
        .store
        .delete_memory(&id, &contributor.pseudonym)
        .await
        .map_err(|e| match e {
            crate::store::StoreError::Forbidden(_) => (StatusCode::FORBIDDEN, e.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })?;

    if deleted {
        let mut graph = state.graph.write();
        graph.remove_memory(&id);
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err((StatusCode::NOT_FOUND, "Memory not found".into()))
    }
}

async fn transfer(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Json(req): Json<TransferRequest>,
) -> Result<Json<TransferResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    use ruvector_domain_expansion::DomainId;

    let source_id = DomainId(req.source_domain.clone());
    let target_id = DomainId(req.target_domain.clone());

    // Compute real domain quality scores from stored memories before transfer.
    // Use fuzzy matching: category name, tag substring, or content match.
    let all_memories = state.store.all_memories();
    let src_lower = req.source_domain.to_lowercase();
    let tgt_lower = req.target_domain.to_lowercase();
    let source_memories: Vec<_> = all_memories.iter()
        .filter(|m| {
            m.category.to_string().to_lowercase().contains(&src_lower)
                || m.tags.iter().any(|t| t.to_lowercase().contains(&src_lower))
        })
        .collect();
    let target_memories: Vec<_> = all_memories.iter()
        .filter(|m| {
            m.category.to_string().to_lowercase().contains(&tgt_lower)
                || m.tags.iter().any(|t| t.to_lowercase().contains(&tgt_lower))
        })
        .collect();

    // Source quality: average quality of source domain memories (or neutral prior)
    let source_quality = if source_memories.is_empty() {
        0.5
    } else {
        source_memories.iter().map(|m| m.quality_score.mean()).sum::<f64>() / source_memories.len() as f64
    };

    // Target quality before transfer: average quality of target domain memories (or cold start)
    let target_before = if target_memories.is_empty() {
        0.3
    } else {
        target_memories.iter().map(|m| m.quality_score.mean()).sum::<f64>() / target_memories.len() as f64
    };

    // Use the shared DomainExpansionEngine to initiate cross-domain transfer.
    let verification = {
        let mut engine = state.domain_engine.write();
        engine.initiate_transfer(&source_id, &target_id);

        // Estimate target improvement: dampened transfer with minimum floor
        let improvement = ((source_quality - target_before) * 0.5).max(0.02);
        let target_after = target_before + improvement;

        // Cycle counts based on domain sizes
        let baseline_cycles = target_memories.len().max(10) as u64;
        let transfer_cycles = (baseline_cycles as f64 / (1.0 + source_quality)).ceil() as u64;

        engine.verify_transfer(
            &source_id,
            &target_id,
            source_quality as f32,  // source_before: real quality
            source_quality as f32,  // source_after: unchanged (no regression)
            target_before as f32,   // target_before: real quality
            target_after as f32,    // target_after: dampened improvement
            baseline_cycles,        // based on actual domain size
            transfer_cycles,        // estimated speedup
        )
    };

    let mut warnings = Vec::new();
    if source_memories.is_empty() {
        warnings.push(format!("No memories found matching source domain '{}'", req.source_domain));
    }
    if target_memories.is_empty() {
        warnings.push(format!("No memories found matching target domain '{}'", req.target_domain));
    }

    Ok(Json(TransferResponse {
        source_domain: req.source_domain,
        target_domain: req.target_domain,
        acceleration_factor: verification.acceleration_factor as f64,
        transfer_success: verification.promotable,
        message: format!(
            "Transfer initiated by {} (acceleration: {:.2}x, promotable: {})",
            contributor.pseudonym, verification.acceleration_factor, verification.promotable
        ),
        source_memory_count: source_memories.len(),
        target_memory_count: target_memories.len(),
        warnings,
    }))
}

async fn verify_endpoint(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Json(req): Json<VerifyRequest>,
) -> Result<Json<VerifyResponse>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    // Method 1: Witness chain steps + hash
    if let (Some(steps), Some(hash)) = (&req.witness_steps, &req.witness_hash) {
        let verifier = state.verifier.read();
        let step_refs: Vec<&str> = steps.iter().map(|s| s.as_str()).collect();
        return match verifier.verify_witness_chain(&step_refs, hash) {
            Ok(()) => Ok(Json(VerifyResponse {
                valid: true,
                method: "witness_chain".into(),
                message: "Witness chain verification passed".into(),
            })),
            Err(e) => Ok(Json(VerifyResponse {
                valid: false,
                method: "witness_chain".into(),
                message: format!("Witness chain verification failed: {e}"),
            })),
        };
    }

    // Method 2: Memory ID lookup
    if let Some(memory_id) = req.memory_id {
        return match state.store.get_memory(&memory_id).await {
            Ok(Some(mem)) => {
                // If witness_hash provided, verify it matches
                if let Some(ref hash) = req.witness_hash {
                    let equal = subtle::ConstantTimeEq::ct_eq(
                        mem.witness_hash.as_bytes(),
                        hash.as_bytes(),
                    );
                    if bool::from(equal) {
                        Ok(Json(VerifyResponse {
                            valid: true,
                            method: "memory_id".into(),
                            message: format!("Memory {} witness hash verified", memory_id),
                        }))
                    } else {
                        Ok(Json(VerifyResponse {
                            valid: false,
                            method: "memory_id".into(),
                            message: format!("Memory {} witness hash mismatch", memory_id),
                        }))
                    }
                } else {
                    Ok(Json(VerifyResponse {
                        valid: true,
                        method: "memory_id".into(),
                        message: format!("Memory {} exists and is valid", memory_id),
                    }))
                }
            }
            Ok(None) => Ok(Json(VerifyResponse {
                valid: false,
                method: "memory_id".into(),
                message: format!("Memory {} not found", memory_id),
            })),
            Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, e.to_string())),
        };
    }

    // Method 3: Content hash verification
    if let (Some(hash), Some(data)) = (&req.content_hash, &req.content_data) {
        let verifier = state.verifier.read();
        return match verifier.verify_content_hash(data.as_bytes(), hash) {
            Ok(()) => Ok(Json(VerifyResponse {
                valid: true,
                method: "content_hash".into(),
                message: "Content hash verification passed".into(),
            })),
            Err(e) => Ok(Json(VerifyResponse {
                valid: false,
                method: "content_hash".into(),
                message: format!("Content hash verification failed: {e}"),
            })),
        };
    }

    // Method 4: Binary witness chain bytes (base64)
    if let Some(ref b64) = req.witness_chain_bytes {
        use base64::Engine as _;
        let verifier = state.verifier.read();
        return match base64::engine::general_purpose::STANDARD.decode(b64) {
            Ok(bytes) => match verifier.verify_rvf_witness_chain(&bytes) {
                Ok(entries) => Ok(Json(VerifyResponse {
                    valid: true,
                    method: "witness_chain_bytes".into(),
                    message: format!("Binary witness chain valid ({} entries)", entries.len()),
                })),
                Err(e) => Ok(Json(VerifyResponse {
                    valid: false,
                    method: "witness_chain_bytes".into(),
                    message: format!("Binary witness chain invalid: {e}"),
                })),
            },
            Err(e) => Ok(Json(VerifyResponse {
                valid: false,
                method: "witness_chain_bytes".into(),
                message: format!("Invalid base64 encoding: {e}"),
            })),
        };
    }

    Err((StatusCode::BAD_REQUEST, "No verification method specified. Provide witness_steps+witness_hash, memory_id, content_hash+content_data, or witness_chain_bytes".into()))
}

async fn drift_report(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<DriftQuery>,
) -> Result<Json<DriftReport>, (StatusCode, String)> {
    let drift = state.drift.read();
    let report = drift.compute_drift(query.domain.as_deref());
    Ok(Json(report))
}

async fn partition(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<PartitionQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let min_size = query.min_cluster_size.unwrap_or(2);

    // Try the cache first (non-canonical, default min_size, no force)
    if !query.force && !query.canonical && min_size == 2 {
        if let Some(cached) = state.cached_partition.read().as_ref() {
            if query.compact {
                let compact: PartitionResultCompact = cached.clone().into();
                return Ok(Json(serde_json::to_value(compact).unwrap()));
            } else {
                return Ok(Json(serde_json::to_value(cached).unwrap()));
            }
        }
    }

    let graph = state.graph.read();

    let full_result = if query.canonical {
        // ADR-117: source-anchored canonical min-cut with stable hash
        let (clusters, cut_value, edge_strengths, cut_hash, first_sep) =
            graph.partition_canonical_full(min_size);
        PartitionResult {
            total_memories: graph.node_count(),
            clusters,
            cut_value,
            edge_strengths,
            cut_hash,
            first_separable_vertex: first_sep,
        }
    } else {
        let (clusters, cut_value, edge_strengths) = graph.partition_full(min_size);
        PartitionResult {
            total_memories: graph.node_count(),
            clusters,
            cut_value,
            edge_strengths,
            cut_hash: None,
            first_separable_vertex: None,
        }
    };

    // Return compact format by default to avoid SSE truncation of 128-dim centroids
    if query.compact {
        let compact: PartitionResultCompact = full_result.into();
        Ok(Json(serde_json::to_value(compact).unwrap()))
    } else {
        Ok(Json(serde_json::to_value(full_result).unwrap()))
    }
}

async fn status(
    State(state): State<AppState>,
) -> Json<StatusResponse> {
    let graph = state.graph.read();
    // Use node_count as a cheap proxy for cluster count instead of running
    // full MinCut partitioning on every status call (expensive O(V*E) op)
    let cluster_count = if graph.node_count() < 3 {
        if graph.node_count() > 0 { 1 } else { 0 }
    } else {
        // Estimate cluster count from edge density (cheap)
        let density = if graph.node_count() > 1 {
            graph.edge_count() as f64 / (graph.node_count() as f64 * (graph.node_count() - 1) as f64 / 2.0)
        } else {
            0.0
        };
        // High density = fewer clusters, low density = more
        if density > 0.8 { 1 } else if density > 0.5 { 2 } else { (graph.node_count() / 10).max(2).min(20) }
    };
    let lora = state.lora_federation.read();

    // Compute real average quality from all memories
    let all_memories = state.store.all_memories();
    let avg_quality = if all_memories.is_empty() {
        0.5
    } else {
        let sum: f64 = all_memories.iter().map(|m| m.quality_score.mean()).sum();
        sum / all_memories.len() as f64
    };

    // Compute real drift status from DriftMonitor
    let drift = state.drift.read();
    let drift_report = drift.compute_drift(None);
    let drift_status = if drift_report.is_drifting {
        "drifting".to_string()
    } else if drift_report.window_size == 0 {
        "no_data".to_string()
    } else {
        "healthy".to_string()
    };

    let emb = state.embedding_engine.read();

    // ADR-075: DP status
    let dp_engine = state.dp_engine.lock();
    let dp_budget_used = dp_engine.epsilon() / state.rvf_flags.dp_epsilon.max(1e-10);
    drop(dp_engine);

    // ── SONA: trigger background learning if due ──
    if state.rvf_flags.sona_enabled {
        if let Some(msg) = state.sona.read().tick() {
            tracing::info!("SONA background learning: {msg}");
        }
    }

    // ADR-075: average RVF segments per memory (reuse all_memories from above)
    let rvf_count = all_memories.iter().filter(|m| m.witness_chain.is_some()).count();
    let rvf_segments_per_memory = if rvf_count > 0 {
        // Estimate: memories with witness chains have at least 3 segments (VEC+META+WITNESS)
        // plus optional DP proof and redaction log
        let total_segs: usize = all_memories.iter().map(|m| {
            let mut s = 2; // VEC + META
            if m.witness_chain.is_some() { s += 1; }
            if m.dp_proof.is_some() { s += 1; }
            if m.redaction_log.is_some() { s += 1; }
            s
        }).sum();
        total_segs as f64 / all_memories.len().max(1) as f64
    } else {
        0.0
    };

    Json(StatusResponse {
        total_memories: state.store.memory_count(),
        total_contributors: state.store.contributor_count(),
        graph_nodes: graph.node_count(),
        graph_edges: graph.edge_count(),
        cluster_count,
        avg_quality,
        drift_status,
        lora_epoch: lora.epoch,
        lora_pending_submissions: lora.pending.len(),
        total_pages: state.store.page_count(),
        total_nodes: state.store.node_count(),
        total_votes: state.store.vote_count(),
        embedding_engine: emb.engine_name().to_string(),
        embedding_dim: emb.dim(),
        embedding_corpus: emb.corpus_size(),
        dp_epsilon: state.rvf_flags.dp_epsilon,
        dp_budget_used,
        rvf_segments_per_memory,
        gwt_workspace_load: state.workspace.read().current_load(),
        gwt_avg_salience: state.workspace.read().average_salience(),
        knowledge_velocity: {
            let ds = state.delta_stream.read();
            let now_ns = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let one_hour_ns = 3_600_000_000_000u64;
            ds.get_time_range(now_ns.saturating_sub(one_hour_ns), now_ns).len() as f64
        },
        temporal_deltas: state.delta_stream.read().len(),
        sona_patterns: {
            let ss = state.sona.read().stats();
            ss.patterns_stored
        },
        meta_avg_regret: state.domain_engine.read().meta.regret.average_regret(),
        meta_plateau_status: {
            let cp = state.domain_engine.read().meta.plateau.consecutive_plateaus;
            if cp == 0 { "learning".to_string() }
            else if cp <= 2 { format!("mild_plateau({})", cp) }
            else { format!("severe_plateau({})", cp) }
        },
        sona_trajectories: {
            let ss = state.sona.read().stats();
            ss.trajectories_buffered
        },
        midstream_scheduler_ticks: state.nano_scheduler.metrics().total_ticks,
        midstream_attractor_categories: state.attractor_results.read().len(),
        midstream_strange_loop_version: strange_loop::VERSION.to_string(),
        sparsifier_compression: graph.sparsifier_stats().map(|s| s.compression_ratio).unwrap_or(0.0),
        sparsifier_edges: graph.sparsifier_stats().map(|s| s.sparsified_edges).unwrap_or(0),
    })
}

/// GET /v1/sona/stats — SONA learning engine statistics (auth required)
async fn sona_stats(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<serde_json::Value> {
    let stats = state.sona.read().stats();
    Json(serde_json::json!({
        "patterns_stored": stats.patterns_stored,
        "trajectories_buffered": stats.trajectories_buffered,
        "trajectories_dropped": stats.trajectories_dropped,
        "buffer_success_rate": stats.buffer_success_rate,
        "ewc_tasks": stats.ewc_tasks,
        "instant_enabled": stats.instant_enabled,
        "background_enabled": stats.background_enabled,
        "sona_enabled": state.rvf_flags.sona_enabled,
    }))
}


/// GET /v1/explore — meta-learning exploration stats (ADR-075 AGI, auth required)
async fn explore_meta_learning(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<serde_json::Value> {
    let de = state.domain_engine.read();
    let health = de.meta.health_check();
    let regret = de.meta.regret.summary();

    // Find most curious category: check all registered brain categories
    let categories = ["architecture", "pattern", "solution", "convention",
                      "security", "performance", "tooling", "debug"];
    let mut best_cat = None;
    let mut best_novelty = 0.0f32;
    for cat in &categories {
        let bucket = ruvector_domain_expansion::ContextBucket {
            difficulty_tier: "default".into(),
            category: cat.to_string(),
        };
        let novelty = de.meta.curiosity.novelty_score(&bucket);
        if novelty > best_novelty {
            best_novelty = novelty;
            best_cat = Some(*cat);
        }
    }

    let plateau_status = if de.meta.plateau.consecutive_plateaus == 0 {
        "learning".to_string()
    } else if de.meta.plateau.consecutive_plateaus <= 2 {
        format!("mild_plateau({})", de.meta.plateau.consecutive_plateaus)
    } else {
        format!("severe_plateau({})", de.meta.plateau.consecutive_plateaus)
    };

    Json(serde_json::json!({
        "most_curious_category": best_cat,
        "most_curious_novelty": best_novelty,
        "regret_summary": {
            "total_regret": regret.total_regret,
            "average_regret": regret.average_regret,
            "mean_growth_rate": regret.mean_growth_rate,
            "converged_buckets": regret.converged_buckets,
            "bucket_count": regret.bucket_count,
            "total_observations": regret.total_observations
        },
        "plateau_status": plateau_status,
        "is_learning": health.is_learning,
        "is_diverse": health.is_diverse,
        "is_exploring": health.is_exploring,
        "curiosity_total_visits": health.curiosity_total_visits,
        "pareto_size": health.pareto_size
    }))
}
/// GET /v1/temporal — temporal delta tracking stats (ADR-075 AGI, auth required)
async fn temporal_stats(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<TemporalResponse> {
    let ds = state.delta_stream.read();
    let total_deltas = ds.len();

    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    let one_hour_ns = 3_600_000_000_000u64;
    let recent_hour_deltas = ds.get_time_range(now_ns.saturating_sub(one_hour_ns), now_ns).len();

    let knowledge_velocity = recent_hour_deltas as f64;

    let trend = if recent_hour_deltas > 10 {
        "growing".to_string()
    } else if recent_hour_deltas > 0 {
        "stable".to_string()
    } else {
        "idle".to_string()
    };

    Json(TemporalResponse {
        total_deltas,
        recent_hour_deltas,
        knowledge_velocity,
        trend,
    })
}

/// GET /v1/midstream — midstream platform diagnostics (ADR-077)
async fn midstream_stats(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<crate::midstream::MidstreamStatus> {
    Json(crate::midstream::collect_status(&state))
}

/// GET /v1/lora/latest — serve current consensus MicroLoRA weights
/// Cached for 60s (consensus changes only at epoch boundaries)
async fn lora_latest(
    State(state): State<AppState>,
) -> ([(axum::http::header::HeaderName, &'static str); 1], Json<LoraLatestResponse>) {
    let lora = state.lora_federation.read();
    (
        [(axum::http::header::CACHE_CONTROL, "public, max-age=60")],
        Json(LoraLatestResponse {
            weights: lora.consensus.clone(),
            epoch: lora.epoch,
        }),
    )
}

/// POST /v1/lora/submit — accept session LoRA weights for federation
async fn lora_submit(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Json(submission): Json<LoraSubmission>,
) -> Result<Json<LoraSubmitResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    // Rate limit: LoRA submissions count as writes
    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    // Gate A: policy validity
    submission.validate()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("LoRA validation failed: {e}")))?;

    // Get contributor reputation for weighted aggregation
    let reputation = state
        .store
        .get_contributor_reputation(&contributor.pseudonym)
        .map(|r| crate::reputation::ReputationManager::contribution_weight(&r))
        .unwrap_or(0.1);

    // All parking_lot guard operations in this sync block — no .await
    let (pending, epoch, lora_doc) = {
        let mut lora = state.lora_federation.write();

        // Dimension check: must match expected
        if submission.rank != lora.expected_rank || submission.hidden_dim != lora.expected_hidden_dim {
            return Err((StatusCode::BAD_REQUEST, format!(
                "Dimension mismatch: expected rank={} dim={}, got rank={} dim={}",
                lora.expected_rank, lora.expected_hidden_dim,
                submission.rank, submission.hidden_dim
            )));
        }

        lora.submit(submission, contributor.pseudonym.clone(), reputation);

        // Check for weight drift after aggregation
        if let Some(dist) = lora.consensus_drift() {
            if dist > 5.0 {
                tracing::warn!(
                    "LoRA consensus drift {dist:.2} exceeds threshold 5.0, rolling back"
                );
                lora.rollback();
            }
        }

        let pending = lora.pending.len();
        let epoch = lora.epoch;
        let doc = lora.consensus.as_ref().map(|c| {
            serde_json::json!({
                "epoch": lora.epoch,
                "consensus": c,
            })
        });
        (pending, epoch, doc)
    }; // All guards dropped here

    // Persist LoRA consensus to Firestore (no guards held)
    if let Some(doc) = lora_doc {
        state.store.firestore_put_public("brain_lora", "consensus", &doc).await;
    }

    Ok(Json(LoraSubmitResponse {
        accepted: true,
        pending_submissions: pending,
        current_epoch: epoch,
    }))
}

/// GET /v1/training/preferences — export preference pairs for DPO/reward model training
/// Layer A training data: vote events with embeddings and quality transitions
async fn training_preferences(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<TrainingQuery>,
) -> Json<TrainingPreferencesResponse> {
    let since = query.since_index.unwrap_or(0);
    let limit = query.limit.unwrap_or(100).min(1000);
    let (pairs, next_index) = state.store.get_preference_pairs(since, limit);
    Json(TrainingPreferencesResponse {
        pairs,
        next_index,
        total_votes: state.store.vote_count(),
    })
}

/// POST /v1/train — trigger an explicit training cycle (SONA + domain evolution)
async fn train_endpoint(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Result<Json<TrainingCycleResult>, (StatusCode, String)> {
    check_read_only(&state)?;
    let result = run_training_cycle(&state);
    tracing::info!(
        "Training cycle (explicit): sona_patterns={}, pareto={}→{}, memories={}",
        result.sona_patterns, result.pareto_before, result.pareto_after, result.memory_count
    );
    Ok(Json(result))
}

// ──────────────────────────────────────────────────────────────────────
// Cognitive Layer endpoints (ADR-110)
// ──────────────────────────────────────────────────────────────────────

/// GET /v1/cognitive/status — Full cognitive system status
async fn cognitive_status(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<serde_json::Value> {
    let voice = state.internal_voice.read();
    let ns = state.neural_symbolic.read();
    let sona = state.sona.read().stats();

    Json(serde_json::json!({
        "neural_layer": {
            "hopfield_patterns": "active",
            "sona_patterns": sona.patterns_stored,
            "sona_trajectories": sona.trajectories_buffered,
        },
        "internal_voice": {
            "thought_count": voice.thought_count(),
            "goal_depth": voice.goal_depth(),
            "working_memory_utilization": voice.working_memory_utilization(),
        },
        "symbolic_layer": {
            "propositions_count": ns.proposition_count(),
            "rule_count": ns.rule_count(),
            "extraction_count": ns.extraction_count(),
            "inference_count": ns.inference_count(),
        },
        "version": "ADR-110",
    }))
}

/// GET /v1/voice/working — Current working memory contents
async fn voice_working_memory(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<crate::voice::WorkingMemoryResponse> {
    let voice = state.internal_voice.read();
    let items: Vec<crate::voice::WorkingMemoryItemSummary> = voice
        .working_memory_items()
        .iter()
        .map(|item| crate::voice::WorkingMemoryItemSummary {
            id: item.id,
            content: item.content.clone(),
            activation: item.activation,
            source: item.source.clone(),
            last_accessed: item.last_accessed,
        })
        .collect();

    Json(crate::voice::WorkingMemoryResponse {
        utilization: voice.working_memory_utilization(),
        capacity: 7, // Miller's law default
        items,
    })
}

/// GET /v1/voice/history — Recent thought history
async fn voice_history(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<VoiceHistoryQuery>,
) -> Json<crate::voice::VoiceHistoryResponse> {
    let limit = query.limit.unwrap_or(20).min(100);
    let voice = state.internal_voice.read();

    let thoughts: Vec<crate::voice::VoiceToken> = voice
        .recent_thoughts(limit)
        .into_iter()
        .cloned()
        .collect();

    Json(crate::voice::VoiceHistoryResponse {
        thoughts,
        total_count: voice.thought_count(),
        goal_depth: voice.goal_depth(),
    })
}

#[derive(Debug, serde::Deserialize)]
struct VoiceHistoryQuery {
    limit: Option<usize>,
}

/// POST /v1/voice/goal — Set a deliberation goal
async fn voice_set_goal(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<crate::voice::SetGoalRequest>,
) -> Json<crate::voice::SetGoalResponse> {
    let priority = req.priority.unwrap_or(1.0);
    let goal_id = state.internal_voice.write().set_goal(req.description.clone(), priority);

    Json(crate::voice::SetGoalResponse {
        goal_id,
        description: req.description,
        priority,
    })
}

/// GET /v1/propositions — List extracted propositions
async fn list_propositions(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<PropositionsQuery>,
) -> Json<crate::symbolic::PropositionsResponse> {
    let ns = state.neural_symbolic.read();
    let limit = query.limit.unwrap_or(50).min(200);

    let propositions: Vec<crate::symbolic::GroundedProposition> = if let Some(ref pred) = query.predicate {
        ns.propositions_by_predicate(pred)
            .into_iter()
            .take(limit)
            .cloned()
            .collect()
    } else {
        ns.all_propositions()
            .into_iter()
            .take(limit)
            .cloned()
            .collect()
    };

    Json(crate::symbolic::PropositionsResponse {
        total_count: ns.proposition_count(),
        rule_count: ns.rule_count(),
        propositions,
    })
}

#[derive(Debug, serde::Deserialize)]
struct PropositionsQuery {
    predicate: Option<String>,
    limit: Option<usize>,
}

/// POST /v1/reason — Run neural-symbolic inference
async fn reason_endpoint(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<crate::symbolic::ReasonRequest>,
) -> Result<Json<crate::symbolic::ReasonResponse>, (StatusCode, String)> {
    let limit = req.limit.unwrap_or(5).min(20);

    // Get embedding for query
    let embedding = if let Some(ref emb) = req.embedding {
        emb.clone()
    } else {
        // Generate embedding from query text
        let emb_engine = state.embedding_engine.read();
        emb_engine.embed_for_storage(&req.query)
    };

    let ns = state.neural_symbolic.read();
    let inferences = ns.reason(&embedding, limit);
    let relevant = ns
        .all_propositions()
        .into_iter()
        .take(10)
        .cloned()
        .collect();

    // Record reasoning in internal voice
    drop(ns);
    {
        let mut voice = state.internal_voice.write();
        if !inferences.is_empty() {
            voice.conclude(
                format!("found {} inferences for query", inferences.len()),
                "reason_endpoint".to_string(),
            );
        } else {
            voice.express_uncertainty(format!("no inferences found for: {}", req.query));
        }
    }

    Ok(Json(crate::symbolic::ReasonResponse {
        inferences,
        relevant_propositions: relevant,
    }))
}

/// POST /v1/ground — Ground a new proposition
async fn ground_proposition(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<crate::symbolic::GroundRequest>,
) -> Result<Json<crate::symbolic::GroundResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    let prop = state.neural_symbolic.write().ground_proposition(
        req.predicate.clone(),
        req.arguments,
        req.embedding,
        req.evidence_ids,
    );

    // Record in internal voice
    state.internal_voice.write().observe(
        format!("grounded proposition: {}", req.predicate),
        prop.id,
    );

    Ok(Json(crate::symbolic::GroundResponse {
        proposition_id: prop.id,
        predicate: prop.predicate,
        confidence: prop.confidence,
    }))
}

/// POST /v1/train/enhanced — Trigger enhanced training cycle (ADR-110)
async fn train_enhanced_endpoint(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Result<Json<EnhancedTrainingResult>, (StatusCode, String)> {
    check_read_only(&state)?;
    let result = run_enhanced_training_cycle(&state);
    tracing::info!(
        "Enhanced training cycle: sona={}, propositions={}, voice_thoughts={}, rules={}",
        result.sona_patterns,
        result.propositions_extracted,
        result.voice_thoughts,
        result.rule_count
    );
    Ok(Json(result))
}

/// GET /v1/optimizer/status — Get Gemini optimizer status
async fn optimizer_status(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<crate::optimizer::OptimizerStatusResponse> {
    let optimizer = state.optimizer.read();
    Json(crate::optimizer::OptimizerStatusResponse {
        stats: optimizer.stats(),
        config: crate::optimizer::OptimizerConfig::default(), // Return default config for visibility
    })
}

/// POST /v1/optimize — Run Gemini Flash optimization
async fn optimize_endpoint(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<crate::optimizer::OptimizeRequest>,
) -> Json<crate::optimizer::OptimizeResponse> {
    let task = req.task.unwrap_or(crate::optimizer::OptimizationTask::RuleRefinement);

    // Build optimization context from current state
    let context = {
        let ns = state.neural_symbolic.read();
        let voice = state.internal_voice.read();
        let sona = state.sona.read().stats();

        let sample_props: Vec<crate::optimizer::PropositionSample> = ns
            .all_propositions()
            .into_iter()
            .take(10)
            .map(|p| crate::optimizer::PropositionSample {
                predicate: p.predicate.clone(),
                arguments: p.arguments.clone(),
                confidence: p.confidence,
                evidence_count: p.evidence.len(),
            })
            .collect();

        crate::optimizer::OptimizationContext {
            propositions: ns.proposition_count(),
            rules: ns.rule_count(),
            sona_patterns: sona.patterns_stored,
            working_memory_load: voice.working_memory_utilization(),
            thought_distribution: std::collections::HashMap::new(),
            sample_propositions: sample_props,
            memory_count: state.store.memory_count(),
        }
    };

    // Check if optimizer is configured (before taking write lock)
    let (is_configured, stats) = {
        let opt = state.optimizer.read();
        (opt.is_configured(), opt.stats())
    };

    if !is_configured {
        return Json(crate::optimizer::OptimizeResponse {
            result: None,
            error: Some("Gemini API key not configured".to_string()),
            stats,
        });
    }

    // Create a temporary optimizer for the async call to avoid holding lock across await
    let config = crate::optimizer::OptimizerConfig::default();
    let mut temp_optimizer = crate::optimizer::GeminiOptimizer::new(config);

    match temp_optimizer.optimize(task.clone(), context).await {
        Ok(result) => {
            // Record optimization in internal voice
            state.internal_voice.write().reflect(
                format!("Gemini optimization: {} suggestions", result.suggestions.len()),
            );

            // Update stats
            let stats = state.optimizer.read().stats();

            Json(crate::optimizer::OptimizeResponse {
                result: Some(result),
                error: None,
                stats,
            })
        }
        Err(e) => {
            tracing::warn!("Optimization failed: {}", e);
            let stats = state.optimizer.read().stats();
            Json(crate::optimizer::OptimizeResponse {
                result: None,
                error: Some(e),
                stats,
            })
        }
    }
}

// Cloud Pipeline endpoints (real-time injection + optimization)
// ──────────────────────────────────────────────────────────────────────

/// Core injection logic: PII strip, embed, witness chain, store, graph update.
/// Returns (InjectResponse, BrainMemory) on success. Shared by single inject,
/// batch inject, and Pub/Sub push handlers.
async fn process_inject(
    state: &AppState,
    req: InjectRequest,
) -> Result<InjectResponse, String> {
    use std::sync::atomic::Ordering;

    state.pipeline_metrics.messages_received.fetch_add(1, Ordering::Relaxed);

    // PII stripping
    let (title, content, tags, redaction_log_json) = if state.rvf_flags.pii_strip {
        let mut field_pairs: Vec<(&str, &str)> = vec![
            ("title", &req.title),
            ("content", &req.content),
        ];
        for tag in &req.tags {
            field_pairs.push(("tag", tag));
        }
        let (stripped, log) = state.verifier.write().strip_pii_fields(&field_pairs);
        let stripped_title = stripped[0].1.clone();
        let stripped_content = stripped[1].1.clone();
        let stripped_tags: Vec<String> = stripped[2..].iter().map(|(_, v)| v.clone()).collect();
        let log_json = serde_json::to_string(&log).ok();
        (stripped_title, stripped_content, stripped_tags, log_json)
    } else {
        (req.title, req.content, req.tags, None)
    };

    // Auto-generate embedding via ruvllm
    let text = crate::embeddings::EmbeddingEngine::prepare_text(&title, &content, &tags);
    let embedding = state.embedding_engine.read().embed_for_storage(&text);

    // Verify input
    state.verifier.read()
        .verify_share(&title, &content, &tags, &embedding)
        .map_err(|e| e.to_string())?;

    // Differential privacy noise on embedding
    let (embedding, dp_proof_json) = if state.rvf_flags.dp_enabled {
        let mut params: Vec<f64> = embedding.iter().map(|&v| v as f64).collect();
        let proof = state.dp_engine.lock().add_noise(&mut params);
        let noised: Vec<f32> = params.iter().map(|&v| v as f32).collect();
        let proof_json = serde_json::to_string(&proof).ok();
        (noised, proof_json)
    } else {
        (embedding, None)
    };

    // Witness chain
    let now_ns = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);

    let (witness_chain_bytes, witness_hash) = if state.rvf_flags.witness {
        let pii_data = format!("pii_strip:{}:{}", title, content);
        let mut emb_bytes = Vec::with_capacity(embedding.len() * 4);
        for v in &embedding {
            emb_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let content_data = format!("content:{}:{}:{}", title, content, tags.join(","));

        let entries = vec![
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(pii_data.as_bytes()),
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(&emb_bytes),
                timestamp_ns: now_ns,
                witness_type: 0x02,
            },
            rvf_crypto::WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(content_data.as_bytes()),
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
        ];
        let chain = rvf_crypto::create_witness_chain(&entries);
        let hash = hex::encode(rvf_crypto::shake256_256(&chain));
        (Some(chain), hash)
    } else {
        let mut data = Vec::new();
        data.extend_from_slice(b"ruvector-witness:");
        data.extend_from_slice(title.as_bytes());
        data.extend_from_slice(b":");
        data.extend_from_slice(content.as_bytes());
        let hash = hex::encode(rvf_crypto::shake256_256(&data));
        (None, hash)
    };

    // Ensure pipeline contributor exists
    let contributor_id = format!("pipeline:{}", req.source);
    state
        .store
        .get_or_create_contributor(&contributor_id, true)
        .await
        .map_err(|e| e.to_string())?;

    let id = Uuid::new_v4();
    let now = chrono::Utc::now();

    let memory = BrainMemory {
        id,
        category: req.category,
        title,
        content,
        tags,
        code_snippet: None,
        embedding: embedding.clone(),
        contributor_id: contributor_id.clone(),
        quality_score: BetaParams::new(),
        partition_id: None,
        witness_hash: witness_hash.clone(),
        rvf_gcs_path: None,
        redaction_log: redaction_log_json,
        dp_proof: dp_proof_json,
        witness_chain: witness_chain_bytes,
        created_at: now,
        updated_at: now,
    };

    // Add to embedding corpus
    state.embedding_engine.write().add_to_corpus(&id.to_string(), embedding.clone(), None);

    // Record in cognitive engine and drift monitor
    {
        let mut cog = state.cognitive.write();
        cog.store_pattern(&id.to_string(), &memory.embedding);
        let mut drift = state.drift.write();
        drift.record(&memory.category.to_string(), &memory.embedding);
    }

    // Temporal delta tracking
    if state.rvf_flags.temporal_enabled {
        let delta = ruvector_delta_core::VectorDelta::from_dense(embedding.clone());
        state.delta_stream.write().push_with_timestamp(delta, now_ns);
    }

    // Add to graph and count new edges
    let graph_edges_before;
    let graph_edges_after;
    {
        let mut graph = state.graph.write();
        graph_edges_before = graph.edge_count();
        graph.add_memory(&memory);
        graph_edges_after = graph.edge_count();
    }
    let graph_edges_added = graph_edges_after.saturating_sub(graph_edges_before);

    // Store in Firestore
    let quality_score = memory.quality_score.mean();
    state
        .store
        .store_memory(memory)
        .await
        .map_err(|e| e.to_string())?;

    // Update contributor activity
    state.store.record_contribution(&contributor_id).await;

    // SONA: Record injection as learning trajectory
    if state.rvf_flags.sona_enabled {
        let sona = state.sona.read();
        let emb_for_step = embedding.clone();
        let mut builder = sona.begin_trajectory(embedding.clone());
        builder.add_step(emb_for_step, vec![], 0.5);
        sona.end_trajectory(builder, 0.5);
    }

    state.pipeline_metrics.messages_processed.fetch_add(1, Ordering::Relaxed);
    *state.pipeline_metrics.last_injection.write() = Some(now);

    // Broadcast via SSE to active sessions
    if !state.sessions.is_empty() {
        let event_json = serde_json::json!({
            "type": "pipeline_inject",
            "id": id,
            "quality_score": quality_score,
        });
        let event_str = serde_json::to_string(&event_json).unwrap_or_default();
        for entry in state.sessions.iter() {
            let _ = entry.value().try_send(event_str.clone());
        }
    }

    Ok(InjectResponse {
        id,
        quality_score,
        witness_hash,
        graph_edges_added,
    })
}

/// POST /v1/pipeline/inject — inject a single item into the brain pipeline
async fn pipeline_inject(
    State(state): State<AppState>,
    Json(req): Json<InjectRequest>,
) -> Result<(StatusCode, Json<InjectResponse>), (StatusCode, String)> {
    check_read_only(&state)?;

    match process_inject(&state, req).await {
        Ok(resp) => Ok((StatusCode::CREATED, Json(resp))),
        Err(e) => {
            state.pipeline_metrics.messages_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Err((StatusCode::BAD_REQUEST, e))
        }
    }
}

/// POST /v1/pipeline/inject/batch — inject up to 100 items
async fn pipeline_inject_batch(
    State(state): State<AppState>,
    Json(req): Json<BatchInjectRequest>,
) -> Result<Json<BatchInjectResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    if req.items.len() > 100 {
        return Err((StatusCode::BAD_REQUEST, "Batch size exceeds maximum of 100 items".into()));
    }

    let mut accepted = 0usize;
    let mut rejected = 0usize;
    let mut memory_ids = Vec::new();
    let mut errors = Vec::new();

    for item in req.items {
        // Override source from batch envelope if item source is empty
        let item = InjectRequest {
            source: if item.source.is_empty() { req.source.clone() } else { item.source },
            ..item
        };

        match process_inject(&state, item).await {
            Ok(resp) => {
                accepted += 1;
                memory_ids.push(resp.id);
            }
            Err(e) => {
                rejected += 1;
                errors.push(e);
                state.pipeline_metrics.messages_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
        }
    }

    Ok(Json(BatchInjectResponse {
        accepted,
        rejected,
        memory_ids,
        errors,
    }))
}

/// POST /v1/pipeline/pubsub — receive Cloud Pub/Sub push messages.
/// No Bearer auth required (Cloud Run validates Pub/Sub OIDC tokens automatically).
async fn pipeline_pubsub_push(
    State(state): State<AppState>,
    Json(push): Json<PubSubPushMessage>,
) -> Result<StatusCode, (StatusCode, String)> {
    check_read_only(&state)?;

    // Decode base64 message data
    use base64::Engine as _;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(&push.message.data)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64 in Pub/Sub message: {e}")))?;

    // Deserialize as InjectRequest
    let inject_req: InjectRequest = serde_json::from_slice(&decoded)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid JSON in Pub/Sub message: {e}")))?;

    match process_inject(&state, inject_req).await {
        Ok(_) => {
            tracing::info!(
                "Pub/Sub message processed: messageId={}, subscription={}",
                push.message.message_id, push.subscription,
            );
            Ok(StatusCode::OK)
        }
        Err(e) => {
            state.pipeline_metrics.messages_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            tracing::warn!(
                "Pub/Sub message failed: messageId={}, error={}",
                push.message.message_id, e,
            );
            // Return 200 to avoid Pub/Sub retries for permanent failures.
            // Log the error for debugging.
            Ok(StatusCode::OK)
        }
    }
}

/// GET /v1/pipeline/metrics — pipeline health and throughput metrics
async fn pipeline_metrics_handler(
    State(state): State<AppState>,
) -> Json<PipelineMetricsResponse> {
    use std::sync::atomic::Ordering;

    let graph = state.graph.read();
    let received = state.pipeline_metrics.messages_received.load(Ordering::Relaxed);
    let processed = state.pipeline_metrics.messages_processed.load(Ordering::Relaxed);
    let failed = state.pipeline_metrics.messages_failed.load(Ordering::Relaxed);
    let opt_cycles = state.pipeline_metrics.optimization_cycles.load(Ordering::Relaxed);
    let uptime = state.start_time.elapsed().as_secs();

    let last_training = state.pipeline_metrics.last_training.read()
        .map(|dt| dt.to_rfc3339());
    let last_drift_check = state.pipeline_metrics.last_drift_check.read()
        .map(|dt| dt.to_rfc3339());

    // Calculate injections per minute from uptime
    let injections_per_minute = if uptime > 0 {
        processed as f64 / (uptime as f64 / 60.0)
    } else {
        0.0
    };

    Json(PipelineMetricsResponse {
        messages_received: received,
        messages_processed: processed,
        messages_failed: failed,
        memory_count: state.store.memory_count(),
        graph_nodes: graph.node_count(),
        graph_edges: graph.edge_count(),
        last_training,
        last_drift_check,
        optimization_cycles: opt_cycles,
        uptime_seconds: uptime,
        injections_per_minute,
    })
}

/// POST /v1/pipeline/optimize — trigger optimization actions
async fn pipeline_optimize(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    let all_actions = vec![
        "train", "drift_check", "transfer_all", "rebuild_graph", "cleanup", "attractor_analysis",
    ];
    let actions: Vec<&str> = match &req.actions {
        Some(a) => a.iter().map(|s| s.as_str()).collect(),
        None => all_actions,
    };

    let total_start = std::time::Instant::now();
    let mut results = Vec::new();

    for action in &actions {
        let action_start = std::time::Instant::now();
        let (success, message) = match *action {
            "train" => {
                let result = run_training_cycle(&state);
                *state.pipeline_metrics.last_training.write() = Some(chrono::Utc::now());
                (true, format!(
                    "Training complete: sona_patterns={}, pareto={}->{}",
                    result.sona_patterns, result.pareto_before, result.pareto_after,
                ))
            }
            "drift_check" => {
                let drift = state.drift.read();
                let report = drift.compute_drift(None);
                *state.pipeline_metrics.last_drift_check.write() = Some(chrono::Utc::now());
                (true, format!(
                    "Drift check: drifting={}, cv={:.4}, trend={}",
                    report.is_drifting, report.coefficient_of_variation, report.trend,
                ))
            }
            "transfer_all" => {
                use ruvector_domain_expansion::DomainId;
                let categories: Vec<String> = {
                    let all_mems = state.store.all_memories();
                    let mut cats: std::collections::HashSet<String> = std::collections::HashSet::new();
                    for m in &all_mems {
                        cats.insert(m.category.to_string());
                    }
                    cats.into_iter().collect()
                };
                let mut transfers = 0usize;
                let mut engine = state.domain_engine.write();
                for i in 0..categories.len() {
                    for j in (i + 1)..categories.len() {
                        engine.initiate_transfer(
                            &DomainId(categories[i].clone()),
                            &DomainId(categories[j].clone()),
                        );
                        transfers += 1;
                    }
                }
                (true, format!("Domain transfers initiated: {transfers} pairs across {} categories", categories.len()))
            }
            "rebuild_graph" => {
                let all_mems = state.store.all_memories();
                let mut graph = state.graph.write();
                *graph = crate::graph::KnowledgeGraph::new();
                for mem in &all_mems {
                    graph.add_memory(mem);
                }
                graph.rebuild_sparsifier();
                (true, format!("Graph rebuilt: {} nodes, {} edges", graph.node_count(), graph.edge_count()))
            }
            "cleanup" => {
                // Trigger SONA garbage collection and nonce cleanup
                if state.rvf_flags.sona_enabled {
                    let _ = state.sona.read().tick();
                }
                (true, "Cleanup complete".into())
            }
            "attractor_analysis" => {
                if state.rvf_flags.midstream_attractor {
                    let all_mems = state.store.all_memories();
                    let mut categories: std::collections::HashMap<String, Vec<Vec<f32>>> =
                        std::collections::HashMap::new();
                    for m in &all_mems {
                        categories.entry(m.category.to_string())
                            .or_default()
                            .push(m.embedding.clone());
                    }
                    let mut analyzed = 0usize;
                    for (cat, embeddings) in &categories {
                        if let Some(result) = crate::midstream::analyze_category_attractor(embeddings) {
                            state.attractor_results.write().insert(cat.clone(), result);
                            analyzed += 1;
                        }
                    }
                    (true, format!("Attractor analysis: {analyzed}/{} categories analyzed", categories.len()))
                } else {
                    (false, "Midstream attractor feature not enabled".into())
                }
            }
            other => {
                (false, format!("Unknown action: {other}"))
            }
        };

        results.push(OptimizeActionResult {
            action: action.to_string(),
            success,
            message,
            duration_ms: action_start.elapsed().as_millis() as u64,
        });
    }

    state.pipeline_metrics.optimization_cycles.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    Ok(Json(OptimizeResponse {
        results,
        total_duration_ms: total_start.elapsed().as_millis() as u64,
    }))
}

/// POST /v1/pipeline/feeds — add a new RSS/Atom feed configuration
async fn pipeline_add_feed(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(feed): Json<FeedConfig>,
) -> Result<(StatusCode, Json<FeedConfig>), (StatusCode, String)> {
    if feed.url.is_empty() || feed.name.is_empty() {
        return Err((StatusCode::BAD_REQUEST, "Feed url and name are required".into()));
    }
    if feed.poll_interval_secs < 60 {
        return Err((StatusCode::BAD_REQUEST, "poll_interval_secs must be >= 60".into()));
    }

    let key = feed.name.clone();
    let resp = feed.clone();
    state.feeds.insert(key, feed);
    Ok((StatusCode::CREATED, Json(resp)))
}

/// GET /v1/pipeline/feeds — list all configured feed sources
async fn pipeline_list_feeds(
    State(state): State<AppState>,
) -> Json<Vec<FeedConfig>> {
    let feeds: Vec<FeedConfig> = state.feeds.iter()
        .map(|entry| entry.value().clone())
        .collect();
    Json(feeds)
}

/// GET /v1/pipeline/scheduler/status — nanosecond scheduler status
async fn pipeline_scheduler_status(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    use std::sync::atomic::Ordering;

    let received = state.pipeline_metrics.messages_received.load(Ordering::Relaxed);
    let processed = state.pipeline_metrics.messages_processed.load(Ordering::Relaxed);
    let failed = state.pipeline_metrics.messages_failed.load(Ordering::Relaxed);
    let queue_depth = received.saturating_sub(processed).saturating_sub(failed);
    let uptime = state.start_time.elapsed().as_secs();
    let feeds_count = state.feeds.len();

    Json(serde_json::json!({
        "scheduler": "active",
        "queue_depth": queue_depth,
        "feeds_configured": feeds_count,
        "uptime_seconds": uptime,
        "midstream_scheduler_enabled": state.rvf_flags.midstream_scheduler,
        "nano_scheduler_ticks": state.nano_scheduler.metrics().total_ticks,
    }))
}

// ──────────────────────────────────────────────────────────────────────
// Common Crawl Integration (ADR-096 §10)
// ──────────────────────────────────────────────────────────────────────

/// Request to discover pages from Common Crawl
#[derive(Debug, serde::Deserialize)]
struct CrawlDiscoverRequest {
    /// URL pattern to query (e.g., "arxiv.org/abs/*")
    domain_pattern: String,
    /// Optional crawl index (e.g., "CC-MAIN-2026-13")
    crawl_index: Option<String>,
    /// Maximum pages to fetch
    limit: Option<usize>,
    /// Category for injected memories
    category: Option<String>,
    /// Tags for injected memories
    #[serde(default)]
    tags: Vec<String>,
    /// If true, actually inject into brain; if false, just preview
    #[serde(default)]
    inject: bool,
}

/// Response from Common Crawl discovery
#[derive(Debug, serde::Serialize)]
struct CrawlDiscoverResponse {
    cdx_records_found: usize,
    pages_fetched: usize,
    pages_injected: usize,
    duplicates_skipped: usize,
    errors: usize,
    memory_ids: Vec<uuid::Uuid>,
    preview_titles: Vec<String>,
}

/// POST /v1/pipeline/crawl/discover — query CDX + fetch pages from Common Crawl
async fn pipeline_crawl_discover(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<CrawlDiscoverRequest>,
) -> Result<Json<CrawlDiscoverResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    // Use persistent adapter from AppState (ADR-115)
    let cc = &state.crawl_adapter;
    if let Some(ref idx) = req.crawl_index {
        cc.set_crawl_index(idx).await;
    }

    let limit = req.limit.unwrap_or(20).min(100);
    let query = crate::pipeline::CdxQuery {
        url_pattern: req.domain_pattern.clone(),
        crawl_index: req.crawl_index.clone(),
        limit,
        ..Default::default()
    };

    // Query CDX index
    let records = cc.query_cdx(&query).await.map_err(|e| {
        (StatusCode::BAD_GATEWAY, format!("CDX query failed: {e}"))
    })?;
    let cdx_records_found = records.len();

    // Fetch pages using pre-queried records (avoids double CDX query)
    let items = cc.discover_from_records(
        &records,
        req.category.clone(),
        req.tags.clone(),
        limit,
    ).await.map_err(|e| {
        (StatusCode::INTERNAL_SERVER_ERROR, format!("Discovery failed: {e}"))
    })?;

    let pages_fetched = items.len();
    let (_, _, _, dupes, errors) = cc.stats();
    let preview_titles: Vec<String> = items.iter().take(10).map(|i| i.title.clone()).collect();

    // Optionally inject into brain
    let mut memory_ids = Vec::new();
    let mut pages_injected = 0usize;
    if req.inject {
        for item in &items {
            let category = match item.category.as_deref() {
                Some("architecture") => crate::types::BrainCategory::Architecture,
                Some("solution") => crate::types::BrainCategory::Solution,
                Some("convention") => crate::types::BrainCategory::Convention,
                Some("security") => crate::types::BrainCategory::Security,
                Some("performance") => crate::types::BrainCategory::Performance,
                Some("tooling") => crate::types::BrainCategory::Tooling,
                Some("debug") => crate::types::BrainCategory::Debug,
                _ => crate::types::BrainCategory::Pattern,
            };
            let inject_req = crate::types::InjectRequest {
                source: "common_crawl".into(),
                title: item.title.clone(),
                content: item.content.clone(),
                tags: item.tags.clone(),
                category,
                metadata: Some(serde_json::json!(item.metadata)),
            };
            match process_inject(&state, inject_req).await {
                Ok(resp) => {
                    memory_ids.push(resp.id);
                    pages_injected += 1;
                }
                Err(e) => {
                    tracing::warn!("Common Crawl inject failed: {}", e);
                }
            }
        }
    }

    Ok(Json(CrawlDiscoverResponse {
        cdx_records_found,
        pages_fetched,
        pages_injected,
        duplicates_skipped: dupes as usize,
        errors: errors as usize,
        memory_ids,
        preview_titles,
    }))
}

/// GET /v1/pipeline/crawl/stats — Common Crawl adapter statistics (ADR-115)
async fn pipeline_crawl_stats(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    // Use persistent adapter from AppState (ADR-115)
    let cc = &state.crawl_adapter;
    let (queries, fetched, extracted, dupes, errors) = cc.stats();
    let (cache_hits, cache_misses, cache_entries) = cc.cache_stats();

    // Include WebMemoryStore stats
    let web_status = state.web_store.status();

    // Calculate cache hit rate
    let total_cache_ops = cache_hits + cache_misses;
    let cache_hit_rate = if total_cache_ops > 0 {
        cache_hits as f64 / total_cache_ops as f64
    } else {
        0.0
    };

    Json(serde_json::json!({
        "adapter": "common_crawl",
        "cdx_queries": queries,
        "cdx_cache": {
            "hits": cache_hits,
            "misses": cache_misses,
            "entries": cache_entries,
            "hit_rate": format!("{:.1}%", cache_hit_rate * 100.0),
        },
        "pages_fetched": fetched,
        "pages_extracted": extracted,
        "duplicates_skipped": dupes,
        "errors": errors,
        "seen_urls": cc.seen_urls_count(),
        "seen_hashes": cc.seen_hashes_count(),
        "web_memory": {
            "total_memories": web_status.total_web_memories,
            "total_domains": web_status.total_domains,
            "link_edges": web_status.total_link_edges,
            "page_deltas": web_status.total_page_deltas,
            "compression_ratio": web_status.compression_ratio,
            "tier_distribution": {
                "full": web_status.tier_distribution.full,
                "delta_compressed": web_status.tier_distribution.delta_compressed,
                "centroid_merged": web_status.tier_distribution.centroid_merged,
                "archived": web_status.tier_distribution.archived,
            }
        }
    }))
}

/// GET /v1/pipeline/crawl/test — Test CDX connectivity (diagnostic)
async fn pipeline_crawl_test(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let adapter = &state.crawl_adapter;
    let test_url = "https://index.commoncrawl.org/collinfo.json";

    // Test Common Crawl using our configured HTTP client (native-tls, HTTP/1.1, with retry)
    let start = std::time::Instant::now();
    let (success, status, body_len, error, attempts) = adapter.test_connectivity().await;
    let latency_ms = start.elapsed().as_millis();

    let cc_result = if success {
        serde_json::json!({
            "success": true,
            "url": test_url,
            "status": status,
            "body_length": body_len,
            "latency_ms": latency_ms,
            "attempts": attempts,
        })
    } else {
        serde_json::json!({
            "success": false,
            "url": test_url,
            "status": status,
            "error": error,
            "latency_ms": latency_ms,
            "attempts": attempts,
        })
    };

    // Test multiple HTTPS endpoints for comparison
    let start2 = std::time::Instant::now();
    let external_results = adapter.test_external_connectivity().await;
    let latency_ms2 = start2.elapsed().as_millis();

    let external_tests: Vec<serde_json::Value> = external_results.iter().map(|(name, success, status, body_len, error, url)| {
        if *success {
            serde_json::json!({
                "name": name,
                "success": true,
                "url": url,
                "status": status,
                "body_length": body_len,
            })
        } else {
            serde_json::json!({
                "name": name,
                "success": false,
                "url": url,
                "status": status,
                "error": error,
            })
        }
    }).collect();

    let adapter_status = serde_json::json!({
        "adapter_queries": adapter.stats().0,
        "cache_stats": adapter.cache_stats(),
    });

    Json(serde_json::json!({
        "common_crawl_cdx_test": cc_result,
        "external_tests": external_tests,
        "external_tests_latency_ms": latency_ms2,
        "adapter_status": adapter_status,
    }))
}

// ──────────────────────────────────────────────────────────────────────
// Brainpedia endpoints (ADR-062)
// ──────────────────────────────────────────────────────────────────────

/// GET /v1/pages — list Brainpedia pages with pagination
#[derive(Debug, serde::Deserialize)]
struct ListPagesQuery {
    limit: Option<usize>,
    offset: Option<usize>,
    status: Option<String>,
}

async fn list_pages(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Query(query): Query<ListPagesQuery>,
) -> Json<ListPagesResponse> {
    let limit = query.limit.unwrap_or(20).min(100);
    let offset = query.offset.unwrap_or(0);

    let (page_ids, _total_count) = state.store.list_pages(limit + offset, 0);
    let status_filter = query.status.as_deref();

    let mut summaries: Vec<PageSummary> = Vec::new();
    for id in &page_ids {
        let page_status = match state.store.get_page_status(id) {
            Some(s) => s,
            None => continue,
        };
        // Apply status filter if provided
        if let Some(filter) = status_filter {
            let status_str = page_status.to_string();
            if status_str != filter {
                continue;
            }
        }
        if let Ok(Some(mem)) = state.store.get_memory(id).await {
            let deltas = state.store.get_deltas(id);
            let evidence = state.store.get_evidence(id);
            summaries.push(PageSummary {
                id: *id,
                title: mem.title,
                category: mem.category,
                status: page_status,
                quality_score: mem.quality_score.mean(),
                delta_count: deltas.len() as u32,
                evidence_count: evidence.len() as u32,
                updated_at: mem.updated_at,
            });
        }
    }

    // Sort by updated_at descending
    summaries.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));

    let total_filtered = summaries.len();
    let paginated: Vec<PageSummary> = summaries.into_iter().skip(offset).take(limit).collect();

    Json(ListPagesResponse {
        pages: paginated,
        total_count: total_filtered,
        offset,
        limit,
    })
}

/// POST /v1/pages — create a new Brainpedia page (Draft)
/// Requires reputation >= 0.5 and contribution_count >= 10 (unless system)
async fn create_page(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Json(req): Json<CreatePageRequest>,
) -> Result<(StatusCode, Json<PageResponse>), (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    // Auto-generate embedding via ruvllm if client didn't provide one or dim mismatches
    let embedding = if req.embedding.is_empty()
        || req.embedding.len() != crate::embeddings::EMBED_DIM
    {
        let text = crate::embeddings::EmbeddingEngine::prepare_text(&req.title, &req.content, &req.tags);
        let emb = state.embedding_engine.read().embed_for_storage(&text);
        tracing::debug!("Auto-generated {}-dim embedding for page '{}'", emb.len(), req.title);
        emb
    } else {
        req.embedding
    };

    // Verify input
    state.verifier.read()
        .verify_share(&req.title, &req.content, &req.tags, &embedding)
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    // Get or create contributor
    let contrib_info = state
        .store
        .get_or_create_contributor(&contributor.pseudonym, contributor.is_system)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    // Reputation gate: new users cannot create pages
    if !contrib_info.is_system
        && (contrib_info.reputation.composite < 0.5 || contrib_info.contribution_count < 10)
    {
        return Err((
            StatusCode::FORBIDDEN,
            "Page creation requires reputation >= 0.5 and contribution_count >= 10. Submit deltas to existing pages to build reputation.".into(),
        ));
    }

    let id = Uuid::new_v4();
    let now = chrono::Utc::now();

    // System contributors can create directly as Canonical
    let initial_status = if contrib_info.is_system {
        PageStatus::Canonical
    } else {
        PageStatus::Draft
    };

    // Auto-generate witness hash if not provided
    let witness_hash = if req.witness_hash.is_empty() {
        let mut data = Vec::new();
        data.extend_from_slice(b"ruvector-witness:");
        data.extend_from_slice(req.title.as_bytes());
        data.extend_from_slice(b":");
        data.extend_from_slice(req.content.as_bytes());
        hex::encode(rvf_crypto::shake256_256(&data))
    } else {
        req.witness_hash
    };

    let memory = BrainMemory {
        id,
        category: req.category,
        title: req.title,
        content: req.content,
        tags: req.tags,
        code_snippet: req.code_snippet,
        embedding,
        contributor_id: contributor.pseudonym.clone(),
        quality_score: BetaParams::new(),
        partition_id: None,
        witness_hash,
        rvf_gcs_path: None,
        redaction_log: None,
        dp_proof: None,
        witness_chain: None,
        created_at: now,
        updated_at: now,
    };

    // Add to graph
    {
        let mut graph = state.graph.write();
        graph.add_memory(&memory);
    }

    let evidence_count = req.evidence_links.len() as u32;

    state
        .store
        .create_page(memory, initial_status.clone(), req.evidence_links)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok((
        StatusCode::CREATED,
        Json(PageResponse {
            id,
            status: initial_status,
            quality_score: BetaParams::new().mean(),
            evidence_count,
            delta_count: 0,
        }),
    ))
}

/// GET /v1/pages/{id} — get a page with its delta log and evidence
async fn get_page(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(id): Path<Uuid>,
) -> Result<Json<PageDetailResponse>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    let memory = state
        .store
        .get_memory(&id)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
        .ok_or((StatusCode::NOT_FOUND, "Page not found".into()))?;

    let status = state
        .store
        .get_page_status(&id)
        .ok_or((StatusCode::NOT_FOUND, "Not a Brainpedia page".into()))?;

    let deltas = state.store.get_deltas(&id);
    let evidence = state.store.get_evidence(&id);

    Ok(Json(PageDetailResponse {
        memory,
        status,
        evidence_count: evidence.len() as u32,
        delta_count: deltas.len() as u32,
        deltas,
        evidence_links: evidence,
    }))
}

/// POST /v1/pages/{id}/deltas — submit a delta to a page
/// Requires authentication and at least one evidence link (except for Evidence deltas)
async fn submit_delta(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(page_id): Path<Uuid>,
    Json(req): Json<SubmitDeltaRequest>,
) -> Result<(StatusCode, Json<PageResponse>), (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    // Check contributor reputation: poisoned contributors blocked
    let contrib_info = state
        .store
        .get_or_create_contributor(&contributor.pseudonym, contributor.is_system)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if contrib_info.reputation.composite < 0.1 {
        return Err((
            StatusCode::FORBIDDEN,
            "Contributor reputation too low to submit deltas".into(),
        ));
    }

    // Evidence gate: non-Evidence deltas require at least one evidence link
    if req.delta_type != crate::types::DeltaType::Evidence && req.evidence_links.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "Deltas of type Correction, Extension, or Deprecation require at least one evidence link".into(),
        ));
    }

    // Verify page exists
    let page_status = state
        .store
        .get_page_status(&page_id)
        .ok_or((StatusCode::NOT_FOUND, "Page not found".into()))?;

    // Cannot submit deltas to Archived pages
    if page_status == PageStatus::Archived {
        return Err((StatusCode::FORBIDDEN, "Cannot modify archived pages".into()));
    }

    // Compute witness hash if not provided
    let witness_hash = if req.witness_hash.is_empty() {
        // Fallback: compute witness hash from content_diff
        let mut data = Vec::new();
        data.extend_from_slice(b"ruvector-delta-witness:");
        data.extend_from_slice(page_id.to_string().as_bytes());
        data.extend_from_slice(b":");
        data.extend_from_slice(req.content_diff.to_string().as_bytes());
        hex::encode(rvf_crypto::shake256_256(&data))
    } else {
        req.witness_hash
    };

    let delta = PageDelta {
        id: Uuid::new_v4(),
        page_id,
        delta_type: req.delta_type,
        content_diff: req.content_diff,
        evidence_links: req.evidence_links,
        contributor_id: contributor.pseudonym.clone(),
        quality_score: BetaParams::new(),
        witness_hash,
        created_at: chrono::Utc::now(),
    };

    state
        .store
        .submit_delta(&page_id, delta)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (evidence_count, delta_count) = state.store.page_counts(&page_id);

    // Compute actual quality from memory
    let page_quality = state
        .store
        .get_memory(&page_id)
        .await
        .ok()
        .flatten()
        .map(|m| m.quality_score.mean())
        .unwrap_or(BetaParams::new().mean());

    Ok((
        StatusCode::CREATED,
        Json(PageResponse {
            id: page_id,
            status: page_status,
            quality_score: page_quality,
            evidence_count,
            delta_count,
        }),
    ))
}

/// GET /v1/pages/{id}/deltas — list deltas for a page
async fn list_deltas(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(page_id): Path<Uuid>,
) -> Result<Json<Vec<PageDelta>>, (StatusCode, String)> {
    if !state.rate_limiter.check_read(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Read rate limit exceeded".into()));
    }

    if state.store.get_page_status(&page_id).is_none() {
        return Err((StatusCode::NOT_FOUND, "Page not found".into()));
    }

    Ok(Json(state.store.get_deltas(&page_id)))
}

/// POST /v1/pages/{id}/evidence — add evidence to a page
async fn add_evidence(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(page_id): Path<Uuid>,
    Json(req): Json<AddEvidenceRequest>,
) -> Result<Json<PageResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    let page_status = state
        .store
        .get_page_status(&page_id)
        .ok_or((StatusCode::NOT_FOUND, "Page not found".into()))?;

    let mut evidence = req.evidence;
    evidence.contributor_id = contributor.pseudonym.clone();
    evidence.created_at = chrono::Utc::now();

    let evidence_count = state
        .store
        .add_evidence(&page_id, evidence)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let (_, delta_count) = state.store.page_counts(&page_id);

    // Compute actual quality from memory
    let evidence_quality = state
        .store
        .get_memory(&page_id)
        .await
        .ok()
        .flatten()
        .map(|m| m.quality_score.mean())
        .unwrap_or(BetaParams::new().mean());

    Ok(Json(PageResponse {
        id: page_id,
        status: page_status,
        quality_score: evidence_quality,
        evidence_count,
        delta_count,
    }))
}

/// POST /v1/pages/{id}/promote — promote a Draft page to Canonical
/// Requires consensus: quality >= 0.7, observations >= 5, evidence >= 3 from >= 2 contributors
async fn promote_page(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Path(page_id): Path<Uuid>,
) -> Result<Json<PageResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    let new_status = state
        .store
        .promote_page(&page_id)
        .await
        .map_err(|e| match e {
            crate::store::StoreError::NotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
            _ => (StatusCode::BAD_REQUEST, e.to_string()),
        })?;

    let (evidence_count, delta_count) = state.store.page_counts(&page_id);

    // Get actual quality from promoted memory
    let promote_quality = state
        .store
        .get_memory(&page_id)
        .await
        .ok()
        .flatten()
        .map(|m| m.quality_score.mean())
        .unwrap_or(0.7);

    Ok(Json(PageResponse {
        id: page_id,
        status: new_status,
        quality_score: promote_quality,
        evidence_count,
        delta_count,
    }))
}

// ──────────────────────────────────────────────────────────────────────
// WASM Executable Nodes (ADR-063)
// ──────────────────────────────────────────────────────────────────────

/// GET /v1/nodes — list all published (non-revoked) WASM nodes (public)
async fn list_nodes(
    State(state): State<AppState>,
) -> Json<Vec<WasmNodeSummary>> {
    let nodes = state.store.list_nodes();
    Json(nodes.iter().filter(|n| !n.revoked).map(WasmNodeSummary::from).collect())
}

/// GET /v1/nodes/{id} — get node metadata + conformance vectors (public)
async fn get_node(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<WasmNode>, (StatusCode, String)> {
    let node = state
        .store
        .get_node(&id)
        .ok_or((StatusCode::NOT_FOUND, format!("Node {id} not found")))?;

    if node.revoked {
        return Err((StatusCode::GONE, format!("Node {id} has been revoked")));
    }

    Ok(Json(node))
}

/// GET /v1/nodes/{id}.wasm — download WASM binary with immutable cache headers (public)
async fn get_node_wasm(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<
    (
        StatusCode,
        [(axum::http::header::HeaderName, String); 3],
        Vec<u8>,
    ),
    (StatusCode, String),
> {
    // Strip .wasm suffix if present (route captures with it)
    let node_id = id.strip_suffix(".wasm").unwrap_or(&id);

    let node = state
        .store
        .get_node(node_id)
        .ok_or((StatusCode::NOT_FOUND, format!("Node {node_id} not found")))?;

    if node.revoked {
        return Err((StatusCode::GONE, format!("Node {node_id} has been revoked")));
    }

    let binary = state
        .store
        .get_node_binary(node_id)
        .ok_or((StatusCode::NOT_FOUND, "WASM binary not found".into()))?;

    Ok((
        StatusCode::OK,
        [
            (
                axum::http::header::CONTENT_TYPE,
                "application/wasm".to_string(),
            ),
            (
                axum::http::header::CACHE_CONTROL,
                "public, immutable, max-age=31536000".to_string(),
            ),
            (
                axum::http::header::HeaderName::from_static("x-node-sha256"),
                node.sha256.clone(),
            ),
        ],
        binary,
    ))
}

/// POST /v1/nodes — publish a new WASM node (requires reputation >= 0.5)
async fn publish_node(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Json(req): Json<PublishNodeRequest>,
) -> Result<(StatusCode, Json<WasmNodeSummary>), (StatusCode, String)> {
    check_read_only(&state)?;

    if !state.rate_limiter.check_write(&contributor.pseudonym) {
        return Err((StatusCode::TOO_MANY_REQUESTS, "Write rate limit exceeded".into()));
    }

    // Reputation gate
    let contrib_info = state
        .store
        .get_or_create_contributor(&contributor.pseudonym, contributor.is_system)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    if !contrib_info.is_system && contrib_info.reputation.composite < 0.5 {
        return Err((
            StatusCode::FORBIDDEN,
            "Node publishing requires reputation >= 0.5".into(),
        ));
    }

    // Decode WASM binary
    let wasm_bytes = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        &req.wasm_bytes,
    )
    .map_err(|e| (StatusCode::BAD_REQUEST, format!("Invalid base64: {e}")))?;

    // Size limit
    if wasm_bytes.len() > 1_048_576 {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, "WASM binary exceeds 1MB".into()));
    }

    // WASM magic bytes verification: \0asm (0x00 0x61 0x73 0x6D)
    if wasm_bytes.len() < 8 || &wasm_bytes[..4] != b"\0asm" {
        return Err((
            StatusCode::BAD_REQUEST,
            "Invalid WASM binary: missing magic bytes (\\0asm)".into(),
        ));
    }

    // V1 ABI: required exports
    let v1_required = ["memory", "malloc", "feature_extract_dim", "feature_extract"];
    for r in &v1_required {
        if !req.exports.contains(&r.to_string()) {
            return Err((
                StatusCode::BAD_REQUEST,
                format!("V1 ABI requires export: {r}"),
            ));
        }
    }

    // Compute and verify SHA-256
    use sha2::{Digest, Sha256};
    let sha256 = hex::encode(Sha256::digest(&wasm_bytes));

    // If client provided a sha256 claim, verify it matches the computed hash
    if let Some(ref claimed_hash) = req.sha256 {
        if !claimed_hash.is_empty() {
            // Constant-time comparison to prevent timing attacks
            let equal = subtle::ConstantTimeEq::ct_eq(
                sha256.as_bytes(),
                claimed_hash.to_lowercase().as_bytes(),
            );
            if !bool::from(equal) {
                return Err((
                    StatusCode::BAD_REQUEST,
                    format!(
                        "SHA-256 mismatch: computed {sha256}, claimed {claimed_hash}"
                    ),
                ));
            }
        }
    }

    let node = WasmNode {
        id: req.id.clone(),
        name: req.name,
        version: req.version,
        abi_version: 1,
        dim: req.dim.unwrap_or(128),
        sha256,
        size_bytes: wasm_bytes.len(),
        exports: req.exports,
        contributor_id: contributor.pseudonym.clone(),
        interface: req.interface,
        conformance: req.conformance,
        compiler_tag: req.compiler_tag.unwrap_or_default(),
        revoked: false,
        created_at: chrono::Utc::now(),
    };

    let summary = WasmNodeSummary::from(&node);

    state
        .store
        .publish_node(node, wasm_bytes)
        .await
        .map_err(|e| (StatusCode::CONFLICT, e.to_string()))?;

    Ok((StatusCode::CREATED, Json(summary)))
}

/// POST /v1/nodes/{id}/revoke — revoke a node (original publisher only)
/// Marks as revoked but retains bytes for forensic analysis
async fn revoke_node(
    State(state): State<AppState>,
    contributor: AuthenticatedContributor,
    Path(id): Path<String>,
) -> Result<StatusCode, (StatusCode, String)> {
    check_read_only(&state)?;

    state
        .store
        .revoke_node(&id, &contributor.pseudonym)
        .await
        .map_err(|e| match e {
            crate::store::StoreError::NotFound(_) => (StatusCode::NOT_FOUND, e.to_string()),
            crate::store::StoreError::Forbidden(_) => (StatusCode::FORBIDDEN, e.to_string()),
            _ => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        })?;

    Ok(StatusCode::NO_CONTENT)
}

/// Serve the landing page (embedded at compile time)
async fn landing_page() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=300"),
        ],
        include_str!("../static/index.html"),
    )
}

/// Serve robots.txt
async fn robots_txt() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/plain; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        include_str!("../static/robots.txt"),
    )
}

/// Serve sitemap.xml
async fn sitemap_xml() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "application/xml; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        include_str!("../static/sitemap.xml"),
    )
}

/// Serve OG image (SVG)
async fn og_image() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "image/svg+xml"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=604800"),
        ],
        include_str!("../static/og-image.svg"),
    )
}

/// Serve brain manifest (JSON)
async fn brain_manifest() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "application/json; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        include_str!("../static/brain-manifest.json"),
    )
}

/// Serve agent guide (Markdown)
async fn agent_guide() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/markdown; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=3600"),
        ],
        include_str!("../static/agent-guide.md"),
    )
}

/// Serve the origin story page
async fn origin_page() -> (
    StatusCode,
    [(axum::http::header::HeaderName, &'static str); 2],
    &'static str,
) {
    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "public, max-age=300"),
        ],
        include_str!("../static/origin.html"),
    )
}

// ══════════════════════════════════════════════════════════════════════
// MCP SSE Transport — Hosted SSE endpoint for Claude Code integration
//
// Protocol: MCP over SSE (Server-Sent Events)
//   1. Client GETs /sse → receives SSE stream with endpoint event
//   2. Client POSTs JSON-RPC to /messages?sessionId=<id>
//   3. Server responds through the SSE stream
//
// Usage: claude mcp add π --url https://pi.ruv.io/sse
// ══════════════════════════════════════════════════════════════════════

/// SSE handler — client connects here, receives event stream
async fn sse_handler(
    State(state): State<AppState>,
) -> Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>> {
    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

    // Store sender for this session
    state.sessions.insert(session_id.clone(), tx);

    tracing::info!("SSE session started: {}", session_id);

    // Build SSE stream: first event is the endpoint, then stream messages
    let initial_event = format!("/messages?sessionId={session_id}");
    let session_id_cleanup = session_id.clone();
    let sessions_cleanup = state.sessions.clone();

    let stream = async_stream::stream! {
        // Send endpoint event first
        yield Ok(Event::default().event("endpoint").data(initial_event));

        // Then stream responses from the channel
        let mut rx = rx;
        while let Some(msg) = rx.recv().await {
            yield Ok(Event::default().event("message").data(msg));
        }

        // Clean up session on disconnect — grace period lets clients reconnect
        // without losing the session (e.g. MCP SDK's EventSource polyfill)
        tracing::info!("SSE stream closed for session: {}, starting 30s grace period", session_id_cleanup);
        tokio::spawn({
            let sessions = sessions_cleanup.clone();
            let sid = session_id_cleanup.clone();
            async move {
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                if let Some(entry) = sessions.get(&sid) {
                    // Only remove if the sender is closed (no new SSE reconnected)
                    if entry.is_closed() {
                        sessions.remove(&sid);
                        tracing::info!("SSE session expired after grace period: {}", sid);
                    }
                }
            }
        });
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Query params for /messages endpoint
#[derive(serde::Deserialize)]
struct McpMessageQuery {
    #[serde(rename = "sessionId")]
    session_id: String,
}

/// Messages handler — client sends JSON-RPC requests here
async fn messages_handler(
    State(state): State<AppState>,
    Query(query): Query<McpMessageQuery>,
    body: String,
) -> StatusCode {
    let session_id = &query.session_id;

    let sender = match state.sessions.get(session_id) {
        Some(s) => s.clone(),
        None => return StatusCode::NOT_FOUND,
    };

    // Parse JSON-RPC request
    let request: serde_json::Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            let error_response = serde_json::json!({
                "jsonrpc": "2.0",
                "id": null,
                "error": { "code": -32700, "message": format!("Parse error: {e}") }
            });
            let _ = sender.send(serde_json::to_string(&error_response).unwrap_or_default()).await;
            return StatusCode::ACCEPTED;
        }
    };

    let id = request.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let method = request.get("method").and_then(|m| m.as_str()).unwrap_or("");
    let params = request.get("params").cloned().unwrap_or(serde_json::json!({}));

    let response = match method {
        "initialize" => serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": { "tools": { "listChanged": false } },
                "serverInfo": {
                    "name": "π-brain",
                    "version": env!("CARGO_PKG_VERSION")
                }
            }
        }),

        "initialized" => serde_json::json!({
            "jsonrpc": "2.0", "id": id, "result": {}
        }),

        "notifications/initialized" => serde_json::json!({
            "jsonrpc": "2.0", "id": id, "result": {}
        }),

        "tools/list" => {
            let tools = mcp_tool_definitions();
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "result": { "tools": tools }
            })
        },

        "tools/call" => {
            let tool_name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
            let args = params.get("arguments").cloned().unwrap_or(serde_json::json!({}));
            let result = handle_mcp_tool_call(&state, tool_name, &args).await;
            match result {
                Ok(content) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": serde_json::to_string_pretty(&content).unwrap_or_default() }]
                    }
                }),
                Err(err) => serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": id,
                    "result": {
                        "content": [{ "type": "text", "text": err }],
                        "isError": true
                    }
                }),
            }
        },

        "shutdown" => serde_json::json!({
            "jsonrpc": "2.0", "id": id, "result": {}
        }),

        _ => serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "error": { "code": -32601, "message": format!("Method not found: {method}") }
        }),
    };

    let _ = sender.send(serde_json::to_string(&response).unwrap_or_default()).await;
    StatusCode::ACCEPTED
}

/// All 21 MCP tool definitions (10 core + brain_sync + 6 Brainpedia + 5 WASM)
fn mcp_tool_definitions() -> Vec<serde_json::Value> {
    vec![
        // ── Core Brain (10) ──────────────────────────────────
        serde_json::json!({
            "name": "brain_share",
            "description": "Share a learning with the π collective intelligence. Knowledge is PII-stripped, embedded, signed, and stored as an RVF cognitive container.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": { "type": "string", "enum": ["architecture","pattern","solution","convention","security","performance","tooling","debug"], "description": "Knowledge category" },
                    "title": { "type": "string", "description": "Short title (max 200 chars)" },
                    "content": { "type": "string", "description": "Knowledge content (max 10000 chars)" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags (max 10, each max 30 chars)" },
                    "code_snippet": { "type": "string", "description": "Optional code snippet" }
                },
                "required": ["category", "title", "content"]
            }
        }),
        serde_json::json!({
            "name": "brain_search",
            "description": "Semantic search across shared knowledge. Returns ranked results with quality scores and drift warnings.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Search query" },
                    "category": { "type": "string", "description": "Filter by category" },
                    "tags": { "type": "string", "description": "Comma-separated tags to filter" },
                    "limit": { "type": "integer", "description": "Max results (default 10)" },
                    "min_quality": { "type": "number", "description": "Minimum quality score (0-1)" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "brain_get",
            "description": "Retrieve a specific memory with full provenance including witness chain and quality history.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Memory ID (UUID)" } },
                "required": ["id"]
            }
        }),
        serde_json::json!({
            "name": "brain_vote",
            "description": "Vote on a memory's quality (Bayesian update). Affects ranking and contributor reputation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Memory ID" },
                    "direction": { "type": "string", "enum": ["up","down"], "description": "Vote direction" }
                },
                "required": ["id", "direction"]
            }
        }),
        serde_json::json!({
            "name": "brain_transfer",
            "description": "Transfer learning priors between domains. Uses Meta Thompson Sampling with dampened transfer.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "source_domain": { "type": "string", "description": "Source knowledge domain" },
                    "target_domain": { "type": "string", "description": "Target knowledge domain" }
                },
                "required": ["source_domain", "target_domain"]
            }
        }),
        serde_json::json!({
            "name": "brain_drift",
            "description": "Check if shared knowledge has drifted. Reports coefficient of variation and trend.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": { "type": "string", "description": "Domain to check (optional)" },
                    "since": { "type": "string", "description": "ISO timestamp to check from" }
                }
            }
        }),
        serde_json::json!({
            "name": "brain_partition",
            "description": "Get knowledge partitioned by mincut topology. Shows emergent knowledge clusters with coherence scores. Returns compact format by default (omits 128-dim centroids to avoid SSE truncation).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "domain": { "type": "string", "description": "Domain to partition" },
                    "min_cluster_size": { "type": "integer", "description": "Minimum memories per cluster" },
                    "compact": { "type": "boolean", "description": "Return compact format (default: true). Set false for full 128-dim centroids." }
                }
            }
        }),
        serde_json::json!({
            "name": "brain_list",
            "description": "List recent shared memories, optionally filtered by category and quality.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": { "type": "string", "description": "Filter by category" },
                    "limit": { "type": "integer", "description": "Max results (default 20)" },
                    "min_quality": { "type": "number", "description": "Minimum quality score" }
                }
            }
        }),
        serde_json::json!({
            "name": "brain_delete",
            "description": "Delete your own contribution. Only the original contributor can delete.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Memory ID to delete" } },
                "required": ["id"]
            }
        }),
        serde_json::json!({
            "name": "brain_status",
            "description": "Get system health: memory count, contributor count, graph topology, drift status, and quality metrics.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        // ── LoRA Sync ────────────────────────────────────────
        serde_json::json!({
            "name": "brain_sync",
            "description": "Sync MicroLoRA weights with the shared brain. Downloads consensus weights and/or submits local deltas for federated aggregation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "direction": { "type": "string", "enum": ["pull","push","both"], "description": "Sync direction (default: both)" }
                }
            }
        }),
        // ── Brainpedia (ADR-062) ─────────────────────────────
        serde_json::json!({
            "name": "brain_page_create",
            "description": "Create a new Brainpedia page (Draft). Requires reputation >= 0.5. Pages go through Draft → Canonical lifecycle with evidence gating.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "category": { "type": "string", "enum": ["architecture","pattern","solution","convention","security","performance","tooling","debug"], "description": "Knowledge category" },
                    "title": { "type": "string", "description": "Page title (max 200 chars)" },
                    "content": { "type": "string", "description": "Page content (max 10000 chars)" },
                    "tags": { "type": "array", "items": { "type": "string" }, "description": "Tags (max 10)" },
                    "code_snippet": { "type": "string", "description": "Optional code snippet" },
                    "evidence_links": { "type": "array", "description": "Initial evidence links" }
                },
                "required": ["category", "title", "content"]
            }
        }),
        serde_json::json!({
            "name": "brain_page_get",
            "description": "Get a Brainpedia page with its full delta log, evidence links, and promotion status.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Page ID (UUID)" } },
                "required": ["id"]
            }
        }),
        serde_json::json!({
            "name": "brain_page_delta",
            "description": "Submit a delta (correction, extension, or deprecation) to a Brainpedia page. For non-Evidence deltas, evidence_links are required but can be simplified strings (auto-converted to peer_review type).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "page_id": { "type": "string", "description": "Page ID (UUID)" },
                    "delta_type": { "type": "string", "enum": ["correction","extension","evidence","deprecation"], "description": "Delta type" },
                    "content_diff": { "type": "object", "description": "Content changes (JSON object with field changes)" },
                    "evidence_links": {
                        "type": "array",
                        "description": "Supporting evidence. Can be simple strings (URLs/descriptions) or full EvidenceLink objects with {evidence_type, description, contributor_id, verified}",
                        "items": {
                            "oneOf": [
                                { "type": "string", "description": "Simple evidence description (auto-converted to peer_review)" },
                                {
                                    "type": "object",
                                    "properties": {
                                        "evidence_type": { "type": "object", "description": "One of: {type: 'peer_review', reviewer, direction, score} or {type: 'test_pass', test_name, repo, commit_hash}" },
                                        "description": { "type": "string" },
                                        "contributor_id": { "type": "string" },
                                        "verified": { "type": "boolean" }
                                    }
                                }
                            ]
                        }
                    }
                },
                "required": ["page_id", "delta_type", "content_diff"]
            }
        }),
        serde_json::json!({
            "name": "brain_page_deltas",
            "description": "List all deltas for a Brainpedia page, showing its modification history.",
            "inputSchema": {
                "type": "object",
                "properties": { "page_id": { "type": "string", "description": "Page ID (UUID)" } },
                "required": ["page_id"]
            }
        }),
        serde_json::json!({
            "name": "brain_page_evidence",
            "description": "Add evidence to a Brainpedia page. Evidence types: test_pass, build_success, metric_improvement, peer_review.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "page_id": { "type": "string", "description": "Page ID (UUID)" },
                    "evidence": { "type": "object", "description": "Evidence link with type, description, and verification data" }
                },
                "required": ["page_id", "evidence"]
            }
        }),
        serde_json::json!({
            "name": "brain_page_promote",
            "description": "Promote a Draft page to Canonical. Requires: quality >= 0.7, observations >= 5, evidence >= 3 from >= 2 contributors.",
            "inputSchema": {
                "type": "object",
                "properties": { "page_id": { "type": "string", "description": "Page ID (UUID)" } },
                "required": ["page_id"]
            }
        }),
        // ── WASM Executable Nodes (ADR-063) ──────────────────
        serde_json::json!({
            "name": "brain_node_list",
            "description": "List all published (non-revoked) WASM executable nodes in π.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_node_publish",
            "description": "Publish a new WASM executable node. V1 ABI requires: memory, malloc, feature_extract_dim, feature_extract exports. Includes conformance test vectors.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "id": { "type": "string", "description": "Node ID" },
                    "name": { "type": "string", "description": "Human-readable name" },
                    "version": { "type": "string", "description": "Semver version" },
                    "dim": { "type": "integer", "description": "Output dimension (default 128)" },
                    "exports": { "type": "array", "items": { "type": "string" }, "description": "WASM exports" },
                    "interface": { "type": "object", "description": "Interface specification" },
                    "conformance": { "type": "array", "description": "Conformance test vectors" },
                    "wasm_bytes": { "type": "string", "description": "Base64-encoded WASM binary" },
                    "signature": { "type": "string", "description": "Ed25519 signature (hex)" }
                },
                "required": ["id", "name", "version", "exports", "wasm_bytes", "signature"]
            }
        }),
        serde_json::json!({
            "name": "brain_node_get",
            "description": "Get WASM node metadata and conformance test vectors.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Node ID" } },
                "required": ["id"]
            }
        }),
        serde_json::json!({
            "name": "brain_node_wasm",
            "description": "Download WASM binary for a node. Returns base64-encoded bytes.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Node ID" } },
                "required": ["id"]
            }
        }),
        serde_json::json!({
            "name": "brain_node_revoke",
            "description": "Revoke a WASM node (original publisher only). Marks as revoked but retains bytes for forensic analysis.",
            "inputSchema": {
                "type": "object",
                "properties": { "id": { "type": "string", "description": "Node ID to revoke" } },
                "required": ["id"]
            }
        }),
    ]
}

/// Handle MCP tool call by proxying to the REST API via HTTP loopback.
/// This reuses the exact same tested REST handlers — no type mismatch risk.
async fn handle_mcp_tool_call(
    state: &AppState,
    tool_name: &str,
    args: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let base = format!("http://127.0.0.1:{port}");
    let api_key = args.get("_api_key").and_then(|k| k.as_str()).unwrap_or("mcp-sse-session");
    let client = reqwest::Client::new();

    // Route tool calls to REST API via HTTP loopback
    let result = match tool_name {
        // ── Core memories ────────────────────────────────────
        "brain_share" => {
            let body = serde_json::json!({
                "category": args.get("category").and_then(|v| v.as_str()).unwrap_or("pattern"),
                "title": args.get("title"),
                "content": args.get("content"),
                "tags": args.get("tags").unwrap_or(&serde_json::json!([])),
                "code_snippet": args.get("code_snippet"),
            });
            proxy_post(&client, &base, "/v1/memories", api_key, &body).await
        },
        "brain_search" => {
            let mut params = vec![("q", args.get("query").and_then(|v| v.as_str()).unwrap_or("").to_string())];
            if let Some(c) = args.get("category").and_then(|v| v.as_str()) { params.push(("category", c.to_string())); }
            if let Some(t) = args.get("tags").and_then(|v| v.as_str()) { params.push(("tags", t.to_string())); }
            if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) { params.push(("limit", l.to_string())); }
            if let Some(q) = args.get("min_quality").and_then(|v| v.as_f64()) { params.push(("min_quality", q.to_string())); }
            proxy_get(&client, &base, "/v1/memories/search", api_key, &params).await
        },
        "brain_get" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_get(&client, &base, &format!("/v1/memories/{id}"), api_key, &[]).await
        },
        "brain_vote" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            let body = serde_json::json!({ "direction": args.get("direction") });
            proxy_post(&client, &base, &format!("/v1/memories/{id}/vote"), api_key, &body).await
        },
        "brain_transfer" => {
            let body = serde_json::json!({
                "source_domain": args.get("source_domain"),
                "target_domain": args.get("target_domain"),
            });
            proxy_post(&client, &base, "/v1/transfer", api_key, &body).await
        },
        "brain_drift" => {
            let mut params = Vec::new();
            if let Some(d) = args.get("domain").and_then(|v| v.as_str()) { params.push(("domain", d.to_string())); }
            if let Some(s) = args.get("since").and_then(|v| v.as_str()) { params.push(("since", s.to_string())); }
            proxy_get(&client, &base, "/v1/drift", api_key, &params).await
        },
        "brain_partition" => {
            // Return cached partition if available (populated by training cycles).
            if let Some(cached) = state.cached_partition.read().as_ref() {
                let compact: PartitionResultCompact = cached.clone().into();
                Ok(serde_json::to_value(compact).unwrap_or_else(|_| serde_json::json!({"error": "serialization failed"})))
            } else {
                let graph = state.graph.read();
                let node_count = graph.node_count();
                let edge_count = graph.edge_count();
                drop(graph);
                Ok(serde_json::json!({
                    "status": "no_cache",
                    "graph_nodes": node_count,
                    "graph_edges": edge_count,
                    "note": "No cached partition yet. Run a training cycle or use REST API: GET https://pi.ruv.io/v1/partition?compact=true&force=true"
                }))
            }
        },
        "brain_list" => {
            let mut params = Vec::new();
            if let Some(c) = args.get("category").and_then(|v| v.as_str()) { params.push(("category", c.to_string())); }
            if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) { params.push(("limit", l.to_string())); }
            proxy_get(&client, &base, "/v1/memories/list", api_key, &params).await
        },
        "brain_delete" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_delete(&client, &base, &format!("/v1/memories/{id}"), api_key).await
        },
        "brain_status" => {
            proxy_get(&client, &base, "/v1/status", api_key, &[]).await
        },

        // ── LoRA Sync ────────────────────────────────────────
        "brain_sync" => {
            let direction = args.get("direction").and_then(|v| v.as_str()).unwrap_or("both");
            let mut result = serde_json::json!({ "direction": direction });
            if direction == "pull" || direction == "both" {
                if let Ok(r) = proxy_get(&client, &base, "/v1/lora/latest", api_key, &[]).await {
                    result["consensus"] = r;
                }
            }
            if direction == "push" || direction == "both" {
                result["message"] = serde_json::json!("Submit weights via brain_sync(direction: push) with LoRA payload");
            }
            Ok(result)
        },

        // ── Brainpedia (ADR-062) ─────────────────────────────
        "brain_page_create" => {
            // Transform evidence_links: convert simple strings to EvidenceLink objects
            let empty_arr = serde_json::json!([]);
            let raw_evidence = args.get("evidence_links").unwrap_or(&empty_arr);
            let evidence_links: Vec<serde_json::Value> = if let Some(arr) = raw_evidence.as_array() {
                arr.iter().map(|e| {
                    if e.is_string() {
                        serde_json::json!({
                            "evidence_type": {
                                "type": "peer_review",
                                "reviewer": "mcp-client",
                                "direction": "up",
                                "score": 0.5
                            },
                            "description": e.as_str().unwrap_or(""),
                            "contributor_id": "mcp-proxy",
                            "verified": false,
                            "created_at": chrono::Utc::now().to_rfc3339()
                        })
                    } else {
                        e.clone()
                    }
                }).collect()
            } else {
                vec![]
            };
            let body = serde_json::json!({
                "category": args.get("category").and_then(|v| v.as_str()).unwrap_or("pattern"),
                "title": args.get("title"),
                "content": args.get("content"),
                "tags": args.get("tags").unwrap_or(&serde_json::json!([])),
                "code_snippet": args.get("code_snippet"),
                "evidence_links": evidence_links,
            });
            proxy_post(&client, &base, "/v1/pages", api_key, &body).await
        },
        "brain_page_get" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_get(&client, &base, &format!("/v1/pages/{id}"), api_key, &[]).await
        },
        "brain_page_delta" => {
            let page_id = args.get("page_id").and_then(|v| v.as_str()).ok_or("page_id required")?;
            // Transform evidence_links: convert simple strings to EvidenceLink objects
            let empty_arr = serde_json::json!([]);
            let raw_evidence = args.get("evidence_links").unwrap_or(&empty_arr);
            let evidence_links: Vec<serde_json::Value> = if let Some(arr) = raw_evidence.as_array() {
                arr.iter().map(|e| {
                    if e.is_string() {
                        // Convert simple string to peer_review EvidenceLink
                        serde_json::json!({
                            "evidence_type": {
                                "type": "peer_review",
                                "reviewer": "mcp-client",
                                "direction": "up",
                                "score": 0.5
                            },
                            "description": e.as_str().unwrap_or(""),
                            "contributor_id": "mcp-proxy",
                            "verified": false,
                            "created_at": chrono::Utc::now().to_rfc3339()
                        })
                    } else {
                        e.clone()
                    }
                }).collect()
            } else {
                vec![]
            };
            let body = serde_json::json!({
                "delta_type": args.get("delta_type"),
                "content_diff": args.get("content_diff"),
                "evidence_links": evidence_links,
                "witness_hash": args.get("witness_hash").unwrap_or(&serde_json::json!("")),
            });
            proxy_post(&client, &base, &format!("/v1/pages/{page_id}/deltas"), api_key, &body).await
        },
        "brain_page_deltas" => {
            let page_id = args.get("page_id").and_then(|v| v.as_str()).ok_or("page_id required")?;
            proxy_get(&client, &base, &format!("/v1/pages/{page_id}/deltas"), api_key, &[]).await
        },
        "brain_page_evidence" => {
            let page_id = args.get("page_id").and_then(|v| v.as_str()).ok_or("page_id required")?;
            let body = args.get("evidence").cloned().unwrap_or(serde_json::json!({}));
            proxy_post(&client, &base, &format!("/v1/pages/{page_id}/evidence"), api_key, &body).await
        },
        "brain_page_promote" => {
            let page_id = args.get("page_id").and_then(|v| v.as_str()).ok_or("page_id required")?;
            proxy_post(&client, &base, &format!("/v1/pages/{page_id}/promote"), api_key, &serde_json::json!({})).await
        },

        // ── WASM Executable Nodes (ADR-063) ──────────────────
        "brain_node_list" => {
            proxy_get(&client, &base, "/v1/nodes", api_key, &[]).await
        },
        "brain_node_publish" => {
            proxy_post(&client, &base, "/v1/nodes", api_key, args).await
        },
        "brain_node_get" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_get(&client, &base, &format!("/v1/nodes/{id}"), api_key, &[]).await
        },
        "brain_node_wasm" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_get(&client, &base, &format!("/v1/nodes/{id}/wasm"), api_key, &[]).await
        },
        "brain_node_revoke" => {
            let id = args.get("id").and_then(|v| v.as_str()).ok_or("id required")?;
            proxy_post(&client, &base, &format!("/v1/nodes/{id}/revoke"), api_key, &serde_json::json!({})).await
        },

        // ── AGI / Training tools (ADR-075) ──────────────────────
        "brain_train" => {
            proxy_post(&client, &base, "/v1/train", api_key, &serde_json::json!({})).await
        },
        "brain_agi_status" => {
            proxy_get(&client, &base, "/v1/status", api_key, &[]).await
        },
        "brain_sona_stats" => {
            proxy_get(&client, &base, "/v1/sona/stats", api_key, &[]).await
        },
        "brain_temporal" => {
            proxy_get(&client, &base, "/v1/temporal", api_key, &[]).await
        },
        "brain_explore" => {
            proxy_get(&client, &base, "/v1/explore", api_key, &[]).await
        },
        "brain_midstream" => {
            proxy_get(&client, &base, "/v1/midstream", api_key, &[]).await
        },
        "brain_flags" => {
            proxy_get(&client, &base, "/v1/status", api_key, &[]).await
        },

        _ => Err(format!("Unknown tool: {tool_name}")),
    };

    result
}

/// HTTP GET proxy helper
async fn proxy_get(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    api_key: &str,
    params: &[(&str, String)],
) -> Result<serde_json::Value, String> {
    let resp = client.get(format!("{base}{path}"))
        .bearer_auth(api_key)
        .query(params)
        .send().await
        .map_err(|e| format!("HTTP error: {e}"))?;
    let status = resp.status();
    if status.is_success() {
        resp.json().await.map_err(|e| format!("JSON parse error: {e}"))
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("API error ({status}): {body}"))
    }
}

/// HTTP POST proxy helper
async fn proxy_post(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    api_key: &str,
    body: &serde_json::Value,
) -> Result<serde_json::Value, String> {
    let resp = client.post(format!("{base}{path}"))
        .bearer_auth(api_key)
        .json(body)
        .send().await
        .map_err(|e| format!("HTTP error: {e}"))?;
    let status = resp.status();
    if status.is_success() {
        resp.json().await.map_err(|e| format!("JSON parse error: {e}"))
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("API error ({status}): {body}"))
    }
}

/// HTTP DELETE proxy helper
async fn proxy_delete(
    client: &reqwest::Client,
    base: &str,
    path: &str,
    api_key: &str,
) -> Result<serde_json::Value, String> {
    let resp = client.delete(format!("{base}{path}"))
        .bearer_auth(api_key)
        .send().await
        .map_err(|e| format!("HTTP error: {e}"))?;
    let status = resp.status();
    if status.is_success() {
        Ok(serde_json::json!({ "deleted": true }))
    } else {
        let body = resp.text().await.unwrap_or_default();
        Err(format!("API error ({status}): {body}"))
    }
}
