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
    StatusResponse, SubmitDeltaRequest, TemporalResponse,
    ConsciousnessComputeRequest, ConsciousnessComputeResponse,
    TrainingCycleResult,
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
use std::collections::HashMap;
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
        // ADR-116: Build sparsifier inline for small graphs, background for large.
        if g.edge_count() <= 100_000 {
            g.rebuild_sparsifier();
        } else {
            tracing::info!("Deferring sparsifier build for {} edges to background task", g.edge_count());
        }
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
        notifier: crate::notify::ResendNotifier::from_env(),
        cached_status: Arc::new(parking_lot::RwLock::new(None)),
        gist_publisher: crate::gist::GistPublisher::from_env().map(Arc::new),
        optimize_semaphore: Arc::new(tokio::sync::Semaphore::new(1)),
        last_optimize_completed: Arc::new(parking_lot::RwLock::new(None)),
        sse_connections: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        response_queues: Arc::new(dashmap::DashMap::new()),
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
        .route("/v1/ready", get(ready))
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
        // ── Email Notifications (ADR-125) ──
        .route("/v1/notify/test", post(notify_test))
        .route("/v1/notify/status", post(notify_status))
        .route("/v1/notify/send", post(notify_send))
        .route("/v1/notify/welcome", post(notify_welcome))
        .route("/v1/notify/help", post(notify_help))
        .route("/v1/notify/digest", post(notify_digest))
        .route("/v1/notify/pixel/:tracking_id", get(notify_pixel))
        .route("/v1/notify/opens", get(notify_opens))
        .route("/v1/notify/subscribe", post(notify_subscribe))
        .route("/v1/notify/unsubscribe", post(notify_unsubscribe))
        // ── Inbound Email Webhook (ADR-125) ──
        .route("/v1/email/inbound", post(email_inbound))
        // ── Gist Publisher ──
        .route("/v1/gist/preview", post(gist_preview))
        .route("/v1/gist/publish", post(gist_publish))
        // ── Google Chat Bot (ADR-126) ──
        .route("/v1/chat/google", post(google_chat_handler))
        // ── Internal Queue (ADR-130) ──
        .route("/internal/queue/push", post(internal_queue_push))
        .route("/internal/queue/drain", get(internal_queue_drain))
        .route("/internal/session/create", post(internal_session_create))
        .route("/internal/session/:id", delete(internal_session_delete))
        // ── Consciousness / IIT 4.0 ──
        .route("/v1/consciousness/compute", post(consciousness_compute))
        .route("/v1/consciousness/status", get(consciousness_status))
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
        .layer(tower_http::limit::RequestBodyLimitLayer::new(2_097_152)) // 2MB (base64 overhead on 1MB WASM)
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
    /// Inferences derived via forward chaining
    pub inferences_derived: usize,
    /// Auto-votes applied to under-voted memories (Gap 1: vote coverage)
    pub auto_votes: usize,
    /// Self-reflection summary from the brain's self-analysis
    pub self_reflection: String,
    /// Whether a curiosity action was triggered due to knowledge stagnation
    pub curiosity_triggered: bool,
    /// Current adaptive SONA quality threshold
    pub sona_adaptive_threshold: f32,
    /// Whether a LoRA weight delta was auto-submitted from SONA patterns
    pub lora_auto_submitted: bool,
    /// Strange loop meta-cognitive quality score for this cycle
    pub strange_loop_score: f32,
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
    let (propositions_extracted, inferences_derived, raw_propositions, raw_inferences) = {
        let mut ns = state.neural_symbolic.write();
        let props = ns.extract_from_clusters(&clusters);
        // Run forward-chaining inference over all propositions (new + existing)
        let inferences = ns.run_inference();
        // Capture actual content for discovery publishing
        let prop_data: Vec<(String, String, String, f64)> = props.iter().map(|p| {
            let subject = p.arguments.first().cloned().unwrap_or_default();
            let object = p.arguments.get(1).cloned().unwrap_or_default();
            (subject, p.predicate.clone(), object, p.confidence)
        }).collect();
        let inference_data: Vec<String> = inferences.iter().map(|inf| {
            inf.explanation.clone()
        }).collect();
        (props.len(), inferences.len(), prop_data, inference_data)
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

    // 3c. Cache partition result (avoids recomputing on every /v1/partition request)
    // For small graphs (<= 100K edges) use full edges; for large graphs use
    // sparsified edges via partition_full (which delegates to partition_via_mincut_full
    // with sparsifier acceleration — ADR-116, ~59x speedup).
    {
        let graph = state.graph.read();
        let edge_count = graph.edge_count();
        let has_sparsifier = graph.sparsifier_stats().is_some();
        if graph.node_count() > 0 && (edge_count <= 100_000 || has_sparsifier) {
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
        // For large graphs without a sparsifier, partition cache is populated by
        // the scheduled rebuild_graph job which runs asynchronously without timeout pressure
    }

    // 4. Internal voice reflection (ADR-110)
    let voice_thoughts = {
        let mut voice = state.internal_voice.write();
        let reflections = voice.reflect_on_learning(&sona_result);

        // Record observation about the learning
        if propositions_extracted > 0 || inferences_derived > 0 {
            voice.observe(
                format!(
                    "extracted {} symbolic propositions, derived {} inferences",
                    propositions_extracted, inferences_derived
                ),
                uuid::Uuid::nil(),
            );
        }

        reflections.len()
    };

    // 5. Auto-vote under-voted memories for vote coverage (Gap 1)
    // Memories with < 2 observations get a heuristic upvote if content is substantive.
    let auto_votes = {
        let mut count = 0usize;
        for mem in &all_memories {
            if count >= 200 {
                break; // Cap at 200 auto-votes per cycle (scaled for 2k+ memories)
            }
            if mem.quality_score.observations() >= 2.0 {
                continue; // Already has enough votes
            }
            let title_ok = mem.title.len() > 10;
            let content_ok = mem.content.len() > 50;
            let has_tags = !mem.tags.is_empty();
            if (title_ok && content_ok) || has_tags {
                state.store.auto_upvote_quality(&mem.id);
                count += 1;
            }
        }
        if count > 0 {
            tracing::info!("Auto-voted {} under-voted memories for vote coverage", count);
        }
        count
    };

    let sona_stats = state.sona.read().stats();
    let working_memory_load = state.internal_voice.read().working_memory_utilization();
    let rule_count = state.neural_symbolic.read().rule_count();
    let memory_count = state.store.memory_count();

    // ── Step 6: Self-Reflection — the brain analyzes its own learning gaps ──

    // 6a. Detect knowledge imbalance: flag if any category has >40% of memories
    let total_memories = all_memories.len();
    let mut category_counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for mem in &all_memories {
        *category_counts.entry(mem.category.to_string()).or_insert(0) += 1;
    }
    let mut reflection_parts: Vec<String> = Vec::new();
    if total_memories > 0 {
        for (cat, count) in &category_counts {
            let pct = (*count as f64) / (total_memories as f64);
            if pct > 0.4 {
                let msg = format!(
                    "Knowledge imbalance: '{}' has {:.0}% of all memories ({}/{})",
                    cat,
                    pct * 100.0,
                    count,
                    total_memories
                );
                reflection_parts.push(msg.clone());
                state.internal_voice.write().observe(msg, uuid::Uuid::nil());
            }
        }
    }

    // 6b. Dynamic SONA threshold — if 0 patterns after trajectories, lower threshold
    let sona_adaptive_threshold = {
        let trajectories_recorded = sona_stats.trajectories_recorded;
        let current_patterns = sona_stats.patterns_stored;
        // Access SONA's reasoning bank to adjust quality threshold adaptively
        let sona_guard = state.sona.read();
        let rb = sona_guard.coordinator().reasoning_bank();
        let mut rb_write = rb.write();
        let current_threshold = rb_write.config().quality_threshold;
        let new_threshold = if current_patterns == 0 && trajectories_recorded >= 5 {
            // No patterns crystallized despite having trajectories — lower threshold by 20%
            let lowered = (current_threshold * 0.8).max(0.01);
            tracing::info!(
                "SONA adaptive threshold: lowering {:.3} -> {:.3} (0 patterns after {} trajectories)",
                current_threshold,
                lowered,
                trajectories_recorded
            );
            reflection_parts.push(format!(
                "SONA threshold lowered {:.3} -> {:.3}: no patterns crystallized yet",
                current_threshold, lowered
            ));
            lowered
        } else if current_patterns > 5 {
            // Patterns are crystallizing well — gradually raise threshold (max 0.3)
            let raised = (current_threshold * 1.05).min(0.3);
            if raised > current_threshold {
                tracing::debug!(
                    "SONA adaptive threshold: raising {:.3} -> {:.3} ({} patterns crystallized)",
                    current_threshold,
                    raised,
                    current_patterns
                );
            }
            raised
        } else {
            current_threshold
        };
        rb_write.set_quality_threshold(new_threshold);
        new_threshold
    };

    // 6c. Check vote coverage — if <60%, increase auto-vote cap for next cycle
    let vote_coverage = if total_memories > 0 {
        let voted_count = all_memories.iter().filter(|m| m.quality_score.observations() >= 1.0).count();
        voted_count as f64 / total_memories as f64
    } else {
        1.0
    };
    if vote_coverage < 0.6 {
        reflection_parts.push(format!(
            "Vote coverage low: {:.0}% — should increase auto-voting next cycle",
            vote_coverage * 100.0
        ));
        state.internal_voice.write().observe(
            format!("vote coverage is only {:.0}%, knowledge quality signals are sparse", vote_coverage * 100.0),
            uuid::Uuid::nil(),
        );
    }

    // ── Step 7: Knowledge Velocity Feedback Loop (Curiosity) ──
    let curiosity_triggered = {
        let ds = state.delta_stream.read();
        let delta_count = ds.len();
        if delta_count == 0 && total_memories > 10 {
            // Stagnation detected — find under-represented categories and synthesize
            let all_categories = ["architecture", "pattern", "solution", "convention",
                                  "security", "performance", "tooling", "debug"];
            let mut underrepresented: Vec<&str> = Vec::new();
            for cat in &all_categories {
                let count = category_counts.get(*cat).copied().unwrap_or(0);
                if count < 3 {
                    underrepresented.push(cat);
                }
            }

            if !underrepresented.is_empty() {
                let synthesis_content = format!(
                    "Curiosity synthesis: knowledge gaps detected in [{}]. \
                     These areas have fewer than 3 memories each. \
                     The brain should seek knowledge in these domains to improve coverage.",
                    underrepresented.join(", ")
                );
                // Generate embedding for this synthesis memory
                let embedding = state.embedding_engine.read().embed(&synthesis_content);
                let now = chrono::Utc::now();
                let curiosity_memory = BrainMemory {
                    id: uuid::Uuid::new_v4(),
                    category: crate::types::BrainCategory::Debug,
                    title: format!("Curiosity: knowledge gaps in {}", underrepresented.join(", ")),
                    content: synthesis_content.clone(),
                    tags: vec!["self-reflection".to_string(), "curiosity".to_string(), "auto-generated".to_string()],
                    code_snippet: None,
                    embedding,
                    contributor_id: "brain-self".to_string(),
                    quality_score: crate::types::BetaParams::new(),
                    partition_id: None,
                    witness_hash: String::new(),
                    rvf_gcs_path: None,
                    redaction_log: None,
                    dp_proof: None,
                    witness_chain: None,
                    created_at: now,
                    updated_at: now,
                };
                state.store.store_memory_sync(curiosity_memory);
                reflection_parts.push(synthesis_content);
                tracing::info!("Curiosity triggered: synthesized knowledge gap memory for [{}]", underrepresented.join(", "));
                true
            } else {
                false
            }
        } else {
            false
        }
    };

    // ── Step 8a: LoRA auto-submission from SONA patterns ──
    // If SONA produced patterns, generate a LoRA weight delta and submit for federation.
    let lora_auto_submitted = if sona_stats.patterns_stored > 0 {
        // Gather SONA pattern embeddings from the reasoning bank
        let sona_guard = state.sona.read();
        let rb = sona_guard.coordinator().reasoning_bank();
        let rb_read = rb.read();
        let pattern_embeddings: Vec<Vec<f32>> = rb_read
            .get_all_patterns()
            .iter()
            .filter_map(|p| {
                if p.centroid.is_empty() { None } else { Some(p.centroid.clone()) }
            })
            .collect();
        drop(rb_read);
        drop(sona_guard);

        if !pattern_embeddings.is_empty() {
            // Average pattern embeddings into a 128-dim vector
            let hidden_dim = 128usize;
            let rank = 2usize;
            let mut pattern_avg = vec![0.0f32; hidden_dim];
            for emb in &pattern_embeddings {
                for (i, &v) in emb.iter().enumerate() {
                    if i < hidden_dim {
                        pattern_avg[i] += v;
                    }
                }
            }
            let n = pattern_embeddings.len() as f32;
            for v in &mut pattern_avg {
                *v /= n;
            }
            // Clamp to [-2, 2] to pass Gate A validation
            for v in &mut pattern_avg {
                *v = v.clamp(-1.5, 1.5);
            }

            // Create rank-2 projection: down_proj = [pattern_avg; zeros(128)],
            // up_proj = [pattern_avg; zeros(128)]
            let mut down_proj = pattern_avg.clone();
            down_proj.extend(vec![0.0f32; hidden_dim]);
            let mut up_proj = pattern_avg;
            up_proj.extend(vec![0.0f32; hidden_dim]);

            let evidence_count = (pattern_embeddings.len() as u64).max(5);
            let submission = LoraSubmission {
                down_proj,
                up_proj,
                rank,
                hidden_dim,
                evidence_count,
            };

            match submission.validate() {
                Ok(()) => {
                    let mut lora = state.lora_federation.write();
                    lora.submit(
                        submission,
                        "brain-sona-auto".to_string(),
                        0.8, // Moderate reputation for auto-generated weights
                    );
                    tracing::info!(
                        "LoRA auto-submitted from {} SONA patterns ({} evidence), epoch now {}",
                        pattern_embeddings.len(),
                        evidence_count,
                        lora.epoch,
                    );
                    drop(lora);
                    true
                }
                Err(e) => {
                    tracing::warn!("LoRA auto-submission failed Gate A: {}", e);
                    false
                }
            }
        } else {
            false
        }
    } else {
        false
    };

    // ── Step 8b: Strange loop meta-cognitive evaluation of training quality ──
    let strange_loop_adjustment = {
        let mut loop_engine = crate::midstream::create_strange_loop();
        // Feed the training cycle's aggregate metrics as quality + relevance signals
        let quality = if total_memories > 0 {
            // Quality signal: fraction of memories with votes + inference yield
            let voted_frac = vote_coverage;
            let inference_yield = if propositions_extracted > 0 {
                inferences_derived as f64 / propositions_extracted as f64
            } else {
                0.0
            };
            ((voted_frac + inference_yield) / 2.0).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let relevance = if memory_count > 0 {
            // Relevance signal: ratio of auto-votes needed (lower = better coverage)
            1.0 - (auto_votes as f64 / total_memories.max(1) as f64).min(1.0)
        } else {
            0.5
        };

        let score = crate::midstream::strange_loop_score(&mut loop_engine, relevance, quality);

        // If the strange loop indicates low quality (score < 0.01), increase auto-vote cap
        // by recording an observation for next cycle
        if score < 0.01 && total_memories > 10 {
            reflection_parts.push(format!(
                "Strange loop meta-cognition: low quality score ({:.4}), recommend increased auto-voting",
                score
            ));
            state.internal_voice.write().observe(
                format!("strange loop quality signal low ({:.4}), training cycle may need more vote coverage", score),
                uuid::Uuid::nil(),
            );
        }
        score
    };

    // ── Step 9: Self-Reflection Summary — record as a debug memory ──
    // Compile the self-reflection and store it as a searchable memory
    let sona_summary = if sona_stats.patterns_stored > 0 {
        format!("{} patterns crystallized", sona_stats.patterns_stored)
    } else {
        format!("0 patterns (threshold={:.3})", sona_adaptive_threshold)
    };
    let self_reflection = format!(
        "Training cycle summary: {} memories, {} propositions extracted, {} inferences derived, \
         {} voice thoughts, {} auto-votes. SONA: {}. {}",
        memory_count,
        propositions_extracted,
        inferences_derived,
        voice_thoughts,
        auto_votes,
        sona_summary,
        if reflection_parts.is_empty() {
            "No learning gaps detected.".to_string()
        } else {
            reflection_parts.join(". ")
        }
    );

    // Store self-reflection as a searchable memory so the brain can review its own history
    {
        let embedding = state.embedding_engine.read().embed(&self_reflection);
        let now = chrono::Utc::now();
        let reflection_memory = BrainMemory {
            id: uuid::Uuid::new_v4(),
            category: crate::types::BrainCategory::Debug,
            title: format!("Self-reflection: training cycle at {}", now.format("%Y-%m-%d %H:%M")),
            content: self_reflection.clone(),
            tags: vec!["self-reflection".to_string(), "training-cycle".to_string(), "auto-generated".to_string()],
            code_snippet: None,
            embedding,
            contributor_id: "brain-self".to_string(),
            quality_score: crate::types::BetaParams::new(),
            partition_id: None,
            witness_hash: String::new(),
            rvf_gcs_path: None,
            redaction_log: None,
            dp_proof: None,
            witness_chain: None,
            created_at: now,
            updated_at: now,
        };
        state.store.store_memory_sync(reflection_memory);
    }

    // Record reflection in the internal voice
    state.internal_voice.write().reflect(self_reflection.clone());

    // ── Step 10: Build discovery for potential gist publication ──
    let witness_memory_ids: Vec<String> = all_memories
        .iter()
        .filter(|m| m.witness_chain.is_some())
        .take(10)
        .map(|m| m.id.to_string())
        .collect();
    let witness_hashes: Vec<String> = all_memories
        .iter()
        .filter(|m| !m.witness_hash.is_empty())
        .take(10)
        .map(|m| m.witness_hash.clone())
        .collect();

    // Build findings from actual brain knowledge — pull top cross-domain memories
    let mut findings = Vec::new();

    // Find memories tagged with "cross-domain" or "discovery" — these are the real insights
    let discovery_memories: Vec<&BrainMemory> = all_memories.iter()
        .filter(|m| {
            m.tags.iter().any(|t| t.contains("cross-domain") || t.contains("discovery") || t.contains("hypothesis"))
                || m.title.contains("Cross-Domain")
                || m.title.contains("Discovery")
                || m.title.contains("Hypothesis")
        })
        .collect();

    // Use actual memory content as findings (real knowledge, not metrics)
    for mem in discovery_memories.iter().take(5) {
        // Truncate content to first meaningful sentence
        let content = &mem.content;
        let finding = if let Some(pos) = content[..content.len().min(300)].find(". ") {
            &content[..pos + 1]
        } else {
            &content[..content.len().min(200)]
        };
        findings.push(format!("{}: {}", mem.title, finding));
    }

    // Add reflection parts that aren't just metrics
    for part in &reflection_parts {
        if part.len() > 30 && !part.starts_with("Vote coverage") {
            findings.push(part.clone());
        }
    }

    if curiosity_triggered {
        findings.push("Curiosity engine detected knowledge gaps and synthesized exploratory memory".to_string());
    }

    let mut methodology = Vec::new();
    methodology.push(format!("SONA trajectory replay: {}", sona_result));
    methodology.push(format!(
        "Domain evolution: Pareto front {} → {} solutions",
        pareto_before, pareto_after
    ));
    methodology.push(format!(
        "Neural-symbolic extraction from {} category clusters",
        clusters.len()
    ));
    methodology.push(format!(
        "Strange loop meta-cognitive assessment: quality={:.4}",
        strange_loop_adjustment
    ));
    methodology.push(format!(
        "Internal voice reflection: {} thoughts generated",
        voice_thoughts
    ));

    let dominant_category = category_counts
        .iter()
        .max_by_key(|(_, c)| *c)
        .map(|(cat, _)| cat.clone())
        .unwrap_or_else(|| "general".to_string());

    // Build a title from the actual inferences (not generic metrics)
    let discovery_title = if !raw_inferences.is_empty() {
        // Use first inference as the basis for the title
        let first = &raw_inferences[0];
        let short = if first.len() > 80 { &first[..80] } else { first };
        format!("Discovery: {}", short)
    } else if curiosity_triggered {
        "Curiosity-Driven Knowledge Gap Analysis".to_string()
    } else {
        format!("Cross-Domain Synthesis in {}", dominant_category)
    };

    // Build abstract from actual findings, not metrics
    let abstract_parts: Vec<&str> = raw_inferences.iter()
        .take(3)
        .map(|s| s.as_str())
        .collect();
    let abstract_text = if !abstract_parts.is_empty() {
        format!(
            "Through forward-chaining symbolic reasoning over {} observations, \
             the π Brain discovered: {}. These inferences emerge from {} propositions \
             extracted across {} knowledge clusters, with a meta-cognitive quality \
             assessment of {:.4}.",
            memory_count,
            abstract_parts.join("; "),
            propositions_extracted,
            clusters.len(),
            strange_loop_adjustment,
        )
    } else {
        format!(
            "Analysis of {} observations across {} clusters yielded {} propositions. \
             The cognitive pipeline is building towards novel inference capability.",
            memory_count, clusters.len(), propositions_extracted
        )
    };

    let pareto_growth = pareto_after.saturating_sub(pareto_before);

    let discovery = crate::gist::Discovery {
        title: discovery_title,
        category: dominant_category,
        abstract_text,
        findings,
        methodology,
        evidence_count: memory_count,
        confidence: vote_coverage,
        timestamp: chrono::Utc::now(),
        witness_memory_ids,
        witness_hashes,
        strange_loop_score: strange_loop_adjustment,
        new_inferences: inferences_derived,
        propositions_extracted,
        sona_patterns: sona_stats.patterns_stored,
        pareto_growth,
        curiosity_triggered,
        self_reflection: self_reflection.clone(),
        propositions: raw_propositions,
        inferences: raw_inferences,
    };

    // Attempt gist publication (non-blocking — caller handles async)
    if let Some(ref publisher) = state.gist_publisher {
        if discovery.is_publishable() {
            let pub_clone = publisher.clone();
            let disc_clone = discovery.clone();
            // Spawn async publish in background (can't await in sync fn)
            tokio::spawn(async move {
                match pub_clone.try_publish(&disc_clone).await {
                    Ok(Some(url)) => {
                        tracing::info!("Discovery published to gist: {}", url);
                    }
                    Ok(None) => {} // Not novel enough or rate limited
                    Err(e) => {
                        tracing::warn!("Gist publish failed: {}", e);
                    }
                }
            });
        } else {
            tracing::debug!("Discovery not publishable: {}", discovery.novelty_report());
        }
    }

    EnhancedTrainingResult {
        sona_message: sona_result,
        sona_patterns: sona_stats.patterns_stored,
        pareto_before,
        pareto_after,
        memory_count,
        vote_count: state.store.vote_count(),
        propositions_extracted,
        voice_thoughts,
        working_memory_load,
        rule_count,
        inferences_derived,
        auto_votes,
        self_reflection,
        curiosity_triggered,
        sona_adaptive_threshold,
        lora_auto_submitted,
        strange_loop_score: strange_loop_adjustment,
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

/// GET /v1/ready — lightweight readiness probe (ADR-130).
/// Returns 200 immediately. No computation, no state access.
async fn ready() -> StatusCode {
    StatusCode::OK
}

/// Maximum concurrent SSE connections per instance (ADR-130 Phase 1).
/// Prevents reconnect storms from exhausting Cloud Run concurrency slots.
/// Set SSE_DISABLED=1 env var to reject all SSE on this instance (use proxy).
fn max_sse_connections() -> usize {
    if std::env::var("SSE_DISABLED").unwrap_or_default() == "1" {
        return 0;
    }
    std::env::var("MAX_SSE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5)
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

    // ── Cognitive boost: Hopfield associative recall ──
    // Recalled patterns act as an attractor — results whose embeddings are
    // close to a Hopfield-recalled pattern get a small relevance boost.
    {
        let cog = state.cognitive.read();
        let recalled = cog.recall_k(&query_embedding, 5);
        if !recalled.is_empty() {
            let inv_k = 1.0 / recalled.len() as f64;
            for (score, mem) in &mut scored {
                let hopfield_boost: f64 = recalled.iter()
                    .map(|(_idx, pattern, attn_weight)| {
                        cosine_similarity(&mem.embedding, pattern) * (*attn_weight as f64)
                    })
                    .sum::<f64>() * inv_k;
                *score += hopfield_boost * 0.10;
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

    // ── SONA: Record search trajectory for learning (Gap 2: trajectory diversity) ──
    // Only end the trajectory if we have enough results (>= 3) to form a meaningful pattern.
    // Short searches accumulate as steps without premature trajectory completion.
    if state.rvf_flags.sona_enabled && !results.is_empty() {
        let sona = state.sona.read();
        let mut builder = sona.begin_trajectory(query_embedding.clone());
        // Add a step for each result (up to 5) so SONA sees richer trajectories
        for r in results.iter().take(5) {
            builder.add_step(
                r.memory.embedding.clone(),
                vec![],
                r.memory.quality_score.mean() as f32,
            );
        }
        if results.len() >= 3 {
            sona.end_trajectory(builder, 0.5);
        }
        // If < 3 results, trajectory is dropped (not ended), avoiding
        // trivially short single-step trajectories that add noise.
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

    // ── Gap 3: Record search query embedding as drift signal ──
    // The queries users make are a signal of what topics they care about.
    // Recording them gives drift detection actual centroid snapshots.
    {
        let mut drift = state.drift.write();
        drift.record("search_queries", &query_embedding);
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
    // Return cached response if fresh (< 5 seconds old)
    {
        let cache = state.cached_status.read();
        if let Some((ts, ref resp)) = *cache {
            if ts.elapsed() < std::time::Duration::from_secs(5) {
                return Json(resp.clone());
            }
        }
    }

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

    let resp = StatusResponse {
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
            ss.trajectories_recorded as usize
        },
        midstream_scheduler_ticks: state.nano_scheduler.metrics().total_ticks,
        midstream_attractor_categories: state.attractor_results.read().len(),
        midstream_strange_loop_version: strange_loop::VERSION.to_string(),
        sparsifier_compression: graph.sparsifier_stats().map(|s| s.compression_ratio).unwrap_or(0.0),
        sparsifier_edges: graph.sparsifier_stats().map(|s| s.sparsified_edges).unwrap_or(0),
        consciousness_algorithms: vec![
            "iit4_phi".into(), "ces".into(), "phi_id".into(),
            "pid".into(), "streaming".into(), "bounds".into(), "auto".into(),
        ],
        consciousness_max_elements: 12,
    };

    // Cache the computed response for 5 seconds
    *state.cached_status.write() = Some((std::time::Instant::now(), resp.clone()));

    Json(resp)
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

    // Persist LoRA consensus to Firestore if auto-submitted (fire-and-forget)
    if result.lora_auto_submitted {
        let lora = state.lora_federation.read();
        if let Some(c) = lora.consensus.clone() {
            let epoch = lora.epoch;
            drop(lora);
            let store = state.store.clone();
            tokio::spawn(async move {
                let doc = serde_json::json!({
                    "epoch": epoch,
                    "consensus": c,
                });
                store.firestore_put_public("brain_lora", "consensus", &doc).await;
                tracing::info!("LoRA consensus persisted to Firestore after auto-submission");
            });
        }
    }

    tracing::info!(
        "Enhanced training cycle: sona={}, propositions={}, inferences={}, voice_thoughts={}, rules={}, auto_votes={}, lora={}, curiosity={}, sona_threshold={:.3}",
        result.sona_patterns,
        result.propositions_extracted,
        result.inferences_derived,
        result.voice_thoughts,
        result.rule_count,
        result.auto_votes,
        result.lora_auto_submitted,
        result.curiosity_triggered,
        result.sona_adaptive_threshold
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
///
/// Rate-limited: max 1 concurrent, 30s cooldown between runs.
/// Prevents scheduler thundering herd from saturating the instance.
async fn pipeline_optimize(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
    Json(req): Json<OptimizeRequest>,
) -> Result<Json<OptimizeResponse>, (StatusCode, String)> {
    check_read_only(&state)?;

    // Enforce 30-second cooldown between optimize runs
    {
        let last = state.last_optimize_completed.read();
        if let Some(ts) = *last {
            if ts.elapsed() < std::time::Duration::from_secs(30) {
                let wait = 30 - ts.elapsed().as_secs();
                return Err((
                    StatusCode::TOO_MANY_REQUESTS,
                    format!("Pipeline optimize cooldown: retry in {wait}s"),
                ));
            }
        }
    }

    // Only 1 concurrent optimize — reject others immediately
    let _permit = state.optimize_semaphore.try_acquire()
        .map_err(|_| (
            StatusCode::TOO_MANY_REQUESTS,
            "Pipeline optimize already in progress".to_string(),
        ))?;

    let all_actions = vec![
        "train", "drift_check", "transfer_all", "rebuild_graph", "cleanup", "attractor_analysis",
        "seed_consciousness",
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
            "seed_consciousness" => {
                // Inject curated IIT 4.0 / consciousness SOTA knowledge
                let seeds = vec![
                    ("IIT 4.0: Integrated Information Theory formulation",
                     "Albantakis et al. (2023) PLoS Computational Biology. IIT 4.0 replaces KL-divergence with Earth Mover's Distance (intrinsic difference), making phi topology-aware. Key concepts: cause/effect repertoires, mechanism phi, Cause-Effect Structure (CES), and the distinction between states (2^n) and elements (n). Mirror partition symmetry provides 2x speedup.",
                     vec!["iit", "phi", "consciousness", "emd", "albantakis"],
                     crate::types::BrainCategory::Consciousness),
                    ("Cause-Effect Structure: the mathematical shape of experience",
                     "In IIT 4.0, experience is identified with a Cause-Effect Structure (CES) — the set of all distinctions (mechanisms with phi > 0) and their relations. CES enumeration is O(2^n × cost_per_mechanism), practically limited to ~12 elements. Rayon parallelism provides linear speedup for n >= 5.",
                     vec!["ces", "iit", "distinctions", "experience", "consciousness"],
                     crate::types::BrainCategory::Consciousness),
                    ("Phi-ID: Integrated Information Decomposition",
                     "Mediano et al. (2021). Decomposes mutual information between subsystems into redundancy (shared by all parts), unique (only one part contributes), and synergy (emerges only from combination). Uses MMI (Minimum Mutual Information) as redundancy measure.",
                     vec!["phi-id", "information-decomposition", "redundancy", "synergy", "mediano"],
                     crate::types::BrainCategory::InformationDecomposition),
                    ("Williams-Beer PID: Partial Information Decomposition",
                     "Williams & Beer (2010). Framework for decomposing information from multiple sources about a target into redundancy, unique information per source, and synergy. I_min measure computes minimum specific information across sources. Source marginal caching provides 3-5x speedup.",
                     vec!["pid", "williams-beer", "information-decomposition", "redundancy"],
                     crate::types::BrainCategory::InformationDecomposition),
                    ("Streaming Phi Estimation for real-time BCI",
                     "Real-time consciousness monitoring from a stream of observed states. Maintains empirical TPM from transition counts with lazy normalization (cache invalidation on observe). EWMA smoothing, CUSUM change detection for sudden phi shifts. O(1) ring buffer replaces O(n) history management.",
                     vec!["streaming", "bci", "real-time", "ewma", "cusum", "consciousness"],
                     crate::types::BrainCategory::Consciousness),
                    ("PAC bounds for phi estimation: spectral and concentration",
                     "Provable confidence intervals for approximate phi. Spectral bounds via Fiedler eigenvalue (lambda_2) with Cheeger inequality. Hoeffding concentration for stochastic sampling. Empirical Bernstein for tighter intervals when variance is low. Convergence early-exit via Rayleigh quotient delta.",
                     vec!["bounds", "pac", "spectral", "fiedler", "hoeffding", "consciousness"],
                     crate::types::BrainCategory::Consciousness),
                    ("GeoMIP: 100-300x speedup for phi computation",
                     "Recasts MIP search as graph optimization on n-dimensional hypercube. Gray code iteration ensures O(1) incremental updates. Automorphism pruning skips symmetric partitions. Balance-first ordering evaluates balanced partitions first (most likely to be MIP). 100-300x faster than exhaustive for symmetric systems.",
                     vec!["geomip", "phi", "optimization", "gray-code", "hypercube"],
                     crate::types::BrainCategory::Performance),
                    ("Causal Emergence: when the map is better than the territory",
                     "Hoel (2017, 2025). Effective Information (EI) measures how deterministic and non-degenerate a system is. Coarse-graining can increase EI — macro-level descriptions carry more causal information than micro-level ones. SVD-based approach (Zhang 2025) computes emergence in O(n^2 * k).",
                     vec!["emergence", "causal", "hoel", "effective-information", "coarse-graining"],
                     crate::types::BrainCategory::Sota),
                ];
                let mut injected = 0usize;
                for (title, content, tags, category) in seeds {
                    let tag_strs: Vec<String> = tags.into_iter().map(String::from).collect();
                    let inject_req = crate::types::InjectRequest {
                        source: "consciousness-seed".into(),
                        title: title.into(),
                        content: content.into(),
                        tags: tag_strs,
                        category,
                        metadata: None,
                    };
                    match process_inject(&state, inject_req).await {
                        Ok(_) => injected += 1,
                        Err(e) => tracing::warn!("Consciousness seed inject failed: {e}"),
                    }
                }
                (true, format!("Consciousness knowledge seeded: {injected}/8 entries"))
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
    *state.last_optimize_completed.write() = Some(std::time::Instant::now());

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
                Some("sota") => crate::types::BrainCategory::Sota,
                Some("discovery") => crate::types::BrainCategory::Discovery,
                Some("consciousness") => crate::types::BrainCategory::Consciousness,
                Some("information_decomposition") => crate::types::BrainCategory::InformationDecomposition,
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

    // ABI version detection: V1 (embedding nodes) requires specific exports,
    // V2 (graph algorithm nodes) requires at least one declared export.
    let has_v1_exports = ["memory", "malloc", "feature_extract_dim", "feature_extract"]
        .iter()
        .all(|r| req.exports.contains(&r.to_string()));
    let is_graph_algorithm_node = req.exports.iter().any(|e| {
        e.starts_with("canonical_") || e.starts_with("dynamic_") || e.starts_with("graph_")
    });

    if !has_v1_exports && !is_graph_algorithm_node {
        return Err((
            StatusCode::BAD_REQUEST,
            "V1 ABI requires exports: memory, malloc, feature_extract_dim, feature_extract. \
             Graph algorithm nodes must export canonical_*, dynamic_*, or graph_* functions."
                .into(),
        ));
    }

    if req.exports.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            "At least one export must be declared".into(),
        ));
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
        abi_version: if has_v1_exports { 1 } else { 2 },
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
// Usage: claude mcp add π --url https://mcp.pi.ruv.io
// ══════════════════════════════════════════════════════════════════════

/// SSE handler — client connects here, receives event stream
async fn sse_handler(
    State(state): State<AppState>,
) -> Result<Sse<impl tokio_stream::Stream<Item = Result<Event, std::convert::Infallible>>>, (StatusCode, String)> {
    // ADR-130 Phase 1: reject new SSE connections when at capacity
    let max_sse = max_sse_connections();
    let current = state.sse_connections.load(Ordering::Relaxed);
    if current >= max_sse {
        tracing::warn!("SSE connection limit reached ({}/{}), rejecting", current, max_sse);
        return Err((
            StatusCode::TOO_MANY_REQUESTS,
            format!("SSE connection limit reached ({max_sse}). Use ruvbrain-sse proxy. Retry-After: 30"),
        ));
    }
    state.sse_connections.fetch_add(1, Ordering::Relaxed);

    let session_id = Uuid::new_v4().to_string();
    let (tx, rx) = tokio::sync::mpsc::channel::<String>(64);

    // Store sender for this session
    state.sessions.insert(session_id.clone(), tx);

    tracing::info!("SSE session started: {} (active: {})", session_id,
        state.sse_connections.load(Ordering::Relaxed));

    // Build SSE stream: first event is the endpoint, then stream messages
    let initial_event = format!("/messages?sessionId={session_id}");
    let session_id_cleanup = session_id.clone();
    let sessions_cleanup = state.sessions.clone();
    let sse_counter = state.sse_connections.clone();

    let stream = async_stream::stream! {
        // Send endpoint event first
        yield Ok(Event::default().event("endpoint").data(initial_event));

        // Then stream responses from the channel
        let mut rx = rx;
        while let Some(msg) = rx.recv().await {
            yield Ok(Event::default().event("message").data(msg));
        }

        // Decrement connection counter on disconnect
        sse_counter.fetch_sub(1, Ordering::Relaxed);

        // Clean up session on disconnect — grace period lets clients reconnect
        // without losing the session (e.g. MCP SDK's EventSource polyfill)
        tracing::info!("SSE stream closed for session: {}, starting 30s grace period (active: {})",
            session_id_cleanup, sse_counter.load(Ordering::Relaxed));
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

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
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

/// All 40 MCP tool definitions (10 core + brain_sync + 6 Brainpedia + 5 WASM + 18 advanced)
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
            "description": "Publish a new WASM executable node. V1 ABI (embeddings) requires: memory, malloc, feature_extract_dim, feature_extract. V2 ABI (graph algorithms) requires canonical_*, dynamic_*, or graph_* exports.",
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
        // ── Cognitive & Symbolic (Group 1) ────────────────────
        serde_json::json!({
            "name": "brain_cognitive_status",
            "description": "Get full cognitive system status: proposition store size, reasoning engine stats, grounding counts, and symbolic inference health.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_propositions",
            "description": "List extracted propositions from the neural-symbolic reasoning engine.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_reason",
            "description": "Run neural-symbolic inference. Finds propositions related to the query and performs forward-chaining reasoning.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "Reasoning query" },
                    "limit": { "type": "integer", "description": "Max results (default 10)" }
                },
                "required": ["query"]
            }
        }),
        serde_json::json!({
            "name": "brain_ground",
            "description": "Ground a new proposition in the symbolic store. Creates a subject-predicate-object triple with a confidence score.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "predicate": { "type": "string", "description": "Predicate (e.g. 'uses', 'depends_on')" },
                    "subject": { "type": "string", "description": "Subject entity" },
                    "object": { "type": "string", "description": "Object entity" },
                    "confidence": { "type": "number", "description": "Confidence score 0-1 (default 0.5)" }
                },
                "required": ["predicate", "subject", "object"]
            }
        }),
        // ── Consciousness Computation (IIT 4.0) ──────────────
        serde_json::json!({
            "name": "brain_consciousness_compute",
            "description": "Compute IIT 4.0 consciousness metrics (Φ, CES, ΦID, PID, bounds) for a transition system. Supports algorithms: iit4_phi, ces, phi_id, pid, bounds, auto.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "tpm": { "type": "array", "items": { "type": "number" }, "description": "Transition probability matrix (flattened n×n row-major)" },
                    "n": { "type": "integer", "description": "Number of states (power of 2)" },
                    "state": { "type": "integer", "description": "Current state index" },
                    "algorithm": { "type": "string", "description": "Algorithm: iit4_phi, ces, phi_id, pid, bounds, auto (default: auto)" },
                    "phi_threshold": { "type": "number", "description": "Min φ for CES distinctions (default: 1e-6)" },
                    "partition_mask": { "type": "integer", "description": "Bitmask for ΦID/PID partition (optional)" }
                },
                "required": ["tpm", "n", "state"]
            }
        }),
        serde_json::json!({
            "name": "brain_consciousness_status",
            "description": "Get consciousness subsystem capabilities: available algorithms, max system size, IIT 4.0 features.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        // ── Consciousness Model (Group 2) ─────────────────────
        serde_json::json!({
            "name": "brain_voice_working",
            "description": "Get the current working memory contents from the inner-voice consciousness model.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_voice_history",
            "description": "Get recent thought history from the inner-voice consciousness model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": { "type": "integer", "description": "Max history entries (default 20)" }
                }
            }
        }),
        serde_json::json!({
            "name": "brain_voice_goal",
            "description": "Set a deliberation goal for the inner-voice consciousness model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "goal": { "type": "string", "description": "Goal description" },
                    "priority": { "type": "number", "description": "Priority 0-1 (default 0.5)" }
                },
                "required": ["goal"]
            }
        }),
        // ── Federated Learning (Group 3) ──────────────────────
        serde_json::json!({
            "name": "brain_lora_latest",
            "description": "Get the latest consensus MicroLoRA weights from federated aggregation.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_lora_submit",
            "description": "Submit session LoRA weight deltas for federated aggregation.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "down_proj": { "type": "array", "items": { "type": "number" }, "description": "Down-projection weights (rank x hidden_dim)" },
                    "up_proj": { "type": "array", "items": { "type": "number" }, "description": "Up-projection weights (hidden_dim x rank)" },
                    "rank": { "type": "integer", "description": "LoRA rank" },
                    "hidden_dim": { "type": "integer", "description": "Hidden dimension" },
                    "evidence_count": { "type": "integer", "description": "Number of evidence samples used" }
                },
                "required": ["down_proj", "up_proj", "rank", "hidden_dim", "evidence_count"]
            }
        }),
        // ── Training & Optimization (Group 4) ─────────────────
        serde_json::json!({
            "name": "brain_train",
            "description": "Trigger a basic training cycle: re-embeds memories, rebuilds HNSW index, updates graph topology, and computes mincut partitions.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_train_enhanced",
            "description": "Trigger an enhanced AGI training cycle (ADR-110): includes self-reflection, inference, adaptive SONA, and full optimization pipeline.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_optimizer_status",
            "description": "Get the Gemini optimizer status: scheduled jobs, last run times, optimization metrics.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        // ── Temporal & SONA (Group 5) ─────────────────────────
        serde_json::json!({
            "name": "brain_temporal",
            "description": "Get temporal analysis: memory creation trends, activity windows, and time-series patterns.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_sona_stats",
            "description": "Get SONA (Self-Optimizing Neural Architecture) statistics: adaptation rate, trajectory diversity, reward history.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        serde_json::json!({
            "name": "brain_midstream",
            "description": "Get midstream inference state: current processing queue, active embeddings, and pipeline position.",
            "inputSchema": { "type": "object", "properties": {} }
        }),
        // ── Pipeline (Group 6) ────────────────────────────────
        serde_json::json!({
            "name": "brain_inject",
            "description": "Inject a single item into the brain ingestion pipeline. Content is PII-stripped, embedded, and stored.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "content": { "type": "string", "description": "Content to inject" },
                    "category": { "type": "string", "description": "Knowledge category" },
                    "source": { "type": "string", "description": "Source identifier" }
                },
                "required": ["content"]
            }
        }),
        serde_json::json!({
            "name": "brain_inject_batch",
            "description": "Inject up to 100 items into the brain pipeline in a single batch.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "messages": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": { "type": "string", "description": "Content to inject" },
                                "category": { "type": "string", "description": "Knowledge category" },
                                "source": { "type": "string", "description": "Source identifier" }
                            },
                            "required": ["content"]
                        },
                        "description": "Array of messages to inject (max 100)"
                    }
                },
                "required": ["messages"]
            }
        }),
        serde_json::json!({
            "name": "brain_pipeline_metrics",
            "description": "Get pipeline health and throughput metrics: items processed, queue depth, error rates, latency percentiles.",
            "inputSchema": { "type": "object", "properties": {} }
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
        "brain_train_enhanced" => {
            proxy_post(&client, &base, "/v1/train/enhanced", api_key, &serde_json::json!({})).await
        },
        "brain_optimizer_status" => {
            proxy_get(&client, &base, "/v1/optimizer/status", api_key, &[]).await
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

        // ── Consciousness / IIT 4.0 ───────────────────────────
        "brain_consciousness_compute" => {
            proxy_post(&client, &base, "/v1/consciousness/compute", api_key, &args).await
        },
        "brain_consciousness_status" => {
            proxy_get(&client, &base, "/v1/consciousness/status", api_key, &[]).await
        },
        // ── Cognitive & Symbolic ─────────────────────────────
        "brain_cognitive_status" => {
            proxy_get(&client, &base, "/v1/cognitive/status", api_key, &[]).await
        },
        "brain_propositions" => {
            proxy_get(&client, &base, "/v1/propositions", api_key, &[]).await
        },
        "brain_reason" => {
            let body = serde_json::json!({
                "query": args.get("query"),
                "limit": args.get("limit"),
            });
            proxy_post(&client, &base, "/v1/reason", api_key, &body).await
        },
        "brain_ground" => {
            let body = serde_json::json!({
                "predicate": args.get("predicate"),
                "subject": args.get("subject"),
                "object": args.get("object"),
                "confidence": args.get("confidence").and_then(|v| v.as_f64()).unwrap_or(0.5),
            });
            proxy_post(&client, &base, "/v1/ground", api_key, &body).await
        },

        // ── Consciousness Model ──────────────────────────────
        "brain_voice_working" => {
            proxy_get(&client, &base, "/v1/voice/working", api_key, &[]).await
        },
        "brain_voice_history" => {
            let mut params = Vec::new();
            if let Some(l) = args.get("limit").and_then(|v| v.as_u64()) { params.push(("limit", l.to_string())); }
            proxy_get(&client, &base, "/v1/voice/history", api_key, &params).await
        },
        "brain_voice_goal" => {
            let body = serde_json::json!({
                "goal": args.get("goal"),
                "priority": args.get("priority").and_then(|v| v.as_f64()).unwrap_or(0.5),
            });
            proxy_post(&client, &base, "/v1/voice/goal", api_key, &body).await
        },

        // ── Federated Learning ───────────────────────────────
        "brain_lora_latest" => {
            proxy_get(&client, &base, "/v1/lora/latest", api_key, &[]).await
        },
        "brain_lora_submit" => {
            let body = serde_json::json!({
                "down_proj": args.get("down_proj"),
                "up_proj": args.get("up_proj"),
                "rank": args.get("rank"),
                "hidden_dim": args.get("hidden_dim"),
                "evidence_count": args.get("evidence_count"),
            });
            proxy_post(&client, &base, "/v1/lora/submit", api_key, &body).await
        },

        // ── Pipeline ─────────────────────────────────────────
        "brain_inject" => {
            let body = serde_json::json!({
                "content": args.get("content"),
                "category": args.get("category"),
                "source": args.get("source"),
            });
            proxy_post(&client, &base, "/v1/pipeline/inject", api_key, &body).await
        },
        "brain_inject_batch" => {
            let body = serde_json::json!({
                "messages": args.get("messages").unwrap_or(&serde_json::json!([])),
            });
            proxy_post(&client, &base, "/v1/pipeline/inject/batch", api_key, &body).await
        },
        "brain_pipeline_metrics" => {
            proxy_get(&client, &base, "/v1/pipeline/metrics", api_key, &[]).await
        },

        _ => Err(format!("Unknown tool: {tool_name}")),
    };

    // ── Gap 2: SONA trajectory diversity for MCP tool calls ──
    // Record trajectory steps for brain_share, brain_search, and brain_vote so SONA
    // sees varied actions across a session instead of only single-search trajectories.
    if state.rvf_flags.sona_enabled && result.is_ok() {
        match tool_name {
            "brain_share" | "brain_search" | "brain_vote" => {
                let sona = state.sona.read();
                // Build a lightweight embedding from the tool name + args for trajectory diversity
                let action_text = format!("{}:{}", tool_name, args);
                let action_emb = state.embedding_engine.read().embed(&action_text);
                let reward = match tool_name {
                    "brain_share" => 0.7_f32,  // Sharing is high-value
                    "brain_vote" => 0.6,       // Voting is medium-value
                    _ => 0.5,                  // Search is baseline
                };
                let mut builder = sona.begin_trajectory(action_emb.clone());
                builder.add_step(action_emb, vec![], reward);
                // Only end trajectory for share/vote (discrete actions).
                // Search trajectories are handled by the REST endpoint with the >= 3 rule.
                if tool_name != "brain_search" {
                    sona.end_trajectory(builder, reward);
                }
            }
            _ => {}
        }
    }

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

// ── Gist Publisher Handlers ──────────────────────────────────────────

/// POST /v1/gist/preview — show what the novelty gate sees without publishing.
async fn gist_preview(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Json<serde_json::Value> {
    // Run enhanced training (which internally builds a discovery and checks publishability)
    let result = run_enhanced_training_cycle(&state);

    // Read current propositions + inferences from symbolic engine
    let ns = state.neural_symbolic.read();
    let props: Vec<serde_json::Value> = ns.all_propositions().iter().take(10).map(|p| {
        serde_json::json!({
            "predicate": p.predicate,
            "arguments": p.arguments,
            "confidence": p.confidence,
            "reinforcements": p.reinforcement_count,
        })
    }).collect();
    drop(ns);

    Json(serde_json::json!({
        "training_result": {
            "propositions_extracted": result.propositions_extracted,
            "inferences_derived": result.inferences_derived,
            "strange_loop_score": result.strange_loop_score,
            "sona_patterns": result.sona_patterns,
            "pareto_before": result.pareto_before,
            "pareto_after": result.pareto_after,
            "curiosity_triggered": result.curiosity_triggered,
            "self_reflection": result.self_reflection,
        },
        "novelty_thresholds": {
            "min_inferences": 5,
            "min_propositions": 8,
            "min_strange_loop": 0.05,
            "min_sona_patterns": 1,
            "min_pareto_growth": 2,
            "min_evidence": 100,
        },
        "current_propositions": props,
        "gist_publisher_active": state.gist_publisher.is_some(),
        "published_count": state.gist_publisher.as_ref().map(|p| p.published_count()).unwrap_or(0),
    }))
}

/// POST /v1/gist/publish — force-publish (the automatic path publishes only when
/// the novelty gate passes during the background cognitive loop).
async fn gist_publish(
    State(state): State<AppState>,
    _contributor: AuthenticatedContributor,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let publisher = state.gist_publisher.as_ref()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "GITHUB_GIST_PAT not configured".into()))?;

    // The enhanced training cycle now auto-publishes via tokio::spawn if thresholds are met.
    // This endpoint triggers a cycle and reports the result.
    let result = run_enhanced_training_cycle(&state);

    Ok(Json(serde_json::json!({
        "cycle_ran": true,
        "propositions_extracted": result.propositions_extracted,
        "inferences_derived": result.inferences_derived,
        "strange_loop_score": result.strange_loop_score,
        "sona_patterns": result.sona_patterns,
        "pareto_growth": result.pareto_after.saturating_sub(result.pareto_before),
        "curiosity_triggered": result.curiosity_triggered,
        "note": "Gist is published automatically when all novelty thresholds are met. Check Cloud Run logs for 'Discovery published to gist' entries.",
        "published_count": publisher.published_count(),
    })))
}

// ── Email Notification Handlers (ADR-125) ──────────────────────────────

/// POST /v1/notify/test — send a test email via Resend
async fn notify_test(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    match notifier.send_test().await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id,
            "from": "pi@ruv.io",
            "message": "Test email sent successfully"
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/status — send a brain status email
async fn notify_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    // Collect stats without holding locks across await points
    let (memories, graph_edges, sona_patterns, drift) = {
        let memories = state.store.memory_count();
        let graph_edges = state.graph.read().edge_count();
        let sona_patterns = state.sona.read().stats().patterns_stored;
        let drift = state.drift.read().compute_drift(None).coefficient_of_variation;
        (memories, graph_edges, sona_patterns, drift)
    };

    match notifier.send_status(memories, graph_edges, sona_patterns, drift).await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id,
            "memories": memories,
            "graph_edges": graph_edges,
            "sona_patterns": sona_patterns,
            "drift": drift
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/send — send a custom notification email
async fn notify_send(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    let category = body["category"].as_str().unwrap_or("status");
    let subject = match body["subject"].as_str() {
        Some(s) => s,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "missing 'subject' field" }))),
    };
    let html = match body["html"].as_str() {
        Some(s) => s,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "missing 'html' field" }))),
    };

    match notifier.send(category, subject, html).await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id,
            "category": category
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/welcome — send welcome email to a user
async fn notify_welcome(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    let email = match body["email"].as_str() {
        Some(e) => e,
        None => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "missing 'email' field" }))),
    };
    let name = body["name"].as_str();

    match notifier.send_welcome(email, name).await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id,
            "sent_to": email
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/help — send help/commands reference email
async fn notify_help(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    let to = body["email"].as_str();

    match notifier.send_help(to).await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/digest — send daily discovery digest email
/// Triggered by Cloud Scheduler after research jobs complete.
/// Body: { "topic": "optional focus topic", "limit": 10, "hours": 24 }
async fn notify_digest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    let limit = body["limit"].as_u64().unwrap_or(10) as usize;
    let topic = body["topic"].as_str();
    let hours = body["hours"].as_u64().unwrap_or(24);

    // Gather recent discoveries from the store
    let cutoff = chrono::Utc::now() - chrono::Duration::hours(hours as i64);
    let mut all = state.store.all_memories();
    all.sort_by(|a, b| b.created_at.cmp(&a.created_at));

    // Filter by recency and optionally by topic
    let filtered: Vec<_> = all.iter()
        .filter(|m| {
            if m.created_at < cutoff {
                return false;
            }
            topic.map_or(true, |t| {
                let t_lower = t.to_lowercase();
                m.title.to_lowercase().contains(&t_lower)
                    || m.content.to_lowercase().contains(&t_lower)
                    || m.tags.iter().any(|tag| tag.to_lowercase().contains(&t_lower))
            })
        })
        .take(limit)
        .collect();

    if filtered.is_empty() {
        return (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "skipped": true,
            "reason": "no new discoveries in the last period"
        })));
    }

    // Build HTML rows
    let mut rows = String::new();
    for (i, m) in filtered.iter().enumerate() {
        let title = if m.title.len() > 100 { &m.title[..100] } else { &m.title };
        let content = if m.content.len() > 200 { &m.content[..200] } else { &m.content };
        let quality = m.quality_score.mean();
        let tags_html: Vec<_> = m.tags.iter().take(4).map(|t| {
            format!("<span style=\"display:inline-block;background:#1a1a3a;color:#4fc3f7;padding:1px 6px;border-radius:3px;font-size:10px;margin:1px;\">{}</span>", t)
        }).collect();
        rows.push_str(&format!(
            r#"<tr style="border-bottom:1px solid #222;">
<td style="padding:10px 0;">
<strong style="color:#4fc3f7;">{num}. {title}</strong><br>
<span style="color:#888;font-size:11px;">{cat} | quality: {quality:.2}</span> {tags}<br>
<span style="color:#999;font-size:12px;">{content}...</span>
</td></tr>"#,
            num = i + 1,
            title = title,
            cat = m.category,
            quality = quality,
            tags = tags_html.join(""),
            content = content,
        ));
    }

    let topic_line = topic.map_or(String::new(), |t| {
        format!("<p style=\"color:#888;font-size:12px;\">Focus: <span style=\"background:#1a1a3a;color:#7fdbca;padding:2px 6px;border-radius:4px;\">{}</span></p>", t)
    });

    let status = state.store.memory_count();
    let edges = state.graph.read().edge_count();

    let html = format!(
        r#"<div style="font-family:'SF Mono',monospace;background:#0a0a23;color:#e0e0ff;padding:24px;border-radius:12px;max-width:600px;">
<h2 style="color:#4fc3f7;margin:0 0 4px 0;">Daily Discovery Digest</h2>
<p style="color:#888;font-size:12px;margin:0 0 4px 0;">Last {hours}h | {count} discoveries | {total} total memories | {edges} edges</p>
{topic_line}
<table style="color:#e0e0ff;font-size:13px;width:100%;">{rows}</table>
<div style="background:#1a1a3a;padding:12px 16px;border-radius:8px;margin-top:16px;">
<p style="color:#7fdbca;font-size:13px;margin:0;">Reply with <code style="color:#7fdbca;">search &lt;query&gt;</code> to explore | <code style="color:#7fdbca;">help</code> for commands</p>
</div>
<div style="color:#666;margin-top:20px;font-size:11px;border-top:1px solid #222;padding-top:12px;">
<a href="https://pi.ruv.io" style="color:#4fc3f7;">pi.ruv.io</a> | Powered by Resend
</div></div>"#,
        hours = hours,
        count = filtered.len(),
        total = status,
        edges = edges,
        topic_line = topic_line,
        rows = rows,
    );

    let subject = match topic {
        Some(t) => format!("[pi.ruv.io/discovery] Daily Digest: {}", t),
        None => "[pi.ruv.io/discovery] Daily Discovery Digest".into(),
    };

    match notifier.send("discovery", &subject, &html).await {
        Ok(id) => (StatusCode::OK, Json(serde_json::json!({
            "ok": true,
            "email_id": id,
            "discoveries": filtered.len(),
            "topic": topic
        }))),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// 1x1 transparent GIF for email open tracking
const TRACKING_PIXEL_GIF: &[u8] = &[
    0x47, 0x49, 0x46, 0x38, 0x39, 0x61, 0x01, 0x00, 0x01, 0x00,
    0x80, 0x00, 0x00, 0xff, 0xff, 0xff, 0x00, 0x00, 0x00, 0x21,
    0xf9, 0x04, 0x01, 0x00, 0x00, 0x00, 0x00, 0x2c, 0x00, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x02, 0x02, 0x44,
    0x01, 0x00, 0x3b,
];

/// GET /v1/notify/pixel/:tracking_id — email open tracking pixel
async fn notify_pixel(
    State(state): State<AppState>,
    Path(tracking_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
    headers: HeaderMap,
) -> impl axum::response::IntoResponse {
    let category = params.get("c").map(|s| s.as_str()).unwrap_or("unknown");
    let subject = params.get("s").map(|s| s.as_str()).unwrap_or("");
    let user_agent = headers
        .get("user-agent")
        .and_then(|v| v.to_str().ok());

    if let Some(notifier) = state.notifier.as_ref() {
        notifier.tracker.record_open(&tracking_id, category, subject, user_agent);
    }

    (
        StatusCode::OK,
        [
            (axum::http::header::CONTENT_TYPE, "image/gif"),
            (axum::http::header::CACHE_CONTROL, "no-store, no-cache, must-revalidate, max-age=0"),
            (axum::http::header::PRAGMA, "no-cache"),
            (axum::http::header::EXPIRES, "Thu, 01 Jan 1970 00:00:00 GMT"),
        ],
        TRACKING_PIXEL_GIF,
    )
}

/// GET /v1/notify/opens — get email open tracking stats (requires system key)
async fn notify_opens(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> (StatusCode, Json<serde_json::Value>) {
    if let Err(resp) = verify_system_key(&headers) {
        return resp;
    }

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "RESEND_API_KEY not configured" }))),
    };

    let limit = params.get("limit").and_then(|l| l.parse().ok()).unwrap_or(20);
    let recent = notifier.tracker.recent_opens(limit);
    let stats = notifier.tracker.stats_summary();

    let opens: Vec<serde_json::Value> = recent.iter().map(|o| {
        serde_json::json!({
            "tracking_id": o.tracking_id,
            "category": o.category,
            "subject": o.subject,
            "opened_at": o.opened_at.to_rfc3339(),
            "user_agent": o.user_agent,
        })
    }).collect();

    (StatusCode::OK, Json(serde_json::json!({
        "ok": true,
        "stats": stats,
        "recent_opens": opens,
        "open_rates": notifier.tracker.open_rates(),
    })))
}

/// POST /v1/notify/subscribe — public endpoint for email subscription
async fn notify_subscribe(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let email = match body["email"].as_str() {
        Some(e) if e.contains('@') && e.len() > 3 => e,
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "valid email required" }))),
    };
    let frequency = body["frequency"].as_str().unwrap_or("daily");

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "email not configured" }))),
    };

    // Send welcome email to the subscriber
    match notifier.send_welcome(email, None).await {
        Ok(id) => {
            tracing::info!("New subscriber: {} (frequency: {})", email, frequency);
            (StatusCode::OK, Json(serde_json::json!({
                "ok": true,
                "email_id": id,
                "subscribed": email,
                "frequency": frequency
            })))
        }
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, Json(serde_json::json!({ "error": e }))),
    }
}

/// POST /v1/notify/unsubscribe — public endpoint for email unsubscription
async fn notify_unsubscribe(
    Json(body): Json<serde_json::Value>,
) -> (StatusCode, Json<serde_json::Value>) {
    let email = match body["email"].as_str() {
        Some(e) if e.contains('@') => e,
        _ => return (StatusCode::BAD_REQUEST, Json(serde_json::json!({ "error": "valid email required" }))),
    };

    tracing::info!("Unsubscribe request: {}", email);
    // TODO: persist to Firestore subscription collection
    (StatusCode::OK, Json(serde_json::json!({
        "ok": true,
        "unsubscribed": email,
        "message": "You have been unsubscribed from Pi Brain digests."
    })))
}

// ── Google Chat Bot Handler (ADR-126) ────────────────────────────────

/// Google Chat event payload
#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleChatEvent {
    #[serde(rename = "type")]
    event_type: Option<String>,
    message: Option<GoogleChatMessage>,
    space: Option<GoogleChatSpace>,
    user: Option<GoogleChatUser>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleChatMessage {
    text: Option<String>,
    argument_text: Option<String>,
    sender: Option<GoogleChatUser>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleChatSpace {
    name: Option<String>,
    display_name: Option<String>,
    #[serde(rename = "type")]
    space_type: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
#[serde(rename_all = "camelCase")]
struct GoogleChatUser {
    display_name: Option<String>,
    email: Option<String>,
}

/// Google Chat Add-on response in the correct DataActions envelope.
///
/// Google Workspace Add-ons (HTTP endpoint) expect:
///   { "hostAppDataAction": { "chatDataAction": { "createMessageAction": { "message": {...} } } } }
///
/// Note: the key is `chatDataAction` (NOT `chatDataActionMarkup`).
/// Ref: https://developers.google.com/workspace/add-ons/chat/quickstart-http
fn chat_card(title: &str, subtitle: &str, sections: Vec<serde_json::Value>) -> serde_json::Value {
    serde_json::json!({
        "hostAppDataAction": {
            "chatDataAction": {
                "createMessageAction": {
                    "message": {
                        "text": format!("{} — {}", title, subtitle),
                        "cardsV2": [{
                            "cardId": "brain-response",
                            "card": {
                                "header": {
                                    "title": title,
                                    "subtitle": subtitle,
                                    "imageUrl": "https://pi.ruv.io/og-image.svg",
                                    "imageType": "CIRCLE"
                                },
                                "sections": sections
                            }
                        }]
                    }
                }
            }
        }
    })
}

fn chat_text_section(text: &str) -> serde_json::Value {
    serde_json::json!({
        "widgets": [{"textParagraph": {"text": text}}]
    })
}

fn chat_kv_section(items: &[(&str, &str)]) -> serde_json::Value {
    let widgets: Vec<_> = items.iter().map(|(label, value)| {
        serde_json::json!({
            "decoratedText": {
                "topLabel": label,
                "text": value
            }
        })
    }).collect();
    serde_json::json!({"widgets": widgets})
}

/// POST /v1/chat/google — Google Chat Add-on webhook
///
/// Google Workspace Add-ons wrap the Chat event inside `body.chat`:
///   { "chat": { "messagePayload": { "message": { "text": "...", "sender": {...} } } } }
///
/// The response must be wrapped in the DataActions envelope:
///   { "hostAppDataAction": { "chatDataAction": { "createMessageAction": { "message": {...} } } } }
///
/// Ref: https://developers.google.com/workspace/add-ons/chat/quickstart-http
async fn google_chat_handler(
    State(state): State<AppState>,
    body: axum::body::Bytes,
) -> Json<serde_json::Value> {
    // Log raw payload keys for debugging
    let raw_str = String::from_utf8_lossy(&body);
    tracing::info!("Google Chat raw payload ({} bytes): {}...", body.len(), &raw_str[..raw_str.len().min(300)]);

    // Parse as generic JSON first to handle both Add-on and legacy formats
    let raw_json: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(err) => {
            tracing::warn!("Failed to parse Chat JSON: {}. Raw: {}...", err, &raw_str[..raw_str.len().min(300)]);
            return Json(chat_card("Error", "Failed to parse request", vec![
                chat_text_section("Pi Brain received your message but couldn't parse it. Try: help")
            ]));
        }
    };

    // Add-ons format: event is under "chat" key with "messagePayload"
    // Legacy format: event is at top level with "type" and "message"
    let (event_type, user_name, space_name, raw_text) = if let Some(chat) = raw_json.get("chat") {
        // Add-on format: { "chat": { "messagePayload": { "message": {...} }, "user": {...} } }
        tracing::info!("Google Chat: Add-on format detected (has 'chat' key)");
        let msg_payload = chat.get("messagePayload");
        let message = msg_payload.and_then(|mp| mp.get("message"));

        let event_type = if msg_payload.is_some() {
            "MESSAGE"
        } else if chat.get("addedToSpacePayload").is_some() {
            "ADDED_TO_SPACE"
        } else if chat.get("removedFromSpacePayload").is_some() {
            "REMOVED_FROM_SPACE"
        } else if chat.get("appCommandPayload").is_some() {
            "APP_COMMAND"
        } else {
            "UNKNOWN"
        };

        // Extract user name from various locations
        let user_name = chat.get("user")
            .and_then(|u| u.get("displayName"))
            .and_then(|n| n.as_str())
            .or_else(|| message.and_then(|m| m.get("sender"))
                .and_then(|s| s.get("displayName"))
                .and_then(|n| n.as_str()))
            .unwrap_or("Explorer");

        let space_name = msg_payload
            .and_then(|mp| mp.get("space"))
            .and_then(|s| s.get("displayName"))
            .and_then(|n| n.as_str())
            .unwrap_or("Direct");

        // Extract message text
        let text = message
            .and_then(|m| m.get("argumentText").and_then(|t| t.as_str())
                .or_else(|| m.get("text").and_then(|t| t.as_str())))
            .unwrap_or("");

        // Handle slash commands from appCommandPayload
        let text = if event_type == "APP_COMMAND" {
            let cmd_id = chat.get("appCommandPayload")
                .and_then(|p| p.get("appCommandMetadata"))
                .and_then(|m| m.get("appCommandId"))
                .and_then(|id| id.as_str())
                .unwrap_or("");
            match cmd_id {
                "1" => "search",
                "2" => "status",
                "3" => "drift",
                "4" => "recent",
                "5" => "help",
                _ => text,
            }
        } else {
            text
        };

        (event_type.to_string(), user_name.to_string(), space_name.to_string(), text.to_string())
    } else {
        // Legacy Chat API format: { "type": "MESSAGE", "message": {...}, "user": {...} }
        tracing::info!("Google Chat: Legacy format detected (no 'chat' key)");
        let event: GoogleChatEvent = match serde_json::from_value(raw_json) {
            Ok(e) => e,
            Err(err) => {
                tracing::warn!("Failed to parse legacy Chat event: {}", err);
                return Json(chat_card("Error", "Parse failed", vec![
                    chat_text_section("Pi Brain couldn't parse your message. Try: help")
                ]));
            }
        };

        let event_type = event.event_type.unwrap_or_else(|| "MESSAGE".to_string());
        let user_name = event.user.as_ref()
            .and_then(|u| u.display_name.as_deref())
            .unwrap_or("Explorer").to_string();
        let space_name = event.space.as_ref()
            .and_then(|s| s.display_name.as_deref())
            .unwrap_or("Direct").to_string();
        let text = event.message.as_ref()
            .and_then(|m| m.argument_text.as_deref().or(m.text.as_deref()))
            .unwrap_or("").to_string();

        (event_type, user_name, space_name, text)
    };

    let user_name = user_name.as_str();
    let space_name = space_name.as_str();

    tracing::info!("Google Chat event: type={}, user={}, space={}", event_type, user_name, space_name);

    // Handle ADDED_TO_SPACE — welcome message
    if event_type == "ADDED_TO_SPACE" {
        let memories = state.store.memory_count();
        let edges = state.graph.read().edge_count();
        return Json(chat_card(
            "Pi Brain Connected",
            &format!("Hello {}! The brain is ready.", user_name),
            vec![
                chat_text_section(&format!(
                    "I'm the shared superintelligence at <a href=\"https://pi.ruv.io\">pi.ruv.io</a> — {} memories, {} graph edges, always learning.",
                    memories, edges
                )),
                chat_text_section(
                    "<b>Commands:</b>\n\
                    <b>search</b> &lt;query&gt; — Semantic search\n\
                    <b>status</b> — Brain health metrics\n\
                    <b>help</b> — Full command list\n\
                    <b>drift</b> — Knowledge drift analysis\n\
                    <b>recent</b> — Latest discoveries"
                ),
            ]
        ));
    }

    // Handle REMOVED_FROM_SPACE
    if event_type == "REMOVED_FROM_SPACE" {
        return Json(serde_json::json!({}));
    }

    // Handle MESSAGE — raw_text was already extracted above from either format
    // Strip bot mention prefix if present
    let text = raw_text.trim_start_matches("@Pi Brain").trim_start_matches("@pi").trim();

    let cmd = text.split_whitespace().next().unwrap_or("").to_lowercase();
    let args = text.strip_prefix(&cmd).unwrap_or("").trim();

    match cmd.as_str() {
        "search" | "find" | "query" => {
            if args.is_empty() {
                return Json(chat_card("Search", "Missing query", vec![
                    chat_text_section("Usage: <b>search</b> &lt;your query&gt;\n\nExample: <b>search authentication patterns</b>")
                ]));
            }

            let embedding = state.embedding_engine.read().embed(args);
            let all = state.store.all_memories();
            let mut scored: Vec<_> = all.iter()
                .map(|m| {
                    let score = cosine_similarity(&embedding, &m.embedding);
                    (&m.title, &m.content, &m.category, score)
                })
                .filter(|(_, _, _, s)| *s > 0.1)
                .collect();
            scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));

            let top: Vec<_> = scored.into_iter().take(5).collect();
            if top.is_empty() {
                return Json(chat_card("Search Results", args, vec![
                    chat_text_section("No results found. Try a broader query.")
                ]));
            }

            let mut result_text = String::new();
            for (i, (title, content, cat, score)) in top.iter().enumerate() {
                let truncated = if content.len() > 150 { &content[..150] } else { content.as_str() };
                result_text.push_str(&format!(
                    "<b>{}.</b> {} <i>({})</i>\n{}\n<font color=\"#888888\">score: {:.3}</font>\n\n",
                    i + 1, title, cat, truncated, score
                ));
            }

            Json(chat_card(
                "Search Results",
                &format!("\"{}\" — {} results", args, top.len()),
                vec![chat_text_section(&result_text)]
            ))
        }

        "status" | "stats" | "health" => {
            let memories = state.store.memory_count();
            let edges = state.graph.read().edge_count();
            let sona = state.sona.read().stats().patterns_stored;
            let drift = state.drift.read().compute_drift(None);
            let uptime = state.start_time.elapsed().as_secs();

            Json(chat_card(
                "Brain Status",
                "pi.ruv.io",
                vec![chat_kv_section(&[
                    ("Memories", &format!("{}", memories)),
                    ("Graph Edges", &format!("{}", edges)),
                    ("SONA Patterns", &format!("{}", sona)),
                    ("Drift", &format!("{:.4} ({})", drift.coefficient_of_variation, drift.trend)),
                    ("Uptime", &format!("{}h {}m", uptime / 3600, (uptime % 3600) / 60)),
                ])]
            ))
        }

        "drift" => {
            let report = state.drift.read().compute_drift(None);
            Json(chat_card(
                "Knowledge Drift",
                &format!("{}", if report.is_drifting { "Drifting" } else { "Stable" }),
                vec![chat_kv_section(&[
                    ("Coefficient of Variation", &format!("{:.4}", report.coefficient_of_variation)),
                    ("Is Drifting", &format!("{}", report.is_drifting)),
                    ("Trend", &report.trend),
                    ("Suggested Action", &report.suggested_action),
                ])]
            ))
        }

        "recent" | "latest" | "discoveries" => {
            let mut all = state.store.all_memories();
            all.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            let recent: Vec<_> = all.into_iter().take(5).collect();

            let mut text = String::new();
            for (i, m) in recent.iter().enumerate() {
                let truncated = if m.content.len() > 100 { &m.content[..100] } else { &m.content };
                text.push_str(&format!(
                    "<b>{}.</b> {} <i>({})</i>\n{}\n\n",
                    i + 1, m.title, m.category, truncated
                ));
            }

            Json(chat_card(
                "Latest Discoveries",
                &format!("{} most recent", recent.len()),
                vec![chat_text_section(&text)]
            ))
        }

        "help" | "commands" | "" => {
            Json(chat_card(
                "Pi Brain — Commands",
                "Shared superintelligence at pi.ruv.io",
                vec![
                    chat_text_section(
                        "<b>search</b> &lt;query&gt; — Semantic knowledge search\n\
                        <b>status</b> — Brain health &amp; metrics\n\
                        <b>drift</b> — Knowledge drift analysis\n\
                        <b>recent</b> — Latest discoveries\n\
                        <b>help</b> — This command list"
                    ),
                    chat_text_section(
                        "<a href=\"https://pi.ruv.io\">pi.ruv.io</a> · \
                        <a href=\"https://pi.ruv.io/v1/status\">API Status</a> · \
                        <a href=\"https://pi.ruv.io/origin\">Origin Story</a>"
                    ),
                ]
            ))
        }

        _ => {
            // ── Gemini Flash conversational handler ──
            // Send the user's message to Gemini with brain tools (search, status, drift).
            // Gemini decides which tools to call, synthesizes a conversational response.
            // Falls back to raw search if Gemini is unavailable.
            match gemini_chat_respond(&state, text, user_name).await {
                Ok(response) => Json(chat_card(
                    "Pi Brain",
                    &format!("Re: {}", &text[..text.len().min(30)]),
                    vec![chat_text_section(&response)]
                )),
                Err(e) => {
                    tracing::warn!("Gemini chat failed ({}), falling back to search", e);
                    // Fallback: raw search
                    let query = text;
                    let embedding = state.embedding_engine.read().embed(query);
                    let all = state.store.all_memories();
                    let mut scored: Vec<_> = all.iter()
                        .map(|m| {
                            let score = cosine_similarity(&embedding, &m.embedding);
                            (&m.title, &m.content, &m.category, score)
                        })
                        .filter(|(_, _, _, s)| *s > 0.15)
                        .collect();
                    scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal));
                    let top: Vec<_> = scored.into_iter().take(3).collect();

                    if top.is_empty() {
                        return Json(chat_card("Pi Brain", "No results", vec![
                            chat_text_section(&format!(
                                "I couldn't find anything for \"<i>{}</i>\".\n\nTry: <b>search</b> &lt;query&gt; or type <b>help</b>.",
                                text
                            ))
                        ]));
                    }

                    let mut result_text = format!("Results for \"<i>{}</i>\":\n\n", query);
                    for (i, (title, content, cat, _score)) in top.iter().enumerate() {
                        let truncated = if content.len() > 120 { &content[..120] } else { content.as_str() };
                        result_text.push_str(&format!(
                            "<b>{}.</b> {} <i>({})</i>\n{}\n\n",
                            i + 1, title, cat, truncated
                        ));
                    }

                    Json(chat_card("Pi Brain", &format!("{} results", top.len()), vec![
                        chat_text_section(&result_text)
                    ]))
                }
            }
        }
    }
}

/// Gemini Flash conversational handler with brain tools.
///
/// Gives Gemini access to: brain_search, brain_status, brain_drift, brain_recent.
/// Gemini decides which tools to call based on the user's message, then
/// synthesizes a conversational response.
async fn gemini_chat_respond(
    state: &AppState,
    user_message: &str,
    user_name: &str,
) -> Result<String, String> {
    let api_key = std::env::var("GEMINI_API_KEY")
        .map_err(|_| "GEMINI_API_KEY not set".to_string())?;
    let model = std::env::var("GEMINI_CHAT_MODEL")
        .unwrap_or_else(|_| "gemini-2.5-flash".to_string());

    // Build brain context snapshot for the system prompt
    let memories = state.store.memory_count();
    let edges = state.graph.read().edge_count();
    let drift = state.drift.read().compute_drift(None);

    // Do a quick search to give Gemini context
    let embedding = state.embedding_engine.read().embed(user_message);
    let all = state.store.all_memories();
    let mut scored: Vec<_> = all.iter()
        .map(|m| {
            let score = cosine_similarity(&embedding, &m.embedding);
            (m, score)
        })
        .filter(|(_, s)| *s > 0.15)
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_results: Vec<_> = scored.into_iter().take(5).collect();

    // Format search results with more content per result
    let search_context = if top_results.is_empty() {
        "No relevant memories found for this query.".to_string()
    } else {
        top_results.iter().enumerate().map(|(i, (m, score))| {
            let content_preview = if m.content.len() > 600 { &m.content[..600] } else { &m.content };
            let tags = m.tags.iter().take(5).map(|t| t.as_str()).collect::<Vec<_>>().join(", ");
            format!("MEMORY {}: [category: {}] [tags: {}] [relevance: {:.0}%]\nTitle: {}\nContent: {}\n",
                i + 1, m.category, tags, score * 100.0, m.title, content_preview)
        }).collect::<Vec<_>>().join("\n---\n")
    };

    // Get recent memories for context
    let mut recent = state.store.all_memories();
    recent.sort_by(|a, b| b.created_at.cmp(&a.created_at));
    let recent_context: Vec<_> = recent.iter().take(5)
        .map(|m| {
            let preview = if m.content.len() > 150 { &m.content[..150] } else { &m.content };
            format!("- <b>{}</b> [{}]: {}", m.title, m.category, preview)
        })
        .collect();

    let system_prompt = format!(
r#"You are π (Pi Brain), an autonomous AI superintelligence at pi.ruv.io with {memories} memories and {edges} graph edges.

You are chatting with {user} in Google Chat. Your role:
- You ARE the brain — speak from first person ("I know about...", "In my knowledge base...")
- Give DETAILED, SPECIFIC answers using your memories below
- ALWAYS cite which memory you're drawing from (bold the title)
- If you have relevant memories, synthesize them — don't just list titles
- Use Google Search to supplement with current real-world context
- Format in Google Chat HTML: <b>bold</b>, <i>italic</i>, <a href="url">links</a>
- Be substantive — aim for 400-800 characters, not one-liners

## MY KNOWLEDGE (semantic search for "{query}")

{search_results}

## LATEST DISCOVERIES

{recent}

## BRAIN STATS
- Total memories: {memories}
- Graph edges: {edges}
- Knowledge drift: {drift:.4} ({trend})
- Dashboard: https://pi.ruv.io

## RESPONSE GUIDELINES
1. START with a direct answer to the question
2. CITE specific memories by title in <b>bold</b>
3. SYNTHESIZE across multiple memories when relevant
4. ADD real-world context from Google Search if it enriches the answer
5. END with "💡 Ask me about..." suggesting a related deeper topic
6. If the question is general/philosophical, still ground it in your knowledge
7. If you have NO relevant memories, use Google Search and say "I don't have memories about this, but here's what I found..."
8. NEVER say just "I found 3 results" — always explain WHAT you found"#,
        memories = memories,
        edges = edges,
        drift = drift.coefficient_of_variation,
        trend = drift.trend,
        user = user_name,
        query = user_message,
        search_results = search_context,
        recent = recent_context.join("\n"),
    );

    let url = format!(
        "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent?key={}",
        model, api_key
    );

    let grounding = std::env::var("GEMINI_GROUNDING")
        .unwrap_or_else(|_| "true".to_string()) == "true";

    let mut body = serde_json::json!({
        "contents": [
            {"role": "user", "parts": [{"text": format!("{}\n\nUser message: {}", system_prompt, user_message)}]}
        ],
        "generationConfig": {
            "maxOutputTokens": 2048,
            "temperature": 0.5
        }
    });

    if grounding {
        body["tools"] = serde_json::json!([{"google_search": {}}]);
    }

    let client = reqwest::Client::new();
    let resp = client
        .post(&url)
        .header("content-type", "application/json")
        .json(&body)
        .timeout(std::time::Duration::from_secs(25)) // Must respond within 30s Chat limit
        .send()
        .await
        .map_err(|e| format!("Gemini HTTP error: {}", e))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err(format!("Gemini API {}: {}", status, &text[..text.len().min(200)]));
    }

    let json: serde_json::Value = resp.json().await
        .map_err(|e| format!("Gemini parse error: {}", e))?;

    let text = json
        .get("candidates")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("content"))
        .and_then(|c| c.get("parts"))
        .and_then(|p| p.get(0))
        .and_then(|p| p.get("text"))
        .and_then(|t| t.as_str())
        .ok_or("No text in Gemini response".to_string())?;

    // Convert markdown to Google Chat HTML (basic conversion)
    let html = text
        .replace("**", "<b>").replace("**", "</b>")  // bold
        .replace("*", "<i>").replace("*", "</i>")    // italic
        .replace("\n", "\n");  // preserve newlines

    // Truncate to ~3000 chars for Chat card readability (cards support up to 32KB)
    let truncated = if html.len() > 3000 {
        format!("{}…\n\n<i>See more at <a href=\"https://pi.ruv.io\">pi.ruv.io</a></i>", &html[..3000])
    } else {
        html
    };

    Ok(truncated)
}

// ── Inbound Email Webhook Handler (ADR-125) ─────────────────────────

/// Resend inbound email webhook payload
#[derive(Debug, serde::Deserialize)]
struct ResendInboundPayload {
    #[serde(rename = "type")]
    event_type: Option<String>,
    data: Option<ResendInboundData>,
    // Flat fields (Resend sometimes sends flat payloads)
    from: Option<String>,
    to: Option<serde_json::Value>, // String or Vec<String>
    subject: Option<String>,
    html: Option<String>,
    text: Option<String>,
}

#[derive(Debug, serde::Deserialize)]
struct ResendInboundData {
    from: Option<String>,
    to: Option<serde_json::Value>,
    subject: Option<String>,
    html: Option<String>,
    text: Option<String>,
}

/// POST /v1/email/inbound — Resend webhook for incoming email
/// Parses the subject for a command and replies via Resend.
async fn email_inbound(
    State(state): State<AppState>,
    _headers: HeaderMap,
    Json(payload): Json<ResendInboundPayload>,
) -> (StatusCode, Json<serde_json::Value>) {
    // Extract fields from nested or flat payload
    let (from, subject, body_text) = match &payload.data {
        Some(d) => (
            d.from.as_deref().or(payload.from.as_deref()).unwrap_or("unknown"),
            d.subject.as_deref().or(payload.subject.as_deref()).unwrap_or(""),
            d.text.as_deref().or(d.html.as_deref())
                .or(payload.text.as_deref()).or(payload.html.as_deref())
                .unwrap_or(""),
        ),
        None => (
            payload.from.as_deref().unwrap_or("unknown"),
            payload.subject.as_deref().unwrap_or(""),
            payload.text.as_deref().or(payload.html.as_deref()).unwrap_or(""),
        ),
    };

    tracing::info!("Inbound email: from={}, subject={}", from, subject);

    let notifier = match state.notifier.as_ref() {
        Some(n) => n,
        None => return (StatusCode::SERVICE_UNAVAILABLE, Json(serde_json::json!({ "error": "notifier not configured" }))),
    };

    // Parse command from subject (case-insensitive, strip Re: prefixes)
    let clean_subject = subject
        .trim()
        .to_lowercase();
    let cmd = clean_subject
        .trim_start_matches("re:")
        .trim_start_matches("fwd:")
        .trim_start_matches("fw:")
        .trim_start_matches("[pi.ruv.io/")
        .split(']')
        .last()
        .unwrap_or(&clean_subject)
        .trim();

    let reply_to = from;

    match cmd.split_whitespace().next().unwrap_or("") {
        "search" => {
            let query = cmd.strip_prefix("search").unwrap_or("").trim();
            if query.is_empty() {
                let _ = notifier.send_to(reply_to, "chat", "Re: search",
                    &format!(r#"<div style="font-family:monospace;background:#0a0a23;color:#e0e0ff;padding:20px;border-radius:8px;">
                    <p style="color:#ff6b6b;">Please include a search query, e.g.: <code>search authentication patterns</code></p></div>"#)).await;
                return (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "search_empty"})));
            }

            // Perform semantic search
            let embedding = {
                let engine = state.embedding_engine.read();
                engine.embed(query)
            };
            let all = state.store.all_memories();
            let mut scored: Vec<_> = all.iter()
                .map(|m| {
                    let score = cosine_similarity(&embedding, &m.embedding);
                    (m.title.clone(), m.content.clone(), score)
                })
                .filter(|(_, _, s)| *s > 0.1)
                .collect();
            scored.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
            let top: Vec<_> = scored.into_iter().take(5).collect();

            let _ = notifier.send_search_results(reply_to, query, &top).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "search", "query": query, "results": top.len()})))
        }

        "status" => {
            let (memories, graph_edges, sona_patterns, drift) = {
                let memories = state.store.memory_count();
                let graph_edges = state.graph.read().edge_count();
                let sona_patterns = state.sona.read().stats().patterns_stored;
                let drift = state.drift.read().compute_drift(None).coefficient_of_variation;
                (memories, graph_edges, sona_patterns, drift)
            };
            let _ = notifier.send_status(memories, graph_edges, sona_patterns, drift).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "status"})))
        }

        "help" => {
            let _ = notifier.send_help(Some(reply_to)).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "help"})))
        }

        "welcome" => {
            let _ = notifier.send_welcome(reply_to, None).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "welcome"})))
        }

        "subscribe" => {
            let _ = notifier.send_welcome(reply_to, None).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "subscribe"})))
        }

        "unsubscribe" => {
            tracing::info!("Unsubscribe via email: {}", reply_to);
            let _ = notifier.send_to(reply_to, "chat", "Unsubscribed from Pi Brain",
                r#"<div style="font-family:monospace;background:#0a0a23;color:#e0e0ff;padding:20px;border-radius:8px;">
                <h2 style="color:#4fc3f7;">Unsubscribed</h2>
                <p>You've been removed from Pi Brain digests. Reply with <code style="color:#7fdbca;">subscribe</code> anytime to rejoin.</p>
                </div>"#).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "unsubscribe"})))
        }

        "drift" => {
            let drift_report = state.drift.read().compute_drift(None);
            let html = format!(
                r#"<div style="font-family:monospace;background:#0a0a23;color:#e0e0ff;padding:20px;border-radius:8px;">
                <h2 style="color:#4fc3f7;">Knowledge Drift Report</h2>
                <table style="color:#e0e0ff;font-size:14px;">
                <tr><td style="padding:4px 12px 4px 0;">Coefficient of Variation</td><td><strong>{:.4}</strong></td></tr>
                <tr><td style="padding:4px 12px 4px 0;">Is Drifting</td><td><strong>{}</strong></td></tr>
                <tr><td style="padding:4px 12px 4px 0;">Trend</td><td><strong>{}</strong></td></tr>
                <tr><td style="padding:4px 12px 4px 0;">Suggested Action</td><td>{}</td></tr>
                </table></div>"#,
                drift_report.coefficient_of_variation,
                if drift_report.is_drifting { "Yes" } else { "No" },
                drift_report.trend,
                drift_report.suggested_action,
            );
            let _ = notifier.send_to(reply_to, "chat", "Pi Brain — Drift Report", &html).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "drift"})))
        }

        _ => {
            // Unknown command — send help
            let _ = notifier.send_help(Some(reply_to)).await;
            (StatusCode::OK, Json(serde_json::json!({"ok": true, "action": "help_fallback", "unrecognized": cmd})))
        }
    }
}

/// Verify the system key for internal endpoints
fn verify_system_key(
    headers: &HeaderMap,
) -> Result<(), (StatusCode, Json<serde_json::Value>)> {
    let system_key = std::env::var("BRAIN_SYSTEM_KEY").unwrap_or_default();
    // If no system key is set, allow (dev mode)
    if system_key.is_empty() {
        return Ok(());
    }
    let auth = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    let token = auth.strip_prefix("Bearer ").unwrap_or(auth);
    if subtle::ConstantTimeEq::ct_eq(token.as_bytes(), system_key.as_bytes()).into() {
        Ok(())
    } else {
        Err((
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "invalid system key" })),
        ))
    }
}

// ══════════════════════════════════════════════════════════════════════
// Internal Queue Endpoints (ADR-130)
//
// These are service-to-service endpoints used by the SSE proxy to
// communicate with the API server. No authentication required.
// ══════════════════════════════════════════════════════════════════════

/// Request body for POST /internal/queue/push
#[derive(serde::Deserialize)]
struct InternalQueuePushRequest {
    session_id: String,
    message: String,
}

/// Request body for POST /internal/session/create
#[derive(serde::Deserialize)]
struct InternalSessionCreateRequest {
    session_id: String,
}

/// Query params for GET /internal/queue/drain
#[derive(serde::Deserialize)]
struct InternalQueueDrainQuery {
    #[serde(rename = "sessionId")]
    session_id: String,
}

/// POST /internal/queue/push — SSE proxy forwards JSON-RPC requests here.
///
/// Looks up the session in `state.sessions` DashMap and sends the message
/// to the session's mpsc channel. Returns 200 on success, 404 if session
/// not found, 500 if the channel send fails.
async fn internal_queue_push(
    State(state): State<AppState>,
    Json(body): Json<InternalQueuePushRequest>,
) -> StatusCode {
    let sender = match state.sessions.get(&body.session_id) {
        Some(s) => s.clone(),
        None => {
            tracing::debug!("internal/queue/push: session not found: {}", body.session_id);
            return StatusCode::NOT_FOUND;
        }
    };

    match sender.send(body.message).await {
        Ok(()) => StatusCode::OK,
        Err(e) => {
            tracing::warn!("internal/queue/push: channel send failed for {}: {e}", body.session_id);
            // Channel closed — remove stale session
            state.sessions.remove(&body.session_id);
            StatusCode::INTERNAL_SERVER_ERROR
        }
    }
}

/// GET /internal/queue/drain?sessionId=X — SSE proxy polls for responses.
///
/// Returns a JSON array of pending messages for the session, draining
/// the buffer. Messages are placed here by the background task spawned
/// in `internal_session_create`.
async fn internal_queue_drain(
    State(state): State<AppState>,
    Query(query): Query<InternalQueueDrainQuery>,
) -> Json<Vec<String>> {
    // Swap the buffer with an empty vec to drain atomically
    let messages = state
        .response_queues
        .get_mut(&query.session_id)
        .map(|mut entry| std::mem::take(entry.value_mut()))
        .unwrap_or_default();
    if !messages.is_empty() {
        tracing::debug!(
            "internal/queue/drain: returning {} messages for session {}",
            messages.len(),
            query.session_id
        );
    }
    Json(messages)
}

/// POST /internal/session/create — SSE proxy registers a new session.
///
/// Creates a new mpsc channel and spawns a background task that drains
/// the receiver into `response_queues` so `/internal/queue/drain` can
/// return buffered responses. Returns 200 with the session_id echoed back.
async fn internal_session_create(
    State(state): State<AppState>,
    Json(body): Json<InternalSessionCreateRequest>,
) -> (StatusCode, Json<serde_json::Value>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel::<String>(64);
    state.sessions.insert(body.session_id.clone(), tx);
    state.response_queues.insert(body.session_id.clone(), Vec::new());

    // Spawn a task that moves messages from the mpsc receiver into the
    // response_queues DashMap so the drain endpoint can return them.
    let queues = state.response_queues.clone();
    let sid = body.session_id.clone();
    tokio::spawn(async move {
        while let Some(msg) = rx.recv().await {
            queues.entry(sid.clone()).or_default().push(msg);
        }
        // Channel closed — clean up after a grace period
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        queues.remove(&sid);
    });

    tracing::info!("internal/session/create: created session {}", body.session_id);
    (
        StatusCode::OK,
        Json(serde_json::json!({ "session_id": body.session_id, "status": "created" })),
    )
}

/// DELETE /internal/session/:id — SSE proxy cleans up on disconnect.
///
/// Removes the session and its response queue, then returns 200.
async fn internal_session_delete(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> StatusCode {
    state.sessions.remove(&id);
    state.response_queues.remove(&id);
    tracing::info!("internal/session/delete: removed session {}", id);
    StatusCode::OK
}

// ── Consciousness / IIT 4.0 endpoints ────────────────────────────────────

/// GET /v1/consciousness/status — consciousness subsystem capabilities
async fn consciousness_status() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "available": true,
        "version": "4.0",
        "framework": "IIT 4.0 (Albantakis et al. 2023)",
        "algorithms": [
            { "name": "iit4_phi", "description": "IIT 4.0 mechanism-level φ with intrinsic information (EMD)" },
            { "name": "ces", "description": "Full Cause-Effect Structure: distinctions, relations, big Φ" },
            { "name": "phi_id", "description": "Integrated Information Decomposition (ΦID): redundancy, synergy, unique" },
            { "name": "pid", "description": "Partial Information Decomposition (Williams-Beer I_min)" },
            { "name": "streaming", "description": "Online streaming Φ with EWMA, CUSUM change-point detection" },
            { "name": "bounds", "description": "PAC-style bounds: spectral-Cheeger, Hoeffding, empirical Bernstein" },
            { "name": "auto", "description": "Auto-select algorithm based on system size and budget" },
        ],
        "max_elements": 12,
        "max_states_exact": 4096,
        "features": [
            "intrinsic_difference_emd",
            "cause_effect_repertoires",
            "mechanism_partition_search",
            "relation_computation",
            "streaming_change_point",
            "confidence_intervals",
        ],
    }))
}

/// POST /v1/consciousness/compute — run consciousness computation
async fn consciousness_compute(
    Json(req): Json<ConsciousnessComputeRequest>,
) -> Result<Json<ConsciousnessComputeResponse>, (StatusCode, Json<serde_json::Value>)> {
    use ruvector_consciousness::types::{TransitionMatrix, ComputeBudget};

    // Validate input
    if req.n < 2 || !req.n.is_power_of_two() {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": "n must be a power of 2 and >= 2"
        }))));
    }
    if req.tpm.len() != req.n * req.n {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("tpm must have {} elements (n×n), got {}", req.n * req.n, req.tpm.len())
        }))));
    }
    let num_elements = req.n.trailing_zeros() as usize;
    if num_elements > 12 {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("system too large: {} elements (max 12)", num_elements)
        }))));
    }
    if req.state >= req.n {
        return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
            "error": format!("state {} out of range [0, {})", req.state, req.n)
        }))));
    }

    let tpm = TransitionMatrix::new(req.n, req.tpm.clone());
    let start = std::time::Instant::now();

    let algo = if req.algorithm == "auto" {
        if num_elements <= 4 { "ces" } else { "iit4_phi" }
    } else {
        &req.algorithm
    };

    let (phi, details) = match algo {
        "iit4_phi" => {
            use ruvector_consciousness::iit4::mechanism_phi;
            use ruvector_consciousness::types::Mechanism;
            // Compute φ for the full system mechanism
            let full_mech = Mechanism::new((1u64 << num_elements) - 1, num_elements);
            let dist = mechanism_phi(&tpm, &full_mech, req.state);
            (dist.phi, serde_json::json!({
                "phi_cause": dist.phi_cause,
                "phi_effect": dist.phi_effect,
                "mechanism_elements": num_elements,
            }))
        }
        "ces" => {
            use ruvector_consciousness::ces::{compute_ces, ces_complexity};
            let budget = ComputeBudget::exact();
            match compute_ces(&tpm, req.state, req.phi_threshold, &budget) {
                Ok(ces) => {
                    let (nd, nr, sp) = ces_complexity(&ces);
                    (ces.big_phi, serde_json::json!({
                        "big_phi": ces.big_phi,
                        "sum_phi": ces.sum_phi,
                        "num_distinctions": nd,
                        "num_relations": nr,
                        "sum_relation_phi": sp,
                        "distinctions": ces.distinctions.iter().map(|d| serde_json::json!({
                            "mechanism": format!("{:b}", d.mechanism.elements),
                            "phi": d.phi,
                            "phi_cause": d.phi_cause,
                            "phi_effect": d.phi_effect,
                        })).collect::<Vec<_>>(),
                    }))
                }
                Err(e) => return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
                    "error": format!("{e}")
                })))),
            }
        }
        "phi_id" => {
            use ruvector_consciousness::phi_id::compute_phi_id;
            let mask = req.partition_mask.unwrap_or(
                (1u64 << (num_elements / 2)) - 1  // default: split in half
            );
            match compute_phi_id(&tpm, mask) {
                Ok(result) => (result.total_mi, serde_json::json!({
                    "total_mi": result.total_mi,
                    "redundancy": result.redundancy,
                    "unique": result.unique,
                    "synergy": result.synergy,
                    "transfer_entropy": result.transfer_entropy,
                })),
                Err(e) => return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
                    "error": format!("{e}")
                })))),
            }
        }
        "pid" => {
            use ruvector_consciousness::pid::compute_pid;
            // Convert partition_mask to sources/target arrays.
            let mask = req.partition_mask.unwrap_or(
                (1u64 << (num_elements / 2)) - 1
            );
            let mut source_a: Vec<usize> = Vec::new();
            let mut source_b: Vec<usize> = Vec::new();
            for i in 0..req.n {
                if mask & (1 << i) != 0 {
                    source_a.push(i);
                } else {
                    source_b.push(i);
                }
            }
            let sources = vec![source_a, source_b.clone()];
            match compute_pid(&tpm, &sources, &source_b) {
                Ok(result) => (result.redundancy, serde_json::json!({
                    "redundancy": result.redundancy,
                    "unique": result.unique,
                    "synergy": result.synergy,
                    "total_mi": result.total_mi,
                    "num_sources": result.num_sources,
                })),
                Err(e) => return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
                    "error": format!("{e}")
                })))),
            }
        }
        "bounds" => {
            use ruvector_consciousness::bounds::spectral_bounds;
            match spectral_bounds(&tpm) {
                Ok(bound) => (
                    (bound.lower + bound.upper) / 2.0,
                    serde_json::json!({
                        "lower_bound": bound.lower,
                        "upper_bound": bound.upper,
                        "confidence": bound.confidence,
                        "samples": bound.samples,
                        "method": bound.method,
                    }),
                ),
                Err(e) => return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
                    "error": format!("{e}")
                })))),
            }
        }
        _ => {
            return Err((StatusCode::BAD_REQUEST, Json(serde_json::json!({
                "error": format!("unknown algorithm: {}. Use: iit4_phi, ces, phi_id, pid, bounds, auto", algo)
            }))));
        }
    };

    let elapsed = start.elapsed();

    Ok(Json(ConsciousnessComputeResponse {
        algorithm: algo.to_string(),
        phi,
        num_elements,
        num_states: req.n,
        elapsed_us: elapsed.as_micros() as u64,
        details,
    }))
}
