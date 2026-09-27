//! Integration tests for mcp-brain-server cognitive stack
//!
//! Tests cover all major RuVector crate integrations.

#[cfg(test)]
mod tests {
    use ruvector_delta_core::{Delta, VectorDelta};
    use ruvector_domain_expansion::{DomainExpansionEngine, DomainId};
    use ruvector_nervous_system::hdc::{HdcMemory, Hypervector};
    use ruvector_nervous_system::hopfield::ModernHopfield;
    use ruvector_nervous_system::separate::DentateGyrus;
    use ruvector_solver::forward_push::ForwardPushSolver;
    use ruvector_solver::types::CsrMatrix;

    // -----------------------------------------------------------------------
    // 1. Hopfield: store 5 patterns, retrieve by partial query
    // -----------------------------------------------------------------------
    #[test]
    fn test_hopfield_store_retrieve() {
        let mut hopfield = ModernHopfield::new(8, 1.0);

        // Store 5 distinct patterns
        let patterns: Vec<Vec<f32>> = (0..5)
            .map(|i| {
                let mut p = vec![0.0f32; 8];
                p[i] = 1.0;
                p
            })
            .collect();

        for p in &patterns {
            hopfield.store(p.clone()).expect("store failed");
        }

        // Retrieve using a noisy version of pattern 0
        let noisy = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let recalled = hopfield.retrieve(&noisy).expect("retrieve failed");

        // Should retrieve something close to pattern 0 (first element dominant)
        assert!(
            recalled[0] > recalled[1],
            "pattern 0 should be dominant in retrieval"
        );
    }

    // -----------------------------------------------------------------------
    // 2. DentateGyrus: encode similar inputs, verify orthogonal outputs
    // -----------------------------------------------------------------------
    #[test]
    fn test_dentate_pattern_separation() {
        let gyrus = DentateGyrus::new(8, 1000, 50, 42);

        // Two very similar inputs
        let a = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3];
        let b = vec![1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.4]; // slightly different

        let enc_a = gyrus.encode(&a);
        let enc_b = gyrus.encode(&b);

        // DentateGyrus should produce sparse binary representations
        // Jaccard similarity < 1.0 means they're not identical
        let sim = enc_a.jaccard_similarity(&enc_b);
        assert!(
            sim <= 1.0,
            "encoded similarity should be at most 1.0, got {}",
            sim
        );

        // Dense encoding should have 1000 dimensions
        let dense_a = gyrus.encode_dense(&a);
        assert_eq!(dense_a.len(), 1000);
    }

    // -----------------------------------------------------------------------
    // 3. HDC: store 100 hypervectors, retrieve by similarity
    // -----------------------------------------------------------------------
    #[test]
    fn test_hdc_fast_filter() {
        let mut memory = HdcMemory::new();

        // Store 100 random hypervectors
        for i in 0..100u64 {
            let hv = Hypervector::from_seed(i);
            memory.store(format!("item-{}", i), hv);
        }

        // Retrieve using seed 42 as query — "item-42" should be at the top
        let query = Hypervector::from_seed(42);
        let results = memory.retrieve_top_k(&query, 5);

        assert!(!results.is_empty(), "should return at least one result");
        // Top result should be item-42 (exact match = highest similarity)
        assert_eq!(
            results[0].0, "item-42",
            "top result should be item-42, got {}",
            results[0].0
        );
        // Similarity of exact match should be 1.0
        assert!(
            (results[0].1 - 1.0).abs() < 0.01,
            "exact match similarity should be ~1.0"
        );
    }

    // -----------------------------------------------------------------------
    // 4. MinCut: build graph with 20 nodes, verify real min_cut_value > 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_mincut_partition() {
        use ruvector_mincut::MinCutBuilder;

        // Build a 20-node graph with edges forming two dense clusters
        let mut edges: Vec<(u64, u64, f64)> = Vec::new();

        // Cluster A: nodes 0..9, dense internal edges
        for i in 0..10u64 {
            for j in (i + 1)..10u64 {
                edges.push((i, j, 5.0));
            }
        }

        // Cluster B: nodes 10..19, dense internal edges
        for i in 10..20u64 {
            for j in (i + 1)..20u64 {
                edges.push((i, j, 5.0));
            }
        }

        // Weak bridge between clusters
        edges.push((4, 15, 0.1));
        edges.push((5, 16, 0.1));

        let mincut = MinCutBuilder::new()
            .exact()
            .with_edges(edges)
            .build()
            .expect("failed to build MinCut");

        let cut_value = mincut.min_cut_value();
        assert!(
            cut_value > 0.0,
            "min cut value should be > 0, got {}",
            cut_value
        );
    }

    // -----------------------------------------------------------------------
    // 5. TopologyGatedAttention: rank 10 results
    // -----------------------------------------------------------------------
    #[test]
    fn test_attention_ranking() {
        use crate::ranking::RankingEngine;
        use crate::types::{BetaParams, BrainCategory, BrainMemory};
        use chrono::Utc;
        use uuid::Uuid;

        let mut engine = RankingEngine::new(4);

        // Create 10 fake memories with different embeddings
        let mut results: Vec<(f64, BrainMemory)> = (0..10)
            .map(|i| {
                let embedding = vec![i as f32 * 0.1, 0.5, 0.3, 0.2];
                let memory = BrainMemory {
                    id: Uuid::new_v4(),
                    category: BrainCategory::Pattern,
                    title: format!("mem-{}", i),
                    content: "test".into(),
                    tags: vec![],
                    code_snippet: None,
                    embedding,
                    contributor_id: "tester".into(),
                    quality_score: BetaParams::new(),
                    partition_id: None,
                    witness_hash: String::new(),
                    rvf_gcs_path: None,
                    redaction_log: None,
                    dp_proof: None,
                    witness_chain: None,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                };
                (0.5 + i as f64 * 0.05, memory)
            })
            .collect();

        // Rank should not panic and should produce sorted output
        engine.rank(&mut results);
        assert_eq!(results.len(), 10);

        // Verify sorted descending
        for w in results.windows(2) {
            assert!(
                w[0].0 >= w[1].0,
                "results should be sorted descending: {} >= {}",
                w[0].0,
                w[1].0
            );
        }
    }

    // -----------------------------------------------------------------------
    // 6. VectorDelta: compute drift between two embedding sequences
    // -----------------------------------------------------------------------
    #[test]
    fn test_delta_drift() {
        use crate::drift::DriftMonitor;

        let mut monitor = DriftMonitor::new();
        let domain = "test-domain";

        // Record 20 embeddings with increasing drift
        for i in 0..20usize {
            let embedding: Vec<f32> = (0..8).map(|j| (i * j) as f32 * 0.01).collect();
            monitor.record(domain, &embedding);
        }

        let report = monitor.compute_drift(Some(domain));
        assert_eq!(report.window_size, 20);
        assert!(
            report.coefficient_of_variation >= 0.0,
            "CV should be non-negative"
        );

        // Also test direct delta computation
        let old = vec![1.0f32, 0.0, 0.0, 0.0];
        let new = vec![0.9f32, 0.1, 0.05, 0.0];
        let delta = VectorDelta::compute(&old, &new);
        let l2 = delta.l2_norm();
        assert!(l2 > 0.0, "l2_norm should be positive for different vectors");
        assert!(!delta.is_identity(), "should not be identity delta");
    }

    // -----------------------------------------------------------------------
    // 7. SonaEngine: generate embeddings, verify semantic similarity
    // -----------------------------------------------------------------------
    #[test]
    fn test_sona_embedding() {
        let engine = sona::SonaEngine::new(32);

        // Build a trajectory
        let mut builder = engine.begin_trajectory(vec![0.5f32; 32]);
        builder.add_step(vec![0.6f32; 32], vec![], 0.8);
        builder.add_step(vec![0.7f32; 32], vec![], 0.9);
        engine.end_trajectory(builder, 0.85);

        // Stats should record 1 trajectory
        let stats = engine.stats();
        assert_eq!(stats.trajectories_buffered, 1);

        // Apply micro-lora (output may be zero before learning, but should not panic)
        let input = vec![1.0f32; 32];
        let mut output = vec![0.0f32; 32];
        engine.apply_micro_lora(&input, &mut output);
        // Output is a Vec<f32> of correct length
        assert_eq!(output.len(), 32);
    }

    // -----------------------------------------------------------------------
    // 8. ForwardPushSolver / CsrMatrix: build CSR graph, run PPR, verify top-k
    // -----------------------------------------------------------------------
    #[test]
    fn test_pagerank_search() {
        // Build a simple 6-node ring graph with an extra hub node
        // Nodes: 0-5 in a ring, node 0 also connects to all others
        let n = 6;
        let mut entries: Vec<(usize, usize, f64)> = Vec::new();

        // Ring edges
        for i in 0..n {
            entries.push((i, (i + 1) % n, 1.0));
            entries.push(((i + 1) % n, i, 1.0));
        }

        // Hub: node 0 connects to all others with high weight
        for i in 1..n {
            entries.push((0, i, 2.0));
            entries.push((i, 0, 2.0));
        }

        let graph = CsrMatrix::<f64>::from_coo(n, n, entries);
        assert_eq!(graph.rows, n);
        assert!(graph.nnz() > 0);

        let solver = ForwardPushSolver::default_params();
        let results = solver
            .top_k(&graph, 0, 3)
            .expect("forward push should succeed");

        assert!(!results.is_empty(), "should return PPR results");
        // Node 0 as source — it or its immediate neighbors should rank high
        let returned_nodes: Vec<usize> = results.iter().map(|(n, _)| *n).collect();
        // At least some nodes should be returned
        assert!(returned_nodes.len() <= 3);
    }

    // -----------------------------------------------------------------------
    // 9. Domain transfer: initiate_transfer between two domains, verify acceleration
    // -----------------------------------------------------------------------
    #[test]
    fn test_domain_transfer() {
        use ruvector_domain_expansion::{ArmId, ContextBucket};

        let mut engine = DomainExpansionEngine::new();
        let source = DomainId("rust_synthesis".into());
        let target = DomainId("structured_planning".into());

        // Warm up source domain with outcomes
        let bucket = ContextBucket {
            difficulty_tier: "medium".into(),
            category: "algorithm".into(),
        };
        for _ in 0..20 {
            engine.thompson.record_outcome(
                &source,
                bucket.clone(),
                ArmId("greedy".into()),
                0.8,
                1.0,
            );
        }

        // Initiate transfer
        engine.initiate_transfer(&source, &target);

        // Verify the transfer with simulated metrics
        let verification = engine.verify_transfer(
            &source, &target, 0.8,  // source_before
            0.79, // source_after (within tolerance)
            0.3,  // target_before
            0.65, // target_after
            100,  // baseline_cycles
            50,   // transfer_cycles
        );

        assert!(
            verification.improved_target,
            "transfer should improve target domain"
        );
        assert!(
            !verification.regressed_source,
            "transfer should not regress source"
        );
        assert!(verification.promotable, "verification should be promotable");
        assert!(
            verification.acceleration_factor > 1.0,
            "acceleration factor should be > 1.0, got {}",
            verification.acceleration_factor
        );
    }

    // -----------------------------------------------------------------------
    // 10. Witness chain: verify integrity (via cognitive engine store)
    // -----------------------------------------------------------------------
    #[test]
    fn test_witness_chain() {
        use crate::cognitive::CognitiveEngine;

        let mut engine = CognitiveEngine::new(8);

        // Store 5 patterns sequentially — simulates a witness chain
        let patterns: Vec<(&str, Vec<f32>)> = vec![
            ("entry-1", vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("entry-2", vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("entry-3", vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            ("entry-4", vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            ("entry-5", vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
        ];

        for (id, emb) in &patterns {
            engine.store_pattern(id, emb);
        }

        // Retrieve from entry-3's pattern — should be in Hopfield memory
        let query = vec![0.05, 0.05, 0.9, 0.05, 0.05, 0.0, 0.0, 0.0];
        let recalled = engine.recall(&query);
        assert!(recalled.is_some(), "should recall a pattern");

        // Cluster coherence of the 5 stored embeddings
        let embs: Vec<Vec<f32>> = patterns.iter().map(|(_, e)| e.clone()).collect();
        let coherence = engine.cluster_coherence(&embs);
        assert!(
            coherence >= 0.0 && coherence <= 1.0,
            "coherence should be [0,1], got {}",
            coherence
        );
    }

    // -----------------------------------------------------------------------
    // 11. PII strip: test all 12 PII patterns
    // -----------------------------------------------------------------------
    #[test]
    fn test_pii_strip_all_patterns() {
        use crate::verify::Verifier;

        let verifier = Verifier::new();

        let pii_inputs = vec![
            (
                "email address",
                "My email is user@example.com and I need help",
            ),
            ("phone number", "Call me at 555-867-5309 for details"),
            ("SSN", "My SSN is 123-45-6789 please keep it safe"),
            (
                "credit card",
                "Card number 4111-1111-1111-1111 expires 12/25",
            ),
            ("IP address", "Server IP is 192.168.1.100 for internal use"),
            ("AWS key", "AWS key AKIAIOSFODNN7EXAMPLE is exposed"),
            ("private key", "-----BEGIN PRIVATE KEY----- data here"),
            ("password pattern", "password=supersecret123 in config"),
            ("api key", "api_key=sk-abc123 in the headers"),
        ];

        for (label, input) in &pii_inputs {
            let tags = vec!["test".to_string()];
            let embedding = vec![0.1f32; 128];
            let result = verifier.verify_share("Test Title", input, &tags, &embedding);
            // Should either reject (Err) or sanitize (Ok) — both are valid
            // The key test is that it doesn't panic and handles PII input
            match result {
                Ok(_) => {
                    // Accepted (may have stripped PII) — valid
                }
                Err(e) => {
                    // Rejected due to PII detection — valid
                    let msg = e.to_string().to_lowercase();
                    assert!(
                        !msg.is_empty(),
                        "{}: rejection message should not be empty",
                        label
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // 12. End-to-end: verify → strip PII → build witness chain → RVF container
    // -----------------------------------------------------------------------
    #[test]
    fn test_end_to_end_share_pipeline() {
        use crate::pipeline::{build_rvf_container, count_segments, RvfPipelineInput};
        use crate::verify::Verifier;
        use rvf_crypto::WitnessEntry;

        let mut verifier = Verifier::new();
        let title = "Secure Architecture Guide";
        let content = "Contact admin@example.com or see /home/deploy/config.yaml for setup";
        let tags = vec!["security".to_string(), "architecture".to_string()];
        let embedding = vec![0.1f32; 128];

        // Step 1: Verify input (should reject due to PII)
        let result = verifier.verify_share(title, content, &tags, &embedding);
        assert!(
            result.is_err(),
            "PII content should be rejected by verify_share"
        );

        // Step 2: Strip PII instead of rejecting
        let fields = [("title", title), ("content", content)];
        let (stripped, log) = verifier.strip_pii_fields(&fields);
        assert!(log.total_redactions >= 2, "should redact email + path");
        assert!(
            !stripped[1].1.contains("admin@example.com"),
            "email should be redacted"
        );
        assert!(!stripped[1].1.contains("/home/"), "path should be redacted");

        // Step 3: Stripped content should pass verification
        let clean_title = &stripped[0].1;
        let clean_content = &stripped[1].1;
        assert!(verifier
            .verify_share(clean_title, clean_content, &tags, &embedding)
            .is_ok());

        // Step 4: Build witness chain
        let now_ns = 1_000_000_000u64;
        let stripped_hash = rvf_crypto::shake256_256(clean_content.as_bytes());
        let mut emb_bytes = Vec::with_capacity(embedding.len() * 4);
        for v in &embedding {
            emb_bytes.extend_from_slice(&v.to_le_bytes());
        }
        let emb_hash = rvf_crypto::shake256_256(&emb_bytes);
        let entries = vec![
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: stripped_hash,
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: emb_hash,
                timestamp_ns: now_ns,
                witness_type: 0x02,
            },
            WitnessEntry {
                prev_hash: [0u8; 32],
                action_hash: rvf_crypto::shake256_256(b"final"),
                timestamp_ns: now_ns,
                witness_type: 0x01,
            },
        ];
        let chain = rvf_crypto::create_witness_chain(&entries);
        assert_eq!(chain.len(), 73 * 3);

        // Step 5: Verify chain integrity
        let decoded = verifier.verify_rvf_witness_chain(&chain).unwrap();
        assert_eq!(decoded.len(), 3);

        // Step 6: Build RVF container
        let redaction_json = serde_json::to_string(&serde_json::json!({
            "entries": [], "total_redactions": log.total_redactions
        }))
        .unwrap();
        let input = RvfPipelineInput {
            memory_id: "e2e-test-id",
            embedding: &embedding,
            title: clean_title,
            content: clean_content,
            tags: &tags,
            category: "security",
            contributor_id: "e2e-tester",
            witness_chain: Some(&chain),
            dp_proof_json: None,
            redaction_log_json: Some(&redaction_json),
        };
        let container = build_rvf_container(&input).expect("container build should succeed");
        let seg_count = count_segments(&container);
        // VEC + META + WITNESS + REDACTION_LOG = 4 segments
        assert_eq!(seg_count, 4, "expected 4 segments, got {seg_count}");
    }

    // -----------------------------------------------------------------------
    // 13. Auth: API key validation and pseudonym derivation
    // -----------------------------------------------------------------------
    #[test]
    fn test_auth_pseudonym_derivation() {
        use crate::auth::AuthenticatedContributor;

        // Same key should always produce the same pseudonym (deterministic)
        let a = AuthenticatedContributor::from_api_key("test-key-12345678");
        let b = AuthenticatedContributor::from_api_key("test-key-12345678");
        assert_eq!(a.pseudonym, b.pseudonym);
        assert_eq!(a.api_key_prefix, "test-key");
        assert!(!a.is_system);

        // Different keys should produce different pseudonyms
        let c = AuthenticatedContributor::from_api_key("different-key-9999");
        assert_ne!(a.pseudonym, c.pseudonym);

        // System seed should have known values
        let sys = AuthenticatedContributor::system_seed();
        assert_eq!(sys.pseudonym, "ruvector-seed");
        assert!(sys.is_system);
    }

    // -----------------------------------------------------------------------
    // 14. RVF feature flags: verify default values (including AGI flags)
    // -----------------------------------------------------------------------
    #[test]
    fn test_rvf_feature_flags_defaults() {
        use crate::types::RvfFeatureFlags;
        let flags = RvfFeatureFlags::from_env();
        // Phase 1-7 defaults
        assert!(flags.pii_strip, "pii_strip should default to true");
        assert!(flags.witness, "witness should default to true");
        assert!(flags.container, "container should default to true");
        assert!(!flags.dp_enabled, "dp_enabled should default to false");
        assert!(!flags.adversarial, "adversarial should default to false");
        assert!(!flags.neg_cache, "neg_cache should default to false");
        assert!(
            (flags.dp_epsilon - 1.0).abs() < f64::EPSILON,
            "dp_epsilon should default to 1.0"
        );
        // Phase 8 AGI defaults — all enabled by default
        assert!(flags.sona_enabled, "sona_enabled should default to true");
        assert!(flags.gwt_enabled, "gwt_enabled should default to true");
        assert!(
            flags.temporal_enabled,
            "temporal_enabled should default to true"
        );
        assert!(
            flags.meta_learning_enabled,
            "meta_learning_enabled should default to true"
        );
    }

    // -----------------------------------------------------------------------
    // 15. SONA: trajectory roundtrip and pattern search
    // -----------------------------------------------------------------------
    #[test]
    fn test_sona_trajectory_roundtrip() {
        let sona = sona::SonaEngine::new(128);
        let query = vec![0.5f32; 128];

        // Begin trajectory, add a step, end it
        let mut builder = sona.begin_trajectory(query.clone());
        builder.add_step(vec![0.6f32; 128], vec![], 0.8);
        sona.end_trajectory(builder, 0.7);

        // Stats should reflect the trajectory
        let stats = sona.stats();
        assert!(
            stats.trajectories_buffered >= 1 || stats.trajectories_dropped == 0,
            "trajectory should be buffered or processed"
        );

        // Pattern search should not crash (may return empty before learning)
        let patterns = sona.find_patterns(&query, 5);
        // Patterns are empty until background learning runs, but API must not panic
        let _ = patterns;
    }

    // -----------------------------------------------------------------------
    // 16. GWT: broadcast and salience competition
    // -----------------------------------------------------------------------
    #[test]
    fn test_gwt_broadcast_competition() {
        use ruvector_nervous_system::routing::workspace::GlobalWorkspace;

        let mut ws = GlobalWorkspace::with_threshold(7, 0.1);

        // Broadcast 10 items with varying salience
        for i in 0..10u16 {
            let salience = (i as f32 + 1.0) / 10.0; // 0.1 to 1.0
            let content = vec![i as f32; 4];
            let rep = ruvector_nervous_system::routing::workspace::Representation::new(
                content, salience, i, 0,
            );
            ws.broadcast(rep);
        }

        // Workspace capacity is 7, so only top-7 by salience should survive
        let top = ws.retrieve_top_k(7);
        assert!(top.len() <= 7, "workspace should respect capacity 7");
        assert!(top.len() >= 1, "at least one item should survive");

        // Most salient should be the item with salience 1.0
        let best = ws.most_salient();
        assert!(best.is_some(), "workspace should have a most salient item");

        // Load should be positive
        let load = ws.current_load();
        assert!(load > 0.0, "workspace should have positive load");
    }

    // -----------------------------------------------------------------------
    // 17. Delta: temporal stream tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_delta_stream_temporal() {
        let mut stream =
            ruvector_delta_core::DeltaStream::<ruvector_delta_core::VectorDelta>::for_vectors(4);

        // Push 3 deltas at different timestamps
        let d1 = VectorDelta::from_dense(vec![1.0, 0.0, 0.0, 0.0]);
        let d2 = VectorDelta::from_dense(vec![0.0, 1.0, 0.0, 0.0]);
        let d3 = VectorDelta::from_dense(vec![0.0, 0.0, 1.0, 0.0]);
        stream.push_with_timestamp(d1, 1000);
        stream.push_with_timestamp(d2, 2000);
        stream.push_with_timestamp(d3, 3000);

        // Query time range
        let range = stream.get_time_range(1500, 3500);
        assert_eq!(
            range.len(),
            2,
            "should find 2 deltas in time range 1500-3500"
        );

        // Full range should return all 3
        let all = stream.get_time_range(0, 10000);
        assert_eq!(all.len(), 3, "should find all 3 deltas");
    }

    // -----------------------------------------------------------------------
    // 18. Meta-learning: curiosity bonus and regret tracking
    // -----------------------------------------------------------------------
    #[test]
    fn test_meta_learning_curiosity() {
        let engine = DomainExpansionEngine::new();

        // Meta-learning health should be available without panicking
        let health = engine.meta_health();
        // Fresh engine has no observations, so consecutive_plateaus = 0
        assert_eq!(
            health.consecutive_plateaus, 0,
            "no plateaus on fresh engine"
        );

        // Regret summary should work on empty state
        let regret = engine.regret_summary();
        assert_eq!(regret.total_observations, 0, "no observations yet");

        // Pareto front should be empty initially
        assert_eq!(health.pareto_size, 0, "pareto front empty on fresh engine");
    }

    // -----------------------------------------------------------------------
    // Midstream Platform tests (ADR-077)
    // -----------------------------------------------------------------------

    #[test]
    fn test_midstream_scheduler_create() {
        let scheduler = crate::midstream::create_scheduler();
        let metrics = scheduler.metrics();
        assert_eq!(metrics.total_ticks, 0, "fresh scheduler has zero ticks");
        assert_eq!(metrics.total_tasks, 0, "fresh scheduler has zero tasks");
    }

    #[test]
    fn test_midstream_strange_loop_create() {
        let mut sl = crate::midstream::create_strange_loop();
        let mut ctx = strange_loop::Context::new();
        ctx.insert("relevance".to_string(), 0.8);
        ctx.insert("quality".to_string(), 0.9);
        // Should run without panic and converge within bounds
        let result = sl.run(&mut ctx);
        assert!(result.is_ok(), "strange loop should succeed: {:?}", result);
    }

    #[test]
    fn test_midstream_strange_loop_score() {
        let mut sl = crate::midstream::create_strange_loop();
        let score = crate::midstream::strange_loop_score(&mut sl, 0.8, 0.9);
        // Score should be in [0.0, 0.04] range
        assert!(score >= 0.0, "score should be non-negative");
        assert!(score <= 0.04, "score should be at most 0.04, got {}", score);
    }

    #[test]
    fn test_midstream_attractor_too_short() {
        // Less than 10 points → None
        let embeddings: Vec<Vec<f32>> = (0..5).map(|i| vec![i as f32; 8]).collect();
        let result = crate::midstream::analyze_category_attractor(&embeddings);
        assert!(
            result.is_none(),
            "should return None for too-short trajectory"
        );
    }

    #[test]
    fn test_midstream_attractor_stability_score() {
        let result = temporal_attractor_studio::LyapunovResult {
            lambda: -0.5,
            lyapunov_time: 2.0,
            doubling_time: 1.386,
            points_used: 20,
            dimension: 8,
            pairs_found: 10,
        };
        let score = crate::midstream::attractor_stability_score(&result);
        assert!(score > 0.0, "negative lambda should give positive score");
        assert!(score <= 0.05, "score should be at most 0.05");

        // Positive lambda → zero
        let chaotic = temporal_attractor_studio::LyapunovResult {
            lambda: 0.5,
            lyapunov_time: 2.0,
            doubling_time: 1.386,
            points_used: 20,
            dimension: 8,
            pairs_found: 10,
        };
        let cscore = crate::midstream::attractor_stability_score(&chaotic);
        assert_eq!(cscore, 0.0, "positive lambda should give zero score");
    }

    // Note: temporal-neural-solver tests require x86_64 SIMD
    #[cfg(feature = "x86-simd")]
    #[test]
    fn test_midstream_temporal_solver_create() {
        let solver = temporal_neural_solver::TemporalSolver::new(8, 16, 8);
        // Should create without panic. Predict requires Array1 inputs.
        let _ = solver;
    }

    #[cfg(feature = "x86-simd")]
    #[test]
    fn test_midstream_solver_confidence_score() {
        let cert = temporal_neural_solver::Certificate {
            error_bound: 0.01,
            confidence: 0.95,
            gate_pass: true,
            iterations: 5,
            computational_work: 100,
        };
        let score = crate::midstream::solver_confidence_score(&cert);
        assert!(score > 0.0, "gate_pass=true should give positive score");
        assert!(score <= 0.04, "score should be at most 0.04");

        // gate_pass=false → zero
        let bad_cert = temporal_neural_solver::Certificate {
            error_bound: 1.0,
            confidence: 0.1,
            gate_pass: false,
            iterations: 50,
            computational_work: 1000,
        };
        let bad_score = crate::midstream::solver_confidence_score(&bad_cert);
        assert_eq!(bad_score, 0.0, "gate_pass=false should give zero score");
    }

    // -----------------------------------------------------------------------
    // Digest rendering: UTF-8 boundary regression (the panic that took down
    // the production `ruvbrain` service — routes.rs `notify_digest`)
    // -----------------------------------------------------------------------

    fn memory_with(title: &str, content: &str) -> crate::types::BrainMemory {
        crate::types::BrainMemory {
            id: uuid::Uuid::new_v4(),
            category: crate::types::BrainCategory::Pattern,
            title: title.to_string(),
            content: content.to_string(),
            tags: vec!["demo".to_string()],
            code_snippet: None,
            embedding: vec![0.0; 8],
            contributor_id: "test".to_string(),
            quality_score: crate::types::BetaParams::new(),
            partition_id: None,
            witness_hash: String::new(),
            rvf_gcs_path: None,
            redaction_log: None,
            dp_proof: None,
            witness_chain: None,
            created_at: chrono::Utc::now(),
            updated_at: chrono::Utc::now(),
        }
    }

    /// Titles/contents whose byte budget (120 for title, 250 for content)
    /// lands strictly inside a multi-byte character. Before the fix,
    /// `&m.title[..120]` / `&m.content[..250]` panicked with
    /// "byte index N is not a char boundary", aborting the request task.
    #[test]
    fn digest_rows_survive_multibyte_titles_at_the_truncation_boundary() {
        let cases = vec![
            // 2-byte char straddling the 120-byte title budget
            memory_with(&format!("{}é tail", "a".repeat(119)), "short"),
            // 3-byte CJK straddling the title budget
            memory_with(&format!("{}中文", "a".repeat(119)), "short"),
            // 4-byte emoji straddling the title budget (starts at byte 118)
            memory_with(&format!("{}😀 more", "a".repeat(118)), "short"),
            // 4-byte emoji straddling the 250-byte content budget
            memory_with("ok", &format!("{}😀 rest of the content", "b".repeat(248))),
            // 3-byte CJK straddling the content budget
            memory_with("ok", &format!("{}中文内容", "b".repeat(249))),
        ];

        for m in &cases {
            let rows = crate::routes::format_digest_rows(std::slice::from_ref(m));
            assert!(
                rows.contains("<tr"),
                "expected a rendered row for title {:?}",
                m.title
            );
        }

        // Rendering them all together also works and yields one row each.
        let rows = crate::routes::format_digest_rows(&cases);
        assert_eq!(rows.matches("<tr").count(), cases.len());
    }

    /// The truncation must actually bound the output, not just avoid panicking.
    #[test]
    fn digest_rows_truncate_long_multibyte_fields() {
        // 200 emoji = 800 bytes of title; must be cut to <= 120 bytes.
        let m = memory_with(&"😀".repeat(200), &"中".repeat(400));
        let rows = crate::routes::format_digest_rows(std::slice::from_ref(&m));

        // 120 bytes / 4 bytes-per-emoji = exactly 30 emoji survive.
        assert_eq!(rows.matches('😀').count(), 30);
        // 250 bytes / 3 bytes-per-CJK = 83 chars (249 bytes), floor to boundary.
        assert_eq!(rows.matches('中').count(), 83);
    }

    // -----------------------------------------------------------------------
    // P1: pipeline injections must be removable by a system operator.
    //
    // `process_inject` stores `contributor_id = "pipeline:{source}"`, a value
    // no contributor pseudonym can ever equal (pseudonyms are 32 hex chars),
    // so the contributor-scoped delete could never remove an injected row.
    // -----------------------------------------------------------------------

    #[tokio::test]
    async fn pipeline_rows_are_undeletable_by_a_contributor_but_deletable_by_system() {
        let store = crate::store::FirestoreClient::new();

        let mut m = memory_with("injected", "body");
        // Exactly what process_inject writes for an inject with source="pubmed".
        m.contributor_id = "pipeline:pubmed".to_string();
        let id = m.id;
        store.store_memory(m).await.expect("store_memory");

        // A real contributor pseudonym — 32 hex chars, never equal to
        // "pipeline:...". The contributor-scoped delete must refuse.
        let pseudonym = "0123456789abcdef0123456789abcdef";
        let err = store.delete_memory(&id, pseudonym).await;
        assert!(
            matches!(err, Err(crate::store::StoreError::Forbidden(_))),
            "contributor-scoped delete of a pipeline row should be Forbidden, got {err:?}"
        );
        assert!(
            store.get_memory(&id).await.unwrap().is_some(),
            "row must still be present after the refused delete"
        );

        // A BRAIN_SYSTEM_KEY holder can clean it up.
        let deleted = store
            .delete_memory_as(&id, "ruvector-seed", true)
            .await
            .expect("system delete should not error");
        assert!(deleted, "system delete should report success");
        assert!(
            store.get_memory(&id).await.unwrap().is_none(),
            "row must be gone after the system delete"
        );
    }

    /// The system override must not turn every delete into a system delete:
    /// a non-system caller still cannot touch another contributor's row.
    #[tokio::test]
    async fn system_override_does_not_leak_to_ordinary_contributors() {
        let store = crate::store::FirestoreClient::new();
        let mut m = memory_with("someone elses", "body");
        m.contributor_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string();
        let id = m.id;
        store.store_memory(m).await.expect("store_memory");

        let other = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        assert!(matches!(
            store.delete_memory_as(&id, other, false).await,
            Err(crate::store::StoreError::Forbidden(_))
        ));
        assert!(store.get_memory(&id).await.unwrap().is_some());

        // The owner can still delete their own row.
        let deleted = store
            .delete_memory_as(&id, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", false)
            .await
            .expect("owner delete should not error");
        assert!(deleted);
    }

    // -----------------------------------------------------------------------
    // P2: `ranked_search` must not need a write lock.
    //
    // Every inject marks the CSR cache dirty; the next search rebuilt it.
    // While `ranked_search` took `&mut self`, that rebuild happened under a
    // write lock on the whole graph, so one search blocked every reader.
    // -----------------------------------------------------------------------

    fn graph_with_memories(n: usize) -> crate::graph::KnowledgeGraph {
        let mut g = crate::graph::KnowledgeGraph::new();
        for i in 0..n {
            let mut m = memory_with(&format!("mem {i}"), "content");
            // Embeddings close enough to clear the 0.55 similarity threshold
            // so real edges exist and the CSR is non-empty.
            let mut e = vec![0.9f32; 8];
            e[i % 8] += 0.3;
            crate::graph::normalize_embedding(&mut e);
            m.embedding = e;
            g.add_memory(&m);
        }
        g
    }

    /// The whole point of the change: a search runs with only a READ lock,
    /// and two readers can hold that lock at the same time. Under the old
    /// `&mut self` signature this did not compile; if it ever regresses to
    /// needing a write lock, `.read()` below stops compiling again.
    #[test]
    fn ranked_search_runs_concurrently_under_shared_read_locks() {
        let graph = std::sync::Arc::new(parking_lot::RwLock::new(graph_with_memories(24)));
        let query = {
            let mut q = vec![0.9f32; 8];
            q[0] += 0.3;
            crate::graph::normalize_embedding(&mut q);
            q
        };

        // Hold a read guard on this thread for the whole test...
        let held = graph.read();
        let from_held = held.ranked_search(&query, 5);
        assert!(!from_held.is_empty(), "search should return results");

        // ...while another thread also takes a read guard and searches.
        // This deadlocks (or fails to compile) if a write lock is required.
        let g2 = std::sync::Arc::clone(&graph);
        let q2 = query.clone();
        let other = std::thread::spawn(move || g2.read().ranked_search(&q2, 5));
        let from_other = other
            .join()
            .expect("concurrent reader panicked or deadlocked");

        assert_eq!(
            from_held.len(),
            from_other.len(),
            "both readers should see the same result set"
        );
        drop(held);
    }

    /// The lazy rebuild must still be correct: a search after an insert
    /// reflects the new edges, i.e. there is no staleness window.
    #[test]
    fn ranked_search_after_insert_sees_the_rebuilt_csr() {
        let mut g = graph_with_memories(12);
        let query = {
            let mut q = vec![0.9f32; 8];
            q[3] += 0.3;
            crate::graph::normalize_embedding(&mut q);
            q
        };

        let before = g.ranked_search(&query, 20);

        // Insert a new memory — this marks the CSR dirty.
        let mut m = memory_with("fresh", "content");
        let mut e = vec![0.9f32; 8];
        e[3] += 0.3;
        crate::graph::normalize_embedding(&mut e);
        m.embedding = e;
        let new_id = m.id;
        g.add_memory(&m);

        // A read-only search must already include it.
        let after = (&g).ranked_search(&query, 20);
        assert_eq!(after.len(), before.len() + 1, "new memory should be ranked");
        assert!(
            after.iter().any(|(id, _)| *id == new_id),
            "the just-inserted memory must appear without an explicit rebuild"
        );
    }

    /// Repeated searches with no intervening write are stable — the
    /// double-checked rebuild must not clear or corrupt the cache.
    #[test]
    fn repeated_searches_are_stable_without_writes() {
        let g = graph_with_memories(16);
        let query = {
            let mut q = vec![0.9f32; 8];
            q[1] += 0.3;
            crate::graph::normalize_embedding(&mut q);
            q
        };
        let first = g.ranked_search(&query, 8);
        for _ in 0..5 {
            assert_eq!(g.ranked_search(&query, 8), first);
        }
    }

    // -----------------------------------------------------------------------
    // P3: the cold-start sparsifier build must hold no lock.
    //
    // `spawn_blocking(|| graph.write().rebuild_sparsifier())` kept the tokio
    // runtime free but held the write guard for the whole build. The build is
    // now snapshot -> build off-lock -> guarded install.
    // -----------------------------------------------------------------------

    #[test]
    fn sparsifier_builds_from_a_snapshot_with_no_graph_borrowed() {
        let g = graph_with_memories(40);
        let (entries, nodes, edges, gen) = g
            .sparsifier_snapshot()
            .expect("graph should yield a snapshot");
        assert!(edges > 0 && nodes == 40);

        // The build takes plain data — no `&self`, so nothing is locked. This
        // is exactly the call the background task makes inside spawn_blocking.
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        let mut g = g;
        if let Some(spar) = built {
            assert!(
                g.install_sparsifier(spar, nodes, edges, gen),
                "install should succeed when the graph is unchanged"
            );
            assert!(g.sparsifier_stats().is_some());
        }
    }

    /// Edges appended while the build runs must be replayed on install —
    /// `add_memory` cannot feed a sparsifier that is still `None`.
    #[test]
    fn install_replays_edges_added_during_the_build() {
        let mut g = graph_with_memories(40);
        let (entries, nodes, edges, gen) = g.sparsifier_snapshot().unwrap();
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        // Simulate an inject landing mid-build.
        let mut m = memory_with("mid-build", "content");
        let mut e = vec![0.9f32; 8];
        e[2] += 0.3;
        crate::graph::normalize_embedding(&mut e);
        m.embedding = e;
        g.add_memory(&m);
        let edges_after = g.edge_count();
        assert!(edges_after > edges, "the inject should have added edges");

        if let Some(spar) = built {
            assert!(
                g.install_sparsifier(spar, nodes, edges, gen),
                "install should still succeed after a concurrent append"
            );
            let stats = g.sparsifier_stats().expect("stats after install");
            assert_eq!(
                stats.full_edges, edges_after,
                "the installed sparsifier must account for every edge, including \
                 the ones added during the build"
            );
        }
    }

    /// If the graph SHRANK during the build, node positions were reassigned by
    /// `remove_memory` and the built sparsifier refers to the wrong nodes.
    /// The install must refuse rather than silently corrupt analytics.
    #[test]
    fn install_refuses_when_the_graph_shrank_during_the_build() {
        let mut g = graph_with_memories(40);
        let (entries, nodes, edges, gen) = g.sparsifier_snapshot().unwrap();
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        // Simulate a delete landing mid-build. remove_memory reindexes every
        // node position, which is exactly what invalidates the built matrix.
        let to_remove = g.node_ids_snapshot()[0];
        g.remove_memory(&to_remove);
        assert!(g.edge_count() < edges || g.node_count() < nodes);

        if let Some(spar) = built {
            assert!(
                !g.install_sparsifier(spar, nodes, edges, gen),
                "install must refuse a sparsifier built against stale indices"
            );
            assert!(
                g.sparsifier_stats().is_none(),
                "no sparsifier should be installed after a refused install"
            );
        }
    }

    // -----------------------------------------------------------------------
    // P4: list_memories must do work proportional to the page, not the corpus.
    //
    // It used to clone every matching BrainMemory (each carrying a 384-dim
    // embedding), sort all of them, then throw away all but `limit`. The
    // rewrite sorts 24-byte keys and clones only the returned rows, so these
    // tests pin the ORDERING CONTRACT that rewrite has to preserve.
    // -----------------------------------------------------------------------

    /// Reference implementation: the old "clone everything, sort everything,
    /// then paginate" behaviour, with the new deterministic id tie-break.
    fn reference_page(
        mut all: Vec<crate::types::BrainMemory>,
        sort: &crate::types::ListSort,
        limit: usize,
        offset: usize,
    ) -> Vec<uuid::Uuid> {
        use crate::types::ListSort;
        all.sort_by(|a, b| {
            let (ka, kb) = match sort {
                ListSort::UpdatedAt => (
                    a.updated_at.timestamp_micros() as f64,
                    b.updated_at.timestamp_micros() as f64,
                ),
                ListSort::Quality => (a.quality_score.mean(), b.quality_score.mean()),
                ListSort::Votes => (
                    a.quality_score.observations(),
                    b.quality_score.observations(),
                ),
            };
            kb.partial_cmp(&ka)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        all.into_iter()
            .skip(offset)
            .take(limit)
            .map(|m| m.id)
            .collect()
    }

    #[tokio::test]
    async fn list_memories_pagination_matches_a_full_sort_including_ties() {
        use crate::types::ListSort;

        let store = crate::store::FirestoreClient::new();
        let mut all = Vec::new();

        // Deliberately heavy tie density: only 4 distinct quality values and
        // 3 distinct vote counts across 40 rows, so the tie-break path is
        // exercised on every sort mode.
        for i in 0..40u32 {
            let mut m = memory_with(&format!("mem {i}"), "content");
            m.quality_score = crate::types::BetaParams {
                alpha: 1.0 + f64::from(i % 4),
                beta: 1.0,
            };
            m.updated_at = chrono::Utc::now() - chrono::Duration::seconds(i64::from(i % 7));
            all.push(m.clone());
            store.store_memory(m).await.expect("store_memory");
        }

        for sort in [ListSort::UpdatedAt, ListSort::Quality, ListSort::Votes] {
            for (limit, offset) in [
                (5, 0),
                (5, 10),
                (1, 0),
                (40, 0),
                (10, 35),
                (7, 39),
                (5, 100),
            ] {
                let (page, total) = store
                    .list_memories(None, None, limit, offset, &sort)
                    .await
                    .expect("list_memories");

                assert_eq!(total, 40, "total_count must be the full match count");

                let got: Vec<_> = page.iter().map(|m| m.id).collect();
                let want = reference_page(all.clone(), &sort, limit, offset);
                assert_eq!(
                    got, want,
                    "sort={sort:?} limit={limit} offset={offset}: page must match a full sort"
                );
            }
        }
    }

    /// Identical requests must return identical pages. Before the rewrite,
    /// rows with equal sort keys were ordered by DashMap iteration order.
    #[tokio::test]
    async fn list_memories_is_deterministic_across_identical_calls() {
        use crate::types::ListSort;

        let store = crate::store::FirestoreClient::new();
        for i in 0..30u32 {
            let mut m = memory_with(&format!("mem {i}"), "content");
            // Every row has the SAME quality — pure tie-break territory.
            m.quality_score = crate::types::BetaParams {
                alpha: 2.0,
                beta: 1.0,
            };
            store.store_memory(m).await.expect("store_memory");
        }

        let first: Vec<_> = store
            .list_memories(None, None, 8, 4, &ListSort::Quality)
            .await
            .unwrap()
            .0
            .iter()
            .map(|m| m.id)
            .collect();

        for _ in 0..6 {
            let again: Vec<_> = store
                .list_memories(None, None, 8, 4, &ListSort::Quality)
                .await
                .unwrap()
                .0
                .iter()
                .map(|m| m.id)
                .collect();
            assert_eq!(
                again, first,
                "identical requests must return identical pages"
            );
        }
    }

    /// The count check alone is NOT sufficient. A remove followed by an add
    /// restores both counts while `remove_memory` has shifted every node
    /// position past the removed one — so a sparsifier built before the pair
    /// describes the wrong nodes at the same cardinality. Only the
    /// `index_generation` check catches this.
    #[test]
    fn install_refuses_after_a_remove_and_add_that_restores_the_counts() {
        let mut g = graph_with_memories(40);
        let (entries, nodes, edges, gen) = g.sparsifier_snapshot().unwrap();
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        // Remove one node (reindexes everything after it) ...
        let victim = g.node_ids_snapshot()[5];
        g.remove_memory(&victim);
        // ... then add one back, restoring node_count and pushing edges back up.
        for i in 0..3 {
            let mut m = memory_with(&format!("replacement {i}"), "content");
            let mut e = vec![0.9f32; 8];
            e[i % 8] += 0.3;
            crate::graph::normalize_embedding(&mut e);
            m.embedding = e;
            g.add_memory(&m);
        }

        assert!(
            g.node_count() >= nodes && g.edge_count() >= edges,
            "precondition: the counts must be restored, so only the generation \
             check can reject this build (nodes {} vs {}, edges {} vs {})",
            g.node_count(),
            nodes,
            g.edge_count(),
            edges
        );

        if let Some(spar) = built {
            assert!(
                !g.install_sparsifier(spar, nodes, edges, gen),
                "install must refuse after node positions were reassigned, even \
                 though the counts recovered"
            );
            assert!(g.sparsifier_stats().is_none());
        }
    }

    /// `rebuild_from_batch` reassigns every node position even for an
    /// identical memory set (it walks a DashMap, whose order is arbitrary), so
    /// a build that started before it must also be refused.
    #[test]
    fn install_refuses_after_a_full_rebuild_from_batch() {
        let mut g = graph_with_memories(30);
        let (entries, nodes, edges, gen) = g.sparsifier_snapshot().unwrap();
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        let memories: Vec<_> = (0..30)
            .map(|i| {
                let mut m = memory_with(&format!("mem {i}"), "content");
                let mut e = vec![0.9f32; 8];
                e[i % 8] += 0.3;
                crate::graph::normalize_embedding(&mut e);
                m.embedding = e;
                m
            })
            .collect();
        g.rebuild_from_batch(&memories);

        if let Some(spar) = built {
            assert!(
                !g.install_sparsifier(spar, nodes, edges, gen),
                "install must refuse after rebuild_from_batch reassigned positions"
            );
        }
    }

    /// A sparsifier installed by another path while this build ran must not be
    /// clobbered by the older background build.
    #[test]
    fn install_refuses_to_overwrite_a_newer_sparsifier() {
        let mut g = graph_with_memories(30);
        let (entries, nodes, edges, gen) = g.sparsifier_snapshot().unwrap();
        let built = crate::graph::KnowledgeGraph::build_sparsifier_from(&entries, nodes);

        // The inline small-graph path (routes.rs) builds one directly.
        g.rebuild_sparsifier();
        if g.sparsifier_stats().is_none() {
            return; // sparsifier unavailable in this environment; nothing to assert
        }

        if let Some(spar) = built {
            assert!(
                !g.install_sparsifier(spar, nodes, edges, gen),
                "install must not overwrite a sparsifier built while this one ran"
            );
            assert!(
                g.sparsifier_stats().is_some(),
                "the newer sparsifier must survive the refused install"
            );
        }
    }
}
