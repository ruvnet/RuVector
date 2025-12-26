// Integration tests for Tiny Dancer Routing module
//
// These tests validate the complete routing functionality including
// agent registration, FastGRNN neural network, and routing decisions.

#[cfg(test)]
mod routing_tests {
    use ruvector_postgres::routing::{
        agents::{Agent, AgentRegistry, AgentType},
        fastgrnn::FastGRNN,
        router::{OptimizationTarget, Router, RoutingConstraints},
    };

    #[test]
    fn test_complete_routing_workflow() {
        // Create registry and router
        let registry = AgentRegistry::new();
        let router = Router::with_registry(std::sync::Arc::new(registry));

        // Register diverse agents
        let agents = vec![
            create_agent("gpt-4", 0.03, 500.0, 0.95, vec!["coding", "reasoning"]),
            create_agent("claude-3", 0.025, 400.0, 0.93, vec!["coding", "writing"]),
            create_agent("gpt-3.5", 0.002, 200.0, 0.75, vec!["general", "fast"]),
            create_agent("llama-2", 0.0, 800.0, 0.70, vec!["local", "private"]),
        ];

        for agent in agents {
            router.registry().register(agent).unwrap();
        }

        // Test cost-optimized routing
        let request_emb = vec![0.1; 384];
        let decision = router
            .route(
                &request_emb,
                &RoutingConstraints::new(),
                OptimizationTarget::Cost,
            )
            .unwrap();

        assert_eq!(decision.agent_name, "llama-2"); // Free option
        assert!(decision.confidence > 0.0);

        // Test quality-optimized routing
        let decision = router
            .route(
                &request_emb,
                &RoutingConstraints::new(),
                OptimizationTarget::Quality,
            )
            .unwrap();

        assert_eq!(decision.agent_name, "gpt-4"); // Highest quality

        // Test latency-optimized routing
        let decision = router
            .route(
                &request_emb,
                &RoutingConstraints::new(),
                OptimizationTarget::Latency,
            )
            .unwrap();

        assert_eq!(decision.agent_name, "gpt-3.5"); // Fastest
    }

    #[test]
    fn test_routing_with_constraints() {
        let registry = AgentRegistry::new();
        let router = Router::with_registry(std::sync::Arc::new(registry));

        router
            .registry()
            .register(create_agent(
                "expensive-high-quality",
                1.0,
                200.0,
                0.99,
                vec!["coding"],
            ))
            .unwrap();

        router
            .registry()
            .register(create_agent(
                "cheap-medium-quality",
                0.01,
                200.0,
                0.75,
                vec!["coding"],
            ))
            .unwrap();

        let request_emb = vec![0.1; 384];

        // Constrain by max cost
        let constraints = RoutingConstraints::new()
            .with_max_cost(0.5)
            .with_min_quality(0.7);

        let decision = router
            .route(&request_emb, &constraints, OptimizationTarget::Quality)
            .unwrap();

        // Should pick cheap option due to cost constraint
        assert_eq!(decision.agent_name, "cheap-medium-quality");
    }

    #[test]
    fn test_fastgrnn_routing() {
        let mut router = Router::new();
        router.init_grnn(64);

        router
            .registry()
            .register(create_agent("agent1", 0.05, 200.0, 0.85, vec!["coding"]))
            .unwrap();

        let request_emb = vec![0.1; 384];

        let decision = router
            .route(
                &request_emb,
                &RoutingConstraints::new(),
                OptimizationTarget::Balanced,
            )
            .unwrap();

        // Verify neural network enhanced confidence
        assert!(decision.confidence > 0.0);
        assert!(decision.confidence <= 1.0);
    }

    #[test]
    fn test_capability_based_routing() {
        let registry = AgentRegistry::new();
        let router = Router::with_registry(std::sync::Arc::new(registry));

        router
            .registry()
            .register(create_agent(
                "coder",
                0.05,
                200.0,
                0.90,
                vec!["coding", "debugging"],
            ))
            .unwrap();

        router
            .registry()
            .register(create_agent(
                "writer",
                0.03,
                150.0,
                0.85,
                vec!["writing", "translation"],
            ))
            .unwrap();

        router
            .registry()
            .register(create_agent(
                "generalist",
                0.02,
                300.0,
                0.70,
                vec!["coding", "writing", "general"],
            ))
            .unwrap();

        let request_emb = vec![0.1; 384];

        // Require coding capability
        let constraints = RoutingConstraints::new().with_capability("coding".to_string());

        let decision = router
            .route(&request_emb, &constraints, OptimizationTarget::Quality)
            .unwrap();

        // Should pick specialized coder (highest quality with coding)
        assert!(decision.agent_name == "coder" || decision.agent_name == "generalist");

        // Verify writer was not selected
        assert_ne!(decision.agent_name, "writer");
    }

    #[test]
    fn test_agent_metrics_update() {
        let registry = AgentRegistry::new();
        let mut agent = create_agent("test-agent", 0.05, 200.0, 0.80, vec!["test"]);

        // Initial state
        assert_eq!(agent.performance.total_requests, 0);
        assert_eq!(agent.performance.avg_latency_ms, 200.0);

        // Update with better latency
        agent.update_metrics(150.0, true, Some(0.85));
        assert_eq!(agent.performance.total_requests, 1);
        assert_eq!(agent.performance.avg_latency_ms, 150.0);
        assert_eq!(agent.performance.success_rate, 1.0);

        // Update with worse latency
        agent.update_metrics(250.0, true, Some(0.75));
        assert_eq!(agent.performance.total_requests, 2);
        assert_eq!(agent.performance.avg_latency_ms, 200.0); // Average of 150 and 250
        assert_eq!(agent.performance.success_rate, 1.0);

        // Failed request
        agent.update_metrics(300.0, false, None);
        assert_eq!(agent.performance.total_requests, 3);
        assert!(agent.performance.success_rate < 1.0);
    }

    #[test]
    fn test_fastgrnn_sequence_processing() {
        let grnn = FastGRNN::new(10, 5);

        let sequence = vec![
            vec![1.0, 0.0, 0.0, 0.5, -0.5, 0.2, -0.2, 0.8, -0.8, 0.0],
            vec![0.0, 1.0, 0.0, -0.5, 0.5, -0.2, 0.2, -0.8, 0.8, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ];

        let outputs = grnn.forward_sequence(&sequence);

        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].len(), 5);

        // Verify state evolution (later states should be different from first)
        let first_state = &outputs[0];
        let last_state = &outputs[2];

        let diff: f32 = first_state
            .iter()
            .zip(last_state.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();

        assert!(diff > 0.0, "Hidden state should evolve across sequence");
    }

    #[test]
    fn test_routing_alternatives() {
        let registry = AgentRegistry::new();
        let router = Router::with_registry(std::sync::Arc::new(registry));

        // Register multiple similar agents
        for i in 0..5 {
            let quality = 0.7 + (i as f32 * 0.05);
            let cost = 0.01 + (i as f32 * 0.01);
            router
                .registry()
                .register(create_agent(
                    &format!("agent-{}", i),
                    cost,
                    200.0,
                    quality,
                    vec!["test"],
                ))
                .unwrap();
        }

        let request_emb = vec![0.1; 384];

        let decision = router
            .route(
                &request_emb,
                &RoutingConstraints::new(),
                OptimizationTarget::Quality,
            )
            .unwrap();

        // Should have alternatives listed
        assert!(!decision.alternatives.is_empty());
        assert!(decision.alternatives.len() <= 3); // Max 3 alternatives

        // Alternatives should have lower scores
        for alt in &decision.alternatives {
            assert!(alt.score < 1.0);
            assert!(!alt.reason.is_empty());
        }
    }

    #[test]
    fn test_excluded_agents() {
        let registry = AgentRegistry::new();
        let router = Router::with_registry(std::sync::Arc::new(registry));

        router
            .registry()
            .register(create_agent("agent-a", 0.05, 200.0, 0.90, vec!["test"]))
            .unwrap();

        router
            .registry()
            .register(create_agent("agent-b", 0.05, 200.0, 0.85, vec!["test"]))
            .unwrap();

        let request_emb = vec![0.1; 384];

        // Exclude the best agent
        let constraints = RoutingConstraints::new().with_excluded_agent("agent-a".to_string());

        let decision = router
            .route(&request_emb, &constraints, OptimizationTarget::Quality)
            .unwrap();

        assert_eq!(decision.agent_name, "agent-b");
    }

    // Helper function to create test agents
    fn create_agent(
        name: &str,
        cost: f32,
        latency: f32,
        quality: f32,
        capabilities: Vec<&str>,
    ) -> Agent {
        let mut agent = Agent::new(
            name.to_string(),
            AgentType::LLM,
            capabilities.iter().map(|s| s.to_string()).collect(),
        );
        agent.cost_model.per_request = cost;
        agent.performance.avg_latency_ms = latency;
        agent.performance.quality_score = quality;
        agent.embedding = Some(vec![0.1; 384]); // Default embedding
        agent
    }
}
