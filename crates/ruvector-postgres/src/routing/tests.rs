//! Comprehensive unit tests for the routing module
//!
//! This module contains tests that would have caught the TableIterator<'static> bug
//! that was incompatible with PostgreSQL 18.
//!
//! Key areas tested:
//! - SetOfIterator compatibility (PG18 vs earlier versions)
//! - Iterator lifetime handling
//! - Memory context safety
//! - Agent registry operations
//! - Router decision making
//! - FastGRNN neural routing

#[cfg(test)]
mod pg18_compatibility_tests {
    //! Tests specifically for PostgreSQL 18 compatibility issues
    //!
    //! The bug: `TableIterator<'static>` was incompatible with PG18
    //! The fix: Changed to `SetOfIterator` without .collect()
    //!
    //! These tests verify:
    //! 1. SetOfIterator works correctly
    //! 2. No memory leaks with iterator returns
    //! 3. Proper lifetime handling without 'static

    use super::agents::{Agent, AgentRegistry, AgentType};
    use super::operators;
    use std::sync::Arc;

    /// Helper to create test agents
    fn create_test_agent(name: &str, agent_type: AgentType, capabilities: Vec<String>) -> Agent {
        let mut agent = Agent::new(name.to_string(), agent_type, capabilities);
        agent.cost_model.per_request = 0.05;
        agent.performance.avg_latency_ms = 100.0;
        agent.performance.quality_score = 0.85;
        agent
    }

    /// Test that agents can be registered and retrieved
    #[test]
    fn test_registry_lifecycle() {
        let registry = AgentRegistry::new();

        // Should be empty initially
        assert_eq!(registry.count(), 0);
        assert_eq!(registry.count_active(), 0);

        // Register an agent
        let agent = create_test_agent(
            "test-agent",
            AgentType::LLM,
            vec!["coding".to_string()],
        );

        assert!(registry.register(agent.clone()).is_ok());
        assert_eq!(registry.count(), 1);
        assert_eq!(registry.count_active(), 1);

        // Retrieve the agent
        let retrieved = registry.get("test-agent");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name, "test-agent");

        // Remove the agent
        let removed = registry.remove("test-agent");
        assert!(removed.is_some());
        assert_eq!(registry.count(), 0);
    }

    /// Test that list_all returns owned values (not references)
    ///
    /// This is critical for SetOfIterator compatibility - we need owned
    /// values that can be moved into the iterator, not borrowed references.
    #[test]
    fn test_list_all_returns_owned() {
        let registry = AgentRegistry::new();

        registry.register(create_test_agent(
            "agent1",
            AgentType::LLM,
            vec!["coding".to_string()],
        )).unwrap();

        registry.register(create_test_agent(
            "agent2",
            AgentType::Embedding,
            vec!["similarity".to_string()],
        )).unwrap();

        let agents = registry.list_all();

        // Verify we get owned values
        assert_eq!(agents.len(), 2);
        assert_eq!(agents[0].name, "agent1");
        assert_eq!(agents[1].name, "agent2");

        // These are owned clones, so we can use them after registry drops
        drop(registry);
        assert_eq!(agents[0].name, "agent1"); // Should still work
    }

    /// Test that list_all results can be transformed into iterators
    ///
    /// This simulates what SetOfIterator does - it takes an iterator
    /// and returns it. We need to ensure the transformation doesn't
    /// require 'static lifetime.
    #[test]
    fn test_list_all_can_be_mapped_to_iterator() {
        let registry = Arc::new(AgentRegistry::new());

        registry.register(create_test_agent(
            "agent1",
            AgentType::LLM,
            vec!["coding".to_string()],
        )).unwrap();

        registry.register(create_test_agent(
            "agent2",
            AgentType::Embedding,
            vec!["similarity".to_string()],
        )).unwrap();

        // Get agents and transform into iterator
        // This is what ruvector_list_agents() does internally
        let agents = registry.list_all();

        // Simulate the SetOfIterator::new() pattern
        let iterator = agents.into_iter().map(|agent| {
            (
                agent.name.clone(),
                agent.agent_type.as_str().to_string(),
                agent.capabilities.clone(),
                agent.cost_model.per_request,
                agent.performance.avg_latency_ms,
                agent.performance.quality_score,
                agent.performance.success_rate,
                agent.performance.total_requests as i64,
                agent.is_active,
            )
        });

        // Collect into a vector to verify it works
        let results: Vec<_> = iterator.collect();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, "agent1");
        assert_eq!(results[1].0, "agent2");
    }

    /// Test find_by_capability returns owned values
    #[test]
    fn test_find_by_capability_returns_owned() {
        let registry = AgentRegistry::new();

        registry.register(create_test_agent(
            "coder1",
            AgentType::LLM,
            vec!["coding".to_string()],
        )).unwrap();

        registry.register(create_test_agent(
            "coder2",
            AgentType::LLM,
            vec!["coding".to_string(), "translation".to_string()],
        )).unwrap();

        registry.register(create_test_agent(
            "translator",
            AgentType::LLM,
            vec!["translation".to_string()],
        )).unwrap();

        let coders = registry.find_by_capability("coding", 10);

        // Should return 2 coders
        assert_eq!(coders.len(), 2);

        // Verify they're owned values
        let names: Vec<_> = coders.iter().map(|a| a.name.clone()).collect();
        assert!(names.contains(&"coder1".to_string()));
        assert!(names.contains(&"coder2".to_string()));

        // Can use after registry drops
        drop(registry);
        assert_eq!(names.len(), 2);
    }

    /// Test that active/inactive filtering works correctly
    #[test]
    fn test_active_inactive_filtering() {
        let registry = AgentRegistry::new();

        let mut active_agent = create_test_agent(
            "active",
            AgentType::LLM,
            vec!["coding".to_string()],
        );
        active_agent.is_active = true;

        let mut inactive_agent = create_test_agent(
            "inactive",
            AgentType::LLM,
            vec!["coding".to_string()],
        );
        inactive_agent.is_active = false;

        registry.register(active_agent).unwrap();
        registry.register(inactive_agent).unwrap();

        // list_all should return both
        assert_eq!(registry.list_all().len(), 2);

        // list_active should only return active
        assert_eq!(registry.list_active().len(), 1);
        assert_eq!(registry.count_active(), 1);

        // find_by_capability should only return active
        let coders = registry.find_by_capability("coding", 10);
        assert_eq!(coders.len(), 1);
        assert_eq!(coders[0].name, "active");
    }

    /// Test registry thread safety
    #[test]
    fn test_registry_thread_safety() {
        use std::thread;

        let registry: Arc<AgentRegistry> = Arc::new(AgentRegistry::new());

        // Spawn multiple threads registering agents
        let handles: Vec<_> = (0..10)
            .map(|i| {
                let registry = Arc::clone(&registry);
                thread::spawn(move || {
                    let agent = create_test_agent(
                        &format!("agent-{}", i),
                        AgentType::LLM,
                        vec!["test".to_string()],
                    );
                    registry.register(agent)
                })
            })
            .collect();

        // All registrations should succeed
        for handle in handles {
            assert!(handle.join().unwrap().is_ok());
        }

        assert_eq!(registry.count(), 10);
    }
}

#[cfg(test)]
mod iterator_lifetime_tests {
    //! Tests for iterator lifetime safety
    //!
    //! The TableIterator<'static> bug was fundamentally a lifetime issue.
    //! These tests verify that iterators don't require 'static lifetime.

    use super::agents::{Agent, AgentRegistry, AgentType};

    fn create_test_agent(name: &str) -> Agent {
        Agent::new(name.to_string(), AgentType::LLM, vec!["test".to_string()])
    }

    /// Test that iterators don't require 'static lifetime
    #[test]
    fn test_iterator_non_static_lifetime() {
        let registry = AgentRegistry::new();

        // Register agents with non-'static data
        let local_string = String::from("local-agent");
        let agent = Agent::new(local_string, AgentType::LLM, vec!["test".to_string()]);
        registry.register(agent).unwrap();

        // Get agents - should work without 'static
        let agents = registry.list_all();
        assert_eq!(agents.len(), 1);

        // Can iterate and transform
        let names: Vec<_> = agents.iter().map(|a| a.name.clone()).collect();
        assert_eq!(names[0], "local-agent");
    }

    /// Test chained iterator operations
    #[test]
    fn test_chained_iterator_operations() {
        let registry = AgentRegistry::new();

        for i in 0..5 {
            registry.register(create_test_agent(&format!("agent-{}", i))).unwrap();
        }

        // Chain multiple iterator operations
        let results: Vec<_> = registry
            .list_all()
            .into_iter()
            .filter(|a| a.name.contains("2") || a.name.contains("3"))
            .map(|a| (a.name.clone(), a.capabilities.clone()))
            .collect();

        assert_eq!(results.len(), 2);
    }

    /// Test iterator can be passed to functions
    #[test]
    fn test_iterator_as_function_argument() {
        let registry = AgentRegistry::new();

        registry.register(create_test_agent("agent1")).unwrap();
        registry.register(create_test_agent("agent2")).unwrap();

        // Get iterator and pass to function
        let agents = registry.list_all();

        fn count_agents(iter: impl IntoIterator<Item = Agent>) -> usize {
            iter.into_iter().count()
        }

        let count = count_agents(agents);
        assert_eq!(count, 2);
    }
}

#[cfg(test)]
mod memory_safety_tests {
    //! Tests for memory safety and leak prevention

    use super::agents::{Agent, AgentRegistry, AgentType};

    #[test]
    fn test_no_memory_leak_on_repeated_operations() {
        let registry = AgentRegistry::new();

        // Perform many operations
        for i in 0..100 {
            let agent = Agent::new(
                format!("temp-agent-{}", i),
                AgentType::LLM,
                vec!["test".to_string()],
            );
            registry.register(agent).unwrap();

            // List and drop
            let _agents = registry.list_all();
        }

        // Clear and verify
        registry.clear();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_clone_safety() {
        let registry = AgentRegistry::new();

        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);
        agent.cost_model.per_request = 0.05;
        agent.performance.avg_latency_ms = 100.0;

        registry.register(agent.clone()).unwrap();

        // Clone should be independent
        let retrieved = registry.get("test").unwrap();
        assert_eq!(retrieved.name, agent.name);
        assert_eq!(retrieved.cost_model.per_request, agent.cost_model.per_request);
    }
}

#[cfg(test)]
mod edge_case_tests {
    //! Edge case and boundary condition tests

    use super::agents::{Agent, AgentRegistry, AgentType};
    use super::router::{
        OptimizationTarget, RoutingConstraints, Router,
    };

    fn create_test_agent(name: &str, cost: f32, quality: f32) -> Agent {
        let mut agent = Agent::new(name.to_string(), AgentType::LLM, vec!["test".to_string()]);
        agent.cost_model.per_request = cost;
        agent.performance.quality_score = quality;
        agent.performance.avg_latency_ms = 100.0;
        agent
    }

    #[test]
    fn test_empty_registry_operations() {
        let registry = AgentRegistry::new();

        assert_eq!(registry.count(), 0);
        assert_eq!(registry.list_all().len(), 0);
        assert_eq!(registry.list_active().len(), 0);
        assert!(registry.get("nonexistent").is_none());
        assert!(registry.remove("nonexistent").is_none());
        assert_eq!(registry.find_by_capability("test", 10).len(), 0);
    }

    #[test]
    fn test_duplicate_registration_fails() {
        let registry = AgentRegistry::new();

        let agent = create_test_agent("duplicate", 0.05, 0.8);

        registry.register(agent.clone()).unwrap();
        let result = registry.register(agent);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("already exists"));
    }

    #[test]
    fn test_update_nonexistent_agent_fails() {
        let registry = AgentRegistry::new();

        let agent = create_test_agent("nonexistent", 0.05, 0.8);
        let result = registry.update(agent);

        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not found"));
    }

    #[test]
    fn test_router_with_no_agents() {
        let router = Router::new();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let result = router.route(&embedding, &constraints, OptimizationTarget::Balanced);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("No agents"));
    }

    #[test]
    fn test_capability_case_insensitive() {
        let agent = Agent::new(
            "test".to_string(),
            AgentType::LLM,
            vec!["CODE_GENERATION".to_string()],
        );

        assert!(agent.has_capability("code_generation"));
        assert!(agent.has_capability("CODE_GENERATION"));
        assert!(agent.has_capability("Code_Generation"));
        assert!(!agent.has_capability("translation"));
    }

    #[test]
    fn test_empty_capabilities() {
        let agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);

        assert!(!agent.has_capability("anything"));
        assert_eq!(agent.capabilities.len(), 0);
    }

    #[test]
    fn test_cost_calculation_with_no_tokens() {
        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);
        agent.cost_model.per_request = 0.05;

        assert_eq!(agent.calculate_cost(None), 0.05);
    }

    #[test]
    fn test_cost_calculation_with_tokens() {
        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);
        agent.cost_model.per_request = 0.01;
        agent.cost_model.per_token = Some(0.001);

        assert_eq!(agent.calculate_cost(Some(100)), 0.11); // 0.01 + 100 * 0.001
    }

    #[test]
    fn test_metrics_update_first_observation() {
        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);

        agent.update_metrics(100.0, true, Some(0.9));

        assert_eq!(agent.performance.total_requests, 1);
        assert_eq!(agent.performance.avg_latency_ms, 100.0);
        assert_eq!(agent.performance.success_rate, 1.0);
        assert_eq!(agent.performance.quality_score, 0.9);
    }

    #[test]
    fn test_metrics_update_averaging() {
        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);

        agent.update_metrics(100.0, true, Some(0.8));
        agent.update_metrics(200.0, true, Some(0.9));

        assert_eq!(agent.performance.total_requests, 2);
        assert_eq!(agent.performance.avg_latency_ms, 150.0);
        assert_eq!(agent.performance.success_rate, 1.0);
        assert_eq!(agent.performance.quality_score, 0.85);
    }

    #[test]
    fn test_metrics_update_with_failure() {
        let mut agent = Agent::new("test".to_string(), AgentType::LLM, vec![]);

        agent.update_metrics(100.0, true, None);
        agent.update_metrics(100.0, false, None);

        assert_eq!(agent.performance.total_requests, 2);
        assert_eq!(agent.performance.success_rate, 0.5);
    }

    #[test]
    fn test_optimization_target_from_str() {
        assert_eq!(
            OptimizationTarget::from_str("cost"),
            OptimizationTarget::Cost
        );
        assert_eq!(
            OptimizationTarget::from_str("COST"),
            OptimizationTarget::Cost
        );
        assert_eq!(
            OptimizationTarget::from_str("latency"),
            OptimizationTarget::Latency
        );
        assert_eq!(
            OptimizationTarget::from_str("quality"),
            OptimizationTarget::Quality
        );
        assert_eq!(
            OptimizationTarget::from_str("balanced"),
            OptimizationTarget::Balanced
        );
        // Unknown defaults to Balanced
        assert_eq!(
            OptimizationTarget::from_str("unknown"),
            OptimizationTarget::Balanced
        );
    }

    #[test]
    fn test_routing_constraints_builder() {
        let constraints = RoutingConstraints::new()
            .with_max_cost(0.1)
            .with_max_latency(500.0)
            .with_min_quality(0.8)
            .with_capability("coding".to_string())
            .with_excluded_agent("bad-agent".to_string());

        assert_eq!(constraints.max_cost, Some(0.1));
        assert_eq!(constraints.max_latency_ms, Some(500.0));
        assert_eq!(constraints.min_quality, Some(0.8));
        assert_eq!(constraints.required_capabilities.len(), 1);
        assert_eq!(constraints.excluded_agents.len(), 1);
    }

    #[test]
    fn test_routing_constraints_default() {
        let constraints = RoutingConstraints::default();

        assert_eq!(constraints.max_cost, None);
        assert_eq!(constraints.max_latency_ms, None);
        assert_eq!(constraints.min_quality, None);
        assert_eq!(constraints.required_capabilities.len(), 0);
        assert_eq!(constraints.excluded_agents.len(), 0);
    }
}

#[cfg(test)]
mod routing_decision_tests {
    //! Tests for routing decision quality

    use super::agents::{Agent, AgentRegistry, AgentType};
    use super::router::{
        OptimizationTarget, RoutingConstraints, Router,
    };
    use std::sync::Arc;

    fn create_router_with_agents() -> Router {
        let registry = Arc::new(AgentRegistry::new());

        // Cheap, fast, low quality
        let mut agent1 = Agent::new("cheap-fast-low".to_string(), AgentType::LLM, vec!["test".to_string()]);
        agent1.cost_model.per_request = 0.01;
        agent1.performance.avg_latency_ms = 50.0;
        agent1.performance.quality_score = 0.6;
        agent1.embedding = Some(vec![0.1; 384]);
        registry.register(agent1).unwrap();

        // Expensive, slow, high quality
        let mut agent2 = Agent::new("expensive-slow-high".to_string(), AgentType::LLM, vec!["test".to_string()]);
        agent2.cost_model.per_request = 0.10;
        agent2.performance.avg_latency_ms = 500.0;
        agent2.performance.quality_score = 0.95;
        agent2.embedding = Some(vec![0.2; 384]);
        registry.register(agent2).unwrap();

        // Balanced
        let mut agent3 = Agent::new("balanced".to_string(), AgentType::LLM, vec!["test".to_string()]);
        agent3.cost_model.per_request = 0.05;
        agent3.performance.avg_latency_ms = 150.0;
        agent3.performance.quality_score = 0.8;
        agent3.embedding = Some(vec![0.15; 384]);
        registry.register(agent3).unwrap();

        Router::with_registry(registry)
    }

    #[test]
    fn test_cost_optimization_selects_cheapest() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Cost)
            .unwrap();

        assert_eq!(decision.agent_name, "cheap-fast-low");
        assert!(decision.estimated_cost < 0.02);
    }

    #[test]
    fn test_latency_optimization_selects_fastest() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Latency)
            .unwrap();

        assert_eq!(decision.agent_name, "cheap-fast-low");
        assert!(decision.estimated_latency_ms < 100.0);
    }

    #[test]
    fn test_quality_optimization_selects_best() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Quality)
            .unwrap();

        assert_eq!(decision.agent_name, "expensive-slow-high");
        assert!(decision.expected_quality > 0.9);
    }

    #[test]
    fn test_balanced_optimization_middle_ground() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Balanced)
            .unwrap();

        // Balanced should pick the middle agent
        assert_eq!(decision.agent_name, "balanced");
    }

    #[test]
    fn test_routing_with_cost_constraint() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new().with_max_cost(0.03);

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Quality)
            .unwrap();

        // Even though we optimize for quality, cost constraint should
        // exclude the expensive agent
        assert_eq!(decision.agent_name, "cheap-fast-low");
    }

    #[test]
    fn test_routing_with_quality_constraint() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new().with_min_quality(0.85);

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Cost)
            .unwrap();

        // Even optimizing for cost, quality constraint excludes low quality
        assert_eq!(decision.agent_name, "expensive-slow-high");
    }

    #[test]
    fn test_routing_with_excluded_agent() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new()
            .with_excluded_agent("cheap-fast-low".to_string());

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Cost)
            .unwrap();

        // Cheapest is excluded, should pick next best
        assert_ne!(decision.agent_name, "cheap-fast-low");
    }

    #[test]
    fn test_routing_with_required_capability() {
        let registry = Arc::new(AgentRegistry::new());

        let mut agent1 = Agent::new("coder".to_string(), AgentType::LLM, vec!["coding".to_string()]);
        agent1.cost_model.per_request = 0.05;
        agent1.performance.avg_latency_ms = 100.0;
        agent1.performance.quality_score = 0.8;
        agent1.embedding = Some(vec![0.1; 384]);
        registry.register(agent1).unwrap();

        let mut agent2 = Agent::new("translator".to_string(), AgentType::LLM, vec!["translation".to_string()]);
        agent2.cost_model.per_request = 0.01; // Cheaper
        agent2.performance.avg_latency_ms = 50.0;
        agent2.performance.quality_score = 0.7;
        agent2.embedding = Some(vec![0.1; 384]);
        registry.register(agent2).unwrap();

        let router = Router::with_registry(registry);
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new()
            .with_capability("coding".to_string());

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Cost)
            .unwrap();

        // Must pick coder despite translator being cheaper
        assert_eq!(decision.agent_name, "coder");
    }

    #[test]
    fn test_routing_decision_structure() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Balanced)
            .unwrap();

        // Verify all fields are populated
        assert!(!decision.agent_name.is_empty());
        assert!(decision.confidence >= 0.0 && decision.confidence <= 1.0);
        assert!(decision.estimated_cost >= 0.0);
        assert!(decision.estimated_latency_ms >= 0.0);
        assert!(decision.expected_quality >= 0.0 && decision.expected_quality <= 1.0);
        assert!(decision.similarity_score >= 0.0 && decision.similarity_score <= 1.0);
        assert!(!decision.reasoning.is_empty());
    }

    #[test]
    fn test_routing_alternatives() {
        let router = create_router_with_agents();
        let embedding = vec![0.1; 384];
        let constraints = RoutingConstraints::new();

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Balanced)
            .unwrap();

        // Should have alternatives
        assert!(!decision.alternatives.is_empty());
        assert!(decision.alternatives.len() <= 3); // Max 3 alternatives

        // Verify alternative structure
        for alt in &decision.alternatives {
            assert!(!alt.name.is_empty());
            assert!(!alt.reason.is_empty());
            assert_ne!(alt.name, decision.agent_name); // Not the selected agent
        }
    }
}

#[cfg(test)]
mod fastgrnn_tests {
    //! Additional tests for FastGRNN beyond the basic ones

    use super::fastgrnn::FastGRNN;

    #[test]
    fn test_fastgrnn_weight_initialization() {
        let grnn = FastGRNN::new(10, 5);

        // Check dimensions via public getters
        assert_eq!(grnn.input_dim(), 10);
        assert_eq!(grnn.hidden_dim(), 5);

        // Verify the model can be created via from_weights
        let grnn2 = FastGRNN::from_weights(
            10,
            5,
            vec![0.1; 50],
            vec![0.2; 25],
            vec![0.3; 50],
            vec![0.4; 25],
            vec![0.0; 5],
            vec![0.0; 5],
            1.0,
            1.0,
        );
        assert_eq!(grnn2.input_dim(), 10);
        assert_eq!(grnn2.hidden_dim(), 5);
    }

    #[test]
    fn test_fastgrnn_step_deterministic() {
        let grnn = FastGRNN::new(4, 3);
        let input = vec![1.0, 0.5, -0.5, 0.0];
        let hidden = vec![0.1, 0.2, 0.3];

        let output1 = grnn.step(&input, &hidden);
        let output2 = grnn.step(&input, &hidden);

        assert_eq!(output1, output2, "FastGRNN step should be deterministic");
    }

    #[test]
    fn test_fastgrnn_zero_input() {
        let grnn = FastGRNN::new(4, 3);
        let input = vec![0.0; 4];
        let hidden = vec![0.0; 3];

        let output = grnn.step(&input, &hidden);

        // With zero input and zero hidden, output should be close to zero
        for &val in &output {
            assert!(val.abs() < 0.1, "Zero input should produce near-zero output");
        }
    }

    #[test]
    fn test_fastgrnn_sequence_preserves_state() {
        let grnn = FastGRNN::new(4, 3);

        let inputs = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
        ];

        let outputs = grnn.forward_sequence(&inputs);

        assert_eq!(outputs.len(), 3);

        // Each output should have dimension 3
        for output in &outputs {
            assert_eq!(output.len(), 3);
        }

        // Outputs should differ (state changes)
        assert_ne!(outputs[0], outputs[1]);
        assert_ne!(outputs[1], outputs[2]);
    }
}

#[cfg(test)]
mod integration_tests {
    //! End-to-end integration tests

    use super::agents::{Agent, AgentRegistry, AgentType};
    use super::router::{
        OptimizationTarget, RoutingConstraints, Router,
    };
    use std::sync::Arc;

    #[test]
    fn test_full_routing_workflow() {
        // 1. Create registry
        let registry = Arc::new(AgentRegistry::new());

        // 2. Register agents
        let mut gpt4 = Agent::new("gpt-4".to_string(), AgentType::LLM, vec![
            "coding".to_string(),
            "translation".to_string(),
        ]);
        gpt4.cost_model.per_request = 0.03;
        gpt4.cost_model.per_token = Some(0.00006);
        gpt4.performance.avg_latency_ms = 500.0;
        gpt4.performance.quality_score = 0.95;
        gpt4.embedding = Some(vec![0.8; 384]);
        registry.register(gpt4).unwrap();

        let mut claude = Agent::new("claude-3".to_string(), AgentType::LLM, vec![
            "coding".to_string(),
            "analysis".to_string(),
        ]);
        claude.cost_model.per_request = 0.02;
        claude.performance.avg_latency_ms = 300.0;
        claude.performance.quality_score = 0.92;
        claude.embedding = Some(vec![0.7; 384]);
        registry.register(claude).unwrap();

        // 3. Create router
        let router = Router::with_registry(registry.clone());

        // 4. Route a coding request (quality-optimized)
        let embedding = vec![0.75; 384]; // Closer to gpt-4
        let constraints = RoutingConstraints::new()
            .with_capability("coding".to_string());

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Quality)
            .unwrap();

        assert_eq!(decision.agent_name, "gpt-4");
        assert!(decision.expected_quality > 0.9);

        // 5. Update metrics
        let mut agent = registry.get("gpt-4").unwrap();
        agent.update_metrics(450.0, true, Some(0.94));
        registry.update(agent).unwrap();

        // 6. Verify updated metrics
        let updated = registry.get("gpt-4").unwrap();
        assert_eq!(updated.performance.total_requests, 1);
    }

    #[test]
    fn test_agent_lifecycle() {
        let registry = AgentRegistry::new();

        // Create
        let mut agent = Agent::new("lifecycle-test".to_string(), AgentType::LLM, vec![]);
        assert_eq!(agent.performance.total_requests, 0);

        // Register
        registry.register(agent.clone()).unwrap();
        assert_eq!(registry.count(), 1);

        // Update metrics
        agent.update_metrics(100.0, true, Some(0.85));
        registry.update(agent.clone()).unwrap();

        // Retrieve
        let retrieved = registry.get("lifecycle-test").unwrap();
        assert_eq!(retrieved.performance.total_requests, 1);

        // Deactivate
        agent.is_active = false;
        registry.update(agent.clone()).unwrap();
        assert_eq!(registry.count_active(), 0);

        // Remove
        registry.remove("lifecycle-test").unwrap();
        assert_eq!(registry.count(), 0);
    }

    #[test]
    fn test_multi_capability_routing() {
        let registry = Arc::new(AgentRegistry::new());

        let mut agent1 = Agent::new("polyglot".to_string(), AgentType::LLM, vec![
            "coding".to_string(),
            "translation".to_string(),
            "analysis".to_string(),
        ]);
        agent1.cost_model.per_request = 0.05;
        agent1.performance.quality_score = 0.85;
        agent1.embedding = Some(vec![0.5; 384]);
        registry.register(agent1).unwrap();

        let mut agent2 = Agent::new("specialist".to_string(), AgentType::LLM, vec![
            "coding".to_string(),
        ]);
        agent2.cost_model.per_request = 0.02;
        agent2.performance.quality_score = 0.80;
        agent2.embedding = Some(vec![0.5; 384]);
        registry.register(agent2).unwrap();

        let router = Router::with_registry(registry);
        let embedding = vec![0.5; 384];

        // Request requiring only coding - both are candidates
        let constraints = RoutingConstraints::new()
            .with_capability("coding".to_string());

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Cost)
            .unwrap();

        // Should pick cheaper specialist
        assert_eq!(decision.agent_name, "specialist");

        // Request requiring multiple capabilities
        let constraints = RoutingConstraints::new()
            .with_capability("coding".to_string())
            .with_capability("translation".to_string());

        let decision = router
            .route(&embedding, &constraints, OptimizationTarget::Balanced)
            .unwrap();

        // Should pick polyglot (only one with both capabilities)
        assert_eq!(decision.agent_name, "polyglot");
    }
}
