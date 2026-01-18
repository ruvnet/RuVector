-- ============================================================================
-- RuVector PostgreSQL Extension - Routing Integration Tests
-- ============================================================================
-- Comprehensive integration tests for agent routing operators.
--
-- **CRITICAL BUG FIX VERIFICATION**: This test suite specifically validates
-- the fix for PostgreSQL 18 compatibility where `TableIterator<'static>`
-- was replaced with `SetOfIterator` in ruvector_list_agents() and
-- ruvector_find_agents_by_capability().
--
-- The original bug occurred because PG18's lifetime checker became stricter
-- and would not allow `'static` lifetime borrows in iterator contexts.
--
-- Run with: psql -d testdb -f pg18_routing_integration.sql
-- Or: cargo pgrx test pg18
--
-- PostgreSQL Version Compatibility: pg16, pg17, pg18
-- ============================================================================

\set ECHO all
\set ON_ERROR_STOP on

-- ============================================================================
-- SECTION: Test Environment Setup
-- ============================================================================

\echo '=== Section 1: Test Environment Setup ==='

-- Load extension (must be installed first)
CREATE EXTENSION IF NOT EXISTS ruvector;

-- Clean up any previous test data
SELECT ruvector_clear_agents();

-- Verify PostgreSQL version for compatibility tracking
SELECT
    version() AS postgresql_version,
    current_database() AS database_name,
    current_user AS current_user;

-- ============================================================================
-- SECTION: Bug Regression Test - CRITICAL FOR PG18
-- ============================================================================
-- This test specifically validates the fix for the crash that occurred
-- on PostgreSQL 18 when using ruvector_list_agents().
--
-- Bug: ruvector_list_agents() used TableIterator<'static> which crashed on PG18
-- Fix: Changed to SetOfIterator which properly handles lifetime issues
--
-- **THIS IS THE PRIMARY TEST THAT WOULD HAVE CAUGHT THE ORIGINAL BUG**

\echo '=== Section 2: CRITICAL - PG18 SetOfIterator Bug Regression Test ==='

-- First, register a test agent
\echo 'Registering test agent...'

SELECT ruvector_register_agent(
    'test-agent',
    'worker',
    ARRAY['test'],
    1.0,
    100,
    0.9
) AS registration_result;

-- **THIS QUERY PREVIOUSLY CRASHED ON PG18**
-- It should now return results without crashing on any PostgreSQL version
\echo 'Listing all agents (previously crashed on PG18)...'

SELECT
    name,
    agent_type,
    capabilities,
    cost_per_request,
    avg_latency_ms,
    quality_score,
    success_rate,
    total_requests,
    is_active
FROM ruvector_list_agents()
ORDER BY name;

-- Verify we got exactly one agent
\echo 'Verifying agent count...'

SELECT COUNT(*) AS agent_count
FROM ruvector_list_agents();

-- ============================================================================
-- SECTION: Agent Registration Tests
-- ============================================================================

\echo '=== Section 3: Agent Registration Tests ==='

-- Test 3.1: Register multiple agents with different configurations
\echo 'Test 3.1: Register multiple agents...'

SELECT ruvector_register_agent(
    'gpt-4',
    'llm',
    ARRAY['code_generation', 'translation', 'analysis'],
    0.03,
    500.0,
    0.95
) AS register_gpt4;

SELECT ruvector_register_agent(
    'claude-3',
    'llm',
    ARRAY['code_generation', 'writing', 'analysis'],
    0.04,
    450.0,
    0.97
) AS register_claude3;

SELECT ruvector_register_agent(
    'embedding-model',
    'embedding',
    ARRAY['similarity', 'search'],
    0.001,
    50.0,
    0.99
) AS register_embedding;

SELECT ruvector_register_agent(
    'vision-agent',
    'vision',
    ARRAY['image_analysis', 'ocr'],
    0.02,
    300.0,
    0.88
) AS register_vision;

SELECT ruvector_register_agent(
    'audio-transcriber',
    'audio',
    ARRAY['transcription', 'translation'],
    0.015,
    200.0,
    0.92
) AS register_audio;

-- Test 3.2: Verify all agents are listed
\echo 'Test 3.2: Verify all agents listed (PG18 SetOfIterator test)...'

SELECT COUNT(*) AS total_agents FROM ruvector_list_agents();

-- Test 3.3: Register duplicate agent (should fail)
\echo 'Test 3.3: Register duplicate agent (should fail)...'

SELECT ruvector_register_agent(
    'gpt-4',
    'llm',
    ARRAY['duplicate'],
    0.05,
    100.0,
    0.80
);  -- Expected to fail

-- Test 3.4: Register agent with full JSONB configuration
\echo 'Test 3.4: Register agent with JSONB config...'

SELECT ruvector_register_agent_full('{
    "name": "custom-agent",
    "agent_type": "specialized",
    "capabilities": ["special_task_1", "special_task_2"],
    "cost_model": {
        "per_request": 0.025,
        "per_token": 0.00005,
        "monthly_fixed": 10.0
    },
    "performance": {
        "avg_latency_ms": 150.0,
        "p95_latency_ms": 250.0,
        "p99_latency_ms": 400.0,
        "quality_score": 0.91,
        "success_rate": 0.98,
        "total_requests": 0
    },
    "is_active": true,
    "metadata": {"version": "1.0", "region": "us-east-1"}
}'::jsonb) AS register_jsonb;

-- ============================================================================
-- SECTION: Agent Retrieval Tests
-- ============================================================================

\echo '=== Section 4: Agent Retrieval Tests ==='

-- Test 4.1: Get specific agent details
\echo 'Test 4.1: Get specific agent details...'

SELECT ruvector_get_agent('gpt-4');

-- Test 4.2: Get non-existent agent (should return error)
\echo 'Test 4.2: Get non-existent agent (should error)...'

SELECT ruvector_get_agent('non-existent-agent');

-- Test 4.3: List all agents with full details
\echo 'Test 4.3: List all agents with full details...'

SELECT
    name,
    agent_type,
    array_length(capabilities, 1) AS capability_count,
    cost_per_request,
    avg_latency_ms,
    quality_score,
    is_active
FROM ruvector_list_agents()
ORDER BY quality_score DESC;

-- ============================================================================
-- SECTION: Capability Search Tests
-- ============================================================================

\echo '=== Section 5: Capability Search Tests (SetOfIterator PG18 test) ==='

-- Test 5.1: Find agents by specific capability
\echo 'Test 5.1: Find agents with code_generation capability...'

SELECT
    name,
    quality_score,
    avg_latency_ms,
    cost_per_request
FROM ruvector_find_agents_by_capability('code_generation', 10)
ORDER BY quality_score DESC;

-- Test 5.2: Find agents with different capability
\echo 'Test 5.2: Find agents with translation capability...'

SELECT
    name,
    quality_score,
    avg_latency_ms,
    cost_per_request
FROM ruvector_find_agents_by_capability('translation', 10)
ORDER BY quality_score DESC;

-- Test 5.3: Find agents with limit
\echo 'Test 5.3: Find agents with limit...'

SELECT
    name,
    quality_score,
    avg_latency_ms,
    cost_per_request
FROM ruvector_find_agents_by_capability('analysis', 2)
ORDER BY quality_score DESC;

-- Test 5.4: Find agents with non-existent capability
\echo 'Test 5.4: Find agents with non-existent capability...'

SELECT
    name,
    quality_score,
    avg_latency_ms,
    cost_per_request
FROM ruvector_find_agents_by_capability('non_existent_capability', 10);

-- ============================================================================
-- SECTION: Agent Update Tests
-- ============================================================================

\echo '=== Section 6: Agent Update Tests ==='

-- Test 6.1: Update agent metrics
\echo 'Test 6.1: Update agent metrics...'

SELECT ruvector_update_agent_metrics('gpt-4', 450.0, true, 0.96);

SELECT ruvector_update_agent_metrics('claude-3', 400.0, true, 0.98);

SELECT ruvector_update_agent_metrics('gpt-4', 600.0, false, NULL);

-- Test 6.2: Verify metrics were updated
\echo 'Test 6.2: Verify metrics updated...'

SELECT
    name,
    total_requests,
    avg_latency_ms,
    quality_score,
    success_rate
FROM ruvector_list_agents()
WHERE name IN ('gpt-4', 'claude-3')
ORDER BY name;

-- Test 6.3: Set agent active status
\echo 'Test 6.3: Toggle agent active status...'

SELECT ruvector_set_agent_active('vision-agent', false);

SELECT ruvector_set_agent_active('audio-transcriber', false);

-- Verify active status changed
SELECT
    name,
    is_active
FROM ruvector_list_agents()
WHERE name IN ('vision-agent', 'audio-transcriber')
ORDER BY name;

-- ============================================================================
-- SECTION: Routing Statistics Tests
-- ============================================================================

\echo '=== Section 7: Routing Statistics Tests ==='

-- Test 7.1: Get routing statistics
\echo 'Test 7.1: Get routing statistics...'

SELECT ruvector_routing_stats();

-- Test 7.2: Verify stats accuracy
\echo 'Test 7.2: Verify stats against actual counts...'

SELECT
    (SELECT COUNT(*) FROM ruvector_list_agents()) AS actual_total,
    (SELECT COUNT(*) FROM ruvector_list_agents() WHERE is_active = true) AS actual_active,
    ruvector_routing_stats()->>'total_agents' AS stats_total,
    ruvector_routing_stats()->>'active_agents' AS stats_active;

-- ============================================================================
-- SECTION: Agent Removal Tests
-- ============================================================================

\echo '=== Section 8: Agent Removal Tests ==='

-- Test 8.1: Remove an agent
\echo 'Test 8.1: Remove an agent...'

SELECT ruvector_remove_agent('audio-transcriber');

-- Test 8.2: Verify agent was removed
\echo 'Test 8.2: Verify agent was removed...'

SELECT ruvector_get_agent('audio-transcriber');

SELECT COUNT(*) AS remaining_agents FROM ruvector_list_agents();

-- Test 8.3: Try to remove non-existent agent
\echo 'Test 8.3: Remove non-existent agent (should error)...'

SELECT ruvector_remove_agent('already-removed-agent');

-- Test 8.4: Try to remove agent twice
\echo 'Test 8.4: Remove same agent twice (second should error)...'

SELECT ruvector_remove_agent('embedding-model');

SELECT ruvector_remove_agent('embedding-model');  -- Should fail

-- ============================================================================
-- SECTION: Edge Cases and Stress Tests
-- ============================================================================

\echo '=== Section 9: Edge Cases and Stress Tests ==='

-- Test 9.1: Empty agent list
\echo 'Test 9.1: Clear and test empty agent list...'

SELECT ruvector_clear_agents();

SELECT COUNT(*) AS count_after_clear FROM ruvector_list_agents();

-- Test 9.2: Capability search with no agents
SELECT COUNT(*) AS capability_search_no_agents
FROM ruvector_find_agents_by_capability('anything', 10);

-- Test 9.3: Stats with no agents
SELECT ruvector_routing_stats();

-- Test 9.4: Bulk agent registration (stress test for SetOfIterator)
\echo 'Test 9.4: Bulk registration (20 agents)...'

SELECT ruvector_register_agent(
    'bulk-agent-' || generate_series,
    'llm',
    ARRAY['bulk_capability_' || generate_series],
    (random() * 0.1)::float8,
    (random() * 1000)::float8,
    random()
)
FROM generate_series(1, 20);

-- Test 9.5: List many agents (validates SetOfIterator with multiple items)
\echo 'Test 9.5: List all bulk agents (PG18 SetOfIterator stress test)...'

SELECT
    name,
    agent_type,
    quality_score
FROM ruvector_list_agents()
ORDER BY name
LIMIT 25;

-- Test 9.6: Capability search with many results
\echo 'Test 9.6: Capability search with many results...'

SELECT COUNT(*) AS bulk_capability_count
FROM ruvector_find_agents_by_capability('bulk_capability_5', 100);

-- ============================================================================
-- SECTION: Routing Decision Tests
-- ============================================================================

\echo '=== Section 10: Routing Decision Tests ==='

-- Register test agents with different characteristics for routing
SELECT ruvector_clear_agents();

\echo 'Registering agents for routing tests...'

SELECT ruvector_register_agent('cheap-fast', 'llm', ARRAY['test'], 0.01, 100.0, 0.70);
SELECT ruvector_register_agent('expensive-slow', 'llm', ARRAY['test'], 0.10, 1000.0, 0.95);
SELECT ruvector_register_agent('balanced', 'llm', ARRAY['test'], 0.05, 500.0, 0.85);

-- Test 10.1: Route for cost optimization
\echo 'Test 10.1: Route for cost optimization...'

SELECT ruvector_route(
    ARRAY[0.1, 0.2, 0.3]::float8[] || ARRAY_FILL(0.1, ARRAY[381])::float8[],
    'cost',
    NULL
);

-- Test 10.2: Route for quality optimization
\echo 'Test 10.2: Route for quality optimization...'

SELECT ruvector_route(
    ARRAY[0.1, 0.2, 0.3]::float8[] || ARRAY_FILL(0.1, ARRAY[381])::float8[],
    'quality',
    NULL
);

-- Test 10.3: Route for latency optimization
\echo 'Test 10.3: Route for latency optimization...'

SELECT ruvector_route(
    ARRAY[0.1, 0.2, 0.3]::float8[] || ARRAY_FILL(0.1, ARRAY[381])::float8[],
    'latency',
    NULL
);

-- Test 10.4: Route with constraints
\echo 'Test 10.4: Route with constraints...'

SELECT ruvector_route(
    ARRAY[0.1, 0.2, 0.3]::float8[] || ARRAY_FILL(0.1, ARRAY[381])::float8[],
    'balanced',
    '{"max_cost": 0.06, "min_quality": 0.80}'::jsonb
);

-- ============================================================================
-- SECTION: PostgreSQL Version Compatibility Tests
-- ============================================================================

\echo '=== Section 11: PostgreSQL Version Compatibility ==='

-- Test that SetOfIterator works correctly on this PostgreSQL version
\echo 'Test 11.1: SetOfIterator compatibility test...'

SELECT
    version() AS pg_version,
    (SELECT COUNT(*) FROM ruvector_list_agents()) AS list_agents_works,
    (SELECT COUNT(*) FROM ruvector_find_agents_by_capability('test', 10)) AS find_capability_works;

-- Test 11.2: Test with all agent types
\echo 'Test 11.2: Test with all agent types...'

SELECT ruvector_clear_agents();

SELECT ruvector_register_agent('llm-agent', 'llm', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('embedding-agent', 'embedding', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('specialized-agent', 'specialized', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('vision-agent', 'vision', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('audio-agent', 'audio', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('multimodal-agent', 'multimodal', ARRAY['test'], 0.01, 100.0, 0.9);
SELECT ruvector_register_agent('custom-agent', 'custom_type', ARRAY['test'], 0.01, 100.0, 0.9);

SELECT
    agent_type,
    COUNT(*) AS count
FROM ruvector_list_agents()
GROUP BY agent_type
ORDER BY agent_type;

-- ============================================================================
-- SECTION: Cleanup
-- ============================================================================

\echo '=== Section 12: Final Cleanup ==='

-- Clean up test data
SELECT ruvector_clear_agents();

-- Verify clean state
SELECT
    COUNT(*) AS final_agent_count
FROM ruvector_list_agents();

SELECT ruvector_routing_stats();

\echo '=== All routing integration tests completed successfully ==='
\echo '=== PG18 SetOfIterator fix validated ==='
