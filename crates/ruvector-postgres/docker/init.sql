-- RuVector-Postgres Initialization Script
-- Creates extension and verifies basic functionality

-- Create the extension
CREATE EXTENSION IF NOT EXISTS ruvector;

-- Create test schema
CREATE SCHEMA IF NOT EXISTS ruvector_test;

-- Test table for basic usage
CREATE TABLE ruvector_test.test_basic (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create ruvector role if it doesn't exist (optional app user)
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ruvector') THEN
        CREATE ROLE ruvector WITH LOGIN PASSWORD 'ruvector';
    END IF;
END $$;

-- Grant permissions to ruvector role and public
GRANT USAGE ON SCHEMA ruvector_test TO PUBLIC;
GRANT ALL ON SCHEMA ruvector_test TO ruvector;
GRANT ALL ON ALL TABLES IN SCHEMA ruvector_test TO ruvector;
GRANT ALL ON ALL SEQUENCES IN SCHEMA ruvector_test TO ruvector;

-- Log initialization and test basic functions
DO $$
DECLARE
    version_info TEXT;
    simd_info TEXT;
BEGIN
    -- Test version function
    SELECT ruvector_version() INTO version_info;
    RAISE NOTICE 'RuVector-Postgres initialized successfully';
    RAISE NOTICE 'Extension version: %', version_info;

    -- Test SIMD info function
    SELECT ruvector_simd_info() INTO simd_info;
    RAISE NOTICE 'SIMD info: %', simd_info;

    -- Test distance functions with array functions
    RAISE NOTICE 'Testing distance functions...';
    RAISE NOTICE 'Inner product: %', inner_product_arr(ARRAY[1.0, 2.0, 3.0]::real[], ARRAY[1.0, 2.0, 3.0]::real[]);
    RAISE NOTICE 'Cosine distance: %', cosine_distance_arr(ARRAY[1.0, 0.0, 0.0]::real[], ARRAY[0.0, 1.0, 0.0]::real[]);

    -- Test embedding functions
    RAISE NOTICE 'Testing embedding functions...';
    RAISE NOTICE 'Default model: %', ruvector_default_model();
    RAISE NOTICE 'MiniLM dims: %', ruvector_embedding_dims('all-MiniLM-L6-v2');

    RAISE NOTICE 'All basic tests passed!';
END $$;
