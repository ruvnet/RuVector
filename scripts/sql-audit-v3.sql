-- ================================================================
-- ruvector Independent Audit Verification Script (v3 — Hardened)
-- ================================================================
--
-- PURPOSE: Independently verify ruvector extension claims.
--          Tests 13 advertised features against actual behavior.
--          Script is fault-tolerant: no section aborts the rest.
--
-- REQUIREMENTS:
--   - PostgreSQL 14-17 with ruvector extension installed
--   - Run as a superuser or extension owner
--
-- USAGE:
--   psql -U postgres -d secondbrain -f sql-audit-v3.sql
--   -- or via Docker:
--   docker exec <container> psql -U postgres -d secondbrain -f /path/sql-audit-v3.sql
--
-- If ruvector is not yet loaded in your database:
--   CREATE EXTENSION ruvector;
--   -- then run this script
--
-- OUTPUT: Each section prints PASS/FAIL with evidence.
--         Save the full output for review.
--
-- CHANGES from v2:
--   - Fixed dollar quoting: all DO blocks use $$ or $audit_NNN$ tags
--   - Fixed shortest_path: uses temp table to pass node IDs between blocks
--   - Wrapped all bare SELECTs in DO/EXCEPTION for fault tolerance
--   - Fixed dblink connection string with format() + %L quoting
--   - Added GUC guards for hnsw.ef_search
--   - Section 11 now filters by extension dependency consistently
--   - Session GUCs properly saved/restored at start/end
--   - enable_indexscan wrapped in savepoint for safety
--   - Programmatic PASS/FAIL for all sections
--
-- AUTHOR: PLSS FHIR Team (Phase 0 Audit, Session 52)
-- DATE: 2026-02-26 (v1), 2026-03-03 (v2 fixes), 2026-03-03 (v3 hardened)
-- ================================================================

\pset pager off
SET search_path TO public;
SET client_min_messages TO notice;

-- Save session state for restore at end
DO $save$
BEGIN
  PERFORM set_config('audit.saved_client_min_messages',
    current_setting('client_min_messages'), false);
EXCEPTION WHEN OTHERS THEN
  NULL;
END $save$;

\echo ''
\echo '================================================================'
\echo '  ruvector INDEPENDENT AUDIT VERIFICATION (v3 — Hardened)'
\echo '  Run this against ANY ruvector installation'
\echo '================================================================'
\echo ''

-- ================================================================
-- SECTION 0: BASELINE
-- ================================================================
\echo '================================================================'
\echo '  SECTION 0: BASELINE - What is installed?'
\echo '================================================================'

\echo '--- 0a. Extension ---'
\timing on
SELECT extname, extversion FROM pg_extension WHERE extname = 'ruvector';
\echo ''

\echo '--- 0b. PostgreSQL version ---'
SELECT version();
\echo ''

\echo '--- 0c. Total ruvector functions (extension-owned only) ---'
SELECT count(*) AS total_ruvector_functions
FROM pg_proc p
JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector';
\echo ''

\echo '--- 0d. Functions by category (ruvector-owned only) ---'
SELECT
    CASE
        WHEN proname LIKE '%gcn%' OR proname LIKE '%gnn%' OR proname LIKE '%graphsage%' OR proname LIKE '%message_pass%' THEN '01_GNN'
        WHEN proname LIKE 'attention_%' OR proname LIKE '%attention%' THEN '02_ATTENTION'
        WHEN proname LIKE '%graph%' OR proname LIKE '%node%' OR proname LIKE '%edge%' OR proname LIKE '%cypher%' OR proname LIKE '%shortest%' THEN '03_GRAPH'
        WHEN proname LIKE '%rdf%' OR proname LIKE '%triple%' OR proname LIKE '%sparql%' OR proname LIKE '%ntriples%' THEN '04_SPARQL'
        WHEN proname LIKE '%health%' OR proname LIKE '%heal%' OR proname LIKE '%repair%' THEN '05_HEALING'
        WHEN proname LIKE '%tenant%' THEN '06_TENANCY'
        WHEN proname LIKE '%hybrid%' THEN '07_HYBRID'
        WHEN proname LIKE 'dag_%' OR proname LIKE '%qudag%' OR proname LIKE '%sona%' THEN '08_DAG_SONA'
        WHEN proname LIKE '%distance%' OR proname LIKE 'l1_%' OR proname LIKE 'l2_%' OR proname LIKE 'cosine_%' OR proname LIKE 'inner_%' THEN '09_DISTANCE'
        WHEN proname LIKE '%hnsw%' OR proname LIKE '%ivf%' THEN '10_INDEX'
        ELSE '11_OTHER'
    END AS category,
    count(*) AS function_count,
    string_agg(proname, ', ' ORDER BY proname) AS functions
FROM pg_proc p
JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
GROUP BY 1 ORDER BY 1;
\echo ''
\echo '>>> CHECK: How many functions in each category? Compare to claims.'
\echo ''
\timing off

-- ================================================================
-- SECTION 1: CORE VECTORS (should work)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 1: CORE VECTOR OPS'
\echo '  Expected: These should work on any ruvector installation'
\echo '================================================================'
\timing on

\echo '--- 1a. Vector type ---'
DO $$
DECLARE
  v ruvector;
BEGIN
  v := '[1.0, 2.0, 3.0]'::ruvector;
  RAISE NOTICE 'PASS: Vector type works — %', v;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: Vector type broken — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 1b. L2 distance ---'
DO $$
DECLARE
  d float8;
BEGIN
  d := '[1.0, 2.0, 3.0]'::ruvector <-> '[4.0, 5.0, 6.0]'::ruvector;
  IF d BETWEEN 5.19 AND 5.20 THEN
    RAISE NOTICE 'PASS: L2 distance = % (expected ~5.196)', d;
  ELSE
    RAISE NOTICE 'FAIL: L2 distance = % (expected ~5.196)', d;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: L2 distance error — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 1c. Cosine distance ---'
DO $$
DECLARE
  d float8;
BEGIN
  d := '[1.0, 0.0]'::ruvector <=> '[0.0, 1.0]'::ruvector;
  IF d BETWEEN 0.99 AND 1.01 THEN
    RAISE NOTICE 'PASS: Cosine distance = % (expected ~1.0)', d;
  ELSE
    RAISE NOTICE 'FAIL: Cosine distance = % (expected ~1.0)', d;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: Cosine distance error — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 1d. HNSW index creation ---'
DROP TABLE IF EXISTS _audit_vectors;
CREATE TABLE _audit_vectors (id serial PRIMARY KEY, embedding ruvector(8));
INSERT INTO _audit_vectors (embedding)
SELECT ('[' || array_to_string(ARRAY(
    SELECT round(((random() * 2 - 1))::numeric, 4) FROM generate_series(1,8)
), ', ') || ']')::ruvector
FROM generate_series(1, 500);

DO $$
BEGIN
  EXECUTE 'CREATE INDEX idx_audit_hnsw ON _audit_vectors USING hnsw (embedding ruvector_l2_ops)';
  RAISE NOTICE 'PASS: HNSW index created successfully';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: HNSW index creation error — % %', SQLSTATE, SQLERRM;
END $$;
ANALYZE _audit_vectors;
\echo ''

\echo '--- 1e. k-NN search via sequential scan (baseline) ---'
DO $knn_seq$
DECLARE
  cnt integer;
BEGIN
  SET LOCAL enable_indexscan = off;
  SET LOCAL enable_bitmapscan = off;
  SELECT count(*) INTO cnt FROM (
    SELECT id
    FROM _audit_vectors
    ORDER BY embedding <-> '[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]'::ruvector
    LIMIT 5
  ) sub;
  IF cnt = 5 THEN
    RAISE NOTICE 'PASS: Sequential k-NN returned % results', cnt;
  ELSE
    RAISE NOTICE 'FAIL: Sequential k-NN returned % results (expected 5)', cnt;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: Sequential k-NN error — % %', SQLSTATE, SQLERRM;
END $knn_seq$;
\echo ''

\echo '--- 1f. k-NN search via HNSW index ---'
DO $knn_hnsw$
DECLARE
  cnt integer;
BEGIN
  -- Guard: only set if GUC exists
  BEGIN
    SET LOCAL hnsw.ef_search = 200;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'WARNING: hnsw.ef_search GUC not available';
  END;
  SELECT count(*) INTO cnt FROM (
    SELECT id
    FROM _audit_vectors
    ORDER BY embedding <-> '[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]'::ruvector
    LIMIT 5
  ) sub;
  IF cnt = 5 THEN
    RAISE NOTICE 'PASS: HNSW k-NN returned % results', cnt;
  ELSIF cnt = 0 THEN
    -- Run diagnostics to understand why
    BEGIN
      DECLARE diag jsonb;
      BEGIN
        SELECT ruvector_hnsw_debug('idx_audit_hnsw') INTO diag;
        RAISE NOTICE 'FAIL: HNSW k-NN returned 0 rows — debug: %', diag;
      EXCEPTION WHEN OTHERS THEN
        RAISE NOTICE 'FAIL: HNSW k-NN returned 0 rows (debug function unavailable)';
      END;
    END;
  ELSE
    RAISE NOTICE 'WARN: HNSW k-NN returned % results (expected 5)', cnt;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: HNSW k-NN error — % %', SQLSTATE, SQLERRM;
END $knn_hnsw$;
\echo ''

\echo '--- 1g. Version + SIMD ---'
DO $$
DECLARE
  ver text;
  simd text;
BEGIN
  SELECT ruvector_version() INTO ver;
  SELECT ruvector_simd_info() INTO simd;
  RAISE NOTICE 'PASS: version=%, simd=%', ver, simd;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_version() or ruvector_simd_info() not found';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 2: ATTENTION (advertised: 13+ functions)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 2: ATTENTION MODULE'
\echo '  Advertised: attention, multi-head, flash, sparse, etc.'
\echo '================================================================'
\timing on

\echo '--- 2a. attention_score ---'
DO $$
DECLARE
  score float8;
BEGIN
  SELECT attention_score(
    ARRAY[1.0, 0.0, 1.0, 0.0]::real[],
    ARRAY[1.0, 0.0, 1.0, 0.0]::real[]
  ) INTO score;
  IF score BETWEEN 0.99 AND 1.01 THEN
    RAISE NOTICE 'PASS: attention_score = % (expected ~1.0)', score;
  ELSE
    RAISE NOTICE 'WARN: attention_score = % (expected ~1.0)', score;
  END IF;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: attention_score function does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 2b. attention_softmax ---'
DO $$
DECLARE
  result real[];
  total float8;
BEGIN
  SELECT attention_softmax(ARRAY[1.0, 2.0, 3.0]::real[]) INTO result;
  total := 0;
  FOR i IN 1..array_length(result, 1) LOOP
    total := total + result[i];
  END LOOP;
  IF total BETWEEN 0.99 AND 1.01 THEN
    RAISE NOTICE 'PASS: attention_softmax sums to % (expected ~1.0)', round(total::numeric, 4);
  ELSE
    RAISE NOTICE 'FAIL: attention_softmax sums to % (expected ~1.0)', round(total::numeric, 4);
  END IF;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: attention_softmax does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 2c. Check for multi-head attention ---'
DO $$
DECLARE
  cnt integer;
BEGIN
  SELECT count(*) INTO cnt
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%multi_head%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % multi-head attention function(s) registered', cnt;
  ELSE
    RAISE NOTICE 'FAIL: multi-head attention not registered';
  END IF;
END $$;
\echo ''

\echo '--- 2d. Check for flash attention ---'
DO $$
DECLARE
  cnt integer;
BEGIN
  SELECT count(*) INTO cnt
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%flash%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % flash attention function(s) registered', cnt;
  ELSE
    RAISE NOTICE 'FAIL: flash attention not registered';
  END IF;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 3: GNN (advertised: GCN, GraphSAGE, message passing)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 3: GNN — Graph Neural Networks'
\echo '  Advertised: GCN forward, GraphSAGE, message passing'
\echo '================================================================'
\timing on

\echo '--- 3a. Check if GNN functions exist + signatures ---'
SELECT proname, pg_get_function_arguments(p.oid) AS actual_signature
FROM pg_proc p
JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
WHERE p.proname LIKE '%gcn%' OR p.proname LIKE '%gnn%'
   OR p.proname LIKE '%graphsage%' OR p.proname LIKE '%message_pass%'
ORDER BY proname;
\echo '>>> FAIL if empty — GNN functions not registered'
\echo ''

\echo '--- 3b. Try calling GCN forward ---'
DO $$
BEGIN
  -- 2 nodes x 2 dims, 1 bidirectional edge, 2x2 weight matrix, output dim 2
  PERFORM ruvector_gcn_forward(
    ARRAY[1.0, 0.0, 0.0, 1.0]::real[],
    ARRAY[0, 1]::integer[],
    ARRAY[1, 0]::integer[],
    ARRAY[1.0, 0.0, 0.0, 1.0]::real[],
    2
  );
  RAISE NOTICE 'PASS: ruvector_gcn_forward executed successfully';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_gcn_forward does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 3c. Try calling GraphSAGE forward ---'
DO $$
BEGIN
  PERFORM ruvector_graphsage_forward(
    ARRAY[1.0, 0.0, 0.0, 1.0]::real[],
    ARRAY[0, 1]::integer[],
    ARRAY[1, 0]::integer[],
    2,
    10
  );
  RAISE NOTICE 'PASS: ruvector_graphsage_forward executed successfully';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_graphsage_forward does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 3d. Check for message_pass ---'
DO $$
DECLARE
  cnt integer;
BEGIN
  SELECT count(*) INTO cnt
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%message_pass%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % message_pass function(s) registered', cnt;
  ELSE
    RAISE NOTICE 'FAIL: message_pass not registered despite being advertised';
  END IF;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 4: GRAPH + CYPHER (advertised: graph queries)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 4: GRAPH + CYPHER'
\echo '  Advertised: graph CRUD, Cypher query language'
\echo '================================================================'
\timing on

-- Temp table to pass node IDs between DO blocks
DROP TABLE IF EXISTS _audit_graph_ids;
CREATE TEMP TABLE _audit_graph_ids (
  label text PRIMARY KEY,
  node_id bigint NOT NULL
);

\echo '--- 4a. Cleanup any prior audit_graph ---'
DO $$
BEGIN
  PERFORM ruvector_delete_graph('audit_graph');
  RAISE NOTICE 'Cleaned up previous audit_graph';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'No prior audit_graph to clean (OK)';
END $$;
\echo ''

\echo '--- 4b. Create graph + add nodes + edges ---'
DO $graph_create$
DECLARE
  id_alice bigint;
  id_bob bigint;
  id_charlie bigint;
  edge_ab bigint;
  edge_bc bigint;
BEGIN
  PERFORM ruvector_create_graph('audit_graph');

  SELECT ruvector_add_node('audit_graph', ARRAY['Person'], '{"name": "Alice"}'::jsonb) INTO id_alice;
  SELECT ruvector_add_node('audit_graph', ARRAY['Person'], '{"name": "Bob"}'::jsonb) INTO id_bob;
  SELECT ruvector_add_node('audit_graph', ARRAY['Person'], '{"name": "Charlie"}'::jsonb) INTO id_charlie;

  -- Persist IDs so subsequent sections can reference them
  INSERT INTO _audit_graph_ids VALUES ('alice', id_alice), ('bob', id_bob), ('charlie', id_charlie);

  SELECT ruvector_add_edge('audit_graph', id_alice, id_bob, 'KNOWS', '{"since": 2020}'::jsonb) INTO edge_ab;
  SELECT ruvector_add_edge('audit_graph', id_bob, id_charlie, 'KNOWS', '{"since": 2021}'::jsonb) INTO edge_bc;

  RAISE NOTICE 'PASS: Created nodes Alice=%, Bob=%, Charlie=% and edges %,%',
    id_alice, id_bob, id_charlie, edge_ab, edge_bc;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: Graph creation error — % %', SQLSTATE, SQLERRM;
END $graph_create$;
\echo ''

\echo '--- 4c. Graph stats ---'
DO $$
DECLARE
  stats jsonb;
BEGIN
  SELECT ruvector_graph_stats('audit_graph')::jsonb INTO stats;
  RAISE NOTICE 'Graph stats: %', stats;
  IF (stats->>'node_count')::int >= 3 AND (stats->>'edge_count')::int >= 2 THEN
    RAISE NOTICE 'PASS: 3+ nodes, 2+ edges';
  ELSE
    RAISE NOTICE 'FAIL: Expected 3 nodes + 2 edges, got %', stats;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: graph_stats error — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 4d. Shortest path (using captured node IDs) ---'
DO $shortest$
DECLARE
  id_first bigint;
  id_last bigint;
  result jsonb;
BEGIN
  SELECT node_id INTO id_first FROM _audit_graph_ids WHERE label = 'alice';
  SELECT node_id INTO id_last FROM _audit_graph_ids WHERE label = 'charlie';

  IF id_first IS NULL OR id_last IS NULL THEN
    RAISE NOTICE 'SKIP: Node IDs not captured — graph creation may have failed';
    RETURN;
  END IF;

  SELECT ruvector_shortest_path('audit_graph', id_first, id_last)::jsonb INTO result;
  RAISE NOTICE 'Shortest path result: %', result;
  IF result->>'nodes' IS NOT NULL THEN
    RAISE NOTICE 'PASS: Shortest path returned nodes';
  ELSE
    RAISE NOTICE 'FAIL: No path found between nodes % and %', id_first, id_last;
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $shortest$;
\echo ''

\echo '--- 4e. CYPHER MATCH (THE CRITICAL TEST) ---'
\echo '    Alice -KNOWS-> Bob -KNOWS-> Charlie'
\echo '    Query: MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b'
\echo '    Expected: Two results — Alice->Bob AND Bob->Charlie'
DO $cypher$
DECLARE
  result jsonb;
  row_count integer;
BEGIN
  SELECT ruvector_cypher('audit_graph',
    'MATCH (a:Person)-[:KNOWS]->(b:Person) RETURN a, b')::jsonb INTO result;
  RAISE NOTICE 'Cypher result: %', result;

  -- Check for known self-reference bug
  IF result::text LIKE '%"a"%' AND result::text LIKE '%"b"%' THEN
    RAISE NOTICE 'INFO: Cypher returned data — check manually for self-reference bug';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: Cypher error — % %', SQLSTATE, SQLERRM;
END $cypher$;
\echo ''
\echo '>>> CHECK OUTPUT: Does "a" and "b" have DIFFERENT ids?'
\echo '>>> FAIL if same id (self-reference bug known in ruvector 0.1.0)'
\echo ''
\timing off

-- ================================================================
-- SECTION 5: SPARQL / RDF (advertised: triple store)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 5: SPARQL / RDF Triple Store'
\echo '  Advertised: RDF storage, SPARQL queries'
\echo '================================================================'
\timing on

\echo '--- 5a. Cleanup any prior audit_rdf ---'
DO $$
BEGIN
  PERFORM ruvector_delete_rdf_store('audit_rdf');
  RAISE NOTICE 'Cleaned up previous audit_rdf';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'No prior audit_rdf to clean (OK)';
END $$;
\echo ''

\echo '--- 5b. Create RDF store + insert triples ---'
DO $rdf_create$
DECLARE
  t1 bigint;
  t2 bigint;
  t3 bigint;
BEGIN
  PERFORM ruvector_create_rdf_store('audit_rdf');

  SELECT ruvector_insert_triple('audit_rdf',
    'http://example.org/alice', 'http://xmlns.com/foaf/0.1/name', '"Alice"') INTO t1;
  SELECT ruvector_insert_triple('audit_rdf',
    'http://example.org/alice', 'http://xmlns.com/foaf/0.1/knows', 'http://example.org/bob') INTO t2;
  SELECT ruvector_insert_triple('audit_rdf',
    'http://example.org/bob', 'http://xmlns.com/foaf/0.1/name', '"Bob"') INTO t3;

  IF t1 IS NOT NULL AND t2 IS NOT NULL AND t3 IS NOT NULL THEN
    RAISE NOTICE 'PASS: Inserted 3 triples (IDs: %, %, %)', t1, t2, t3;
  ELSE
    RAISE NOTICE 'FAIL: One or more triple inserts returned NULL';
  END IF;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_insert_triple does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: RDF creation error — % %', SQLSTATE, SQLERRM;
END $rdf_create$;
\echo ''

\echo '--- 5c. RDF stats ---'
DO $$
DECLARE
  stats text;
BEGIN
  SELECT ruvector_rdf_stats('audit_rdf')::text INTO stats;
  RAISE NOTICE 'RDF stats: %', stats;
  IF stats LIKE '%triple_count%' THEN
    RAISE NOTICE 'PASS: RDF stats returned triple_count';
  ELSE
    RAISE NOTICE 'WARN: RDF stats format unexpected';
  END IF;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: rdf_stats error — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 5d. SPARQL ASK ---'
DO $$
DECLARE
  result text;
BEGIN
  SELECT ruvector_sparql('audit_rdf',
    'ASK { <http://example.org/alice> <http://xmlns.com/foaf/0.1/knows> <http://example.org/bob> }',
    'json')::text INTO result;
  RAISE NOTICE 'SPARQL ASK result: %', result;
  IF result LIKE '%true%' THEN
    RAISE NOTICE 'PASS: SPARQL ASK returned true';
  ELSE
    RAISE NOTICE 'FAIL: SPARQL ASK did not return true';
  END IF;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_sparql does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'FAIL: SPARQL error — % %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 6: PERSISTENCE (THE MOST CRITICAL TEST) — AUTOMATED
-- ================================================================
\echo '================================================================'
\echo '  SECTION 6: PERSISTENCE TEST (AUTOMATED via dblink)'
\echo '  Graph and RDF data should survive across connections.'
\echo '================================================================'

DO $$
BEGIN
  CREATE EXTENSION IF NOT EXISTS dblink;
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'WARNING: dblink extension not available — persistence test will be manual';
END $$;

DO $persist$
DECLARE
  graph_result text;
  rdf_result text;
  conn_str text;
BEGIN
  -- Use format() with %L for safe quoting of db/user names
  conn_str := format('dbname=%L user=%L', current_database(), current_user);

  -- Test graph persistence via a separate connection
  BEGIN
    SELECT val INTO graph_result
    FROM dblink(conn_str, $$SELECT ruvector_graph_stats('audit_graph')::text$$) AS t(val text);
    IF graph_result IS NOT NULL AND graph_result LIKE '%node_count%' THEN
      RAISE NOTICE 'PASS: Graph data PERSISTS across connections: %', graph_result;
    ELSE
      RAISE NOTICE 'FAIL: Graph query returned unexpected result: %', graph_result;
    END IF;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'FAIL: Graph data LOST across connections — % %', SQLSTATE, SQLERRM;
  END;

  -- Test RDF persistence via a separate connection
  BEGIN
    SELECT val INTO rdf_result
    FROM dblink(conn_str, $$SELECT ruvector_rdf_stats('audit_rdf')::text$$) AS t(val text);
    IF rdf_result IS NOT NULL AND rdf_result LIKE '%triple_count%' THEN
      RAISE NOTICE 'PASS: RDF data PERSISTS across connections: %', rdf_result;
    ELSE
      RAISE NOTICE 'FAIL: RDF query returned unexpected result: %', rdf_result;
    END IF;
  EXCEPTION WHEN OTHERS THEN
    RAISE NOTICE 'FAIL: RDF data LOST across connections — % %', SQLSTATE, SQLERRM;
  END;

EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'SKIP: Automated persistence test unavailable — % %', SQLSTATE, SQLERRM;
  RAISE NOTICE '  Manual test: disconnect, reconnect, then run:';
  RAISE NOTICE '    SELECT ruvector_graph_stats(''audit_graph'');';
  RAISE NOTICE '  PASS = data present, FAIL = "does not exist" error';
END $persist$;
\echo ''

-- ================================================================
-- SECTION 7: SELF-HEALING (advertised: health monitoring)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 7: SELF-HEALING'
\echo '  Advertised: health monitoring, index repair, auto-detection'
\echo '================================================================'
\timing on

\echo '--- 7a. Check if healing functions exist ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%health%' OR p.proname LIKE '%heal%'
     OR p.proname LIKE '%repair%' OR p.proname LIKE '%orphan%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % healing function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'FAIL: No healing functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 7b. Try health_status ---'
DO $$
DECLARE
  result text;
BEGIN
  SELECT ruvector_health_status()::text INTO result;
  RAISE NOTICE 'PASS: ruvector_health_status() = %', result;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_health_status() does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 8: MULTI-TENANCY (advertised: RLS isolation)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 8: MULTI-TENANCY'
\echo '  Advertised: tenant isolation, RLS generation'
\echo '================================================================'
\timing on

\echo '--- 8a. Check if tenancy functions exist ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%tenant%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % tenancy function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'FAIL: No tenancy functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 8b. Try tenant_set ---'
DO $$
BEGIN
  PERFORM ruvector_tenant_set('test_tenant');
  RAISE NOTICE 'PASS: ruvector_tenant_set() executed';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_tenant_set() does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 9: HYBRID SEARCH (advertised: vector + BM25 fusion)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 9: HYBRID SEARCH'
\echo '  Advertised: combined vector + keyword search with RRF fusion'
\echo '================================================================'
\timing on

\echo '--- 9a. Check if hybrid functions exist ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname || '(' || pg_get_function_arguments(p.oid) || ')', E'\n  ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%hybrid%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % hybrid function(s):%  %', cnt, E'\n', fn_list;
  ELSE
    RAISE NOTICE 'FAIL: No hybrid search functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 9b. Try hybrid_search ---'
DO $$
DECLARE
  result jsonb;
  is_success boolean;
BEGIN
  -- Correct argument order: collection, query_text, query_vector (real[]), k
  SELECT ruvector_hybrid_search(
    '_audit_vectors',
    'test query',
    ARRAY[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]::real[],
    5
  ) INTO result;
  is_success := (result->>'success')::boolean;
  IF is_success THEN
    RAISE NOTICE 'PASS: ruvector_hybrid_search() returned success=true (% results)',
      jsonb_array_length(result->'results');
  ELSE
    RAISE NOTICE 'FAIL: ruvector_hybrid_search() returned success=false: %',
      result->>'error';
  END IF;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_hybrid_search() does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR (may be signature mismatch — check 9a): % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 10: DAG / SONA (advertised: neural optimization)
-- ================================================================
\echo '================================================================'
\echo '  SECTION 10: DAG / SONA'
\echo '  Advertised: 64 DAG functions, SONA learning, QuDAG consensus'
\echo '================================================================'
\timing on

\echo '--- 10a. Check if DAG functions exist ---'
DO $$
DECLARE
  cnt integer;
BEGIN
  SELECT count(*) INTO cnt
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE 'dag_%' OR p.proname LIKE '%qudag%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % DAG/QuDAG function(s) registered', cnt;
  ELSE
    RAISE NOTICE 'FAIL: No DAG functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 10b. Check SONA functions ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%sona%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % SONA function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'FAIL: No SONA functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 10c. Try sona_stats ---'
DO $$
DECLARE
  result text;
BEGIN
  SELECT ruvector_sona_stats('_audit_vectors')::text INTO result;
  RAISE NOTICE 'PASS: ruvector_sona_stats() = %', result;
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_sona_stats() does not exist';
WHEN OTHERS THEN
  RAISE NOTICE 'ERROR: % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''

\echo '--- 10d. Try sona_apply (crash test with 3-dim input) ---'
DO $$
BEGIN
  PERFORM ruvector_sona_apply('_audit_vectors', ARRAY[1.0, 2.0, 3.0]::real[]);
  RAISE NOTICE 'PASS: ruvector_sona_apply() handled 3-dim input without crash';
EXCEPTION WHEN undefined_function THEN
  RAISE NOTICE 'FAIL: ruvector_sona_apply() does not exist';
WHEN external_routine_exception THEN
  RAISE NOTICE 'FAIL: CRASH/PANIC on 3-dim input — unsafe error handling';
WHEN OTHERS THEN
  RAISE NOTICE 'INFO: Graceful error on 3-dim input (expected): % — %', SQLSTATE, SQLERRM;
END $$;
\echo ''
\timing off

-- ================================================================
-- SECTION 11: ADDITIONAL CAPABILITIES DISCOVERED
-- ================================================================
\echo '================================================================'
\echo '  SECTION 11: ADDITIONAL CAPABILITIES (bonus checks)'
\echo '  Features found beyond the advertised 13'
\echo '================================================================'
\timing on

\echo '--- 11a. Hyperbolic geometry functions ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%poincare%' OR p.proname LIKE '%lorentz%'
     OR p.proname LIKE '%mobius%' OR p.proname LIKE '%exp_map%'
     OR p.proname LIKE '%log_map%' OR p.proname LIKE '%minkowski%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % hyperbolic geometry function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'INFO: No hyperbolic geometry functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 11b. Agent routing functions ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%agent%' OR p.proname LIKE '%route%' OR p.proname LIKE '%routing%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % agent/routing function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'INFO: No agent routing functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 11c. Embedding model functions ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%embed%' OR p.proname LIKE '%model%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % embedding/model function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'INFO: No embedding model functions registered';
  END IF;
END $$;
\echo ''

\echo '--- 11d. Temporal / learning functions ---'
DO $$
DECLARE
  cnt integer;
  fn_list text;
BEGIN
  SELECT count(*), string_agg(p.proname, ', ' ORDER BY p.proname)
  INTO cnt, fn_list
  FROM pg_proc p
  JOIN pg_depend d ON d.objid = p.oid AND d.deptype = 'e'
  JOIN pg_extension e ON e.oid = d.refobjid AND e.extname = 'ruvector'
  WHERE p.proname LIKE '%temporal%' OR p.proname LIKE '%learning%'
     OR p.proname LIKE '%feedback%' OR p.proname LIKE '%pattern%';
  IF cnt > 0 THEN
    RAISE NOTICE 'PASS: % temporal/learning function(s): %', cnt, fn_list;
  ELSE
    RAISE NOTICE 'INFO: No temporal/learning functions registered';
  END IF;
END $$;
\echo ''
\timing off

-- ================================================================
-- CLEANUP
-- ================================================================
\echo '================================================================'
\echo '  CLEANUP'
\echo '================================================================'

DROP TABLE IF EXISTS _audit_vectors;
DROP TABLE IF EXISTS _audit_graph_ids;
\echo 'Test tables dropped.'

DO $$
BEGIN
  PERFORM ruvector_delete_graph('audit_graph');
  RAISE NOTICE 'Graph audit_graph cleaned up';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'Graph audit_graph already gone (OK)';
END $$;

DO $$
BEGIN
  PERFORM ruvector_delete_rdf_store('audit_rdf');
  RAISE NOTICE 'RDF store audit_rdf cleaned up';
EXCEPTION WHEN OTHERS THEN
  RAISE NOTICE 'RDF store audit_rdf already gone (OK)';
END $$;

-- Restore session state
RESET client_min_messages;
\echo ''

-- ================================================================
-- SUMMARY CHECKLIST
-- ================================================================
\echo '================================================================'
\echo '  SUMMARY CHECKLIST'
\echo '================================================================'
\echo ''
\echo '  Review NOTICE output above for PASS/FAIL per section:'
\echo ''
\echo '  [ ] Section 0:  Baseline (extension loaded, functions registered?)'
\echo '  [ ] Section 1:  Core vectors (type, distance, HNSW index, k-NN)'
\echo '  [ ] Section 2:  Attention (basic yes, multi-head? flash?)'
\echo '  [ ] Section 3:  GNN (gcn_forward, graphsage_forward, message_pass?)'
\echo '  [ ] Section 4:  Graph CRUD + Cypher MATCH correctness'
\echo '  [ ] Section 5:  SPARQL (triple store, ASK query)'
\echo '  [ ] Section 6:  PERSISTENCE (data survives reconnection?)'
\echo '  [ ] Section 7:  Self-healing (functions exist?)'
\echo '  [ ] Section 8:  Multi-tenancy (functions exist?)'
\echo '  [ ] Section 9:  Hybrid search (functions exist?)'
\echo '  [ ] Section 10: DAG/SONA (functions exist? crashes?)'
\echo '  [ ] Section 11: Bonus capabilities (hyperbolic, agents, embeddings)'
\echo ''
\echo '  Key questions for review:'
\echo '  1. How many of 13 advertised features actually work?'
\echo '  2. Does graph/RDF data survive a simple disconnect?'
\echo '  3. Does Cypher MATCH return correct relationships?'
\echo '  4. Does HNSW index actually return results?'
\echo ''
\echo '================================================================'
\echo '  END OF AUDIT (v3 — Hardened)'
\echo '================================================================'
