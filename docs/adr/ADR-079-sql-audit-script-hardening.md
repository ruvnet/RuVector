# ADR-079: SQL Audit Script Hardening & Bug Fixes

**Status:** Accepted
**Date:** 2026-03-03
**Author:** ruvnet

## Context

The ruvector independent audit verification script (`sql-audit.sql`) v2 contained 12 bugs ranging from syntax errors that prevent execution to logic errors that produce misleading results. The script tests 13 advertised ruvector extension features against actual behavior — correctness of the audit tool itself is critical for trust.

## Issues Found (v2 -> v3)

### Critical (5)

| # | Issue | Impact |
|---|-------|--------|
| 1 | **Dollar quoting broken** — All DO blocks use single `$` instead of `$$`. PostgreSQL requires `$$` or `$tag$` for dollar-quoted string literals. | Every DO block is a syntax error — script cannot run at all |
| 2 | **Hardcoded node IDs in shortest_path** (Section 4d) — Uses literal `1, 3` but auto-generated IDs vary by database state. IDs from Section 4b's DO block are local variables, unreachable in 4d. | Shortest path test fails on any non-empty database |
| 3 | **Section 5b bare SELECTs** — `ruvector_insert_triple()` calls have no DO/EXCEPTION wrapper. If the function doesn't exist, the script aborts entirely. | Breaks fault-tolerance guarantee |
| 4 | **dblink connection string unquoted** — `'dbname=' \|\| current_database()` is vulnerable to breakage with special characters in database/user names. | Persistence test fails on databases with spaces/special chars |
| 5 | **GUC `hnsw.ef_search` unguarded** — `SET hnsw.ef_search = 200` throws an error if ruvector doesn't register this custom GUC parameter. | HNSW test section aborts |

### Important (4)

| # | Issue | Impact |
|---|-------|--------|
| 6 | **Section 11 inconsistent filtering** — Uses `pg_namespace` join instead of `pg_depend + pg_extension`, unlike Section 0. May report non-ruvector functions as ruvector features. | False positives in bonus capabilities check |
| 7 | **Session GUCs not restored** — `hnsw.ef_search`, `client_min_messages` never reset. | Affects user's psql session after audit |
| 8 | **Section 5b results not validated** — Triple INSERT output printed but never checked PASS/FAIL. | Misleading — user sees output but no verdict |
| 9 | **Section 4c graph_stats outside exception block** — Bare SELECT aborts script if graph creation failed in 4b. | Breaks fault tolerance |

### Minor (3)

| # | Issue | Impact |
|---|-------|--------|
| 10 | `\timing` scope inconsistent across sections | Timing data missing for Sections 3-10 |
| 11 | Cypher test (4e) not programmatically validated | Relies on human to spot self-reference bug |
| 12 | `enable_indexscan = off` not wrapped in savepoint | Script interruption leaves index scans disabled |

## Decision

Create v3 (`scripts/sql-audit-v3.sql`) with all 12 fixes applied:

1. **Dollar quoting** — All DO blocks use `$$` or named tags (`$audit_NNN$`, `$graph_create$`, etc.)
2. **Node ID passing** — Temp table `_audit_graph_ids` bridges DO blocks; shortest_path reads from it
3. **Full fault tolerance** — Every external call wrapped in DO/EXCEPTION; no bare SELECTs for ruvector functions
4. **Safe dblink** — `format('dbname=%L user=%L', current_database(), current_user)` with proper quoting
5. **GUC guards** — `SET LOCAL hnsw.ef_search` inside nested BEGIN/EXCEPTION
6. **Consistent filtering** — All Section 11 queries use `pg_depend + pg_extension` join
7. **Session restore** — `RESET client_min_messages` at cleanup; `SET LOCAL` for all temporary GUCs
8. **Programmatic verdicts** — All sections emit PASS/FAIL/ERROR via RAISE NOTICE with value checks
9. **Savepoint safety** — `SET LOCAL enable_indexscan` scoped to DO block transaction

## Consequences

- Audit script is now fully executable on any PostgreSQL 14-17 installation
- No section can abort the rest — all wrapped in exception handlers
- Results are machine-parseable (grep for `PASS:` / `FAIL:` / `ERROR:`)
- Session state is clean after script completes

## Known ruvector Issues Discovered by Audit

| # | Issue | Status | Fix |
|---|-------|--------|-----|
| 1 | Cypher MATCH self-reference bug (`a.id == b.id`) | **Fixed (v0.3.1)** | Rewrote `match_pattern()` in `executor.rs` to properly traverse edges, reject self-references when variables differ, and generate per-edge binding rows |
| 2 | Graph/RDF persistence failure (in-memory only) | **Fixed (v0.3.1)** | Added PostgreSQL backing tables (`_ruvector_graphs`, `_ruvector_nodes`, `_ruvector_edges`, `_ruvector_rdf_stores`, `_ruvector_triples`) with auto-load on cache miss |
| 3 | HNSW index scan returns 0 rows despite correct query planning | Open | v0.1.0 SQL schema issue — requires investigation of index AM registration |
| 4 | Self-healing, multi-tenancy, hybrid search "not registered" | **Fixed (v0.3.1)** | 46 missing `CREATE FUNCTION` statements added to `ruvector--0.3.0.sql`: GNN (5), healing (17), tenancy (17), hybrid (7). Modules were always compiled but SQL schema lacked function registrations. All 46 verified in Docker container. |
| 5 | SONA apply panics on non-256-dim input | **Fixed (v0.3.1)** | Dynamic dimension detection with per-dim engine caching and `catch_unwind` panic guard |

## Related Changes (v0.3.1)

### Rust Source Fixes
- `crates/ruvector-postgres/src/graph/cypher/executor.rs` — Cypher self-reference fix
- `crates/ruvector-postgres/src/graph/mod.rs` — Graph persistence tables + `use pgrx::JsonB` + `get_by_name::<T, _>()` fix
- `crates/ruvector-postgres/src/graph/sparql/mod.rs` — RDF persistence tables + `get_by_name::<T, _>()` fix
- `crates/ruvector-postgres/src/graph/operators.rs` — Persist calls after node/edge/triple inserts
- `crates/ruvector-postgres/src/sona/mod.rs` — Dynamic dimension engine cache (`dim as usize` cast)
- `crates/ruvector-postgres/src/sona/operators.rs` — Dimension detection + `catch_unwind` panic guard

### SQL Schema
- `crates/ruvector-postgres/sql/ruvector--0.3.0.sql` — Added 46 `CREATE FUNCTION` statements for GNN (5), healing (17), tenancy (17), hybrid (7). Total extension functions: **190**

### Docker
- `crates/ruvector-postgres/Dockerfile` — Updated labels, features, SQL copy for v0.3.1
- `crates/ruvector-postgres/Dockerfile.prebuilt` — New slim image using pre-compiled artifacts (~12s build)
- `crates/ruvector-postgres/docker/Dockerfile` — Updated Rust 1.85, features, labels
- `crates/ruvector-postgres/docker/docker-compose.yml` — Updated Rust version to 1.85
- **Published**: `docker.io/ruvnet/ruvector-postgres:0.3.1` and `:latest` (sha256:6d2f28ed5efd, 151 MB)

### Verification Summary

All 46 new functions verified in Docker container (`ruvnet/ruvector-postgres:0.3.1`):

| Module | Functions | Status |
|--------|-----------|--------|
| GNN | `ruvector_gcn_forward`, `ruvector_gnn_aggregate`, `ruvector_message_pass`, `ruvector_graphsage_forward`, `ruvector_gnn_batch_forward` | 5/5 PASS |
| Self-Healing | `ruvector_health_status`, `ruvector_is_healthy`, `ruvector_system_metrics`, `ruvector_healing_history`, `ruvector_healing_history_since`, `ruvector_healing_history_for_strategy`, `ruvector_healing_trigger`, `ruvector_healing_execute`, `ruvector_healing_configure`, `ruvector_healing_get_config`, `ruvector_healing_enable`, `ruvector_healing_strategies`, `ruvector_healing_effectiveness`, `ruvector_healing_stats`, `ruvector_healing_thresholds`, `ruvector_healing_set_thresholds`, `ruvector_healing_problem_types` | 17/17 PASS |
| Multi-Tenancy | `ruvector_tenant_create`, `ruvector_tenant_set`, `ruvector_tenant_stats`, `ruvector_tenant_quota_check`, `ruvector_tenant_suspend`, `ruvector_tenant_resume`, `ruvector_tenant_delete`, `ruvector_tenants`, `ruvector_enable_tenant_rls`, `ruvector_tenant_migrate`, `ruvector_tenant_migration_status`, `ruvector_tenant_isolate`, `ruvector_tenant_set_policy`, `ruvector_tenant_update_quota`, `ruvector_generate_rls_sql`, `ruvector_generate_tenant_column_sql`, `ruvector_generate_roles_sql` | 17/17 PASS |
| Hybrid Search | `ruvector_register_hybrid`, `ruvector_hybrid_update_stats`, `ruvector_hybrid_configure`, `ruvector_hybrid_search`, `ruvector_hybrid_stats`, `ruvector_hybrid_score`, `ruvector_hybrid_list` | 7/7 PASS |
| SONA (prev fix) | `ruvector_sona_apply` with 3-dim and 5-dim inputs | 2/2 PASS |
