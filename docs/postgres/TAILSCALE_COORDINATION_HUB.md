# PostgreSQL Coordination Hub over Tailscale

**Date:** 2026-01-14
**Status:** Design Document
**Purpose:** PostgreSQL 18 as coordination hub for distributed infrastructure

---

## Architecture Overview

```
                    TAILNET (ACL-Controlled)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
    Cloud Infra      Mac (this)        gmktec-k9
   tag:prod        PostgreSQL 18       tag:gmktec
   tag:containers  Coordination Hub    Heavy Processing
                      5TB Storage
                   port 28818 (pg)
                   port 6432 (pgbouncer)
```

**Role:** Coordination hub - NOT the primary data store. Vector metadata, queue state, agent coordination.

---

## Security Model

### Tailscale ACL Integration

**Existing relevant tags:**
```json
"tag:signing-workstation"  // Mac hosting PostgreSQL
"tag:gmktec"               // Remote worker server
"tag:prod"                 // Cloud infrastructure
"tag:containers"           // Container workloads
"tag:devops-admin"         // Admin workstations
```

**Proposed addition:**
```json
"tag:coordination-hub": ["autogroup:admin"],
```

**ACL grant for PostgreSQL access:**
```json
{
    "src": ["tag:gmktec", "tag:prod", "tag:containers", "tag:devops-admin"],
    "dst": ["tag:coordination-hub"],
    "ip": ["28818", "6432"],  // PostgreSQL + PgBouncer
}
```

### Defense in Depth

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| 1 | Tailscale ACLs | IP-level access control |
| 2 | `pg_hba.conf` | PostgreSQL auth |
| 3 | Database roles | Least privilege per client |
| 4 | Row-level security | Tenant isolation (if needed) |

---

## Implementation Plan

### Phase 1: Connection Pooling (15 min)

**Install PgBouncer:**
```bash
brew install pgbouncer
```

**Configuration: `/opt/homebrew/etc/pgbouncer.ini`**
```ini
[databases]
ruvector = host=localhost port=28818 dbname=ruvector

[pgbouncer]
listen_addr = 100.x.y.z  # Tailscale IP of Mac
listen_port = 6432
auth_type = md5
auth_file = /opt/homebrew/etc/userlist.txt
pool_mode = transaction
max_client_conn = 1000
default_pool_size = 50
reserve_pool_size = 10
reserve_pool_timeout = 3
server_lifetime = 3600
server_idle_timeout = 600
```

**Userlist: `/opt/homebrew/etc/userlist.txt`**
```
"gmktec" "md5<hash>"
"cloud" "md5<hash>"
"admin" "md5<hash>"
```

**Launch:**
```bash
brew services start pgbouncer
```

---

### Phase 2: PostgreSQL Security (10 min)

**`pg_hba.conf` - Tailscale-aware rules:**
```conf
# TYPE  DATABASE        USER            ADDRESS                 METHOD

# Local only
local   all             postgres                                trust
local   all             all                                     md5

# Tailscale network only (100.x.y.0/24)
host    all             all             100.64.0.0/10          scram-sha-256

# Reject everything else
host    all             all             0.0.0.0/0              reject
```

**Firewall layer (macOS pf):**
```bash
# Block non-Tailscale access to PostgreSQL
block in on en0 proto tcp to any port 28818
# Allow Tailscale interface
pass in on utun0 proto tcp to any port 28818
```

---

### Phase 3: WAL Archiving (20 min)

**Enable WAL for Point-in-Time Recovery:**

`postgresql.conf` additions:
```conf
# WAL Settings
wal_level = replica
archive_mode = on
archive_command = 'cp %p /Volumes/pg-archive/%f'
max_wal_senders = 3
wal_keep_size = 1GB
```

**Setup archive directory:**
```bash
mkdir -p /Volumes/pg-archive
chmod 700 /Volumes/pg-archive
```

**Recovery test (when needed):**
```bash
# Restore to specific point in time
pg_ctl stop -D ~/.pgrx/18.1/data
cp -r ~/.pgrx/18.1/data ~/.pgrx/18.1/data.backup
pg_ctl start -D ~/.pgrx/18.1/data -o "-c restore_command='cp /Volumes/pg-archive/%f %p'"
```

---

### Phase 4: Monitoring (30 min)

**Health check endpoint:**

`/usr/local/bin/pg-health.sh:`
```bash
#!/bin/bash
# PostgreSQL health check for Tailscale Funnel

PORT=28818
TS_FUNNEL_URL="https://login.tailscale.com/..."

if pg_isready -h localhost -p $PORT; then
    # Run basic query
    RESULT=$(psql -h localhost -p $PORT -d postgres -tAc "SELECT 1")
    if [ "$RESULT" = "1" ]; then
        echo '{"status":"healthy","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
        exit 0
    fi
fi

echo '{"status":"unhealthy","timestamp":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}'
exit 1
```

**Metrics collection:**
```sql
-- HNSW performance tracking
CREATE EXTENSION IF NOT EXISTS ruvector;

CREATE MATERIALIZED VIEW hnsw_metrics AS
SELECT
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch,
    pg_size_pretty(pg_relation_size(indexrelid)) as index_size
FROM pg_stat_user_indexes
WHERE indexname LIKE '%hnsw%';

-- Refresh every minute
-- REFRESH MATERIALIZED VIEW hnsw_metrics;
```

---

## Database Roles & Access

**Role hierarchy:**
```sql
-- Coordination clients (gmktec workers)
CREATE ROLE coordination_worker WITH LOGIN PASSWORD 'xxx';
GRANT CONNECT ON DATABASE ruvector TO coordination_worker;
GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO coordination_worker;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO coordination_worker;

-- Cloud infra (read-heavy, some writes)
CREATE ROLE cloud_client WITH LOGIN PASSWORD 'xxx';
GRANT CONNECT ON DATABASE ruvector TO cloud_client;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO cloud_client;
GRANT INSERT, UPDATE ON SPECIFIC TABLES TO cloud_client;

-- Admin (full access)
CREATE ROLE coordination_admin WITH LOGIN PASSWORD 'xxx' SUPERUSER;
```

**PgBouncer user mapping:**
```ini
[databases]
ruvector = host=localhost port=28818 dbname=ruvector

[pgbouncer]
auth_type = md5
auth_file = /opt/homebrew/etc/userlist.txt
```

---

## Connection Strings

**From gmktec-k9:**
```bash
# Direct (fallback)
export DATABASE_URL="postgres://coordination_worker:pass@100.x.y.z:28818/ruvector"

# Via PgBouncer (recommended)
export DATABASE_URL="postgres://coordination_worker:pass@100.x.y.z:6432/ruvector"
```

**From cloud infra:**
```bash
export DATABASE_URL="postgres://cloud_client:pass@100.x.y.z:6432/ruvector"
```

---

## Disaster Recovery

### Backup Strategy

| Type | Frequency | Retention | Location |
|------|-----------|-----------|----------|
| WAL | Continuous | 7 days | `/Volumes/pg-archive/` |
| Full dump | Daily | 30 days | `/Volumes/pg-backups/` |
| Snapshot | Weekly | 4 weeks | Time Machine |

**Automated backup script:**
```bash
#!/bin/bash
# /usr/local/bin/pg-backup.sh

BACKUP_DIR="/Volumes/pg-backups"
DATE=$(date +%Y%m%d)
pg_dumpall -h localhost -p 28818 | gzip > "$BACKUP_DIR/pg_all_$DATE.sql.gz"
# Keep last 30 days
find "$BACKUP_DIR" -name "pg_all_*.sql.gz" -mtime +30 -delete
```

### Recovery Procedures

**Restore from dump:**
```bash
gunzip -c /Volumes/pg-backups/pg_all_20260114.sql.gz | psql -h localhost -p 28818
```

**PITR from WAL:**
```bash
# 1. Stop PostgreSQL
pg_ctl stop -D ~/.pgrx/18.1/data

# 2. Create recovery config
echo "restore_command = 'cp /Volumes/pg-archive/%f %p'" > ~/.pgrx/18.1/data/recovery.signal

# 3. Start PostgreSQL (replays WAL)
pg_ctl start -D ~/.pgrx/18.1/data
```

---

## Performance Tuning

**Current config from earlier session:**
```conf
# Memory
shared_buffers = 16GB
effective_cache_size = 48GB
work_mem = 256MB

# Parallelism
max_parallel_workers = 12
max_parallel_workers_per_gather = 8

# RuVector extension
ruvector.ef_search = 128
ruvector.probes = 10
```

**PgBouncer tuning for coordination workload:**
```ini
pool_mode = transaction          # Best for coordination/queuing
default_pool_size = 50           # 50 concurrent backend connections
max_client_conn = 1000           # 1000 concurrent clients
server_lifetime = 3600           # Rotate connections hourly
```

---

## Rollout Checklist

- [ ] Install PgBouncer
- [ ] Configure pgbouncer.ini with Tailscale IP
- [ ] Create database roles (coordination_worker, cloud_client)
- [ ] Update pg_hba.conf for Tailscale network only
- [ ] Enable WAL archiving
- [ ] Set up backup cron job
- [ ] Deploy health check script
- [ ] Test connectivity from gmktec-k9
- [ ] Test connectivity from cloud infra (if available)
- [ ] Update Tailscale ACL with coordination-hub tag
- [ ] Document connection strings in shared location

---

## estimated Timeline

| Phase | Time | Dependencies |
|-------|------|--------------|
| PgBouncer setup | 15 min | None |
| Security hardening | 10 min | PgBouncer |
| WAL archiving | 20 min | Storage mount |
| Monitoring | 30 min | None |
| Testing | 15 min | All above |
| **Total** | **90 min** | |

---

## References

- Tailscale ACL: Current policy at `/Users/devops/.claude/...`
- PostgreSQL docs: https://www.postgresql.org/docs/18/
- PgBouncer docs: https://www.pgbouncer.org/usage.html
- RuVector docs: `docs/postgres/`
