CREATE TABLE IF NOT EXISTS memberships (
  issuer TEXT NOT NULL,
  subject TEXT NOT NULL,
  tenant_id TEXT NOT NULL,
  role TEXT NOT NULL CHECK(role IN ('viewer', 'editor', 'admin')),
  enabled INTEGER NOT NULL DEFAULT 1 CHECK(enabled IN (0, 1)),
  PRIMARY KEY (issuer, subject, tenant_id)
);
CREATE INDEX IF NOT EXISTS memberships_subject ON memberships (issuer, subject, enabled);

CREATE TABLE IF NOT EXISTS vectors (
  tenant_id TEXT NOT NULL,
  collection TEXT NOT NULL,
  id TEXT NOT NULL,
  dimension INTEGER NOT NULL,
  vector_json TEXT NOT NULL,
  metadata_json TEXT NOT NULL,
  updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (tenant_id, collection, id)
);
CREATE INDEX IF NOT EXISTS vectors_scope ON vectors (tenant_id, collection);

CREATE TRIGGER IF NOT EXISTS vectors_dimension_guard BEFORE INSERT ON vectors
WHEN EXISTS (SELECT 1 FROM vectors WHERE tenant_id = NEW.tenant_id AND collection = NEW.collection AND dimension != NEW.dimension)
BEGIN SELECT RAISE(ABORT, 'dimension_mismatch'); END;

CREATE TRIGGER IF NOT EXISTS vectors_capacity_guard BEFORE INSERT ON vectors
WHEN (SELECT COUNT(*) FROM vectors WHERE tenant_id = NEW.tenant_id AND collection = NEW.collection) >= 500
 AND NOT EXISTS (SELECT 1 FROM vectors WHERE tenant_id = NEW.tenant_id AND collection = NEW.collection AND id = NEW.id)
BEGIN SELECT RAISE(ABORT, 'collection_capacity_reached'); END;

CREATE TABLE IF NOT EXISTS usage_daily (
  tenant_id TEXT NOT NULL,
  day TEXT NOT NULL,
  used INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (tenant_id, day)
);
