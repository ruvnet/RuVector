//! Shared fixture: the production `RESOURCE_ALLOWLIST` (edge/auth-worker
//! wrangler.toml) and the DCR policy the Worker derives from it.

use ruvector_edge_authz::client::{DcrPolicy, DEFAULT_CLIENT_SCOPE};
use ruvector_edge_authz::ResourceAllowlist;

pub const ALLOWLIST: &str = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 ruvector:read ruvector:write ruvector:admin offline_access, https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp ruvector:read ruvector:write offline_access, https://team.ruv.io/mcp team:read team:write team:run offline_access";

pub fn allowlist() -> ResourceAllowlist {
    ResourceAllowlist::from_config(ALLOWLIST).expect("production allowlist loads")
}

pub fn dcr_policy(allowlist: &ResourceAllowlist) -> DcrPolicy {
    DcrPolicy {
        scopes_supported: allowlist.scopes_supported(),
        default_scope: DEFAULT_CLIENT_SCOPE.iter().map(|s| s.to_string()).collect(),
    }
}
