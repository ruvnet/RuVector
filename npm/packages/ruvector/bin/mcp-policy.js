'use strict';

/**
 * MCP tool-access policy (ADR-256 — default-deny posture).
 *
 * Borrows metaharness's "default-deny allowlist" security concept using a small,
 * pure, testable module. Operators restrict the exposed/callable MCP tool surface
 * via environment variables:
 *
 *   RUVECTOR_MCP_ALLOW="hooks_route,hooks_recall"  → only these are exposed/callable
 *   RUVECTOR_MCP_DENY="hooks_security_scan"        → these are blocked (wins over allow)
 *   RUVECTOR_MCP_PROFILE=readonly                  → a curated safe subset
 *
 * Precedence: DENY > ALLOW/PROFILE > (default allow-all).
 * With no policy configured, all registered tools are allowed (backward compatible);
 * `policy.configured` is false so the server can nudge toward an allowlist.
 *
 * This module is dependency-free and side-effect-free so it can be unit-tested
 * without spawning an MCP stdio server.
 */

/** Curated read-only / low-risk tool subsets. */
const MCP_PROFILES = {
  readonly: [
    'ruvector',
    'hooks_stats',
    'hooks_recall',
    'hooks_route',
    'hooks_route_enhanced',
    'hooks_suggest_context',
    'hooks_capabilities',
    'hooks_export',
    'hooks_doctor',
    'hooks_attention_info',
    'hooks_gnn_info',
  ],
};

/** Split a comma/space separated env list into a clean array. */
function parseList(value) {
  if (!value || typeof value !== 'string') return [];
  return value
    .split(/[,\s]+/)
    .map((s) => s.trim())
    .filter(Boolean);
}

/**
 * Build a tool-access policy from an env-like object (defaults to process.env).
 * Returns { allowSet: Set|null, deny: Set, profile: string|null, configured: bool }.
 */
function buildToolPolicy(env) {
  const e = env || process.env;
  const allow = parseList(e.RUVECTOR_MCP_ALLOW);
  const deny = new Set(parseList(e.RUVECTOR_MCP_DENY));
  const profileName = e.RUVECTOR_MCP_PROFILE && e.RUVECTOR_MCP_PROFILE.trim();

  let allowSet = null;
  if (profileName && MCP_PROFILES[profileName]) {
    allowSet = new Set(MCP_PROFILES[profileName]);
  }
  if (allow.length) {
    allowSet = new Set([...(allowSet || []), ...allow]);
  }

  return {
    allowSet,
    deny,
    profile: profileName && MCP_PROFILES[profileName] ? profileName : null,
    // An unknown profile name with no allow/deny is treated as "not configured".
    configured: Boolean(allowSet || deny.size),
  };
}

/** Is a tool name permitted under the given policy? DENY always wins. */
function isToolAllowed(name, policy) {
  if (!policy) return true;
  if (policy.deny.has(name)) return false;
  if (policy.allowSet) return policy.allowSet.has(name);
  return true;
}

/** Filter a TOOLS array (objects with a `.name`) down to the allowed subset. */
function filterAllowedTools(tools, policy) {
  if (!Array.isArray(tools)) return [];
  return tools.filter((t) => t && isToolAllowed(t.name, policy));
}

module.exports = {
  MCP_PROFILES,
  parseList,
  buildToolPolicy,
  isToolAllowed,
  filterAllowedTools,
};
