# ADR-067: MCP Gate Permit System

**Status**: Accepted, Implemented
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-014 (Coherence Engine), ADR-058 (Hash Security Optimization), ADR-059 (Shared Brain Google Cloud), ADR-060 (Shared Brain Capabilities)

## 1. Context

AI agents executing in production environments need a decision gate between intent and action. An agent that can call any tool without verification is a liability. The coherence gate (ADR-014) provides the mathematical foundation — anytime-valid e-processes, conformal prediction sets, mincut structural witnesses — but it needs an MCP-native interface so Claude Code agents can request permissions via the standard tool-calling protocol.

The `mcp-gate` crate (`crates/mcp-gate/`) wraps the `cognitum-gate-tilezero` coherence gate in an MCP stdio server. It exposes three tools: `permit_action` (request permission), `get_receipt` (audit trail), and `replay_decision` (deterministic verification). All decisions are recorded in a cryptographic witness chain. Contributor authentication uses SHAKE-256 pseudonym derivation, and access is rate-limited via BudgetTokenBucket.

## 2. Decision

Implement `mcp-gate` as a standalone MCP stdio server that wraps the existing `cognitum-gate-tilezero` gate. Use SHAKE-256 for pseudonym derivation (consistent with the brain server's authentication scheme), challenge nonces for replay protection, and BudgetTokenBucket for rate limiting. All gate decisions produce cryptographically chained witness receipts.

## 3. Architecture

### 3.1 Crate Structure

```
crates/mcp-gate/
  src/
    lib.rs       -- Public API, re-exports
    main.rs      -- Entry point (McpGateServer::run_stdio)
    server.rs    -- MCP protocol handler (JSON-RPC over stdio)
    tools.rs     -- Tool implementations (permit, receipt, replay)
    types.rs     -- Request/response types, JSON-RPC types
  Cargo.toml
```

The crate depends on:
- `cognitum-gate-tilezero`: The coherence gate engine (TileZero)
- `tokio`: Async runtime for stdio processing
- `serde` / `serde_json`: JSON-RPC serialization
- `thiserror`: Error types

### 3.2 MCP Tools

**`permit_action`**: Request permission for an action. The gate evaluates three witnesses:

| Witness | What it measures | Output |
|---------|-----------------|--------|
| Structural | Mincut graph analysis of the action's connectivity | `cut_value`, `partition` status, critical edges |
| Predictive | Conformal prediction set for the action's outcome | `set_size`, `coverage` target |
| Evidential | Anytime-valid e-process accumulation | `e_value`, `verdict` (accept/continue/reject) |

Returns one of three decisions:
- **Permit**: Action allowed. Returns a `PermitToken` (base64-encoded, time-bounded) and a witness receipt.
- **Defer**: Action escalated. Returns escalation info (reason, suggested reviewer) and a witness receipt.
- **Deny**: Action blocked. Returns denial reason and a witness receipt.

**`get_receipt`**: Retrieve a witness receipt by sequence number. Each gate decision produces a receipt containing the decision, timestamp, witness summary, and a hash linking to the previous receipt. The chain is cryptographic — tampering with any receipt breaks the chain.

**`replay_decision`**: Deterministically replay a past decision given the same inputs and state snapshot. Optionally verifies the hash chain integrity up to the replayed sequence number. Returns whether the replayed decision matches the original.

### 3.3 SHAKE-256 Pseudonym Derivation

Contributor pseudonyms are derived from API keys using SHAKE-256, identical to the brain server (ADR-059, ADR-060):

```rust
pub fn from_api_key(api_key: &str) -> AuthenticatedContributor {
    let mut hasher = Shake256::default();
    hasher.update(b"ruvector-brain-pseudonym:");
    hasher.update(api_key.as_bytes());
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 16];
    reader.read(&mut buf);
    let pseudonym = hex::encode(buf);
    // ...
}
```

This produces a 32-character hex pseudonym (128-bit) that:
- Is deterministic: same API key always produces the same pseudonym
- Is irreversible: the API key cannot be recovered from the pseudonym
- Uses a domain-separated prefix (`ruvector-brain-pseudonym:`) to prevent cross-system correlation

### 3.4 Challenge Nonce Replay Protection

Write operations (including `permit_action`) require a challenge nonce to prevent replay attacks:

1. Client calls `GET /v1/challenge` to receive a fresh nonce
2. Server generates a random nonce, stores it with a 5-minute TTL
3. Client includes the nonce in the write request
4. Server verifies the nonce exists and has not expired
5. Server consumes the nonce (single-use)

Nonces are stored in a `NonceStore` backed by a `DashMap` with periodic TTL eviction.

### 3.5 Rate Limiting

The `BudgetTokenBucket` pattern (shared with the brain server) provides per-contributor rate limiting:

```rust
pub struct RateLimiter {
    write_buckets: DashMap<String, TokenBucket>,
    read_buckets: DashMap<String, TokenBucket>,
    write_limit: u32,   // 100 writes/hour
    read_limit: u32,    // 1000 reads/hour
    window: Duration,   // 1 hour
}
```

Each contributor (identified by pseudonym) gets independent token buckets for reads and writes. Buckets refill at window boundaries. Stale buckets (unused for 2 windows) are periodically evicted to prevent unbounded memory growth. The cleanup runs every 1000 operations.

### 3.6 Multi-Factor Reputation Scoring

The reputation system gates access levels:

```
composite = accuracy^2 * uptime * stake_weight
```

| Factor | Measurement | Update |
|--------|-------------|--------|
| `accuracy` | Bayesian Beta(1,1) prior, updated from vote outcomes | `(upvotes+1)/(upvotes+downvotes+2)` after min 5 observations |
| `uptime` | EMA of activity frequency | `uptime = 0.95 * uptime + 0.05` per contribution |
| `stake_weight` | Fixed at 1.0 for v1 | Future: based on contribution volume |

- New contributors start at composite ~0.1. Their gate requests are weighted 10x less.
- Contributors with composite < 0.1 are flagged as potential poisoners and their permit requests are auto-denied.
- Inactivity decay: `accuracy *= 0.95^months_inactive`, `uptime *= 0.90^months_inactive`.

### 3.7 Contributor Pseudonym Revocation

A contributor can be revoked by setting their reputation below the poisoning threshold (0.1). The `check_poisoning_penalty` function triggers when:
- The contributor has >= 5 downvotes on their contributions
- Their quality score falls below 0.2

Once penalized, the contributor's pseudonym is effectively revoked — all gate requests return Deny. The pseudonym remains in the system for audit purposes. The contributor can only recover by generating a new API key (and thus a new pseudonym), starting at the cold-start reputation of 0.1.

## 4. Implementation

### 4.1 Server Entry Point

```rust
let server = McpGateServer::new();
server.run_stdio().await.expect("Server failed");
```

The server reads JSON-RPC requests from stdin, dispatches to tool handlers, and writes responses to stdout. The protocol version is `2024-11-05`.

### 4.2 Request/Response Flow

```
stdin -> parse JSON-RPC -> match method:
  "initialize"   -> return server info + capabilities
  "tools/list"   -> return [permit_action, get_receipt, replay_decision]
  "tools/call"   -> match tool name:
    "permit_action"   -> TileZero::decide() -> PermitResponse | DeferResponse | DenyResponse
    "get_receipt"     -> TileZero::get_receipt(seq) -> GetReceiptResponse
    "replay_decision" -> TileZero::replay(seq) -> ReplayDecisionResponse
-> serialize JSON-RPC -> stdout
```

### 4.3 Permit Action Request

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "permit_action",
    "arguments": {
      "action_id": "cfg-push-7a3f",
      "action_type": "config_change",
      "target": { "device": "router-west-03", "path": "/network/interfaces/eth0" },
      "context": { "agent_id": "ops-agent-12", "session_id": "sess-abc123", "urgency": "normal" }
    }
  }
}
```

### 4.4 Permit Response (Permitted)

```json
{
  "decision": "permit",
  "token": "eyJ0eXAi...",
  "valid_until_ns": 1737158400000000000,
  "witness": {
    "structural": { "cut_value": 12.7, "partition": "stable", "critical_edges": 0 },
    "predictive": { "set_size": 3, "coverage": 0.92 },
    "evidential": { "e_value": 847.3, "verdict": "accept" }
  },
  "receipt_sequence": 1847392
}
```

### 4.5 Error Codes

| Code | Meaning |
|------|---------|
| -32001 | Receipt not found |
| -32002 | Chain verification failed |
| -32602 | Invalid request parameters |
| -32603 | Internal server error |
| -32700 | JSON parse error |

## 5. Consequences

### Positive

- **Standard MCP interface**: Any MCP-compatible client (Claude Code, custom agents) can request gate permits without custom integration
- **Cryptographic audit trail**: Every decision is chained. Tampering is detectable. Replays are deterministic.
- **Shared auth scheme**: SHAKE-256 pseudonym derivation is identical between `mcp-gate` and `mcp-brain-server`, enabling cross-system contributor identity without sharing API keys
- **Defense in depth**: Rate limiting + nonce replay protection + reputation gating + cryptographic witnesses form layered protection

### Negative

- **Stdio only**: The MCP gate is stdio-based, not SSE. It must run as a local process. Remote access requires wrapping in a network transport (deferred).
- **Stateful**: The TileZero gate accumulates witness receipts in memory. Long-running sessions with many decisions grow linearly. Persistence to disk is deferred.

### Neutral

- The three-witness model (structural, predictive, evidential) is inherited from `cognitum-gate-tilezero`. The MCP gate does not add new decision logic — it provides protocol access to existing capabilities.
- Pseudonym revocation is soft (reputation penalty) not hard (key revocation). A determined attacker can generate new API keys. This is acceptable because cold-start reputation (0.1) limits the blast radius of new pseudonyms.
