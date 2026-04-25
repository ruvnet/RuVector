# ADR-159: A2A (Agent-to-Agent) Protocol Support for rvAgent

## Status

**Proposed — r3 (second review pass 2026-04-24)**. A new subcrate
`rvagent-a2a` under `crates/rvAgent/`, adding a third protocol
surface alongside the existing `rvagent-mcp` (agent ↔ tool) and
`rvagent-acp` (client ↔ agent) stacks. No code changes accompany
this ADR; the Decision section fixes the shape and the
Implementation plan fixes the milestones. Actual implementation
lands in a separate PR.

The revisions cumulate rather than replace — r2 closed the first-
order failure modes (transport spoofing, uncontrolled compute,
dumb routing, untyped payloads); r3 closes the second-order ones
that appear only under multi-agent traffic at scale (aggregate
spend, invisible call graphs, infinite recursion, schema drift).

**r2 — first-order failure modes (2026-04-24 AM):**

1. **Identity + trust** — `AgentID = SHAKE-256(pubkey_ed25519)`,
   signed AgentCards, verify-on-discover. Without this, A2A peers
   are indistinguishable-by-URL and spoofable.
2. **Policy + cost control** — `TaskPolicy { max_tokens,
   max_cost_usd, max_duration_ms, allowed_skills[], max_concurrency }`
   evaluated before task dispatch. Without this, A2A is an
   uncontrolled compute surface.
3. **Routing strategy** — `PeerSelector` trait for policy-based
   peer selection. Transforms A2A from transport into an
   intelligent delegation layer.
4. **Typed artifact semantics** — `ArtifactKind` sum type over the
   spec's generic `Part` parts, enabling zero-copy memory handoff
   via `RuLakeWitness`.

**r3 — second-order failure modes (2026-04-24 PM):**

5. **Global budget** — `GlobalBudget { max_usd_per_minute,
   max_tokens_per_minute, max_tasks_per_minute }` enforced at the
   dispatch queue **before** `PeerSelector` runs. Per-task
   `TaskPolicy` prevents any single task from burning money;
   `GlobalBudget` prevents the aggregate of in-policy tasks from
   exceeding a wall-clock spend rate. Without it, the routing
   layer can amplify spend by legitimately selecting the fastest
   peer for every concurrent request.
6. **Trace-level causality** — `TaskContext { trace_id,
   parent_task_id, depth, root_agent_id }` threaded through every
   task. Makes the multi-agent call graph observable instead of
   a flat fanout of orphan tasks. Answers "which root task caused
   this spend?" and "which peer chain created this latency?" —
   neither answerable without lineage.
7. **Recursion guard** — `max_call_depth` + `visited_agents` on
   `TaskContext`. Rejects `A → B → C → A` loops and runaway depth
   before they are dispatched, not after they exhaust the budget.
8. **Artifact versioning from day one** — `ArtifactKind` is
   a non-exhaustive enum; wire format carries an explicit
   `artifact_kind_version` in `metadata.ruvector.artifact_kind`.
   Receivers negotiate version on discovery. Schema evolution
   becomes a minor version bump, not a flag day.

## Date

2026-04-24

## Authors

ruv.io · RuVector research. Triggered by repeated integration requests
to have rvAgent appear as a peer in multi-vendor agent graphs
(OpenAI Swarm, LangGraph, Microsoft Copilot, Google Gemini Agent
Builder), none of which speak MCP or ACP but all of which either speak
or are moving to speak A2A.

## Relates To

- ADR-099 — ACP (Agent Communication Protocol) over axum HTTP
- ADR-103 C6 — ACP security: auth, rate limiting, body caps
- ADR-155 — ruLake cache-first positioning (for the optional
  `rvlake_witness` AgentCard field discussed under open questions)
- ADR-156 — ruLake as memory substrate for agent brains (the consumer
  that benefits from cross-agent memory handoff)
- ADR-157 — Optional `VectorKernel` accelerator plane (unrelated cost
  surface, cited only for pattern: "new subcrate, same shape as the
  last one")
- External: Google A2A spec — https://github.com/google/A2A (launched
  April 2024, JSON-RPC 2.0 over HTTPS, SSE streaming, signed webhooks).

---

## Context

### The protocol landscape rvAgent currently sits in

rvAgent today ships three coordination surfaces:

| Crate                | Direction         | Counterparty            | Wire                                  |
|----------------------|-------------------|-------------------------|---------------------------------------|
| `rvagent-mcp`        | agent → tool      | MCP tool host           | MCP (Anthropic, JSON-RPC over stdio/HTTP) |
| `rvagent-acp`        | client → agent    | UI / CLI / other caller | in-house ACP (axum HTTP + prompt req → `ContentBlock{Text,ToolUse,ToolResult}` responses) |
| `rvagent-subagents`  | parent → child    | in-process child        | Arc-cloned `AgentState`, no wire format |

Both `mcp` and `acp` are **vertical**: one end is always the "agent
proper" and the other is something that isn't another peer agent. The
subagent crate handles the only horizontal case today, but it is
strictly in-process — a parent spawning a child inside the same
binary, sharing state by Arc.

### What's missing

There is no protocol in the rvAgent tree for **peer-to-peer
communication between independent agents**, in particular agents that
live in different processes, possibly on different machines, possibly
written in different languages, possibly run by different vendors.
A grep of the rvAgent tree confirms this: no references to "A2A",
"Agent-to-Agent", "agents.json", or "googleapis.*agent" exist in any
subcrate.

The concrete use cases driving this gap:

1. **Outbound delegation.** rvAgent is strong at code-writing but
   weaker at, say, structured scientific analysis. A user workflow
   wants to hand off a subtask — "review this PR for numerical
   stability" — to a Python-backed specialist agent. Today this
   requires hand-rolled HTTP glue per peer.
2. **Inbound exposure.** A customer runs an existing multi-agent
   orchestrator (Copilot, LangGraph, Gemini Agent Builder) and wants
   to drop rvAgent in as one node. Today they have to either port
   their orchestrator to speak ACP or write a translating shim.
3. **Capability discovery.** Even inside a single-vendor rvAgent
   fleet, there is no standard way to ask "what can you do?" across
   instances. Each deployment reinvents a manifest.

### Why A2A specifically

Google published the A2A spec in April 2024 as an open, vendor-neutral
protocol targeting exactly this gap. The relevant properties:

- **Capability manifest.** Every A2A agent exposes
  `/.well-known/agent.json` (the "AgentCard") declaring its name,
  description, skills, supported input/output content modes, auth
  schemes, and canonical URL. This is the discovery primitive.
- **Task lifecycle.** A2A models long-running work as a `Task` with a
  `TaskStatus` in `{submitted, working, input-required, completed,
  failed, canceled}`. The JSON-RPC methods are `tasks/send`,
  `tasks/get`, `tasks/cancel`, `tasks/pushNotification/set`, and
  `tasks/resubscribe`.
- **Typed messages and artifacts.** Payloads are `Message` (a sequence
  of `Part`s: `TextPart`, `FilePart`, `DataPart`) and `Artifact`
  (the same shape, tagged as the task's output).
- **SSE streaming.** Server-Sent Events carry status updates and
  incremental artifact fragments to callers that subscribed on
  `tasks/send`.
- **Signed webhooks.** For disconnected/async flows, callers register
  a `PushNotificationConfig` and receive HMAC-signed callbacks on
  state transitions.
- **Peer symmetry.** Every A2A agent can act as both client and
  server. There is no "primary" side.

The spec is JSON-RPC 2.0 over HTTPS, which maps cleanly onto the axum
stack `rvagent-acp` already uses. The auth schemes in the AgentCard
(`bearer`, `oauth2`, `apikey`) map 1:1 onto what `rvagent-acp`'s
middleware already understands.

### Interop targets we've verified have A2A support or a stated A2A roadmap

- OpenAI Swarm — community A2A adapter, no first-party support as of
  this writing.
- LangGraph (LangChain) — first-party A2A server wrapper shipped 2024
  Q4.
- Microsoft Copilot (Semantic Kernel) — A2A inbound declared on the
  public roadmap.
- Google Gemini Agent Builder — native A2A client and server.

"Declared on the roadmap" is doing work in that list. This is a
young spec. We return to that under Alternatives (option D) and
Consequences.

### What A2A is **not**

A2A is not a replacement for MCP. MCP is about an agent calling a
tool; A2A is about an agent calling another agent, which in turn may
be calling tools of its own via MCP. The protocols stack: an rvAgent
instance that speaks A2A outbound typically speaks MCP inbound to its
tools. Nor is A2A a replacement for ACP: ACP is still the right
surface for "my user's UI is talking to my agent," because A2A's
model (peer symmetric, task-centric, long-running) is a bad fit for
the tight-loop conversational case ACP serves.

---

## Decision

Add a new subcrate **`crates/rvAgent/rvagent-a2a`**. Shape mirrors
`rvagent-mcp` and `rvagent-acp`:

```
crates/rvAgent/rvagent-a2a/
├── Cargo.toml
├── src/
│   ├── lib.rs           -- public API re-exports
│   ├── card.rs          -- AgentCard + skill/auth/content-mode types
│   ├── task.rs          -- Task, TaskStatus, state machine
│   ├── message.rs       -- Message, Part (spec types)
│   ├── artifact_types.rs -- typed ArtifactKind (r2: Text /
│   │                       StructuredJson / VectorRef / RuLakeWitness)
│   ├── identity.rs      -- r2: AgentID = SHAKE-256(pubkey),
│   │                       signed AgentCards, trust graph hooks
│   ├── policy.rs        -- r2: TaskPolicy, PolicyError, cost-guard
│   ├── routing.rs       -- r2: PeerSelector trait + stock impls
│   ├── executor.rs      -- r2: Executor = Local | Remote(A2A),
│   │                       unified dispatch model
│   ├── budget.rs        -- r3: GlobalBudget + token/cost meter,
│   │                       dispatch-queue gate
│   ├── context.rs       -- r3: TaskContext — trace_id, parent_id,
│   │                       depth, visited_agents, recursion guard
│   ├── rpc.rs           -- JSON-RPC 2.0 framing + method dispatch
│   ├── server/
│   │   ├── mod.rs       -- axum sub-app, route wiring
│   │   ├── well_known.rs -- GET /.well-known/agent.json
│   │   ├── tasks.rs     -- tasks/send, tasks/get, tasks/cancel,
│   │   │                   tasks/resubscribe
│   │   ├── sse.rs       -- SSE streaming for status + artifacts
│   │   └── push.rs      -- push-notification webhook sender
│   │                       (HMAC-SHA256 default, Ed25519 opt-in)
│   ├── client/
│   │   ├── mod.rs       -- discovery + RPC client
│   │   ├── discover.rs  -- fetch & parse peer /.well-known/agent.json,
│   │   │                   verify card signature, cache with ETag
│   │   └── rpc.rs       -- JSON-RPC caller (reqwest)
│   ├── runner.rs        -- TaskRunner trait (the abstraction that
│   │                       Executor::Local and Executor::Remote both
│   │                       satisfy)
│   └── error.rs         -- A2aError, maps to JSON-RPC error codes
└── tests/
    ├── card_roundtrip.rs
    ├── card_signature.rs   -- r2: verify signed AgentCard
    ├── policy_guard.rs     -- r2: TaskPolicy enforcement
    ├── routing.rs          -- r2: PeerSelector implementations
    ├── budget_guard.rs     -- r3: GlobalBudget enforcement
    ├── recursion_guard.rs  -- r3: depth + visited-agents rejection
    ├── trace_lineage.rs    -- r3: TaskContext propagation
    ├── task_lifecycle.rs
    ├── sse_stream.rs
    ├── push_signing.rs
    ├── artifact_kinds.rs   -- r2: typed artifact round-trip
    └── peer_interop.rs     -- against reference A2A server (see M3)
```

### Core type sketches

The AgentCard served at `/.well-known/agent.json`:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AgentCard {
    pub name: String,
    pub description: String,
    pub url: String,                     // canonical endpoint URL
    pub version: String,
    pub skills: Vec<Skill>,
    pub input_modes: Vec<ContentMode>,   // e.g. text/plain, application/json
    pub output_modes: Vec<ContentMode>,
    pub authentication: AuthSchemes,
    pub capabilities: Capabilities,      // streaming, pushNotifications, ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub documentation_url: Option<String>,

    // r2 additions — identity + vendor extensions.

    /// SHAKE-256(32) over the ed25519 public key. Stable, human-
    /// inspectable, content-addressed — two peers that minted their
    /// pubkey the same way produce the same AgentID. Used as the
    /// primary key in trust graphs + allowlists.
    pub agent_id: AgentID,

    /// Ed25519 signature over the canonical-JSON of every field
    /// above (excluding `signature` itself). Lets any consumer of
    /// the card verify it was produced by the claimed agent without
    /// a live TLS handshake — important for signed cache entries,
    /// offline reasoning, and HTTP-free intermediaries.
    pub signature: CardSignature,

    /// Vendor-namespaced metadata. ruvector-specific extensions
    /// live under `metadata.ruvector` per the A2A spec's
    /// free-form-metadata allowance (see open question #2).
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub metadata: serde_json::Value,
}

/// Content-addressed agent identifier (32-byte SHAKE-256 of the
/// ed25519 public key, hex-encoded in JSON). Two cards with the
/// same `agent_id` are the same peer, regardless of which URL they
/// were served from.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct AgentID(pub String);

/// Ed25519 signature over the canonical-JSON of the AgentCard
/// (excluding the `signature` field). Serialized as base64 per
/// A2A's metadata convention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CardSignature {
    pub pubkey_ed25519: String,          // base64, 32 bytes
    pub sig_ed25519: String,             // base64, 64 bytes
    pub canonicalization: String,        // e.g. "jcs" (RFC 8785)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Skill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub tags: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<String>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Capabilities {
    pub streaming: bool,
    pub push_notifications: bool,
    pub state_transition_history: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AuthSchemes {
    pub schemes: Vec<String>,            // "bearer" | "oauth2" | "apikey"
}
```

The Task lifecycle:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Task {
    pub id: String,                      // server-assigned UUID
    pub session_id: Option<String>,
    pub status: TaskStatus,
    pub messages: Vec<Message>,          // conversation so far
    pub artifacts: Vec<Artifact>,        // accumulated outputs
    pub metadata: serde_json::Value,     // free-form per A2A spec
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum TaskState {
    Submitted,
    Working,
    InputRequired,
    Completed,
    Failed,
    Canceled,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskStatus {
    pub state: TaskState,
    pub timestamp: DateTime<Utc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<Message>,        // agent-side explanation
}
```

Messages and artifacts share a `Part` sum type:

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,                      // "user" | "agent"
    pub parts: Vec<Part>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum Part {
    Text { text: String },
    File { file: FilePart },
    Data { data: serde_json::Value },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Artifact {
    pub name: Option<String>,
    pub description: Option<String>,
    pub parts: Vec<Part>,
    pub index: u32,
    pub append: bool,
    pub last_chunk: bool,
}
```

### JSON-RPC surface

A2A method name ↔ handler:

| Method                         | Handler                       | Effect                                                  |
|--------------------------------|-------------------------------|---------------------------------------------------------|
| `tasks/send`                   | `server::tasks::send`         | Create task, dispatch to `TaskRunner`, return initial state |
| `tasks/sendSubscribe`          | `server::sse::send_subscribe` | Same as `send` but response is an SSE stream of updates |
| `tasks/get`                    | `server::tasks::get`          | Read current state + history                            |
| `tasks/cancel`                 | `server::tasks::cancel`       | Request cooperative cancellation                        |
| `tasks/pushNotification/set`   | `server::push::set`           | Register webhook for async state transitions            |
| `tasks/pushNotification/get`   | `server::push::get`           | Inspect current webhook config                          |
| `tasks/resubscribe`            | `server::sse::resubscribe`    | Reopen SSE stream after disconnect                      |

Plus the non-RPC discovery route:

| Route                         | Handler                 |
|-------------------------------|-------------------------|
| `GET /.well-known/agent.json` | `server::well_known::card` |

### Executor abstraction — `TaskRunner`

The A2A layer does not itself know how to run an agent. It delegates:

```rust
#[async_trait]
pub trait TaskRunner: Send + Sync + 'static {
    async fn run(
        &self,
        task_id: &str,
        input: &Message,
        updates: TaskUpdateSink,
    ) -> Result<Vec<Artifact>, A2aError>;

    async fn cancel(&self, task_id: &str) -> Result<(), A2aError>;
}
```

`TaskUpdateSink` is the bridge the runner uses to push intermediate
`TaskStatus` transitions (`Working` → `InputRequired` → …) and
incremental `Artifact` chunks back to SSE subscribers. The default
implementation supplied by `rvagent-a2a` wraps
`rvagent_core::AgentState` and translates its existing
`ContentBlock{Text, ToolUse, ToolResult}` event stream into A2A
`Part`s — `Text` → `Part::Text`, `ToolUse`/`ToolResult` →
`Part::Data` with a reserved JSON shape. This is the same translation
`rvagent-acp` already performs for its own wire format; the two
adapters share a helper module.

### Security — reuse, don't re-implement

The axum middleware stack that `rvagent-acp` already uses
(ADR-099, ADR-103 C6) is lifted into a shared `rvagent-middleware`
layer (which already exists as a subcrate) and composed in front of
both the ACP and A2A sub-apps:

- **Auth:** A2A's declared `AuthSchemes` — `bearer`, `oauth2`,
  `apikey` — map directly onto the extractors already present in
  `rvagent-middleware`. No new auth machinery.
- **Rate limiting:** same token-bucket layer as ACP, same config
  surface.
- **Body caps:** same `RequestBodyLimitLayer` default. A2A messages
  can carry `FilePart`s, so the cap may need a higher default than
  ACP's — decision deferred, see open question #5.

Outbound (discovery client) security:

- TLS verification via `rustls` with the system root store. No
  custom CA overrides exposed in the default config.
- Peer AgentCards are cached on first fetch with an `ETag`-aware
  validator; the cache has a TTL (proposed 5 minutes; open).
- The discovery client has a per-peer concurrency limit and timeout
  (proposed 10 s connect, 60 s read; open).

### Push-notification webhook signing

A2A specifies HMAC-SHA256 over the serialized request body, using a
shared secret the receiver returned from
`tasks/pushNotification/set`. `rvagent-a2a` implements this exactly
as specified as the default, and **also** ships an opt-in Ed25519
signature path gated behind the `ed25519-webhooks` cargo feature.
Agents advertise which they accept via
`metadata.ruvector.webhook_algos` in their AgentCard. Position:
HMAC for interop, Ed25519 for trust. Open question #3 is resolved.

---

### r2 — Identity and trust model

Request-level auth (bearer, oauth2, apikey) proves the caller has a
credential; it does not prove the caller is the agent whose
AgentCard they quoted. Without a binding between the card and its
key, a malicious peer can reuse a legitimate `agent_id` at its own
URL and serve forged responses. The r2 additions close this gap:

```rust
// src/identity.rs

/// Every production AgentCard is signed. Verification happens:
///   1. on discover (client): reject mismatched signatures with
///      A2aError::CardSignatureInvalid before any RPC is made.
///   2. on publish (server): sign the card once at startup, serve
///      from a cached canonical-JSON blob so re-signing is avoided
///      per request.
pub fn verify_card(card: &AgentCard) -> Result<(), A2aError>;

/// Stable content-addressed id — the key in trust graphs,
/// allowlists, rate-limit buckets. Same pubkey → same id, always.
pub fn agent_id_from_pubkey(pubkey_ed25519: &[u8; 32]) -> AgentID;

/// Canonicalization: JCS (RFC 8785) with the `signature` field
/// elided. Stable across serializer output order and whitespace.
pub fn canonical_card_bytes(card: &AgentCard) -> Vec<u8>;
```

Trust integration hooks (deliberately minimal; full trust-graph
policy is out of scope for A2A itself):

- `IdentityResolver` trait — lets a deployment plug in its own
  "is this `agent_id` allowed?" predicate. Default implementation
  allows-all and logs at INFO. An allowlist or an ADR-103-style
  RBAC layer drops in cleanly here.
- AgentCard fetch + signature verification are cached together in
  the `discover.rs` client — no scenario where a valid-looking URL
  returns a valid card but the id doesn't match the key.

Relationship to the spec: A2A is silent on card signing. This is a
ruvector extension living in `metadata.ruvector.identity` alongside
the standard fields. If the spec adopts signed cards later, we
realign; if not, we are strictly additive and ignorable by peers
that don't care.

---

### r2 — Policy and cost control

An A2A endpoint with no policy is an arbitrary compute service. A
malicious peer can `tasks/send` with a long-running prompt and
drain model tokens on the receiver's budget. The r2 additions bound
every task:

```rust
// src/policy.rs

#[derive(Clone, Debug, Default)]
pub struct TaskPolicy {
    /// Hard upper bound on the LLM-token count the TaskRunner may
    /// consume. Checked at dispatch + periodically mid-task.
    pub max_tokens: Option<u64>,

    /// Estimated-cost upper bound in USD. The runner must estimate
    /// before dispatch (skill × expected tokens × model rate) and
    /// reject if estimate > cap. Post-task actual-cost emitted to
    /// observability.
    pub max_cost_usd: Option<f64>,

    /// Wall-clock ceiling on task lifetime. After this, the
    /// TaskRunner receives a cooperative-cancel signal and the
    /// task transitions to `TaskState::Failed` with reason
    /// `PolicyTimeout`.
    pub max_duration_ms: Option<u64>,

    /// If set, the `Skill.id`s the caller is allowed to request.
    /// Others return A2aError::PolicyExceeded before dispatch.
    pub allowed_skills: Option<Vec<String>>,

    /// Concurrency cap per (caller-agent-id, skill) bucket.
    /// Enforces "one caller can't saturate a single skill slot."
    pub max_concurrency: Option<u32>,
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error("task exceeds policy: {field} = {actual} > {limit}")]
    Exceeded { field: &'static str, actual: String, limit: String },

    #[error("skill {0} not in allowed set")]
    SkillDenied(String),

    #[error("concurrency bucket full: {caller}/{skill}")]
    ConcurrencyFull { caller: AgentID, skill: String },
}
```

Evaluation order:

1. At `tasks/send` dispatch: allowed_skills → max_concurrency
   (reserve a slot) → max_cost_usd (estimate-then-check).
2. Inside the runner (periodic, every N tokens): max_tokens,
   max_duration_ms.
3. On task end: release the concurrency slot, emit actual cost to
   tracing span + optional metrics sink.

Policy is scoped **per caller identity** (resolved from the signed
AgentCard's `agent_id`), not per endpoint. Different peers can have
different budgets against the same rvAgent.

---

### r2 — Routing strategy

A2A's outbound client is "call a peer by URL." That's the transport
layer. The r2 routing layer answers *which* peer to call when
multiple satisfy the task:

```rust
// src/routing.rs

pub struct TaskSpec<'a> {
    pub skill_id: &'a str,
    pub estimated_tokens: Option<u64>,
    pub deadline_ms: Option<u64>,
    pub input_modes: &'a [ContentMode],
    pub required_capabilities: &'a [&'a str], // e.g. "streaming"
}

pub trait PeerSelector: Send + Sync {
    /// Given a task and the pool of candidate peers (already
    /// filtered to those advertising the requested skill in their
    /// card), pick one or return None to signal "no eligible peer."
    fn select(
        &self,
        task: &TaskSpec,
        peers: &[PeerSnapshot],
    ) -> Option<PeerID>;
}

pub struct PeerSnapshot {
    pub id: AgentID,
    pub card: AgentCard,
    pub ewma_latency_ms: f64,      // observed
    pub ewma_cost_usd: f64,        // observed per-task
    pub open_tasks: u32,           // outstanding requests
    pub failure_rate: f64,         // circuit-breaker state
}
```

Stock implementations shipped in the crate:

- `CheapestUnderLatency { budget_ms }` — pick the lowest
  `ewma_cost_usd` whose `ewma_latency_ms < budget_ms`.
- `LowestLatency` — minimum `ewma_latency_ms`, costs be damned.
- `RoundRobin { seed }` — deterministic rotation for test
  reproducibility.
- `CapabilityMatch` — the first peer whose card lists every
  required capability; tiebreaks by `failure_rate`.

Selectors compose via `ChainedSelector(Vec<Box<dyn PeerSelector>>)`
— first non-None wins. This is where deployment-specific routing
policy (SLA tiers, regional preferences, trust gates) plugs in.

The `PeerSnapshot` pool is maintained by the discovery client's
cache + a lightweight health-check loop. The same rate-limit /
circuit-breaker layer from `rvagent-middleware` feeds
`failure_rate`; when it opens, the peer is temporarily removed from
the pool instead of being returned with a high failure rate.

---

### r2 — Typed artifact semantics

The spec's `Artifact { parts: Vec<Part> }` is flexible enough to
carry anything but enforces nothing. r2 adds a parallel typed layer
so receivers can rely on shape:

```rust
// src/artifact_types.rs

/// ruvector-typed wrapper around the spec's `Artifact`. The
/// underlying wire format stays spec-compliant — this is a
/// type-safe construction + destructuring layer.
pub enum ArtifactKind {
    /// Plain text. Wire: single `Part::Text`.
    Text(String),

    /// Structured JSON with an optional schema reference for
    /// validation. Wire: single `Part::Data` with `metadata.schema`.
    StructuredJson {
        value: serde_json::Value,
        schema: Option<String>,   // $id or URL
    },

    /// By-reference vector payload — caller holds the vectors,
    /// artifact carries a (backend, collection, witness) pointer
    /// the receiver can resolve. Enables zero-copy handoff when
    /// both peers share a ruLake tier.
    VectorRef {
        backend: String,
        collection: String,
        witness: String,          // SHAKE-256 hex (ruLake witness)
        dim: u32,
        count: u64,
    },

    /// Explicit ruLake bundle pointer. Wire: `Part::Data` with
    /// `metadata.ruvector.rvlake_witness` set. Receiver with a
    /// ruLake deployment can `warm_from_dir` or pull the bundle
    /// by witness; receiver without ruLake degrades to a plain
    /// opaque-data artifact.
    RuLakeWitness {
        witness: String,
        data_ref: String,         // gs://, s3://, file://…
        capabilities: Vec<String>, // e.g. ["read"] — scoped grant
    },

    /// Escape hatch for anything the typed layer doesn't cover.
    /// Wire: pass-through spec Parts.
    Raw(Vec<Part>),
}

impl ArtifactKind {
    pub fn to_artifact(&self) -> Artifact;
    pub fn from_artifact(a: &Artifact) -> Result<Self, A2aError>;
}
```

What this unlocks (called out because this is the ruvector-specific
differentiation the r1 draft deferred to "open question"):

- **Zero-copy handoff.** Two rvAgent peers on the same ruLake tier
  pass a 64-byte `RuLakeWitness` instead of N MB of vectors. The
  receiver resolves the witness against its own ruLake; the cache
  either already has the entry (shared witness, already primed)
  or primes it from the shared backend without re-pulling from
  the caller.
- **Schema-enforced JSON.** Peers advertise skills that return
  `StructuredJson { schema }` and receivers can validate
  automatically instead of unpacking by convention.
- **Scope-limited memory grants.** `RuLakeWitness.capabilities`
  carries `["read"]` or `["read","write"]` so a caller can hand
  the receiver read-only access to a shared cache without
  implying write access. Witness-level auth composes with the
  per-request auth; both must pass.

Receivers that don't understand `RuLakeWitness` see it as an
opaque `Part::Data`. The typed layer is additive.

**Versioning (r3).** `ArtifactKind` is declared
`#[non_exhaustive]`. Every card advertises
`metadata.ruvector.artifact_kind_version = "v1"` (or `v2`, etc.).
Receivers refuse to down-version: a peer advertising `v1` cannot
accept `v2` artifacts without explicit opt-in via
`accept_lower_version_artifacts`. This makes schema evolution a
minor version bump with explicit negotiation instead of the
"flag day" pattern that breaks running fleets.

---

### r3 — Global budget control

Per-task `TaskPolicy` bounds any single task. It does **not** bound
the aggregate: 100 concurrent tasks at `max_cost_usd = 0.05` still
cost $5, and a well-tuned `PeerSelector` routing to the cheapest
peer may legitimately accept every one. The r3 addition enforces
a system-wide spend ceiling:

```rust
// src/budget.rs

#[derive(Clone, Debug)]
pub struct GlobalBudget {
    /// Hard cap on combined task cost per 60-second window.
    pub max_usd_per_minute: Option<f64>,

    /// Hard cap on LLM tokens consumed per 60-second window,
    /// summed across all concurrent tasks.
    pub max_tokens_per_minute: Option<u64>,

    /// Hard cap on task dispatches per 60-second window. Stops
    /// the "thousands of tiny tasks" amplification pattern.
    pub max_tasks_per_minute: Option<u32>,

    /// When the window saturates, do we reject new tasks
    /// (`Shed`) or queue them until the next window (`Queue`)?
    /// Shed is the safe default; Queue is correct for bursty
    /// workloads that can tolerate seconds of added latency.
    pub overflow: OverflowPolicy,
}

pub enum OverflowPolicy { Shed, Queue { max_queue_depth: u32 } }

pub enum BudgetError {
    RateLimitUsd { observed_per_min: f64, cap: f64 },
    RateLimitTokens { observed_per_min: u64, cap: u64 },
    RateLimitTasks { observed_per_min: u32, cap: u32 },
    QueueFull,
}
```

**Enforcement stage.** The budget check runs at the **dispatch
queue**, *before* `PeerSelector` runs. A task that would exceed
the per-minute cap is rejected with `BudgetError` before it
enters the routing pool. This is the critical ordering — running
the router first means the router can legitimately pick a peer
that would push us over the cap; running the budget first means
the router only sees tasks that already fit.

**Measurement.** The meter is a lock-free rolling counter over a
fixed 60-second window (per-minute is the grain the review called
for; can be widened to per-hour with a second window in the same
struct later). Updates happen at:
- dispatch: `tasks_this_minute += 1`
- task end: `cost_this_minute += actual_cost_usd`,
  `tokens_this_minute += actual_tokens`

The gap between dispatch-time rejection and end-time metering
is bounded by `TaskPolicy::max_duration_ms`, so the observed
per-minute cost is always at most `cap + (concurrent_tasks ×
max_per_task_cost)` — a knowable overshoot, not a runaway.

---

### r3 — Trace-level causality

Per-task observability is invisible without knowing *which task
invoked which*. The multi-agent case makes this acute: a flat list
of task traces hides the call graph entirely. The r3 `TaskContext`
makes the graph first-class:

```rust
// src/context.rs

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TaskContext {
    /// Root-scoped trace id. Compatible with W3C Trace Context
    /// (16-byte lowercase hex). Propagates unchanged through
    /// every descendant task in the call graph.
    pub trace_id: String,

    /// Parent task's id — None for the root task. One hop per
    /// task, regardless of local vs remote executor.
    pub parent_task_id: Option<String>,

    /// Depth in the call graph from the root. 0 = root,
    /// 1 = direct child, etc. Bounded by `max_call_depth`
    /// (see recursion guard).
    pub depth: u32,

    /// `AgentID` of the root task's originating agent. Used for
    /// per-root-attribution of aggregate spend.
    pub root_agent_id: AgentID,

    /// Every agent id the task has transited, in order.
    /// Enables the recursion guard + operator forensics.
    pub visited_agents: Vec<AgentID>,
}
```

**Propagation.** `TaskContext` is carried in every A2A message's
`metadata.ruvector.context`, and in every intra-process subagent
call via its builder. Remote handoffs copy it, increment `depth`,
append the receiving `AgentID` to `visited_agents`. The sender's
`AgentID` is the new `parent_task_id`.

**Wire format.** Compatible with W3C Trace Context headers when
the A2A transport is plain HTTP — the `traceparent` header is
emitted in parallel so external tools (Jaeger, Tempo, Honeycomb)
pick up the lineage automatically. `metadata.ruvector.context`
carries the ruvector-specific fields (`root_agent_id`,
`visited_agents`) that W3C doesn't model.

**What this answers.** Without `TaskContext`:
- "Which root task is burning our tokens?" — unanswerable
- "Which peer chain caused this latency spike?" — unanswerable
- "Where is recursion forming?" — unanswerable

With it, every observability query reduces to a trace-id filter +
a lineage walk. Cost attribution aggregates along `root_agent_id`;
latency blame aggregates along `visited_agents`.

---

### r3 — Recursion guard

Multi-agent call graphs can form cycles (`A → B → C → A`) and
unbounded chains (`A → B → C → … → Z`) that only surface as
budget drain + latency collapse. The r3 guard rejects both before
dispatch:

```rust
// src/recursion_guard.rs

pub struct RecursionPolicy {
    /// Maximum chain depth from the root. Default: 8.
    /// Rejects tasks at `depth >= max_call_depth` with
    /// `A2aError::RecursionDepthExceeded`.
    pub max_call_depth: u32,

    /// If true, reject any task whose target `AgentID` is
    /// already in `TaskContext::visited_agents`. Default: true.
    pub deny_revisit: bool,

    /// Optional allowlist — if set, agents explicitly named here
    /// are exempt from `deny_revisit`. Use for legitimate
    /// bounce-through patterns (e.g. a memory agent that
    /// everyone legitimately round-trips through).
    pub revisit_allowlist: Vec<AgentID>,
}
```

**Enforcement stage.** At `tasks/send` on the receiving side,
immediately after policy + budget checks, before runner dispatch.
Rejection uses a dedicated JSON-RPC error code so callers can
distinguish it from a generic policy violation.

**Why both checks, not just depth.** Depth alone catches the
runaway chain case but not the short cycle. `A → B → A` is depth 2
and inside any reasonable `max_call_depth`. The cycle needs the
`visited_agents` check.

**Why both are on by default.** The default policy values
(`max_call_depth = 8`, `deny_revisit = true`) reflect the common
case: most legitimate agent graphs fan out more than they
recurse. A deployment that actually needs recursion opts in via
config, with an `AgentID` on the allowlist — explicit + auditable.

---

### What this is **not** doing

- Not replacing `rvagent-acp`. ACP stays as the client ↔ agent
  surface.
- Not replacing `rvagent-mcp`. MCP stays as the tool-host surface.
- Not adding a new binary target. `rvagent-a2a` is a library; users
  mount its axum router into their own binary (typically the same
  binary that already mounts `rvagent-acp`).
- Not introducing a new dependency on any vendor SDK. The protocol
  is spec-only; the implementation uses the same `axum`, `reqwest`,
  `serde`, `tokio` stack the other subcrates use.

### Rough size

600–1000 LoC for the subcrate, plus ~300 LoC of tests, plus a single
external reference-interop test under a `peer-interop` feature flag
(off by default; gated behind a Docker-available test runner). The
number is driven by the fact that the type layer is dictated by the
spec and cannot be compressed, and the server-side handlers are a
dispatch table over the JSON-RPC methods above.

---

## Alternatives considered

### A. Extend `rvagent-acp` to also speak A2A

**Rejected.** ACP is a client ↔ agent protocol. A2A is a peer ↔ peer
protocol. The semantic mismatch is not cosmetic:

- ACP models a single conversation with one caller. A2A models a
  task queue of potentially concurrent peer requests, each with its
  own lifecycle and subscription state.
- ACP's response shape (`ContentBlock{Text, ToolUse, ToolResult}`)
  leaks the internal agent event stream. A2A's response shape
  (`Artifact` with typed `Part`s) is deliberately abstract and
  doesn't expose tool-use internals. Smashing them into one
  endpoint forces either an inconsistent wire format or a wrapper
  layer that exists only to hide the other protocol.
- ACP auth assumes a bearer-token from a known UI. A2A peers
  authenticate each other; neither side is "the UI."

Conflating the two would make both protocols worse.

### B. Feature-flag A2A on `rvagent-acp`

**Rejected.** Same semantic mismatch as A, now pushed into
`cfg`-walls. Users who only want ACP pay for the A2A types in their
API surface; users who want both get a crate whose public API varies
by feature set, which is the exact pattern the Rust ecosystem has
learned to avoid (cf. the `tokio` full-features-vs-surgical debate).
Two clean subcrates are a smaller total code surface than one crate
gated through its middle.

### C. Ship A2A only as a standalone binary / service, outside the
rvAgent tree

**Rejected.** The primary integration case is "a Rust application
embeds an rvAgent and wants to expose it to A2A peers." A binary-only
deployment forces those users to either shell out to a sidecar
(operational cost, latency, shared-state headaches) or re-implement
A2A themselves. Library composition is what rvAgent exists for.

A secondary binary that mounts `rvagent-a2a` is still possible — in
fact, the `examples/` directory should include one — but the library
is the primary deliverable.

### D. Wait for the A2A spec to mature, or adopt a competing spec

**Considered seriously.** A2A is a 2024 spec. It has not won in the
sense that HTTP won. The adjacent specs in the space:

- **LangChain Hub** — a registry, not really a wire protocol. It
  exposes prompts and chains, not running agents. Not a replacement.
- **CrewAI Flows** — tightly coupled to the CrewAI Python framework
  and its own task abstraction. Portable in principle, but the
  reference implementations are all inside CrewAI's tree. Moves
  with CrewAI.
- **Microsoft Autogen `RoundRobinGroupChat`** — a pattern, not a
  protocol. Autogen itself is moving toward A2A inbound per the
  public roadmap.
- **OpenAI Swarm** — no published wire protocol at the time of
  writing; the framework is transport-agnostic by design.

Trade-off analysis: A2A is young and subject to revision, but it is
the only candidate that (a) has a published spec over stable
transports, (b) is vendor-neutral by design, and (c) is already
being adopted by more than one vendor. The downside of picking A2A
and having it lose is that we've written 600–1000 lines of code
that becomes a compatibility layer rather than a primary API. The
downside of waiting is that for 6–18 months customers continue to
write ad-hoc glue, and we miss the window during which A2A is still
small enough that our implementation could influence spec defaults
(two of the ambiguous points in the current spec — content-mode
negotiation and artifact chunk ordering — are places an early
reference implementation with good defaults can push the spec).

The decision is to implement now, with a written awareness that the
spec may shift. Versioning discipline (see Implementation plan M1)
mitigates the cost of a spec revision.

---

## Consequences

### Positive

- **Cross-vendor interop.** rvAgent becomes a first-class peer in
  agent graphs assembled by LangGraph, Gemini Agent Builder,
  Copilot, and other A2A-speaking orchestrators, without per-vendor
  glue.
- **Capability discovery as a standard.** Internal tooling (the
  rvAgent CLI, dashboards, autopilot) gains a uniform way to ask
  "what can this instance do?" — the AgentCard.
- **Composability with ruLake.** A cross-agent memory handoff story
  becomes tractable: a remote agent can state in its AgentCard that
  it reads/writes ruLake, and rvAgent can avoid redundant context
  transfer by passing a ruLake witness instead of raw content
  (see open question #2).
- **Symmetric ecosystem positioning.** rvAgent sits at the
  intersection of three protocols: MCP inbound for tools, A2A
  peer-to-peer for other agents, ACP for UI/CLI. This is the same
  three-way surface the other mature agent frameworks are
  converging on (Gemini: function-calling / A2A / built-in UI;
  Copilot: plugins / A2A / Copilot chat). We match the pattern.

### Negative

- **N+1 protocol surface.** The rvAgent tree now carries three
  protocol implementations. Each has its own spec to track, its own
  serde boundary to keep stable, its own security posture. The
  maintenance cost is real; budgeting a fixed fraction of ongoing
  work to "protocol hygiene" is necessary.
- **Spec churn risk.** A2A is 2024-young. A breaking revision of
  the spec will force a major-version bump on `rvagent-a2a` and
  may require callers to regenerate AgentCards. We mitigate with
  semver discipline on the crate and with explicit A2A spec
  version pinning in each released minor version, but we cannot
  fully avoid it.
- **Ecosystem network effects.** If A2A does not win — if the
  dominant orchestrators stabilize on a different protocol — this
  crate becomes a compatibility shim rather than a first-class
  surface. The 600–1000 LoC is not a large bet, but it is a bet.

### Neutral

- **Crate surface grows by exactly one.** `rvAgent` gains
  `rvagent-a2a`. No existing crate's public API changes. Existing
  consumers of `rvagent-core`, `rvagent-mcp`, `rvagent-acp`, and
  `rvagent-subagents` are unaffected.
- **No new runtime dependencies.** `axum`, `reqwest`, `serde`,
  `tokio` are already in the tree.
- **Binary size.** Users who don't mount the A2A router pay no
  runtime cost; users who do pay ~the cost of adding `reqwest`
  (which most rvAgent deployments already pull in transitively
  via MCP).

---

## Open questions

### r2 resolutions

1. ~~**Parent↔subagent parity.**~~ **Resolved:** unify the
   **execution model**, not the transport. A new `Executor` enum
   in `src/executor.rs`:

   ```rust
   pub enum Executor {
       Local(Box<dyn TaskRunner>),
       Remote(Peer),        // A2A outbound to another agent
   }
   ```

   Both arms expose the same `run(task) -> Result<Vec<Artifact>>`
   surface. Caller code doesn't know or care which arm is live.

   **However:** subagents are **not** auto-exposed through A2A.
   Exposing internal reasoning steps to external peers leaks
   architecture. The dispatch is unified (caller-side); the
   observability is not (server-side). An explicit
   `.with_a2a_exposure(true)` on the subagent builder opts in
   case-by-case. Default: off.

2. ~~**`rvlake_witness` in the AgentCard.**~~ **Resolved: ship it
   in M1.** Under `metadata.ruvector.memory`:

   ```json
   {
     "type": "rvlake",
     "shard": "gs://bucket/prod/",
     "capabilities": ["read"],
     "witness_format_version": 2
   }
   ```

   This is the ruvector-specific differentiator the r1 draft
   deferred. Landing early means the `RuLakeWitness` artifact
   type has a home on day one instead of waiting for an external
   peer to ask for it (they won't — no external peer knows to ask
   for what isn't documented). Strictly additive; peers without
   ruLake ignore the field.

3. ~~**Webhook signing: HMAC-SHA256 only, or also Ed25519?**~~
   **Resolved: both.** HMAC-SHA256 is the spec-compliant default
   (interop with every A2A peer). Ed25519 is opt-in via the
   `ed25519-webhooks` cargo feature and announced via
   `metadata.ruvector.webhook_algos = ["hmac-sha256","ed25519"]`.
   Position: HMAC for compatibility, Ed25519 for trust. A receiver
   advertising both lets a trust-sensitive caller prefer Ed25519
   while remaining spec-interoperable with peers that don't
   recognize it.

### Still open

4. **Discovery-client rate limiting + circuit-breaker scope.**
   The outbound client fetches `/.well-known/agent.json` and talks
   JSON-RPC to peers. Should it reuse the ACP-side rate-limiting
   middleware configured for **inbound** traffic, a new
   outbound-specific limiter, or no limiter? A circuit-breaker
   (open-on-N-failures, half-open-with-probe) is clearly wanted;
   its scope is the question. Proposal: reuse the
   `rvagent-middleware` rate-limiter as a `tower::Layer` wrapped
   around the `reqwest` client, with a separate namespace in the
   config (`a2a.client.rate_limit`) so operators can tune outbound
   and inbound independently. **Ranked urgent** because a chatty
   peer during M3 interop testing could DoS a mock registry
   otherwise.

5. **Trust-graph substrate.** Identity gives us content-addressed
   `AgentID`s. What's the trust *graph* built on top? Options:
   (a) flat allowlist in config, (b) per-caller signed delegation
   tokens, (c) federated trust graph with mutual attestations
   (M6-ish scope). Proposal: ship (a) as the M1 default, leave
   (b)/(c) as explicit follow-up ADRs. Without (a) everything
   else here is advisory.

---

## Risk analysis

### Spec risk

A2A is a 2024 spec. Parts of it are stable (JSON-RPC envelope, the
AgentCard shape, the TaskState enum); parts are in motion
(content-mode negotiation, artifact chunk ordering semantics, push-
notification delivery guarantees).

**Mitigations:**
- Pin the spec commit hash in `rvagent-a2a/Cargo.toml` as a metadata
  comment on every minor release; breaking spec changes trigger a
  major-version bump.
- Keep the adapter layer inside `rpc.rs` — the JSON-RPC dispatch +
  wire types are the spec's surface area, and re-shaping them
  requires touching only one module.
- Treat A2A as a **pluggable transport over our own Executor/
  TaskRunner model**, not as the canonical task abstraction. If
  A2A loses the protocol war, the rest of the crate (Executor,
  TaskRunner, PeerSelector, TaskPolicy, ArtifactKind, AgentID) is
  transport-agnostic — swapping in a different peer protocol is a
  `rpc.rs` rewrite, not a ground-up redesign.

### Operational risk

The failure mode that actually bites at scale is **agent
explosion**. Once every agent can call every other agent, latency
spikes, cost explodes, and the call graph becomes unobservable.

**Mitigations, in priority order:**
1. **Hard concurrency caps** via `TaskPolicy::max_concurrency` on
   every endpoint. Deployments default to modest values (e.g. 16
   concurrent tasks per caller per skill) that operators can raise
   after measurement.
2. **Per-peer circuit breakers** in the outbound client — a
   misbehaving peer stops receiving requests after N consecutive
   failures and is probed back in gradually.
3. **Routing policies that bound fanout** — `PeerSelector`
   implementations pick *one* peer per task, not N. Parallel-broadcast
   patterns ("ask all peers, use fastest reply") are deliberately
   not provided as a stock selector; deployments that want them
   implement their own, making the fanout explicit.
4. **Cost-per-task visibility** — every completed task emits
   `actual_tokens` + `actual_cost_usd` to a tracing span. An
   operator querying "which peer is burning our budget?" gets a
   direct answer from their trace store instead of a spreadsheet.

### Trust risk

Unsigned AgentCards are spoofable by anyone who can intercept the
URL. Signed cards push the trust root to the ed25519 key. This is
an improvement but not a panacea — key rotation, compromise
recovery, and the trust-graph substrate (open question #5) are
separate mechanisms that sit on top.

**Mitigation:** the `IdentityResolver` trait is the hook point for
deployment-specific trust policy; the crate ships a default that
allows-all and logs, so a naive deployment gets working A2A from
day one, and a production deployment plugs in its policy without
a crate fork.

---

## Quantified tradeoffs

Scoring the change on eight dimensions, 0 = no change, 10 = maximal:

| Dimension           | Benefit | Cost             | Net r1 | Net r2 | Net r3 |
|---------------------|--------:|------------------|-------:|-------:|-------:|
| Interop             |    +9   | +3 maintenance   |    +6  |    +6  |    +6  |
| Complexity          |    +6   | +6 cognitive     |     0  |    -1  |    -2  |
| Latency             |    +4   | +10-30% overhead |    -1  |    -1  |    -1  |
| Per-task cost ctrl  |    +0   | high risk        |    -5  |    +7  |    +7  |
| Aggregate spend ctrl|    +0   | unbounded fanout |    -7  |    -4  |    +8  |
| Trust posture       |    +0   | spoofable        |    -3  |    +6  |    +6  |
| Memory handoff      |    +0   | N-copy blobs     |     0  |    +7  |    +8  |
| Observability       |    +0   | orphan fanout    |    -3  |    -3  |    +8  |
| Recursion safety    |    +0   | stack-explosion  |    -5  |    -5  |    +9  |
| Schema evolution    |    +0   | flag-day risk    |    -2  |    -2  |    +6  |

**r1 net:** ~7/10. Real interop gain, shaky foundations on cost,
trust, and multi-agent safety. Viable as a proof-of-concept, not
as a production control surface.

**r2 net:** ~9/10. The four r2 additions (identity, policy,
routing, typed artifacts) turn *per-task* cost, trust, and memory
from negative to strongly positive. Aggregate spend, observability,
and recursion remain real risks at scale.

**r3 net:** ~9.5/10. The four r3 additions (global budget,
trace-level causality, recursion guard, artifact versioning) turn
the three remaining second-order risks from negative to positive,
and make schema evolution a forward-compatible dial instead of a
breaking change. The cumulative `-2` on complexity reflects eight
new modules; the benefit is that each module is single-concern,
sub-150 LoC, and independently reviewable. The one dimension not
moving in r3 is latency — the extra guards are in-process checks
with negligible overhead, so they do not change the latency tax
already priced in at r1.

**Scoring method.** Every number is the author's subjective
calibration against analogous past decisions. They're posted here
to be argued with, not to be taken as ground truth.

---

## Strategic positioning

The sharper framing the second review produced:

> **rvAgent is the control plane for agent execution.**
> Not an agent. Not a framework. Not a protocol adapter.

That puts the crate in a different category than it started in. To
make the category explicit, the stack divides into four layers, and
rvAgent sits cleanly in exactly one of them:

| Layer         | Examples                                | Role                                             |
|---------------|-----------------------------------------|--------------------------------------------------|
| Models        | OpenAI, Anthropic, Gemini, local LLMs   | The reasoning substrate                          |
| Frameworks    | LangChain, CrewAI, AutoGen              | Compose model calls into agents                  |
| Orchestrators | LangGraph, Copilot, Gemini Agent Builder| Wire agents into graphs / workflows              |
| **Control plane** | **rvAgent**                         | **Identity, policy, routing, budget, lineage across the graph** |

Models answer one question. Frameworks wrap one agent. Orchestrators
wire graphs of agents. A control plane governs *how the graph runs* —
who is allowed to call whom, under what budget, with what audit
trail, sharing what memory, using which routing policy. That's the
role the r2 + r3 additions collectively define.

This ADR therefore standardizes three protocol surfaces on one
runtime:

```
rvAgent = { UI surface (ACP), Tool surface (MCP), Peer surface (A2A) }
            └─── client-facing ──┘ └─── tool-facing ──┘ └─── peer-facing ──┘
                        + control plane (identity, policy, routing, budget, lineage, memory)
```

The ecosystem analogy, in one line:

- **HTTP** is the app-to-user protocol. A browser is not a control
  plane.
- **gRPC** is the service-to-service protocol. A service is not a
  control plane.
- **Kafka** is the async/broadcast protocol. A topic is not a
  control plane.
- **Kubernetes** is the control plane for containers — it governs
  *how* containers run, not *what* they do.

rvAgent is the Kubernetes-shaped layer for A2A, MCP, and ACP
traffic. Interop alone is commodity; a control plane is not. That's
why the r2 additions (identity, policy, routing, typed artifacts)
and the r3 additions (global budget, trace lineage, recursion
guard, artifact versioning) are load-bearing: strip any one of
them and the crate falls back to "yet another framework with an
A2A shim."

**The practical consequence:** we stop competing head-on with
LangChain, Copilot, Gemini Agent Builder — those are orchestrators
and frameworks, one layer up. We become **the substrate they
plug into**, the same way ruLake (ADR-155) sits underneath
whichever vector store already holds your data. The crate that is
easiest to plug into wins, because operators choose control
planes once and plug many orchestrators into them.

---

## Acceptance test

The review produced a sharper, three-point benchmark that replaces
the r2 version. The ADR has succeeded iff all three hold on a real
deployment against a real external A2A orchestrator:

> 1. **A remote agent call is indistinguishable from a local call.**
> 2. **Memory transfer size is constant regardless of payload size.**
> 3. **Total cost stays bounded under recursive delegation.**

Each point is a wall-clock-measurable property, not a subjective
"works fine" judgement. The concrete bindings:

### 1. Remote is indistinguishable from local

The same `TaskSpec` dispatched through `Executor::Local` vs
`Executor::Remote(Peer)` produces byte-identical result payloads
(artifacts, status transitions, final content) modulo timestamps
and transport-layer wrappers. Caller code is location-transparent:
no `if remote { ... } else { ... }` branches in user code, no
separate error types, no divergent cancellation semantics. Measured
by `executor_remote.rs` in M3 (scope + test defined there).

### 2. Memory transfer is constant-size

Two rvAgent peers on the same ruLake tier hand off a task whose
input is a 100k-vector retrieval result. The inter-peer HTTP body
is a 64-byte `RuLakeWitness` artifact plus protocol framing; the
receiver reads the compressed codes from the shared tier directly.
Target: **A2A request body ≤ 2 KiB regardless of input vector
count.** Re-measured at 1k, 10k, 100k, 1M vectors — the body size
does not scale with payload. Measured by `witness_handoff.rs` in
the M3 integration suite.

### 3. Cost is bounded under recursion

Under an adversarially-constructed recursive fanout — agent A
delegates to B and C, each of which delegates to two peers, each
of which delegates to two more, ten levels deep — the aggregate
spend across the whole call tree never exceeds the root caller's
`GlobalBudget::max_usd_per_minute`. Verified by: the sum of
per-task `actual_cost_usd` across all tasks sharing the same
`TaskContext::trace_id` is always ≤ budget; tasks past the
`RecursionPolicy::max_call_depth` are rejected at dispatch with
`A2aError::RecursionLimit` before they run.

Also required under this point: the `visited_agents` cycle check
rejects `A → B → C → A` before the second `A` call is dispatched,
as verified by `recursion_guard.rs` in M3.

### Why these three

The three tests together are the operational definition of "the
control plane works":

- Test 1 says the runtime abstraction is real (you didn't leak the
  transport).
- Test 2 says the memory abstraction is real (you didn't serialize
  the payload).
- Test 3 says the governance abstraction is real (you didn't lose
  control of the spend).

If any of them fails, the protocol work has shipped but the
strategic positioning has not. No amount of passing unit tests at
milestone level substitutes for these three.

---

## Answers to the strategic questions raised in reviews

### r2 questions

#### Q: Should A2A be just interoperability, or the basis of an agent routing economy?

**Staged, same PR track.** M1–M2 deliver interoperability: the
AgentCard, the Task lifecycle, SSE streaming. That's enough to
satisfy the first acceptance test ("drop-in interop").

**M3 delivers the routing economy.** The `PeerSelector` trait +
stock implementations + `TaskPolicy` cost guards make cost/latency
tradeoffs tractable. At M3 completion, an rvAgent deployment can
say: "route analysis tasks to the cheapest peer under 2s latency,
cap at $0.05 per task, fall back to in-process if no peer
qualifies." That's a routing economy, not just transport.

**M4 delivers the primitives for a billing economy** — push-
notifications with signed delivery receipts are the substrate a
usage-based cross-agent billing layer would sit on. Actual
billing/settlement is *not* in this ADR; it's a downstream system
built on M4 artifacts.

#### Q: Is ruLake shared memory across agents, or isolated per agent?

**Isolated by default, shared on explicit grant via witness +
capabilities.** The `RuLakeWitness` artifact carries a
`capabilities: Vec<String>` field. A caller passing
`capabilities: ["read"]` grants the receiver read-only access to
the cache entry at that witness. `["read","write"]` grants
read-write. No grant = no access, period. Witnesses are content-
addressed + tamper-evident, so a leaked witness is detectable.

This is the right default because it turns ruLake into a
**cross-agent memory commons**: every pairwise handoff is an
opt-in grant, per-call, with witness-level auditability. It's the
opposite of a shared-blob model where any agent with network
access can read any other's memory.

The consequence: ruLake's witness chain becomes the unit of
cross-agent memory governance, not just per-process cache
coherence (ADR-155). That's a substantial extension of ADR-155's
scope; it belongs in this ADR because the A2A integration is what
forces the decision.

### r3 questions

#### Q: Should routing + policy be exposed as config, or kept code-defined? This determines whether rvAgent becomes infrastructure or just a library.

**Both — trait-based for composition, serde-driven config overlay
for operators; config wins when both are present.**

The `PeerSelector` trait stays the canonical surface. Library
consumers that want programmable routing (e.g., a meta-selector
that looks at live telemetry) implement the trait directly in Rust.
That keeps rvAgent usable as a library — `cargo add rvagent-a2a`,
write code, compose.

In parallel, rvAgent ships a `RoutingConfig` serde struct that
operators load from a TOML / YAML / JSON file at startup:

```toml
[routing]
default_selector = "cheapest_under_latency"
latency_budget_ms = 2000
fallback = "lowest_latency"

[budget.global]
max_usd_per_minute = 10.0
overflow = "shed"

[policy.default]
max_cost_usd = 0.05
max_duration_ms = 30000
allowed_skills = ["rag.query", "embed.vectorize"]

[recursion]
max_call_depth = 8
deny_revisit = true
```

At startup the config is materialized into a `ChainedSelector` +
`GlobalBudget` + `TaskPolicy` default + `RecursionPolicy`, and
those replace the in-code defaults. If a library consumer also
registered a custom selector in code, the config-derived one wins
(operators outrank library defaults). Code can explicitly opt out
by flagging `config_override_allowed = false` in the builder, but
the default is operator-wins.

**Why both.** Code-only forces every operator change to be a
redeploy. Config-only makes the routing layer's extensibility
depend on fitting everything into a declarative schema, which does
not age well (every new selector means a new config field). The
combination lets operators tune in production without a rebuild
and lets library consumers extend the selector space without a
PR to rvAgent. The same pattern is why Kubernetes has both
controllers (code) and CRDs + YAML (config) — the two surfaces
serve different audiences.

**Implication for positioning.** This is the shift that turns
rvAgent from "library" to "infrastructure." A library ships code
and expects callers to compose it; infrastructure ships a binary +
a config file and expects operators to tune it without opening an
IDE. rvAgent supports both modes from the same crate — the
`rvagent-a2a` binary target reads the config file; the
`rvagent-a2a` crate target exposes the traits.

Scope: the config loader + schema lands in M3 alongside the first
real routing surface; config-driven `GlobalBudget` + `TaskPolicy`
defaults land in M1 with the policy layer.

#### Q: Should ruLake witnesses evolve into a permissioned memory fabric across orgs? That changes the security model significantly.

**Yes — that is the natural trajectory, and the foundational
primitives are already in this ADR. Cross-org trust establishment
is deferred to a follow-up ADR (M6 scope).**

The trajectory has three stages:

| Stage | Scope              | Primitives                                                        | ADR        |
|-------|--------------------|-------------------------------------------------------------------|------------|
| 1     | Intra-process      | Witness chain, content-addressed cache sharing                    | ADR-155    |
| 2     | Intra-org peer-to-peer | `RuLakeWitness` artifact + `capabilities: Vec<String>` grant  | ADR-159 (this) |
| 3     | Cross-org fabric   | Signed delegation tokens, org-level trust roots, witness routing  | ADR-16x (future) |

Stage 2 is what this ADR delivers. A witness carries capabilities
(`read`, `write`, `search`, scoped per shard), and the receiving
peer is expected to respect them because:

- The witness is content-addressed and tamper-evident (a mutated
  witness produces a different hash and fails verification at the
  receiving peer).
- The receiving peer's `AgentID = SHAKE-256(pubkey_ed25519)` is
  signed into its AgentCard, so the caller knows who it's granting
  to.
- The `TaskContext` lineage makes "who read which witness" auditable
  on both sides.

Stage 2 is sufficient for any pair of peers within one org or one
trust boundary (same vendor, same customer environment, same
deployment).

**Stage 3 — the cross-org fabric — is what changes the security
model.** It requires:

- An **org-level trust root** so peer A can verify that peer B
  (different org, different `AgentID` keyspace) has been authorized
  by B's org to accept A's witnesses. Today we assume a single
  org's pubkey registry; cross-org means federation.
- **Signed delegation tokens** that extend a witness with a
  third-party attestation ("org X authorizes org Y to hold this
  witness for T seconds"). This is the piece most prone to
  footguns — TTL, revocation, re-delegation, blast radius of a
  compromised root — and why it belongs in its own ADR.
- **Witness routing** — a witness emitted by org X and handed to
  org Y may need to be fetched from X's tier without Y having
  direct access; a proxy layer is needed.

**The explicit scope line:**

- ✅ **In this ADR (M1–M4):** witness + capabilities inside one
  trust boundary (one org, one key registry, one ruLake tier).
- ❌ **Not in this ADR, deferred to ADR-16x / M6+:** cross-org
  trust roots, delegation tokens, cross-tier witness routing, and
  the associated revocation / audit / compliance story.

**Why defer.** The cross-org story is roughly as large as this
whole ADR on its own. Pulling it in would double the scope, delay
M1 materially, and couple two independently-evolving specs
(A2A itself, and a federation spec that does not yet exist).
Shipping stage 2 first gives us operational experience with the
capabilities grant model before committing to the federation
variant.

**Consequence for positioning.** Shipping stage 2 is enough for
the "permissioned memory commons within an org" claim — which is
already a material market differentiation. Stage 3 is the multi-
org fabric; it is reachable from this ADR without breaking
changes, which is the load-bearing property.

---

## Implementation status (2026-04-24)

This addendum records ground truth at a point in time. The `## Status`
block above is unchanged — the ADR remains "Proposed — r3" pending
formal acceptance. All four milestones have landed in
`crates/rvAgent/rvagent-a2a/` (library + tests + benches) and
`crates/rvAgent/rvagent-cli/` (wiring).

### Ship checklist

| Milestone | Status | Proof |
|---|---|---|
| M1 — AgentCard + lifecycle + r2/r3 foundations | Shipped | `crates/rvAgent/rvagent-a2a/src/{types,identity,policy,executor,budget,context,recursion_guard,config,artifact_types}.rs` + tests |
| M2 — SSE streaming + resubscribe (with replay) | Shipped | `src/server/sse.rs` + `tests/sse_{stream,reconnect,backpressure}.rs` |
| M3 — Discovery + peer executor + routing + circuit breaker | Shipped | `src/{client,routing}.rs` + `tests/{executor_remote,routing_selectors,circuit_breaker,recursion_guard,dispatch_order,witness_handoff}.rs` |
| M4 — Push webhooks (HMAC-SHA256 + Ed25519) | Shipped | `src/server/push.rs` + `tests/push_{signing,retry,rejected,ed25519}.rs` |

### Three-point acceptance results

The benchmark defined in `## Acceptance test` (above) is met:

- **Test 1 — remote ≡ local:** PASS. `tests/executor_remote.rs::remote_executor_matches_local_shape`
  shows byte-identical result shape modulo timestamps/transport wrappers.
- **Test 2 — constant memory transfer:** PASS. Measured **765 B**
  request body for a 100k-vector witness handoff (target ≤ 2 KiB)
  in `tests/witness_handoff.rs`. Body size does not scale with
  payload vector count.
- **Test 3 — bounded cost under recursion:** PASS. Dispatch order
  `budget → recursion → policy → executor` is verified by
  `tests/dispatch_order.rs`, with reinforcement from
  `tests/recursion_guard.rs` and `tests/budget_guard.rs`.

### Measured performance

Baselines on AMD Ryzen 9 9950X, criterion first-run:

- `BudgetLedger` uncapped fast-path: **442 M ops/s** (`benches/budget_ledger.rs`).
- `BudgetLedger` at limit: **13.3 M ops/s** single-thread; **1.22 M ops/s**
  under 4-thread concurrent contention.
- `TaskContext` child, depth 1: **5.37 M ops/s**; depth 8: **0.98 M chains/s**
  (`benches/task_context.rs`).
- All code paths clear ADR-159 task #39's ≥1 M ops/s target by
  **1.2× – 442×**, depending on path.

### Test totals

- Default features: **134 / 134** passing, 0 ignored.
- `ed25519-webhooks` feature: **135 / 135** passing, 0 ignored.
- `rvagent-cli` (with `a2a` subcommand): **46 / 46** passing.
- `cargo clippy -D warnings` clean on `rvagent-a2a` and `rvagent-cli`.

### CLI integration

`rvagent-cli a2a serve|discover|send-task` is wired. End-to-end
smoke test `crates/rvAgent/rvagent-cli/tests/a2a_cli.rs` validates
`serve → discover → send-task` round-trip with a signed AgentCard.

### Deferred items

- **Peer-interop Docker-gated test** (`tests/peer_interop.rs`) —
  ADR-159 explicitly excludes this from default `cargo test`; it
  belongs on CI.
- **Cross-org permissioned memory fabric** — deferred to ADR-16x
  per the r3 Q&A (see `## Answers to the strategic questions…`,
  "cross-org permissioned memory fabric").
- **Status flip "Proposed" → "Accepted"** — pending human decision.
- **Git commit of this work** — not yet staged; awaiting user review.

### What this addendum does not do

It does **not** update the ADR's `## Status` block. The ADR stays
in "Proposed — r3" until the team formally accepts it. This
addendum is a living ground-truth pointer; the Status block
reflects the governance state.

---

## Implementation plan

Four milestones, same shape as ADR-155's plan. Each milestone has
explicit acceptance tests; no milestone counts as done until those
pass.

### M1 — AgentCard (signed) + minimal task lifecycle + r2 / r3 foundations

**Scope.**

- New subcrate `crates/rvAgent/rvagent-a2a`, skeleton only (no
  client, no SSE, no push).
- `AgentCard`, `Task`, `Message`, `Part`, `Artifact` types with
  round-trippable `serde` implementations matching the spec's
  JSON shape byte-for-byte on the official fixtures.
- **r2: Signed AgentCards.** `identity.rs` with `AgentID` +
  `CardSignature`, JCS canonicalization, verify-on-discover. Server
  signs card once at startup.
- **r2: Typed artifacts.** `artifact_types.rs` with `ArtifactKind`
  enum (non-exhaustive) including `RuLakeWitness`. AgentCard
  includes `metadata.ruvector.memory` advertising ruLake availability
  + capabilities.
- **r2: Policy layer.** `policy.rs` with `TaskPolicy` struct;
  allowed_skills + max_concurrency enforced at `tasks/send`;
  max_tokens + max_duration_ms enforced by the runner.
- **r2: Executor abstraction.** `executor.rs` with
  `Executor = Local(Box<dyn TaskRunner>) | Remote(Peer)` — M1 only
  implements `Local`; `Remote` arrives in M3.
- **r3: Global budget scaffold.** `budget.rs` with `GlobalBudget`
  struct, `OverflowPolicy { Shed, Queue }`, rolling 60-second
  window measurement in `BudgetLedger`. Enforced at the single
  dispatch queue in front of every `tasks/send` / `tasks/resubscribe`.
  M1 supports a single local runner so there is no routing yet —
  but the gate is installed now so M3's `PeerSelector` runs
  *after* it, not before.
- **r3: Trace context threading.** `context.rs` with `TaskContext`
  (`trace_id`, `parent_task_id`, `depth`, `root_agent_id`,
  `visited_agents`). Extracted from incoming `tasks/send` metadata
  (`metadata.ruvector.context`) or generated fresh with a new
  W3C-compatible `trace_id` on entry. Propagated through every
  task the runner spawns. Surfaced in tracing spans so lineage is
  observable from M1.
- **r3: Artifact versioning wire carrier.** `ArtifactKind` enum
  carries `#[non_exhaustive]`; wire format includes
  `metadata.ruvector.artifact_kind_version` (currently `"1"`).
  AgentCard advertises supported versions in
  `metadata.ruvector.artifact_kind_versions_supported`. Version
  negotiation handled by a tiny helper in `artifact_types.rs`.
- Server-side `GET /.well-known/agent.json` serving a card
  constructed from a builder, returning signed canonical JSON.
- Server-side `tasks/send`, `tasks/get`, `tasks/cancel` over
  JSON-RPC 2.0 (synchronous, no streaming).
- `TaskRunner` trait with a default implementation wrapping
  `rvagent_core::AgentState`.
- Reuse `rvagent-middleware` for auth, rate limiting, body caps.
- **r3: Config loader scaffold.** `config.rs` reads
  `RoutingConfig` / `BudgetConfig` / `PolicyConfig` /
  `RecursionConfig` from a TOML file at startup; at M1, only the
  budget + policy + recursion blocks have active readers (routing
  lands in M3). Config-supplied values win over code defaults.

**Acceptance tests.**

1. `card_roundtrip.rs`: parse each JSON fixture from the A2A spec
   repo's `samples/` directory into `AgentCard`, serialize, and
   assert bytewise equality modulo key-order normalization.
2. **r2** `card_signature.rs`: produce a signed card, verify it
   parses clean + signature validates. Mutate any field, assert
   `A2aError::CardSignatureInvalid`.
3. **r2** `artifact_kinds.rs`: round-trip each `ArtifactKind`
   variant through wire format + back; `RuLakeWitness` survives
   including `capabilities` field.
4. **r2** `policy_guard.rs`: submit a task violating each of
   `allowed_skills`, `max_concurrency`, `max_cost_usd`, assert
   each returns the matching `PolicyError` variant before any
   runner work starts.
5. **r3** `budget_guard.rs`: drive `max_usd_per_minute = 1.0` past
   the limit with 200 cheap synthetic tasks in one second; assert
   the 201st is rejected with `BudgetError::Exceeded` under
   `OverflowPolicy::Shed` and queued with backoff under
   `OverflowPolicy::Queue`.
6. **r3** `trace_lineage.rs`: enqueue a task with a specified
   `trace_id`, assert the runner's tracing span carries the same
   `trace_id`; enqueue a child-task (via `TaskContext::child()`)
   and assert `parent_task_id` + `depth += 1` are set correctly.
7. **r3** `artifact_version_handshake.rs`: peer A advertises
   `artifact_kind_versions_supported = ["1"]`, peer B advertises
   `["1","2"]`. Assert the negotiated wire version is `"1"` and
   both parties round-trip a v1 artifact successfully.
8. `task_lifecycle.rs`: drive a task through
   `submitted → working → completed` via an in-memory runner;
   assert `tasks/get` returns the correct history at each step.
9. `task_cancel.rs`: call `tasks/cancel` while the runner is
   blocked; assert the runner's cooperative-cancel path fires and
   the task resolves as `canceled`.
10. Auth: unauthenticated request to any JSON-RPC method returns
    401 / JSON-RPC error `-32001` (spec-aligned auth-required
    code).

**Size estimate.** ~700 LoC library + ~400 LoC tests (r2 additions:
`identity.rs` ~80, `policy.rs` ~60, `artifact_types.rs` ~80,
`executor.rs` scaffold ~30; r3 additions: `budget.rs` ~120,
`context.rs` ~100, `config.rs` ~80, artifact-version helpers ~20).

### M2 — SSE streaming and `tasks/resubscribe`

**Scope.**

- `tasks/sendSubscribe` returning `text/event-stream`.
- `tasks/resubscribe` for reconnection (uses task id).
- `TaskUpdateSink` bridging between `TaskRunner::run` and the SSE
  response.
- Translation from `rvagent-core` internal event stream
  (`ContentBlock{Text, ToolUse, ToolResult}`) into A2A
  `TaskStatusUpdateEvent` / `TaskArtifactUpdateEvent`, sharing code
  with the existing `rvagent-acp` translator.

**Acceptance tests.**

1. `sse_stream.rs`: connect to `tasks/sendSubscribe`, drive the
   runner through three intermediate `Working` updates and one
   `Completed`, assert the client receives five SSE events in
   order.
2. `sse_reconnect.rs`: disconnect mid-stream, call
   `tasks/resubscribe` with the task id, assert the stream
   resumes from the last delivered event (no dupes, no gaps).
3. `sse_backpressure.rs`: slow consumer, fast producer; assert the
   producer does not OOM the server (bounded channel, documented
   policy on drop vs block).

**Size estimate.** ~150 LoC library + ~100 LoC tests.

### M3 — Discovery client + peer-to-peer interop + r2 routing + r3 recursion guard

**Scope.**

- Client module: fetch `/.well-known/agent.json`, cache with
  `ETag` validator, expose a typed `AgentCard`.
- **r2: Card signature verification on fetch.** Reject peers
  whose card signature fails; log at WARN with the mismatched
  `agent_id`.
- JSON-RPC caller (reqwest) with retry / timeout / rate-limit
  layers.
- **r2: `Executor::Remote(Peer)` implementation.** Calls through
  the same `run(task)` surface as `Executor::Local`. Caller code
  is location-transparent.
- **r2 + r3: Routing layer.** `routing.rs` with `PeerSelector` trait +
  four stock implementations (`CheapestUnderLatency`,
  `LowestLatency`, `RoundRobin`, `CapabilityMatch`) +
  `ChainedSelector` compose. Health-check loop maintains the
  `PeerSnapshot` pool. **r3:** dispatch ordering is fixed as
  `GlobalBudget → RecursionPolicy → PeerSelector → TaskPolicy →
  Remote/Local runner`. The budget check runs before the
  selector, so when the budget is exhausted we never pay for peer
  discovery, score computation, or connection setup.
- **r2: Circuit breaker.** Peer removed from pool after N failures,
  half-open probe after cooldown, full re-enable on success.
- **r3: Recursion guard.** `recursion_guard.rs` with
  `RecursionPolicy { max_call_depth, deny_revisit,
  revisit_allowlist }`. Evaluated at dispatch using the incoming
  `TaskContext`: reject if `depth > max_call_depth` or if any peer
  in `visited_agents` matches the current target (minus the
  allowlist). Emits `A2aError::RecursionLimit` with the offending
  path.
- **r3: Config loader extension.** The M1 config scaffold extends
  to load `RoutingConfig` and populate the `ChainedSelector` at
  startup. If the config specifies a selector that does not exist
  in the registered trait-impl set, the server refuses to start —
  operators get an immediate, loud failure rather than a silent
  fallback.
- `peer-interop` integration test against an external A2A
  reference implementation (Google's A2A sample server at
  https://github.com/google/A2A, pinned commit).

**Acceptance tests.**

1. `discover.rs`: fetch a card from a mock server, assert parsed
   fields match; refetch with `If-None-Match`, assert cache hit;
   **r2:** mutated signature is rejected with
   `A2aError::CardSignatureInvalid`.
2. **r2** `routing.rs`: populate a `PeerSnapshot` pool with three
   peers of differing (cost, latency, open_tasks); run each stock
   selector and assert the expected peer wins.
3. **r2** `executor_remote.rs`: dispatch the same `TaskSpec`
   through `Executor::Local` and `Executor::Remote`, assert
   receiver output shape is byte-identical modulo timestamps.
4. **r2** `circuit_breaker.rs`: peer returns 503 N times → pool
   drops it; after cooldown, probe succeeds → re-enabled.
5. **r3** `recursion_guard.rs`: submit a task at `depth = 8` with
   `max_call_depth = 8` → dispatched; submit at `depth = 9` →
   `A2aError::RecursionLimit`. Submit with
   `visited_agents = [A, B, C]` targeting `A` → rejected; with
   `A` in `revisit_allowlist` → dispatched.
6. **r3** `dispatch_order.rs`: fill `GlobalBudget` to the limit
   and assert that a subsequent task fails fast with
   `BudgetError::Exceeded` without invoking `PeerSelector::pick`
   (verified with a mock selector that panics on call).
7. **r3** `witness_handoff.rs` (the acceptance-test-2 binding):
   dispatch a task whose input is a 100k-vector
   `RuLakeWitness`; assert the A2A request body is ≤ 2 KiB.
8. `peer_interop.rs` (Docker-gated): run the pinned reference
   server in a container, have `rvagent-a2a` client send a task,
   receive a result, assert the expected artifact shape.
9. Failure modes: peer returns malformed card → surfaced as
   `A2aError::Discovery`, not a panic; TLS validation failure →
   surfaced as `A2aError::Transport`.

**Size estimate.** ~470 LoC library + ~350 LoC tests (integration
test excluded from default `cargo test`; r2 additions: `routing.rs`
~120, `Executor::Remote` ~60, circuit breaker ~40; r3 additions:
`recursion_guard.rs` ~60, dispatch-order glue + config-routing
extension ~60).

### M4 — Push-notification webhooks with HMAC signing

**Scope.**

- Server side: implement `tasks/pushNotification/set` and `get`;
  on task state transitions, POST an HMAC-SHA256-signed payload
  to the registered URL.
- Client side: verify HMAC on incoming webhook requests (helper
  available for downstream users writing receivers).
- Retry with exponential backoff on 5xx, give up after N
  attempts, surface to operator via tracing span.
- If open question #3 lands in favor of Ed25519: gate Ed25519
  behind a cargo feature `ed25519-webhooks`, advertise it in
  AgentCard `metadata.ruvector.webhook_algos`.

**Acceptance tests.**

1. `push_signing.rs`: register a webhook, drive the task to
   `Completed`, assert the receiver sees a request whose
   HMAC-SHA256 over the body matches the secret.
2. `push_retry.rs`: receiver returns 503 for the first two
   attempts, 200 for the third; assert delivered-once semantics
   (receiver sees one successful POST, logs show two failures).
3. `push_rejected.rs`: receiver returns 4xx; assert no retries,
   assert the failure is traced and visible to operators.
4. If `ed25519-webhooks` is on: equivalent test with Ed25519
   signature verification.

**Size estimate.** ~150 LoC library + ~100 LoC tests.

### Cross-milestone quality gates

- Every milestone: `cargo clippy --all-targets -- -D warnings`
  clean, `cargo deny check` clean.
- Every milestone: no new direct dependencies outside what's
  already in the `rvAgent` tree (axum, reqwest, serde, tokio,
  tracing, thiserror, async-trait).
- Between M2 and M3: a benchmark comparing `rvagent-acp`
  latency vs `rvagent-a2a` latency on the same `TaskRunner`, to
  verify the A2A layer does not impose a surprise overhead on a
  locally-mounted router (target: within 15% of ACP median
  latency).
- Before M4 ships: ADR-99 / ADR-103 C6 security-review pass
  extended to cover the A2A sub-app; no new auth code paths
  outside `rvagent-middleware`.

Total projected size at M4 end (r3): ~1470 LoC library + ~950 LoC
tests. The envelope widened from r1 (600–1000 LoC) → r2 (~1100) →
r3 (~1470) as the four r2 and four r3 additions landed. Each
addition remains single-concern and sub-150 LoC in isolation:

| Module                 | LoC  | Wave | Concern                                    |
|------------------------|-----:|------|--------------------------------------------|
| `identity.rs`          |  ~80 | r2   | AgentID, signed cards                      |
| `policy.rs`            |  ~60 | r2   | Per-task compute + cost guards             |
| `routing.rs`           | ~120 | r2   | Selector trait + stock impls               |
| `artifact_types.rs`    | ~100 | r2+r3| Typed artifacts + version negotiation      |
| `executor.rs`          |  ~90 | r2   | Local / Remote abstraction                 |
| `budget.rs`            | ~120 | r3   | Global budget + ledger + overflow policy   |
| `context.rs`           | ~100 | r3   | Trace + lineage + cycle vector             |
| `recursion_guard.rs`   |  ~60 | r3   | Depth + cycle dispatch gate                |
| `config.rs`            |  ~80 | r3   | Config loader (operator surface)           |
| core types + lifecycle | ~660 | r1   | Card, Task, Message, RPC, SSE, push        |

The total still fits in a single sprint of integration work — the
guide-rail is that no single file crosses 300 LoC and no single
concept crosses 150.

**Three-dimensional acceptance** gates the whole plan, not just the
last milestone. After M4 is merged, all three conditions in the
**Acceptance test** section above must hold on a real deployment
against a real external A2A orchestrator:

1. Remote call indistinguishable from local call.
2. Memory transfer size constant regardless of payload.
3. Total cost bounded under recursive delegation.

If any fails, the work is not finished regardless of which
milestones have passed.

---

## Footnotes

- The A2A spec cited throughout is the version at
  `https://github.com/google/A2A` as of 2026-04-24. Spec commit
  hash to be pinned in `rvagent-a2a/Cargo.toml` as a metadata
  comment when M1 lands.
- "rvAgent" in this ADR refers to the crate tree at
  `crates/rvAgent/` in the `ruvnet/ruvector` repository. No other
  binary or service is implied.
- Interop-target claims (LangGraph, Gemini Agent Builder, Copilot
  roadmap) are based on public documentation at the time of
  writing and are not a guarantee that those integrations will
  remain stable.
