//! A2A server — axum sub-app exposing the JSON-RPC surface + discovery.
//!
//! Submodules:
//! - [`json_rpc`] — the JSON-RPC 2.0 framing types and the method dispatch
//!   entry point (`POST /`).
//! - [`sse`] — `tasks/sendSubscribe` and `tasks/resubscribe` streaming.
//! - [`push`] — push-notification webhook registry (M4).
//!
//! The only public handle is [`A2aServer`], which builds an
//! [`axum::Router`]. Users mount that router into their own binary, usually
//! alongside the `rvagent-acp` router on a different path prefix.

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{broadcast, RwLock};

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::budget::BudgetLedger;
use crate::config::RoutingConfig;
use crate::executor::Executor;
use crate::routing::Router as PeerRouter;
use crate::types::{AgentCard, Task, TaskArtifactUpdateEvent, TaskStatusUpdateEvent};

pub mod json_rpc;
pub mod push;
pub mod sse;

pub use json_rpc::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};

// ---------------------------------------------------------------------------
// Shared server state.
// ---------------------------------------------------------------------------

/// In-memory task store for M1. Production deployments will swap this out
/// for a persistent store (redb / sled / postgres) via a trait; for now the
/// API is simple enough that we keep the type concrete and inline.
pub type TaskStore = Arc<RwLock<HashMap<String, Task>>>;

/// Per-task event channel. Capacity is bounded (256 events); on overflow
/// we drop the oldest slot and emit a tracing WARN — documented policy
/// per ADR-159 M2 acceptance test `sse_backpressure.rs`.
pub type TaskEventTx = broadcast::Sender<TaskEvent>;

/// Capacity of the per-task event channel + retained replay history.
/// Kept in sync so `tasks/resubscribe` can replay at most what the
/// live channel would have retained had the client stayed connected.
pub(crate) const EVENT_CHANNEL_CAPACITY: usize = 256;

/// One of the two SSE payload shapes. The server broadcasts these and the
/// SSE handler converts each into a `text/event-stream` frame.
#[derive(Clone, Debug)]
pub enum TaskEvent {
    Status(TaskStatusUpdateEvent),
    Artifact(TaskArtifactUpdateEvent),
}

/// Shared application state behind the router.
#[derive(Clone)]
pub struct A2aState {
    /// Pre-signed canonical JSON bytes of the card. Produced once at
    /// startup (see `identity::sign_card`) and served directly so we
    /// don't re-canonicalize per request.
    pub card: Arc<AgentCard>,

    /// Signed card bytes — served verbatim from
    /// `GET /.well-known/agent.json`. Filled in by the server builder.
    pub card_bytes: Arc<Vec<u8>>,

    /// Task store — simple `HashMap` for M1. The same `RwLock` is locked
    /// for reads + writes; contention is expected to be negligible at M1
    /// task rates.
    pub tasks: TaskStore,

    /// Per-task event channels. Entries live as long as any subscriber,
    /// then are garbage-collected on task-end.
    pub streams: Arc<RwLock<HashMap<String, TaskEventTx>>>,

    /// Pluggable executor — `Local(Box<dyn TaskRunner>)` in M1,
    /// `Remote(Peer)` or `ChainedSelector` in M3.
    pub executor: Arc<Executor>,

    /// Global budget ledger (r3). Enforced at the dispatch queue.
    pub budget: Arc<BudgetLedger>,

    /// Push-notification webhook registry (M4). Shared with the
    /// `tasks/pushNotification/set`/`get` handlers + the state-change
    /// dispatcher.
    pub push_registry: Arc<push::PushNotificationRegistry>,

    /// HTTP client reused for outbound webhook POSTs. Sharing a single
    /// `reqwest::Client` keeps the connection pool warm across
    /// dispatches.
    pub push_client: reqwest::Client,

    /// Optional Ed25519 signing key for outbound webhook signatures
    /// (feature `ed25519-webhooks`). Generated at server startup and the
    /// public half is advertised in the AgentCard
    /// `metadata.ruvector.webhook_ed25519_pubkey`. When `None` (feature
    /// off, or feature on but disabled), the Ed25519 header is not
    /// emitted and the HMAC path remains the sole signature channel.
    #[cfg(feature = "ed25519-webhooks")]
    pub ed25519_signing_key: Arc<Option<ed25519_dalek::SigningKey>>,

    /// Recent status events retained per task so `tasks/resubscribe`
    /// can replay the suffix a disconnected client missed. Capacity
    /// matches the live broadcast channel (256); oldest events evict
    /// first. Keyed by task id. See `sse::resubscribe`.
    pub status_history:
        Arc<RwLock<HashMap<String, std::collections::VecDeque<TaskStatusUpdateEvent>>>>,

    /// Optional peer router (ADR-159 M3). When `Some`, `tasks/send`
    /// consults the router after policy admission and forwards the task
    /// to the selected peer via `Executor::Remote`. `None` → local-only
    /// dispatch. Populated from `A2aServerConfig.routing` at startup;
    /// peers are seeded asynchronously in the background.
    pub router: Option<Arc<PeerRouter>>,
}

// ---------------------------------------------------------------------------
// Server builder.
// ---------------------------------------------------------------------------

/// Configuration for the axum sub-app. Intentionally small — all the
/// policy / budget / recursion knobs live on the types they govern.
#[derive(Debug, Clone)]
pub struct A2aServerConfig {
    /// Maximum inbound request body size, bytes. A2A can carry file
    /// parts so this defaults higher than `rvagent-acp`.
    pub max_body_size: usize,
    /// Permissive CORS — mount in front-facing deployments at your
    /// discretion.
    pub permissive_cors: bool,
    /// Optional routing block (ADR-159 M3). When `Some` and
    /// `peers` is non-empty, the server spawns a background task at
    /// `new()` to discover each peer and seed the `PeerRegistry`, then
    /// installs a `Router` on `A2aState.router` so `tasks/send` can
    /// forward. `None` keeps the M1 local-only behavior.
    pub routing: Option<RoutingConfig>,
}

impl Default for A2aServerConfig {
    fn default() -> Self {
        Self {
            max_body_size: 8 * 1024 * 1024, // 8 MB — see ADR-159 open Q#5
            permissive_cors: false,
            routing: None,
        }
    }
}

/// The A2A server. Construct via `A2aServer::new`, then call `.router()`
/// to mount into a parent axum `Router`.
pub struct A2aServer {
    state: A2aState,
    config: A2aServerConfig,
}

impl A2aServer {
    /// Construct a new server.
    ///
    /// `card_bytes` must be the canonical-JSON serialization of `card`
    /// with a valid `metadata.ruvector.identity` signature — produced by
    /// `identity::sign_card` at binary startup.
    pub fn new(
        card: AgentCard,
        card_bytes: Vec<u8>,
        executor: Arc<Executor>,
        budget: Arc<BudgetLedger>,
        config: A2aServerConfig,
    ) -> Self {
        // When the `ed25519-webhooks` feature is on, mint a fresh
        // Ed25519 keypair for webhook signing and advertise the public
        // half in the AgentCard under `metadata.ruvector.webhook_algos`
        // + `metadata.ruvector.webhook_ed25519_pubkey`. The HMAC header
        // stays the default; Ed25519 is additive, not replacement.
        //
        // Preserve the caller's `card_bytes` unchanged when the card
        // carries a `metadata.ruvector.signature` — mutating the JSON
        // would invalidate the identity signature. In that case the
        // webhook pubkey still lives on the in-memory `card` so
        // server-internal paths can reach it, but the public
        // discovery document keeps its original signed form. Callers
        // who need the advertisement to land in the signed card should
        // build the card with the extra fields and re-sign before
        // constructing `A2aServer`.
        #[cfg(feature = "ed25519-webhooks")]
        let (ed_key, card, card_bytes) = {
            use base64::Engine;
            use rand_core::OsRng;
            let sk = ed25519_dalek::SigningKey::generate(&mut OsRng);
            let pubkey_b64 =
                base64::engine::general_purpose::STANDARD.encode(sk.verifying_key().to_bytes());

            // Does the card already carry a signature? If yes, leave
            // both the struct and the bytes alone.
            let already_signed = card
                .metadata
                .get("ruvector")
                .and_then(|v| v.get("signature"))
                .is_some();

            if already_signed {
                tracing::warn!(
                    "ed25519-webhooks: AgentCard carries an identity signature; \
                     leaving card_bytes unchanged. Re-sign the card after adding \
                     webhook_ed25519_pubkey to advertise it via discovery."
                );
                (Arc::new(Some(sk)), card, card_bytes)
            } else {
                let mut patched_card = card.clone();
                let meta = patched_card.metadata.as_object_mut().and_then(|m| {
                    m.entry("ruvector").or_insert_with(|| serde_json::json!({}));
                    m.get_mut("ruvector").and_then(|v| v.as_object_mut())
                });
                if let Some(ruvector) = meta {
                    ruvector.insert(
                        "webhook_algos".into(),
                        serde_json::json!(["hmac-sha256", "ed25519"]),
                    );
                    ruvector.insert(
                        "webhook_ed25519_pubkey".into(),
                        serde_json::Value::String(pubkey_b64.clone()),
                    );
                } else {
                    patched_card.metadata = serde_json::json!({
                        "ruvector": {
                            "webhook_algos": ["hmac-sha256", "ed25519"],
                            "webhook_ed25519_pubkey": pubkey_b64,
                        }
                    });
                }
                let new_bytes = serde_json::to_vec(&patched_card).unwrap_or(card_bytes);
                (Arc::new(Some(sk)), patched_card, new_bytes)
            }
        };

        // Build an optional Router from `config.routing` and spawn a
        // background discovery task for each configured peer URL.
        // Seeding is async so server startup is never gated on a peer
        // that's still coming up — the registry starts empty and fills
        // in as discoveries complete. A failed discovery logs + retries
        // once; subsequent failures surface as "no healthy peers" to
        // the selector and the router falls through to local.
        let router = build_router(config.routing.as_ref());

        let state = A2aState {
            card: Arc::new(card),
            card_bytes: Arc::new(card_bytes),
            tasks: Arc::new(RwLock::new(HashMap::new())),
            streams: Arc::new(RwLock::new(HashMap::new())),
            executor,
            budget,
            push_registry: push::PushNotificationRegistry::new(),
            // Redirects are never followed: `push::validate_url` vets only
            // the registered URL, so a redirect could steer the POST to an
            // unvalidated host or scheme.
            push_client: reqwest::Client::builder()
                .timeout(std::time::Duration::from_secs(10))
                .redirect(reqwest::redirect::Policy::none())
                .build()
                .unwrap_or_default(),
            #[cfg(feature = "ed25519-webhooks")]
            ed25519_signing_key: ed_key,
            status_history: Arc::new(RwLock::new(HashMap::new())),
            router,
        };
        Self { state, config }
    }

    /// Attach a pre-built [`PeerRouter`] to this server's state. Useful
    /// for tests that want to supply a fully-seeded `PeerRegistry`
    /// instead of waiting for async discovery. Returns `self` for
    /// chaining.
    pub fn with_router(mut self, router: Arc<PeerRouter>) -> Self {
        self.state.router = Some(router);
        self
    }

    /// Access the assembled `A2aState`. Exposed for tests and for callers
    /// that want to mount additional routes on the same store.
    pub fn state(&self) -> &A2aState {
        &self.state
    }

    /// Build the axum `Router`. Mount at `/` of your choice — usually
    /// this is the whole vhost since `/.well-known/agent.json` is
    /// path-hardcoded by the spec.
    #[tracing::instrument(skip(self), level = "debug")]
    pub fn router(&self) -> Router {
        let mut router = Router::new()
            .route("/.well-known/agent.json", get(handle_card))
            .route("/", post(json_rpc::dispatch))
            // SSE streaming endpoints per A2A spec (ADR-159 M2). These are
            // the GET variants of `tasks/sendSubscribe` and
            // `tasks/resubscribe` — the JSON-RPC POST surface retains the
            // spec-conformant stubs in `json_rpc::dispatch`.
            .route("/tasks/sendSubscribe", get(sse::send_subscribe))
            .route("/tasks/resubscribe", get(sse::resubscribe))
            .with_state(self.state.clone())
            .layer(RequestBodyLimitLayer::new(self.config.max_body_size))
            .layer(TraceLayer::new_for_http());

        if self.config.permissive_cors {
            router = router.layer(CorsLayer::permissive());
        }

        router
    }
}

// ---------------------------------------------------------------------------
// Router bootstrap — selector instantiation + async peer discovery.
// ---------------------------------------------------------------------------

/// Instantiate a named selector. Returns `None` for unknown names so the
/// caller can fall back / log — we don't want a typo in the TOML to
/// panic server startup.
fn selector_by_name(
    name: &str,
    latency_budget_ms: u64,
) -> Option<Arc<dyn crate::routing::PeerSelector>> {
    match name {
        "cheapest_under_latency" => Some(Arc::new(crate::routing::CheapestUnderLatency {
            budget_ms: latency_budget_ms,
        })),
        "lowest_latency" => Some(Arc::new(crate::routing::LowestLatency)),
        "round_robin" => Some(Arc::new(crate::routing::RoundRobin::new())),
        _ => None,
    }
}

/// Build a [`Router`] from a [`RoutingConfig`]. Spawns a background
/// discovery task per peer entry — each `fetch_card` call populates the
/// registry on success and logs + retries (once) on failure.
///
/// Returns `None` when `routing` is `None` or the selector can't be
/// resolved — the server then falls through to local-only dispatch.
fn build_router(routing: Option<&RoutingConfig>) -> Option<Arc<PeerRouter>> {
    let routing = routing?;

    // Resolve the default selector first; an unknown name disables
    // routing entirely so a typo surfaces as a single WARN at startup.
    if boxed_selector_by_name(&routing.default_selector, routing.latency_budget_ms).is_none() {
        tracing::warn!(
            target = "a2a::routing",
            selector = %routing.default_selector,
            "unknown default_selector; routing disabled",
        );
        return None;
    }

    // Build the selector: default alone, or default-then-fallback via
    // `ChainedSelector` when a fallback is named AND resolvable. An
    // unknown fallback name logs + proceeds with the default only.
    let selector: Arc<dyn crate::routing::PeerSelector> = match routing.fallback.as_deref() {
        Some(fb) => match boxed_selector_by_name(fb, routing.latency_budget_ms) {
            Some(fb_boxed) => {
                let default_boxed =
                    boxed_selector_by_name(&routing.default_selector, routing.latency_budget_ms)
                        .expect("default selector already validated above");
                Arc::new(crate::routing::ChainedSelector::new(vec![
                    default_boxed,
                    fb_boxed,
                ]))
            }
            None => {
                tracing::warn!(
                    target = "a2a::routing",
                    fallback = %fb,
                    "unknown fallback selector; using default only",
                );
                // Arc-ify the default on its own.
                selector_by_name(&routing.default_selector, routing.latency_budget_ms)
                    .expect("default selector already validated above")
            }
        },
        None => selector_by_name(&routing.default_selector, routing.latency_budget_ms)
            .expect("default selector already validated above"),
    };

    let registry = Arc::new(crate::routing::PeerRegistry::new());
    let latency_budget_ms = Some(routing.latency_budget_ms);
    let router = Arc::new(PeerRouter::new(
        registry.clone(),
        selector,
        latency_budget_ms,
    ));

    // Fire off discovery tasks. Each peer is fetched in parallel; the
    // registry fills in as discoveries complete. Runs inside the tokio
    // runtime `A2aServer::new` is called from.
    for entry in &routing.peers {
        let url = entry.url.clone();
        let strict = entry.verify_card.unwrap_or(false);
        let registry = registry.clone();
        tokio::spawn(async move {
            discover_and_register(registry, url, strict).await;
        });
    }

    Some(router)
}

/// Same as [`selector_by_name`] but returns a `Box` for use inside a
/// [`crate::routing::ChainedSelector`].
fn boxed_selector_by_name(
    name: &str,
    latency_budget_ms: u64,
) -> Option<Box<dyn crate::routing::PeerSelector>> {
    match name {
        "cheapest_under_latency" => Some(Box::new(crate::routing::CheapestUnderLatency {
            budget_ms: latency_budget_ms,
        })),
        "lowest_latency" => Some(Box::new(crate::routing::LowestLatency)),
        "round_robin" => Some(Box::new(crate::routing::RoundRobin::new())),
        _ => None,
    }
}

/// Discover a single peer's AgentCard and insert it into `registry` as
/// a fresh `PeerSnapshot`. One retry on failure; after that the peer
/// simply never appears in the pool (selector returns None → local
/// fallback). Logs at INFO on success, WARN on miss.
async fn discover_and_register(
    registry: Arc<crate::routing::PeerRegistry>,
    base_url: String,
    strict_verify: bool,
) {
    let client = match crate::client::A2aClient::builder()
        .strict_verify(strict_verify)
        .build()
    {
        Ok(c) => Arc::new(c),
        Err(e) => {
            tracing::warn!(
                target = "a2a::routing",
                peer_url = %base_url,
                error = %e,
                "failed to build A2aClient for peer discovery; skipping",
            );
            return;
        }
    };

    let mut last_err: Option<crate::error::A2aError> = None;
    for attempt in 1..=2u32 {
        match client.fetch_card(&base_url).await {
            Ok(card) => {
                // Derive the AgentID from the card's signature if
                // present; otherwise synthesize a stable-ish id from
                // the URL so the peer is at least addressable.
                let id = card
                    .metadata
                    .pointer("/ruvector/signature/public_key")
                    .and_then(|v| v.as_str())
                    .and_then(|hex_pk| hex::decode(hex_pk).ok())
                    .and_then(|bytes| {
                        if bytes.len() == 32 {
                            let mut arr = [0u8; 32];
                            arr.copy_from_slice(&bytes);
                            ed25519_dalek::VerifyingKey::from_bytes(&arr)
                                .ok()
                                .map(|vk| crate::identity::agent_id_from_pubkey(&vk))
                        } else {
                            None
                        }
                    })
                    .unwrap_or_else(|| crate::identity::AgentID(format!("peer:{}", base_url)));

                // Patch the card's `url` field to the configured
                // `base_url` so downstream forwards always hit the
                // operator-supplied endpoint, not whatever the card
                // self-advertises (which may be a stale or internal
                // hostname). The JSON-RPC handler reads
                // `snapshot.card.url` when building the `Peer`.
                let mut card = card;
                card.url = base_url.clone();

                let snapshot = crate::routing::PeerSnapshot {
                    id: id.clone(),
                    card,
                    ewma_latency_ms: 0.0,
                    ewma_cost_usd: 0.0,
                    open_tasks: 0,
                    failure_rate: 0.0,
                };
                registry.add(snapshot);
                tracing::info!(
                    target = "a2a::routing",
                    peer_url = %base_url,
                    peer_id = %id,
                    "peer discovered and registered",
                );
                return;
            }
            Err(e) => {
                tracing::warn!(
                    target = "a2a::routing",
                    peer_url = %base_url,
                    attempt,
                    error = %e,
                    "peer discovery failed",
                );
                last_err = Some(e);
                // Brief backoff between the two attempts.
                tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            }
        }
    }
    if let Some(e) = last_err {
        tracing::warn!(
            target = "a2a::routing",
            peer_url = %base_url,
            error = %e,
            "peer discovery giving up after retries — peer will not appear in pool",
        );
    }
}

// ---------------------------------------------------------------------------
// `/.well-known/agent.json`.
// ---------------------------------------------------------------------------

/// Serve the pre-signed card bytes verbatim. No auth required — discovery
/// is public by A2A spec.
#[tracing::instrument(skip(state), level = "debug")]
async fn handle_card(State(state): State<A2aState>) -> impl IntoResponse {
    (
        StatusCode::OK,
        [("content-type", "application/json")],
        state.card_bytes.as_ref().clone(),
    )
}

// ---------------------------------------------------------------------------
// Helper: ensure or create a task's broadcast channel.
// ---------------------------------------------------------------------------

/// Get the existing broadcast sender for `task_id`, or create a bounded
/// (256 capacity) one if none yet exists. Bounded so a slow consumer
/// can't OOM the server — on overflow `broadcast` drops the oldest slot
/// and receivers get a `Lagged` notification, which we log as WARN.
pub(crate) async fn ensure_stream(state: &A2aState, task_id: &str) -> TaskEventTx {
    {
        let guard = state.streams.read().await;
        if let Some(tx) = guard.get(task_id) {
            return tx.clone();
        }
    }
    let (tx, _rx) = broadcast::channel::<TaskEvent>(EVENT_CHANNEL_CAPACITY);
    let mut guard = state.streams.write().await;
    guard
        .entry(task_id.to_string())
        .or_insert_with(|| tx.clone());
    tx
}

/// Record a `TaskStatusUpdateEvent` into the replay history for
/// `task_id`. Capped at `EVENT_CHANNEL_CAPACITY`; oldest events evict
/// first. Called from the JSON-RPC dispatch path on every status emit,
/// so `resubscribe` can replay whatever a disconnected client missed.
pub(crate) async fn record_status_event(state: &A2aState, ev: &TaskStatusUpdateEvent) {
    let mut guard = state.status_history.write().await;
    let entry = guard
        .entry(ev.id.clone())
        .or_insert_with(std::collections::VecDeque::new);
    entry.push_back(ev.clone());
    while entry.len() > EVENT_CHANNEL_CAPACITY {
        entry.pop_front();
    }
}

/// The task has no live broadcast channel, so [`emit_status_event`] had
/// nobody to deliver to. Not an error condition for the caller — the event
/// is still recorded into the replay history either way — which is why it
/// carries no payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("task has no live broadcast channel")]
pub struct NoLiveChannel;

/// Record + broadcast a status event atomically. Preferred over raw
/// `tx.send(TaskEvent::Status(..))` from custom `TaskRunner`s because
/// it keeps the `resubscribe` replay history in sync — a subscriber
/// that reconnects after missing events still sees them on replay.
///
/// Returns `Ok(receiver_count)` on successful broadcast.
/// [`NoLiveChannel`] means the task has no live channel (nothing was
/// broadcast, but the event was still recorded into the replay history).
pub async fn emit_status_event(
    state: &A2aState,
    ev: TaskStatusUpdateEvent,
) -> Result<usize, NoLiveChannel> {
    record_status_event(state, &ev).await;
    let tx = state.streams.read().await.get(&ev.id).cloned();
    match tx {
        Some(tx) => tx.send(TaskEvent::Status(ev)).map_err(|_| NoLiveChannel),
        None => Err(NoLiveChannel),
    }
}

// ---------------------------------------------------------------------------
// Tests.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_is_sensible() {
        let c = A2aServerConfig::default();
        assert_eq!(c.max_body_size, 8 * 1024 * 1024);
        assert!(!c.permissive_cors);
    }
}
