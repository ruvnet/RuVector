//! Push webhooks must not follow redirects.
//!
//! `tasks/pushNotification/set` validates the registered URL (https, or
//! plain http only to loopback). A redirect returned by the receiver is
//! never re-validated, so following it would let a registered https URL
//! steer the server's POST to any host or scheme, including plaintext
//! internal services. The dispatcher must stop at the first hop.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::{
    extract::State,
    http::{header, StatusCode},
    response::IntoResponse,
    routing::post,
    Router,
};

use rvagent_a2a::budget::{BudgetLedger, GlobalBudget};
use rvagent_a2a::context::TaskContext;
use rvagent_a2a::executor::{Executor, InMemoryRunner};
use rvagent_a2a::identity::{agent_id_from_pubkey, AgentID};
use rvagent_a2a::server::{A2aServer, A2aServerConfig};
use rvagent_a2a::types::{
    AgentCapabilities, AgentCard, AgentProvider, AgentSkill, AuthScheme, Message, Part, Role,
    TaskSpec,
};

use ed25519_dalek::SigningKey;
use rand_core::OsRng;
use serde_json::json;

#[derive(Clone)]
struct ReceiverState {
    hook_hits: Arc<AtomicUsize>,
    target_hits: Arc<AtomicUsize>,
    target_url: String,
}

async fn hook(State(s): State<ReceiverState>, _body: axum::body::Bytes) -> impl IntoResponse {
    s.hook_hits.fetch_add(1, Ordering::SeqCst);
    (
        StatusCode::TEMPORARY_REDIRECT,
        [(header::LOCATION, s.target_url.clone())],
    )
}

async fn target(State(s): State<ReceiverState>, _body: axum::body::Bytes) -> impl IntoResponse {
    s.target_hits.fetch_add(1, Ordering::SeqCst);
    StatusCode::OK
}

fn make_card(base_url: &str) -> AgentCard {
    AgentCard {
        name: "push-redirect-test".into(),
        description: "push_redirect.rs fixture".into(),
        url: base_url.into(),
        provider: AgentProvider {
            organization: "ruvector".into(),
            url: None,
        },
        version: "0.1.0".into(),
        capabilities: AgentCapabilities {
            streaming: false,
            push_notifications: true,
        },
        skills: vec![AgentSkill {
            id: "rag.query".into(),
            name: "RAG".into(),
            description: "x".into(),
            tags: vec![],
            input_modes: vec![],
            output_modes: vec![],
        }],
        authentication: AuthScheme {
            schemes: vec!["bearer".into()],
        },
        metadata: json!({ "ruvector": {} }),
    }
}

fn spec(id: &str) -> TaskSpec {
    TaskSpec {
        id: id.into(),
        skill: "rag.query".into(),
        message: Message {
            role: Role::User,
            parts: vec![Part::Text {
                text: "ping".into(),
            }],
            metadata: serde_json::Value::Null,
        },
        policy: None,
        context: TaskContext::new_root(AgentID(
            agent_id_from_pubkey(&SigningKey::generate(&mut OsRng).verifying_key()).0,
        )),
        metadata: serde_json::Value::Null,
    }
}

#[tokio::test]
async fn push_webhook_does_not_follow_redirects() {
    let receiver_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let receiver_addr = receiver_listener.local_addr().unwrap();
    let receiver_url = format!("http://{}/hook", receiver_addr);
    let hook_hits = Arc::new(AtomicUsize::new(0));
    let target_hits = Arc::new(AtomicUsize::new(0));
    let app = Router::new()
        .route("/hook", post(hook))
        .route("/internal", post(target).get(target))
        .with_state(ReceiverState {
            hook_hits: hook_hits.clone(),
            target_hits: target_hits.clone(),
            target_url: format!("http://{}/internal", receiver_addr),
        });
    tokio::spawn(async move {
        axum::serve(receiver_listener, app).await.unwrap();
    });

    let server_listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let server_addr = server_listener.local_addr().unwrap();
    let server_url = format!("http://{}", server_addr);
    let card = make_card(&server_url);
    let card_bytes = serde_json::to_vec(&card).unwrap();
    let executor = Arc::new(Executor::Local(Arc::new(InMemoryRunner::new())));
    let budget = Arc::new(BudgetLedger::new(GlobalBudget::default()));
    let server = A2aServer::new(
        card,
        card_bytes,
        executor,
        budget,
        A2aServerConfig::default(),
    );
    let router = server.router();
    tokio::spawn(async move {
        axum::serve(server_listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;

    let http = reqwest::Client::new();
    let set_req = json!({
        "jsonrpc": "2.0",
        "method": "tasks/pushNotification/set",
        "params": {
            "id": "t-redir-1",
            "pushNotificationConfig": { "url": receiver_url, "token": "redir-token" },
        },
        "id": 1,
    });
    let set_resp = http.post(&server_url).json(&set_req).send().await.unwrap();
    assert!(set_resp.status().is_success());

    let send_req = json!({
        "jsonrpc": "2.0",
        "method": "tasks/send",
        "params": spec("t-redir-1"),
        "id": 2,
    });
    let send_resp = http.post(&server_url).json(&send_req).send().await.unwrap();
    assert!(send_resp.status().is_success());

    // Long enough for every retry (100ms + 400ms backoff) to land.
    tokio::time::sleep(Duration::from_millis(1500)).await;

    assert!(
        hook_hits.load(Ordering::SeqCst) >= 1,
        "registered webhook was never called"
    );
    assert_eq!(
        target_hits.load(Ordering::SeqCst),
        0,
        "push dispatcher followed a redirect to an unvalidated URL"
    );
}
