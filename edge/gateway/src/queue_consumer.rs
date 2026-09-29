//! Queue consumers of the gateway Worker (wrangler `[[queues.consumers]]`).
//! Message bodies are JSON text (see `m3_ports::WorkerQueues`).
//!
//! - `ruvector-edge-ingest` (DLQ `ruvector-edge-ingest-dlq`): one import
//!   job delivery per message (`ingest::process`). `Continue` re-enqueues
//!   the same `{tenant_key, job_id}` and acks; `Retry` leaves the message
//!   for redelivery after [`RETRY_DELAY_S`] (cursor already persisted; the
//!   delay × `max_retries` far exceeds the 120 s pending-`op_id` TTL, so a
//!   batch reserved by a killed invocation frees up before the retries run
//!   out); malformed bodies are acked. A job that finishes or fails ships
//!   its audit event.
//! - `ruvector-edge-audit` (DLQ `ruvector-edge-audit-dlq`): the whole batch
//!   is chained per tenant in the ledger and written under `audit/`
//!   (`audit::ship`, keyed by queue message id, so redelivery never
//!   duplicates a line); on any failure the batch is retried as a unit
//!   after [`RETRY_DELAY_S`].

use crate::audit::{self, AuditEvent};
use crate::durable::DoBackend;
use crate::ingest::{self, Delivery, UpsertSink, BATCHES_PER_DELIVERY};
use crate::jobs::IngestMsg;
use crate::m3_ports::{
    QueueName, Queues, WorkerQueues, AUDIT_QUEUE, DATA_BINDING, INGEST_QUEUE, R2,
};
use crate::platform::WorkerClock;
use ruvector_edge_auth::Clock;
use serde_json::Value as Json;
use worker::{event, Context, Env, MessageBatch, MessageExt, QueueRetryOptionsBuilder, Result};

/// Delay before a retried message is redelivered (matches wrangler.toml
/// `retry_delay`).
pub const RETRY_DELAY_S: u32 = 30;

/// Queue consumer entry point.
#[event(queue)]
pub async fn queue(batch: MessageBatch<String>, env: Env, _ctx: Context) -> Result<()> {
    console_error_panic_hook::set_once();
    let blob = R2(env.bucket(DATA_BINDING)?);
    let b = DoBackend { env: &env };
    let later = QueueRetryOptionsBuilder::new()
        .with_delay_seconds(RETRY_DELAY_S)
        .build();
    match batch.queue().as_str() {
        INGEST_QUEUE => {
            let queues = WorkerQueues(&env);
            for msg in batch.messages()? {
                let Ok(m) = serde_json::from_str::<IngestMsg>(msg.body()) else {
                    msg.ack();
                    continue;
                };
                let now_ms = WorkerClock.now_unix() * 1000;
                let sink = UpsertSink {
                    b: &b,
                    now: now_ms / 1000,
                };
                let r = ingest::process(&b, &blob, &sink, &m, now_ms, BATCHES_PER_DELIVERY).await;
                let outcome = match r {
                    Ok(report) => {
                        if let Some(ev) = &report.audit {
                            audit::emit(&queues, ev).await;
                        }
                        Some(report.outcome)
                    }
                    Err(_) => None,
                };
                match outcome {
                    Some(Delivery::Done) => msg.ack(),
                    Some(Delivery::Continue) => {
                        let again = serde_json::to_value(&m).unwrap_or(Json::Null);
                        match queues.send(QueueName::Ingest, again).await {
                            Ok(()) => msg.ack(),
                            Err(_) => msg.retry_with_options(&later),
                        }
                    }
                    Some(Delivery::Retry) | None => msg.retry_with_options(&later),
                }
            }
        }
        AUDIT_QUEUE => {
            let mut events = Vec::new();
            for msg in batch.messages()? {
                if let Ok(ev) = serde_json::from_str::<AuditEvent>(msg.body()) {
                    events.push((msg.id(), ev));
                }
            }
            let now_ms = WorkerClock.now_unix() * 1000;
            if audit::ship(&b, &blob, events, now_ms).await.is_ok() {
                batch.ack_all();
            } else {
                batch.retry_all_with_options(&later);
            }
        }
        _ => batch.ack_all(),
    }
    Ok(())
}
