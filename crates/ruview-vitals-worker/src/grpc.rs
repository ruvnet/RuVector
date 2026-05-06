//! gRPC `Vitals` service implementation. Bind addr + port live in
//! [`Config::grpc_listen`] (`:50054` by default).

use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;

use futures_core::Stream;
use tokio::sync::broadcast::error::RecvError;
use tonic::{Request, Response, Status};

use crate::proto;
use crate::state::WorkerState;
use crate::types::{VitalEstimate, VitalReading};

/// Convert a domain [`VitalEstimate`] into its proto wire shape.
#[must_use]
pub fn to_proto_estimate(e: &VitalEstimate) -> proto::Estimate {
    proto::Estimate {
        value_bpm: e.value_bpm,
        confidence: e.confidence,
        status: e.status.as_proto(),
    }
}

/// Convert a domain [`VitalReading`] into its proto wire shape.
#[must_use]
pub fn to_proto_reading(r: &VitalReading) -> proto::VitalReading {
    proto::VitalReading {
        node_id: u32::from(r.node_id),
        timestamp_us: r.timestamp_us,
        breathing: Some(to_proto_estimate(&r.breathing)),
        heart_rate: Some(to_proto_estimate(&r.heart_rate)),
        snr_db: r.snr_db,
        subcarrier_count: r.subcarrier_count,
        window_frames: r.window_frames,
        status: r.status.as_proto(),
    }
}

#[derive(Debug, Clone)]
pub struct VitalsService {
    state: Arc<WorkerState>,
}

impl VitalsService {
    #[must_use]
    pub const fn new(state: Arc<WorkerState>) -> Self {
        Self { state }
    }
}

#[tonic::async_trait]
impl proto::vitals_server::Vitals for VitalsService {
    async fn health(
        &self,
        _req: Request<proto::HealthRequest>,
    ) -> std::result::Result<Response<proto::HealthResponse>, Status> {
        Ok(Response::new(proto::HealthResponse {
            version: crate::VERSION.to_string(),
            node_name: self.state.config.node_name.clone(),
            listen_port: u32::from(self.state.config.grpc_listen.port()),
            ready: true,
            uptime_seconds: self.state.uptime_seconds(),
        }))
    }

    async fn get_stats(
        &self,
        _req: Request<proto::StatsRequest>,
    ) -> std::result::Result<Response<proto::StatsResponse>, Status> {
        let s = self.state.stats.snapshot();
        Ok(Response::new(proto::StatsResponse {
            packets_received: s.packets_received,
            packets_dropped: s.packets_dropped,
            windows_processed: s.windows_processed,
            readings_emitted: s.readings_emitted,
            brain_posts_ok: s.brain_posts_ok,
            brain_posts_failed: s.brain_posts_failed,
            uptime_seconds: self.state.uptime_seconds(),
        }))
    }

    async fn get_latest(
        &self,
        req: Request<proto::GetLatestRequest>,
    ) -> std::result::Result<Response<proto::VitalReading>, Status> {
        let asked = req.into_inner().node_id;
        let g = self.state.latest.read().await;
        if asked == 0 {
            // Any node — pick the most recently-stamped entry.
            if let Some(r) = g.values().max_by_key(|r| r.timestamp_us) {
                return Ok(Response::new(to_proto_reading(r)));
            }
        } else if asked <= u32::from(u8::MAX) {
            let nid = asked as u8;
            if let Some(r) = g.get(&nid) {
                return Ok(Response::new(to_proto_reading(r)));
            }
        }
        Err(Status::not_found("no readings available for node"))
    }

    type StreamVitalsStream =
        Pin<Box<dyn Stream<Item = std::result::Result<proto::VitalReading, Status>> + Send + 'static>>;

    async fn stream_vitals(
        &self,
        req: Request<proto::StreamVitalsRequest>,
    ) -> std::result::Result<Response<Self::StreamVitalsStream>, Status> {
        let filter_raw = req.into_inner().node_id_filter;
        let filter = if filter_raw == 0 {
            None
        } else if filter_raw <= u32::from(u8::MAX) {
            Some(filter_raw as u8)
        } else {
            return Err(Status::invalid_argument("node_id_filter exceeds u8 range"));
        };
        let mut rx = self.state.subscribe();

        // Bump readings_emitted every time we forward a reading on
        // this stream; gives operators visibility into how lively the
        // gRPC fan-out is vs how many readings the pipeline produced.
        let stats = self.state.stats.clone();

        let stream = async_stream::stream! {
            loop {
                match rx.recv().await {
                    Ok(reading) => {
                        if let Some(want) = filter {
                            if reading.node_id != want { continue; }
                        }
                        stats.readings_emitted.fetch_add(1, Ordering::Relaxed);
                        yield Ok(to_proto_reading(&reading));
                    }
                    Err(RecvError::Lagged(n)) => {
                        tracing::warn!(skipped = n, "stream_vitals subscriber lagged");
                        continue;
                    }
                    Err(RecvError::Closed) => break,
                }
            }
        };

        Ok(Response::new(Box::pin(stream)))
    }
}

/// Spin up a tonic server on `state.config.grpc_listen` and serve
/// the `Vitals` service. Blocks until the server exits (graceful
/// shutdown is left as future work — the worker is process-managed
/// by systemd which sends SIGTERM).
pub async fn serve(state: Arc<WorkerState>) -> crate::Result<()> {
    let svc = VitalsService::new(state.clone());
    let addr = state.config.grpc_listen;
    tracing::info!(%addr, "gRPC Vitals service listening");
    tonic::transport::Server::builder()
        .add_service(proto::vitals_server::VitalsServer::new(svc))
        .serve(addr)
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::unavailable_reading;
    use crate::types::{VitalEstimate, VitalStatus};

    #[test]
    fn estimate_roundtrip_preserves_status() {
        let e = VitalEstimate {
            value_bpm: 14.5,
            confidence: 0.81,
            status: VitalStatus::Degraded,
        };
        let p = to_proto_estimate(&e);
        assert!((p.value_bpm - 14.5).abs() < 1e-9);
        assert!((p.confidence - 0.81).abs() < 1e-9);
        assert_eq!(p.status, VitalStatus::Degraded.as_proto());
    }

    #[test]
    fn reading_roundtrip_node_id_widens() {
        let r = unavailable_reading(255, 99);
        let p = to_proto_reading(&r);
        assert_eq!(p.node_id, 255);
        assert_eq!(p.timestamp_us, 99);
        assert!(p.breathing.is_some());
        assert!(p.heart_rate.is_some());
    }
}
