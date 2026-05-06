//! Single-node gRPC Vitals client with connection reuse.

use std::time::Duration;

use tonic::transport::Channel;

use crate::error::Result;
use crate::proto::{
    vitals_client::VitalsClient as TonicClient, GetLatestRequest, HealthRequest, HealthResponse,
    StatsRequest, StatsResponse, StreamVitalsRequest, VitalReading,
};

/// Thin wrapper around the tonic-generated `VitalsClient` that adds a
/// per-call deadline and hides the raw proto types from callers.
#[derive(Clone)]
pub struct VitalsClient {
    inner: TonicClient<Channel>,
    node_name: String,
}

impl VitalsClient {
    /// Connect to a single `ruview-vitals-worker` node.
    ///
    /// `endpoint` is an HTTP/2 URI, e.g. `http://100.80.54.16:50055`.
    /// Connection is lazy — the first RPC triggers the actual TCP handshake.
    pub async fn connect(node_name: impl Into<String>, endpoint: impl AsRef<str>) -> Result<Self> {
        let channel = Channel::from_shared(endpoint.as_ref().to_owned())
            .map_err(|_| tonic::Status::invalid_argument("invalid endpoint URI"))?
            .timeout(Duration::from_secs(5))
            .connect_lazy();
        Ok(Self {
            inner: TonicClient::new(channel),
            node_name: node_name.into(),
        })
    }

    pub fn node_name(&self) -> &str {
        &self.node_name
    }

    /// Cheap liveness probe.
    pub async fn health(&mut self) -> Result<HealthResponse> {
        Ok(self
            .inner
            .health(HealthRequest {})
            .await?
            .into_inner())
    }

    /// Service counters.
    pub async fn stats(&mut self) -> Result<StatsResponse> {
        Ok(self
            .inner
            .get_stats(StatsRequest {})
            .await?
            .into_inner())
    }

    /// Latest cached reading (any node if `node_id == 0`).
    pub async fn latest(&mut self, node_id: u32) -> Result<VitalReading> {
        Ok(self
            .inner
            .get_latest(GetLatestRequest { node_id })
            .await?
            .into_inner())
    }

    /// Streaming readings — caller drives the returned stream.
    pub async fn stream(
        &mut self,
        node_id_filter: u32,
    ) -> Result<tonic::Streaming<VitalReading>> {
        Ok(self
            .inner
            .stream_vitals(StreamVitalsRequest { node_id_filter })
            .await?
            .into_inner())
    }
}
