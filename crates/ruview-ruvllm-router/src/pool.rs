use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tonic::transport::Channel;

use crate::llm_proto::llm_service_client::LlmServiceClient;
use crate::llm_proto::{HealthRequest, HealthResponse};

pub struct Backend {
    pub addr: String,
    pub model: String,
    pub active: AtomicU32,
    pub healthy: AtomicBool,
    channel: Channel,
    pub last_health: RwLock<Option<HealthResponse>>,
    pub last_check: RwLock<Instant>,
}

impl Backend {
    pub async fn connect(addr: &str, model: String) -> Result<Arc<Self>, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = format!("http://{addr}");
        let channel = Channel::from_shared(endpoint)?.connect_lazy();
        Ok(Arc::new(Self {
            addr: addr.to_string(),
            model,
            active: AtomicU32::new(0),
            healthy: AtomicBool::new(true),
            channel,
            last_health: RwLock::new(None),
            last_check: RwLock::new(Instant::now()),
        }))
    }

    pub fn client(&self) -> LlmServiceClient<Channel> {
        LlmServiceClient::new(self.channel.clone())
    }

    pub async fn check_health(&self) {
        let mut client = self.client();
        match client.health(HealthRequest {}).await {
            Ok(resp) => {
                let r = resp.into_inner();
                let ok = r.hailo_ok;
                *self.last_health.write().await = Some(r);
                *self.last_check.write().await = Instant::now();
                self.healthy.store(ok, Ordering::Relaxed);
            }
            Err(e) => {
                tracing::warn!(backend = %self.addr, error = %e, "health check failed");
                self.healthy.store(false, Ordering::Relaxed);
                *self.last_check.write().await = Instant::now();
            }
        }
    }
}

pub struct Pool;

impl Pool {
    pub async fn new(specs: &[(String, String)]) -> Vec<Arc<Backend>> {
        let mut backends = Vec::with_capacity(specs.len());
        for (addr, model) in specs {
            match Backend::connect(addr, model.clone()).await {
                Ok(b) => {
                    tracing::info!(addr = %addr, model = %model, "backend registered");
                    backends.push(b);
                }
                Err(e) => tracing::error!(addr = %addr, error = %e, "backend connect failed"),
            }
        }
        backends
    }

    /// Pick the healthy backend with the fewest active requests.
    pub fn least_busy(backends: &[Arc<Backend>]) -> Option<Arc<Backend>> {
        backends
            .iter()
            .filter(|b| b.healthy.load(Ordering::Relaxed))
            .min_by_key(|b| b.active.load(Ordering::Relaxed))
            .cloned()
    }

    /// Run a health check sweep every `interval`. Marks stale backends unhealthy.
    pub async fn health_loop(backends: Arc<Vec<Arc<Backend>>>, interval: Duration) {
        loop {
            tokio::time::sleep(interval).await;
            for b in backends.iter() {
                b.check_health().await;
                tracing::debug!(
                    backend  = %b.addr,
                    healthy  = b.healthy.load(Ordering::Relaxed),
                    active   = b.active.load(Ordering::Relaxed),
                    "health tick"
                );
            }
        }
    }
}

/// RAII guard that decrements `active` when dropped (even on cancel/panic).
pub struct ActiveGuard(Arc<Backend>);

impl ActiveGuard {
    pub fn new(b: &Arc<Backend>) -> Self {
        b.active.fetch_add(1, Ordering::Relaxed);
        Self(Arc::clone(b))
    }
}

impl Drop for ActiveGuard {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::Relaxed);
    }
}
