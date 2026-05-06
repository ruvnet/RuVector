//! Shared worker state — counters, latest-per-node cache, and a
//! broadcast channel that fans readings out to gRPC `StreamVitals`
//! subscribers and to the brain POST loop.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::SystemTime;

use tokio::sync::{broadcast, RwLock};

use crate::config::Config;
use crate::types::{NodeId, VitalReading};

/// Capacity of the broadcast channel that fans readings out to gRPC
/// streamers + the brain loop. Sized for ~10 s of buffer at the
/// worker's natural ~0.6 Hz reading cadence (≈ 30 fps × 0.6 s window).
pub const READING_BROADCAST_CAPACITY: usize = 256;

/// Atomic counters scraped by `GetStats` and the periodic heartbeat.
#[derive(Debug, Default)]
pub struct WorkerStats {
    pub packets_received: AtomicU64,
    pub packets_dropped: AtomicU64,
    pub packets_relayed: AtomicU64,
    pub windows_processed: AtomicU64,
    pub readings_emitted: AtomicU64,
    pub brain_posts_ok: AtomicU64,
    pub brain_posts_failed: AtomicU64,
}

impl WorkerStats {
    /// Cheap snapshot reader — used by `GetStats` and the heartbeat
    /// log. We don't atomically snapshot all fields; a slightly
    /// inconsistent cross-field view is fine for telemetry.
    #[must_use]
    pub fn snapshot(&self) -> WorkerStatsSnapshot {
        WorkerStatsSnapshot {
            packets_received: self.packets_received.load(Ordering::Relaxed),
            packets_dropped: self.packets_dropped.load(Ordering::Relaxed),
            packets_relayed: self.packets_relayed.load(Ordering::Relaxed),
            windows_processed: self.windows_processed.load(Ordering::Relaxed),
            readings_emitted: self.readings_emitted.load(Ordering::Relaxed),
            brain_posts_ok: self.brain_posts_ok.load(Ordering::Relaxed),
            brain_posts_failed: self.brain_posts_failed.load(Ordering::Relaxed),
        }
    }
}

/// Plain-old-data snapshot of [`WorkerStats`]. Implements `Copy` so
/// it travels through async boundaries with no allocation.
#[derive(Debug, Clone, Copy, Default)]
pub struct WorkerStatsSnapshot {
    pub packets_received: u64,
    pub packets_dropped: u64,
    pub packets_relayed: u64,
    pub windows_processed: u64,
    pub readings_emitted: u64,
    pub brain_posts_ok: u64,
    pub brain_posts_failed: u64,
}

/// Shared worker state — held behind `Arc` and cloned into the gRPC
/// service, the brain loop, and the UDP ingest task.
#[derive(Debug)]
pub struct WorkerState {
    pub config: Arc<Config>,
    pub stats: Arc<WorkerStats>,
    pub started_at: SystemTime,
    pub latest: Arc<RwLock<HashMap<NodeId, VitalReading>>>,
    pub tx: broadcast::Sender<VitalReading>,
}

impl WorkerState {
    /// Construct a fresh state. Returns the state plus a primary
    /// broadcast receiver — held by the caller so the channel never
    /// closes while at least one subscriber may still appear.
    #[must_use]
    pub fn new(config: Config) -> (Arc<Self>, broadcast::Receiver<VitalReading>) {
        let (tx, rx) = broadcast::channel(READING_BROADCAST_CAPACITY);
        let state = Arc::new(Self {
            config: Arc::new(config),
            stats: Arc::new(WorkerStats::default()),
            started_at: SystemTime::now(),
            latest: Arc::new(RwLock::new(HashMap::new())),
            tx,
        });
        (state, rx)
    }

    /// Subscribe to the reading broadcast.
    pub fn subscribe(&self) -> broadcast::Receiver<VitalReading> {
        self.tx.subscribe()
    }

    /// Seconds since the worker booted. Saturates at zero on a clock
    /// rewind (rare, but the cluster Pis run NTP and may slew).
    #[must_use]
    pub fn uptime_seconds(&self) -> u64 {
        SystemTime::now()
            .duration_since(self.started_at)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Record a new reading: bumps counters, replaces the per-node
    /// cache entry, and broadcasts. Subscribers that have lagged are
    /// implicitly dropped — `tx.send` fails silently when no
    /// subscriber is alive, which is fine.
    pub async fn record(&self, reading: VitalReading) {
        self.stats.readings_emitted.fetch_add(1, Ordering::Relaxed);
        self.latest.write().await.insert(reading.node_id, reading);
        let _ = self.tx.send(reading);
    }

    /// Snapshot of latest readings keyed by `node_id`.
    pub async fn latest_snapshot(&self) -> Vec<VitalReading> {
        self.latest.read().await.values().copied().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::unavailable_reading;

    fn cfg() -> Config {
        Config {
            udp_listen: "127.0.0.1:0".parse().unwrap(),
            grpc_listen: "127.0.0.1:0".parse().unwrap(),
            brain_url: "http://127.0.0.1:9876".to_string(),
            window_frames: 50,
            brain_post_interval: std::time::Duration::from_secs(60),
            node_name: "test-host".to_string(),
            verbose: false,
            relay_targets: Vec::new(),
            csi_model_path: None,
            csi_lora_path: None,
        }
    }

    #[tokio::test]
    async fn record_updates_latest_and_counters() {
        let (state, _initial_rx) = WorkerState::new(cfg());
        let r = unavailable_reading(7, 12_345);
        state.record(r).await;
        let snap = state.latest_snapshot().await;
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].node_id, 7);
        assert_eq!(state.stats.readings_emitted.load(Ordering::Relaxed), 1);
    }

    #[tokio::test]
    async fn record_broadcasts_to_subscribers() {
        let (state, _initial) = WorkerState::new(cfg());
        let mut sub = state.subscribe();
        let r = unavailable_reading(3, 100);
        state.record(r).await;
        let received = sub.recv().await.expect("broadcast");
        assert_eq!(received.node_id, 3);
        assert_eq!(received.timestamp_us, 100);
    }

    #[tokio::test]
    async fn stats_snapshot_round_trips() {
        let (state, _) = WorkerState::new(cfg());
        state.stats.packets_received.fetch_add(7, Ordering::Relaxed);
        state.stats.brain_posts_ok.fetch_add(3, Ordering::Relaxed);
        let s = state.stats.snapshot();
        assert_eq!(s.packets_received, 7);
        assert_eq!(s.brain_posts_ok, 3);
    }
}
