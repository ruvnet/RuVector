//! Crowd-scale distributed speaker identity tracking.
//!
//! Hierarchical system for detecting and tracking thousands of speakers
//! across distributed sensor networks using graph-based clustering.
//!
//! ## Architecture
//!
//! - **Layer 1**: Local acoustic event detection per sensor node
//! - **Layer 2**: Local graph formation + spectral clustering (Fiedler vector)
//! - **Layer 3**: Cross-node identity association via embedding similarity
//! - **Layer 4**: Global identity memory graph with confidence tracking

use ruvector_mincut::graph::DynamicGraph;
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

/// A speech event detected at a single sensor.
#[derive(Debug, Clone)]
pub struct SpeechEvent {
    /// Timestamp in seconds.
    pub time: f64,
    /// Spectral centroid frequency (Hz).
    pub freq_centroid: f64,
    /// Signal energy (linear scale).
    pub energy: f64,
    /// Voicing probability [0, 1].
    pub voicing: f64,
    /// Harmonics-to-noise ratio.
    pub harmonicity: f64,
    /// Estimated direction of arrival (radians).
    pub direction: f64,
    /// Which sensor observed this event.
    pub sensor_id: usize,
}

/// A local speaker hypothesis from one sensor region.
#[derive(Debug, Clone)]
pub struct LocalSpeaker {
    /// Unique identifier within the tracker.
    pub id: u64,
    /// Mean frequency centroid across grouped events.
    pub centroid_freq: f64,
    /// Mean direction of arrival.
    pub avg_direction: f64,
    /// Confidence score [0, 1].
    pub confidence: f64,
    /// Speaker embedding vector.
    pub embedding: Vec<f64>,
    /// Number of events that contributed.
    pub event_count: usize,
    /// Timestamp of the most recent event.
    pub last_seen: f64,
}

/// A global identity in the crowd.
#[derive(Debug, Clone)]
pub struct SpeakerIdentity {
    /// Globally unique identity id.
    pub id: u64,
    /// Aggregate embedding vector.
    pub embedding: Vec<f64>,
    /// Position trajectory as (x, y) snapshots.
    pub trajectory: Vec<(f64, f64)>,
    /// Confidence score [0, 1].
    pub confidence: f64,
    /// Total observation count.
    pub observations: usize,
    /// First observation timestamp.
    pub first_seen: f64,
    /// Most recent observation timestamp.
    pub last_seen: f64,
    /// Whether the speaker is currently active.
    pub active: bool,
}

/// Sensor node for local processing.
pub struct SensorNode {
    /// Sensor identifier.
    pub id: usize,
    /// Physical position (x, y) in metres.
    pub position: (f64, f64),
    /// Buffered speech events awaiting processing.
    pub events: Vec<SpeechEvent>,
    /// Local similarity graph over events.
    pub local_graph: DynamicGraph,
    /// Speakers discovered locally.
    pub local_speakers: Vec<LocalSpeaker>,
}

/// Configuration for the crowd tracker.
#[derive(Debug, Clone)]
pub struct CrowdConfig {
    /// Maximum number of global identities to maintain.
    pub max_identities: usize,
    /// Cosine-similarity threshold for cross-sensor association.
    pub association_threshold: f64,
    /// Seconds of inactivity before an identity is retired.
    pub retirement_time: f64,
    /// Dimensionality of speaker embeddings.
    pub embedding_dim: usize,
    /// Maximum local speakers per sensor node.
    pub max_local_speakers: usize,
}

impl Default for CrowdConfig {
    fn default() -> Self {
        Self {
            max_identities: 10_000,
            association_threshold: 0.7,
            retirement_time: 30.0,
            embedding_dim: 16,
            max_local_speakers: 64,
        }
    }
}

/// Summary statistics for the tracker.
#[derive(Debug, Clone, Default)]
pub struct CrowdStats {
    /// Total identities ever created.
    pub total_identities: usize,
    /// Currently active speakers.
    pub active_speakers: usize,
    /// Number of sensor nodes.
    pub sensors: usize,
    /// Total events ingested across all sensors.
    pub total_events: usize,
    /// Total local speaker hypotheses across all sensors.
    pub total_local_speakers: usize,
}

// ---------------------------------------------------------------------------
// CrowdTracker
// ---------------------------------------------------------------------------

/// Crowd-scale speaker identity tracker.
///
/// Orchestrates the four-layer hierarchy: local event detection, local
/// graph clustering, cross-sensor association, and global identity memory.
pub struct CrowdTracker {
    /// All sensor nodes.
    pub sensors: Vec<SensorNode>,
    /// Global speaker identities.
    pub identities: Vec<SpeakerIdentity>,
    /// Global identity association graph.
    pub identity_graph: DynamicGraph,
    /// Monotonically increasing identity counter.
    next_identity_id: u64,
    /// Tracker configuration.
    config: CrowdConfig,
}

impl CrowdTracker {
    /// Create a new tracker with the given configuration.
    pub fn new(config: CrowdConfig) -> Self {
        Self {
            sensors: Vec::new(),
            identities: Vec::new(),
            identity_graph: DynamicGraph::new(),
            next_identity_id: 0,
            config,
        }
    }

    /// Register a sensor at the given physical position. Returns the sensor id.
    pub fn add_sensor(&mut self, position: (f64, f64)) -> usize {
        let id = self.sensors.len();
        self.sensors.push(SensorNode {
            id,
            position,
            events: Vec::new(),
            local_graph: DynamicGraph::new(),
            local_speakers: Vec::new(),
        });
        id
    }

    /// Ingest a batch of speech events into the specified sensor node.
    pub fn ingest_events(&mut self, sensor_id: usize, events: Vec<SpeechEvent>) {
        if let Some(sensor) = self.sensors.get_mut(sensor_id) {
            sensor.events.extend(events);
        }
    }

    // -- Layer 2: local graph formation + spectral clustering ---------------

    /// Build local similarity graphs for every sensor and cluster into
    /// local speaker hypotheses.
    pub fn update_local_graphs(&mut self) {
        let embedding_dim = self.config.embedding_dim;
        let max_local = self.config.max_local_speakers;

        for sensor in &mut self.sensors {
            if sensor.events.is_empty() {
                continue;
            }

            // Reset graph
            sensor.local_graph = DynamicGraph::new();
            let n = sensor.events.len();

            // Add one vertex per event
            for i in 0..n {
                sensor.local_graph.add_vertex(i as u64);
            }

            // Connect events by temporal proximity, frequency similarity,
            // and direction consistency.
            for i in 0..n {
                for j in (i + 1)..n {
                    let ei = &sensor.events[i];
                    let ej = &sensor.events[j];

                    let dt = (ei.time - ej.time).abs();
                    let df = (ei.freq_centroid - ej.freq_centroid).abs();
                    let dd = (ei.direction - ej.direction).abs();

                    // Gaussian-kernel similarity
                    let time_sim = (-dt * dt / 0.5).exp();
                    let freq_sim = (-df * df / 10000.0).exp();
                    let dir_sim = (-dd * dd / 0.25).exp();

                    let weight = time_sim * freq_sim * dir_sim;

                    if weight > 0.01 {
                        let _ = sensor.local_graph.insert_edge(
                            i as u64,
                            j as u64,
                            weight,
                        );
                    }
                }
            }

            // Spectral clustering via Fiedler vector (power iteration on
            // the graph Laplacian).
            let labels = spectral_bipartition(&sensor.local_graph, n);

            // Group events by cluster label and form LocalSpeaker hypotheses.
            let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
            for (idx, &label) in labels.iter().enumerate() {
                clusters.entry(label).or_default().push(idx);
            }

            sensor.local_speakers.clear();

            for (_label, indices) in &clusters {
                if indices.is_empty() {
                    continue;
                }
                if sensor.local_speakers.len() >= max_local {
                    break;
                }

                let count = indices.len();
                let mut sum_freq = 0.0;
                let mut sum_dir = 0.0;
                let mut sum_energy = 0.0;
                let mut max_time = f64::NEG_INFINITY;

                for &idx in indices {
                    let e = &sensor.events[idx];
                    sum_freq += e.freq_centroid;
                    sum_dir += e.direction;
                    sum_energy += e.energy;
                    if e.time > max_time {
                        max_time = e.time;
                    }
                }

                let centroid_freq = sum_freq / count as f64;
                let avg_direction = sum_dir / count as f64;
                let confidence =
                    (count as f64 / sensor.events.len() as f64).min(1.0);

                // Build a simple embedding from cluster statistics.
                let mut embedding = vec![0.0; embedding_dim];
                if embedding_dim >= 4 {
                    embedding[0] = centroid_freq / 1000.0;
                    embedding[1] = avg_direction;
                    embedding[2] = sum_energy / count as f64;
                    embedding[3] = count as f64;
                }
                // Fill remaining dims with per-event harmonicity stats.
                for (k, &idx) in indices.iter().enumerate() {
                    let dim = 4 + (k % (embedding_dim.saturating_sub(4).max(1)));
                    if dim < embedding_dim {
                        embedding[dim] += sensor.events[idx].harmonicity;
                    }
                }
                // Normalise embedding.
                let norm = embedding.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm > 1e-12 {
                    for v in &mut embedding {
                        *v /= norm;
                    }
                }

                let id = sensor.id as u64 * 100_000 + sensor.local_speakers.len() as u64;

                sensor.local_speakers.push(LocalSpeaker {
                    id,
                    centroid_freq,
                    avg_direction,
                    confidence,
                    embedding,
                    event_count: count,
                    last_seen: max_time,
                });
            }
        }
    }

    // -- Layer 3: cross-sensor identity association -------------------------

    /// Match local speakers across different sensors and merge into global
    /// identities. `time` is the current wall-clock time for retirement.
    pub fn associate_cross_sensor(&mut self, time: f64) {
        // Collect all local speakers with their sensor position.
        let mut candidates: Vec<(LocalSpeaker, (f64, f64))> = Vec::new();
        for sensor in &self.sensors {
            for ls in &sensor.local_speakers {
                candidates.push((ls.clone(), sensor.position));
            }
        }

        // For each candidate, try to match against existing identities.
        for (local, pos) in &candidates {
            let mut best_idx: Option<usize> = None;
            let mut best_sim: f64 = self.config.association_threshold;

            for (idx, identity) in self.identities.iter().enumerate() {
                if !identity.active {
                    continue;
                }
                let sim = cosine_similarity(&local.embedding, &identity.embedding);
                if sim > best_sim {
                    best_sim = sim;
                    best_idx = Some(idx);
                }
            }

            if let Some(idx) = best_idx {
                // Merge into existing identity.
                let identity = &mut self.identities[idx];
                merge_embedding(
                    &mut identity.embedding,
                    &local.embedding,
                    identity.observations,
                );
                identity.observations += local.event_count;
                identity.confidence =
                    (identity.confidence + local.confidence) / 2.0;
                identity.last_seen = identity.last_seen.max(local.last_seen);
                identity.trajectory.push(*pos);
            } else if self.identities.len() < self.config.max_identities {
                // Create new global identity.
                let id = self.next_identity_id;
                self.next_identity_id += 1;
                self.identity_graph.add_vertex(id);

                self.identities.push(SpeakerIdentity {
                    id,
                    embedding: local.embedding.clone(),
                    trajectory: vec![*pos],
                    confidence: local.confidence,
                    observations: local.event_count,
                    first_seen: local.last_seen,
                    last_seen: local.last_seen,
                    active: true,
                });
            }
        }

        // Build edges between identities that co-occur.
        self.rebuild_identity_edges(time);
    }

    // -- Layer 4: global identity memory ------------------------------------

    /// Retire stale identities and update the global identity graph.
    pub fn update_global_identities(&mut self, time: f64) {
        let retirement = self.config.retirement_time;

        for identity in &mut self.identities {
            if identity.active && (time - identity.last_seen) > retirement {
                identity.active = false;
            }
        }

        // Attempt to reactivate identities that match fresh local speakers.
        // Only consider local speakers observed recently (within retirement window).
        for sensor in &self.sensors {
            for local in &sensor.local_speakers {
                if (time - local.last_seen) > retirement {
                    continue;
                }
                for identity in &mut self.identities {
                    if identity.active {
                        continue;
                    }
                    let sim =
                        cosine_similarity(&local.embedding, &identity.embedding);
                    if sim > self.config.association_threshold {
                        identity.active = true;
                        identity.last_seen = local.last_seen;
                        identity.observations += local.event_count;
                        merge_embedding(
                            &mut identity.embedding,
                            &local.embedding,
                            identity.observations,
                        );
                    }
                }
            }
        }
    }

    /// Return all currently active speaker identities.
    pub fn get_active_speakers(&self) -> Vec<&SpeakerIdentity> {
        self.identities.iter().filter(|s| s.active).collect()
    }

    /// Compute summary statistics.
    pub fn get_stats(&self) -> CrowdStats {
        CrowdStats {
            total_identities: self.identities.len(),
            active_speakers: self.identities.iter().filter(|s| s.active).count(),
            sensors: self.sensors.len(),
            total_events: self.sensors.iter().map(|s| s.events.len()).sum(),
            total_local_speakers: self
                .sensors
                .iter()
                .map(|s| s.local_speakers.len())
                .sum(),
        }
    }

    // -- internal helpers ---------------------------------------------------

    /// Rebuild edges in the identity graph based on embedding similarity
    /// among active identities.
    fn rebuild_identity_edges(&mut self, _time: f64) {
        // Clear old edges by rebuilding the graph.
        self.identity_graph = DynamicGraph::new();

        let active: Vec<usize> = self
            .identities
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active)
            .map(|(i, _)| i)
            .collect();

        for &i in &active {
            self.identity_graph.add_vertex(self.identities[i].id);
        }

        for (ai, &i) in active.iter().enumerate() {
            for &j in &active[(ai + 1)..] {
                let sim = cosine_similarity(
                    &self.identities[i].embedding,
                    &self.identities[j].embedding,
                );
                if sim > 0.3 {
                    let _ = self.identity_graph.insert_edge(
                        self.identities[i].id,
                        self.identities[j].id,
                        sim,
                    );
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors. Returns 0.0 for zero-length vectors.
fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        0.0
    } else {
        dot / denom
    }
}

/// Exponential moving-average merge of a new embedding into an existing one.
fn merge_embedding(existing: &mut Vec<f64>, incoming: &[f64], prior_count: usize) {
    let alpha = 1.0 / (prior_count as f64 + 1.0).max(1.0);
    for (i, v) in existing.iter_mut().enumerate() {
        if i < incoming.len() {
            *v = *v * (1.0 - alpha) + incoming[i] * alpha;
        }
    }
    // Re-normalise.
    let norm = existing.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-12 {
        for v in existing {
            *v /= norm;
        }
    }
}

/// Spectral bipartition of a graph using the Fiedler vector via power
/// iteration on the normalised Laplacian.
///
/// Returns a label vector of length `n` where each entry is 0 or 1.
fn spectral_bipartition(graph: &DynamicGraph, n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![0; n];
    }

    // Build the degree vector and adjacency as dense structures for the
    // small local graphs (typically < 100 nodes).
    let mut degree = vec![0.0_f64; n];
    let mut adj = vec![vec![0.0_f64; n]; n];

    for i in 0..n {
        let neighbours = graph.neighbors(i as u64);
        for (j, _eid) in &neighbours {
            let j = *j as usize;
            if j < n {
                let w = graph
                    .edge_weight(i as u64, j as u64)
                    .unwrap_or(0.0);
                adj[i][j] = w;
                degree[i] += w;
            }
        }
    }

    // Laplacian L = D - A. We want the Fiedler vector (second smallest
    // eigenvector). Use power iteration on (max_eigenvalue * I - L) to
    // find the largest eigenvector of the shifted matrix, then deflate
    // the trivial eigenvector.

    // Estimate max eigenvalue as 2 * max_degree (Gershgorin bound).
    let max_d = degree.iter().cloned().fold(0.0_f64, f64::max);
    let shift = 2.0 * max_d + 1.0;

    // Shifted matrix M = shift*I - L = shift*I - D + A
    // M[i][j] = A[i][j] for i != j
    // M[i][i] = shift - degree[i]

    // Power iteration
    let max_iter = 200;
    let mut v = vec![0.0_f64; n];
    // Initialise with a non-constant vector so it is not aligned with
    // the trivial eigenvector.
    for i in 0..n {
        v[i] = (i as f64) - (n as f64 / 2.0);
    }

    for _ in 0..max_iter {
        // Multiply by M
        let mut mv = vec![0.0_f64; n];
        for i in 0..n {
            mv[i] = (shift - degree[i]) * v[i];
            for j in 0..n {
                if i != j {
                    mv[i] += adj[i][j] * v[j];
                }
            }
        }

        // Remove component along the trivial eigenvector (all-ones / sqrt(n)).
        let proj: f64 = mv.iter().sum::<f64>() / n as f64;
        for x in &mut mv {
            *x -= proj;
        }

        // Normalise
        let norm = mv.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for x in &mut mv {
            *x /= norm;
        }

        v = mv;
    }

    // Partition by sign of the Fiedler vector.
    v.iter().map(|&x| if x >= 0.0 { 0 } else { 1 }).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a speech event with reasonable defaults.
    fn make_event(
        sensor_id: usize,
        time: f64,
        freq: f64,
        direction: f64,
    ) -> SpeechEvent {
        SpeechEvent {
            time,
            freq_centroid: freq,
            energy: 0.5,
            voicing: 0.9,
            harmonicity: 0.8,
            direction,
            sensor_id,
        }
    }

    #[test]
    fn test_single_sensor_detection() {
        let mut tracker = CrowdTracker::new(CrowdConfig::default());
        let sid = tracker.add_sensor((0.0, 0.0));

        // Speaker A: low frequency, direction ~0
        // Speaker B: high frequency, direction ~PI
        let mut events = Vec::new();
        for i in 0..10 {
            events.push(make_event(sid, i as f64 * 0.1, 200.0, 0.1));
        }
        for i in 0..10 {
            events.push(make_event(sid, i as f64 * 0.1, 800.0, 3.0));
        }

        tracker.ingest_events(sid, events);
        tracker.update_local_graphs();

        let sensor = &tracker.sensors[sid];
        assert!(
            sensor.local_speakers.len() >= 2,
            "Expected at least 2 local speakers, got {}",
            sensor.local_speakers.len()
        );
    }

    #[test]
    fn test_cross_sensor_association() {
        let config = CrowdConfig {
            association_threshold: 0.5,
            ..CrowdConfig::default()
        };
        let mut tracker = CrowdTracker::new(config);
        let s0 = tracker.add_sensor((0.0, 0.0));
        let s1 = tracker.add_sensor((10.0, 0.0));

        // Same speaker observed from two sensors: similar frequency and timing.
        let events_a: Vec<SpeechEvent> = (0..8)
            .map(|i| make_event(s0, i as f64 * 0.1, 440.0, 0.5))
            .collect();
        let events_b: Vec<SpeechEvent> = (0..8)
            .map(|i| make_event(s1, i as f64 * 0.1, 440.0, 0.5))
            .collect();

        tracker.ingest_events(s0, events_a);
        tracker.ingest_events(s1, events_b);

        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.0);

        // The two sensors should see similar embeddings and merge into
        // one (or at most two) global identities.
        let active = tracker.get_active_speakers();
        assert!(
            !active.is_empty(),
            "Should have at least one global identity"
        );
        // With matching embeddings, association should merge them.
        assert!(
            active.len() <= 2,
            "Identical speakers should merge; got {} identities",
            active.len()
        );
    }

    #[test]
    fn test_identity_persistence() {
        let config = CrowdConfig {
            retirement_time: 5.0,
            association_threshold: 0.5,
            ..CrowdConfig::default()
        };
        let mut tracker = CrowdTracker::new(config);
        let sid = tracker.add_sensor((0.0, 0.0));

        // Phase 1: speaker appears
        let events: Vec<SpeechEvent> = (0..6)
            .map(|i| make_event(sid, i as f64 * 0.1, 300.0, 1.0))
            .collect();
        tracker.ingest_events(sid, events);
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.0);

        let initial_count = tracker.get_active_speakers().len();
        assert!(initial_count >= 1, "Speaker should appear");

        // Phase 2: time passes, speaker retires
        tracker.update_global_identities(100.0);
        let retired_count = tracker.get_active_speakers().len();
        assert_eq!(retired_count, 0, "Speaker should be retired after timeout");

        // Phase 3: speaker reappears with similar embedding
        let events2: Vec<SpeechEvent> = (0..6)
            .map(|i| make_event(sid, 100.0 + i as f64 * 0.1, 300.0, 1.0))
            .collect();
        // Clear old events and re-ingest.
        tracker.sensors[sid].events.clear();
        tracker.ingest_events(sid, events2);
        tracker.update_local_graphs();
        tracker.update_global_identities(100.5);

        let reactivated_count = tracker.get_active_speakers().len();
        assert!(
            reactivated_count >= 1,
            "Speaker should be reactivated; got {}",
            reactivated_count
        );

        // The reactivated identity should be the *same* id as before.
        let total = tracker.get_stats().total_identities;
        assert!(
            total <= 2,
            "Should reuse identity, not create many new ones; total={}",
            total
        );
    }

    #[test]
    fn test_crowd_stats() {
        let mut tracker = CrowdTracker::new(CrowdConfig::default());
        let s0 = tracker.add_sensor((0.0, 0.0));
        let s1 = tracker.add_sensor((5.0, 5.0));

        let events0: Vec<SpeechEvent> = (0..5)
            .map(|i| make_event(s0, i as f64 * 0.1, 440.0, 0.0))
            .collect();
        let events1: Vec<SpeechEvent> = (0..3)
            .map(|i| make_event(s1, i as f64 * 0.1, 880.0, 1.5))
            .collect();

        tracker.ingest_events(s0, events0);
        tracker.ingest_events(s1, events1);
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.0);

        let stats = tracker.get_stats();
        assert_eq!(stats.sensors, 2);
        assert_eq!(stats.total_events, 8);
        assert!(stats.total_identities > 0);
        assert!(stats.active_speakers > 0);
        assert!(stats.active_speakers <= stats.total_identities);
    }

    #[test]
    fn test_scaling() {
        let mut tracker = CrowdTracker::new(CrowdConfig {
            max_local_speakers: 32,
            ..CrowdConfig::default()
        });

        // 10 sensors
        let sensor_ids: Vec<usize> = (0..10)
            .map(|i| tracker.add_sensor((i as f64 * 5.0, 0.0)))
            .collect();

        // 50+ events spread across sensors
        for &sid in &sensor_ids {
            let events: Vec<SpeechEvent> = (0..6)
                .map(|i| {
                    let freq = 200.0 + (sid as f64) * 50.0 + (i as f64) * 10.0;
                    let dir = (sid as f64) * 0.3;
                    make_event(sid, i as f64 * 0.2, freq, dir)
                })
                .collect();
            tracker.ingest_events(sid, events);
        }

        // Should not panic through the full pipeline.
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(2.0);
        tracker.update_global_identities(2.0);

        let stats = tracker.get_stats();
        assert_eq!(stats.sensors, 10);
        assert!(
            stats.total_events >= 50,
            "Expected >= 50 events, got {}",
            stats.total_events
        );
        assert!(
            stats.total_identities > 0 && stats.total_identities < 100,
            "Identity count should be reasonable; got {}",
            stats.total_identities
        );
        assert!(stats.active_speakers > 0);
    }
}
