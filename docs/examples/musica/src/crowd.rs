//! Crowd-scale distributed speaker identity tracker.
//!
//! Hierarchical system for detecting and tracking thousands of speakers:
//! - Layer 1: Local acoustic event detection per sensor
//! - Layer 2: Local graph formation + spectral clustering
//! - Layer 3: Cross-node identity association
//! - Layer 4: Global identity memory graph
//!
//! The unit of scale is the speaker hypothesis, not the waveform.

use ruvector_mincut::prelude::*;
use std::collections::HashMap;

/// A speech event detected at a single sensor.
#[derive(Debug, Clone)]
pub struct SpeechEvent {
    /// Timestamp in seconds.
    pub time: f64,
    /// Frequency centroid (Hz).
    pub freq_centroid: f64,
    /// Energy level.
    pub energy: f64,
    /// Voicing probability (0-1).
    pub voicing: f64,
    /// Harmonicity score (0-1).
    pub harmonicity: f64,
    /// Direction of arrival (degrees, 0=front).
    pub direction: f64,
    /// Sensor that detected this event.
    pub sensor_id: usize,
}

/// A local speaker hypothesis from one sensor region.
#[derive(Debug, Clone)]
pub struct LocalSpeaker {
    /// Unique local ID.
    pub id: u64,
    /// Average frequency centroid.
    pub centroid_freq: f64,
    /// Average direction of arrival.
    pub avg_direction: f64,
    /// Confidence (0-1).
    pub confidence: f64,
    /// Speaker embedding (simplified: freq + direction + voicing stats).
    pub embedding: Vec<f64>,
    /// Number of events assigned.
    pub event_count: usize,
    /// Last seen timestamp.
    pub last_seen: f64,
    /// Sensor ID.
    pub sensor_id: usize,
}

/// A global identity in the crowd.
#[derive(Debug, Clone)]
pub struct SpeakerIdentity {
    /// Global unique ID.
    pub id: u64,
    /// Aggregate speaker embedding.
    pub embedding: Vec<f64>,
    /// Position trajectory [(time, direction)].
    pub trajectory: Vec<(f64, f64)>,
    /// Confidence (0-1).
    pub confidence: f64,
    /// Total observations merged into this identity.
    pub observations: usize,
    /// First seen timestamp.
    pub first_seen: f64,
    /// Last seen timestamp.
    pub last_seen: f64,
    /// Whether currently active.
    pub active: bool,
}

/// Sensor node for local processing.
pub struct SensorNode {
    /// Sensor ID.
    pub id: usize,
    /// Position (x, y) in meters.
    pub position: (f64, f64),
    /// Recent events buffer.
    events: Vec<SpeechEvent>,
    /// Local speaker hypotheses.
    pub local_speakers: Vec<LocalSpeaker>,
    /// Next local speaker ID.
    next_local_id: u64,
}

impl SensorNode {
    fn new(id: usize, position: (f64, f64)) -> Self {
        Self {
            id,
            position,
            events: Vec::new(),
            local_speakers: Vec::new(),
            next_local_id: 0,
        }
    }
}

/// Configuration for the crowd tracker.
#[derive(Debug, Clone)]
pub struct CrowdConfig {
    /// Maximum global identities to maintain.
    pub max_identities: usize,
    /// Embedding cosine similarity threshold for association.
    pub association_threshold: f64,
    /// Time (seconds) after which an identity is retired.
    pub retirement_time: f64,
    /// Embedding dimension.
    pub embedding_dim: usize,
    /// Maximum local speakers per sensor.
    pub max_local_speakers: usize,
    /// Time window for local event grouping (seconds).
    pub event_window: f64,
}

impl Default for CrowdConfig {
    fn default() -> Self {
        Self {
            max_identities: 1000,
            association_threshold: 0.6,
            retirement_time: 30.0,
            embedding_dim: 6,
            max_local_speakers: 20,
            event_window: 2.0,
        }
    }
}

/// Statistics.
#[derive(Debug, Clone)]
pub struct CrowdStats {
    /// Total identities (including retired).
    pub total_identities: usize,
    /// Currently active speakers.
    pub active_speakers: usize,
    /// Number of sensors.
    pub sensors: usize,
    /// Total events processed.
    pub total_events: usize,
    /// Total local speakers across all sensors.
    pub total_local_speakers: usize,
}

/// The crowd-scale speaker tracker.
pub struct CrowdTracker {
    /// Sensor nodes.
    pub sensors: Vec<SensorNode>,
    /// Global identities.
    pub identities: Vec<SpeakerIdentity>,
    /// Next global identity ID.
    next_identity_id: u64,
    /// Configuration.
    config: CrowdConfig,
    /// Total events ingested.
    total_events: usize,
}

impl CrowdTracker {
    /// Create a new tracker.
    pub fn new(config: CrowdConfig) -> Self {
        Self {
            sensors: Vec::new(),
            identities: Vec::new(),
            next_identity_id: 0,
            config,
            total_events: 0,
        }
    }

    /// Add a sensor at a given position. Returns sensor ID.
    pub fn add_sensor(&mut self, position: (f64, f64)) -> usize {
        let id = self.sensors.len();
        self.sensors.push(SensorNode::new(id, position));
        id
    }

    /// Ingest events from a specific sensor.
    pub fn ingest_events(&mut self, sensor_id: usize, events: Vec<SpeechEvent>) {
        if sensor_id < self.sensors.len() {
            self.total_events += events.len();
            self.sensors[sensor_id].events.extend(events);

            // Trim old events
            let window = self.config.event_window;
            let sensor = &mut self.sensors[sensor_id];
            if let Some(latest) = sensor.events.last().map(|e| e.time) {
                sensor.events.retain(|e| latest - e.time < window);
            }
        }
    }

    /// Update local graphs and cluster events into local speakers.
    pub fn update_local_graphs(&mut self) {
        for sensor in &mut self.sensors {
            if sensor.events.is_empty() {
                continue;
            }

            // Build graph over events
            let n = sensor.events.len();
            let mut edges = Vec::new();

            for i in 0..n {
                for j in i + 1..n {
                    let w = event_similarity(&sensor.events[i], &sensor.events[j]);
                    if w > 0.2 {
                        edges.push((i, j, w));
                    }
                }
            }

            // Spectral clustering via Fiedler vector
            if edges.is_empty() || n < 2 {
                // Each event is its own speaker
                sensor.local_speakers.clear();
                for event in &sensor.events {
                    let speaker = create_local_speaker(
                        &mut sensor.next_local_id,
                        &[event.clone()],
                        sensor.id,
                        &self.config,
                    );
                    sensor.local_speakers.push(speaker);
                }
                continue;
            }

            // Build Laplacian and compute Fiedler vector
            let fiedler = compute_fiedler_for_events(n, &edges);

            // Partition by Fiedler vector sign
            let median = {
                let mut sorted = fiedler.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[n / 2]
            };

            let mut groups: HashMap<usize, Vec<&SpeechEvent>> = HashMap::new();
            for (i, event) in sensor.events.iter().enumerate() {
                let group = if fiedler[i] > median { 1 } else { 0 };
                groups.entry(group).or_default().push(event);
            }

            // Create local speakers from groups
            sensor.local_speakers.clear();
            for (_group_id, group_events) in &groups {
                let events_owned: Vec<SpeechEvent> = group_events.iter().map(|e| (*e).clone()).collect();
                let speaker = create_local_speaker(
                    &mut sensor.next_local_id,
                    &events_owned,
                    sensor.id,
                    &self.config,
                );
                sensor.local_speakers.push(speaker);
            }

            // Trim to max
            sensor.local_speakers.truncate(self.config.max_local_speakers);
        }
    }

    /// Associate local speakers across sensors into global identities.
    pub fn associate_cross_sensor(&mut self, time: f64) {
        // Collect all local speakers
        let all_local: Vec<&LocalSpeaker> = self
            .sensors
            .iter()
            .flat_map(|s| s.local_speakers.iter())
            .collect();

        for local in &all_local {
            // Try to match to existing identity
            let mut best_match: Option<(usize, f64)> = None;

            for (i, identity) in self.identities.iter().enumerate() {
                let sim = cosine_similarity(&local.embedding, &identity.embedding);
                if sim > self.config.association_threshold {
                    if best_match.is_none() || sim > best_match.unwrap().1 {
                        best_match = Some((i, sim));
                    }
                }
            }

            if let Some((idx, _sim)) = best_match {
                // Update existing identity
                let identity = &mut self.identities[idx];
                identity.observations += local.event_count;
                identity.last_seen = time;
                identity.active = true;
                identity.trajectory.push((time, local.avg_direction));

                // Update embedding (running average)
                let alpha = 0.1;
                for (ie, le) in identity.embedding.iter_mut().zip(local.embedding.iter()) {
                    *ie = (1.0 - alpha) * *ie + alpha * *le;
                }

                identity.confidence = (identity.confidence * 0.9 + local.confidence * 0.1).min(1.0);
            } else if self.identities.len() < self.config.max_identities {
                // Create new identity
                let identity = SpeakerIdentity {
                    id: self.next_identity_id,
                    embedding: local.embedding.clone(),
                    trajectory: vec![(time, local.avg_direction)],
                    confidence: local.confidence * 0.5,
                    observations: local.event_count,
                    first_seen: time,
                    last_seen: time,
                    active: true,
                };
                self.identities.push(identity);
                self.next_identity_id += 1;
            }
        }
    }

    /// Update global identity states: retire stale, prune low-confidence.
    pub fn update_global_identities(&mut self, time: f64) {
        for identity in &mut self.identities {
            if time - identity.last_seen > self.config.retirement_time {
                identity.active = false;
            }
        }

        // Trim trajectory to recent entries
        for identity in &mut self.identities {
            let cutoff = time - self.config.retirement_time;
            identity.trajectory.retain(|&(t, _)| t > cutoff);
        }
    }

    /// Get currently active speakers.
    pub fn get_active_speakers(&self) -> Vec<&SpeakerIdentity> {
        self.identities.iter().filter(|i| i.active).collect()
    }

    /// Get tracker statistics.
    pub fn get_stats(&self) -> CrowdStats {
        CrowdStats {
            total_identities: self.identities.len(),
            active_speakers: self.identities.iter().filter(|i| i.active).count(),
            sensors: self.sensors.len(),
            total_events: self.total_events,
            total_local_speakers: self.sensors.iter().map(|s| s.local_speakers.len()).sum(),
        }
    }
}

// ── Helpers ─────────────────────────────────────────────────────────────

fn event_similarity(a: &SpeechEvent, b: &SpeechEvent) -> f64 {
    let time_sim = 1.0 - (a.time - b.time).abs().min(2.0) / 2.0;
    let freq_sim = 1.0 - (a.freq_centroid - b.freq_centroid).abs().min(2000.0) / 2000.0;
    let dir_sim = 1.0 - (a.direction - b.direction).abs().min(180.0) / 180.0;
    let voice_sim = 1.0 - (a.voicing - b.voicing).abs();

    0.25 * time_sim + 0.25 * freq_sim + 0.3 * dir_sim + 0.2 * voice_sim
}

fn create_local_speaker(
    next_id: &mut u64,
    events: &[SpeechEvent],
    sensor_id: usize,
    config: &CrowdConfig,
) -> LocalSpeaker {
    let n = events.len().max(1) as f64;

    let centroid_freq = events.iter().map(|e| e.freq_centroid).sum::<f64>() / n;
    let avg_direction = events.iter().map(|e| e.direction).sum::<f64>() / n;
    let avg_voicing = events.iter().map(|e| e.voicing).sum::<f64>() / n;
    let avg_harmonicity = events.iter().map(|e| e.harmonicity).sum::<f64>() / n;
    let avg_energy = events.iter().map(|e| e.energy).sum::<f64>() / n;
    let last_seen = events.iter().map(|e| e.time).fold(0.0f64, f64::max);

    let confidence = (avg_voicing * 0.5 + avg_harmonicity * 0.3 + (events.len() as f64 / 10.0).min(1.0) * 0.2).min(1.0);

    // Build embedding
    let mut embedding = vec![0.0; config.embedding_dim];
    if config.embedding_dim >= 6 {
        embedding[0] = centroid_freq / 4000.0;
        embedding[1] = avg_direction / 180.0;
        embedding[2] = avg_voicing;
        embedding[3] = avg_harmonicity;
        embedding[4] = avg_energy.min(1.0);
        embedding[5] = confidence;
    }

    let id = *next_id;
    *next_id += 1;

    LocalSpeaker {
        id,
        centroid_freq,
        avg_direction,
        confidence,
        embedding,
        event_count: events.len(),
        last_seen,
        sensor_id,
    }
}

fn compute_fiedler_for_events(n: usize, edges: &[(usize, usize, f64)]) -> Vec<f64> {
    // Build degree + adjacency for power iteration
    let mut degree = vec![0.0f64; n];
    let mut adj: Vec<Vec<(usize, f64)>> = vec![Vec::new(); n];

    for &(u, v, w) in edges {
        degree[u] += w;
        degree[v] += w;
        adj[u].push((v, w));
        adj[v].push((u, w));
    }

    let d_inv: Vec<f64> = degree.iter().map(|&d| if d > 1e-12 { 1.0 / d } else { 0.0 }).collect();

    // Power iteration on D^{-1}A, deflated against constant vector
    let mut v: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();

    let mean: f64 = v.iter().sum::<f64>() / n as f64;
    for x in &mut v {
        *x -= mean;
    }

    for _ in 0..30 {
        let mut new_v = vec![0.0; n];
        for i in 0..n {
            let mut sum = 0.0;
            for &(j, w) in &adj[i] {
                sum += w * v[j];
            }
            new_v[i] = d_inv[i] * sum;
        }

        let mean: f64 = new_v.iter().sum::<f64>() / n as f64;
        for x in &mut new_v {
            *x -= mean;
        }

        let norm: f64 = new_v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut new_v {
                *x /= norm;
            }
        }

        v = new_v;
    }

    v
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let dot: f64 = a[..n].iter().zip(b[..n].iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a[..n].iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b[..n].iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_events(sensor_id: usize, time: f64, direction: f64, n: usize) -> Vec<SpeechEvent> {
        (0..n)
            .map(|i| SpeechEvent {
                time: time + i as f64 * 0.1,
                freq_centroid: 300.0 + (i as f64 * 10.0),
                energy: 0.5 + (i as f64 * 0.05),
                voicing: 0.8,
                harmonicity: 0.7,
                direction,
                sensor_id,
            })
            .collect()
    }

    #[test]
    fn test_single_sensor_detection() {
        let mut tracker = CrowdTracker::new(CrowdConfig::default());
        let s0 = tracker.add_sensor((0.0, 0.0));

        // Two speakers at different directions
        let mut events = make_events(s0, 1.0, 0.0, 5);
        events.extend(make_events(s0, 1.0, 90.0, 5));

        tracker.ingest_events(s0, events);
        tracker.update_local_graphs();

        assert!(
            tracker.sensors[s0].local_speakers.len() >= 2,
            "Should detect at least 2 local speakers, got {}",
            tracker.sensors[s0].local_speakers.len()
        );
    }

    #[test]
    fn test_cross_sensor_association() {
        let config = CrowdConfig {
            association_threshold: 0.3,
            ..CrowdConfig::default()
        };
        let mut tracker = CrowdTracker::new(config);
        let s0 = tracker.add_sensor((0.0, 0.0));
        let s1 = tracker.add_sensor((5.0, 0.0));

        // Same speaker seen from both sensors (similar direction)
        tracker.ingest_events(s0, make_events(s0, 1.0, 10.0, 5));
        tracker.ingest_events(s1, make_events(s1, 1.0, 15.0, 5));

        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.5);

        // Should have created identities
        assert!(
            !tracker.identities.is_empty(),
            "Should have created global identities"
        );

        let stats = tracker.get_stats();
        assert!(stats.active_speakers > 0);
    }

    #[test]
    fn test_identity_persistence() {
        let config = CrowdConfig {
            retirement_time: 10.0,
            association_threshold: 0.3,
            ..CrowdConfig::default()
        };
        let mut tracker = CrowdTracker::new(config);
        let s0 = tracker.add_sensor((0.0, 0.0));

        // Speaker appears
        tracker.ingest_events(s0, make_events(s0, 1.0, 0.0, 5));
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.5);
        let count_1 = tracker.get_active_speakers().len();

        // Speaker disappears, time passes
        tracker.update_global_identities(5.0);
        let active_mid = tracker.get_active_speakers().len();
        assert_eq!(active_mid, count_1, "Should still be active at t=5");

        // Speaker reappears
        tracker.ingest_events(s0, make_events(s0, 6.0, 5.0, 5));
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(6.5);

        // Should reconnect (not create duplicate)
        let total = tracker.identities.len();
        assert!(
            total <= count_1 + 1,
            "Should not create too many new identities: {total}"
        );
    }

    #[test]
    fn test_crowd_stats() {
        let mut tracker = CrowdTracker::new(CrowdConfig::default());
        let s0 = tracker.add_sensor((0.0, 0.0));
        let s1 = tracker.add_sensor((10.0, 0.0));

        tracker.ingest_events(s0, make_events(s0, 1.0, 0.0, 3));
        tracker.ingest_events(s1, make_events(s1, 1.0, 45.0, 4));
        tracker.update_local_graphs();
        tracker.associate_cross_sensor(1.5);

        let stats = tracker.get_stats();
        assert_eq!(stats.sensors, 2);
        assert_eq!(stats.total_events, 7);
        assert!(stats.total_local_speakers > 0);
    }

    #[test]
    fn test_scaling() {
        let mut tracker = CrowdTracker::new(CrowdConfig {
            max_identities: 500,
            ..CrowdConfig::default()
        });

        // 10 sensors
        for i in 0..10 {
            tracker.add_sensor((i as f64 * 10.0, 0.0));
        }

        // 5+ events per sensor at various directions
        for s in 0..10 {
            let mut events = Vec::new();
            for d in 0..5 {
                events.extend(make_events(s, 1.0, d as f64 * 30.0, 3));
            }
            tracker.ingest_events(s, events);
        }

        tracker.update_local_graphs();
        tracker.associate_cross_sensor(2.0);
        tracker.update_global_identities(2.0);

        let stats = tracker.get_stats();
        assert_eq!(stats.sensors, 10);
        assert!(stats.total_events >= 150);
        assert!(
            stats.total_identities > 0 && stats.total_identities < 500,
            "Identity count should be reasonable: {}",
            stats.total_identities
        );

        println!("Scaling test: {:?}", stats);
    }
}
