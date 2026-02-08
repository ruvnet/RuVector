//! Append-only lineage log for world-model provenance tracking.
//!
//! Every mutation to the world model (tile creation, update, merge, entity changes)
//! is recorded as a [`LineageEvent`] with full provenance information. The log is
//! append-only and supports queries by tile ID, time range, and rollback-point
//! discovery.

use crate::coherence::CoherenceDecision;

/// A lineage event recording provenance of a world-model mutation.
#[derive(Clone, Debug)]
pub struct LineageEvent {
    /// Unique event identifier (monotonically increasing).
    pub event_id: u64,
    /// Wall-clock timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Tile affected by this event.
    pub tile_id: u64,
    /// Type of mutation.
    pub event_type: LineageEventType,
    /// Provenance (who/what produced this data).
    pub provenance: Provenance,
    /// If set, the event_id of a known-good state to rollback to.
    pub rollback_pointer: Option<u64>,
    /// The coherence decision that governed this event.
    pub coherence_decision: CoherenceDecision,
    /// Coherence score at the time of the event.
    pub coherence_score: f32,
}

/// Types of lineage events.
#[derive(Clone, Debug)]
pub enum LineageEventType {
    /// A new tile was created.
    TileCreated,
    /// An existing tile was updated.
    TileUpdated {
        /// Size of the delta in bytes.
        delta_size: u32,
    },
    /// Multiple tiles were merged.
    TileMerged {
        /// IDs of the source tiles.
        source_tiles: Vec<u64>,
    },
    /// An entity was added to a tile.
    EntityAdded {
        /// The entity that was added.
        entity_id: u64,
    },
    /// An entity was updated within a tile.
    EntityUpdated {
        /// The entity that was updated.
        entity_id: u64,
    },
    /// The tile was rolled back.
    Rollback {
        /// Human-readable reason for rollback.
        reason: String,
    },
    /// The tile was frozen.
    Freeze {
        /// Human-readable reason for freeze.
        reason: String,
    },
}

/// Provenance information for a lineage event.
#[derive(Clone, Debug)]
pub struct Provenance {
    /// Source of the data.
    pub source: ProvenanceSource,
    /// Confidence in the data (0.0 to 1.0).
    pub confidence: f32,
    /// Optional cryptographic signature over the event.
    pub signature: Option<Vec<u8>>,
}

/// Source of a data update.
#[derive(Clone, Debug)]
pub enum ProvenanceSource {
    /// Data from a physical sensor.
    Sensor {
        /// Identifier of the sensor.
        sensor_id: String,
    },
    /// Data from a model inference.
    Inference {
        /// Identifier of the model.
        model_id: String,
    },
    /// Data from manual entry.
    Manual {
        /// Identifier of the user.
        user_id: String,
    },
    /// Data merged from multiple sources.
    Merge {
        /// Event IDs of the source events.
        sources: Vec<u64>,
    },
}

/// Append-only lineage log.
pub struct LineageLog {
    events: Vec<LineageEvent>,
    next_id: u64,
}

impl LineageLog {
    /// Create a new empty lineage log.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            next_id: 0,
        }
    }

    /// Append a new event to the log.
    ///
    /// Returns the assigned event ID.
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &mut self,
        timestamp_ms: u64,
        tile_id: u64,
        event_type: LineageEventType,
        provenance: Provenance,
        rollback_pointer: Option<u64>,
        coherence_decision: CoherenceDecision,
        coherence_score: f32,
    ) -> u64 {
        let event_id = self.next_id;
        self.next_id += 1;

        self.events.push(LineageEvent {
            event_id,
            timestamp_ms,
            tile_id,
            event_type,
            provenance,
            rollback_pointer,
            coherence_decision,
            coherence_score,
        });

        event_id
    }

    /// Retrieve an event by its ID.
    pub fn get(&self, event_id: u64) -> Option<&LineageEvent> {
        // Events are stored in order of ID, so we can index directly
        // if event_id < next_id.
        if event_id < self.next_id {
            self.events.get(event_id as usize)
        } else {
            None
        }
    }

    /// Query all events for a given tile ID, in chronological order.
    pub fn query_tile(&self, tile_id: u64) -> Vec<&LineageEvent> {
        self.events
            .iter()
            .filter(|e| e.tile_id == tile_id)
            .collect()
    }

    /// Query all events within a timestamp range (inclusive).
    pub fn query_range(&self, start_ms: u64, end_ms: u64) -> Vec<&LineageEvent> {
        self.events
            .iter()
            .filter(|e| e.timestamp_ms >= start_ms && e.timestamp_ms <= end_ms)
            .collect()
    }

    /// Find the most recent rollback point for a tile.
    ///
    /// Searches backwards through the tile's events to find the most recent
    /// event that has a rollback pointer set, and returns that pointer's target
    /// event ID. If no rollback pointer exists, returns the ID of the most recent
    /// event with an `Accept` coherence decision.
    pub fn find_rollback_point(&self, tile_id: u64) -> Option<u64> {
        let tile_events: Vec<&LineageEvent> = self
            .events
            .iter()
            .filter(|e| e.tile_id == tile_id)
            .collect();

        // First, look for an explicit rollback pointer (search from most recent)
        for event in tile_events.iter().rev() {
            if let Some(ptr) = event.rollback_pointer {
                return Some(ptr);
            }
        }

        // Fallback: find the most recent accepted event
        for event in tile_events.iter().rev() {
            if event.coherence_decision == CoherenceDecision::Accept {
                return Some(event.event_id);
            }
        }

        None
    }

    /// Return the total number of events in the log.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Check if the log is empty.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

impl Default for LineageLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sensor_provenance() -> Provenance {
        Provenance {
            source: ProvenanceSource::Sensor {
                sensor_id: "cam-0".to_string(),
            },
            confidence: 0.95,
            signature: None,
        }
    }

    #[test]
    fn test_append_and_get() {
        let mut log = LineageLog::new();
        let id = log.append(
            1000,
            42,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            0.95,
        );
        assert_eq!(id, 0);
        assert_eq!(log.len(), 1);

        let event = log.get(0).unwrap();
        assert_eq!(event.tile_id, 42);
        assert_eq!(event.timestamp_ms, 1000);
    }

    #[test]
    fn test_sequential_ids() {
        let mut log = LineageLog::new();
        let id0 = log.append(
            100,
            1,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        let id1 = log.append(
            200,
            2,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn test_query_tile() {
        let mut log = LineageLog::new();
        log.append(
            100,
            1,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        log.append(
            200,
            2,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        log.append(
            300,
            1,
            LineageEventType::TileUpdated { delta_size: 100 },
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            0.9,
        );

        let tile1_events = log.query_tile(1);
        assert_eq!(tile1_events.len(), 2);
        assert_eq!(tile1_events[0].timestamp_ms, 100);
        assert_eq!(tile1_events[1].timestamp_ms, 300);
    }

    #[test]
    fn test_query_range() {
        let mut log = LineageLog::new();
        log.append(
            100,
            1,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        log.append(
            200,
            1,
            LineageEventType::TileUpdated { delta_size: 50 },
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            0.9,
        );
        log.append(
            300,
            1,
            LineageEventType::TileUpdated { delta_size: 75 },
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            0.85,
        );

        let range = log.query_range(150, 250);
        assert_eq!(range.len(), 1);
        assert_eq!(range[0].timestamp_ms, 200);
    }

    #[test]
    fn test_find_rollback_point_explicit() {
        let mut log = LineageLog::new();
        let id0 = log.append(
            100,
            1,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        log.append(
            200,
            1,
            LineageEventType::TileUpdated { delta_size: 50 },
            sensor_provenance(),
            Some(id0), // rollback to creation
            CoherenceDecision::Defer,
            0.5,
        );

        let rb = log.find_rollback_point(1);
        assert_eq!(rb, Some(0));
    }

    #[test]
    fn test_find_rollback_point_fallback_to_accept() {
        let mut log = LineageLog::new();
        log.append(
            100,
            1,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        log.append(
            200,
            1,
            LineageEventType::TileUpdated { delta_size: 50 },
            sensor_provenance(),
            None,
            CoherenceDecision::Defer,
            0.5,
        );

        let rb = log.find_rollback_point(1);
        // Should find the Accept event (id=0)
        assert_eq!(rb, Some(0));
    }

    #[test]
    fn test_find_rollback_point_no_events() {
        let log = LineageLog::new();
        assert_eq!(log.find_rollback_point(999), None);
    }

    #[test]
    fn test_get_nonexistent() {
        let log = LineageLog::new();
        assert!(log.get(0).is_none());
        assert!(log.get(100).is_none());
    }

    #[test]
    fn test_is_empty() {
        let mut log = LineageLog::new();
        assert!(log.is_empty());
        log.append(
            0,
            0,
            LineageEventType::TileCreated,
            sensor_provenance(),
            None,
            CoherenceDecision::Accept,
            1.0,
        );
        assert!(!log.is_empty());
    }
}
