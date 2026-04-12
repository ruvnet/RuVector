//! Witness log — spec section 14.
//!
//! Every mutating operation appends exactly one [`WitnessEvent`]. Reads emit
//! nothing. [`WitnessLog::flush`] drains the buffer for downstream shipping
//! without losing the `WitnessCursor` ordering invariant.
//!
//! # Example
//!
//! ```
//! use ruvector_field::witness::{WitnessEvent, WitnessLog};
//! use ruvector_field::model::NodeId;
//! let mut log = WitnessLog::new();
//! log.emit(WitnessEvent::FieldNodeCreated { node: NodeId(1), ts_ns: 0 });
//! assert_eq!(log.len(), 1);
//! let flushed = log.flush();
//! assert_eq!(flushed.len(), 1);
//! assert_eq!(log.len(), 0);
//! ```

use crate::model::{EdgeKind, HintId, NodeId, Shell, WitnessCursor};

/// Nine witness events defined by the spec.
#[derive(Debug, Clone, PartialEq)]
pub enum WitnessEvent {
    /// New node was appended.
    FieldNodeCreated {
        /// Node id.
        node: NodeId,
        /// Timestamp, nanoseconds.
        ts_ns: u64,
    },
    /// Edge inserted or its weight bumped.
    FieldEdgeUpserted {
        /// Source.
        src: NodeId,
        /// Destination.
        dst: NodeId,
        /// Kind.
        kind: EdgeKind,
        /// Weight.
        weight: f32,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Semantic antipode link committed.
    AntipodeBound {
        /// First node.
        a: NodeId,
        /// Second node.
        b: NodeId,
        /// Binding weight.
        weight: f32,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Node was promoted to a deeper shell.
    ShellPromoted {
        /// Node id.
        node: NodeId,
        /// Previous shell.
        from: Shell,
        /// New shell.
        to: Shell,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Node was demoted to a shallower shell.
    ShellDemoted {
        /// Node id.
        node: NodeId,
        /// Previous shell.
        from: Shell,
        /// New shell.
        to: Shell,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Contradiction observed and flagged.
    ContradictionFlagged {
        /// Node whose antipode fired.
        node: NodeId,
        /// Antipode node.
        antipode: NodeId,
        /// Confidence of the flag.
        confidence: f32,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Routing hint issued (advisory only).
    RoutingHintIssued {
        /// Hint id.
        hint: HintId,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Routing hint committed through the proof gate.
    RoutingHintCommitted {
        /// Hint id.
        hint: HintId,
        /// Timestamp.
        ts_ns: u64,
    },
    /// Snapshot committed to storage.
    FieldSnapshotCommitted {
        /// Snapshot's witness cursor.
        cursor: WitnessCursor,
        /// Timestamp.
        ts_ns: u64,
    },
}

impl WitnessEvent {
    /// Short variant tag, useful for log filtering.
    pub fn tag(&self) -> &'static str {
        match self {
            WitnessEvent::FieldNodeCreated { .. } => "field_node_created",
            WitnessEvent::FieldEdgeUpserted { .. } => "field_edge_upserted",
            WitnessEvent::AntipodeBound { .. } => "antipode_bound",
            WitnessEvent::ShellPromoted { .. } => "shell_promoted",
            WitnessEvent::ShellDemoted { .. } => "shell_demoted",
            WitnessEvent::ContradictionFlagged { .. } => "contradiction_flagged",
            WitnessEvent::RoutingHintIssued { .. } => "routing_hint_issued",
            WitnessEvent::RoutingHintCommitted { .. } => "routing_hint_committed",
            WitnessEvent::FieldSnapshotCommitted { .. } => "field_snapshot_committed",
        }
    }
}

/// Append-only witness log with a monotonically increasing cursor.
#[derive(Debug, Clone, Default)]
pub struct WitnessLog {
    events: Vec<WitnessEvent>,
    cursor: u64,
}

impl WitnessLog {
    /// Empty log.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append one event and advance the cursor.
    pub fn emit(&mut self, event: WitnessEvent) {
        self.events.push(event);
        self.cursor += 1;
    }

    /// Number of queued events.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// `true` if there are no queued events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Current cursor (monotonic across the lifetime of the log).
    pub fn cursor(&self) -> WitnessCursor {
        WitnessCursor(self.cursor)
    }

    /// Borrow the event buffer without draining.
    pub fn events(&self) -> &[WitnessEvent] {
        &self.events
    }

    /// Drain the queued events. Cursor is preserved for downstream dedup.
    pub fn flush(&mut self) -> Vec<WitnessEvent> {
        std::mem::take(&mut self.events)
    }
}
