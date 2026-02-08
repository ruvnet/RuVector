//! Streaming protocol for network transport of world-model data.
//!
//! Defines packet types for keyframes, deltas, and semantic updates. Includes
//! [`ActiveMask`] for tracking which Gaussians are active in a time window and
//! [`BandwidthBudget`] for rate-limiting outgoing data.

use crate::entity::AttributeValue;
use crate::tile::QuantTier;

/// A streaming packet sent over the network.
#[derive(Clone, Debug)]
pub enum StreamPacket {
    /// Full tile snapshot.
    Keyframe(KeyframePacket),
    /// Incremental update relative to a base keyframe.
    Delta(DeltaPacket),
    /// Semantic entity updates (embeddings, attributes).
    Semantic(SemanticPacket),
}

/// Full keyframe packet containing an entire tile's data.
#[derive(Clone, Debug)]
pub struct KeyframePacket {
    /// Tile that this keyframe represents.
    pub tile_id: u64,
    /// Reference time for this keyframe.
    pub time_anchor: f32,
    /// Encoded primitive block data.
    pub primitive_block: Vec<u8>,
    /// Quantization tier of the primitive block.
    pub quant_tier: QuantTier,
    /// Total number of Gaussians in the block.
    pub total_gaussians: u32,
    /// Monotonically increasing sequence number.
    pub sequence: u64,
}

/// Delta packet containing only changed Gaussians since a base keyframe.
#[derive(Clone, Debug)]
pub struct DeltaPacket {
    /// Tile that this delta applies to.
    pub tile_id: u64,
    /// Sequence number of the base keyframe this delta applies to.
    pub base_sequence: u64,
    /// Time range this delta covers.
    pub time_range: [f32; 2],
    /// Bit mask of active Gaussians in this time window.
    pub active_mask: ActiveMask,
    /// Encoded data for updated Gaussians only.
    pub updated_gaussians: Vec<u8>,
    /// Number of Gaussians that were updated.
    pub update_count: u32,
}

/// Bit mask indicating which Gaussians are active in a time window.
///
/// Uses packed `u64` words for efficient storage. Each bit corresponds to
/// one Gaussian; bit index `i` maps to word `i / 64`, bit `i % 64`.
#[derive(Clone, Debug)]
pub struct ActiveMask {
    /// Packed bit storage.
    pub bits: Vec<u64>,
    /// Total number of Gaussians this mask covers.
    pub total_count: u32,
}

impl ActiveMask {
    /// Create a new mask with all Gaussians inactive.
    pub fn new(total_count: u32) -> Self {
        let word_count = (total_count as usize).div_ceil(64);
        Self {
            bits: vec![0u64; word_count],
            total_count,
        }
    }

    /// Set or clear the active bit for a Gaussian.
    #[inline]
    pub fn set(&mut self, index: u32, active: bool) {
        if index >= self.total_count {
            return;
        }
        let word = (index / 64) as usize;
        let bit = index % 64;
        if active {
            self.bits[word] |= 1u64 << bit;
        } else {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    /// Check if a Gaussian is active.
    #[inline]
    pub fn is_active(&self, index: u32) -> bool {
        if index >= self.total_count {
            return false;
        }
        let word = (index / 64) as usize;
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Count the number of active Gaussians.
    pub fn active_count(&self) -> u32 {
        self.bits.iter().map(|w| w.count_ones()).sum()
    }

    /// Return the byte size of the packed bit storage.
    pub fn byte_size(&self) -> usize {
        self.bits.len() * 8
    }
}

/// Semantic update packet containing entity-level changes.
#[derive(Clone, Debug)]
pub struct SemanticPacket {
    /// Entity updates in this packet.
    pub entities: Vec<EntityUpdate>,
    /// Sequence number for ordering.
    pub sequence: u64,
}

/// An individual entity update within a semantic packet.
#[derive(Clone, Debug)]
pub struct EntityUpdate {
    /// Entity being updated.
    pub entity_id: u64,
    /// New embedding vector (if changed).
    pub embedding: Option<Vec<f32>>,
    /// Range of Gaussian IDs associated with this entity (if changed).
    pub gaussian_id_range: Option<(u32, u32)>,
    /// Updated attributes.
    pub attributes: Vec<(String, AttributeValue)>,
}

/// Bandwidth budget controller for rate-limiting outgoing stream data.
///
/// Tracks bytes sent within a sliding window and refuses sends that would
/// exceed the configured maximum bytes-per-second.
pub struct BandwidthBudget {
    /// Maximum bytes allowed per second.
    pub max_bytes_per_second: u64,
    /// Bytes sent in the current window.
    bytes_sent: u64,
    /// Start of the current measurement window (milliseconds).
    window_start_ms: u64,
}

impl BandwidthBudget {
    /// Create a new bandwidth budget.
    pub fn new(max_bps: u64) -> Self {
        Self {
            max_bytes_per_second: max_bps,
            bytes_sent: 0,
            window_start_ms: 0,
        }
    }

    /// Check if sending `bytes` at time `now_ms` would stay within budget.
    ///
    /// If the current window has expired (more than 1 second since start),
    /// the check considers only the new send against a fresh window.
    pub fn can_send(&self, bytes: u64, now_ms: u64) -> bool {
        let elapsed = now_ms.saturating_sub(self.window_start_ms);
        if elapsed >= 1000 {
            // Window has expired; the new send starts a fresh window
            return bytes <= self.max_bytes_per_second;
        }
        self.bytes_sent + bytes <= self.max_bytes_per_second
    }

    /// Record that `bytes` were sent at time `now_ms`.
    ///
    /// Automatically resets the window if more than 1 second has elapsed.
    pub fn record_sent(&mut self, bytes: u64, now_ms: u64) {
        let elapsed = now_ms.saturating_sub(self.window_start_ms);
        if elapsed >= 1000 {
            self.reset_window(now_ms);
        }
        self.bytes_sent += bytes;
    }

    /// Reset the measurement window to start at `now_ms`.
    pub fn reset_window(&mut self, now_ms: u64) {
        self.window_start_ms = now_ms;
        self.bytes_sent = 0;
    }

    /// Return current utilization as a fraction (0.0 to 1.0+).
    pub fn utilization(&self) -> f32 {
        if self.max_bytes_per_second == 0 {
            return 1.0;
        }
        self.bytes_sent as f32 / self.max_bytes_per_second as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- ActiveMask tests --

    #[test]
    fn test_active_mask_new() {
        let mask = ActiveMask::new(100);
        assert_eq!(mask.total_count, 100);
        assert_eq!(mask.active_count(), 0);
        assert_eq!(mask.bits.len(), 2); // ceil(100/64) = 2
    }

    #[test]
    fn test_active_mask_set_get() {
        let mut mask = ActiveMask::new(128);
        assert!(!mask.is_active(0));
        mask.set(0, true);
        assert!(mask.is_active(0));
        mask.set(63, true);
        assert!(mask.is_active(63));
        mask.set(64, true);
        assert!(mask.is_active(64));
        mask.set(127, true);
        assert!(mask.is_active(127));
        assert_eq!(mask.active_count(), 4);
    }

    #[test]
    fn test_active_mask_clear() {
        let mut mask = ActiveMask::new(64);
        mask.set(10, true);
        assert!(mask.is_active(10));
        mask.set(10, false);
        assert!(!mask.is_active(10));
        assert_eq!(mask.active_count(), 0);
    }

    #[test]
    fn test_active_mask_out_of_bounds() {
        let mut mask = ActiveMask::new(10);
        mask.set(100, true); // should be a no-op
        assert!(!mask.is_active(100));
    }

    #[test]
    fn test_active_mask_byte_size() {
        let mask = ActiveMask::new(200);
        // ceil(200/64) = 4 words * 8 bytes = 32 bytes
        assert_eq!(mask.byte_size(), 32);
    }

    #[test]
    fn test_active_mask_zero_count() {
        let mask = ActiveMask::new(0);
        assert_eq!(mask.total_count, 0);
        assert_eq!(mask.active_count(), 0);
        assert_eq!(mask.byte_size(), 0);
    }

    // -- BandwidthBudget tests --

    #[test]
    fn test_bandwidth_budget_new() {
        let budget = BandwidthBudget::new(1_000_000);
        assert_eq!(budget.max_bytes_per_second, 1_000_000);
        assert!((budget.utilization() - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_can_send_within_budget() {
        let mut budget = BandwidthBudget::new(1000);
        budget.reset_window(0);
        assert!(budget.can_send(500, 0));
        budget.record_sent(500, 0);
        assert!(budget.can_send(500, 0));
        assert!(!budget.can_send(501, 0));
    }

    #[test]
    fn test_can_send_after_window_reset() {
        let mut budget = BandwidthBudget::new(1000);
        budget.reset_window(0);
        budget.record_sent(1000, 0);
        assert!(!budget.can_send(1, 500)); // still in window
        assert!(budget.can_send(1000, 1000)); // window expired
    }

    #[test]
    fn test_record_sent_auto_resets() {
        let mut budget = BandwidthBudget::new(1000);
        budget.reset_window(0);
        budget.record_sent(800, 0);
        assert!((budget.utilization() - 0.8).abs() < 1e-6);

        // After window expires, recording should reset
        budget.record_sent(200, 1500);
        assert!((budget.utilization() - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_utilization_zero_budget() {
        let budget = BandwidthBudget::new(0);
        assert!((budget.utilization() - 1.0).abs() < 1e-6);
    }
}
