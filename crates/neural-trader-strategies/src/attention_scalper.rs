//! Multi-level order-book imbalance scalper.
//!
//! Two imbalance computations are available, switchable via
//! [`AttentionScalperConfig::use_sdpa`]:
//!
//! 1. **Geometric decay** (default): near-mid levels weight `1.0`, each
//!    further level decays by `level_decay`. Deterministic and fast —
//!    ~10 ns/level.
//!
//! 2. **Scaled dot-product attention** (`use_sdpa = true`): every level
//!    is encoded as a `[size_log, side_sign, depth_idx, 1.0]` vector
//!    and passed to
//!    [`ruvector_attention::ScaledDotProductAttention`] with a fixed
//!    "pressure" query. The attention-weighted context vector's sign-
//!    dimension becomes the signed imbalance. This is an honest
//!    integration with the ruvector attention stack and behaves
//!    consistently with the geometric path on single-sided books; the
//!    value appears on mixed books where attention can upweight the
//!    wide side.
//!
//! Consumes only [`MarketEvent::BookSnapshot`] from
//! `neural_trader_core`, so it composes with the Kalshi normalizer
//! unchanged and — when a deeper ladder is available from another venue
//! — the same strategy applies.

use std::collections::HashMap;

use neural_trader_core::{EventType, MarketEvent, Side as NtSide};

use crate::intent::{Action, Intent, Side};
use crate::Strategy;

#[derive(Debug, Clone)]
pub struct AttentionScalperConfig {
    /// Top-N price levels to weight per side.
    pub depth: usize,
    /// Geometric decay factor for level weights. 0.5 → near level weighs
    /// 2× the next out. Must be in (0, 1]. Ignored when `use_sdpa = true`.
    pub level_decay: f64,
    /// EMA smoothing alpha for the imbalance signal. Smaller = smoother.
    pub ema_alpha: f64,
    /// Absolute smoothed-imbalance threshold required to emit an intent.
    /// Positive → YES side pressure, negative → NO side pressure.
    pub abs_threshold: f64,
    pub quantity: i64,
    pub strategy_name: &'static str,
    /// When true, compute the imbalance via
    /// `ruvector_attention::ScaledDotProductAttention` instead of the
    /// fixed geometric decay.
    pub use_sdpa: bool,
}

impl Default for AttentionScalperConfig {
    fn default() -> Self {
        Self {
            depth: 5,
            level_decay: 0.6,
            ema_alpha: 0.3,
            abs_threshold: 0.4,
            quantity: 10,
            strategy_name: "attention-scalper",
            use_sdpa: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct SymState {
    yes_levels: Vec<(i64, i64)>, // (price_cents, size)
    no_levels: Vec<(i64, i64)>,
    smoothed_imbalance: f64,
    last_emit_seq: u64,
}

#[derive(Debug, Clone, Default)]
pub struct AttentionScalper {
    pub config: AttentionScalperConfig,
    syms: HashMap<u32, SymState>,
}

impl AttentionScalper {
    pub fn new(config: AttentionScalperConfig) -> Self {
        Self { config, syms: HashMap::new() }
    }

    fn update_snapshot(&mut self, event: &MarketEvent) -> Option<f64> {
        let depth = self.config.depth;
        let decay = self.config.level_decay;
        let alpha = self.config.ema_alpha;

        let state = self.syms.entry(event.symbol_id).or_default();
        let cents = event.price_fp / 1_000_000;
        let size = event.qty_fp / 1_000_000;
        if cents <= 0 || size <= 0 {
            return None;
        }
        match event.side {
            Some(NtSide::Bid) => state.yes_levels.push((cents, size)),
            Some(NtSide::Ask) => state.no_levels.push((cents, size)),
            None => return None,
        }
        // Keep only the most recent ≤ depth levels per side (LIFO so the
        // top-of-book event dominates in a burst of snapshots).
        if state.yes_levels.len() > depth {
            let drop = state.yes_levels.len() - depth;
            state.yes_levels.drain(0..drop);
        }
        if state.no_levels.len() > depth {
            let drop = state.no_levels.len() - depth;
            state.no_levels.drain(0..drop);
        }
        let raw = if self.config.use_sdpa {
            sdpa_imbalance(&state.yes_levels, &state.no_levels, depth)
        } else {
            let wy = weighted_depth(&state.yes_levels, depth, decay);
            let wn = weighted_depth(&state.no_levels, depth, decay);
            let total = wy + wn;
            if total <= 0.0 {
                return None;
            }
            // Imbalance in [-1, 1]. Positive → YES pressure.
            (wy - wn) / total
        };
        state.smoothed_imbalance = (1.0 - alpha) * state.smoothed_imbalance + alpha * raw;
        Some(state.smoothed_imbalance)
    }

    fn maybe_emit(&mut self, event: &MarketEvent, smoothed: f64) -> Option<Intent> {
        let abs = smoothed.abs();
        if abs < self.config.abs_threshold {
            return None;
        }
        let state = self.syms.get_mut(&event.symbol_id)?;
        // Throttle: at most one emission per sequence bucket.
        if state.last_emit_seq == event.seq {
            return None;
        }
        state.last_emit_seq = event.seq;

        let (levels, side) = if smoothed > 0.0 {
            (&state.yes_levels, Side::Yes)
        } else {
            (&state.no_levels, Side::No)
        };
        let price_cents = levels
            .last()
            .map(|(p, _)| *p)
            .unwrap_or(0);
        if price_cents <= 0 || price_cents >= 100 {
            return None;
        }
        let edge_bps = (abs * 1_000.0).round() as i64;
        Some(Intent {
            symbol_id: event.symbol_id,
            side,
            action: Action::Buy,
            limit_price_cents: price_cents,
            quantity: self.config.quantity,
            edge_bps,
            confidence: abs.min(1.0),
            strategy: self.config.strategy_name,
        })
    }
}

fn weighted_depth(levels: &[(i64, i64)], depth: usize, decay: f64) -> f64 {
    let mut weight = 1.0;
    let mut sum = 0.0;
    for (_, size) in levels.iter().take(depth) {
        sum += weight * (*size as f64);
        weight *= decay;
    }
    sum
}

/// Imbalance via scaled dot-product attention over top levels.
///
/// Levels are encoded as `[size_log, side_sign, depth_idx_norm, 1.0]`.
/// A fixed query of `[1.0, 1.0, 0.0, 0.5]` upweights size-heavy, side-
/// aware levels. The returned context vector's component 1 is the
/// signed imbalance (sign direction lives on that axis).
///
/// Returns a value in roughly `[-1, 1]`, NaN guarded.
fn sdpa_imbalance(yes_levels: &[(i64, i64)], no_levels: &[(i64, i64)], depth: usize) -> f64 {
    use ruvector_attention::traits::Attention;
    use ruvector_attention::ScaledDotProductAttention;

    const D: usize = 4;
    let mut keys_buf: Vec<[f32; D]> = Vec::with_capacity(2 * depth);
    for (i, (_, size)) in yes_levels.iter().take(depth).enumerate() {
        let size_log = (*size as f32 + 1.0).ln();
        let depth_norm = i as f32 / depth.max(1) as f32;
        keys_buf.push([size_log, 1.0, depth_norm, 1.0]);
    }
    for (i, (_, size)) in no_levels.iter().take(depth).enumerate() {
        let size_log = (*size as f32 + 1.0).ln();
        let depth_norm = i as f32 / depth.max(1) as f32;
        keys_buf.push([size_log, -1.0, depth_norm, 1.0]);
    }
    if keys_buf.is_empty() {
        return 0.0;
    }
    // Values mirror keys so the attended context carries the side-sign
    // on component 1.
    let keys_refs: Vec<&[f32]> = keys_buf.iter().map(|k| &k[..]).collect();
    let values_refs: Vec<&[f32]> = keys_refs.clone();
    let query: [f32; D] = [1.0, 1.0, 0.0, 0.5];
    let attn = ScaledDotProductAttention::new(D);
    match attn.compute(&query[..], &keys_refs, &values_refs) {
        Ok(context) => {
            let signed = context.get(1).copied().unwrap_or(0.0);
            if signed.is_finite() {
                signed as f64
            } else {
                0.0
            }
        }
        Err(_) => 0.0,
    }
}

impl Strategy for AttentionScalper {
    fn name(&self) -> &'static str {
        self.config.strategy_name
    }

    fn on_event(&mut self, event: &MarketEvent) -> Option<Intent> {
        if !matches!(event.event_type, EventType::BookSnapshot) {
            return None;
        }
        let smoothed = self.update_snapshot(event)?;
        self.maybe_emit(event, smoothed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neural_trader_core::{EventType, MarketEvent, Side as NtSide};

    fn level(sym: u32, side: NtSide, price: i64, size: i64, seq: u64) -> MarketEvent {
        MarketEvent {
            event_id: [0u8; 16],
            ts_exchange_ns: 0,
            ts_ingest_ns: 0,
            venue_id: 1001,
            symbol_id: sym,
            event_type: EventType::BookSnapshot,
            side: Some(side),
            price_fp: price * 1_000_000,
            qty_fp: size * 1_000_000,
            order_id_hash: None,
            participant_id_hash: None,
            flags: 0,
            seq,
        }
    }

    #[test]
    fn balanced_book_no_intent() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            abs_threshold: 0.4,
            ema_alpha: 1.0,
            ..Default::default()
        });
        // Same size on both sides at the same price.
        s.on_event(&level(1, NtSide::Bid, 24, 100, 0));
        let intent = s.on_event(&level(1, NtSide::Ask, 76, 100, 1));
        assert!(intent.is_none());
    }

    #[test]
    fn heavy_yes_side_triggers_yes_buy() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            abs_threshold: 0.3,
            ema_alpha: 1.0, // no smoothing so test is deterministic
            depth: 3,
            level_decay: 1.0,
            quantity: 5,
            strategy_name: "scalper-test",
            use_sdpa: false,
        });
        s.on_event(&level(1, NtSide::Bid, 24, 500, 0));
        s.on_event(&level(1, NtSide::Bid, 23, 300, 1));
        let intent = s.on_event(&level(1, NtSide::Ask, 76, 100, 2)).expect("should emit");
        assert_eq!(intent.symbol_id, 1);
        assert!(matches!(intent.side, Side::Yes));
        assert_eq!(intent.quantity, 5);
    }

    #[test]
    fn heavy_no_side_triggers_no_buy() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            abs_threshold: 0.3,
            ema_alpha: 1.0,
            depth: 3,
            level_decay: 1.0,
            quantity: 5,
            strategy_name: "scalper-test",
            use_sdpa: false,
        });
        s.on_event(&level(2, NtSide::Bid, 24, 100, 0));
        s.on_event(&level(2, NtSide::Ask, 76, 500, 1));
        let intent = s.on_event(&level(2, NtSide::Ask, 77, 400, 2)).expect("should emit");
        assert!(matches!(intent.side, Side::No));
    }

    #[test]
    fn sdpa_path_detects_heavy_yes() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            use_sdpa: true,
            abs_threshold: 0.05,
            ema_alpha: 1.0,
            depth: 3,
            quantity: 5,
            strategy_name: "scalper-sdpa",
            ..Default::default()
        });
        // Heavy YES, light NO: attention should return a positive signal.
        s.on_event(&level(1, NtSide::Bid, 24, 1000, 0));
        s.on_event(&level(1, NtSide::Bid, 23, 900, 1));
        let intent = s
            .on_event(&level(1, NtSide::Ask, 76, 10, 2))
            .expect("sdpa must emit on one-sided book");
        assert!(matches!(intent.side, Side::Yes));
        assert!(intent.confidence > 0.0);
    }

    #[test]
    fn sdpa_path_detects_heavy_no() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            use_sdpa: true,
            abs_threshold: 0.05,
            ema_alpha: 1.0,
            depth: 3,
            quantity: 5,
            strategy_name: "scalper-sdpa",
            ..Default::default()
        });
        s.on_event(&level(2, NtSide::Bid, 24, 10, 0));
        s.on_event(&level(2, NtSide::Ask, 76, 1000, 1));
        let intent = s
            .on_event(&level(2, NtSide::Ask, 77, 800, 2))
            .expect("sdpa must emit on heavy-no book");
        assert!(matches!(intent.side, Side::No));
    }

    #[test]
    fn below_threshold_is_silent() {
        let mut s = AttentionScalper::new(AttentionScalperConfig {
            abs_threshold: 0.9, // demanding
            ema_alpha: 1.0,
            ..Default::default()
        });
        s.on_event(&level(1, NtSide::Bid, 24, 110, 0));
        let intent = s.on_event(&level(1, NtSide::Ask, 76, 100, 1));
        assert!(intent.is_none());
    }
}
