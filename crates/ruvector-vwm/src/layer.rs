//! Static/dynamic Gaussian layer management.
//!
//! Most real-world scenes are mostly static (walls, floors, furniture).
//! Only a small subset of Gaussians are dynamic (people, vehicles, moving objects).
//! This module separates them so the medium and slow loops only touch the dynamic subset.

use crate::gaussian::Gaussian4D;
use crate::streaming::ActiveMask;
use crate::tile::{PrimitiveBlock, QuantTier, Tile, TileCoord};

/// Layer type for Gaussian classification.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LayerType {
    /// Static background layer -- updated rarely, quantized aggressively.
    Static,
    /// Dynamic foreground layer -- updated frequently, kept at high quality.
    Dynamic,
}

/// A managed Gaussian layer with its own tile set and update policy.
#[derive(Clone, Debug)]
pub struct GaussianLayer {
    /// Whether this is a static or dynamic layer.
    pub layer_type: LayerType,
    /// Tiles belonging to this layer.
    pub tiles: Vec<Tile>,
    /// Default quantization tier for this layer.
    pub quant_tier: QuantTier,
    /// Target update rate in Hz.
    pub update_rate_hz: f32,
    /// Total Gaussian count across all tiles (cached).
    pub total_gaussians: u32,
    /// Epoch of the last update applied to this layer.
    pub last_update_epoch: u64,
}

impl GaussianLayer {
    /// Create a new layer with sensible defaults per type.
    ///
    /// - Static layers use aggressive `Cold3` quantization and low update rate (0.1 Hz).
    /// - Dynamic layers use high-quality `Hot8` quantization and higher update rate (5.0 Hz).
    pub fn new(layer_type: LayerType) -> Self {
        match layer_type {
            LayerType::Static => Self {
                layer_type,
                tiles: Vec::new(),
                quant_tier: QuantTier::Cold3,
                update_rate_hz: 0.1,
                total_gaussians: 0,
                last_update_epoch: 0,
            },
            LayerType::Dynamic => Self {
                layer_type,
                tiles: Vec::new(),
                quant_tier: QuantTier::Hot8,
                update_rate_hz: 5.0,
                total_gaussians: 0,
                last_update_epoch: 0,
            },
        }
    }

    /// Add a tile to this layer and update the cached Gaussian count.
    pub fn add_tile(&mut self, tile: Tile) {
        self.total_gaussians += tile.primitive_block.count;
        self.tiles.push(tile);
    }

    /// Get total Gaussian count across all tiles (recomputed from tiles).
    pub fn gaussian_count(&self) -> u32 {
        self.tiles.iter().map(|t| t.primitive_block.count).sum()
    }

    /// Get tile count.
    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

/// Scene with separated static and dynamic layers.
pub struct LayeredScene {
    /// Layer containing static background Gaussians.
    pub static_layer: GaussianLayer,
    /// Layer containing dynamic foreground Gaussians.
    pub dynamic_layer: GaussianLayer,
    epoch: u64,
}

impl LayeredScene {
    /// Create a new empty layered scene.
    pub fn new() -> Self {
        Self {
            static_layer: GaussianLayer::new(LayerType::Static),
            dynamic_layer: GaussianLayer::new(LayerType::Dynamic),
            epoch: 0,
        }
    }

    /// Classify and add Gaussians to the appropriate layer.
    ///
    /// A Gaussian is classified as **dynamic** if any of the following hold:
    /// - It has non-zero velocity on any axis.
    /// - Its time range is bounded (not `[-inf, +inf]`).
    ///
    /// Everything else is **static**.
    ///
    /// Static and dynamic Gaussians are packed into separate tiles sharing the
    /// same [`TileCoord`].
    pub fn classify_and_add(&mut self, gaussians: &[Gaussian4D], coord: TileCoord) {
        let mut static_gs: Vec<Gaussian4D> = Vec::new();
        let mut dynamic_gs: Vec<Gaussian4D> = Vec::new();

        for g in gaussians {
            if is_dynamic(g) {
                dynamic_gs.push(g.clone());
            } else {
                static_gs.push(g.clone());
            }
        }

        if !static_gs.is_empty() {
            let block =
                PrimitiveBlock::encode(&static_gs, self.static_layer.quant_tier);
            let tile = Tile {
                coord: coord.clone(),
                primitive_block: block,
                entity_refs: Vec::new(),
                coherence_score: 1.0,
                last_update_epoch: self.epoch,
            };
            self.static_layer.add_tile(tile);
        }

        if !dynamic_gs.is_empty() {
            let block =
                PrimitiveBlock::encode(&dynamic_gs, self.dynamic_layer.quant_tier);
            let tile = Tile {
                coord,
                primitive_block: block,
                entity_refs: Vec::new(),
                coherence_score: 1.0,
                last_update_epoch: self.epoch,
            };
            self.dynamic_layer.add_tile(tile);
        }
    }

    /// Get the active Gaussian count for a given time.
    ///
    /// Static Gaussians are always active (they have unbounded time ranges by
    /// definition). Dynamic Gaussians are checked individually via
    /// [`Gaussian4D::is_active_at`].
    pub fn active_count_at(&self, t: f32) -> u32 {
        let static_count = self.static_layer.gaussian_count();
        let mut dynamic_active = 0u32;
        for tile in &self.dynamic_layer.tiles {
            let decoded = tile.primitive_block.decode();
            for g in &decoded {
                if g.is_active_at(t) {
                    dynamic_active += 1;
                }
            }
        }
        static_count + dynamic_active
    }

    /// Build an [`ActiveMask`] for the dynamic layer at time `t`.
    ///
    /// Each bit corresponds to one dynamic Gaussian in tile order. The mask
    /// covers `dynamic_layer.gaussian_count()` entries.
    pub fn dynamic_active_mask_at(&self, t: f32) -> ActiveMask {
        let total = self.dynamic_layer.gaussian_count();
        let mut mask = ActiveMask::new(total);
        let mut idx = 0u32;
        for tile in &self.dynamic_layer.tiles {
            let decoded = tile.primitive_block.decode();
            for g in &decoded {
                if g.is_active_at(t) {
                    mask.set(idx, true);
                }
                idx += 1;
            }
        }
        mask
    }

    /// Get combined Gaussian count (static + dynamic).
    pub fn total_gaussians(&self) -> u32 {
        self.static_layer.gaussian_count() + self.dynamic_layer.gaussian_count()
    }

    /// Ratio of dynamic to total Gaussians.
    ///
    /// Returns `0.0` if the scene is empty.
    pub fn dynamic_ratio(&self) -> f32 {
        let total = self.total_gaussians();
        if total == 0 {
            return 0.0;
        }
        self.dynamic_layer.gaussian_count() as f32 / total as f32
    }

    /// Advance epoch (called after successful update cycle).
    ///
    /// Returns the new epoch value.
    pub fn advance_epoch(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    /// Get current epoch.
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl Default for LayeredScene {
    fn default() -> Self {
        Self::new()
    }
}

/// Determine whether a Gaussian should be classified as dynamic.
///
/// Dynamic if it has non-zero velocity on any axis or a bounded time range.
fn is_dynamic(g: &Gaussian4D) -> bool {
    let has_velocity =
        g.velocity[0] != 0.0 || g.velocity[1] != 0.0 || g.velocity[2] != 0.0;
    let has_bounded_time =
        g.time_range[0] != f32::NEG_INFINITY || g.time_range[1] != f32::INFINITY;
    has_velocity || has_bounded_time
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_coord() -> TileCoord {
        TileCoord {
            x: 0,
            y: 0,
            z: 0,
            time_bucket: 0,
            lod: 0,
        }
    }

    fn make_static_gaussian(id: u32) -> Gaussian4D {
        // Default Gaussian: zero velocity, unbounded time range -> static
        Gaussian4D::new([0.0, 0.0, 0.0], id)
    }

    fn make_dynamic_gaussian_velocity(id: u32) -> Gaussian4D {
        let mut g = Gaussian4D::new([1.0, 0.0, 0.0], id);
        g.velocity = [0.5, 0.0, 0.0];
        g.time_range = [0.0, 10.0]; // bounded
        g
    }

    fn make_dynamic_gaussian_time_only(id: u32) -> Gaussian4D {
        let mut g = Gaussian4D::new([0.0, 0.0, 0.0], id);
        g.velocity = [0.0, 0.0, 0.0];
        g.time_range = [1.0, 5.0]; // bounded but no velocity
        g
    }

    // ---- GaussianLayer tests ----

    #[test]
    fn test_static_layer_defaults() {
        let layer = GaussianLayer::new(LayerType::Static);
        assert_eq!(layer.layer_type, LayerType::Static);
        assert_eq!(layer.quant_tier, QuantTier::Cold3);
        assert!((layer.update_rate_hz - 0.1).abs() < f32::EPSILON);
        assert_eq!(layer.gaussian_count(), 0);
        assert_eq!(layer.tile_count(), 0);
    }

    #[test]
    fn test_dynamic_layer_defaults() {
        let layer = GaussianLayer::new(LayerType::Dynamic);
        assert_eq!(layer.layer_type, LayerType::Dynamic);
        assert_eq!(layer.quant_tier, QuantTier::Hot8);
        assert!((layer.update_rate_hz - 5.0).abs() < f32::EPSILON);
        assert_eq!(layer.gaussian_count(), 0);
        assert_eq!(layer.tile_count(), 0);
    }

    #[test]
    fn test_layer_add_tile() {
        let mut layer = GaussianLayer::new(LayerType::Static);
        let gs = vec![make_static_gaussian(0), make_static_gaussian(1)];
        let block = PrimitiveBlock::encode(&gs, QuantTier::Cold3);
        let tile = Tile {
            coord: make_coord(),
            primitive_block: block,
            entity_refs: Vec::new(),
            coherence_score: 1.0,
            last_update_epoch: 0,
        };
        layer.add_tile(tile);
        assert_eq!(layer.tile_count(), 1);
        assert_eq!(layer.gaussian_count(), 2);
        assert_eq!(layer.total_gaussians, 2);
    }

    // ---- Classification tests ----

    #[test]
    fn test_static_gaussians_go_to_static_layer() {
        let mut scene = LayeredScene::new();
        let gs = vec![make_static_gaussian(0), make_static_gaussian(1)];
        scene.classify_and_add(&gs, make_coord());

        assert_eq!(scene.static_layer.gaussian_count(), 2);
        assert_eq!(scene.dynamic_layer.gaussian_count(), 0);
    }

    #[test]
    fn test_dynamic_gaussians_with_velocity_go_to_dynamic_layer() {
        let mut scene = LayeredScene::new();
        let gs = vec![
            make_dynamic_gaussian_velocity(0),
            make_dynamic_gaussian_velocity(1),
        ];
        scene.classify_and_add(&gs, make_coord());

        assert_eq!(scene.static_layer.gaussian_count(), 0);
        assert_eq!(scene.dynamic_layer.gaussian_count(), 2);
    }

    #[test]
    fn test_dynamic_gaussians_with_bounded_time_go_to_dynamic_layer() {
        let mut scene = LayeredScene::new();
        let gs = vec![make_dynamic_gaussian_time_only(0)];
        scene.classify_and_add(&gs, make_coord());

        assert_eq!(scene.static_layer.gaussian_count(), 0);
        assert_eq!(scene.dynamic_layer.gaussian_count(), 1);
    }

    #[test]
    fn test_mixed_scene_classification() {
        let mut scene = LayeredScene::new();
        let gs = vec![
            make_static_gaussian(0),
            make_static_gaussian(1),
            make_static_gaussian(2),
            make_dynamic_gaussian_velocity(10),
            make_dynamic_gaussian_time_only(11),
        ];
        scene.classify_and_add(&gs, make_coord());

        assert_eq!(scene.static_layer.gaussian_count(), 3);
        assert_eq!(scene.dynamic_layer.gaussian_count(), 2);
        assert_eq!(scene.total_gaussians(), 5);
    }

    // ---- active_count_at tests ----

    #[test]
    fn test_active_count_at_all_static() {
        let mut scene = LayeredScene::new();
        let gs = vec![make_static_gaussian(0), make_static_gaussian(1)];
        scene.classify_and_add(&gs, make_coord());

        // Static Gaussians are always active
        assert_eq!(scene.active_count_at(0.0), 2);
        assert_eq!(scene.active_count_at(999.0), 2);
    }

    #[test]
    fn test_active_count_at_filters_dynamic_by_time() {
        let mut scene = LayeredScene::new();
        // g0: static, always active
        // g1: dynamic, active [0, 10]
        // g2: dynamic, active [1, 5]
        let gs = vec![
            make_static_gaussian(0),
            make_dynamic_gaussian_velocity(1),   // time_range [0, 10]
            make_dynamic_gaussian_time_only(2),  // time_range [1, 5]
        ];
        scene.classify_and_add(&gs, make_coord());

        // At t=-1: only static (1)
        assert_eq!(scene.active_count_at(-1.0), 1);
        // At t=0: static(1) + g1 active(1) = 2, g2 not yet active (starts at 1)
        assert_eq!(scene.active_count_at(0.0), 2);
        // At t=3: static(1) + g1(1) + g2(1) = 3
        assert_eq!(scene.active_count_at(3.0), 3);
        // At t=7: static(1) + g1(1) = 2, g2 ended at 5
        assert_eq!(scene.active_count_at(7.0), 2);
        // At t=11: static(1) only, both dynamic ended
        assert_eq!(scene.active_count_at(11.0), 1);
    }

    // ---- dynamic_active_mask_at tests ----

    #[test]
    fn test_dynamic_active_mask_at_empty() {
        let scene = LayeredScene::new();
        let mask = scene.dynamic_active_mask_at(0.0);
        assert_eq!(mask.total_count, 0);
        assert_eq!(mask.active_count(), 0);
    }

    #[test]
    fn test_dynamic_active_mask_at_produces_correct_mask() {
        let mut scene = LayeredScene::new();
        // Two dynamic Gaussians:
        // idx 0: time_range [0, 10] (from make_dynamic_gaussian_velocity)
        // idx 1: time_range [1, 5]  (from make_dynamic_gaussian_time_only)
        let gs = vec![
            make_dynamic_gaussian_velocity(10),
            make_dynamic_gaussian_time_only(11),
        ];
        scene.classify_and_add(&gs, make_coord());

        assert_eq!(scene.dynamic_layer.gaussian_count(), 2);

        // At t=3, both are active
        let mask = scene.dynamic_active_mask_at(3.0);
        assert_eq!(mask.total_count, 2);
        assert!(mask.is_active(0));
        assert!(mask.is_active(1));
        assert_eq!(mask.active_count(), 2);

        // At t=7, only first is active (second ended at 5)
        let mask = scene.dynamic_active_mask_at(7.0);
        assert!(mask.is_active(0));
        assert!(!mask.is_active(1));
        assert_eq!(mask.active_count(), 1);

        // At t=11, neither is active
        let mask = scene.dynamic_active_mask_at(11.0);
        assert!(!mask.is_active(0));
        assert!(!mask.is_active(1));
        assert_eq!(mask.active_count(), 0);
    }

    // ---- dynamic_ratio tests ----

    #[test]
    fn test_dynamic_ratio_empty_scene() {
        let scene = LayeredScene::new();
        assert!((scene.dynamic_ratio() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dynamic_ratio_all_static() {
        let mut scene = LayeredScene::new();
        scene.classify_and_add(&[make_static_gaussian(0)], make_coord());
        assert!((scene.dynamic_ratio() - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dynamic_ratio_all_dynamic() {
        let mut scene = LayeredScene::new();
        scene.classify_and_add(
            &[make_dynamic_gaussian_velocity(0)],
            make_coord(),
        );
        assert!((scene.dynamic_ratio() - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dynamic_ratio_mixed() {
        let mut scene = LayeredScene::new();
        let gs = vec![
            make_static_gaussian(0),
            make_static_gaussian(1),
            make_static_gaussian(2),
            make_dynamic_gaussian_velocity(3),
        ];
        scene.classify_and_add(&gs, make_coord());
        // 1 dynamic out of 4 total = 0.25
        assert!((scene.dynamic_ratio() - 0.25).abs() < f32::EPSILON);
    }

    // ---- epoch tests ----

    #[test]
    fn test_epoch_starts_at_zero() {
        let scene = LayeredScene::new();
        assert_eq!(scene.epoch(), 0);
    }

    #[test]
    fn test_epoch_advancement() {
        let mut scene = LayeredScene::new();
        assert_eq!(scene.advance_epoch(), 1);
        assert_eq!(scene.advance_epoch(), 2);
        assert_eq!(scene.advance_epoch(), 3);
        assert_eq!(scene.epoch(), 3);
    }

    // ---- Default trait ----

    #[test]
    fn test_default_layered_scene() {
        let scene = LayeredScene::default();
        assert_eq!(scene.epoch(), 0);
        assert_eq!(scene.total_gaussians(), 0);
        assert_eq!(scene.static_layer.layer_type, LayerType::Static);
        assert_eq!(scene.dynamic_layer.layer_type, LayerType::Dynamic);
    }

    // ---- is_dynamic helper ----

    #[test]
    fn test_is_dynamic_zero_velocity_unbounded_time() {
        let g = make_static_gaussian(0);
        assert!(!is_dynamic(&g));
    }

    #[test]
    fn test_is_dynamic_nonzero_velocity() {
        let g = make_dynamic_gaussian_velocity(0);
        assert!(is_dynamic(&g));
    }

    #[test]
    fn test_is_dynamic_bounded_time_only() {
        let g = make_dynamic_gaussian_time_only(0);
        assert!(is_dynamic(&g));
    }

    // ---- Multi-tile tests ----

    #[test]
    fn test_multiple_classify_and_add_calls() {
        let mut scene = LayeredScene::new();

        let coord1 = TileCoord {
            x: 0,
            y: 0,
            z: 0,
            time_bucket: 0,
            lod: 0,
        };
        let coord2 = TileCoord {
            x: 1,
            y: 0,
            z: 0,
            time_bucket: 0,
            lod: 0,
        };

        scene.classify_and_add(
            &[make_static_gaussian(0), make_dynamic_gaussian_velocity(1)],
            coord1,
        );
        scene.classify_and_add(
            &[make_static_gaussian(2), make_static_gaussian(3)],
            coord2,
        );

        assert_eq!(scene.static_layer.gaussian_count(), 3);
        assert_eq!(scene.static_layer.tile_count(), 2);
        assert_eq!(scene.dynamic_layer.gaussian_count(), 1);
        assert_eq!(scene.dynamic_layer.tile_count(), 1);
        assert_eq!(scene.total_gaussians(), 4);
    }
}
