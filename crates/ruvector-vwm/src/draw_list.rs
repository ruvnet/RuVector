//! Packed draw list protocol for GPU submission.
//!
//! A [`DrawList`] is a sequence of [`DrawCommand`]s that describe how to render
//! one frame. Commands bind tiles, set per-screen-tile budgets, and issue draw
//! calls for primitive blocks. The draw list can be serialized to bytes for GPU
//! upload or network transport.

use crate::tile::QuantTier;

/// Draw list header with epoch, sequence, and integrity metadata.
#[derive(Clone, Debug)]
pub struct DrawListHeader {
    /// Monotonically increasing epoch (world-model version).
    pub epoch: u64,
    /// Frame sequence number within this epoch.
    pub sequence: u32,
    /// Identifier of the budget profile used to build this list.
    pub budget_profile_id: u32,
    /// Checksum over the command stream (set by [`DrawList::finalize`]).
    pub checksum: u32,
}

/// Individual draw commands within a draw list.
#[derive(Clone, Debug)]
pub enum DrawCommand {
    /// Bind a tile's primitive block for subsequent draw calls.
    TileBind {
        tile_id: u64,
        block_ref: u32,
        quant_tier: QuantTier,
    },
    /// Set the rendering budget for a screen tile.
    SetBudget {
        screen_tile_id: u32,
        max_gaussians: u32,
        max_overdraw: f32,
    },
    /// Issue a draw call for a bound primitive block.
    DrawBlock {
        block_ref: u32,
        sort_key: f32,
        opacity_mode: OpacityMode,
    },
    /// Sentinel marking the end of the command stream.
    End,
}

/// Blending mode for Gaussian rasterization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OpacityMode {
    /// Standard alpha blending (front-to-back or back-to-front).
    AlphaBlend,
    /// Additive blending for emissive/glow effects.
    Additive,
    /// Opaque pass (no blending).
    Opaque,
}

/// A complete packed draw list for one frame.
#[derive(Clone, Debug)]
pub struct DrawList {
    pub header: DrawListHeader,
    pub commands: Vec<DrawCommand>,
}

impl DrawList {
    /// Create a new empty draw list.
    pub fn new(epoch: u64, sequence: u32, budget_profile_id: u32) -> Self {
        Self {
            header: DrawListHeader {
                epoch,
                sequence,
                budget_profile_id,
                checksum: 0,
            },
            commands: Vec::new(),
        }
    }

    /// Append a tile-bind command.
    pub fn bind_tile(&mut self, tile_id: u64, block_ref: u32, quant_tier: QuantTier) {
        self.commands.push(DrawCommand::TileBind {
            tile_id,
            block_ref,
            quant_tier,
        });
    }

    /// Append a budget-set command for a screen tile.
    pub fn set_budget(&mut self, screen_tile_id: u32, max_gaussians: u32, max_overdraw: f32) {
        self.commands.push(DrawCommand::SetBudget {
            screen_tile_id,
            max_gaussians,
            max_overdraw,
        });
    }

    /// Append a draw-block command.
    pub fn draw_block(&mut self, block_ref: u32, sort_key: f32, opacity_mode: OpacityMode) {
        self.commands.push(DrawCommand::DrawBlock {
            block_ref,
            sort_key,
            opacity_mode,
        });
    }

    /// Finalize the draw list by appending an `End` command and computing the checksum.
    ///
    /// Returns the computed checksum.
    pub fn finalize(&mut self) -> u32 {
        // Remove any existing End commands
        self.commands.retain(|c| !matches!(c, DrawCommand::End));
        self.commands.push(DrawCommand::End);

        let bytes = self.serialize_commands();
        let checksum = fnv1a_checksum(&bytes);
        self.header.checksum = checksum;
        checksum
    }

    /// Return the number of commands (excluding End).
    pub fn command_count(&self) -> usize {
        self.commands
            .iter()
            .filter(|c| !matches!(c, DrawCommand::End))
            .count()
    }

    /// Serialize the entire draw list to bytes for GPU upload or network transport.
    ///
    /// Wire format (all little-endian):
    /// - Header: epoch(8) + sequence(4) + budget_profile_id(4) + checksum(4) = 20 bytes
    /// - For each command:
    ///   - tag(1 byte): 0=TileBind, 1=SetBudget, 2=DrawBlock, 3=End
    ///   - payload (varies by tag)
    pub fn to_bytes(&self) -> Vec<u8> {
        // Header = 20 bytes, plus command payload
        let mut buf = Vec::with_capacity(20 + self.commands.len() * 14);

        // Header
        buf.extend_from_slice(&self.header.epoch.to_le_bytes());
        buf.extend_from_slice(&self.header.sequence.to_le_bytes());
        buf.extend_from_slice(&self.header.budget_profile_id.to_le_bytes());
        buf.extend_from_slice(&self.header.checksum.to_le_bytes());

        // Commands
        buf.extend_from_slice(&self.serialize_commands());

        buf
    }

    /// Serialize only the command portion (used internally for checksumming).
    fn serialize_commands(&self) -> Vec<u8> {
        // Max command payload: TileBind = 1+8+4+1 = 14 bytes
        let mut buf = Vec::with_capacity(self.commands.len() * 14);
        for cmd in &self.commands {
            match cmd {
                DrawCommand::TileBind {
                    tile_id,
                    block_ref,
                    quant_tier,
                } => {
                    buf.push(0u8);
                    buf.extend_from_slice(&tile_id.to_le_bytes());
                    buf.extend_from_slice(&block_ref.to_le_bytes());
                    buf.push(quant_tier_to_byte(*quant_tier));
                }
                DrawCommand::SetBudget {
                    screen_tile_id,
                    max_gaussians,
                    max_overdraw,
                } => {
                    buf.push(1u8);
                    buf.extend_from_slice(&screen_tile_id.to_le_bytes());
                    buf.extend_from_slice(&max_gaussians.to_le_bytes());
                    buf.extend_from_slice(&max_overdraw.to_le_bytes());
                }
                DrawCommand::DrawBlock {
                    block_ref,
                    sort_key,
                    opacity_mode,
                } => {
                    buf.push(2u8);
                    buf.extend_from_slice(&block_ref.to_le_bytes());
                    buf.extend_from_slice(&sort_key.to_le_bytes());
                    buf.push(opacity_mode_to_byte(*opacity_mode));
                }
                DrawCommand::End => {
                    buf.push(3u8);
                }
            }
        }
        buf
    }
}

fn quant_tier_to_byte(tier: QuantTier) -> u8 {
    match tier {
        QuantTier::Hot8 => 0,
        QuantTier::Warm7 => 1,
        QuantTier::Warm5 => 2,
        QuantTier::Cold3 => 3,
    }
}

fn opacity_mode_to_byte(mode: OpacityMode) -> u8 {
    match mode {
        OpacityMode::AlphaBlend => 0,
        OpacityMode::Additive => 1,
        OpacityMode::Opaque => 2,
    }
}

/// FNV-1a hash for checksumming.
fn fnv1a_checksum(data: &[u8]) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for &byte in data {
        hash ^= byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_draw_list() {
        let mut dl = DrawList::new(1, 0, 100);
        assert_eq!(dl.command_count(), 0);
        let checksum = dl.finalize();
        assert_ne!(checksum, 0);
        // After finalize there should be an End command
        assert!(matches!(dl.commands.last(), Some(DrawCommand::End)));
    }

    #[test]
    fn test_draw_list_commands() {
        let mut dl = DrawList::new(1, 0, 0);
        dl.bind_tile(42, 1, QuantTier::Hot8);
        dl.set_budget(0, 1000, 2.0);
        dl.draw_block(1, 0.5, OpacityMode::AlphaBlend);
        assert_eq!(dl.command_count(), 3);
        dl.finalize();
        // command_count excludes End
        assert_eq!(dl.command_count(), 3);
    }

    #[test]
    fn test_to_bytes_not_empty() {
        let mut dl = DrawList::new(1, 0, 0);
        dl.bind_tile(1, 0, QuantTier::Cold3);
        dl.finalize();
        let bytes = dl.to_bytes();
        // header = 20 bytes, at least some command bytes
        assert!(bytes.len() > 20);
    }

    #[test]
    fn test_finalize_idempotent() {
        let mut dl = DrawList::new(1, 0, 0);
        dl.draw_block(0, 1.0, OpacityMode::Opaque);
        let c1 = dl.finalize();
        let c2 = dl.finalize();
        assert_eq!(c1, c2);
        // Should only have one End
        let end_count = dl
            .commands
            .iter()
            .filter(|c| matches!(c, DrawCommand::End))
            .count();
        assert_eq!(end_count, 1);
    }
}
