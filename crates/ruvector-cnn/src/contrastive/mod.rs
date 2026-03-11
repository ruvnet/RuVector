//! # Contrastive Learning Module
//!
//! This module provides contrastive learning primitives for self-supervised representation learning:
//!
//! - **InfoNCE Loss**: NT-Xent loss used in SimCLR and CLIP
//! - **Triplet Loss**: Classic metric learning loss with margin
//! - **Contrastive Augmentation**: SimCLR-style data augmentation pipeline
//!
//! ## Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                  Contrastive Learning                        │
//! └───────────────────────────────────────────────────────────────┘
//!                              │
//!         ┌────────────────────┼────────────────────┐
//!         ▼                    ▼                    ▼
//!    ┌─────────┐         ┌──────────┐        ┌─────────────┐
//!    │ InfoNCE │         │ Triplet  │        │ Augmentation│
//!    │  Loss   │         │  Loss    │        │   Pipeline  │
//!    └─────────┘         └──────────┘        └─────────────┘
//!    NT-Xent with        Margin-based         SimCLR-style
//!    temperature         metric learning      random crops,
//!    scaling                                  flips, jitter
//! ```
//!
//! ## Usage Example
//!
//! ```rust
//! use ruvector_cnn::contrastive::{InfoNCELoss, TripletLoss, ContrastiveAugmentation};
//!
//! // InfoNCE for self-supervised learning
//! let infonce = InfoNCELoss::new(0.07);
//!
//! // Triplet loss for metric learning
//! let triplet = TripletLoss::new(1.0);
//!
//! // Data augmentation pipeline
//! let augmentation = ContrastiveAugmentation::builder()
//!     .crop_scale(0.08, 1.0)
//!     .horizontal_flip_prob(0.5)
//!     .color_jitter(0.4, 0.4, 0.4, 0.1)
//!     .build();
//! ```

mod augmentation;
mod infonce;
mod triplet;

pub use augmentation::{
    AugmentationConfig, ContrastiveAugmentation, ContrastiveAugmentationBuilder,
};
pub use infonce::{InfoNCELoss, InfoNCEResult};
pub use triplet::{TripletDistance, TripletLoss, TripletResult};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = InfoNCELoss::new(0.07);
        let _ = TripletLoss::new(1.0);
        let _ = ContrastiveAugmentation::default();
    }
}
