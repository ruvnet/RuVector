//! INT8 Quantization Module (ADR-091)
//!
//! This module provides comprehensive INT8 quantization support for CNN models:
//! - **Phase 1** (params, tensor): Core quantization infrastructure (NEW)
//! - **Phase 2** (calibration): Histogram-based range estimation
//! - **Phase 3** (graph_rewrite): BatchNorm fusion, zero-point optimization, Q/DQ insertion
//! - **Phase 4**: Kernel Dispatch - Runtime selection of optimized INT8 kernels
//!
//! ## ADR-091 Phase 1 Components (New)
//!
//! - `params`: Quantization parameters (scale, zero_point, qmin, qmax)
//! - `tensor`: Quantized tensor types with metadata
//! - Enhanced `calibration`: CalibrationCollector with MinMax, Percentile, MSE, Entropy methods

// ADR-091 Phase 1: Core infrastructure (NEW)
pub mod params;
pub mod tensor;

// Existing implementation (Phase 2-3)
pub mod calibration;
pub mod graph_rewrite;

// Phase 1 exports
pub use params::{QuantizationMode, QuantizationParams as QuantParams, QuantizationScheme};
pub use tensor::{QuantizationMetadata, QuantizedTensor};

// Existing exports (kept for backward compatibility)
pub use calibration::{CalibrationHistogram, QuantizationParams, Quantizer};
pub use graph_rewrite::{
    fuse_batchnorm_to_conv, fuse_hardswish, fuse_relu, fuse_zp_to_bias, generate_hardswish_lut,
    insert_qdq_nodes, ComputationGraph, GraphNode, NodeParams, NodeType,
};
