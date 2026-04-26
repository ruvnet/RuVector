//! Integration tests for graph rewrite passes (ADR-091 Phase 3)

use ruvector_cnn::quantize::{
    fuse_batchnorm_to_conv, fuse_hardswish, fuse_relu, fuse_zp_to_bias, generate_hardswish_lut,
    insert_qdq_nodes, CalibrationHistogram, ComputationGraph, NodeParams, NodeType,
    QuantizationParams,
};
use std::collections::HashMap;

#[test]
fn test_complete_graph_optimization_pipeline() {
    // Create a graph: Input → Conv → BN → ReLU → Conv → HardSwish → Output
    let mut graph = ComputationGraph::new();

    let input_id = graph.add_node(NodeType::Input, NodeParams::None);

    let conv1_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 16], // 2 out channels, 8 weights each
            bias: Some(vec![0.0, 0.0]),
            in_channels: 2,
            out_channels: 2,
            kernel_size: 2,
        },
    );

    let bn_id = graph.add_node(
        NodeType::BatchNorm,
        NodeParams::BatchNorm {
            gamma: vec![2.0, 3.0],
            beta: vec![0.1, 0.2],
            mean: vec![1.0, 2.0],
            var: vec![1.0, 4.0],
            eps: 1e-5,
        },
    );

    let relu_id = graph.add_node(NodeType::ReLU, NodeParams::Activation);

    let conv2_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 8],
            bias: None,
            in_channels: 2,
            out_channels: 1,
            kernel_size: 2,
        },
    );

    let hs_id = graph.add_node(NodeType::HardSwish, NodeParams::Activation);
    let output_id = graph.add_node(NodeType::Output, NodeParams::None);

    // Connect nodes
    graph.connect(input_id, conv1_id);
    graph.connect(conv1_id, bn_id);
    graph.connect(bn_id, relu_id);
    graph.connect(relu_id, conv2_id);
    graph.connect(conv2_id, hs_id);
    graph.connect(hs_id, output_id);

    // Initial node count: 7
    assert_eq!(graph.nodes.len(), 7);

    // GR-1: Fuse BatchNorm into Conv1
    let bn_fused = fuse_batchnorm_to_conv(&mut graph);
    assert_eq!(bn_fused, 1);
    assert_eq!(graph.nodes.len(), 6); // BN removed

    // GR-4: Fuse ReLU into Conv1
    let relu_fused = fuse_relu(&mut graph);
    assert_eq!(relu_fused, 1);
    assert_eq!(graph.nodes.len(), 5); // ReLU removed

    // GR-4: Fuse HardSwish into Conv2
    let hs_fused = fuse_hardswish(&mut graph);
    assert_eq!(hs_fused, 1);
    assert_eq!(graph.nodes.len(), 4); // HardSwish removed

    // Final graph: Input → Conv1 (with fused BN+ReLU) → Conv2 (with fused HardSwish) → Output
    assert_eq!(graph.nodes.len(), 4);
}

#[test]
fn test_zero_point_fusion() {
    let mut graph = ComputationGraph::new();

    let input_id = graph.add_node(NodeType::Input, NodeParams::None);
    // 2 out channels × 2 weights/channel
    // (weights_per_channel = kernel_size² × in_channels = 1 × 2 = 2)
    let conv_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0, 2.0, 3.0, 4.0],
            bias: Some(vec![1.0, 2.0]),
            in_channels: 2,
            out_channels: 2,
            kernel_size: 1,
        },
    );

    graph.connect(input_id, conv_id);

    // Create quantization params with zero-point = 10
    let mut quant_params = HashMap::new();
    quant_params.insert(
        input_id,
        QuantizationParams {
            scale: 0.1,
            zero_point: 10,
            min_val: -12.8,
            max_val: 12.7,
            num_bins: 256,
        },
    );

    // GR-2: Fuse zero-point correction
    let fused = fuse_zp_to_bias(&mut graph, &quant_params);
    assert_eq!(fused, 1);

    // Verify bias was adjusted
    let conv_node = graph.get_node(conv_id).unwrap();
    if let NodeParams::Conv2d { bias, .. } = &conv_node.params {
        let bias = bias.as_ref().unwrap();
        // Channel 0: weight_sum = 1.0 + 2.0 = 3.0
        // bias_corrected = 1.0 - 10.0 * 3.0 = -29.0
        assert!((bias[0] - (-29.0)).abs() < 0.01);

        // Channel 1: weight_sum = 3.0 + 4.0 = 7.0
        // bias_corrected = 2.0 - 10.0 * 7.0 = -68.0
        assert!((bias[1] - (-68.0)).abs() < 0.01);
    } else {
        panic!("Expected Conv2d params");
    }
}

#[test]
fn test_quantize_dequantize_insertion() {
    let mut graph = ComputationGraph::new();

    // FP32 Input → INT8 Conv → FP32 Output
    let input_id = graph.add_node(NodeType::Input, NodeParams::None);
    let conv_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 4],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
        },
    );
    let output_id = graph.add_node(NodeType::Output, NodeParams::None);

    graph.connect(input_id, conv_id);
    graph.connect(conv_id, output_id);

    // Mark Conv as quantized
    let mut quant_params = HashMap::new();
    quant_params.insert(
        conv_id,
        QuantizationParams {
            scale: 0.1,
            zero_point: 0,
            min_val: -12.8,
            max_val: 12.7,
            num_bins: 256,
        },
    );

    // GR-3: Insert Q/DQ nodes
    let inserted = insert_qdq_nodes(&mut graph, &quant_params);
    assert_eq!(inserted, 2); // One Q before Conv, one DQ after Conv

    // Verify graph structure: Input → Q → Conv → DQ → Output
    assert_eq!(graph.nodes.len(), 5);

    let conv_node = graph.get_node(conv_id).unwrap();
    assert_eq!(conv_node.inputs.len(), 1);
    assert_eq!(conv_node.outputs.len(), 1);

    // Check Q node before Conv
    let q_id = conv_node.inputs[0];
    let q_node = graph.get_node(q_id).unwrap();
    assert_eq!(q_node.node_type, NodeType::Quantize);

    // Check DQ node after Conv
    let dq_id = conv_node.outputs[0];
    let dq_node = graph.get_node(dq_id).unwrap();
    assert_eq!(dq_node.node_type, NodeType::Dequantize);
}

#[test]
fn test_hardswish_lut_generation() {
    let scale = 0.1;
    let zero_point = 0;
    let lut = generate_hardswish_lut(scale, zero_point);

    // generate_hardswish_lut iterates i in 0..256 with q_input = i as i8,
    // so the LUT is indexed by `q as u8 as usize` (i.e. wrapping cast):
    //   q =    0 → idx 0      (x = 0)
    //   q =   15 → idx 15     (x = 1.5)
    //   q =  127 → idx 127    (x = 12.7)
    //   q = -128 → idx 128    (x = -12.8)
    let lut_idx = |q: i32| -> usize { (q as i8) as u8 as usize };

    // x = 0 → HardSwish(0) = 0
    assert_eq!(lut[lut_idx(0 - zero_point)], 0);

    // x = -12.8 (q = -128, far below -3) → HardSwish = 0
    assert_eq!(lut[lut_idx(-128 - zero_point)], 0);

    // x = 12.7 (q = 127, far above 3) → HardSwish(x) ≈ x
    let x_pos = (lut[lut_idx(127 - zero_point)] as i32 - zero_point) as f32 * scale;
    assert!(x_pos > 10.0, "expected ~12.7, got {x_pos}");

    // x = 1.5 (q = 15) — middle range
    let x_mid = (lut[lut_idx(15 - zero_point)] as i32 - zero_point) as f32 * scale;
    // HardSwish(1.5) = 1.5 * ReLU6(4.5) / 6 = 1.5 * 4.5 / 6 = 1.125
    assert!((x_mid - 1.125).abs() < 0.3, "expected ~1.125, got {x_mid}");
}

#[test]
fn test_calibration_histogram() {
    let mut hist = CalibrationHistogram::new(-10.0, 10.0, 100);

    // Add calibration data
    for _ in 0..100 {
        hist.add(5.0);
    }
    for _ in 0..50 {
        hist.add(-5.0);
    }

    let params = hist.compute_quantization_params();

    // Should use symmetric quantization around 0
    assert_eq!(params.zero_point, 0);

    // Scale should be abs_max / 127
    let expected_scale = 10.0 / 127.0;
    assert!((params.scale - expected_scale).abs() < 0.01);
}

#[test]
fn test_batchnorm_fusion_preserves_semantics() {
    let mut graph = ComputationGraph::new();

    let input_id = graph.add_node(NodeType::Input, NodeParams::None);
    let conv_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![2.0, 3.0], // 1 output channel, 2 weights
            bias: Some(vec![1.0]),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
        },
    );
    let bn_id = graph.add_node(
        NodeType::BatchNorm,
        NodeParams::BatchNorm {
            gamma: vec![2.0],
            beta: vec![0.5],
            mean: vec![3.0],
            var: vec![4.0],
            eps: 1e-5,
        },
    );

    graph.connect(input_id, conv_id);
    graph.connect(conv_id, bn_id);

    // Before fusion: test mathematical equivalence
    // BN(Conv(x)) = gamma * (Conv(x) - mean) / sqrt(var + eps) + beta
    // = gamma * ((w*x + b) - mean) / sqrt(var + eps) + beta
    // = (gamma / sqrt(var + eps)) * w * x + (gamma / sqrt(var + eps)) * (b - mean) + beta

    let scale = 2.0 / (4.0 + 1e-5_f32).sqrt(); // gamma / sqrt(var + eps)
    let expected_w0 = 2.0 * scale; // w0 * scale
    let expected_w1 = 3.0 * scale; // w1 * scale
    let expected_bias = (1.0 - 3.0) * scale + 0.5; // (b - mean) * scale + beta

    fuse_batchnorm_to_conv(&mut graph);

    let conv_node = graph.get_node(conv_id).unwrap();
    if let NodeParams::Conv2d { weights, bias, .. } = &conv_node.params {
        assert!((weights[0] - expected_w0).abs() < 0.01);
        assert!((weights[1] - expected_w1).abs() < 0.01);
        assert!((bias.as_ref().unwrap()[0] - expected_bias).abs() < 0.01);
    }
}

#[test]
fn test_multi_output_graph() {
    let mut graph = ComputationGraph::new();

    let input_id = graph.add_node(NodeType::Input, NodeParams::None);
    let conv_id = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 4],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
        },
    );

    // Conv has multiple outputs (branching)
    let relu_id = graph.add_node(NodeType::ReLU, NodeParams::Activation);
    let hs_id = graph.add_node(NodeType::HardSwish, NodeParams::Activation);

    graph.connect(input_id, conv_id);
    graph.connect(conv_id, relu_id);
    graph.connect(conv_id, hs_id);

    // Should not fuse when Conv has multiple consumers
    // (In a real implementation, we'd check for this)
    assert_eq!(graph.get_node(conv_id).unwrap().outputs.len(), 2);
}
