//! Graph Rewrite Passes Demo (ADR-091 Phase 3)
//!
//! This example demonstrates all four graph rewrite passes:
//! - GR-1: BatchNorm fusion into Conv
//! - GR-2: Zero-point correction fusion
//! - GR-3: Q/DQ node insertion
//! - GR-4: Activation fusion (ReLU, HardSwish)

use ruvector_cnn::quantize::calibration::QuantizationParams;
use ruvector_cnn::quantize::graph_rewrite::*;
use std::collections::HashMap;

fn main() {
    println!("=== ADR-091 Phase 3: Graph Rewrite Passes Demo ===\n");

    demo_batchnorm_fusion();
    demo_zero_point_fusion();
    demo_qdq_insertion();
    demo_activation_fusion();
    demo_complete_pipeline();
}

fn demo_batchnorm_fusion() {
    println!("--- GR-1: BatchNorm Fusion ---");

    let mut graph = ComputationGraph::new();

    let input = graph.add_node(NodeType::Input, NodeParams::None);
    let conv = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0, 2.0, 3.0, 4.0],
            bias: Some(vec![0.5, 1.0]),
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
        },
    );
    let bn = graph.add_node(
        NodeType::BatchNorm,
        NodeParams::BatchNorm {
            gamma: vec![2.0, 3.0],
            beta: vec![0.1, 0.2],
            mean: vec![0.5, 1.0],
            var: vec![1.0, 4.0],
            eps: 1e-5,
        },
    );

    graph.connect(input, conv);
    graph.connect(conv, bn);

    println!("Before fusion: {} nodes", graph.nodes.len());

    let fused = fuse_batchnorm_to_conv(&mut graph);
    println!(
        "After fusion: {} nodes (fused {} BatchNorm layers)",
        graph.nodes.len(),
        fused
    );

    if let Some(conv_node) = graph.get_node(conv) {
        if let NodeParams::Conv2d { weights, bias, .. } = &conv_node.params {
            println!("Fused weights: {:?}", &weights[0..2]);
            println!("Fused bias: {:?}", bias.as_ref().unwrap());
        }
    }
    println!();
}

fn demo_zero_point_fusion() {
    println!("--- GR-2: Zero-Point Correction Fusion ---");

    let mut graph = ComputationGraph::new();

    let input = graph.add_node(NodeType::Input, NodeParams::None);
    let conv = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0, 2.0, 3.0, 4.0],
            bias: Some(vec![1.0, 2.0]),
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
        },
    );

    graph.connect(input, conv);

    let mut quant_params = HashMap::new();
    quant_params.insert(
        input,
        QuantizationParams {
            scale: 0.1,
            zero_point: 10,
            min_val: -12.8,
            max_val: 12.7,
            num_bins: 256,
        },
    );

    println!("Before fusion:");
    if let Some(conv_node) = graph.get_node(conv) {
        if let NodeParams::Conv2d { bias, .. } = &conv_node.params {
            println!("  Original bias: {:?}", bias.as_ref().unwrap());
        }
    }

    let fused = fuse_zp_to_bias(&mut graph, &quant_params);
    println!("After fusion ({} corrections):", fused);
    if let Some(conv_node) = graph.get_node(conv) {
        if let NodeParams::Conv2d { bias, .. } = &conv_node.params {
            println!("  Corrected bias: {:?}", bias.as_ref().unwrap());
            println!("  (Pre-computed zero-point correction: zp_input × Σweights)");
        }
    }
    println!();
}

fn demo_qdq_insertion() {
    println!("--- GR-3: Quantize/Dequantize Node Insertion ---");

    let mut graph = ComputationGraph::new();

    let input = graph.add_node(NodeType::Input, NodeParams::None);
    let conv = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 4],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
        },
    );
    let output = graph.add_node(NodeType::Output, NodeParams::None);

    graph.connect(input, conv);
    graph.connect(conv, output);

    let mut quant_params = HashMap::new();
    quant_params.insert(
        conv,
        QuantizationParams {
            scale: 0.1,
            zero_point: 0,
            min_val: -12.8,
            max_val: 12.7,
            num_bins: 256,
        },
    );

    println!("Before Q/DQ insertion: {} nodes", graph.nodes.len());
    println!("Graph: Input → Conv → Output");

    let inserted = insert_qdq_nodes(&mut graph, &quant_params);
    println!(
        "After Q/DQ insertion: {} nodes ({} Q/DQ nodes added)",
        graph.nodes.len(),
        inserted
    );
    println!("Graph: Input → Quantize → Conv → Dequantize → Output");
    println!();
}

fn demo_activation_fusion() {
    println!("--- GR-4: Activation Fusion ---");

    let mut graph1 = ComputationGraph::new();
    let conv1 = graph1.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 4],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
        },
    );
    let relu = graph1.add_node(NodeType::ReLU, NodeParams::Activation);
    graph1.connect(conv1, relu);

    println!("ReLU Fusion:");
    println!("  Before: {} nodes (Conv → ReLU)", graph1.nodes.len());
    let fused_relu = fuse_relu(&mut graph1);
    println!(
        "  After: {} nodes (Conv with fused ReLU, {} activations fused)",
        graph1.nodes.len(),
        fused_relu
    );

    let mut graph2 = ComputationGraph::new();
    let conv2 = graph2.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 4],
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
        },
    );
    let hs = graph2.add_node(NodeType::HardSwish, NodeParams::Activation);
    graph2.connect(conv2, hs);

    println!("\nHardSwish Fusion:");
    println!("  Before: {} nodes (Conv → HardSwish)", graph2.nodes.len());
    let fused_hs = fuse_hardswish(&mut graph2);
    println!(
        "  After: {} nodes (Conv with LUT-based HardSwish, {} activations fused)",
        graph2.nodes.len(),
        fused_hs
    );

    // Generate HardSwish LUT
    let lut = generate_hardswish_lut(0.1, 0);
    println!("\n  Generated 256-entry HardSwish LUT (i8→i8):");
    println!("    LUT[0] (x=-12.8): {}", lut[0]);
    println!("    LUT[128] (x=0): {}", lut[128]);
    println!("    LUT[255] (x=12.7): {}", lut[255]);
    println!();
}

fn demo_complete_pipeline() {
    println!("--- Complete Optimization Pipeline ---");

    let mut graph = ComputationGraph::new();

    let input = graph.add_node(NodeType::Input, NodeParams::None);
    let conv1 = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 16],
            bias: Some(vec![0.0, 0.0]),
            in_channels: 2,
            out_channels: 2,
            kernel_size: 2,
        },
    );
    let bn = graph.add_node(
        NodeType::BatchNorm,
        NodeParams::BatchNorm {
            gamma: vec![2.0, 3.0],
            beta: vec![0.1, 0.2],
            mean: vec![1.0, 2.0],
            var: vec![1.0, 4.0],
            eps: 1e-5,
        },
    );
    let relu = graph.add_node(NodeType::ReLU, NodeParams::Activation);
    let conv2 = graph.add_node(
        NodeType::Conv2d,
        NodeParams::Conv2d {
            weights: vec![1.0; 8],
            bias: None,
            in_channels: 2,
            out_channels: 1,
            kernel_size: 2,
        },
    );
    let hs = graph.add_node(NodeType::HardSwish, NodeParams::Activation);
    let output = graph.add_node(NodeType::Output, NodeParams::None);

    graph.connect(input, conv1);
    graph.connect(conv1, bn);
    graph.connect(bn, relu);
    graph.connect(relu, conv2);
    graph.connect(conv2, hs);
    graph.connect(hs, output);

    println!("Original graph: {} nodes", graph.nodes.len());
    println!("  Input → Conv1 → BN → ReLU → Conv2 → HardSwish → Output");

    // Apply optimization passes
    println!("\nApplying optimization passes:");

    let bn_fused = fuse_batchnorm_to_conv(&mut graph);
    println!(
        "  ✓ GR-1: Fused {} BatchNorm layers → {} nodes",
        bn_fused,
        graph.nodes.len()
    );

    let relu_fused = fuse_relu(&mut graph);
    println!(
        "  ✓ GR-4: Fused {} ReLU activations → {} nodes",
        relu_fused,
        graph.nodes.len()
    );

    let hs_fused = fuse_hardswish(&mut graph);
    println!(
        "  ✓ GR-4: Fused {} HardSwish activations → {} nodes",
        hs_fused,
        graph.nodes.len()
    );

    println!("\nOptimized graph: {} nodes", graph.nodes.len());
    println!("  Input → Conv1(+BN+ReLU) → Conv2(+HardSwish) → Output");
    println!(
        "\nMemory savings: {} nodes eliminated",
        7 - graph.nodes.len()
    );
    println!("Runtime benefit: 3 fewer ops, fused activations");
}
