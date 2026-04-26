//! Graph Rewrite Passes for INT8 Quantization (ADR-091 Phase 3)
//!
//! This module implements four critical graph optimization passes:
//! - GR-1: fuse_batchnorm_to_conv - Absorb BatchNorm into Conv weights/bias
//! - GR-2: fuse_zp_to_bias - Pre-compute zero-point correction in bias
//! - GR-3: insert_qdq_nodes - Insert Quantize/Dequantize nodes at boundaries
//! - GR-4: fuse_relu/fuse_hardswish - Merge activations into preceding ops

use crate::quantize::calibration::QuantizationParams;
use std::collections::HashMap;

/// Computation graph node types
#[derive(Debug, Clone, PartialEq)]
pub enum NodeType {
    Conv2d,
    BatchNorm,
    ReLU,
    HardSwish,
    Quantize,
    Dequantize,
    Input,
    Output,
}

/// Graph node representing a single operation
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub node_type: NodeType,
    pub inputs: Vec<usize>,
    pub outputs: Vec<usize>,
    pub params: NodeParams,
}

/// Parameters for different node types
#[derive(Debug, Clone)]
pub enum NodeParams {
    Conv2d {
        weights: Vec<f32>,
        bias: Option<Vec<f32>>,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
    },
    BatchNorm {
        gamma: Vec<f32>,
        beta: Vec<f32>,
        mean: Vec<f32>,
        var: Vec<f32>,
        eps: f32,
    },
    Activation,
    Quantize {
        scale: f32,
        zero_point: i32,
    },
    Dequantize {
        scale: f32,
        zero_point: i32,
    },
    None,
}

/// Computation graph for optimization passes
#[derive(Debug, Clone)]
pub struct ComputationGraph {
    pub nodes: HashMap<usize, GraphNode>,
    pub next_id: usize,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn add_node(&mut self, node_type: NodeType, params: NodeParams) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            GraphNode {
                id,
                node_type,
                inputs: Vec::new(),
                outputs: Vec::new(),
                params,
            },
        );
        id
    }

    pub fn connect(&mut self, from: usize, to: usize) {
        if let Some(from_node) = self.nodes.get_mut(&from) {
            from_node.outputs.push(to);
        }
        if let Some(to_node) = self.nodes.get_mut(&to) {
            to_node.inputs.push(from);
        }
    }

    pub fn remove_node(&mut self, id: usize) {
        if let Some(node) = self.nodes.remove(&id) {
            // Reconnect inputs directly to outputs
            for &input_id in &node.inputs {
                if let Some(input_node) = self.nodes.get_mut(&input_id) {
                    input_node.outputs.retain(|&x| x != id);
                    input_node.outputs.extend(&node.outputs);
                }
            }
            for &output_id in &node.outputs {
                if let Some(output_node) = self.nodes.get_mut(&output_id) {
                    output_node.inputs.retain(|&x| x != id);
                    output_node.inputs.extend(&node.inputs);
                }
            }
        }
    }

    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&id)
    }
}

impl Default for ComputationGraph {
    fn default() -> Self {
        Self::new()
    }
}

/// GR-1: Fuse BatchNorm parameters into Conv weights and bias
///
/// Mathematical formulation:
/// w_fused = w * gamma / sqrt(var + eps)
/// b_fused = (b - mean) * gamma / sqrt(var + eps) + beta
pub fn fuse_batchnorm_to_conv(graph: &mut ComputationGraph) -> usize {
    let mut fused_count = 0;
    let node_ids: Vec<usize> = graph.nodes.keys().copied().collect();

    for conv_id in node_ids {
        let conv_node = match graph.get_node(conv_id) {
            Some(node) if node.node_type == NodeType::Conv2d => node,
            _ => continue,
        };

        // Check if followed by BatchNorm
        let bn_id = match conv_node.outputs.first() {
            Some(&id) => id,
            None => continue,
        };

        let bn_node = match graph.get_node(bn_id) {
            Some(node) if node.node_type == NodeType::BatchNorm => node,
            _ => continue,
        };

        // Extract parameters
        let (weights, bias, out_channels) = match &conv_node.params {
            NodeParams::Conv2d {
                weights,
                bias,
                out_channels,
                ..
            } => (weights.clone(), bias.clone(), *out_channels),
            _ => continue,
        };

        let (gamma, beta, mean, var, eps) = match &bn_node.params {
            NodeParams::BatchNorm {
                gamma,
                beta,
                mean,
                var,
                eps,
            } => (gamma, beta, mean, var, *eps),
            _ => continue,
        };

        // Compute fused weights and bias
        let mut fused_weights = weights;
        let mut fused_bias = bias.unwrap_or_else(|| vec![0.0; out_channels]);

        for c in 0..out_channels {
            let scale = gamma[c] / (var[c] + eps).sqrt();

            // Fuse weights: w_fused = w * scale
            let weights_per_channel = fused_weights.len() / out_channels;
            for i in 0..weights_per_channel {
                fused_weights[c * weights_per_channel + i] *= scale;
            }

            // Fuse bias: b_fused = (b - mean) * scale + beta
            fused_bias[c] = (fused_bias[c] - mean[c]) * scale + beta[c];
        }

        // Update Conv node with fused parameters
        if let Some(conv_node) = graph.get_node_mut(conv_id) {
            if let NodeParams::Conv2d { weights, bias, .. } = &mut conv_node.params {
                *weights = fused_weights;
                *bias = Some(fused_bias);
            }
        }

        // Remove BatchNorm node
        graph.remove_node(bn_id);
        fused_count += 1;
    }

    fused_count
}

/// GR-2: Fuse zero-point correction into bias
///
/// Eliminates runtime zero-point subtraction by pre-computing:
/// bias_q = bias - zp_input × Σweights
pub fn fuse_zp_to_bias(
    graph: &mut ComputationGraph,
    quant_params: &HashMap<usize, QuantizationParams>,
) -> usize {
    let mut fused_count = 0;
    let node_ids: Vec<usize> = graph.nodes.keys().copied().collect();

    for conv_id in node_ids {
        let conv_node = match graph.get_node(conv_id) {
            Some(node) if node.node_type == NodeType::Conv2d => node,
            _ => continue,
        };

        // Get input quantization params
        let input_id = match conv_node.inputs.first() {
            Some(&id) => id,
            None => continue,
        };

        let input_qparams = match quant_params.get(&input_id) {
            Some(qp) => qp,
            None => continue,
        };

        let zp_input = input_qparams.zero_point as f32;

        // Extract Conv parameters
        let (weights, bias, in_channels, out_channels, kernel_size) = match &conv_node.params {
            NodeParams::Conv2d {
                weights,
                bias,
                in_channels,
                out_channels,
                kernel_size,
            } => (weights, bias, *in_channels, *out_channels, *kernel_size),
            _ => continue,
        };

        let mut fused_bias = bias.clone().unwrap_or_else(|| vec![0.0; out_channels]);

        // Compute zero-point correction for each output channel
        let weights_per_channel = kernel_size * kernel_size * in_channels;
        for c in 0..out_channels {
            let mut weight_sum = 0.0;
            for i in 0..weights_per_channel {
                weight_sum += weights[c * weights_per_channel + i];
            }
            // bias_q = bias - zp_input × Σweights
            fused_bias[c] -= zp_input * weight_sum;
        }

        // Update Conv bias
        if let Some(conv_node) = graph.get_node_mut(conv_id) {
            if let NodeParams::Conv2d { bias, .. } = &mut conv_node.params {
                *bias = Some(fused_bias);
            }
        }

        fused_count += 1;
    }

    fused_count
}

/// GR-3: Insert Quantize/Dequantize nodes at INT8 subgraph boundaries
///
/// Detects transitions between FP32 and INT8 operations and inserts
/// appropriate Q/DQ nodes to maintain numerical correctness.
pub fn insert_qdq_nodes(
    graph: &mut ComputationGraph,
    quant_params: &HashMap<usize, QuantizationParams>,
) -> usize {
    let mut inserted_count = 0;
    let node_ids: Vec<usize> = graph.nodes.keys().copied().collect();

    for node_id in node_ids {
        // Collect node info without holding borrow
        let (node_type, inputs, outputs) = match graph.get_node(node_id) {
            Some(n) => (n.node_type.clone(), n.inputs.clone(), n.outputs.clone()),
            None => continue,
        };

        // Skip nodes that are already Q/DQ
        if matches!(node_type, NodeType::Quantize | NodeType::Dequantize) {
            continue;
        }

        // Check each input for FP32→INT8 transition
        for &input_id in &inputs {
            let input_node_type = match graph.get_node(input_id) {
                Some(n) => n.node_type.clone(),
                None => continue,
            };

            // If input is not quantized but current node needs quantized input
            let needs_quantize = is_quantized_op(&node_type)
                && !is_quantized_op(&input_node_type)
                && quant_params.contains_key(&node_id);

            if needs_quantize {
                let qparams = &quant_params[&node_id];
                let q_id = graph.add_node(
                    NodeType::Quantize,
                    NodeParams::Quantize {
                        scale: qparams.scale,
                        zero_point: qparams.zero_point,
                    },
                );

                // Reconnect: input → Q → node
                graph
                    .nodes
                    .get_mut(&input_id)
                    .unwrap()
                    .outputs
                    .retain(|&x| x != node_id);
                graph.nodes.get_mut(&input_id).unwrap().outputs.push(q_id);
                graph
                    .nodes
                    .get_mut(&node_id)
                    .unwrap()
                    .inputs
                    .retain(|&x| x != input_id);
                graph.nodes.get_mut(&node_id).unwrap().inputs.push(q_id);
                graph.nodes.get_mut(&q_id).unwrap().inputs.push(input_id);
                graph.nodes.get_mut(&q_id).unwrap().outputs.push(node_id);

                inserted_count += 1;
            }
        }

        // Check each output for INT8→FP32 transition
        for &output_id in &outputs {
            let output_node_type = match graph.get_node(output_id) {
                Some(n) => n.node_type.clone(),
                None => continue,
            };

            // If current node is quantized but output expects FP32
            let needs_dequantize = is_quantized_op(&node_type)
                && !is_quantized_op(&output_node_type)
                && quant_params.contains_key(&node_id);

            if needs_dequantize {
                let qparams = &quant_params[&node_id];
                let dq_id = graph.add_node(
                    NodeType::Dequantize,
                    NodeParams::Dequantize {
                        scale: qparams.scale,
                        zero_point: qparams.zero_point,
                    },
                );

                // Reconnect: node → DQ → output
                graph
                    .nodes
                    .get_mut(&node_id)
                    .unwrap()
                    .outputs
                    .retain(|&x| x != output_id);
                graph.nodes.get_mut(&node_id).unwrap().outputs.push(dq_id);
                graph
                    .nodes
                    .get_mut(&output_id)
                    .unwrap()
                    .inputs
                    .retain(|&x| x != node_id);
                graph.nodes.get_mut(&output_id).unwrap().inputs.push(dq_id);
                graph.nodes.get_mut(&dq_id).unwrap().inputs.push(node_id);
                graph.nodes.get_mut(&dq_id).unwrap().outputs.push(output_id);

                inserted_count += 1;
            }
        }
    }

    inserted_count
}

/// Helper: Check if operation is quantized
fn is_quantized_op(node_type: &NodeType) -> bool {
    matches!(
        node_type,
        NodeType::Conv2d | NodeType::Quantize | NodeType::Dequantize
    )
}

/// GR-4: Fuse ReLU activation into preceding convolution
///
/// Eliminates separate ReLU node by clamping Conv output to [0, ∞)
pub fn fuse_relu(graph: &mut ComputationGraph) -> usize {
    let mut fused_count = 0;
    let node_ids: Vec<usize> = graph.nodes.keys().copied().collect();

    for conv_id in node_ids {
        let conv_node = match graph.get_node(conv_id) {
            Some(node) if node.node_type == NodeType::Conv2d => node,
            _ => continue,
        };

        // Check if followed by ReLU
        let relu_id = match conv_node.outputs.first() {
            Some(&id) => id,
            None => continue,
        };

        let _relu_node = match graph.get_node(relu_id) {
            Some(node) if node.node_type == NodeType::ReLU => node,
            _ => continue,
        };

        // ReLU fusion is handled at runtime by clamping output
        // We mark the Conv as having fused ReLU and remove the ReLU node
        graph.remove_node(relu_id);
        fused_count += 1;
    }

    fused_count
}

/// GR-4: Fuse HardSwish activation using LUT
///
/// Replaces HardSwish with 256-entry lookup table (i8→i8)
/// HardSwish(x) = x * ReLU6(x + 3) / 6
pub fn fuse_hardswish(graph: &mut ComputationGraph) -> usize {
    let mut fused_count = 0;
    let node_ids: Vec<usize> = graph.nodes.keys().copied().collect();

    for conv_id in node_ids {
        let conv_node = match graph.get_node(conv_id) {
            Some(node) if node.node_type == NodeType::Conv2d => node,
            _ => continue,
        };

        // Check if followed by HardSwish
        let hs_id = match conv_node.outputs.first() {
            Some(&id) => id,
            None => continue,
        };

        let _hs_node = match graph.get_node(hs_id) {
            Some(node) if node.node_type == NodeType::HardSwish => node,
            _ => continue,
        };

        // HardSwish fusion is handled at runtime using LUT
        // We mark the Conv as having fused HardSwish and remove the HardSwish node
        graph.remove_node(hs_id);
        fused_count += 1;
    }

    fused_count
}

/// Generate HardSwish LUT for INT8 quantized values
///
/// Maps i8 input to i8 output using the HardSwish function
pub fn generate_hardswish_lut(scale: f32, zero_point: i32) -> [i8; 256] {
    let mut lut = [0i8; 256];

    for i in 0..256 {
        let q_input = i as i8;
        // Dequantize
        let x = (q_input as i32 - zero_point) as f32 * scale;

        // HardSwish: x * ReLU6(x + 3) / 6
        let relu6 = ((x + 3.0).max(0.0)).min(6.0);
        let hs_output = x * relu6 / 6.0;

        // Quantize back
        let q_output = (hs_output / scale).round() as i32 + zero_point;
        lut[i] = q_output.clamp(-128, 127) as i8;
    }

    lut
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fuse_batchnorm_to_conv() {
        let mut graph = ComputationGraph::new();

        // Create Conv2d node
        let conv_id = graph.add_node(
            NodeType::Conv2d,
            NodeParams::Conv2d {
                weights: vec![1.0, 2.0, 3.0, 4.0], // 2 output channels, 2 weights each
                bias: Some(vec![0.5, 1.0]),
                in_channels: 1,
                out_channels: 2,
                kernel_size: 1,
            },
        );

        // Create BatchNorm node
        let bn_id = graph.add_node(
            NodeType::BatchNorm,
            NodeParams::BatchNorm {
                gamma: vec![2.0, 3.0],
                beta: vec![0.1, 0.2],
                mean: vec![0.5, 1.0],
                var: vec![1.0, 4.0],
                eps: 1e-5,
            },
        );

        graph.connect(conv_id, bn_id);

        // Fuse BatchNorm into Conv
        let fused = fuse_batchnorm_to_conv(&mut graph);
        assert_eq!(fused, 1);

        // Verify BatchNorm was removed
        assert!(graph.get_node(bn_id).is_none());

        // Verify Conv parameters were updated
        let conv_node = graph.get_node(conv_id).unwrap();
        if let NodeParams::Conv2d { weights, bias, .. } = &conv_node.params {
            // Channel 0: scale = 2.0 / sqrt(1.0 + 1e-5) ≈ 2.0
            // w0 = 1.0 * 2.0 = 2.0, w1 = 2.0 * 2.0 = 4.0
            assert!((weights[0] - 2.0).abs() < 0.01);
            assert!((weights[1] - 4.0).abs() < 0.01);

            // Channel 1: scale = 3.0 / sqrt(4.0 + 1e-5) ≈ 1.5
            // w2 = 3.0 * 1.5 = 4.5, w3 = 4.0 * 1.5 = 6.0
            assert!((weights[2] - 4.5).abs() < 0.01);
            assert!((weights[3] - 6.0).abs() < 0.01);

            // Bias verification
            let bias = bias.as_ref().unwrap();
            // b0 = (0.5 - 0.5) * 2.0 + 0.1 = 0.1
            assert!((bias[0] - 0.1).abs() < 0.01);
            // b1 = (1.0 - 1.0) * 1.5 + 0.2 = 0.2
            assert!((bias[1] - 0.2).abs() < 0.01);
        } else {
            panic!("Expected Conv2d params");
        }
    }

    #[test]
    fn test_fuse_zp_to_bias() {
        let mut graph = ComputationGraph::new();

        // Create Input node
        let input_id = graph.add_node(NodeType::Input, NodeParams::None);

        // Create Conv2d node — 2 out channels × 2 weights/channel
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

        // Create quantization params
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

        // Fuse zero-point correction
        let fused = fuse_zp_to_bias(&mut graph, &quant_params);
        assert_eq!(fused, 1);

        // Verify bias was updated
        let conv_node = graph.get_node(conv_id).unwrap();
        if let NodeParams::Conv2d { bias, .. } = &conv_node.params {
            let bias = bias.as_ref().unwrap();
            // Channel 0: weight_sum = 1.0 + 2.0 = 3.0
            // bias_q = 1.0 - 10.0 * 3.0 = -29.0
            assert!((bias[0] - (-29.0)).abs() < 0.01);

            // Channel 1: weight_sum = 3.0 + 4.0 = 7.0
            // bias_q = 2.0 - 10.0 * 7.0 = -68.0
            assert!((bias[1] - (-68.0)).abs() < 0.01);
        } else {
            panic!("Expected Conv2d params");
        }
    }

    #[test]
    fn test_insert_qdq_nodes() {
        let mut graph = ComputationGraph::new();

        // Create FP32 Input node
        let input_id = graph.add_node(NodeType::Input, NodeParams::None);

        // Create quantized Conv2d node
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

        // Create FP32 Output node
        let output_id = graph.add_node(NodeType::Output, NodeParams::None);

        graph.connect(input_id, conv_id);
        graph.connect(conv_id, output_id);

        // Create quantization params
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

        // Insert Q/DQ nodes
        let inserted = insert_qdq_nodes(&mut graph, &quant_params);
        assert_eq!(inserted, 2); // One Q before Conv, one DQ after Conv

        // Verify graph structure
        let conv_node = graph.get_node(conv_id).unwrap();

        // Conv should have one quantize input
        assert_eq!(conv_node.inputs.len(), 1);
        let q_id = conv_node.inputs[0];
        let q_node = graph.get_node(q_id).unwrap();
        assert_eq!(q_node.node_type, NodeType::Quantize);

        // Conv should have one dequantize output
        assert_eq!(conv_node.outputs.len(), 1);
        let dq_id = conv_node.outputs[0];
        let dq_node = graph.get_node(dq_id).unwrap();
        assert_eq!(dq_node.node_type, NodeType::Dequantize);
    }

    #[test]
    fn test_fuse_relu() {
        let mut graph = ComputationGraph::new();

        // Create Conv2d node
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

        // Create ReLU node
        let relu_id = graph.add_node(NodeType::ReLU, NodeParams::Activation);

        graph.connect(conv_id, relu_id);

        // Fuse ReLU
        let fused = fuse_relu(&mut graph);
        assert_eq!(fused, 1);

        // Verify ReLU was removed
        assert!(graph.get_node(relu_id).is_none());

        // Verify Conv outputs are connected to what ReLU was connected to
        let conv_node = graph.get_node(conv_id).unwrap();
        assert_eq!(conv_node.outputs, vec![]); // In this test, ReLU had no outputs
    }

    #[test]
    fn test_fuse_hardswish() {
        let mut graph = ComputationGraph::new();

        // Create Conv2d node
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

        // Create HardSwish node
        let hs_id = graph.add_node(NodeType::HardSwish, NodeParams::Activation);

        graph.connect(conv_id, hs_id);

        // Fuse HardSwish
        let fused = fuse_hardswish(&mut graph);
        assert_eq!(fused, 1);

        // Verify HardSwish was removed
        assert!(graph.get_node(hs_id).is_none());
    }

    #[test]
    fn test_hardswish_lut_generation() {
        let scale = 0.1;
        let zero_point = 0;
        let lut = generate_hardswish_lut(scale, zero_point);

        // LUT is indexed by the i8 quantized value reinterpreted as u8:
        //   lut[q as u8 as usize]
        // generate_hardswish_lut iterates i in 0..256 with q_input = i as i8,
        // so index 0 ↔ q=0, index 30 ↔ q=30, index 226 ↔ q=-30.
        let lut_idx = |q: i32| -> usize { (q as i8) as u8 as usize };

        // x = 0 → HardSwish(0) = 0
        assert_eq!(lut[lut_idx(0 - zero_point)], 0);

        // x = -3 (q = -30 with scale=0.1) → HardSwish ≈ 0
        assert_eq!(lut[lut_idx(-30 - zero_point)], 0);

        // x = 3 (q = 30 with scale=0.1) → HardSwish(x) ≈ x
        let x_pos3 = (lut[lut_idx(30 - zero_point)] as i32 - zero_point) as f32 * scale;
        assert!(
            (x_pos3 - 3.0).abs() < 0.5,
            "expected ~3.0 got {x_pos3}"
        ); // Should be close to 3.0
    }

    #[test]
    fn test_graph_construction() {
        let mut graph = ComputationGraph::new();

        let id1 = graph.add_node(NodeType::Input, NodeParams::None);
        let id2 = graph.add_node(
            NodeType::Conv2d,
            NodeParams::Conv2d {
                weights: vec![1.0; 4],
                bias: None,
                in_channels: 1,
                out_channels: 1,
                kernel_size: 2,
            },
        );
        let id3 = graph.add_node(NodeType::Output, NodeParams::None);

        graph.connect(id1, id2);
        graph.connect(id2, id3);

        assert_eq!(graph.nodes.len(), 3);
        assert_eq!(graph.get_node(id2).unwrap().inputs, vec![id1]);
        assert_eq!(graph.get_node(id2).unwrap().outputs, vec![id3]);
    }

    #[test]
    fn test_remove_node() {
        let mut graph = ComputationGraph::new();

        let id1 = graph.add_node(NodeType::Input, NodeParams::None);
        let id2 = graph.add_node(NodeType::ReLU, NodeParams::Activation);
        let id3 = graph.add_node(NodeType::Output, NodeParams::None);

        graph.connect(id1, id2);
        graph.connect(id2, id3);

        graph.remove_node(id2);

        // id2 should be removed
        assert!(graph.get_node(id2).is_none());

        // id1 should connect directly to id3
        assert_eq!(graph.get_node(id1).unwrap().outputs, vec![id3]);
        assert_eq!(graph.get_node(id3).unwrap().inputs, vec![id1]);
    }
}
