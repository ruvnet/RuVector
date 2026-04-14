//! Depth-configurable EML tree evaluation.
//!
//! An [`EmlTree`] is a fixed-depth tree of EML operators with trainable
//! mixing weights. It maps N input features to a single scalar output.
//!
//! Supported depths: 2, 3, 4, 5.

use crate::operator::{eml_safe, softmax3};

/// Depth-configurable EML evaluation tree.
///
/// The tree maps `input_count` features through layers of affine mixing
/// and EML operators to produce a single scalar output.
///
/// # Architecture
///
/// - **Level 0**: `2^(depth-1)` affine combinations of input features
///   (3 params each via softmax3 mixing).
/// - **Levels 1..depth-1**: EML nodes halving the width at each level,
///   with mixing weights.
/// - **Output**: final EML node producing a single scalar.
#[derive(Debug, Clone)]
pub struct EmlTree {
    depth: usize,
    input_count: usize,
    param_count: usize,
}

impl EmlTree {
    /// Create a new EML tree specification.
    ///
    /// # Arguments
    /// - `depth`: Tree depth (2, 3, 4, or 5).
    /// - `input_count`: Number of input features.
    ///
    /// # Panics
    /// Panics if depth is not in {2, 3, 4, 5}.
    pub fn new(depth: usize, input_count: usize) -> Self {
        assert!(
            (2..=5).contains(&depth),
            "EmlTree depth must be 2, 3, 4, or 5, got {depth}"
        );
        let param_count = Self::compute_param_count(depth, input_count);
        Self {
            depth,
            input_count,
            param_count,
        }
    }

    /// Number of trainable parameters for this tree configuration.
    pub fn param_count(&self) -> usize {
        self.param_count
    }

    /// Tree depth.
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// Number of input features.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// Compute the total parameter count for a given depth and input count.
    ///
    /// Level 0: `width` nodes * 3 params each (softmax3 mixing of 2 inputs + bias).
    /// Each subsequent level halves the width and adds mixing params.
    fn compute_param_count(depth: usize, _input_count: usize) -> usize {
        let width = 1usize << (depth - 1); // 2^(depth-1) nodes at level 0

        // Level 0: each node has 3 softmax params
        let mut total = width * 3;

        // Levels 2..depth: each level halves, each node needs mixing weights.
        // Level 1 is pure EML (no extra params — just pairs level-0 outputs).
        let mut w = width / 2; // level 1 width (after first EML pairing)
        for level in 2..depth {
            // Each node at this level mixes two inputs: 2 weights
            // plus for deeper trees we use 3-weight softmax mixing
            let params_per_node = if level < depth - 1 { 3 } else { 2 };
            total += w * params_per_node;
            w /= 2;
            if w == 0 {
                w = 1;
            }
        }

        // Output level: 2 mixing weights for the final EML
        total += 2;

        total
    }

    /// Evaluate the tree with given parameters and inputs.
    ///
    /// # Arguments
    /// - `params`: Trainable parameters (length must equal `param_count()`).
    /// - `inputs`: Input feature values (length must equal `input_count`).
    ///
    /// # Panics
    /// Panics if `params.len() != param_count()` or `inputs.len() != input_count`.
    pub fn evaluate(&self, params: &[f64], inputs: &[f64]) -> f64 {
        assert_eq!(
            params.len(),
            self.param_count,
            "expected {} params, got {}",
            self.param_count,
            params.len()
        );
        assert_eq!(
            inputs.len(),
            self.input_count,
            "expected {} inputs, got {}",
            self.input_count,
            inputs.len()
        );

        let width = 1usize << (self.depth - 1);

        // Level 0: affine combinations via softmax3
        let mut a = vec![0.0f64; width];
        for i in 0..width {
            let base = i * 3;
            let (alpha, beta, gamma) = softmax3(params[base], params[base + 1], params[base + 2]);
            // Pick two input features (cycling through available inputs)
            let j = (i * 2) % self.input_count;
            let k = (i * 2 + 1) % self.input_count;
            a[i] = (alpha + beta * inputs[j] + gamma * inputs[k]).clamp(-10.0, 10.0);
        }

        // Level 1: pair up with EML (no extra params)
        let mut current: Vec<f64> = a
            .chunks(2)
            .map(|pair| eml_safe(pair[0], pair[1].max(0.01)))
            .collect();

        // Levels 2..depth-1: mix + EML
        let mut param_offset = width * 3;
        for level in 2..self.depth {
            let is_last_mix = level == self.depth - 1;
            let params_per_node = if is_last_mix { 2 } else { 3 };
            let next_width = (current.len() + 1) / 2;
            let mut next = Vec::with_capacity(next_width);

            for i in 0..next_width {
                let li = i * 2;
                let ri = (i * 2 + 1).min(current.len() - 1);

                if params_per_node == 3 {
                    let (alpha, beta, gamma) = softmax3(
                        params[param_offset],
                        params[param_offset + 1],
                        params[param_offset + 2],
                    );
                    let mixed = (alpha + beta * current[li] + gamma * current[ri])
                        .clamp(-10.0, 10.0);
                    // Use shifted softmax for the right side
                    let (ar, br, gr) = softmax3(
                        params[param_offset] + 0.5,
                        params[param_offset + 1] - 0.5,
                        params[param_offset + 2],
                    );
                    let mixed_r = (ar + br * current[ri] + gr * current[li]).clamp(0.01, 10.0);
                    next.push(eml_safe(mixed, mixed_r));
                } else {
                    let w0 = params[param_offset];
                    let w1 = params[param_offset + 1];
                    let left = (w0 * current[li] + (1.0 - w0) * current[ri]).clamp(-10.0, 10.0);
                    let right = (w1 * current[li] + (1.0 - w1) * current[ri]).clamp(0.01, 10.0);
                    next.push(eml_safe(left, right));
                }

                param_offset += params_per_node;
            }

            current = next;
        }

        // Output: final mixing
        let w0 = params[param_offset];
        let w1 = params[param_offset + 1];
        let (left, right) = if current.len() >= 2 {
            (
                (w0 * current[0] + (1.0 - w0) * current[1]).clamp(-10.0, 10.0),
                (w1 * current[0] + (1.0 - w1) * current[1]).clamp(0.01, 10.0),
            )
        } else {
            (
                (w0 * current[0]).clamp(-10.0, 10.0),
                (w1 * current[0]).clamp(0.01, 10.0),
            )
        };

        eml_safe(left, right).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tree_depth_2() {
        let tree = EmlTree::new(2, 3);
        assert_eq!(tree.depth(), 2);
        assert_eq!(tree.input_count(), 3);
        let pc = tree.param_count();
        assert!(pc > 0, "param count should be positive");

        let params = vec![0.1; pc];
        let inputs = vec![0.5, 0.3, 0.7];
        let result = tree.evaluate(&params, &inputs);
        assert!(result.is_finite(), "depth-2 result should be finite");
    }

    #[test]
    fn tree_depth_3() {
        let tree = EmlTree::new(3, 5);
        let pc = tree.param_count();
        let params = vec![0.0; pc];
        let inputs = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let result = tree.evaluate(&params, &inputs);
        assert!(result.is_finite());
    }

    #[test]
    fn tree_depth_4() {
        let tree = EmlTree::new(4, 7);
        let pc = tree.param_count();
        let params = vec![0.1; pc];
        let inputs = vec![0.1; 7];
        let result = tree.evaluate(&params, &inputs);
        assert!(result.is_finite());
    }

    #[test]
    fn tree_depth_5() {
        let tree = EmlTree::new(5, 4);
        let pc = tree.param_count();
        assert!(pc > 0);
        let params = vec![0.0; pc];
        let inputs = vec![0.5; 4];
        let result = tree.evaluate(&params, &inputs);
        assert!(result.is_finite());
    }

    #[test]
    #[should_panic(expected = "EmlTree depth must be 2, 3, 4, or 5")]
    fn tree_invalid_depth() {
        EmlTree::new(1, 3);
    }

    #[test]
    fn tree_output_non_negative() {
        for depth in 2..=5 {
            let tree = EmlTree::new(depth, 4);
            let params = vec![0.5; tree.param_count()];
            let inputs = vec![0.3; 4];
            let result = tree.evaluate(&params, &inputs);
            assert!(
                result >= 0.0,
                "depth-{depth} output should be non-negative, got {result}"
            );
        }
    }

    #[test]
    fn param_count_increases_with_depth() {
        let pc2 = EmlTree::new(2, 4).param_count();
        let pc3 = EmlTree::new(3, 4).param_count();
        let pc4 = EmlTree::new(4, 4).param_count();
        let pc5 = EmlTree::new(5, 4).param_count();
        assert!(pc3 > pc2, "depth 3 should have more params than depth 2");
        assert!(pc4 > pc3, "depth 4 should have more params than depth 3");
        assert!(pc5 > pc4, "depth 5 should have more params than depth 4");
    }
}
