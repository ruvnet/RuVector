//! SIREN-style neural network for learned manifold representation
//!
//! SIREN (Sinusoidal Representation Networks) uses periodic activation
//! functions to better represent continuous signals and implicit functions.
//!
//! Reference: "Implicit Neural Representations with Periodic Activation Functions"
//! (Sitzmann et al., 2020)

use burn::prelude::*;
use burn::nn;

/// SIREN layer with sinusoidal activation
#[derive(Module, Debug)]
pub struct SirenLayer<B: Backend> {
    linear: nn::Linear<B>,
    omega_0: f32,
}

impl<B: Backend> SirenLayer<B> {
    /// Create a new SIREN layer
    ///
    /// # Arguments
    /// * `input_dim` - Input dimension
    /// * `output_dim` - Output dimension
    /// * `omega_0` - Frequency parameter for sin activation
    /// * `is_first` - Whether this is the first layer (affects initialization)
    /// * `device` - Device for computation
    pub fn new(
        input_dim: usize,
        output_dim: usize,
        omega_0: f32,
        is_first: bool,
        device: &B::Device,
    ) -> Self {
        // SIREN initialization
        let linear = if is_first {
            // First layer: uniform(-1/n, 1/n)
            let bound = 1.0 / input_dim as f32;
            nn::LinearConfig::new(input_dim, output_dim)
                .with_initializer(nn::Initializer::Uniform {
                    min: -bound as f64,
                    max: bound as f64,
                })
                .init(device)
        } else {
            // Hidden layers: uniform(-sqrt(6/n)/omega_0, sqrt(6/n)/omega_0)
            let bound = (6.0_f32 / input_dim as f32).sqrt() / omega_0;
            nn::LinearConfig::new(input_dim, output_dim)
                .with_initializer(nn::Initializer::Uniform {
                    min: -bound as f64,
                    max: bound as f64,
                })
                .init(device)
        };

        Self { linear, omega_0 }
    }

    /// Forward pass with sinusoidal activation
    pub fn forward<const D: usize>(&self, input: Tensor<B, D>) -> Tensor<B, D> {
        let x = self.linear.forward(input);
        // sin(omega_0 * x)
        let scaled = x * self.omega_0;
        scaled.sin()
    }
}

/// Learned manifold network architecture
#[derive(Module, Debug)]
pub struct LearnedManifold<B: Backend> {
    /// SIREN-style sinusoidal layers
    layers: Vec<SirenLayer<B>>,
    /// Final output layer (no activation)
    output: nn::Linear<B>,
    /// Input dimension
    input_dim: usize,
}

impl<B: Backend> LearnedManifold<B> {
    /// Create a new learned manifold network
    ///
    /// # Arguments
    /// * `input_dim` - Embedding dimension
    /// * `hidden_dim` - Hidden layer dimension
    /// * `num_layers` - Number of hidden layers
    /// * `omega_0` - SIREN frequency parameter
    /// * `device` - Device for computation
    pub fn new(
        input_dim: usize,
        hidden_dim: usize,
        num_layers: usize,
        omega_0: f32,
        device: &B::Device,
    ) -> Self {
        let mut layers = Vec::new();

        // First layer
        layers.push(SirenLayer::new(
            input_dim,
            hidden_dim,
            omega_0,
            true,
            device,
        ));

        // Hidden layers
        for _ in 1..num_layers {
            layers.push(SirenLayer::new(
                hidden_dim,
                hidden_dim,
                omega_0,
                false,
                device,
            ));
        }

        // Output layer (relevance score)
        let output = nn::LinearConfig::new(hidden_dim, 1).init(device);

        Self {
            layers,
            output,
            input_dim,
        }
    }

    /// Forward pass: embedding → relevance score
    ///
    /// Returns a scalar relevance value indicating how salient
    /// the manifold is at the given position.
    pub fn forward(&self, input: Tensor<B, 1>) -> Tensor<B, 1> {
        let mut x = input;

        // Pass through SIREN layers
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Output layer (no activation)
        let relevance = self.output.forward(x);

        relevance
    }

    /// Forward pass with batch dimension
    pub fn forward_batch(&self, input: Tensor<B, 2>) -> Tensor<B, 2> {
        let mut x = input;

        // Pass through SIREN layers
        for layer in &self.layers {
            x = layer.forward(x);
        }

        // Output layer
        let relevance = self.output.forward(x);

        relevance
    }

    /// Get input dimension
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray;

    #[test]
    fn test_siren_layer() {
        let device = Default::default();
        let layer = SirenLayer::<TestBackend>::new(64, 128, 30.0, true, &device);

        let input = Tensor::<TestBackend, 1>::zeros([64], &device);
        let output = layer.forward(input);

        assert_eq!(output.dims(), [128]);
    }

    #[test]
    fn test_learned_manifold() {
        let device = Default::default();
        let network = LearnedManifold::<TestBackend>::new(64, 256, 3, 30.0, &device);

        let input = Tensor::<TestBackend, 1>::zeros([64], &device);
        let relevance = network.forward(input);

        assert_eq!(relevance.dims(), [1]);
    }

    #[test]
    fn test_learned_manifold_batch() {
        let device = Default::default();
        let network = LearnedManifold::<TestBackend>::new(64, 256, 3, 30.0, &device);

        let input = Tensor::<TestBackend, 2>::zeros([8, 64], &device);
        let relevance = network.forward_batch(input);

        assert_eq!(relevance.dims(), [8, 1]);
    }
}
