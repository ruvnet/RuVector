//! Structural validation for deserialized GNN checkpoints.
use super::{Linear, RuvectorLayer};
use crate::error::GnnError;

impl RuvectorLayer {
    /// Validate a deserialized checkpoint against its declared configuration.
    ///
    /// Checks tensor shapes, finite parameters, normalization epsilon, and
    /// attention configuration without replacing or reinitializing weights.
    /// This scans parameters once at load time; it does not establish trained
    /// model quality or numerical stability for arbitrary forward-pass inputs.
    pub fn validate_checkpoint(
        &self,
        input_dim: usize,
        hidden_dim: usize,
        heads: usize,
        dropout: f32,
    ) -> Result<(), GnnError> {
        if input_dim == 0 || hidden_dim == 0 || heads == 0 || hidden_dim % heads != 0 {
            return Err(GnnError::layer_config(
                "Checkpoint dimensions and attention heads are inconsistent",
            ));
        }
        if !(0.0..=1.0).contains(&dropout) || self.dropout != dropout {
            return Err(GnnError::layer_config(
                "Checkpoint dropout does not match its declared configuration",
            ));
        }
        if self.attention.num_heads != heads || self.attention.head_dim != hidden_dim / heads {
            return Err(GnnError::layer_config(
                "Checkpoint attention does not match its declared configuration",
            ));
        }
        let validate_linear = |layer: &Linear, input: usize, output: usize, name: &str| {
            if layer.weights.dim() != (output, input) || layer.bias.len() != output {
                return Err(GnnError::layer_config(format!(
                    "Checkpoint {name} tensor dimensions are inconsistent"
                )));
            }
            if !layer
                .weights
                .iter()
                .chain(layer.bias.iter())
                .all(|value| value.is_finite())
            {
                return Err(GnnError::layer_config(format!(
                    "Checkpoint {name} parameters must be finite"
                )));
            }
            Ok(())
        };
        validate_linear(&self.w_msg, input_dim, hidden_dim, "message")?;
        for (name, layer) in [
            ("aggregation", &self.w_agg),
            ("update gate input", &self.w_update.w_z),
            ("update gate recurrent", &self.w_update.u_z),
            ("reset gate input", &self.w_update.w_r),
            ("reset gate recurrent", &self.w_update.u_r),
            ("candidate input", &self.w_update.w_h),
            ("candidate recurrent", &self.w_update.u_h),
            ("attention query", &self.attention.q_linear),
            ("attention key", &self.attention.k_linear),
            ("attention value", &self.attention.v_linear),
            ("attention output", &self.attention.out_linear),
        ] {
            validate_linear(layer, hidden_dim, hidden_dim, name)?;
        }
        if self.norm.gamma.len() != hidden_dim || self.norm.beta.len() != hidden_dim {
            return Err(GnnError::layer_config(
                "Checkpoint normalization tensor dimensions are inconsistent",
            ));
        }
        if !self.norm.eps.is_finite()
            || self.norm.eps <= 0.0
            || !self
                .norm
                .gamma
                .iter()
                .chain(self.norm.beta.iter())
                .all(|value| value.is_finite())
        {
            return Err(GnnError::layer_config(
                "Checkpoint normalization parameters must be finite with positive epsilon",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_validation_preserves_valid_parameters() {
        let mut layer = RuvectorLayer::new(4, 8, 2, 0.0).unwrap();
        layer.w_msg.weights[[0, 0]] = 0.125;
        layer.norm.beta[0] = 2.0;
        let before = serde_json::to_string(&layer).unwrap();
        layer.validate_checkpoint(4, 8, 2, 0.0).unwrap();
        assert_eq!(serde_json::to_string(&layer).unwrap(), before);
    }

    #[test]
    fn checkpoint_validation_checks_recurrent_parameters_and_configuration() {
        let mut layer = RuvectorLayer::new(4, 8, 2, 0.0).unwrap();
        assert!(layer.validate_checkpoint(4, 8, 0, 0.0).is_err());
        assert!(layer.validate_checkpoint(4, 8, 2, 0.5).is_err());
        layer.w_update.u_r.bias[0] = f32::NAN;
        assert!(layer.validate_checkpoint(4, 8, 2, 0.0).is_err());
    }
}
