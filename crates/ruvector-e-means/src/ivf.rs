//! Stable inverted-file postings over a trained e-means centroid model.

use crate::{fit, validate_rows, Config, Error, Model};

/// A centroid model plus compact, deterministic inverted lists of input row IDs.
/// Posting `j` occupies `indices[offsets[j]..offsets[j + 1]]`. Row IDs within
/// every list are ascending and every training row occurs exactly once.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct IvfPartition {
    model: Model,
    offsets: Vec<usize>,
    indices: Vec<usize>,
}

impl IvfPartition {
    /// Train centroids and build postings from the final nearest assignments.
    pub fn fit(data: &[f32], dimensions: usize, config: &Config) -> Result<Self, Error> {
        Self::from_model(fit(data, dimensions, config)?)
    }

    /// Build postings for an existing, validated fitted model.
    pub fn from_model(model: Model) -> Result<Self, Error> {
        let clusters = validate_model(&model)?;
        let mut offsets = vec![0usize; clusters + 1];
        for &assignment in &model.assignments {
            offsets[assignment + 1] += 1;
        }
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let mut cursors = offsets[..clusters].to_vec();
        let mut indices = vec![0usize; model.assignments.len()];
        for (row, &assignment) in model.assignments.iter().enumerate() {
            indices[cursors[assignment]] = row;
            cursors[assignment] += 1;
        }
        Ok(Self {
            model,
            offsets,
            indices,
        })
    }

    /// Restore a persisted partition, checking offsets, labels, and row IDs.
    /// Repeated, omitted, reordered, or incorrectly assigned row IDs reject.
    pub fn from_parts(
        model: Model,
        offsets: Vec<usize>,
        indices: Vec<usize>,
    ) -> Result<Self, Error> {
        let clusters = validate_model(&model)?;
        let samples = model.assignments.len();
        if offsets.len() != clusters + 1
            || offsets.first() != Some(&0)
            || offsets.last() != Some(&samples)
            || indices.len() != samples
        {
            return Err(Error(
                "IVF offsets and indices must cover every training row",
            ));
        }
        if offsets
            .windows(2)
            .any(|pair| pair[0] > pair[1] || pair[1] > samples)
        {
            return Err(Error(
                "IVF offsets must be monotonic and within the row count",
            ));
        }
        for cluster in 0..clusters {
            let posting = &indices[offsets[cluster]..offsets[cluster + 1]];
            if posting.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err(Error("IVF posting row IDs must be strictly ascending"));
            }
            for &row in posting {
                if row >= samples || model.assignments[row] != cluster {
                    return Err(Error("IVF row ID is out of range or in the wrong posting"));
                }
            }
        }
        Ok(Self {
            model,
            offsets,
            indices,
        })
    }

    /// Trained centroids, final labels, and training metrics.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Exclusive posting boundaries, of length `clusters + 1`.
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Input row IDs grouped by cluster, stable within each cluster.
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Borrow one inverted list, returning an error for an invalid cluster.
    pub fn posting(&self, cluster: usize) -> Result<&[usize], Error> {
        if cluster >= self.offsets.len() - 1 {
            return Err(Error("IVF cluster index is out of range"));
        }
        Ok(&self.indices[self.offsets[cluster]..self.offsets[cluster + 1]])
    }

    /// Consume this partition without copying its trained centroid model.
    pub fn into_model(self) -> Model {
        self.model
    }
}

fn validate_model(model: &Model) -> Result<usize, Error> {
    validate_rows(&model.centroids, model.dimensions)?;
    let clusters = model.centroids.len() / model.dimensions;
    let samples = model.assignments.len();
    if clusters == 0 || samples < clusters || model.assignments.iter().any(|&j| j >= clusters) {
        return Err(Error(
            "IVF model must have valid nonempty centroids and assignments",
        ));
    }
    if !model.mse.is_finite()
        || model.mse < 0.0
        || !model.peak_step.is_finite()
        || model.peak_step <= 0.0
        || model.peak_step > 1.0
        || model.batch_size == 0
        || model.batch_size > samples
        || model.updates == 0
    {
        return Err(Error("IVF model has invalid fitted training metrics"));
    }
    Ok(clusters)
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for IvfPartition {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            model: Model,
            offsets: Vec<usize>,
            indices: Vec<usize>,
        }
        let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_parts(wire.model, wire.offsets, wire.indices).map_err(serde::de::Error::custom)
    }
}
