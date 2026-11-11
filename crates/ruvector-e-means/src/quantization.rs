//! Checked product-quantization codebooks trained by the shared e-means kernel.

use crate::{fit, nearest, validate_rows, Config, Error};

/// A reusable PQ codebook with private, validated layout metadata.
///
/// Centroids are subspace-major: `[subspace][cluster][subspace_dimension]`.
/// Codes are row-major: one u8 per subspace per input vector. Each subspace
/// uses the supplied configuration unchanged, including its deterministic seed.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "camelCase"))]
pub struct ProductQuantizer {
    dimensions: usize,
    subspaces: usize,
    clusters: usize,
    centroids: Vec<f32>,
}

impl ProductQuantizer {
    /// Train one e-means codebook per equally sized subspace.
    pub fn fit(
        data: &[f32],
        dimensions: usize,
        subspaces: usize,
        config: &Config,
    ) -> Result<Self, Error> {
        validate_rows(data, dimensions)?;
        config.parameters(data.len() / dimensions)?;
        Self::validate_shape(dimensions, subspaces, config.clusters)?;
        let sub_dim = dimensions / subspaces;
        let mut centroids = Vec::with_capacity(dimensions * config.clusters);
        let mut sub_data = Vec::with_capacity(data.len() / subspaces);
        for subspace in 0..subspaces {
            sub_data.clear();
            let start = subspace * sub_dim;
            for row in data.chunks_exact(dimensions) {
                sub_data.extend_from_slice(&row[start..start + sub_dim]);
            }
            centroids.extend(fit(&sub_data, sub_dim, config)?.centroids);
        }
        Self::from_centroids(dimensions, subspaces, config.clusters, centroids)
    }

    /// Restore a codebook, rejecting inconsistent shapes or nonfinite values.
    pub fn from_centroids(
        dimensions: usize,
        subspaces: usize,
        clusters: usize,
        centroids: Vec<f32>,
    ) -> Result<Self, Error> {
        Self::validate_parts(dimensions, subspaces, clusters, &centroids)?;
        Ok(Self {
            dimensions,
            subspaces,
            clusters,
            centroids,
        })
    }

    /// Validate a borrowed codebook layout without allocating a copy.
    pub fn validate_parts(
        dimensions: usize,
        subspaces: usize,
        clusters: usize,
        centroids: &[f32],
    ) -> Result<(), Error> {
        Self::validate_shape(dimensions, subspaces, clusters)?;
        if centroids.len() != dimensions * clusters {
            return Err(Error(
                "PQ centroid length must equal dimensions times clusters",
            ));
        }
        validate_rows(centroids, dimensions / subspaces)
    }

    fn validate_shape(dimensions: usize, subspaces: usize, clusters: usize) -> Result<(), Error> {
        if dimensions == 0 || subspaces == 0 || dimensions % subspaces != 0 {
            return Err(Error(
                "PQ subspaces must be positive and divide positive dimensions",
            ));
        }
        if !(1..=256).contains(&clusters) {
            return Err(Error("PQ clusters must be in 1..=256 for u8 codes"));
        }
        dimensions
            .checked_mul(clusters)
            .ok_or(Error("PQ centroid shape overflows usize"))?;
        Ok(())
    }

    /// Original vector dimensionality.
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }

    /// Number of equally sized subspaces, and bytes in each encoded vector.
    pub fn subspaces(&self) -> usize {
        self.subspaces
    }

    /// Number of centroids in each subspace.
    pub fn clusters(&self) -> usize {
        self.clusters
    }

    /// Dimensionality of each subvector.
    pub fn subspace_dimensions(&self) -> usize {
        self.dimensions / self.subspaces
    }

    /// Borrow the flat subspace-major codebook.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Consume the quantizer and return its existing flat codebook allocation.
    pub fn into_centroids(self) -> Vec<f32> {
        self.centroids
    }

    /// Encode finite row-major vectors with exact f64 nearest-centroid assignment.
    /// Empty input produces an empty code array; ties choose the lowest code.
    pub fn encode(&self, data: &[f32]) -> Result<Vec<u8>, Error> {
        validate_rows(data, self.dimensions)?;
        let sub_dim = self.subspace_dimensions();
        let mut codes = Vec::with_capacity(data.len() / sub_dim);
        for row in data.chunks_exact(self.dimensions) {
            for (subspace, subvector) in row.chunks_exact(sub_dim).enumerate() {
                let start = subspace * self.clusters * sub_dim;
                let book = &self.centroids[start..start + self.clusters * sub_dim];
                codes.push(nearest(subvector, book, sub_dim).0 as u8);
            }
        }
        Ok(codes)
    }

    /// Decode row-major PQ codes, rejecting incomplete rows and invalid codes.
    pub fn decode(&self, codes: &[u8]) -> Result<Vec<f32>, Error> {
        if codes.len() % self.subspaces != 0 {
            return Err(Error("PQ code length must be divisible by subspaces"));
        }
        if codes.iter().any(|&code| usize::from(code) >= self.clusters) {
            return Err(Error("PQ code is outside the trained codebook"));
        }
        let capacity = (codes.len() / self.subspaces)
            .checked_mul(self.dimensions)
            .ok_or(Error("decoded PQ shape overflows usize"))?;
        let sub_dim = self.subspace_dimensions();
        let mut data = Vec::with_capacity(capacity);
        for row in codes.chunks_exact(self.subspaces) {
            for (subspace, &code) in row.iter().enumerate() {
                let start = (subspace * self.clusters + usize::from(code)) * sub_dim;
                data.extend_from_slice(&self.centroids[start..start + sub_dim]);
            }
        }
        Ok(data)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ProductQuantizer {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(serde::Deserialize)]
        #[serde(rename_all = "camelCase", deny_unknown_fields)]
        struct Wire {
            dimensions: usize,
            subspaces: usize,
            clusters: usize,
            centroids: Vec<f32>,
        }
        let wire = <Wire as serde::Deserialize>::deserialize(deserializer)?;
        Self::from_centroids(
            wire.dimensions,
            wire.subspaces,
            wire.clusters,
            wire.centroids,
        )
        .map_err(serde::de::Error::custom)
    }
}
