//! Byzantine-tolerant federated aggregation

/// Federated aggregator with Byzantine outlier detection
pub struct ByzantineAggregator {
    std_threshold: f64,
    min_contributions: usize,
}

impl ByzantineAggregator {
    pub fn new() -> Self {
        Self {
            std_threshold: 2.0,
            min_contributions: 3,
        }
    }

    /// Aggregate embeddings, filtering outliers beyond 2 sigma
    pub fn aggregate(&self, embeddings: &[Vec<f32>]) -> Option<Vec<f32>> {
        if embeddings.len() < self.min_contributions {
            return None;
        }
        let dim = embeddings[0].len();

        // Compute mean per dimension
        let n = embeddings.len() as f64;
        let mut mean = vec![0.0f64; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }

        // Compute std dev per dimension
        let mut std_dev = vec![0.0f64; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                std_dev[i] += (v as f64 - mean[i]).powi(2);
            }
        }
        for s in &mut std_dev {
            *s = (*s / n).sqrt();
        }

        // Filter outliers: exclude embeddings with any dimension > 2 sigma from mean
        let inliers: Vec<&Vec<f32>> = embeddings
            .iter()
            .filter(|emb| {
                emb.iter().enumerate().all(|(i, &v)| {
                    let dev = (v as f64 - mean[i]).abs();
                    std_dev[i] < 1e-10 || dev <= self.std_threshold * std_dev[i]
                })
            })
            .collect();

        if inliers.len() < self.min_contributions {
            return None;
        }

        // Compute filtered mean
        let n_inliers = inliers.len() as f64;
        let mut result = vec![0.0f32; dim];
        for emb in &inliers {
            for (i, &v) in emb.iter().enumerate() {
                result[i] += v / n_inliers as f32;
            }
        }

        Some(result)
    }

    /// Aggregate with reputation-weighted contributions
    ///
    /// Each contribution has a reputation weight. The weighted mean is
    /// computed first, then the standard 2 sigma outlier filter is applied.
    pub fn aggregate_weighted(&self, contributions: &[(Vec<f32>, f64)]) -> Option<Vec<f32>> {
        if contributions.len() < self.min_contributions {
            return None;
        }
        let dim = contributions[0].0.len();
        let total_weight: f64 = contributions.iter().map(|(_, w)| *w).sum();
        if total_weight < 1e-10 {
            return None;
        }

        // Compute weighted mean
        let mut mean = vec![0.0f64; dim];
        for (emb, weight) in contributions {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v as f64 * weight;
            }
        }
        for m in &mut mean {
            *m /= total_weight;
        }

        // Compute weighted std dev
        let mut std_dev = vec![0.0f64; dim];
        for (emb, weight) in contributions {
            for (i, &v) in emb.iter().enumerate() {
                std_dev[i] += weight * (v as f64 - mean[i]).powi(2);
            }
        }
        for s in &mut std_dev {
            *s = (*s / total_weight).sqrt();
        }

        // Filter outliers: exclude contributions with any dimension > 2 sigma
        let inliers: Vec<(&Vec<f32>, f64)> = contributions
            .iter()
            .filter(|(emb, _)| {
                emb.iter().enumerate().all(|(i, &v)| {
                    let dev = (v as f64 - mean[i]).abs();
                    std_dev[i] < 1e-10 || dev <= self.std_threshold * std_dev[i]
                })
            })
            .map(|(emb, w)| (emb, *w))
            .collect();

        if inliers.len() < self.min_contributions {
            return None;
        }

        // Compute filtered weighted mean
        let inlier_weight: f64 = inliers.iter().map(|(_, w)| w).sum();
        if inlier_weight < 1e-10 {
            return None;
        }
        let mut result = vec![0.0f32; dim];
        for (emb, weight) in &inliers {
            for (i, &v) in emb.iter().enumerate() {
                result[i] += (v as f64 * weight / inlier_weight) as f32;
            }
        }

        Some(result)
    }

    /// Count how many embeddings would be filtered as outliers
    pub fn count_outliers(&self, embeddings: &[Vec<f32>]) -> usize {
        if embeddings.len() < 2 {
            return 0;
        }
        let dim = embeddings[0].len();
        let n = embeddings.len() as f64;
        let mut mean = vec![0.0f64; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                mean[i] += v as f64;
            }
        }
        for m in &mut mean {
            *m /= n;
        }
        let mut std_dev = vec![0.0f64; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                std_dev[i] += (v as f64 - mean[i]).powi(2);
            }
        }
        for s in &mut std_dev {
            *s = (*s / n).sqrt();
        }

        embeddings
            .iter()
            .filter(|emb| {
                emb.iter().enumerate().any(|(i, &v)| {
                    let dev = (v as f64 - mean[i]).abs();
                    std_dev[i] > 1e-10 && dev > self.std_threshold * std_dev[i]
                })
            })
            .count()
    }
}

impl Default for ByzantineAggregator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aggregate_normal() {
        let agg = ByzantineAggregator::new();
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
        ];
        let result = agg.aggregate(&embeddings).unwrap();
        assert!((result[0] - 1.0).abs() < 0.2);
    }

    #[test]
    fn test_filter_outliers() {
        let agg = ByzantineAggregator::new();
        let embeddings = vec![
            vec![1.0, 2.0, 3.0],
            vec![1.1, 2.1, 3.1],
            vec![0.9, 1.9, 2.9],
            vec![1.0, 2.0, 3.0],
            vec![1.05, 2.05, 3.05],
            vec![0.95, 1.95, 2.95],
            vec![100.0, 200.0, 300.0], // Outlier — far beyond 2 sigma with enough inliers
        ];
        assert!(agg.count_outliers(&embeddings) >= 1);
    }

    #[test]
    fn test_insufficient_contributions() {
        let agg = ByzantineAggregator::new();
        let embeddings = vec![vec![1.0, 2.0]];
        assert!(agg.aggregate(&embeddings).is_none());
    }

    #[test]
    fn test_aggregate_weighted() {
        let agg = ByzantineAggregator::new();
        let contributions = vec![
            (vec![1.0, 2.0, 3.0], 1.0),
            (vec![1.1, 2.1, 3.1], 2.0), // Higher reputation
            (vec![0.9, 1.9, 2.9], 1.0),
        ];
        let result = agg.aggregate_weighted(&contributions).unwrap();
        // Weighted mean should lean towards the second contribution
        assert!(result[0] > 0.9 && result[0] < 1.2);
        assert!(result[1] > 1.9 && result[1] < 2.2);
    }

    #[test]
    fn test_weighted_outlier_filtering() {
        let agg = ByzantineAggregator::new();
        let contributions = vec![
            (vec![1.0, 2.0, 3.0], 1.0),
            (vec![1.1, 2.1, 3.1], 1.0),
            (vec![0.9, 1.9, 2.9], 1.0),
            (vec![1.0, 2.0, 3.0], 1.0),
            (vec![100.0, 200.0, 300.0], 0.5), // Outlier with low reputation
        ];
        let result = agg.aggregate_weighted(&contributions);
        // Should succeed after filtering the outlier
        assert!(result.is_some());
        let r = result.unwrap();
        assert!((r[0] - 1.0).abs() < 0.2);
    }
}
