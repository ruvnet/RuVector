//! ANN-Benchmarks HDF5 dataset loader.
//!
//! Downloads and loads standard ANN-Benchmarks datasets from GitHub:
//!   - SIFT-128-euclidean  (1M train, 10K test)
//!   - GloVe-25-angular    (1.18M train, 10K test)
//!   - GloVe-100-angular   (1.18M train, 10K test)
//!   - Deep-image-96-angular (10M train, 10K test)
//!
//! HDF5 format: each file contains `train` (corpus), `test` (queries),
//! and `neighbors` (ground truth top-100 ids) datasets.
//!
//! Usage: enable `real-datasets` feature to compile. Without it, all
//! functions in this module return descriptive errors and the rest of
//! the benchmark suite still works with synthetic data.

use crate::Dataset;

/// Dataset descriptor for ANN-Benchmarks standard sets.
pub struct AnnDatasetSpec {
    pub name: &'static str,
    pub url: &'static str,
    pub dims: usize,
}

/// All standard ANN-Benchmarks datasets (feasible to download + run).
pub const ANN_DATASETS: &[AnnDatasetSpec] = &[
    AnnDatasetSpec {
        name: "sift-128-euclidean",
        url: "https://ann-benchmarks.com/sift-128-euclidean.hdf5",
        dims: 128,
    },
    AnnDatasetSpec {
        name: "glove-25-angular",
        url: "https://ann-benchmarks.com/glove-25-angular.hdf5",
        dims: 25,
    },
    AnnDatasetSpec {
        name: "glove-100-angular",
        url: "https://ann-benchmarks.com/glove-100-angular.hdf5",
        dims: 100,
    },
    AnnDatasetSpec {
        name: "deep-image-96-angular",
        url: "https://ann-benchmarks.com/deep-image-96-angular.hdf5",
        dims: 96,
    },
];

/// Download an ANN-Benchmarks HDF5 file to a local cache directory.
/// Returns the local path.
#[cfg(feature = "real-datasets")]
pub fn download_dataset(
    spec: &AnnDatasetSpec,
    cache_dir: &std::path::Path,
) -> anyhow::Result<std::path::PathBuf> {
    use std::io::Write;

    std::fs::create_dir_all(cache_dir)?;
    let filename = spec.url.split('/').last().unwrap_or("dataset.hdf5");
    let local = cache_dir.join(filename);

    if local.exists() {
        println!(
            "  [cache] {} already exists, skipping download",
            local.display()
        );
        return Ok(local);
    }

    println!("  [download] {} → {}", spec.url, local.display());
    let resp = reqwest::blocking::get(spec.url)?;
    let bytes = resp.bytes()?;
    let mut f = std::fs::File::create(&local)?;
    f.write_all(&bytes)?;
    println!("  [done] {:.1} MB", bytes.len() as f64 / (1024.0 * 1024.0));
    Ok(local)
}

/// Load a downloaded HDF5 ANN-Benchmarks file into a Dataset.
///
/// HDF5 layout:
///   /train        — float32 [n_corpus, dims] — corpus vectors
///   /test         — float32 [n_queries, dims] — query vectors
///   /neighbors    — int32   [n_queries, 100]  — true top-100 neighbour ids
#[cfg(feature = "real-datasets")]
pub fn load_hdf5(
    spec: &AnnDatasetSpec,
    path: &std::path::Path,
    max_corpus: usize,
    max_queries: usize,
) -> anyhow::Result<Dataset> {
    use hdf5::File;

    let file = File::open(path)?;

    let train_ds = file.dataset("train")?;
    let test_ds = file.dataset("test")?;
    let nn_ds = file.dataset("neighbors")?;

    // Read corpus (capped for memory)
    let train_data: ndarray::Array2<f32> = train_ds.read_2d()?;
    let n_corpus = max_corpus.min(train_data.nrows());
    let corpus: Vec<Vec<f32>> = (0..n_corpus).map(|i| train_data.row(i).to_vec()).collect();

    // Read queries (capped)
    let test_data: ndarray::Array2<f32> = test_ds.read_2d()?;
    let n_queries = max_queries.min(test_data.nrows());
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|i| test_data.row(i).to_vec()).collect();

    // Read ground-truth top-100 ids (int32 in the HDF5 format)
    let nn_data: ndarray::Array2<i32> = nn_ds.read_2d()?;
    let ground_truth: Vec<Vec<u64>> = (0..n_queries)
        .map(|i| {
            nn_data
                .row(i)
                .iter()
                .take(100)
                .map(|&id| id as u64)
                .collect()
        })
        .collect();

    Ok(Dataset {
        name: spec.name.to_string(),
        dims: spec.dims,
        corpus,
        queries,
        ground_truth,
    })
}

/// Load (downloading if necessary) a standard ANN-Benchmarks dataset.
#[cfg(feature = "real-datasets")]
pub fn load_ann_dataset(
    spec: &AnnDatasetSpec,
    cache_dir: &std::path::Path,
    max_corpus: usize,
    max_queries: usize,
) -> anyhow::Result<Dataset> {
    let path = download_dataset(spec, cache_dir)?;
    load_hdf5(spec, &path, max_corpus, max_queries)
}

/// Without the `real-datasets` feature, return a clear error.
#[cfg(not(feature = "real-datasets"))]
pub fn load_ann_dataset(
    spec: &AnnDatasetSpec,
    _cache_dir: &std::path::Path,
    _max_corpus: usize,
    _max_queries: usize,
) -> anyhow::Result<Dataset> {
    anyhow::bail!(
        "Real dataset '{}' requires the `real-datasets` feature and HDF5 headers.\n\
         Build with: cargo run -p ruvector-sota-bench --features real-datasets --bin sota-all\n\
         Or run on synthetic data: cargo run -p ruvector-sota-bench --bin sota-all -- --smoke",
        spec.name
    )
}

/// Standard 100K-cap datasets for rapid benchmarking (still real vectors).
#[cfg(feature = "real-datasets")]
pub fn load_rapid_datasets(cache_dir: &std::path::Path) -> Vec<anyhow::Result<Dataset>> {
    ANN_DATASETS
        .iter()
        .map(|spec| load_ann_dataset(spec, cache_dir, 100_000, 1_000))
        .collect()
}

/// Full 1M datasets for publication-quality benchmarking (Tier 3, ADR-267).
#[cfg(feature = "real-datasets")]
pub fn load_full_datasets(cache_dir: &std::path::Path) -> Vec<anyhow::Result<Dataset>> {
    ANN_DATASETS
        .iter()
        .map(|spec| {
            let max_c = if spec.name.starts_with("deep-image") {
                10_000_000
            } else {
                1_000_000
            };
            load_ann_dataset(spec, cache_dir, max_c, 10_000)
        })
        .collect()
}
