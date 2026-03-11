# RuVector CNN Integration Architecture

## Executive Summary

This document outlines the integration architecture for CNN-based feature extraction with RuVector's vector similarity search capabilities. The goal is to provide end-to-end image embedding generation and retrieval using SIMD-optimized Rust implementations.

## Integration Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RuVector CNN Pipeline                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐              │
│  │  Image   │───▶│ Preprocessor  │───▶│    CNN       │              │
│  │  Input   │    │ (Resize/Norm) │    │  Backbone    │              │
│  └──────────┘    └───────────────┘    └──────┬───────┘              │
│                                              │                       │
│                                              ▼                       │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐              │
│  │ RuVector │◀───│  L2 Normalize │◀───│  Embedding   │              │
│  │  Index   │    │               │    │  Extraction  │              │
│  └──────────┘    └───────────────┘    └──────────────┘              │
│       │                                                              │
│       ▼                                                              │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    Vector Search (HNSW/IVF)                   │   │
│  │  • Approximate Nearest Neighbors                             │   │
│  │  • Cosine Similarity (via inner product on normalized vecs)  │   │
│  │  • Filtering and Re-ranking                                  │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

## Module Structure

### Proposed Directory Layout

```
ruvector/
├── crates/
│   └── ruvector-cnn/
│       ├── Cargo.toml
│       ├── src/
│       │   ├── lib.rs
│       │   ├── backbone/
│       │   │   ├── mod.rs
│       │   │   ├── mobilenet.rs
│       │   │   ├── efficientnet.rs
│       │   │   └── shufflenet.rs
│       │   ├── layers/
│       │   │   ├── mod.rs
│       │   │   ├── conv.rs
│       │   │   ├── batchnorm.rs
│       │   │   ├── pooling.rs
│       │   │   └── activation.rs
│       │   ├── simd/
│       │   │   ├── mod.rs
│       │   │   ├── avx2.rs
│       │   │   ├── avx512.rs
│       │   │   └── neon.rs
│       │   ├── quantization/
│       │   │   ├── mod.rs
│       │   │   ├── int8.rs
│       │   │   └── calibration.rs
│       │   ├── contrastive/
│       │   │   ├── mod.rs
│       │   │   ├── infonce.rs
│       │   │   └── triplet.rs
│       │   └── embedding.rs
│       └── tests/
│           └── integration.rs
```

### Cargo.toml

```toml
[package]
name = "ruvector-cnn"
version = "0.1.0"
edition = "2021"
description = "CNN feature extraction for RuVector"

[features]
default = ["avx2"]
avx2 = []
avx512 = []
neon = []
quantized = []
onnx = ["tract-onnx"]

[dependencies]
ndarray = { version = "0.16", features = ["blas"] }
ruvector-core = { path = "../ruvector-core" }

# Optional: ONNX model loading
tract-onnx = { version = "0.21", optional = true }

# Image preprocessing
image = { version = "0.25", default-features = false, features = ["jpeg", "png"] }

# Parallelism
rayon = "1.8"

[dev-dependencies]
criterion = "0.5"
rand = "0.8"
```

## Core Interfaces

### Embedding Trait

```rust
// src/embedding.rs

use ndarray::Array2;
use ruvector_core::Vector;

/// Trait for models that extract embeddings from images
pub trait EmbeddingExtractor: Send + Sync {
    /// Dimension of output embedding
    fn embedding_dim(&self) -> usize;

    /// Extract embedding from a single image
    fn extract(&self, image: &[u8], width: u32, height: u32) -> Result<Vector, CnnError>;

    /// Batch extraction for efficiency
    fn extract_batch(
        &self,
        images: &[(&[u8], u32, u32)],
    ) -> Result<Vec<Vector>, CnnError> {
        images.iter()
            .map(|(data, w, h)| self.extract(data, *w, *h))
            .collect()
    }
}

/// Configuration for embedding extraction
#[derive(Clone, Debug)]
pub struct EmbeddingConfig {
    /// Input image size (square)
    pub input_size: u32,
    /// Normalize embeddings to unit length
    pub normalize: bool,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Use INT8 quantization
    pub quantized: bool,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            input_size: 224,
            normalize: true,
            embedding_dim: 512,
            quantized: false,
        }
    }
}
```

### CNN Backbone Interface

```rust
// src/backbone/mod.rs

use ndarray::Array4;

/// Trait for CNN backbones
pub trait Backbone: Send + Sync {
    /// Forward pass returning feature maps
    fn forward(&self, input: &Array4<f32>) -> Array4<f32>;

    /// Output feature dimension (channels)
    fn output_channels(&self) -> usize;

    /// Expected input size
    fn input_size(&self) -> (usize, usize);
}

/// Factory for creating backbones
pub enum BackboneType {
    MobileNetV3Small,
    MobileNetV3Large,
    ShuffleNetV2_0_5,
    ShuffleNetV2_1_0,
    EfficientNetB0,
}

pub fn create_backbone(
    backbone_type: BackboneType,
    weights_path: Option<&str>,
) -> Result<Box<dyn Backbone>, CnnError> {
    match backbone_type {
        BackboneType::MobileNetV3Small => {
            Ok(Box::new(MobileNetV3::small(weights_path)?))
        }
        BackboneType::MobileNetV3Large => {
            Ok(Box::new(MobileNetV3::large(weights_path)?))
        }
        BackboneType::ShuffleNetV2_0_5 => {
            Ok(Box::new(ShuffleNetV2::new(0.5, weights_path)?))
        }
        BackboneType::ShuffleNetV2_1_0 => {
            Ok(Box::new(ShuffleNetV2::new(1.0, weights_path)?))
        }
        BackboneType::EfficientNetB0 => {
            Ok(Box::new(EfficientNet::b0(weights_path)?))
        }
    }
}
```

## Integration with RuVector Index

### Vector Index Integration

```rust
// Integration with ruvector-core

use ruvector_core::{Index, Vector, SearchResult};
use ruvector_cnn::{EmbeddingExtractor, MobileNetEmbedder};

/// Image index combining CNN embeddings with vector search
pub struct ImageIndex {
    embedder: Box<dyn EmbeddingExtractor>,
    index: Index,
    metadata: Vec<ImageMetadata>,
}

pub struct ImageMetadata {
    pub id: u64,
    pub path: String,
    pub size: (u32, u32),
    pub format: ImageFormat,
}

impl ImageIndex {
    pub fn new(config: ImageIndexConfig) -> Result<Self, Error> {
        let embedder = create_embedder(&config)?;
        let index = Index::new(IndexConfig {
            dimension: embedder.embedding_dim(),
            metric: Metric::InnerProduct, // Cosine on normalized vectors
            index_type: config.index_type,
        })?;

        Ok(Self {
            embedder,
            index,
            metadata: Vec::new(),
        })
    }

    /// Add image to index
    pub fn add_image(
        &mut self,
        id: u64,
        image_data: &[u8],
        width: u32,
        height: u32,
        path: String,
    ) -> Result<(), Error> {
        // Extract embedding
        let embedding = self.embedder.extract(image_data, width, height)?;

        // Add to vector index
        self.index.add(id, embedding)?;

        // Store metadata
        self.metadata.push(ImageMetadata {
            id,
            path,
            size: (width, height),
            format: detect_format(image_data),
        });

        Ok(())
    }

    /// Search for similar images
    pub fn search(
        &self,
        query_image: &[u8],
        width: u32,
        height: u32,
        k: usize,
    ) -> Result<Vec<ImageSearchResult>, Error> {
        // Extract query embedding
        let query_embedding = self.embedder.extract(query_image, width, height)?;

        // Vector search
        let results = self.index.search(&query_embedding, k)?;

        // Attach metadata
        Ok(results
            .into_iter()
            .map(|r| ImageSearchResult {
                id: r.id,
                score: r.score,
                metadata: self.get_metadata(r.id),
            })
            .collect())
    }

    /// Batch add for efficiency
    pub fn add_images_batch(
        &mut self,
        images: &[(u64, &[u8], u32, u32, String)],
    ) -> Result<(), Error> {
        use rayon::prelude::*;

        // Parallel embedding extraction
        let embeddings: Vec<_> = images
            .par_iter()
            .map(|(_, data, w, h, _)| {
                self.embedder.extract(data, *w, *h)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Batch add to index
        for ((id, _, w, h, path), embedding) in images.iter().zip(embeddings) {
            self.index.add(*id, embedding)?;
            self.metadata.push(ImageMetadata {
                id: *id,
                path: path.clone(),
                size: (*w, *h),
                format: ImageFormat::Unknown,
            });
        }

        Ok(())
    }
}
```

## Contrastive Learning Integration

### Training Pipeline

```rust
// src/contrastive/mod.rs

use ndarray::Array2;

/// InfoNCE / NT-Xent loss for contrastive learning
pub struct InfoNCELoss {
    temperature: f32,
}

impl InfoNCELoss {
    pub fn new(temperature: f32) -> Self {
        Self { temperature }
    }

    /// Compute loss for positive pairs
    /// embeddings: [2N, D] where (i, i+N) are positive pairs
    pub fn forward(&self, embeddings: &Array2<f32>) -> f32 {
        let batch_size = embeddings.nrows() / 2;
        let dim = embeddings.ncols();

        // Compute similarity matrix
        let sim = self.compute_similarity_matrix(embeddings);

        // InfoNCE loss
        let mut loss = 0.0;
        for i in 0..batch_size {
            let positive_idx = i + batch_size;

            // Positive similarity
            let pos_sim = sim[[i, positive_idx]] / self.temperature;

            // Denominator: sum over all negatives + positive
            let mut denom = 0.0f32;
            for j in 0..(2 * batch_size) {
                if j != i {
                    denom += (sim[[i, j]] / self.temperature).exp();
                }
            }

            loss -= pos_sim - denom.ln();
        }

        loss / batch_size as f32
    }

    #[target_feature(enable = "avx2")]
    unsafe fn compute_similarity_matrix(&self, embeddings: &Array2<f32>) -> Array2<f32> {
        let n = embeddings.nrows();
        let mut sim = Array2::zeros((n, n));

        // Cosine similarity = dot product of normalized vectors
        for i in 0..n {
            for j in i..n {
                let a = embeddings.row(i);
                let b = embeddings.row(j);
                let dot = dot_product_avx2(a.as_slice().unwrap(), b.as_slice().unwrap());
                sim[[i, j]] = dot;
                sim[[j, i]] = dot;
            }
        }

        sim
    }
}

/// Triplet loss for metric learning
pub struct TripletLoss {
    margin: f32,
}

impl TripletLoss {
    pub fn new(margin: f32) -> Self {
        Self { margin }
    }

    /// anchor, positive, negative: [N, D]
    pub fn forward(
        &self,
        anchor: &Array2<f32>,
        positive: &Array2<f32>,
        negative: &Array2<f32>,
    ) -> f32 {
        let batch_size = anchor.nrows();
        let mut loss = 0.0;

        for i in 0..batch_size {
            let d_pos = euclidean_distance(
                anchor.row(i).as_slice().unwrap(),
                positive.row(i).as_slice().unwrap(),
            );
            let d_neg = euclidean_distance(
                anchor.row(i).as_slice().unwrap(),
                negative.row(i).as_slice().unwrap(),
            );

            loss += (d_pos - d_neg + self.margin).max(0.0);
        }

        loss / batch_size as f32
    }
}
```

### Data Augmentation for Contrastive Learning

```rust
// src/contrastive/augmentation.rs

use image::{DynamicImage, ImageBuffer, Rgb};
use rand::Rng;

/// Augmentation pipeline for SimCLR/MoCo
pub struct ContrastiveAugmentation {
    crop_scale: (f32, f32),      // Random crop scale range
    flip_prob: f32,              // Horizontal flip probability
    color_jitter: ColorJitter,   // Color augmentation
    blur_prob: f32,              // Gaussian blur probability
}

impl ContrastiveAugmentation {
    pub fn simclr_default() -> Self {
        Self {
            crop_scale: (0.08, 1.0),
            flip_prob: 0.5,
            color_jitter: ColorJitter {
                brightness: 0.8,
                contrast: 0.8,
                saturation: 0.8,
                hue: 0.2,
            },
            blur_prob: 0.5,
        }
    }

    /// Generate two augmented views of the same image
    pub fn generate_pair(&self, image: &DynamicImage) -> (DynamicImage, DynamicImage) {
        let mut rng = rand::thread_rng();

        let view1 = self.augment(image, &mut rng);
        let view2 = self.augment(image, &mut rng);

        (view1, view2)
    }

    fn augment<R: Rng>(&self, image: &DynamicImage, rng: &mut R) -> DynamicImage {
        let mut img = image.clone();

        // Random resized crop
        img = self.random_resized_crop(&img, rng);

        // Horizontal flip
        if rng.gen::<f32>() < self.flip_prob {
            img = img.fliph();
        }

        // Color jitter
        img = self.apply_color_jitter(&img, rng);

        // Gaussian blur
        if rng.gen::<f32>() < self.blur_prob {
            img = self.gaussian_blur(&img);
        }

        img
    }

    fn random_resized_crop<R: Rng>(
        &self,
        image: &DynamicImage,
        rng: &mut R,
    ) -> DynamicImage {
        let (w, h) = (image.width(), image.height());
        let area = w * h;

        // Random scale
        let scale = rng.gen_range(self.crop_scale.0..self.crop_scale.1);
        let target_area = (area as f32 * scale) as u32;

        // Random aspect ratio (3/4 to 4/3)
        let aspect = rng.gen_range(0.75..1.333);

        let crop_w = ((target_area as f32 * aspect).sqrt() as u32).min(w);
        let crop_h = ((target_area as f32 / aspect).sqrt() as u32).min(h);

        let x = rng.gen_range(0..=(w - crop_w));
        let y = rng.gen_range(0..=(h - crop_h));

        image.crop_imm(x, y, crop_w, crop_h)
    }
}
```

## ONNX Model Loading

### Using tract-onnx

```rust
// src/backbone/onnx.rs

use tract_onnx::prelude::*;
use ndarray::Array4;

pub struct OnnxBackbone {
    model: SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>,
    input_size: (usize, usize),
    output_channels: usize,
}

impl OnnxBackbone {
    pub fn load(model_path: &str) -> Result<Self, CnnError> {
        let model = tract_onnx::onnx()
            .model_for_path(model_path)?
            .with_input_fact(0, f32::fact(&[1, 3, 224, 224]).into())?
            .into_optimized()?
            .into_runnable()?;

        // Infer output shape
        let dummy = tract_ndarray::Array4::<f32>::zeros((1, 3, 224, 224));
        let output = model.run(tvec!(dummy.into()))?;
        let output_channels = output[0].shape()[1];

        Ok(Self {
            model,
            input_size: (224, 224),
            output_channels,
        })
    }
}

impl Backbone for OnnxBackbone {
    fn forward(&self, input: &Array4<f32>) -> Array4<f32> {
        let input_tract: tract_ndarray::ArrayD<f32> = input
            .clone()
            .into_dyn();

        let result = self.model
            .run(tvec!(input_tract.into()))
            .expect("Forward pass failed");

        result[0]
            .to_array_view::<f32>()
            .expect("Output conversion failed")
            .to_owned()
            .into_dimensionality::<ndarray::Ix4>()
            .expect("Shape conversion failed")
    }

    fn output_channels(&self) -> usize {
        self.output_channels
    }

    fn input_size(&self) -> (usize, usize) {
        self.input_size
    }
}
```

## Performance Benchmarks

### Target Performance

| Component | Target Latency | Throughput |
|-----------|---------------|------------|
| Image resize (224x224) | <1ms | >1000 img/s |
| MobileNet-V3 Small forward | <5ms | >200 img/s |
| ShuffleNet-V2 forward | <3ms | >300 img/s |
| L2 normalization (512-d) | <0.01ms | >100k vec/s |
| HNSW search (1M vectors) | <1ms | >1000 qps |

### Benchmark Suite

```rust
// benches/embedding_bench.rs

use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use ruvector_cnn::{MobileNetEmbedder, EmbeddingExtractor};

fn benchmark_embedding_extraction(c: &mut Criterion) {
    let embedder = MobileNetEmbedder::new_v3_small().unwrap();

    // Generate random test images
    let images: Vec<Vec<u8>> = (0..100)
        .map(|_| random_image_rgb(224, 224))
        .collect();

    let mut group = c.benchmark_group("embedding_extraction");

    group.bench_function("mobilenet_v3_small", |b| {
        b.iter(|| {
            for img in &images {
                let _ = embedder.extract(img, 224, 224);
            }
        })
    });

    group.finish();
}

fn benchmark_simd_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("simd_operations");

    for size in [256, 512, 1024, 2048].iter() {
        let a: Vec<f32> = (0..*size).map(|i| i as f32 / *size as f32).collect();
        let b: Vec<f32> = (0..*size).map(|i| (size - i) as f32 / *size as f32).collect();

        group.bench_with_input(
            BenchmarkId::new("dot_product_scalar", size),
            size,
            |bench, _| {
                bench.iter(|| dot_product_scalar(&a, &b))
            },
        );

        group.bench_with_input(
            BenchmarkId::new("dot_product_avx2", size),
            size,
            |bench, _| {
                bench.iter(|| unsafe { dot_product_avx2(&a, &b) })
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_embedding_extraction, benchmark_simd_operations);
criterion_main!(benches);
```

## Usage Examples

### Basic Image Embedding

```rust
use ruvector_cnn::{MobileNetEmbedder, EmbeddingConfig, EmbeddingExtractor};
use ruvector_core::Index;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create embedder with MobileNet-V3 Small
    let embedder = MobileNetEmbedder::new(EmbeddingConfig {
        input_size: 224,
        embedding_dim: 512,
        normalize: true,
        quantized: false,
    })?;

    // Load and embed an image
    let image_data = std::fs::read("image.jpg")?;
    let (width, height) = get_image_dimensions(&image_data)?;

    let embedding = embedder.extract(&image_data, width, height)?;

    println!("Embedding dimension: {}", embedding.len());
    println!("First 5 values: {:?}", &embedding.as_slice()[..5]);

    Ok(())
}
```

### Building an Image Search Index

```rust
use ruvector_cnn::ImageIndex;
use std::path::Path;

fn build_image_index(image_dir: &Path) -> Result<ImageIndex, Box<dyn std::error::Error>> {
    let mut index = ImageIndex::new(ImageIndexConfig {
        backbone: BackboneType::MobileNetV3Small,
        index_type: IndexType::HNSW { ef_construction: 200, m: 16 },
        ..Default::default()
    })?;

    // Add all images from directory
    let mut id = 0u64;
    for entry in std::fs::read_dir(image_dir)? {
        let path = entry?.path();
        if path.extension().map_or(false, |e| e == "jpg" || e == "png") {
            let data = std::fs::read(&path)?;
            let (w, h) = get_image_dimensions(&data)?;

            index.add_image(id, &data, w, h, path.to_string_lossy().to_string())?;
            id += 1;
        }
    }

    println!("Indexed {} images", id);
    Ok(index)
}

fn search_similar(index: &ImageIndex, query_path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let query_data = std::fs::read(query_path)?;
    let (w, h) = get_image_dimensions(&query_data)?;

    let results = index.search(&query_data, w, h, 10)?;

    println!("Top 10 similar images:");
    for (i, result) in results.iter().enumerate() {
        println!("  {}. {} (score: {:.4})", i + 1, result.metadata.path, result.score);
    }

    Ok(())
}
```

### Contrastive Pre-training

```rust
use ruvector_cnn::contrastive::{InfoNCELoss, ContrastiveAugmentation};
use ruvector_cnn::backbone::MobileNetV3;

fn pretrain_embedder(
    images: &[DynamicImage],
    epochs: usize,
    batch_size: usize,
) -> Result<MobileNetV3, Box<dyn std::error::Error>> {
    let augmentation = ContrastiveAugmentation::simclr_default();
    let loss_fn = InfoNCELoss::new(0.1); // temperature = 0.1
    let mut model = MobileNetV3::small_untrained()?;

    for epoch in 0..epochs {
        let mut epoch_loss = 0.0;

        for batch in images.chunks(batch_size) {
            // Generate augmented pairs
            let pairs: Vec<_> = batch
                .iter()
                .flat_map(|img| {
                    let (v1, v2) = augmentation.generate_pair(img);
                    vec![v1, v2]
                })
                .collect();

            // Forward pass
            let embeddings = model.forward_batch(&pairs)?;

            // Compute loss
            let loss = loss_fn.forward(&embeddings);
            epoch_loss += loss;

            // Backward pass and update (pseudo-code)
            // let gradients = loss.backward();
            // optimizer.step(&mut model, gradients);
        }

        println!("Epoch {}: loss = {:.4}", epoch + 1, epoch_loss / images.len() as f32);
    }

    Ok(model)
}
```

## SONA Integration (Self-Optimizing Neural Architecture)

The `sona` crate provides adaptive learning capabilities that enhance CNN embeddings through trajectory-based learning.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    CNN + SONA Pipeline                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌───────────────┐    ┌──────────────┐          │
│  │  Image   │───▶│    CNN        │───▶│  Embedding   │          │
│  │  Input   │    │  Backbone     │    │  (512-d)     │          │
│  └──────────┘    └───────────────┘    └──────┬───────┘          │
│                                              │                   │
│                                              ▼                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    SONA Engine                             │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │  │
│  │  │ Micro-LoRA  │  │ Base-LoRA   │  │  EWC++          │    │  │
│  │  │ (rank=1-2)  │  │ (rank=8-16) │  │  (anti-forget)  │    │  │
│  │  └─────────────┘  └─────────────┘  └─────────────────┘    │  │
│  │                         │                                  │  │
│  │                         ▼                                  │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │              ReasoningBank                          │  │  │
│  │  │  • Pattern extraction from successful embeddings    │  │  │
│  │  │  • Similarity search for related contexts           │  │  │
│  │  │  • Trajectory storage for replay learning           │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### SONA-Enhanced Embedding Extraction

```rust
use sona::{SonaEngine, SonaConfig, TrajectoryBuilder};
use ruvector_cnn::{MobileNetEmbedder, EmbeddingExtractor};

/// CNN embedder enhanced with SONA adaptive learning
pub struct SonaEnhancedEmbedder {
    cnn: MobileNetEmbedder,
    sona: SonaEngine,
}

impl SonaEnhancedEmbedder {
    pub fn new(cnn_config: EmbeddingConfig, sona_config: SonaConfig) -> Result<Self, Error> {
        Ok(Self {
            cnn: MobileNetEmbedder::new(cnn_config)?,
            sona: SonaEngine::new(sona_config),
        })
    }

    /// Extract embedding with SONA transformation
    pub fn extract_enhanced(&self, image_data: &[u8], w: u32, h: u32) -> Result<Vec<f32>, Error> {
        // Base CNN embedding
        let base_embedding = self.cnn.extract(image_data, w, h)?;

        // Apply SONA's learned Micro-LoRA transformation
        let mut enhanced = vec![0.0; base_embedding.len()];
        self.sona.apply_micro_lora(&base_embedding, &mut enhanced);

        // L2 normalize
        l2_normalize(&mut enhanced);

        Ok(enhanced)
    }

    /// Record contrastive pair for trajectory learning
    pub fn record_contrastive_pair(
        &self,
        anchor: &[u8],
        positive: &[u8],
        w: u32,
        h: u32,
    ) -> Result<(), Error> {
        let anchor_emb = self.cnn.extract(anchor, w, h)?;
        let positive_emb = self.cnn.extract(positive, w, h)?;

        // Begin trajectory with anchor
        let mut trajectory = self.sona.begin_trajectory(anchor_emb.clone());

        // Add positive as a step with high similarity signal
        let similarity = cosine_similarity(&anchor_emb, &positive_emb);
        trajectory.add_step(positive_emb, vec![], similarity);

        // End trajectory - triggers SONA learning
        self.sona.end_trajectory(trajectory, similarity);

        Ok(())
    }

    /// Query ReasoningBank for similar patterns
    pub fn find_similar_patterns(&self, embedding: &[f32], k: usize) -> Vec<LearnedPattern> {
        self.sona.reasoning_bank().search_similar(embedding, k)
    }
}
```

### Trajectory-Based Contrastive Learning

```rust
/// Contrastive learning loop with SONA integration
pub fn contrastive_train_with_sona(
    embedder: &SonaEnhancedEmbedder,
    images: &[DynamicImage],
    augmentation: &ContrastiveAugmentation,
    epochs: usize,
) -> Result<(), Error> {
    for epoch in 0..epochs {
        for image in images {
            // Generate two augmented views
            let (view1, view2) = augmentation.generate_pair(image);

            // Record as contrastive pair - SONA learns the invariance
            embedder.record_contrastive_pair(
                &view1.as_bytes(),
                &view2.as_bytes(),
                view1.width(),
                view1.height(),
            )?;
        }

        // Periodic consolidation to prevent forgetting
        if epoch % 10 == 0 {
            embedder.sona.consolidate_memory();
        }

        println!("Epoch {}: SONA patterns = {}", epoch + 1,
            embedder.sona.reasoning_bank().pattern_count());
    }

    Ok(())
}
```

## Hyperbolic HNSW Integration

Leverage `ruvector-hyperbolic-hnsw` for hierarchical image search where semantic concepts have natural tree-like relationships.

### Why Hyperbolic for Images?

| Property | Euclidean | Hyperbolic |
|----------|-----------|------------|
| Hierarchy | Poor | Excellent |
| Example | "car" vs "sedan" | "vehicle" → "car" → "sedan" |
| Volume growth | Polynomial | Exponential |
| Tree embedding | Distorted | Isometric |

### Poincaré Ball Projection

```rust
use ruvector_hyperbolic_hnsw::{
    HyperbolicHnsw, PoincareBall, HyperbolicConfig,
    exp_map_zero, log_map_zero, poincare_distance,
};

/// Project Euclidean CNN embedding to Poincaré ball
pub struct HyperbolicImageIndex {
    embedder: MobileNetEmbedder,
    poincare: PoincareBall,
    hnsw: HyperbolicHnsw,
}

impl HyperbolicImageIndex {
    pub fn new(config: HyperbolicConfig) -> Result<Self, Error> {
        Ok(Self {
            embedder: MobileNetEmbedder::new_v3_small()?,
            poincare: PoincareBall::new(config.curvature),
            hnsw: HyperbolicHnsw::new(config)?,
        })
    }

    /// Add image with hyperbolic embedding
    pub fn add_image(&mut self, id: u64, image: &[u8], w: u32, h: u32) -> Result<(), Error> {
        // Extract Euclidean embedding
        let euclidean_emb = self.embedder.extract(image, w, h)?;

        // Project to Poincaré ball using exponential map at origin
        let hyperbolic_emb = exp_map_zero(&euclidean_emb, self.poincare.curvature());

        // Add to hyperbolic HNSW index
        self.hnsw.add(id, &hyperbolic_emb)?;

        Ok(())
    }

    /// Search using hyperbolic distance
    pub fn search(&self, query_image: &[u8], w: u32, h: u32, k: usize) -> Result<Vec<SearchResult>, Error> {
        let euclidean_emb = self.embedder.extract(query_image, w, h)?;
        let hyperbolic_query = exp_map_zero(&euclidean_emb, self.poincare.curvature());

        self.hnsw.search(&hyperbolic_query, k)
    }

    /// Hierarchical search: find images at specific semantic level
    pub fn search_at_level(
        &self,
        query: &[f32],
        target_norm: f32,  // Norm ~ hierarchy level in Poincaré ball
        k: usize,
    ) -> Result<Vec<SearchResult>, Error> {
        // In Poincaré ball, points near origin are "general" (e.g., "animal")
        // Points near boundary are "specific" (e.g., "golden retriever puppy")
        let scaled_query = self.poincare.scale_to_norm(query, target_norm);
        self.hnsw.search(&scaled_query, k)
    }
}
```

### Mixed-Curvature Embedding (via ruvector-math)

```rust
use ruvector_math::product_manifold::{ProductManifold, ManifoldType};

/// Embed images in mixed-curvature space
/// - Euclidean: color/texture features
/// - Hyperbolic: semantic hierarchy
/// - Spherical: cyclical patterns (time of day, seasons)
pub struct MixedCurvatureImageEmbedder {
    cnn: MobileNetEmbedder,
    manifold: ProductManifold,
}

impl MixedCurvatureImageEmbedder {
    pub fn new() -> Result<Self, Error> {
        Ok(Self {
            cnn: MobileNetEmbedder::new(EmbeddingConfig {
                embedding_dim: 512,
                ..Default::default()
            })?,
            // 256-dim Euclidean + 200-dim Hyperbolic + 56-dim Spherical = 512
            manifold: ProductManifold::new(256, 200, 56),
        })
    }

    pub fn embed(&self, image: &[u8], w: u32, h: u32) -> Result<ProductEmbedding, Error> {
        let flat_emb = self.cnn.extract(image, w, h)?;

        // Split into components
        let euclidean = flat_emb[..256].to_vec();
        let hyperbolic = self.manifold.project_hyperbolic(&flat_emb[256..456]);
        let spherical = self.manifold.project_spherical(&flat_emb[456..]);

        Ok(ProductEmbedding { euclidean, hyperbolic, spherical })
    }

    pub fn distance(&self, a: &ProductEmbedding, b: &ProductEmbedding) -> f32 {
        self.manifold.distance(
            &[&a.euclidean[..], &a.hyperbolic[..], &a.spherical[..]].concat(),
            &[&b.euclidean[..], &b.hyperbolic[..], &b.spherical[..]].concat(),
        ).unwrap()
    }
}
```

## CNN vs GNN: Complementary Roles

RuVector has both `ruvector-cnn` (proposed) and `ruvector-gnn` (existing). They serve different purposes:

### Comparison

| Aspect | CNN (`ruvector-cnn`) | GNN (`ruvector-gnn`) |
|--------|---------------------|---------------------|
| **Input** | Grid-structured (images) | Graph-structured (relations) |
| **Learns** | Spatial hierarchies | Relational patterns |
| **Invariance** | Translation invariance | Permutation invariance |
| **Key ops** | Convolution, pooling | Message passing, aggregation |
| **Use case** | Image → embedding | Embedding → reasoning |

### Combined Pipeline

```
┌────────────────────────────────────────────────────────────────┐
│                    Multimodal Pipeline                         │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Images          ┌─────────┐                                   │
│  ─────────────▶  │   CNN   │ ──▶ Image Embeddings              │
│                  └─────────┘           │                        │
│                                        │                        │
│                                        ▼                        │
│                            ┌─────────────────────┐             │
│  Relationships             │  Image Similarity   │             │
│  (co-occurrence,     ───▶  │  Graph Construction │             │
│   captions, etc.)          └──────────┬──────────┘             │
│                                       │                        │
│                                       ▼                        │
│                            ┌─────────────────────┐             │
│                            │        GNN          │             │
│                            │  (ruvector-gnn)     │             │
│                            │  • Message passing  │             │
│                            │  • Cross-image      │             │
│                            │    reasoning        │             │
│                            └──────────┬──────────┘             │
│                                       │                        │
│                                       ▼                        │
│                            ┌─────────────────────┐             │
│                            │  Enhanced Embeddings│             │
│                            │  (context-aware)    │             │
│                            └─────────────────────┘             │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

### Code Example: CNN + GNN

```rust
use ruvector_cnn::{MobileNetEmbedder, EmbeddingExtractor};
use ruvector_gnn::{GraphAttentionNetwork, GnnConfig};

pub struct MultimodalImageReasoner {
    cnn: MobileNetEmbedder,
    gnn: GraphAttentionNetwork,
}

impl MultimodalImageReasoner {
    /// Extract image embeddings and reason over relationships
    pub fn reason_over_images(
        &self,
        images: &[(&[u8], u32, u32)],
        edges: &[(usize, usize)],  // Image relationships
    ) -> Result<Vec<Vec<f32>>, Error> {
        // Step 1: CNN extracts initial embeddings
        let embeddings: Vec<_> = images
            .iter()
            .map(|(data, w, h)| self.cnn.extract(data, *w, *h))
            .collect::<Result<Vec<_>, _>>()?;

        // Step 2: Build graph from relationships
        let graph = self.build_graph(&embeddings, edges);

        // Step 3: GNN refines embeddings based on relationships
        let refined = self.gnn.forward(&graph)?;

        Ok(refined)
    }

    fn build_graph(&self, nodes: &[Vec<f32>], edges: &[(usize, usize)]) -> Graph {
        // ... construct graph structure
    }
}
```

## Future Enhancements

### Phase 1: Core Implementation
- [ ] Basic MobileNet-V3 Small/Large in Rust
- [ ] SIMD-optimized convolution (AVX2)
- [ ] Integration with RuVector index
- [ ] ONNX model loading support

### Phase 2: Performance Optimization
- [ ] INT8 quantization support
- [ ] AVX-512 optimizations
- [ ] Winograd convolution for 3x3 kernels
- [ ] Multi-threaded batch processing

### Phase 3: Training Support
- [ ] InfoNCE/NT-Xent loss implementation
- [ ] Data augmentation pipeline
- [ ] Gradient computation
- [ ] Model checkpointing

### Phase 4: Advanced Features
- [ ] ShuffleNet and EfficientNet backbones
- [ ] Domain-specific fine-tuning
- [ ] Hard negative mining
- [ ] Multi-modal embedding support

### Phase 5: SONA & Hyperbolic Integration
- [ ] SONA trajectory learning for contrastive pairs
- [ ] ReasoningBank pattern extraction from embeddings
- [ ] Hyperbolic HNSW integration
- [ ] Mixed-curvature product manifold support
- [ ] CNN + GNN combined pipeline

## References

1. [FAISS Library](https://github.com/facebookresearch/faiss)
2. [FAISS Documentation](https://faiss.ai/index.html)
3. [Embedding with FAISS and HuggingFace](https://huggingface.co/learn/cookbook/en/faiss_with_hf_datasets_and_clip)
4. [LangChain FAISS Integration](https://python.langchain.com/docs/integrations/vectorstores/faiss/)
5. [SimCLR Paper](https://arxiv.org/abs/2002.05709)
6. [MoCo Paper](https://arxiv.org/abs/1911.05722)
7. [MobileNet-V3](https://arxiv.org/abs/1905.02244)
8. [ShuffleNet](https://arxiv.org/abs/1707.01083)
9. [tract-onnx](https://github.com/sonos/tract)
