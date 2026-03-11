# CNN State-of-the-Art: Contrastive Learning Overview

## Executive Summary

This document surveys the current state-of-the-art in contrastive learning for CNNs, focusing on self-supervised methods that can generate high-quality embeddings without labeled data. These techniques are particularly relevant for RuVector's vector similarity search capabilities.

## Key Contrastive Learning Frameworks

### 1. SimCLR (Simple Contrastive Learning of Visual Representations)

**Paper**: [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709) (Google, 2020)

**Architecture**:
```
Image -> Data Augmentation (2 views) -> CNN Encoder -> Projection Head -> Contrastive Loss
```

**Key Innovations**:
- Composition of data augmentations is critical for effective learning
- Learnable nonlinear projection head between representation and loss
- Benefits from larger batch sizes (4096-8192) and longer training

**Performance**: 76.5% top-1 accuracy on ImageNet (linear evaluation)

**Limitations**:
- Requires large batch sizes for sufficient negative samples
- Computationally expensive (many GPU hours)

### 2. MoCo (Momentum Contrast)

**Paper**: Momentum Contrast for Unsupervised Visual Representation Learning (Meta AI)

**Key Innovation**: Dynamic memory queue decouples batch size from number of negatives

**Architecture**:
```
Query Encoder -> Current batch features
Momentum Encoder -> Memory Queue (65536 negatives)
```

**Advantages over SimCLR**:
- Works with smaller batch sizes (256)
- Memory-efficient via queue mechanism
- Momentum update prevents encoder collapse

**MoCo v3 Performance**: Approaches supervised learning baselines

### 3. BYOL (Bootstrap Your Own Latent)

**Paper**: Bootstrap Your Own Latent (DeepMind)

**Key Innovation**: Eliminates negative samples entirely

**Architecture**:
```
Online Network: Encoder -> Projector -> Predictor
Target Network: Encoder -> Projector (momentum updated)
```

**Loss**: L2 distance between online prediction and target projection

**Advantages**:
- No need for large batch sizes
- No negative sampling required
- Avoids false negative problem
- More stable training

**Considerations**: Requires careful architecture design to prevent collapse

### 4. SwAV (Swapped Assignments between Views)

**Paper**: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments

**Key Innovation**: Clustering-based approach with swapped assignments

**Architecture**:
```
Image Views -> Encoder -> Prototype Assignment
Swapped Prediction: View1 predicts View2's cluster, vice versa
```

**Performance**: 75.30% top-1 ImageNet (outperforms direct comparison methods)

**Advantages**:
- No large batch or memory bank needed
- Multi-crop augmentation strategy
- Online clustering is efficient

## Contrastive Loss Functions

### InfoNCE / NT-Xent Loss

The **Normalized Temperature-scaled Cross Entropy (NT-Xent)** loss, also known as InfoNCE, is the standard loss for contrastive learning:

```
L = -log(exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ))
```

**Components**:
- `sim(z_i, z_j)`: Similarity function (cosine or dot product)
- `τ`: Temperature parameter (typically 0.1-0.5)
- Numerator: Positive pair similarity
- Denominator: Sum over all pairs (1 positive + N-1 negatives)

**Temperature Effects**:
- Lower τ (0.1): Sharper distributions, focus on hard negatives
- Higher τ (0.5): Smoother distributions, more uniform gradients

**Implementation**:
```rust
fn info_nce_loss(embeddings: &[Vec<f32>], temperature: f32) -> f32 {
    // For each positive pair (i, j):
    // 1. Compute cosine similarity between all pairs
    // 2. Apply temperature scaling
    // 3. Compute softmax cross-entropy
    // 4. Average over batch
}
```

### Triplet Loss

Classic metric learning loss:

```
L = max(0, d(a, p) - d(a, n) + margin)
```

**Components**:
- Anchor (a), Positive (p), Negative (n) triplets
- Margin (typically 0.2-1.0)

**Hard Negative Mining**: Critical for training efficiency

**Use Cases**: Face verification (FaceNet), fine-grained similarity

### Comparison Table

| Loss | Negatives Required | Batch Size | Training Stability | Best For |
|------|-------------------|------------|-------------------|----------|
| NT-Xent | Yes (in-batch) | Large (4096+) | High | Self-supervised pretraining |
| InfoNCE | Yes | Medium-Large | High | General contrastive |
| Triplet | Yes (mined) | Any | Moderate | Metric learning |
| BYOL Loss | No | Any | Requires care | No-negative scenarios |

## 2024-2025 State-of-the-Art Performance

### ImageNet Benchmarks

| Method | Architecture | Top-1 Accuracy | Notes |
|--------|-------------|----------------|-------|
| MAE ViT-Huge | ViT-H | 87.8% | Masked autoencoder (generative) |
| ReLICv2 | ResNet-50 | 77.1% | Contrastive |
| SwAV | ResNet-50 | 75.3% | Clustering-based |
| SimCLR | ResNet-50 | 76.5% | Requires large batch |
| MoCo v3 | ViT-B | 76.7% | Momentum contrast |
| BYOL | ResNet-50 | 74.3% | No negatives |

### Key Insights from Recent Research

1. **Data augmentation strategy matters more than SSL paradigm**
2. **Vision Transformers benefit significantly from SSL pre-training**
3. **Masked image modeling provides simplicity and efficiency benefits**
4. **Scaling to larger models and datasets improves performance**
5. **SSL has become de-facto standard for ImageNet pre-training**

## Efficient Architectures for Embedding Extraction

### MobileNet Series

**MobileNet-V2/V3**: Depthwise separable convolutions

```
Standard Conv: H×W×C_in×K×K×C_out multiplications
Depthwise Separable: H×W×C_in×K×K + H×W×C_in×C_out
Reduction: ~8-9x fewer operations for 3×3 kernels
```

**Performance**: 98.15% accuracy on activity recognition (6MB model)

### ShuffleNet

**Key Innovation**: Channel shuffle after group convolution

**FLOPs**: 10-150 MFLOPs range

**Performance**: Surpasses MobileNet by 7.8% at ~40 MFLOPs

### EfficientNet

**Compound Scaling**: Balanced depth, width, resolution scaling

**Trade-off**: Higher accuracy but more parameters/MACs

**Best for**: When compute budget is flexible

## Practical Recommendations for RuVector

### Architecture Selection

| Use Case | Recommended Architecture | Notes |
|----------|-------------------------|-------|
| Real-time embedding | MobileNet-V3 Small | ~3ms inference |
| Balanced accuracy/speed | ShuffleNet-V2 | Good SIMD compatibility |
| Maximum quality | EfficientNet-B0 | INT8 quantizable |

### Pre-training Strategy

1. **For domain-specific data**:
   - Use SimCLR/MoCo with domain augmentations
   - Fine-tune projection head on similarity task

2. **For general embeddings**:
   - Start with ImageNet pre-trained weights
   - Extract features from penultimate layer

3. **For CPU deployment**:
   - MobileNet-V3 with INT8 quantization
   - Winograd convolution for 3x3 kernels

### Embedding Dimensions

| Dimension | Use Case | Trade-off |
|-----------|----------|-----------|
| 128-256 | Real-time search | Fast, compact |
| 512 | Balanced | Good quality/size |
| 1024-2048 | Maximum recall | Higher memory |

## Integration with Vector Search

### FAISS Integration Pattern

```rust
// 1. CNN extracts embedding
let embedding = cnn.forward(image); // [batch, 512]

// 2. L2 normalize for cosine similarity
let normalized = l2_normalize(embedding);

// 3. Add to FAISS index or RuVector
index.add(normalized);

// 4. Search uses inner product (= cosine after normalization)
let results = index.search(query, k);
```

### Recommended Index Types

| Dataset Size | Index Type | Build Time | Search Time |
|-------------|------------|------------|-------------|
| <10K | Flat | O(n) | O(n) |
| 10K-1M | IVF | O(n) | O(sqrt(n)) |
| >1M | HNSW | O(n log n) | O(log n) |

## References

1. [SimCLR Paper](https://arxiv.org/abs/2002.05709)
2. [Contrastive Learning Overview](https://lilianweng.github.io/posts/2021-05-31-contrastive/)
3. [LearnOpenCV: SimCLR and BYOL](https://learnopencv.com/contrastive-learning-simclr-and-byol-with-code-example/)
4. [Self-Supervised Learning Survey](https://www.ijert.org/advances-in-self-supervised-learning--a-comprehensive-review-of-contrastive-and-generative-approaches)
5. [InfoNCE PyTorch](https://github.com/RElbers/info-nce-pytorch)
6. [PyTorch Metric Learning Losses](https://kevinmusgrave.github.io/pytorch-metric-learning/losses/)
7. [FAISS Library](https://github.com/facebookresearch/faiss)
8. [ShuffleNet Paper](https://arxiv.org/abs/1707.01083)
