# exo-manifold: Learned Manifold Engine

Continuous manifold storage using implicit neural representations (SIREN networks) for the EXO-AI cognitive substrate.

## Overview

Instead of discrete vector storage, memories are encoded as continuous functions on a learned manifold using SIREN (Sinusoidal Representation Networks).

## Key Features

### 1. **Gradient Descent Retrieval** (`src/retrieval.rs`)
- Query via optimization toward high-relevance regions
- Implements ManifoldRetrieve algorithm from PSEUDOCODE.md
- Converges to semantically relevant patterns

### 2. **Continuous Deformation** (`src/deformation.rs`)
- No discrete insert operations
- Manifold weights updated via gradient descent
- Deformation proportional to pattern salience

### 3. **Strategic Forgetting** (`src/forgetting.rs`)
- Identify low-salience regions
- Apply Gaussian smoothing kernel
- Prune near-zero weights

### 4. **SIREN Network** (`src/network.rs`)
- Sinusoidal activation functions
- Specialized initialization for implicit functions
- Multi-layer architecture with Fourier features

## Architecture

```
Query → Gradient Descent → Converged Position → Extract Patterns
           ↓
     SIREN Network
   (Learned Manifold)
           ↓
    Relevance Field
```

## Implementation Status

✅ **Complete Implementation**:
- ManifoldEngine core structure
- SIREN neural network layers
- Gradient descent retrieval algorithm
- Continuous manifold deformation
- Strategic forgetting with smoothing
- Comprehensive tests

⚠️ **Known Issue**: 
The `burn` crate v0.14 has a compatibility issue with `bincode` v2.x. 

**Workaround**:
Add to workspace `Cargo.toml`:
```toml
[patch.crates-io]
bincode = { version = "1.3" }
```

Or wait for burn v0.15 which resolves this issue.

## Usage Example

```rust
use exo_manifold::ManifoldEngine;
use exo_core::{ManifoldConfig, Pattern};
use burn::backend::NdArray;

// Create engine
let config = ManifoldConfig::default();
let device = Default::default();
let mut engine = ManifoldEngine::<NdArray>::new(config, device);

// Deform manifold with pattern
let pattern = Pattern { /* ... */ };
engine.deform(pattern, 0.9)?;

// Retrieve similar patterns
let query = vec![/* embedding */];
let results = engine.retrieve(&query, 10)?;

// Strategic forgetting
engine.forget(0.5, 0.1)?;
```

## Algorithm Details

### Retrieval (from PSEUDOCODE.md)

```pseudocode
position = query_vector
FOR step IN 1..MAX_DESCENT_STEPS:
    relevance_field = manifold_network.forward(position)
    gradient = manifold_network.backward(relevance_field)
    position = position - LEARNING_RATE * gradient
    IF norm(gradient) < CONVERGENCE_THRESHOLD:
        BREAK
results = ExtractPatternsNear(position, k)
```

### Deformation (from PSEUDOCODE.md)

```pseudocode
embedding = Tensor(pattern.embedding)
current_relevance = manifold_network.forward(embedding)
target_relevance = salience
deformation_loss = (current_relevance - target_relevance)^2
smoothness_loss = ManifoldCurvatureRegularizer(manifold_network)
total_loss = deformation_loss + LAMBDA * smoothness_loss
gradients = total_loss.backward()
optimizer.step(gradients)
```

### Forgetting (from PSEUDOCODE.md)

```pseudocode
FOR region IN manifold_network.sample_regions():
    avg_salience = ComputeAverageSalience(region)
    IF avg_salience < salience_threshold:
        ForgetKernel = GaussianKernel(sigma=decay_rate)
        manifold_network.apply_kernel(region, ForgetKernel)
manifold_network.prune_weights(threshold=1e-6)
```

## Dependencies

- `exo-core`: Core types and traits
- `burn`: Deep learning framework
- `burn-ndarray`: NdArray backend
- `ndarray`: N-dimensional arrays
- `parking_lot`: Lock-free data structures

## Testing

```bash
cargo test -p exo-manifold
```

## References

- SIREN: "Implicit Neural Representations with Periodic Activation Functions" (Sitzmann et al., 2020)
- EXO-AI Architecture: `../../architecture/ARCHITECTURE.md`
- Pseudocode: `../../architecture/PSEUDOCODE.md`
