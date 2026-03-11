# ADR-089: CNN Browser Demo for GitHub Pages

## Status
Accepted

## Context
The `@ruvector/cnn` npm package provides WASM-based CNN feature extraction for browsers. To showcase its capabilities and provide a hands-on experience, we need an interactive demo deployable to GitHub Pages.

## Decision
Create an interactive browser demo with:

### Features
1. **Image Upload** - Drag & drop or file picker
2. **Camera Capture** - Real-time webcam integration
3. **Feature Extraction** - Visual display of embedding vectors
4. **Similarity Search** - Compare images and show similarity scores
5. **Live Heatmap** - Feature activation visualization
6. **Performance Metrics** - Real-time latency display

### Technical Approach
- Single HTML file with embedded CSS/JS for easy deployment
- ES modules loading WASM directly from npm CDN (unpkg/jsdelivr)
- Canvas-based visualizations
- No build step required

### Deployment
- GitHub Pages via `docs/demo/cnn/` directory
- Direct URL: `https://ruvnet.github.io/ruvector/demo/cnn/`

## Consequences

### Positive
- Zero-install demo experience
- Showcases real WASM performance (~5ms/image)
- Educational tool for understanding embeddings
- Marketing asset for the package

### Negative
- CDN dependency for WASM module
- Browser compatibility limited to WASM-supporting browsers

## Implementation
- `docs/demo/cnn/index.html` - Main demo page
- Uses `@ruvector/cnn` from unpkg CDN
