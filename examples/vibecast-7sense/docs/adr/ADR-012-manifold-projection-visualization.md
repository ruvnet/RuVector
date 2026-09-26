# ADR-012: Manifold Projection and Visualization Data Model

## Status
Accepted

## Date
2026-08-03

## Context

ADR-009 designed a visualization layer and nothing was built. The `sevensense-viz`
crate it specifies does not exist; the workspace contains no frontend source of any
kind. The API returns a `umap_url` field pointing at
`/api/v1/evidence/{id}/umap`, but no such route is registered, so the link 404s.

ADR-009 also specified a stack — `umap-rs`, `plotly`, `wasm-bindgen`, `web-sys` — that
appears in no `Cargo.toml`. Rather than resurrect a two-year-old dependency list, this
ADR supersedes ADR-009's technology choices while keeping its intent.

Two requirements have changed since ADR-009 was written:

1. **Live operation.** ADR-010 introduces streaming, so the view must accept points
   arriving continuously, not just render a finished batch.
2. **Interpretable axes.** ADR-011 produces named acoustic descriptors, so the view can
   encode meaning in colour and position instead of showing an unlabelled cloud.

## Decision

We separate *projection* (Rust, server-side) from *rendering* (browser), connected by
an explicit versioned data contract. The server never emits pixels; the client never
computes a projection.

### 1. Projection algorithm: PCA, then optional UMAP

We implement **PCA via randomized SVD** in `sevensense-vector` as the default
projection, with UMAP as an opt-in refinement.

This inverts ADR-009, which specified UMAP as the primary. The reasons are practical:

- PCA is deterministic. The same corpus projects to the same coordinates every time,
  so a saved viewport stays valid and two users discussing "the cluster on the left"
  mean the same cluster.
- PCA is **incrementally extensible**. Once components are fitted, projecting a new
  point is one matrix multiply — O(d·k) — which is what streaming requires. UMAP has no
  cheap exact out-of-sample extension; adding a point means either re-fitting or
  approximating.
- Randomized SVD on 1536 dimensions with `k = 3` and one power iteration is
  milliseconds for corpora up to ~10⁵ points.

UMAP preserves local neighbourhood structure better, which matters for finding
subtle dialect clusters. So it remains available for batch analysis via
`?method=umap`, fitted once over a snapshot, with new points appended by PCA-projected
approximation until the next re-fit. The mode is reported in the response so the client
can label the view honestly.

We do not depend on `umap-rs`; it is unmaintained. UMAP is implemented over the
existing HNSW index, which already provides the k-NN graph UMAP needs — this is a
genuine advantage of building it in-tree rather than pulling a crate that would rebuild
its own neighbour structure.

### 2. Hyperbolic projection

`sevensense-vector::hyperbolic` already implements the Poincaré ball (geodesic
distance, exponential and logarithmic maps, Möbius addition, `euclidean_to_poincare`).
It is unused by anything.

Taxonomic and dialect relationships are hierarchical, and hyperbolic space embeds
hierarchies with far less distortion than Euclidean space. We expose
`?method=poincare`, which projects to 3D via PCA and then maps into the Poincaré ball,
so hierarchical structure appears as radial depth — general calls near the origin,
specialized variants near the boundary.

### 3. Data contract

```jsonc
{
  "version": 1,
  "method": "pca",                  // pca | umap | poincare
  "dims": 3,
  "fitted_at": "2026-08-03T09:50:00Z",
  "explained_variance": [0.31, 0.19, 0.11],   // pca only
  "bounds": { "min": [-1.0,-1.0,-1.0], "max": [1.0,1.0,1.0] },
  "points": [
    {
      "id": "seg_01J…",
      "xyz": [0.12, -0.44, 0.87],
      "centroid_hz": 4200.0,        // colour channel  (ADR-011)
      "energy_db": -18.4,           // size channel
      "tonality": 0.72,
      "cluster": 3,
      "label": "Anthus trivialis",  // null when unlabelled
      "t_ms": 152340
    }
  ],
  "edges": [ { "a": 0, "b": 17, "w": 0.94 } ]   // k-NN, w = cosine similarity
}
```

Coordinates are normalized to `[-1, 1]` server-side. The client should not need to know
the data's scale to frame it, and normalizing once server-side avoids every client
reimplementing it slightly differently.

Edges are indices into `points`, not ids — for 10⁴ points with k=8 that is the
difference between roughly 160 KB and 2 MB of JSON.

### 4. Endpoints

| Route | Purpose |
|---|---|
| `GET /api/v1/projection?method=&dims=&limit=` | Fitted projection of the indexed corpus |
| `GET /api/v1/segments/:id/features` | Per-frame and summary features (ADR-011) |
| `GET /api/v1/segments/:id/spectrogram?format=png\|json` | Rendered or raw spectrogram |
| `GET /api/v1/evidence/:id/umap` | Evidence-pack neighbourhood (the currently-dangling link) |
| `WS /ws/stream` | Live `frame` / `segment` / `detection` events (ADR-010) |

### 5. Rendering

**Three.js** with `InstancedMesh`, served as a static bundle from the API.

deck.gl is the obvious alternative and is rejected: it is optimized for geospatial
layers, and its 3D scatter support is weaker than what a direct Three.js instanced
mesh gives. Plotly.js (ADR-009's choice) renders a few thousand 3D points acceptably
and then degrades badly; the target is 10⁴–10⁵.

One `InstancedMesh` holds every point; colour and scale go in instance attributes.
Draw calls stay constant as the corpus grows. Colour maps `centroid_hz` through
**viridis** — perceptually uniform and colourblind-safe, unlike the blue-green-red
ramp that a naive HSV mapping produces.

The 2D panels (feature scatter, radar) use plain `<canvas>` 2D. A charting library for
two static plots is not worth its bundle size.

### 6. Live updates

The client holds a fixed-capacity ring of the most recent N points (default 2000).
`detection` events append; the oldest point is evicted. Instance attributes are updated
in place and the buffer is marked dirty — no reallocation per frame.

Points fade by age via an `age` instance attribute, so recent activity is visually
distinct from history without a separate draw pass.

### 7. Accessibility and theming

The page honours `prefers-color-scheme` and `prefers-reduced-motion` (the latter
disables auto-rotation). Viridis is colourblind-safe, and every colour-encoded value is
also available numerically on hover, so colour is never the sole carrier of
information.

## Consequences

### Positive
- Deterministic default projection; stable coordinates across sessions.
- Streaming points cost one matrix multiply.
- The existing, unused hyperbolic module earns its place.
- One `InstancedMesh` scales to 10⁵ points.

### Negative
- PCA on 1536-D acoustic embeddings typically explains only ~30% of variance in three
  components. The view is a genuine simplification, and `explained_variance` is in the
  payload so the UI can say so rather than implying the picture is complete.
- Implementing UMAP in-tree is real work, deferred behind the PCA default.
- Approximating new UMAP points by PCA between re-fits means the live view is
  slightly inconsistent with the batch view in UMAP mode. Reported via `method`.

### Risks
- A 10⁵-point payload is large even with index-based edges. `limit` is mandatory-by-
  default (2000) and the endpoint paginates.

## Alternatives Considered

**t-SNE.** Slower than UMAP, no out-of-sample extension at all, and its cluster sizes
and inter-cluster distances are not meaningful — actively misleading for a view whose
purpose is to make distance meaningful.

**Client-side projection in WASM.** Attractive for interactivity, but it ships all 1536
dimensions of every point to the browser. At 10⁴ points that is ~60 MB.

**Server-rendered images.** Trivial to build, kills interactivity, and interactivity is
the entire point of a manifold explorer.

## References
- ADR-009 (supersedes its technology choices), ADR-010, ADR-011, ADR-013
