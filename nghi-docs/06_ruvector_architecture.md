# 06. Ruvector Architecture

Now that you know Rust, let's understand how `ruvector` is put together.

## The Workspace
`ruvector` is a Cargo Workspace. It consists of multiple crates that work together.

### Key Crates

*   **`ruvector-core`**: The brain. Contains the vector index implementation, data structures, and core logic.
*   **`ruvector-server`**: The interface. Wraps the core in an HTTP/gRPC server (likely using `axum` or `tonic`).
*   **`ruvector-cli`**: The tool. A command-line interface to interact with the database.
*   **`ruvector-node` / `ruvector-wasm`**: The bridges. Bindings for other environments.

## Data Flow: Inserting a Vector

1.  **Request**: A request comes in (via CLI, HTTP, or Node.js).
2.  **Parsing**: The request is parsed into a Rust struct (e.g., `InsertRequest`).
3.  **Core**: The `ruvector-core` crate takes over.
    *   It might validate the vector dimensions.
    *   It adds the vector to the storage (WAL - Write Ahead Log).
    *   It updates the HNSW (Hierarchical Navigable Small World) index for fast searching.
4.  **Response**: A success result is returned up the chain.

## Data Flow: Searching

1.  **Query**: A query vector is received.
2.  **Index Search**: The HNSW index is traversed to find the nearest neighbors.
3.  **Distance Calculation**: SIMD instructions (via `simsimd` crate) might be used to calculate distances (Euclidean, Cosine) extremely fast.
4.  **Filtering**: Results might be filtered based on metadata.
5.  **Result**: The top K matches are returned.

## Conclusion
You now have a high-level understanding of `ruvector` and the Rust concepts that power it.
The best way to learn more is to start hacking! Pick a small issue or try to add a tiny feature.

**Happy Coding!**
