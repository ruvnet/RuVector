# rvf-server

TCP/HTTP streaming server exposing the RuVector Format runtime via a REST API.

## Overview

`rvf-server` wraps `rvf-runtime` in an Axum-based HTTP server for networked vector operations:

- **REST API** -- CRUD endpoints for vectors and collections
- **Streaming** -- chunked transfer for large batch imports/exports
- **CLI** -- configurable via `clap` command-line arguments

## Usage

```bash
cargo run -p rvf-server -- --port 8080
```

The server binds `127.0.0.1` by default. To listen on another address, set a
bearer token in `RVF_SERVER_TOKEN`; startup is refused without one. HTTP
requests then need `Authorization: Bearer <token>` (`GET /v1/health` stays
open). The binary TCP protocol has no authentication and stays on loopback.

```bash
RVF_SERVER_TOKEN=... cargo run -p rvf-server -- --host 0.0.0.0 --port 8080
```

## License

MIT OR Apache-2.0
