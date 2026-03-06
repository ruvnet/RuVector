# ADR-072: RVF Example Management and Downloads in npx ruvector

**Status**: Proposed
**Date**: 2026-02-28
**Authors**: RuVector Team
**Deciders**: ruv
**Supersedes**: N/A
**Related**: ADR-070 (npx ruvector Unified Integration), ADR-044 (RVF Format Specification), ADR-065 (npm Publishing Strategy)

## 1. Context

The RuVector ecosystem currently ships 46 `.rvf` example files in `examples/rvf/output/` totaling ~11 MB. These demonstrate every RVF capability: basic vector stores, HNSW indexes, COW lineage chains, eBPF accelerators, TEE attestation, ZK witnesses, agent memory, MCP-in-RVF, self-booting kernels, and more.

Today these examples are accessed three ways, each with problems:

| Access Method | Problem |
|---|---|
| `npx ruvector rvf examples` (CLI) | Lists metadata only. `rvf download` fetches from GitHub raw — slow, no caching, no versioning, raw.githubusercontent.com has rate limits |
| `rvf_examples` MCP tool | Returns hardcoded list of 12 examples (out of 46). No download capability |
| Clone the repo | 800 MB+ clone for 11 MB of examples |

Additionally:
- The example catalog is hardcoded in two places (`cli.js` and `mcp-server.js`) and they're out of sync (CLI has 45, MCP has 12)
- No version pinning — examples may break with format changes
- No integrity verification — downloaded files aren't checksummed
- No offline cache — re-downloads every time
- New examples require a new npm publish to update the catalog

## 2. Decision

Host `.rvf` example files on Google Cloud Storage with a manifest-driven catalog. The CLI and MCP server read a versioned manifest from GCS to discover examples, download files with SHA-256 verification, and cache locally at `~/.ruvector/examples/`. A GitHub Actions workflow syncs examples to GCS on every push to `main`.

## 3. Architecture

### 3.1 GCS Bucket Layout

```
gs://ruvector-examples/
  ├── manifest.json              ← current catalog (always latest)
  ├── v0.2.1/
  │   ├── manifest.json          ← pinned catalog for this version
  │   ├── basic_store.rvf
  │   ├── semantic_search.rvf
  │   ├── rag_pipeline.rvf
  │   ├── ...                    ← all 46+ examples
  │   └── checksums.sha256       ← SHA-256 for every file
  ├── v0.2.0/
  │   ├── manifest.json
  │   └── ...
  └── latest -> v0.2.1/          ← symlink-like redirect
```

**Public read access** via `allUsers:objectViewer` IAM. No authentication required for downloads.

**Cloud CDN** in front of the bucket for global edge caching. Typical latency: 20-50ms worldwide vs 200-500ms from raw.githubusercontent.com.

### 3.2 Manifest Format

```json
{
  "version": "0.2.1",
  "updated": "2026-02-28T12:00:00Z",
  "base_url": "https://storage.googleapis.com/ruvector-examples/v0.2.1",
  "total_size": "11.2 MB",
  "examples": [
    {
      "name": "basic_store",
      "file": "basic_store.rvf",
      "size": 155648,
      "size_human": "152 KB",
      "sha256": "a1b2c3d4...",
      "description": "1,000 vectors, dim 128, cosine metric",
      "category": "core",
      "tags": ["vectors", "cosine", "basic"],
      "rvf_version": "1.0",
      "segments": ["VEC", "META", "MANIFEST"],
      "created": "2026-02-15"
    }
  ],
  "categories": {
    "core": "Basic vector storage and search",
    "ai": "AI agent, embedding, and RAG examples",
    "security": "Attestation, ZK proofs, access control",
    "compute": "eBPF, WASM, self-booting, kernels",
    "lineage": "COW chains, derivation, reasoning",
    "industry": "Finance, medical, legal domain examples",
    "network": "Sync, handoff, distributed examples",
    "integration": "MCP, PostgreSQL, serverless bridges"
  }
}
```

### 3.3 Local Cache

```
~/.ruvector/
  └── examples/
      ├── manifest.json           ← cached manifest (TTL: 1 hour)
      ├── basic_store.rvf         ← downloaded example
      ├── semantic_search.rvf
      └── .cache-meta.json        ← cache timestamps + checksums
```

Cache behavior:
- **Manifest TTL**: 1 hour. After that, fetch fresh manifest on next `examples` or `download` command
- **File cache**: Permanent until `rvf cache clear` or version change
- **Integrity**: SHA-256 verified on every download. Cached files re-verified on access if `--verify` flag used
- **Disk budget**: Default 100 MB. Oldest files evicted when budget exceeded. Configurable via `~/.ruvector/config.json`

### 3.4 CLI Commands

```
npx ruvector rvf examples                      # List all examples (from cached manifest)
npx ruvector rvf examples --category security   # Filter by category
npx ruvector rvf examples --refresh             # Force manifest refresh
npx ruvector rvf download basic_store           # Download one example (cached)
npx ruvector rvf download --all                 # Download all examples (~11 MB)
npx ruvector rvf download --category ai         # Download all AI examples
npx ruvector rvf download --verify              # Re-verify cached files
npx ruvector rvf cache status                   # Show cache size, file count
npx ruvector rvf cache clear                    # Clear local cache
```

### 3.5 MCP Tool Updates

The `rvf_examples` MCP tool reads from the same manifest:

```json
{
  "name": "rvf_examples",
  "description": "List and download RVF example files from the ruvector catalog",
  "inputSchema": {
    "type": "object",
    "properties": {
      "filter": { "type": "string", "description": "Filter by name or description" },
      "category": { "type": "string", "description": "Filter by category" },
      "download": { "type": "string", "description": "Download a specific example by name" }
    }
  }
}
```

When `download` is specified, the tool downloads the file to the current working directory and returns the local path. This allows Claude Code to directly work with `.rvf` files.

### 3.6 Sync Pipeline (GitHub Actions)

```yaml
# .github/workflows/sync-rvf-examples.yml
name: Sync RVF Examples to GCS
on:
  push:
    branches: [main]
    paths:
      - 'examples/rvf/output/**'
      - 'examples/rvf/generate_examples.rs'

jobs:
  sync:
    runs-on: ubuntu-latest
    permissions:
      id-token: write  # Workload Identity Federation
    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
          service_account: ${{ secrets.GCS_SA }}

      - name: Generate manifest
        run: |
          python3 scripts/generate-rvf-manifest.py \
            --input examples/rvf/output/ \
            --version $(jq -r .version npm/packages/ruvector/package.json) \
            --output manifest.json

      - name: Sync to GCS
        uses: google-github-actions/upload-cloud-storage@v2
        with:
          path: examples/rvf/output/
          destination: ruvector-examples/v${{ env.VERSION }}/
          gzip: false  # .rvf files are already compact

      - name: Upload manifest
        uses: google-github-actions/upload-cloud-storage@v2
        with:
          path: manifest.json
          destination: ruvector-examples/v${{ env.VERSION }}/manifest.json

      - name: Update latest manifest
        run: |
          gsutil cp gs://ruvector-examples/v${{ env.VERSION }}/manifest.json \
                    gs://ruvector-examples/manifest.json
```

### 3.7 Manifest Generator Script

```python
# scripts/generate-rvf-manifest.py
# Scans examples/rvf/output/, computes SHA-256, extracts RVF segment info,
# categorizes by naming convention, produces manifest.json
```

Categories are derived from the example name or explicit `category` field in a sidecar `.meta.json` file if present:

| Pattern | Category |
|---|---|
| `basic_store`, `semantic_search`, `filtered_search`, `quantization` | core |
| `agent_*`, `rag_*`, `embedding_*`, `ruvllm_*`, `ruvbot` | ai |
| `tee_*`, `zero_knowledge`, `access_control`, `sealed_engine` | security |
| `self_booting`, `ebpf_*`, `browser_wasm`, `linux_microkernel` | compute |
| `lineage_*`, `reasoning_*` | lineage |
| `financial_*`, `medical_*`, `legal_*` | industry |
| `network_*`, `agent_handoff_*` | network |
| `mcp_*`, `postgres_*`, `serverless`, `claude_code_*` | integration |

## 4. Fallback Strategy

If GCS is unreachable:
1. **Cached manifest**: Use `~/.ruvector/examples/manifest.json` if within TTL
2. **Stale manifest**: Use expired cached manifest with warning
3. **Hardcoded fallback**: Built-in minimal catalog (top 12 most popular examples) with GitHub raw URLs
4. **Offline mode**: `--offline` flag uses only cached files, no network

```
Download priority:
  1. Local cache (~/.ruvector/examples/)     ← 0ms, SHA-256 verified
  2. GCS via Cloud CDN                       ← 20-50ms global
  3. GCS direct                              ← 50-200ms
  4. GitHub raw (fallback)                   ← 200-500ms, rate limited
```

## 5. Security

### 5.1 Integrity Verification

Every downloaded `.rvf` file is verified against the SHA-256 in the manifest before being cached or returned to the user. This prevents:
- **CDN poisoning**: Tampered files at the edge are detected
- **MITM attacks**: Even over HTTPS, defense-in-depth with content hashes
- **Cache corruption**: Local disk corruption caught on re-verify

### 5.2 Manifest Signing (Future)

Phase 2 will add Ed25519 signing of manifests:
```json
{
  "version": "0.2.1",
  "examples": [...],
  "signature": "base64(Ed25519(manifest_without_signature))",
  "signer": "ruvector-release-key-001"
}
```
The CLI will verify manifests against a pinned public key shipped in the npm package.

### 5.3 Path Safety

Download destination paths are validated to prevent directory traversal:
- `path.basename()` strips parent directory references
- Regex allows only `[a-zA-Z0-9_\-.]` characters in filenames
- Final path must resolve within the output directory
- These checks already exist in the current `rvf download` implementation

### 5.4 GCS Access Control

- Bucket: `allUsers:objectViewer` (public read)
- Write: Only the CI/CD service account via Workload Identity Federation
- No API keys or credentials shipped in the npm package
- Cloud Armor WAF rules for DDoS protection on CDN ingress

## 6. Implementation

### 6.1 File Changes

| File | Change |
|---|---|
| `npm/packages/ruvector/bin/cli.js` | Update `rvf examples` to fetch manifest from GCS. Update `rvf download` to use GCS URLs + SHA-256 verify + cache. Add `rvf cache` subcommand |
| `npm/packages/ruvector/bin/mcp-server.js` | Update `rvf_examples` handler to read manifest. Add download capability |
| `npm/packages/ruvector/src/rvf-catalog.ts` | Shared manifest fetcher, cache manager, integrity verifier |
| `scripts/generate-rvf-manifest.py` | Manifest generator from local examples |
| `.github/workflows/sync-rvf-examples.yml` | CI/CD sync pipeline |

### 6.2 Single Source of Truth

The `RVF_EXAMPLES` array currently hardcoded in `cli.js` (45 entries) and `mcp-server.js` (12 entries) is replaced by a shared manifest. Both read from:

```javascript
async function getRvfCatalog(opts = {}) {
  const cacheDir = path.join(os.homedir(), '.ruvector', 'examples');
  const manifestPath = path.join(cacheDir, 'manifest.json');

  // Check cache
  if (!opts.refresh && fs.existsSync(manifestPath)) {
    const stat = fs.statSync(manifestPath);
    const age = Date.now() - stat.mtimeMs;
    if (age < 3600000) { // 1 hour TTL
      return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    }
  }

  // Fetch from GCS
  const GCS_URL = 'https://storage.googleapis.com/ruvector-examples/manifest.json';
  try {
    const resp = await fetch(GCS_URL);
    const manifest = await resp.json();
    fs.mkdirSync(cacheDir, { recursive: true });
    fs.writeFileSync(manifestPath, JSON.stringify(manifest, null, 2));
    return manifest;
  } catch {
    // Fallback to cached (even if stale)
    if (fs.existsSync(manifestPath)) {
      return JSON.parse(fs.readFileSync(manifestPath, 'utf8'));
    }
    // Final fallback: hardcoded minimal catalog
    return BUILTIN_CATALOG;
  }
}
```

## 7. GCS Setup

### 7.1 Bucket Creation

```bash
# Create bucket with Standard storage (frequently accessed)
gcloud storage buckets create gs://ruvector-examples \
  --location=us-central1 \
  --uniform-bucket-level-access \
  --public-access-prevention=inherited

# Enable public read
gcloud storage buckets add-iam-policy-binding gs://ruvector-examples \
  --member=allUsers \
  --role=roles/storage.objectViewer

# Enable Cloud CDN (via load balancer backend bucket)
gcloud compute backend-buckets create ruvector-examples-backend \
  --gcs-bucket-name=ruvector-examples \
  --enable-cdn \
  --cache-mode=CACHE_ALL_STATIC \
  --default-ttl=3600
```

### 7.2 CI/CD Service Account

```bash
gcloud iam service-accounts create rvf-examples-sync \
  --display-name="RVF Examples Sync"

gcloud storage buckets add-iam-policy-binding gs://ruvector-examples \
  --member=serviceAccount:rvf-examples-sync@PROJECT.iam.gserviceaccount.com \
  --role=roles/storage.objectAdmin

# Workload Identity Federation for GitHub Actions (no key file)
gcloud iam workload-identity-pools create github-pool \
  --location=global
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --issuer-uri=https://token.actions.githubusercontent.com \
  --attribute-mapping="google.subject=assertion.sub"
```

### 7.3 Cost Estimate

| Resource | Monthly Cost |
|---|---|
| GCS Standard (50 MB, 46 files x ~5 versions) | ~$0.01 |
| Cloud CDN egress (1 GB/month estimate) | ~$0.08 |
| Cloud CDN requests (10K/month estimate) | ~$0.01 |
| **Total** | **~$0.10/month** |

## 8. Example Categories and Counts

| Category | Count | Total Size | Description |
|---|---|---|---|
| core | 8 | ~5.1 MB | Basic stores, HNSW, quantization, filtering |
| ai | 8 | ~1.2 MB | Agent memory, RAG, embeddings, chatbot |
| security | 4 | ~439 KB | TEE, ZK, access control, sealed engine |
| compute | 4 | ~213 KB | eBPF, WASM, self-boot, microkernel |
| lineage | 5 | ~102 KB | COW chains, reasoning chains |
| industry | 3 | ~1.4 MB | Finance, medical, legal |
| network | 4 | ~146 KB | Sync, handoff, telemetry |
| integration | 6 | ~862 KB | MCP, PostgreSQL, serverless, Claude Code |
| **Total** | **46** | **~11.2 MB** | |

## 9. Migration Path

### Phase 1: GCS Setup + Sync (1 day)
- Create GCS bucket with public read
- Write `scripts/generate-rvf-manifest.py`
- Upload current 46 examples + manifest
- Create GitHub Actions workflow

### Phase 2: CLI Update (1 day)
- Replace hardcoded `RVF_EXAMPLES` with manifest fetcher
- Add SHA-256 integrity verification to downloads
- Add local cache at `~/.ruvector/examples/`
- Add `rvf cache status` and `rvf cache clear` subcommands
- Add `--category`, `--refresh`, `--offline`, `--verify` flags

### Phase 3: MCP Update (half day)
- Update `rvf_examples` handler to read from manifest
- Add `download` parameter to MCP tool
- Unify catalog between CLI and MCP (single code path)

### Phase 4: CDN + Monitoring (half day)
- Enable Cloud CDN via backend bucket
- Set up monitoring alerts for download errors
- Add analytics (download counts per example)

## 10. Testing

| Test | Description |
|---|---|
| Manifest fetch from GCS | Returns valid JSON with all 46 examples |
| Manifest cache TTL | Second fetch within 1 hour uses cache |
| Manifest refresh | `--refresh` bypasses cache |
| Download + SHA-256 verify | Downloaded file matches checksum |
| Cache hit | Second download of same file is instant |
| Offline mode | `--offline` uses only cached files |
| GCS unreachable | Falls back to cached manifest, then hardcoded catalog |
| Tampered file | SHA-256 mismatch detected and reported |
| Path traversal | `../../../etc/passwd.rvf` rejected |
| Category filter | `--category security` returns only 4 examples |
| Disk budget | Cache evicts oldest files when >100 MB |
| CI/CD sync | Push to `examples/rvf/output/` triggers GCS upload |
| Manifest generator | Produces valid manifest with correct checksums |
| MCP download | `rvf_examples` with `download` param returns local path |

## 11. Consequences

### Positive
- **Single source of truth**: Manifest replaces 2 hardcoded catalogs
- **Fast global downloads**: Cloud CDN edge caching (20-50ms vs 200-500ms)
- **Integrity**: SHA-256 verification on every download
- **Offline support**: Local cache + `--offline` flag
- **Auto-sync**: New examples available without npm republish
- **Cheap**: ~$0.10/month for storage + CDN
- **Version pinning**: Each ruvector version maps to a pinned manifest

### Negative
- **GCS dependency**: New infrastructure to maintain (mitigated by GitHub raw fallback)
- **Cache management**: Users accumulate cached files (mitigated by disk budget + clear command)
- **Manifest staleness**: 1-hour TTL means new examples take up to 1 hour to appear

### Neutral
- Existing `rvf download` command retains same UX, just faster and verified
- `rvf_examples` MCP tool gains download capability
- GitHub raw URLs continue to work as fallback

## 12. Related ADRs

| ADR | Relationship |
|---|---|
| ADR-044 | RVF Format Specification — defines the `.rvf` binary format these examples demonstrate |
| ADR-059 | Shared Brain Google Cloud — same GCP project, similar GCS patterns |
| ADR-065 | npm Publishing Strategy — example catalog decoupled from npm publish cycle |
| ADR-070 | npx ruvector Unified Integration — `rvf` command group that hosts these commands |
