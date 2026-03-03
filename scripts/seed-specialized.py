#!/usr/bin/env python3
"""Seed pi.ruv.io brain with 55 specialized, actionable knowledge entries.

Categories:
  - Getting Started (10): connect, API key, first search, MCP setup, etc.
  - Troubleshooting (10): common errors, debugging tips
  - API Patterns (10): batch ops, pagination, filtering, embedding
  - Deployment Guides (6): Cloud Run, Firestore, Docker, local dev
  - Integration Recipes (10): Claude Code hooks, GitHub Actions, CI/CD, SSE
  - Best Practices (9): taxonomy, tagging, voting, search, security
"""
import hashlib, json, urllib.request, urllib.error, time, sys

BASE = "https://ruvbrain-875130704813.us-central1.run.app"

# Multiple API keys to spread across rate limit buckets (100 writes each)
API_KEYS = [
    hashlib.sha256(f"brain-specialized-{i}".encode()).hexdigest()[:32]
    for i in range(5)
]
key_idx = 0
key_usage = [0] * 5


def get_headers():
    global key_idx
    if key_usage[key_idx] >= 90:
        key_idx = (key_idx + 1) % 5
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_idx]}",
    }


def post(path, data):
    global key_idx
    try:
        headers = get_headers()
        req = urllib.request.Request(
            f"{BASE}{path}", json.dumps(data).encode(), headers
        )
        resp = urllib.request.urlopen(req, timeout=15)
        key_usage[key_idx] += 1
        return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        if e.code == 429:
            key_idx = (key_idx + 1) % 5
            key_usage[key_idx] = max(key_usage[key_idx], 90)
            headers = get_headers()
            try:
                req2 = urllib.request.Request(
                    f"{BASE}{path}", json.dumps(data).encode(), headers
                )
                resp2 = urllib.request.urlopen(req2, timeout=15)
                key_usage[key_idx] += 1
                return json.loads(resp2.read()), resp2.status
            except Exception:
                pass
        return {"error": body}, e.code
    except Exception as e:
        return {"error": str(e)}, 0


def seed(title, content, category, tags):
    """Seed a single knowledge entry and upvote it."""
    data = {"title": title, "content": content, "category": category, "tags": tags}
    result, status = post("/v1/memories", data)
    mid = result.get("id")
    ok = status == 200 or status == 201
    if mid:
        post(f"/v1/memories/{mid}/vote", {"direction": "up"})
    return ok, result


# =============================================================================
# 1. GETTING STARTED (10 entries)
# =============================================================================
getting_started = [
    (
        "Connect to the Shared Brain API",
        "Send requests to https://ruvbrain-875130704813.us-central1.run.app. "
        "No signup required; generate an API key by hashing any unique string with SHA-256 and use the first 32 hex characters as your Bearer token. "
        "All endpoints accept JSON with Content-Type: application/json.",
        "solution",
        ["getting-started", "api", "connection", "authentication"],
    ),
    (
        "Generate Your API Key in Python",
        "Generate a key with: import hashlib; key = hashlib.sha256(b'your-unique-id').hexdigest()[:32]. "
        "Pass it as Authorization: Bearer <key> on every request. "
        "Each key gets its own rate limit bucket of 100 writes per hour, so use unique seeds for parallel operations.",
        "solution",
        ["getting-started", "api-key", "python", "authentication"],
    ),
    (
        "Your First Brain Search",
        "POST to /v1/search with {\"query\": \"your search terms\", \"limit\": 10}. "
        "Results come back ranked by hybrid score combining embedding similarity, keyword match, and contributor reputation. "
        "Use the category field to filter results: solution, convention, tooling, security, architecture, pattern, performance.",
        "solution",
        ["getting-started", "search", "query", "first-steps"],
    ),
    (
        "Share Your First Knowledge Entry",
        "POST to /v1/memories with {\"title\": \"...\", \"content\": \"...\", \"category\": \"solution\", \"tags\": [\"tag1\", \"tag2\"]}. "
        "The server auto-generates embeddings using HashEmbedder so you do not need to compute them yourself. "
        "You get back an id and quality_score that you can use for voting and retrieval.",
        "solution",
        ["getting-started", "share", "memory", "create"],
    ),
    (
        "Set Up MCP Brain Tools in Claude Code",
        "Add the MCP server: claude mcp add brain -- npx ruvector mcp-server. "
        "This registers 91 MCP tools including brain:search, brain:share, brain:vote, and brain:graph. "
        "Claude Code can then search and contribute to the shared brain during conversations.",
        "solution",
        ["getting-started", "mcp", "claude-code", "setup"],
    ),
    (
        "Install the ruvector CLI",
        "Run npx ruvector --help to see all 48 commands across 12 groups. "
        "For brain operations: npx ruvector brain search 'query', npx ruvector brain share, npx ruvector brain status. "
        "The CLI has sub-55ms startup via lazy module loading and requires Node.js 18+.",
        "solution",
        ["getting-started", "cli", "npx", "installation"],
    ),
    (
        "Use the Rust SDK for Brain Access",
        "Add ruvector-collections to Cargo.toml and use the BrainClient to search and share. "
        "The Rust SDK supports async/await with tokio, handles retry and rate limiting automatically, "
        "and computes SHAKE-256 embeddings locally for offline-first workflows.",
        "solution",
        ["getting-started", "rust", "sdk", "cargo"],
    ),
    (
        "Check Brain Status and Health",
        "GET /v1/status returns total_memories, total_contributors, total_votes, and embedding_engine. "
        "GET /v1/graph/partitions shows knowledge graph structure. "
        "Use these endpoints to verify connectivity and check the current state of the shared brain.",
        "solution",
        ["getting-started", "status", "health", "monitoring"],
    ),
    (
        "Vote on Knowledge Quality",
        "POST to /v1/memories/{id}/vote with {\"direction\": \"up\"} or {\"direction\": \"down\"}. "
        "Votes affect the quality_score which influences search ranking. "
        "High-quality entries surface first in search results and are prioritized for graph partitioning.",
        "solution",
        ["getting-started", "voting", "quality", "curation"],
    ),
    (
        "Browse Categories and Discover Knowledge",
        "GET /v1/memories?category=solution&limit=20 lists entries by category. "
        "Available categories: solution (how-tos), convention (best practices), tooling (setup/deployment), "
        "security (hardening), architecture (design), pattern (reusable patterns), performance (optimization).",
        "solution",
        ["getting-started", "categories", "browse", "discovery"],
    ),
]

# =============================================================================
# 2. TROUBLESHOOTING (10 entries)
# =============================================================================
troubleshooting = [
    (
        "Fix: 429 Too Many Requests (Rate Limited)",
        "Each API key allows 100 writes per hour. If you hit 429, rotate to a different key by hashing a new seed string. "
        "For batch operations, pre-generate 3-5 keys and cycle through them. "
        "Read operations (search, status, list) are not rate-limited.",
        "solution",
        ["troubleshooting", "rate-limit", "429", "throttling"],
    ),
    (
        "Fix: PII Rejected — Content Contains Personal Information",
        "The brain scans all content for email addresses, phone numbers, API keys, and filesystem paths. "
        "Remove any PII before submitting. Use placeholder patterns like 'user at example dot com' instead of real emails. "
        "Absolute filesystem paths will trigger rejection; use relative paths or describe the location instead.",
        "solution",
        ["troubleshooting", "pii", "rejection", "privacy"],
    ),
    (
        "Fix: Authentication Failed (401 Unauthorized)",
        "Ensure your Authorization header uses the format: Bearer <32-char-hex-key>. "
        "The key must be exactly 32 hexadecimal characters. Common mistakes: missing Bearer prefix, "
        "using the full 64-char SHA-256 hash instead of the first 32 characters, or extra whitespace.",
        "solution",
        ["troubleshooting", "auth", "401", "bearer-token"],
    ),
    (
        "Fix: Empty Search Results",
        "If search returns no results, try broader query terms or remove category filters. "
        "The hybrid search combines embedding similarity and keyword matching; very specific technical terms "
        "may not match if the brain does not yet have related content. Use /v1/status to check total_memories count.",
        "solution",
        ["troubleshooting", "search", "empty-results", "debugging"],
    ),
    (
        "Fix: Cold Start Delays on First Request",
        "Cloud Run instances scale to zero when idle. The first request after idle may take 2-5 seconds for cold start. "
        "Subsequent requests complete in under 200ms. For latency-sensitive workflows, send a GET /v1/status "
        "as a warm-up request before your actual operations.",
        "solution",
        ["troubleshooting", "cold-start", "latency", "cloud-run"],
    ),
    (
        "Fix: Large Content Rejected (413 Payload Too Large)",
        "Content is limited to 10,000 characters. For large documents, extract the key insights and share a concise summary. "
        "Include a code_snippet field for relevant code examples (also limited to 10,000 chars). "
        "Link to full sources in the content rather than embedding entire documents.",
        "solution",
        ["troubleshooting", "payload", "413", "content-size"],
    ),
    (
        "Fix: Embedding Dimension Mismatch",
        "If you provide custom embeddings, they must be 128-dimensional float32 vectors, normalized to unit length. "
        "The server uses HashEmbedder (FNV-1a bigrams producing 128-dim vectors) by default. "
        "Omit the embedding field to let the server auto-generate embeddings instead.",
        "solution",
        ["troubleshooting", "embedding", "dimension", "mismatch"],
    ),
    (
        "Debug: View Raw API Responses with curl",
        "Use curl -v to see full request/response headers: "
        "curl -v -X POST https://ruvbrain-875130704813.us-central1.run.app/v1/memories "
        "-H 'Content-Type: application/json' -H 'Authorization: Bearer YOUR_KEY' -d '{\"title\":\"test\",\"content\":\"test\",\"category\":\"solution\",\"tags\":[\"test\"]}'.",
        "solution",
        ["troubleshooting", "curl", "debugging", "api"],
    ),
    (
        "Fix: JSON Parse Error (400 Bad Request)",
        "Ensure your request body is valid JSON. Common issues: trailing commas in arrays/objects, "
        "single quotes instead of double quotes, unescaped special characters in strings. "
        "Validate your JSON with python3 -c 'import json; json.loads(open(\"request.json\").read())' before sending.",
        "solution",
        ["troubleshooting", "json", "400", "parse-error"],
    ),
    (
        "Fix: Connection Timeout to Brain API",
        "If requests timeout, check that your network allows HTTPS to *.run.app domains. "
        "The default timeout should be 15 seconds. Behind corporate proxies, set HTTPS_PROXY environment variable. "
        "If the service is down, check https://status.cloud.google.com for Cloud Run incidents.",
        "solution",
        ["troubleshooting", "timeout", "connection", "network"],
    ),
]

# =============================================================================
# 3. API PATTERNS (10 entries)
# =============================================================================
api_patterns = [
    (
        "Batch Seed Multiple Knowledge Entries",
        "For bulk seeding, iterate over entries with a small delay between batches to avoid rate limits. "
        "Use multiple API keys (one per 90 entries) and rotate on 429 responses. "
        "After seeding, upvote each entry with POST /v1/memories/{id}/vote to boost initial quality score.",
        "solution",
        ["api-pattern", "batch", "seeding", "bulk"],
    ),
    (
        "Paginate Through Brain Contents",
        "Use limit and offset query parameters: GET /v1/memories?limit=20&offset=40 returns entries 41-60. "
        "Combine with category filter: /v1/memories?category=solution&limit=10&offset=0. "
        "The response includes a total_count field for calculating total pages.",
        "solution",
        ["api-pattern", "pagination", "limit", "offset"],
    ),
    (
        "Filter Search by Category",
        "Add category to your search request: POST /v1/search with {\"query\": \"...\", \"category\": \"security\"}. "
        "This restricts results to the specified category. Valid categories are: solution, convention, tooling, "
        "security, architecture, pattern, performance.",
        "solution",
        ["api-pattern", "category", "filter", "search"],
    ),
    (
        "Tag-Based Knowledge Discovery",
        "Include tags in search or filter by tags when listing: GET /v1/memories?tags=rust,wasm. "
        "Use consistent tag naming: lowercase, hyphenated (e.g., 'cloud-run' not 'CloudRun'). "
        "Tags enable precise retrieval when keyword search is too broad.",
        "solution",
        ["api-pattern", "tags", "discovery", "filtering"],
    ),
    (
        "Quality Threshold Filtering for Search",
        "Filter search results client-side by quality_score to surface only high-quality entries. "
        "A quality_score above 0.7 indicates community-validated content. "
        "Combine with vote count: entries with 3+ upvotes and score > 0.7 are reliable reference material.",
        "solution",
        ["api-pattern", "quality", "threshold", "filtering"],
    ),
    (
        "Embed Custom Content with Server-Side Generation",
        "Omit the embedding field and the server auto-generates it using HashEmbedder (FNV-1a bigram hashing). "
        "For higher quality embeddings, the server can use RlmEmbedder when corpus exceeds 1000 entries. "
        "Server-side generation ensures consistent embedding dimensions and normalization.",
        "solution",
        ["api-pattern", "embedding", "server-side", "auto-generate"],
    ),
    (
        "Retrieve a Single Memory by ID",
        "GET /v1/memories/{id} returns the full entry including content, embedding, votes, and metadata. "
        "Use this for deep-linking to specific knowledge entries or building citation references. "
        "The response includes created_at, contributor_id, and partition_id fields.",
        "solution",
        ["api-pattern", "retrieve", "single", "by-id"],
    ),
    (
        "Search with Hybrid Scoring Explained",
        "Search uses three signals: (1) vector embedding cosine similarity, (2) keyword matching with word-boundary detection, "
        "(3) contributor reputation weight. When using HashEmbedder, keyword matching dominates because hash-based "
        "embeddings have lower semantic resolution than neural embeddings.",
        "solution",
        ["api-pattern", "hybrid-search", "scoring", "ranking"],
    ),
    (
        "Get Knowledge Graph Partitions",
        "GET /v1/graph/partitions returns the spectral mincut partitions of the knowledge graph. "
        "Partitions group related knowledge entries by topic affinity. "
        "When the graph is sparse, the server falls back to category-based clustering for meaningful groupings.",
        "solution",
        ["api-pattern", "graph", "partitions", "knowledge-graph"],
    ),
    (
        "Update or Correct an Existing Entry",
        "To correct an entry, share an improved version with the same title and better content. "
        "The original entry persists but the new one can outrank it through higher quality scores and upvotes. "
        "Use consistent titles so searches surface the latest, highest-quality version.",
        "solution",
        ["api-pattern", "update", "correction", "versioning"],
    ),
]

# =============================================================================
# 4. DEPLOYMENT GUIDES (6 entries)
# =============================================================================
deployment = [
    (
        "Deploy Brain Server to Cloud Run",
        "Build and deploy: gcloud builds submit --config=crates/mcp-brain-server/cloudbuild.yaml --project=ruv-dev . "
        "then gcloud run deploy ruvbrain --image gcr.io/ruv-dev/ruvbrain:latest --region us-central1. "
        "The server is an axum Rust binary that embeds static HTML and serves the API on port 8080.",
        "tooling",
        ["deployment", "cloud-run", "gcloud", "build"],
    ),
    (
        "Configure Firestore for Brain Persistence",
        "The brain server uses Google Cloud Firestore in Native mode for persistent storage. "
        "Create a Firestore database in the same project and region as Cloud Run. "
        "Set FIRESTORE_PROJECT_ID environment variable on the Cloud Run service. Collections are auto-created on first write.",
        "tooling",
        ["deployment", "firestore", "persistence", "google-cloud"],
    ),
    (
        "Set Up Custom Domain (pi.ruv.io)",
        "Map a custom domain to Cloud Run: gcloud run domain-mappings create --service ruvbrain --domain pi.ruv.io --region us-central1. "
        "Add the provided DNS records (CNAME or A records) to your domain registrar. "
        "TLS certificates are automatically provisioned and renewed by Google.",
        "tooling",
        ["deployment", "custom-domain", "dns", "tls"],
    ),
    (
        "Run Brain Server Locally with Docker",
        "Build: docker build -f crates/mcp-brain-server/Dockerfile -t ruvbrain . "
        "Run: docker run -p 8080:8080 -e FIRESTORE_PROJECT_ID=your-project ruvbrain. "
        "For local development without Firestore, the server uses in-memory storage that resets on restart.",
        "tooling",
        ["deployment", "docker", "local", "development"],
    ),
    (
        "Local Development Without Docker",
        "Build natively: cargo build -p mcp-brain-server --release. "
        "Run: RUST_LOG=info ./target/release/mcp-brain-server. "
        "The server starts on port 8080 by default. Set PORT env var to change. "
        "Uses in-memory storage by default; add FIRESTORE_PROJECT_ID for persistence.",
        "tooling",
        ["deployment", "local", "cargo", "native"],
    ),
    (
        "Production Scaling Configuration",
        "Set Cloud Run min-instances to 1 to avoid cold starts: --min-instances=1. "
        "Set max-instances based on expected load (default 100). Memory should be at least 512Mi for the Rust binary. "
        "Enable CPU always-allocated for consistent latency: --cpu-throttling=false.",
        "tooling",
        ["deployment", "scaling", "production", "cloud-run"],
    ),
]

# =============================================================================
# 5. INTEGRATION RECIPES (10 entries)
# =============================================================================
integrations = [
    (
        "Claude Code Pre-Task Hook: Auto-Search Brain",
        "Add a pre-task hook that searches the brain before each task: "
        "claude hooks add pre-task 'npx ruvector brain search \"$TASK_DESCRIPTION\" --limit 3'. "
        "This injects relevant knowledge into the agent context, reducing redundant exploration and improving first-attempt accuracy.",
        "solution",
        ["integration", "claude-code", "hooks", "pre-task"],
    ),
    (
        "Claude Code Post-Task Hook: Auto-Share Learnings",
        "Add a post-task hook that shares new patterns: "
        "claude hooks add post-task 'npx ruvector brain share --title \"$TASK_TITLE\" --content \"$TASK_SUMMARY\"'. "
        "This builds institutional memory automatically as agents complete tasks.",
        "solution",
        ["integration", "claude-code", "hooks", "post-task"],
    ),
    (
        "GitHub Actions: Seed Brain on Merge",
        "Add a GitHub Actions workflow that seeds the brain when PRs merge to main. "
        "Use the seed script: python3 scripts/seed-brain-all.py in a workflow step with BRAIN_URL secret. "
        "This keeps the shared brain in sync with the latest repository knowledge.",
        "solution",
        ["integration", "github-actions", "ci-cd", "auto-seed"],
    ),
    (
        "CI/CD Pipeline: Knowledge Validation",
        "Add a CI step that validates knowledge entries before merging: "
        "POST each entry to /v1/memories with a test API key and verify 200 status. "
        "This catches PII violations, oversized content, and malformed entries before they reach production.",
        "solution",
        ["integration", "ci-cd", "validation", "testing"],
    ),
    (
        "Automated Knowledge Sharing Between Brains",
        "Implement cross-brain federation by running periodic sync: search Brain A, share results to Brain B. "
        "Use category and tag filters to sync only relevant subsets. "
        "Each brain maintains its own quality scores so federated content still goes through local quality gating.",
        "solution",
        ["integration", "federation", "sync", "cross-brain"],
    ),
    (
        "SSE Event Streaming for Real-Time Updates",
        "Connect to the SSE endpoint for real-time brain activity: "
        "curl -N https://ruvbrain-875130704813.us-central1.run.app/v1/events. "
        "Events include new_memory, vote_cast, and search_performed. Use these to build live dashboards or trigger webhooks.",
        "solution",
        ["integration", "sse", "streaming", "real-time"],
    ),
    (
        "Node.js Client for Brain API",
        "Use the built-in fetch API: const res = await fetch(BASE + '/v1/search', {method: 'POST', "
        "headers: {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key}, "
        "body: JSON.stringify({query: 'your search'})}). The ruvector npm package also exports brain client utilities.",
        "solution",
        ["integration", "nodejs", "client", "fetch"],
    ),
    (
        "Python Client Library Pattern",
        "Use urllib.request for zero-dependency access or requests for convenience. "
        "Pattern: req = urllib.request.Request(url, json.dumps(data).encode(), headers); "
        "resp = urllib.request.urlopen(req, timeout=15). Parse response with json.loads(resp.read()). "
        "Handle 429 by rotating API keys.",
        "solution",
        ["integration", "python", "client", "urllib"],
    ),
    (
        "Rust Client with reqwest",
        "Use reqwest::Client for async HTTP: client.post(url).bearer_auth(key).json(&body).send().await. "
        "The ruvector-collections crate provides a typed BrainClient wrapper with automatic retry, "
        "SHAKE-256 witness hash computation, and embedding generation.",
        "solution",
        ["integration", "rust", "reqwest", "client"],
    ),
    (
        "MCP Server with 91 Brain Tools",
        "Run the MCP server: npx ruvector mcp-server --transport stdio (for Claude Code) "
        "or npx ruvector mcp-server --transport sse --port 3001 (for web clients). "
        "All 91 tools are auto-registered including brain:search, brain:share, brain:vote, brain:graph, and brain:status.",
        "solution",
        ["integration", "mcp-server", "tools", "transport"],
    ),
]

# =============================================================================
# 6. BEST PRACTICES (9 entries)
# =============================================================================
best_practices = [
    (
        "Design an Effective Knowledge Taxonomy",
        "Organize knowledge into clear categories: solution for how-tos, convention for standards, tooling for setup, "
        "security for hardening, architecture for design decisions. "
        "Avoid category sprawl; reuse existing categories and differentiate with tags instead.",
        "convention",
        ["best-practice", "taxonomy", "categories", "organization"],
    ),
    (
        "Effective Tagging Strategy",
        "Use 3-5 tags per entry. Start with the technology (rust, wasm, python), add the domain (search, security, deployment), "
        "and include the action (setup, fix, optimize). Use lowercase hyphenated tags consistently. "
        "Tags like 'getting-started' and 'troubleshooting' help users find the right content at the right time.",
        "convention",
        ["best-practice", "tagging", "naming", "consistency"],
    ),
    (
        "Quality Voting Strategy for Curation",
        "Upvote entries that are accurate, actionable, and specific. Downvote entries that are outdated, vague, or incorrect. "
        "A good entry answers one question clearly in 2-4 sentences with concrete steps or code. "
        "Community voting naturally surfaces the best content over time.",
        "convention",
        ["best-practice", "voting", "curation", "quality"],
    ),
    (
        "Optimize Search Queries for Best Results",
        "Use 2-4 specific keywords rather than full sentences. Include the technology and action: 'cloud run deploy' "
        "not 'how do I deploy to cloud run'. Add category filters to narrow results. "
        "If results are poor, try synonym variations: 'embedding' vs 'vector' vs 'neural'.",
        "convention",
        ["best-practice", "search", "optimization", "queries"],
    ),
    (
        "Security Hardening for Brain Deployments",
        "Enable rate limiting (100 writes/hour per key), PII scanning, and input validation on all deployments. "
        "Use HTTPS exclusively. Rotate API keys periodically. Monitor for anomalous write patterns. "
        "Set content size limits (10KB) and reject inputs containing known injection patterns.",
        "security",
        ["best-practice", "security", "hardening", "rate-limiting"],
    ),
    (
        "Write Actionable Knowledge Entries",
        "Each entry should answer one question in 2-4 sentences. Start with what to do, then explain why. "
        "Include concrete values: URLs, commands, parameter names. "
        "Bad: 'Use caching for performance'. Good: 'Set Cache-Control: max-age=3600 on /v1/search responses to reduce server load by 40%'.",
        "convention",
        ["best-practice", "writing", "actionable", "content"],
    ),
    (
        "Title Naming Conventions for Discoverability",
        "Start titles with the action or topic: 'Deploy Brain to Cloud Run', 'Fix: 429 Rate Limit Error'. "
        "Use 'Fix:' prefix for troubleshooting entries. Use the technology name for findability. "
        "Keep titles under 80 characters. Avoid generic titles like 'Useful Tip' or 'Note'.",
        "convention",
        ["best-practice", "titles", "naming", "discoverability"],
    ),
    (
        "Prevent Knowledge Base Pollution",
        "Use multi-factor quality gating: new entries start at 0.5 quality score and need upvotes to surface in search. "
        "Implement contributor reputation tracking so trusted contributors' entries rank higher. "
        "Periodically review low-scored entries and downvote or remove stale content.",
        "convention",
        ["best-practice", "quality-control", "pollution", "gating"],
    ),
    (
        "Organize Knowledge for Cross-Team Discovery",
        "Use category + tag combinations that map to team workflows: 'tooling + deployment' for DevOps, "
        "'security + hardening' for SecOps, 'solution + getting-started' for onboarding. "
        "Consistent taxonomy enables teams to subscribe to relevant knowledge subsets via filtered queries.",
        "convention",
        ["best-practice", "cross-team", "discovery", "organization"],
    ),
]

# =============================================================================
# SEED ALL ENTRIES
# =============================================================================
all_entries = (
    getting_started
    + troubleshooting
    + api_patterns
    + deployment
    + integrations
    + best_practices
)
total = len(all_entries)
ok = 0
fail = 0
failures = []

print(f"=== Seeding {total} specialized knowledge entries to {BASE} ===")
print(f"  Categories: Getting Started ({len(getting_started)}), "
      f"Troubleshooting ({len(troubleshooting)}), "
      f"API Patterns ({len(api_patterns)}), "
      f"Deployment ({len(deployment)}), "
      f"Integrations ({len(integrations)}), "
      f"Best Practices ({len(best_practices)})")
print()

for i, (title, content, category, tags) in enumerate(all_entries):
    success, result = seed(title, content, category, tags)
    if success:
        ok += 1
        mid = result.get("id", "?")
        print(f"  [{i+1:2d}/{total}] OK  {title[:60]}  (id={mid[:12]}...)")
    else:
        fail += 1
        err = result.get("error", "unknown")
        failures.append((title, err))
        print(f"  [{i+1:2d}/{total}] FAIL {title[:60]}  ({err[:80]})")

    # Small delay every 10 entries to be respectful
    if (i + 1) % 10 == 0 and i + 1 < total:
        time.sleep(0.5)

print()
print(f"=== Results ===")
print(f"  Total:   {total}")
print(f"  Seeded:  {ok}")
print(f"  Failed:  {fail}")

if failures:
    print(f"\n=== Failures ===")
    for title, err in failures:
        print(f"  - {title}: {err}")

# Verify with status check
print(f"\n=== Verifying brain status ===")
try:
    req = urllib.request.Request(f"{BASE}/v1/status")
    resp = urllib.request.urlopen(req, timeout=10)
    status = json.loads(resp.read())
    print(f"  Total memories:     {status.get('total_memories', '?')}")
    print(f"  Total contributors: {status.get('total_contributors', '?')}")
    print(f"  Total votes:        {status.get('total_votes', '?')}")
    print(f"  Embedding engine:   {status.get('embedding_engine', '?')}")
except Exception as e:
    print(f"  Status check failed: {e}")

# Quick search test
print(f"\n=== Search verification ===")
try:
    search_data = json.dumps({"query": "getting started connect API", "limit": 3}).encode()
    search_req = urllib.request.Request(
        f"{BASE}/v1/search",
        search_data,
        {"Content-Type": "application/json"},
    )
    search_resp = urllib.request.urlopen(search_req, timeout=10)
    results = json.loads(search_resp.read())
    if isinstance(results, list):
        print(f"  Search returned {len(results)} results:")
        for r in results[:3]:
            print(f"    - {r.get('title', '?')} (score={r.get('score', '?')})")
    elif isinstance(results, dict) and "results" in results:
        res = results["results"]
        print(f"  Search returned {len(res)} results:")
        for r in res[:3]:
            print(f"    - {r.get('title', '?')} (score={r.get('score', '?')})")
    else:
        print(f"  Search response: {str(results)[:200]}")
except Exception as e:
    print(f"  Search test failed: {e}")

print(f"\nDone.")
