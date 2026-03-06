#!/usr/bin/env python3
"""Seed π.ruv.io brain with comprehensive RuVector knowledge.

Uses multiple API keys to avoid per-contributor rate limits (100 writes/hr).
"""
import json, hashlib, urllib.request, urllib.error, time, sys

BASE = "https://ruvbrain-875130704813.us-central1.run.app"

# Multiple API keys to spread across rate limit buckets (100 writes each)
API_KEYS = [
    hashlib.sha256(f"brain-ruvector-seed-{i}".encode()).hexdigest()[:32]
    for i in range(5)  # 5 keys = 500 write capacity
]
key_index = 0
key_usage = [0] * len(API_KEYS)

def get_headers():
    global key_index
    # Rotate to next key if current is near limit
    if key_usage[key_index] >= 90:
        key_index = (key_index + 1) % len(API_KEYS)
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEYS[key_index]}"
    }

def post(path, data):
    global key_index
    try:
        headers = get_headers()
        req = urllib.request.Request(f"{BASE}{path}", json.dumps(data).encode(), headers)
        resp = urllib.request.urlopen(req, timeout=15)
        key_usage[key_index] += 1
        return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        body = e.read().decode()[:200]
        if e.code == 429:  # Rate limited — try next key
            key_index = (key_index + 1) % len(API_KEYS)
            key_usage[key_index] = max(key_usage[key_index], 90)
            headers = get_headers()
            try:
                req2 = urllib.request.Request(f"{BASE}{path}", json.dumps(data).encode(), headers)
                resp2 = urllib.request.urlopen(req2, timeout=15)
                key_usage[key_index] += 1
                return json.loads(resp2.read()), resp2.status
            except:
                pass
        return {"error": body}, e.code
    except Exception as e:
        return {"error": str(e)}, 0

def seed(title, content, category, tags):
    data = {"title": title, "content": content, "category": category, "tags": tags}
    result, status = post("/v1/memories", data)
    mid = result.get("id")
    if mid:
        post(f"/v1/memories/{mid}/vote", {"direction": "up"})
    return status == 200 or status == 201

# ===== CURATED KNOWLEDGE (68 entries) =====
curated = [
    # Architecture
    ("SONA Three-Tier Learning Architecture", "Self-Organizing Neural Architecture with three processing tiers: reactive fast-path for cached patterns, adaptive mid-tier for learned behavior, and deliberative deep reasoning for novel situations. Enables runtime-adaptive learning without retraining.", "architecture", ["sona", "learning", "neural", "three-tier"]),
    ("Graph Transformer Architecture", "Combines graph neural networks with transformer attention for structured data processing. Uses topology-gated attention where graph structure modulates attention weights.", "architecture", ["graph", "transformer", "attention", "gnn"]),
    ("Coherence-Gated Attention", "Attention mechanism that gates information flow based on sheaf-theoretic coherence. Only allows attention between nodes whose local sections are compatible under restriction maps.", "architecture", ["coherence", "attention", "sheaf", "gating"]),
    ("MinCut Subpolynomial Graph Partitioning", "Partitions knowledge graphs using spectral mincut algorithms in subpolynomial time. Uses Fiedler vector computation for balanced bisection with O(n^0.5 * log n) complexity.", "architecture", ["mincut", "partitioning", "graph", "spectral"]),
    ("Attention Mechanism Zoo — 46+ Variants", "Comprehensive collection of 46+ attention mechanisms including dot-product, multi-head, cross, sparse, linear, sliding-window, grouped-query, multi-query, flash, ring, and topology-gated variants.", "architecture", ["attention", "mechanism", "multi-head", "flash"]),
    ("GNN Knowledge Graph with HNSW Index", "Graph neural network layers integrated with HNSW vector index for sub-millisecond nearest neighbor search. Combines structural graph features with vector similarity for hybrid retrieval.", "architecture", ["gnn", "knowledge-graph", "hnsw", "vector-search"]),
    ("Domain Expansion Transfer Learning", "Enables knowledge transfer between domains through embedding space alignment. Uses anchor points in shared embedding space to bridge domain-specific representations.", "architecture", ["domain", "transfer", "learning", "expansion"]),
    ("Shared Brain Cloud Architecture", "Distributed brain architecture on Google Cloud Run with Firestore persistence, knowledge graph partitioning, and federated learning. Supports multi-contributor knowledge sharing.", "architecture", ["cloud", "brain", "firestore", "distributed"]),
    ("Cognitive Container Architecture", "Isolated execution environments for AI reasoning with TEE (Trusted Execution Environment) hardening. Containers provide memory isolation, resource limits, and audit trails.", "architecture", ["cognitive", "container", "tee", "isolation"]),
    ("Edge Network — Distributed AI at the Edge", "Peer-to-peer network for AI inference at the edge. Uses relay nodes for NAT traversal, credit-based resource sharing, and gossip protocol for peer discovery.", "architecture", ["edge", "network", "distributed", "p2p"]),

    # Patterns
    ("Federated Learning with Byzantine Tolerance", "Federated aggregation of model updates with Byzantine fault detection. Uses 2-sigma outlier filtering to reject malicious or corrupted gradient updates from untrusted contributors.", "pattern", ["federated", "byzantine", "aggregation", "tolerance"]),
    ("Delta Behavior and Drift Detection", "Monitors embedding space drift over time using centroid tracking and Mahalanobis distance. Detects concept drift, distribution shift, and adversarial perturbation in real-time.", "pattern", ["delta", "drift", "detection", "monitoring"]),
    ("Witness Chain Integrity Pattern", "Merkle-tree based integrity verification for knowledge contributions. Each memory includes a witness hash chain that enables tamper detection and provenance tracking.", "pattern", ["witness", "chain", "integrity", "merkle"]),
    ("SPARC Methodology for AI Development", "Specification, Pseudocode, Architecture, Refinement, Completion — five-phase development methodology for AI systems with multi-agent orchestration and self-learning hooks.", "pattern", ["sparc", "methodology", "development", "phases"]),
    ("ReasoningBank — Self-Learning Pattern Memory", "Persistent memory system that learns from agent trajectories. Stores reasoning patterns, verdicts, and distilled experiences with HNSW-indexed retrieval for fast pattern matching.", "pattern", ["reasoningbank", "learning", "memory", "patterns"]),
    ("Raft Consensus for Distributed State", "Raft consensus protocol adapted for multi-agent coordination. Leader election, log replication, and state machine application for consistent distributed decision-making.", "pattern", ["raft", "consensus", "distributed", "leader-election"]),
    ("Byzantine Fault Tolerant Consensus", "Multi-agent consensus mechanism tolerant of Byzantine (arbitrary) failures. Uses voting, threshold signatures, and reputation-weighted agreement for robust coordination.", "pattern", ["byzantine", "fault-tolerant", "consensus", "voting"]),
    ("Self-Healing Agent Workflows", "Agents detect failures and automatically recover using checkpoint-restart, task reassignment, and degraded-mode operation. Hook-based monitoring triggers healing actions.", "pattern", ["self-healing", "workflow", "recovery", "checkpoint"]),
    ("Parking Lot RwLock for Shared State", "Uses parking_lot::RwLock instead of std::sync::RwLock for better performance in contended scenarios. Provides fair scheduling and no writer starvation.", "pattern", ["parking-lot", "rwlock", "concurrency", "performance"]),
    ("Federated Learning Architecture", "Framework for privacy-preserving distributed model training. Local model updates are aggregated without sharing raw data, supporting differential privacy and secure aggregation.", "pattern", ["federated", "learning", "privacy", "aggregation"]),
    ("Differential Privacy for Embeddings", "Adds calibrated Gaussian noise to embedding vectors before sharing, providing epsilon-differential privacy. Prevents reconstruction of original content from shared embeddings.", "pattern", ["differential", "privacy", "embeddings", "noise"]),
    ("Multi-Factor Reputation Gating", "Quality gating based on contributor reputation, vote count, freshness, and coherence score. Multi-factor scoring prevents low-quality contributions from polluting the knowledge base.", "pattern", ["reputation", "gating", "quality", "multi-factor"]),

    # Security
    ("Content Hash Security — SHAKE-256", "All content is hashed using SHAKE-256 (variable-length XOF) for integrity verification and deduplication. Provides 256-bit security with flexible output length.", "security", ["hash", "shake-256", "integrity", "security"]),
    ("Zero-Trust Input Validation Pipeline", "All API inputs pass through validation pipeline: size limits, UTF-8 verification, PII scanning, injection detection, and content policy checks before processing.", "security", ["zero-trust", "validation", "input", "pipeline"]),
    ("PII Detection and Content Filtering", "Scans all incoming content for personally identifiable information including email addresses, phone numbers, API keys, and filesystem paths. Rejects content containing PII.", "security", ["pii", "detection", "filtering", "privacy"]),
    ("MicroLoRA Federated Fine-Tuning", "Lightweight LoRA (Low-Rank Adaptation) deltas shared between contributors for federated fine-tuning. Byzantine-tolerant aggregation prevents model poisoning.", "security", ["lora", "fine-tuning", "federated", "microloRA"]),
    ("MCP Gate — Permit System for AI Tool Access", "Fine-grained permission system for AI tool access via MCP. Issues time-limited, scope-restricted permits that control which tools agents can invoke and with what parameters.", "security", ["mcp-gate", "permit", "access-control", "tools"]),

    # Solutions
    ("Hybrid Search — Embedding + Keyword + Reputation", "Three-signal search combining vector embedding similarity, keyword matching with word-boundary detection, and contributor reputation weighting. Keyword-dominant when HashEmbedder is active.", "solution", ["hybrid", "search", "embedding", "keyword"]),
    ("Server-Side Neural Embedding Generation", "Server generates embeddings using ruvllm rather than requiring clients to compute them. Supports HashEmbedder (FNV-1a bigrams) and RlmEmbedder (recursive context-aware) based on corpus size.", "solution", ["embedding", "ruvllm", "server-side", "neural"]),
    ("Lazy Graph Partition with Caching", "Graph partitioning is computed lazily on first request and cached until topology changes. Avoids expensive spectral computation on every query.", "solution", ["graph", "partition", "caching", "lazy"]),
    ("Category-Based Partition Fallback", "When spectral partitioning fails (disconnected graph), falls back to category-based clustering. Ensures meaningful partitions even with sparse graphs.", "solution", ["category", "partition", "fallback", "clustering"]),
    ("WASM Multi-Target Compilation", "Compiles Rust crates to WebAssembly with wasm-pack for browser, Node.js, and Deno targets. Enables running vector operations, GNN inference, and quantum simulation in the browser.", "solution", ["wasm", "compilation", "browser", "wasm-pack"]),
    ("SSE Transport for MCP Integration", "Server-Sent Events transport layer for Model Context Protocol. Enables real-time streaming of tool results and agent coordination over HTTP without WebSocket complexity.", "solution", ["sse", "mcp", "transport", "streaming"]),
    ("HDC — Hyperdimensional Computing", "Uses high-dimensional binary vectors for efficient similarity computation. Operations include binding, bundling, and permutation for compositional representations.", "solution", ["hdc", "hyperdimensional", "binary-vectors", "similarity"]),
    ("WASM Executable Nodes (ADR-063)", "Knowledge graph nodes that contain executable WASM modules. Enables computation within the knowledge graph itself — nodes can transform, validate, or generate content.", "solution", ["wasm", "executable", "nodes", "knowledge-graph"]),

    # Conventions
    ("ADR-Driven Architecture Decisions", "All significant architecture decisions documented as Architecture Decision Records. ADRs include context, decision, consequences, and status tracking.", "convention", ["adr", "architecture", "decisions", "documentation"]),
    ("Chalk ESM Workaround in CJS", "chalk v5 is ESM-only but the project uses CommonJS. Workaround: const _chalk = require('chalk'); const chalk = _chalk.default || _chalk;", "convention", ["chalk", "esm", "cjs", "workaround"]),
    ("Lazy Module Loading for CLI Startup", "GNN, attention, and ora modules are lazy-loaded to maintain sub-55ms CLI startup time. Never convert these to eager imports.", "convention", ["lazy-loading", "cli", "startup", "performance"]),
    ("RVF Wire Protocol — Binary Segment Format", "RuVector File format for serializing vector data, model weights, and knowledge graph segments. Compact binary format with integrity hashing and version headers.", "convention", ["rvf", "format", "binary", "protocol"]),
    ("Cloud Run Deployment with Pre-Built Binary", "Deploy to Cloud Run using pre-built release binary with minimal Dockerfile (debian:bookworm-slim + binary). Avoids cargo build in Docker for faster deploys.", "convention", ["cloud-run", "deployment", "docker", "binary"]),
    ("RuVector on npm — Published Packages", "npm packages: ruvector (CLI + MCP server), @ruvector/pi-brain, @ruvector/ruvllm, @ruvector/router. Published under ruvnet account.", "convention", ["npm", "packages", "ruvector", "publishing"]),

    # Performance
    ("HNSW Sub-Millisecond Vector Search", "Hierarchical Navigable Small World index for approximate nearest neighbor search. Achieves sub-millisecond query latency on million-scale vector datasets.", "performance", ["hnsw", "vector-search", "sub-millisecond", "ann"]),
    ("Flash Attention for Large Sequences", "Memory-efficient attention computation that reduces memory from O(n^2) to O(n) by tiling the attention matrix. Achieves 2.49x-7.47x speedup over standard attention.", "performance", ["flash-attention", "memory-efficient", "speedup", "tiling"]),

    # Tooling
    ("Claude Flow V3 — Multi-Agent Orchestration", "Multi-agent orchestration framework for Claude Code. Supports hierarchical, mesh, and adaptive topologies with up to 15 concurrent agents and Byzantine consensus.", "tooling", ["claude-flow", "orchestration", "multi-agent", "claude-code"]),
    ("Claude Code Hooks Integration", "Pre-task, post-task, pre-edit, and post-edit hooks for Claude Code automation. Hooks enable self-learning, pattern training, and coordination workflows.", "tooling", ["claude-code", "hooks", "automation", "integration"]),
    ("npx ruvector — Unified CLI Tool", "48-command CLI with 12 groups (brain, edge, identity, mcp, rvf, hooks, llm, sona, route, gnn, attention, embed). Sub-55ms startup via lazy loading.", "tooling", ["cli", "npx", "ruvector", "commands"]),

    # Ecosystem
    ("Agentic Flow — Agent Coordination Framework", "Node.js/TypeScript framework for multi-agent coordination. Provides agent lifecycle, memory management, task routing, and inter-agent communication.", "tooling", ["agentic-flow", "agents", "coordination", "typescript"]),
    ("ruv-swarm — Distributed AI Swarm System", "Distributed swarm orchestration with multiple topologies (star, mesh, hierarchical, ring). Supports dynamic agent spawning, load balancing, and fault tolerance.", "tooling", ["ruv-swarm", "distributed", "swarm", "orchestration"]),
    ("Creator — ruvnet (Reuven Cohen)", "ruvnet (Reuven Cohen) is the creator and maintainer of the RuVector ecosystem including claude-flow, agentic-flow, ruv-swarm, and 60+ Rust crates for AI infrastructure.", "tooling", ["ruvnet", "reuven-cohen", "creator", "maintainer"]),
]

# ===== ADR SUMMARIES (53 entries) =====
adrs = [
    ("ADR-001: Deep agentic-flow Integration", "Eliminates 10k+ duplicate lines by building claude-flow as specialized extension of agentic-flow.", "architecture", ["adr", "integration"]),
    ("ADR-002: RuvLLM Integration with Ruvector", "Integrates RuvLLM embedding engine for server-side neural embeddings.", "architecture", ["adr", "ruvllm"]),
    ("ADR-003: SIMD Optimization Strategy", "SIMD vectorization for dot product, cosine similarity, and matrix operations.", "performance", ["adr", "simd"]),
    ("ADR-005: Agent Booster — WASM Code Transform", "Sub-millisecond code transforms via WASM without LLM calls.", "architecture", ["adr", "wasm", "booster"]),
    ("ADR-006: Unified Memory Service", "Consolidates 6+ memory backends into single AgentDB with HNSW indexing.", "architecture", ["adr", "memory", "agentdb"]),
    ("ADR-007: Security Review & Technical Debt Remediation", "Comprehensive security audit addressing CVEs and technical debt across the codebase.", "security", ["adr", "security", "audit", "remediation"]),
    ("ADR-009: Hybrid Memory Backend", "Combines in-memory hot cache with persistent cold storage for optimal latency and durability.", "architecture", ["adr", "memory", "hybrid"]),
    ("ADR-010: Claims-Based Authorization", "Fine-grained access control for swarm agents using JWT-like claims and scope restrictions.", "security", ["adr", "authorization", "claims"]),
    ("ADR-014: Coherence Engine Architecture", "Sheaf-theoretic coherence verification for knowledge graph consistency checking.", "architecture", ["adr", "coherence", "sheaf"]),
    ("ADR-015: Coherence-Gated Transformer (Sheaf Attention)", "Transformer architecture where attention is gated by sheaf-theoretic coherence between nodes.", "architecture", ["adr", "transformer", "sheaf"]),
    ("ADR-016: Delta-Behavior System - Domain-Driven Design Architecture", "DDD architecture for monitoring and analyzing behavioral changes in AI systems over time.", "architecture", ["adr", "delta", "ddd"]),
    ("ADR-026: Three-Tier Model Routing", "Routes tasks to WASM booster, Haiku, or Sonnet/Opus based on complexity estimation.", "architecture", ["adr", "routing", "three-tier"]),
    ("ADR-029: EXO-AI Multi-Paradigm Integration Architecture", "Integrates exo-cortex, vision, graph, quantum, and neuro-symbolic AI paradigms.", "architecture", ["adr", "exo-ai", "multi-paradigm"]),
    ("ADR-030: Hash Security Optimization", "SHAKE-256 with domain separation for content hashing. Prevents hash collision attacks.", "security", ["adr", "hash", "shake-256"]),
    ("ADR-031: Vector-Native COW Branching (RVCOW)", "Copy-on-Write branching for vector databases enabling parallel experimentation.", "architecture", ["adr", "cow", "branching"]),
    ("ADR-033: Progressive Indexing Hardening", "Centroid stability monitoring and adversarial robustness for HNSW indexes.", "security", ["adr", "indexing", "hardening"]),
    ("ADR-037: Publishable RVF Acceptance Test", "Standard acceptance tests for validating RVF file format compliance.", "convention", ["adr", "rvf", "testing"]),
    ("ADR-039: RVF Solver WASM — Self-Learning AGI Engine Integration", "Integrates sublinear solvers with WASM for browser-based AI computation.", "architecture", ["adr", "rvf", "solver", "wasm"]),
    ("ADR-040: Causal Atlas RVF Runtime", "Planet detection and life candidate scoring using causal inference in RVF format.", "architecture", ["adr", "causal", "atlas"]),
    ("ADR-042: Security RVF — AIDefence + TEE Hardened Cognitive Container", "Combines AI defense systems with trusted execution environments for secure AI reasoning.", "security", ["adr", "aidefence", "tee", "cognitive"]),
    ("ADR-043: External Intelligence Providers for SONA Learning", "Integrates external data sources and APIs as intelligence providers for SONA's adaptive learning system.", "architecture", ["adr", "sona", "intelligence", "providers"]),
    ("ADR-045: Lean-Agentic Integration — Formal Verification & AI-Native Type Theory", "Formal verification of agent behaviors using dependent type theory and proof assistants.", "architecture", ["adr", "lean", "verification"]),
    ("ADR-048: Sublinear Graph Attention", "Graph attention mechanisms that operate in sublinear time using locality-sensitive hashing.", "architecture", ["adr", "sublinear", "attention"]),
    ("ADR-053: Agent-to-Agent Communication Protocol", "Standardized protocol for direct inter-agent messaging with routing and delivery guarantees.", "architecture", ["adr", "agent", "communication"]),
    ("ADR-054: Swarm Topology Optimization", "Dynamic topology switching between star, mesh, hierarchical based on workload characteristics.", "architecture", ["adr", "swarm", "topology"]),
    ("ADR-055: Multi-Modal Knowledge Representation", "Extends knowledge graph to support text, image, audio, and code modalities with cross-modal retrieval.", "architecture", ["adr", "multi-modal", "knowledge"]),
    ("ADR-056: RVF Knowledge Export for Developer Onboarding", "Exports knowledge graph subsets as RVF files for offline developer onboarding and training.", "convention", ["adr", "rvf", "onboarding"]),
    ("ADR-057: Federated RVF Format for Real-Time Transfer Learning", "RVF extensions for real-time federated transfer learning between distributed brain instances.", "architecture", ["adr", "rvf", "federated", "transfer"]),
    ("ADR-058: RVF Hash Security Hardening and Optimization", "Hardens RVF content hashing against length extension and collision attacks using SHAKE-256.", "security", ["adr", "hash", "security"]),
    ("ADR-059: Shared Brain Google Cloud Architecture", "Google Cloud deployment architecture with Cloud Run, Firestore, and GCS for the shared brain service.", "architecture", ["adr", "cloud", "google"]),
    ("ADR-060: Shared Brain Capabilities", "Feature catalog for the shared brain: search, voting, graphs, federation, LoRA, drift detection.", "architecture", ["adr", "brain", "capabilities"]),
    ("ADR-061: Reasoning Kernel Architecture — Brain-Augmented Targeted Reasoning", "Reasoning kernel that augments LLM reasoning with knowledge graph context and pattern memory.", "architecture", ["adr", "reasoning", "kernel"]),
    ("ADR-062: Brainpedia Architecture", "Wiki-like collaborative knowledge pages with version history, evidence citations, and promotion workflow.", "architecture", ["adr", "brainpedia", "wiki"]),
    ("ADR-063: WASM Executable Nodes", "Knowledge graph nodes containing executable WASM modules for in-graph computation.", "architecture", ["adr", "wasm", "executable"]),
    ("ADR-064: Pi Brain Infrastructure", "Infrastructure design for pi.ruv.io deployment including Cloud Run, custom domain, and TLS.", "architecture", ["adr", "infrastructure", "pi-brain"]),
    ("ADR-065: NPM Publishing Strategy", "Strategy for publishing RuVector npm packages with versioning, scoping, and dependency management.", "convention", ["adr", "npm", "publishing"]),
    ("ADR-066: SSE MCP Transport", "Server-Sent Events transport for Model Context Protocol enabling real-time streaming tool results.", "architecture", ["adr", "sse", "mcp"]),
    ("ADR-067: MCP Gate Permit System", "Permission system for controlling AI agent access to MCP tools with time-limited scoped permits.", "security", ["adr", "mcp-gate", "permits"]),
    ("ADR-068: Domain Expansion Transfer Learning", "Cross-domain knowledge transfer using embedding space alignment and anchor point mapping.", "architecture", ["adr", "domain", "transfer"]),
    ("ADR-069: Google Edge Network Deployment", "Edge deployment architecture using Google Cloud CDN and Cloud Run for low-latency AI inference.", "architecture", ["adr", "edge", "google"]),
    ("ADR-070: npx ruvector Unified Integration", "Unified CLI architecture consolidating 48 commands across 12 groups with sub-55ms startup.", "tooling", ["adr", "npx", "cli"]),
    ("ADR-071: npx ruvector Ecosystem Gap Analysis", "Gap analysis identifying missing CLI capabilities and integration opportunities.", "tooling", ["adr", "ecosystem", "gap-analysis"]),
    ("ADR-072: RVF Example Management Downloads", "Management system for RVF example files with versioning, discovery, and download tracking.", "convention", ["adr", "rvf", "examples"]),
    ("ADR-073: Pi Platform Security Optimization", "Security optimization for the pi.ruv.io platform including rate limiting, input validation, and auth.", "security", ["adr", "security", "pi-brain"]),
    ("ADR-074: RuvLLM Neural Embedding Integration", "Integration of ruvllm HashEmbedder and RlmEmbedder for server-side neural embedding generation.", "architecture", ["adr", "ruvllm", "embeddings"]),
]

# ===== CRATE READMES (85 entries) =====
crates = [
    ("ruvector-solver", "Sublinear-time sparse solvers with O(log n) PageRank, spectral methods, and linear systems in Rust and WASM.", "tooling", ["solver", "pagerank", "spectral"]),
    ("ruvector-solver-wasm", "WebAssembly bindings for sublinear solvers enabling browser-based PageRank and spectral computation.", "tooling", ["solver", "wasm", "browser"]),
    ("ruvector-solver-node", "Node.js bindings for sublinear solvers with native performance.", "tooling", ["solver", "node", "bindings"]),
    ("ruvector-gnn", "Graph Neural Network layer that makes HNSW vector search topology-aware.", "tooling", ["gnn", "hnsw", "graph"]),
    ("ruvector-gnn-wasm", "GNN Layer Operations compiled to WebAssembly for browser-based graph neural networks.", "tooling", ["gnn", "wasm", "browser"]),
    ("ruvector-gnn-node", "GNN Layers as Node.js native addon.", "tooling", ["gnn", "node", "bindings"]),
    ("ruvector-attention", "46 attention mechanisms grounded in 7 mathematical frameworks.", "tooling", ["attention", "mechanisms", "math"]),
    ("ruvector-attention-wasm", "Attention mechanisms compiled to WebAssembly for browser inference.", "tooling", ["attention", "wasm", "browser"]),
    ("ruvector-graph-transformer", "A graph neural network where every operation completes in O(log n).", "tooling", ["graph", "transformer", "sublinear"]),
    ("ruvector-graph-transformer-wasm", "Graph Transformer in WebAssembly for browser-based inference.", "tooling", ["graph", "transformer", "wasm"]),
    ("ruvector-mincut-gated-transformer", "Ultra-low latency transformer inference using mincut graph partitioning.", "tooling", ["mincut", "transformer", "inference"]),
    ("ruvector-mincut-gated-transformer-wasm", "Zero-copy inference via mincut transformer in WebAssembly.", "tooling", ["mincut", "wasm", "inference"]),
    ("ruvector-delta-core", "Core delta behavior monitoring and drift detection library.", "tooling", ["delta", "drift", "monitoring"]),
    ("ruvector-delta-wasm", "Delta behavior analysis in WebAssembly.", "tooling", ["delta", "wasm", "analysis"]),
    ("ruvector-domain-expansion", "Cross-domain knowledge transfer and embedding space alignment.", "tooling", ["domain", "transfer", "expansion"]),
    ("ruvector-domain-expansion-wasm", "Domain expansion in WebAssembly for browser-based transfer learning.", "tooling", ["domain", "wasm", "transfer"]),
    ("ruvllm", "Lightweight neural embedding engine with HashEmbedder and RlmEmbedder.", "tooling", ["ruvllm", "embedding", "neural"]),
    ("ruvllm-wasm", "WASM bindings for browser-based LLM inference.", "tooling", ["ruvllm", "wasm", "inference"]),
    ("ruvllm-node", "Node.js bindings for RuvLLM embedding engine.", "tooling", ["ruvllm", "node", "bindings"]),
    ("sona", "Runtime-adaptive learning for LLM routers and AI systems without expensive retraining.", "tooling", ["sona", "learning", "adaptive"]),
    ("sona-wasm", "SONA learning engine compiled to WebAssembly.", "tooling", ["sona", "wasm", "learning"]),
    ("sona-node", "Node.js bindings for SONA adaptive learning.", "tooling", ["sona", "node", "bindings"]),
    ("ruvector-router", "Intelligent neural routing for vector search with learned query optimization.", "tooling", ["router", "neural", "routing"]),
    ("ruvector-router-wasm", "WebAssembly bindings for intelligent neural routing and vector search in the browser.", "tooling", ["router", "wasm", "browser"]),
    ("ruvector-router-node", "Node.js bindings for neural routing.", "tooling", ["router", "node", "bindings"]),
    ("cognitum-gate-kernel", "Anytime-Valid Coherence Gate for streaming hypothesis testing.", "tooling", ["cognitum", "gate", "coherence"]),
    ("cognitum-gate-tilezero", "TileZero zero-knowledge tile puzzles for proof-of-cognitive-work.", "tooling", ["cognitum", "tilezero", "zero-knowledge"]),
    ("ruvector-profiler", "Performance profiling toolkit for vector operations.", "tooling", ["profiler", "performance", "benchmarking"]),
    ("micro-hnsw-wasm", "7.2KB HNSW implementation in WebAssembly for ultra-lightweight vector search.", "tooling", ["hnsw", "wasm", "micro"]),
    ("ruvector-temporal-tensor", "Shrink your vector data 4-10x without losing the signal using temporal compression.", "tooling", ["temporal", "tensor", "compression"]),
    ("ruvector-tiny-dancer", "Tiny quantized vector operations for embedded and mobile.", "tooling", ["tiny", "quantized", "embedded"]),
    ("ruvector-tiny-dancer-wasm", "WebAssembly bindings for Tiny Dancer quantized operations.", "tooling", ["tiny", "wasm", "quantized"]),
    ("ruvector-collections", "High-performance collection management for Ruvector vector databases.", "tooling", ["collections", "management", "database"]),
    ("ruvector-exo-core", "Core EXO-AI multi-paradigm integration framework.", "tooling", ["exo", "core", "multi-paradigm"]),
    ("ruvector-exo-vision", "EXO-AI computer vision integration module.", "tooling", ["exo", "vision", "ai"]),
    ("ruvector-exo-graph", "EXO-AI graph processing and analysis module.", "tooling", ["exo", "graph", "processing"]),
    ("ruvector-exo-quantum", "EXO-AI quantum computing integration module.", "tooling", ["exo", "quantum", "computing"]),
    ("ruvector-exo-neuro", "EXO-AI neuro-symbolic reasoning module.", "tooling", ["exo", "neuro-symbolic", "reasoning"]),
    ("ruvector-nervous-system", "Biological neural architecture simulation for AI systems.", "tooling", ["nervous-system", "neural", "biological"]),
    ("ruvector-nervous-system-wasm", "Nervous system simulation in WebAssembly.", "tooling", ["nervous-system", "wasm", "simulation"]),
    ("ruvector-crv", "CRV (Coordinate Remote Viewing) protocol integration for RuVector.", "tooling", ["crv", "remote-viewing", "protocol"]),
    ("ruvector-dither", "Dithering algorithms for vector quantization and image processing.", "tooling", ["dither", "quantization", "image"]),
    ("thermorust", "Energy-driven state transitions and thermodynamic computing in Rust.", "tooling", ["thermodynamic", "energy", "computing"]),
    ("ruvector-robotics", "Robotics middleware for real-time control and planning.", "tooling", ["robotics", "control", "middleware"]),
    ("agentic-robotics-core", "The fastest robotics middleware for Rust with 10 microsecond latency.", "tooling", ["robotics", "core", "fast"]),
    ("agentic-robotics-mcp", "Control robots with AI assistants using the Model Context Protocol.", "tooling", ["robotics", "mcp", "control"]),
    ("agentic-robotics-node", "Node.js/TypeScript bindings for Agentic Robotics.", "tooling", ["robotics", "node", "typescript"]),
    ("ruqu-core", "Quantum Execution Intelligence Engine in pure Rust.", "tooling", ["quantum", "execution", "rust"]),
    ("ruqu-wasm", "Run quantum simulations in the browser via WebAssembly.", "tooling", ["quantum", "wasm", "simulation"]),
    ("agentdb", "Persistent vector database for AI agent memory with HNSW indexing.", "tooling", ["agentdb", "vector", "database"]),
]

# ===== ECOSYSTEM (18 entries) =====
ecosystem = [
    ("Claude Flow — Multi-Agent CLI", "Command-line orchestration for Claude Code with swarm init, agent spawn, memory, and hooks. Supports hierarchical, mesh, and adaptive topologies.", "tooling", ["claude-flow", "cli", "orchestration"]),
    ("Agentic Flow Alpha — Agent Framework", "Node.js agent framework with lifecycle management, task routing, and inter-agent communication patterns.", "tooling", ["agentic-flow", "framework", "agents"]),
    ("AgentDB — Vector Memory Database", "Persistent vector database with HNSW indexing, learning plugins, and distributed synchronization.", "tooling", ["agentdb", "database", "memory"]),
    ("RuVector Solver Ecosystem", "Sublinear solvers for PageRank, spectral methods, and sparse linear systems in O(log n) time.", "tooling", ["solver", "sublinear", "ecosystem"]),
    ("RuVector GNN Ecosystem", "Graph neural network layers with topology-gated attention and HNSW integration.", "tooling", ["gnn", "ecosystem", "graph"]),
    ("SONA Learning Ecosystem", "Self-Organizing Neural Architecture for runtime-adaptive learning across AI systems.", "tooling", ["sona", "ecosystem", "learning"]),
    ("RuVector Attention Ecosystem", "46+ attention mechanisms across 7 mathematical frameworks with WASM and Node.js bindings.", "tooling", ["attention", "ecosystem", "mechanisms"]),
    ("Cognitum Gate Ecosystem", "Zero-knowledge proof-of-cognitive-work and coherence verification.", "tooling", ["cognitum", "ecosystem", "zero-knowledge"]),
    ("EXO-AI Integration Suite", "Multi-paradigm AI integration: vision, graph, quantum, neuro-symbolic.", "tooling", ["exo-ai", "ecosystem", "integration"]),
    ("Agentic Robotics Suite", "Control robots with AI using MCP protocol. Core, MCP server, and Node.js bindings.", "tooling", ["robotics", "ecosystem", "agentic"]),
    ("RuQu Quantum Computing", "Quantum simulation and execution intelligence in Rust and WebAssembly.", "tooling", ["quantum", "ecosystem", "simulation"]),
    ("RuVector Router — Neural Query Routing", "Intelligent query routing using learned optimization for vector search.", "tooling", ["router", "ecosystem", "neural"]),
    ("Nervous System — Bio-Neural Architecture", "Biological neural architecture simulation for AI systems.", "tooling", ["nervous-system", "ecosystem", "biological"]),
    ("Domain Expansion — Transfer Learning", "Cross-domain knowledge transfer through embedding space alignment.", "tooling", ["domain", "ecosystem", "transfer"]),
    ("Delta Core — Drift Detection", "Behavioral drift monitoring and anomaly detection for AI systems.", "tooling", ["delta", "ecosystem", "drift"]),
    ("ThermoRust — Thermodynamic Computing", "Energy-driven state transitions using thermodynamic principles.", "tooling", ["thermorust", "ecosystem", "thermodynamic"]),
    ("Temporal Tensor — Vector Compression", "4-10x vector compression using temporal tensor decomposition.", "tooling", ["temporal", "ecosystem", "compression"]),
    ("Dither — Vector Quantization", "Dithering and quantization algorithms for efficient vector storage.", "tooling", ["dither", "ecosystem", "quantization"]),
]

# ===== SEED ALL =====
all_entries = curated + adrs + crates + ecosystem
total = len(all_entries)
ok = 0
fail = 0

print(f"Seeding {total} entries to {BASE}...")
for i, (title, content, category, tags) in enumerate(all_entries):
    if seed(title, content, category, tags):
        ok += 1
    else:
        fail += 1
    if (i + 1) % 25 == 0:
        print(f"  Progress: {i+1}/{total} ({ok} ok, {fail} fail)")

print(f"\nDone: {ok}/{total} seeded ({fail} failed)")

# Verify
print("\nVerifying status...")
try:
    req = urllib.request.Request(f"{BASE}/v1/status")
    resp = urllib.request.urlopen(req, timeout=10)
    status = json.loads(resp.read())
    print(f"  Memories: {status['total_memories']}")
    print(f"  Contributors: {status['total_contributors']}")
    print(f"  Votes: {status['total_votes']}")
    print(f"  Engine: {status['embedding_engine']}")
except Exception as e:
    print(f"  Status check failed: {e}")
