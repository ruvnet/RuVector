# RuVector animated walkthrough

This is the readable companion to the 23 chapter, 4 minute 36 second SVG tour. Each chapter lasts 12 seconds. The SVG loops automatically and displays a static title card when reduced motion is requested.

The original README header is preserved. The tour uses an entirely vector based intro and conceptual illustrations inspired by the supplied RuVector Explorer. The original attached bitmap is not embedded. The visual style follows the supplied Explorer screenshot: a black instrument canvas, branching search trees, orange path traversal, mint recall hints, and compact monospace labels. Tree, hyperbolic and vector space views are conceptual SVG animations; no Three.js or scripts are required. It is an explanation, not a live benchmark or interactive explorer.

## INTRO: Memory that stays. Search that understands.

RuVector combines local retrieval, relationships and feedback.

* Remember useful context
* Find related information
* Improve with measured outcomes

A guided tour of the system • Illustrations are conceptual

## 01 / THE BIG PICTURE: Give your agent a useful memory.

Capture → encode → store → recall → act → record feedback.

* Keep facts and past decisions
* Find context for the next task
* Choose what the agent may act on

Your application decides what to remember and which evidence to trust.

## 02 / EMBEDDINGS: Turn meaning into coordinates.

An embedding model turns text into a list of numbers.

* Similar ideas sit near each other
* Different wording can still match
* Use one model and dimension per store

Local semantic mode uses a model downloaded and cached on first use.

## 03 / SEMANTIC RECALL: Ask naturally. Find the relevant memory.

A new question becomes a vector in the same space.

* Question: Where can data run?
* Memory: Keep inference in Canada
* Retrieve context, then let the agent reason

Similarity helps find context. It does not establish truth.

## 04 / HNSW SEARCH: Take shortcuts through your data.

HNSW is a layered network of nearby vectors.

* Start with a few long distance hops
* Move into a denser local neighborhood
* Return the closest candidates found

Approximate search trades some recall for less search work.

## 05 / THE EXPLORER: Watch the search find its way.

Your attached explorer makes search behavior visible.

* Canopy and Tree reveal each hop
* Space shows a projected vector map
* Hyper shows branches inside a disk

The WASM engine and animated JavaScript trace are separate implementations.

## 06 / SEARCH QUALITY: Fast is useful when recall holds.

Compare approximate results with an exact flat scan.

* Recall: how many true neighbors return?
* Latency: how long does a query take?
* Search breadth: how much work is allowed?

Higher search breadth usually costs more work. Measure on your data.

## 07 / PERSISTENCE: Close the process. Keep the memory.

VectorDB persists vectors, metadata and configuration.

* Store a fact with source and tenant
* Reopen the same storagePath later
* Recall it in a new agent session

Current HNSW reopening rebuilds the index. Measure cold start time.

## 08 / MEMORY TYPES: Facts, experiences and ways of working.

Use memory types to organize what the agent knows.

* Working: current task and session
* Episodic: what happened and its outcome
* Semantic: facts • Procedural: skills

AgenticDB has persistent typed records; the unified ruvllm manager is in memory.

## 09 / RELATIONSHIPS: Find connections, not just similar text.

Graph memory links people, projects, events and decisions.

* Follow who approved a decision
* Connect an outcome to its procedure
* Reconstruct context across several hops

Graph reconstruction and learned shortcuts require separate integration.

## 10 / HYBRID RETRIEVAL: Combine meaning with exact details.

Different retrieval components answer different questions.

* Dense search finds similar meaning
* Sparse search preserves exact terms
* Filters and graph paths narrow context

Core filters apply to candidates; selective filters can return fewer results.

## 11 / TIME AND CONTEXT: Old facts may need a second look.

Temporal recall can prefer recent, supported memories.

* Let stale observations lose weight
* Keep timestamps and supporting sources
* Tune retention for the actual domain

Temporal coherence is a proof of concept for moderate memory sets.

## 12 / EXPLORER LEARNING: Remember a route. Adjust the search effort.

The demo learns query hints and tunes search breadth.

* Recall similar past queries and answers
* Score against an exact reference
* Store the route and adjust ef or nprobe

This demo does not change the index or run SONA. Exact scoring is a demo aid.

## 13 / SONA AND FEEDBACK: Learning needs an outcome.

SONA can adapt small weights from trajectories and rewards.

* Record what the agent tried
* Supply a success or quality signal
* Consolidate to protect useful learning

Reading memory alone does not train weights or guarantee improvement.

## 14 / LOCAL DECISIONS: Some tasks need a decision, not an essay.

The separate typesafe package supports bounded decisions.

* Route a support ticket to a department
* Estimate urgency with a score
* Abstain when confidence is insufficient

Use real semantic embeddings and labeled evaluation; the hash embedder is a test double.

## 15 / COMPRESSION: Fit more memory into less space.

Specialized components reduce storage and search costs.

* Quantization uses smaller representations
* Coarse search narrows the candidate set
* Reranking checks the strongest candidates

Compression can reduce recall. Core quantization settings alone do not compress storage.

## 16 / MEMORY LIFECYCLE: Keep what matters. Recover when needed.

Manage memory beyond insertion and search.

* Compact according to a chosen policy
* Use snapshots and portable RVF artifacts
* Branch memory for isolated experiments

Compaction and snapshot restore are not all wired into the default VectorDB path.

## 17 / TRUST AND ACCESS: Memory informs. Policy authorizes.

Keep retrieval separate from permission to act.

* Enforce user identity and access rules
* Track sources and tamper evident lineage
* Apply retention to exports and copies too

Filters are not full authorization. Witness logs do not encrypt the data.

## 18 / DEPLOYMENT: Start local. Add the pieces you need.

Choose the runtime that fits your application.

* Node.js or Rust for embedded services
* Separate WASM package for the browser
* HTTP service, PostgreSQL or RVF as needed

Installing ruvector does not activate every component in the repository.

## 19 / SHARED MEMORY: Share deliberately across agents.

Optional shared memory adds a separate data boundary.

* Review what leaves the local process
* Retain provenance and trust controls
* Evaluate replication for your topology

Shared Brain is optional and hosted. Replication primitives are not a complete network plane.

## 20 / AGENT TOOLS: Connect memory to your agent.

MCP and coding hooks expose memory operations.

* Start with an explicit tool policy
* Pin the installed package for automation
* Inspect and verify generated hook config

Read only MCP profile: RUVECTOR_MCP_PROFILE=readonly

## 21 / VALIDATION: Test the outcome, not just the animation.

Choose a workload and measure quality alongside cost.

* Record recall, p95 latency and memory
* Compare against a tuned fixed baseline
* Promote changes only after evaluation

Explorer timings and demo savings are not production performance guarantees.

## 22 / GET STARTED: Give your next agent a memory that lasts.

Start with npx. Remember one fact. Recall it in a new process.

* npx ruvector
* npx ruvector info
* npm install ruvector

No database server or API key required for the local memory quick start.

## Try persistent semantic memory

Use Node.js 20 or newer for the current npm package. `npx` fetches and runs the CLI; `npm install ruvector` installs it as a project dependency.

```bash
npx ruvector
npx ruvector info
npx ruvector hooks remember --semantic --type decision "The customer requires all inference to remain in Canada."
npx ruvector hooks recall --semantic --top-k 3 "Where may customer data be processed?"
```

Run the recall command as a new process in the same project. The acceptance test is that it retrieves the stored Canada decision. The first semantic command downloads and caches its embedding model.

## Sources and scope

* [Current capability map and boundaries](../../README.md)
* [Node.js API](../../docs/api/NODEJS_API.md)
* [SONA](../../crates/sona)
* [Local typed decisions](../../npm/packages/typesafe)
* [Benchmark guide](../../docs/benchmarks/BENCHMARKING_GUIDE.md)
* User supplied RuVector Explorer: index.html, ruvector_wasm.js and ruvector_wasm_bg.wasm. The JS demo stores query/answer hints and adjusts ef or nprobe; it does not train SONA or change the underlying index.

Source review: 2026-10-06. Repository README blob: 251d6ae7b6ce0fbe668224dd8eefccfb445c9cef.
