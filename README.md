<a href="https://cognitum.one/ruvector"><img src="assets/ruvector/ruvector-neon-header.gif" alt="RuVector animated neon logo: self learning vector intelligence" width="100%"></a>

# RuVector: Vector Search, Persistent Agent Memory, and Local AI Decisions

RuVector is a Rust native substrate for fast local decisions and agent memory across sessions. It combines local semantic embeddings, persistent vector retrieval, graph relationships, explicit feedback learning, memory lifecycle controls, and optional shared memory.

Built by [Reuven Cohen (rUv)](https://ruv.io/) as part of the [ruvnet open source AI stack](https://github.com/ruvnet/ruvnet). [Cognitum One](https://cognitum.one/ruvector) provides the commercial enterprise layer.

<a href="https://ruvnet.github.io/RuVector/explorer/"><img src="assets/ruvector/ruvector-trailer-v2.svg" alt="RuVector cinematic trailer: search, remember, learn. Open the interactive RuVector Explorer and follow vector search trajectories in your browser." width="100%"></a>

<p align="center"><sub><a href="https://ruvnet.github.io/RuVector/explorer/">Launch the interactive Explorer</a> · 32 second trailer · <a href="assets/ruvector/ruvector-walkthrough.md">Watch the full animation and explore all 23 chapters</a> · <code>npx ruvector</code></sub></p>

[![Crates.io](https://img.shields.io/crates/v/ruvector-core.svg)](https://crates.io/crates/ruvector-core)
[![npm](https://img.shields.io/npm/v/ruvector.svg)](https://www.npmjs.com/package/ruvector)
[![npm monthly downloads](https://img.shields.io/npm/dm/ruvector.svg?label=monthly%20downloads)](https://www.npmjs.com/package/ruvector)
[![npm all-time downloads](https://img.shields.io/npm/dt/ruvector.svg?label=all-time%20downloads)](https://www.npmjs.com/package/ruvector)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)


**Explore:** [Quick starts](#quick-start-choose-your-track) · [Build recipes](#build-by-example) · [Tutorials](./examples/README.md) · [Library catalog](#library-and-capability-map) · [Contrastive AI](#where-does-contrastive-ai-fit) · [Deployment](#deployment-surfaces) · [Benchmarks](#reproduce-the-evidence)

## System 0, System 1, and System 2

RuVector supports three complementary roles in the wider ruvnet stack. These are architecture groups, not automatic execution tiers or a claim that every component is installed together.

| System | Role | Libraries | Tutorials and examples |
| :--- | :--- | :--- | :--- |
| **[System 0: Sense & Respond](#system-0-sense--respond)** | Encode observations and make bounded local decisions | [ONNX embeddings](./crates/ruvector-core/src/embeddings.rs), [typed decisions](./npm/packages/typesafe), [browser WASM](https://www.npmjs.com/package/@ruvector/wasm) | [30 second memory quick start](#remember-and-recall-in-30-seconds), [typed decision guide](./npm/packages/typesafe/README.md), [browser example](./examples/wasm-vanilla/README.md) |
| **[System 1: Learn & Remember](#system-1-learn--remember)** | Persist context, retrieve evidence, and adapt from explicit feedback | [VectorDB](./crates/ruvector-core), [graph memory](./crates/ruvector-graph), [SONA](./crates/sona), [contrastive primitives](./crates/ruvector-cnn/src/contrastive/mod.rs) | [Node.js memory tutorial](#embed-persistent-memory-in-nodejs), [Python guide](./docs/python/README.md), [SONA guide](./crates/sona/README.md) |
| **[System 2: Reason & Orchestrate](#system-2-reason--orchestrate)** | Reconstruct multi step context, coordinate work, and govern execution | [RuvLLM](./crates/ruvllm), [RVF](./crates/rvf), [Ruflo](https://github.com/ruvnet/ruflo), [MetaHarness](https://github.com/ruvnet/metaharness) | [MRAgent example](./examples/mragent), [MCP integration](#agent-integration), [Ruflo getting started](https://github.com/ruvnet/ruflo#readme) |

![Animated RuVector systems diagram: System 0 encodes and responds, System 1 learns and remembers, System 2 reasons and orchestrates](assets/ruvector/three-systems.svg)

## Quick start: choose your track

| Track | Use it for | Prerequisites |
| :--- | :--- | :--- |
| [npm `ruvector`](#track-1-npm-ruvector) | Project memory and a Node.js application | Node.js and npm; supported native backend |
| [MCP via npm `ruvector`](#track-2-mcp-via-npm-ruvector) | Give an MCP client access to RuVector tools | Node.js, npm, and an MCP compatible client |
| [PyPI `ruvector`](#track-3-pypi-ruvector) | Vector search from Python | Python 3.9+, Rust, and a virtual environment for the source install |
| [crates.io `ruvector-core`](#track-4-cratesio-ruvector-core) | Embed the Rust core directly | Rust toolchain and Cargo |

### Track 1: npm ruvector

Package: [`ruvector` on npm](https://www.npmjs.com/package/ruvector).

Install the npm package and run the CLI:

```bash
npm install ruvector
npx ruvector info
npx ruvector hooks remember --semantic --type decision \
  "The customer requires all inference to remain in Canada."
npx ruvector hooks recall --semantic --top-k 3 \
  "Where may customer data be processed?"
```

The first semantic command downloads a local embedding model. Reuse the same project directory and model to retain searchable context. [Node.js SDK example](#embed-persistent-memory-in-nodejs) · [Node.js API](./docs/api/NODEJS_API.md).

### Track 2: MCP via npm ruvector

The MCP server is included in the [`ruvector` npm package](https://www.npmjs.com/package/ruvector).

Inspect the tools and start with the read only profile:

```bash
npx ruvector mcp tools
RUVECTOR_MCP_PROFILE=readonly npx ruvector mcp start
```

Configure your MCP client to launch `npx` with arguments `ruvector mcp start`, environment `RUVECTOR_MCP_PROFILE=readonly`, and the project as its working directory. Use the client configuration format it supports. Enable writes only through an explicit tool policy. [MCP integration and policy](#agent-integration).

#### Claude Code setup

Run these commands from your project directory with Node.js, npm, and Claude Code installed:

```bash
npm install ruvector
claude mcp add --scope project --env RUVECTOR_MCP_PROFILE=readonly --transport stdio ruvector -- npx -y ruvector mcp start
claude mcp get ruvector
```

Open Claude Code in the same project, approve the project MCP server when prompted, and use `/mcp` to check the connection. Project scope stores the configuration in `.mcp.json`. Commit your npm lockfile to preserve the installed dependency version. See [Claude Code's MCP documentation](https://code.claude.com/docs/en/mcp).

**Try this prompt:**

> Use the ruvector MCP server to inspect the available memory and retrieve context relevant to this project. Report which records support your answer. If the store is empty, say so.

**Optional project instruction:** add this to your project's `CLAUDE.md`:

```markdown
## RuVector memory

* Retrieve relevant project context through the ruvector MCP server before using remembered decisions.
* Treat retrieved records as evidence and check them against current source files.
* Include record IDs or provenance when available.
* Respect the configured read only tool policy. Do not bypass it with shell commands.
* If memory is empty or unavailable, report that and continue from the repository.
```

**Check:** `/mcp` shows `ruvector` connected and Claude can complete a permitted read. To populate memory, use the [npm track](#track-1-npm-ruvector) yourself or configure an explicit write policy through [agent integration](#agent-integration).

### Track 3: PyPI ruvector

Python distribution name: `ruvector`; import name: `ruvector`. See the [Python package manifest](./crates/ruvector-py/pyproject.toml) and [installation guide](./docs/python/README.md#install).

The [Python guide](./docs/python/README.md#install) currently documents a source installation. This path avoids assuming a published PyPI wheel is available.

```bash
git clone https://github.com/ruvnet/RuVector.git
cd RuVector
python3 -m venv .venv
source .venv/bin/activate
python -m pip install maturin
cd crates/ruvector-py
maturin develop --release
```

The activation command above is for bash or zsh; on Windows use `.venv\Scripts\Activate.ps1`. Rust and its platform build tools are required.

```python
import numpy as np
from ruvector import Collection

memory = Collection.create(dim=3)
vector = np.array([1.0, 0.0, 0.0], dtype=np.float32)
memory.insert(vector, metadata={"text": "A stored example"})
hits = memory.search(vector, k=1)
print(hits[0].metadata)
memory.save("my-memory")
restored = Collection.load("my-memory")
assert len(restored) == 1
```

This example uses a fixed vector to demonstrate storage, recall, and persistence. Use your embedding model for semantic text search. [Python SDK tutorial and integrations](./docs/python/README.md).


### Track 4: crates.io ruvector-core

Package: [`ruvector-core` on crates.io](https://crates.io/crates/ruvector-core); Rust import: `ruvector_core`.

Create a small application with the persistent storage feature. This example uses exact search and disables the default optional features.

```bash
cargo new ruvector-memory
cd ruvector-memory
cargo add ruvector-core --no-default-features --features storage
```

Replace `src/main.rs` with:

```rust
use ruvector_core::{DbOptions, SearchQuery, VectorDB, VectorEntry};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let db = VectorDB::new(DbOptions {
        dimensions: 3,
        storage_path: "./agent-memory.db".into(),
        hnsw_config: None,
        quantization: None,
        ..Default::default()
    })?;

    db.insert(VectorEntry {
        id: Some("example-1".into()),
        vector: vec![1.0, 0.0, 0.0],
        metadata: None,
    })?;

    let hits = db.search(SearchQuery {
        vector: vec![1.0, 0.0, 0.0],
        k: 1,
        filter: None,
        ef_search: None,
    })?;

    assert_eq!(hits[0].id, "example-1");
    println!("Nearest memory: {} (score: {})", hits[0].id, hits[0].score);
    Ok(())
}
```

```bash
cargo run --release
```

The fixed vector demonstrates insertion and retrieval; supply embeddings for semantic search. Keep `Cargo.lock` for reproducible dependency resolution. [Rust API reference](./docs/api/RUST_API.md) · [Current core types](./crates/ruvector-core/src/types.rs) · [Storage and search implementation](./crates/ruvector-core/src/vector_db.rs).

### Where does contrastive AI fit?

Contrastive AI spans these groups: learn useful distinctions in System 1, apply them to bounded System 0 decisions, and evaluate their use in System 2 workflows. Representation learning, graph diagnostics, and promotion policy are distinct mechanisms.

| Concept | Purpose | Library or implementation | Learn by example |
| :--- | :--- | :--- | :--- |
| **Similarity and separation** | Learn representations that bring related examples closer and separate mismatches | [InfoNCE and triplet losses](./crates/ruvector-cnn/src/contrastive/mod.rs) | [Contrastive training example](./crates/ruvllm/examples/train_contrastive.rs), [training guide](./crates/ruvllm/src/training/README.md) |
| **Structure and coherence** | Examine relationships, weak graph connections, and time sensitive recall | [MinCut](./crates/ruvector-mincut), [temporal coherence](./crates/ruvector-temporal-coherence) | [MinCut guide](./crates/ruvector-mincut/README.md), [temporal memory design](./docs/adr/ADR-211-temporal-coherence-agent-memory.md) |
| **Feedback and bounded adaptation** | Update learning state from outcomes and evaluate proposed changes | [SONA](./crates/sona), [MRAgent optimization example](./examples/mragent) | [SONA guide](./crates/sona/README.md), [MRAgent design](./docs/adr/ADR-269-mragent-graph-memory-darwin-optimization.md) |

[Watch rUv's illustrated contrastive AI walkthrough](https://github.com/ruvnet/ruvnet/blob/main/docs/contrastive-ai-walkthrough.md). Contrastive losses do not establish factual truth; graph coherence does not replace task evaluation, access control, or promotion approval. These components require explicit integration and are not all enabled in the default search path.

## Build by example

Choose one outcome, follow its guide, and check the result before adding more components.

| Build | Libraries to start with | Example or tutorial | Acceptance check |
| :--- | :--- | :--- | :--- |
| **Persistent agent memory** | npm `ruvector` or Rust `ruvector-core` | [Node.js](#embed-persistent-memory-in-nodejs), [Rust](#track-4-cratesio-ruvector-core), [Python](#track-3-pypi-ruvector) | A fresh process retrieves a record written by the previous process |
| **Local ticket routing** | npm `@ruvector/typesafe` | [Typed decisions and evaluation](./npm/packages/typesafe/README.md) | Measure accuracy, abstention, and p95 latency on your held out tickets |
| **Knowledge graph retrieval** | npm `@ruvector/graph-node`, `@ruvector/kge` | [Graph examples](./examples/graph/README.md), [KGE guide](./npm/packages/kge/README.md) | Return source relationships; evaluate predicted links separately from stored facts |
| **Browser vector search** | npm `@ruvector/wasm` | [Vanilla JavaScript](./examples/wasm-vanilla/README.md), [React](./examples/wasm-react/README.md) | Insert and query in the browser; explicitly test persistence if required |
| **Feedback driven memory** | SONA, MRAgent reconstruction | [SONA](./crates/sona/README.md), [MRAgent](./examples/mragent/README.md) | Compare a frozen baseline with the candidate on a separate evaluation set |
| **Portable memory artifacts** | npm `@ruvector/rvf`, `@ruvector/rvf-mcp-server` | [RVF examples](./examples/rvf/README.md), [MCP setup](./npm/packages/rvf-mcp-server/README.md) | Reopen the artifact and confirm expected records and configured tool permissions |

### How the components work together

![Animated reference architecture: encode observations, retrieve evidence, act under policy, and evaluate feedback before adaptation](./assets/ruvector/build-loop.svg)

The application connects these components. Keep source IDs with retrieved evidence, record the outcome of an action, and evaluate any proposed learning change before retaining it. See [contrastive AI](#where-does-contrastive-ai-fit) for the distinction between representation learning, graph diagnostics, and promotion policy.

## What is RuVector used for?

| Goal | Start here |
| :--- | :--- |
| Search vectors or retain agent context | [`ruvector` Node.js SDK](#embed-persistent-memory-in-nodejs), [`ruvector-core` Rust crate](./crates/ruvector-core) |
| Classify text or return typed local decisions | [`@ruvector/typesafe`](./npm/packages/typesafe) |
| Explore vectors visually | [Interactive RuVector Explorer](https://ruvnet.github.io/RuVector/explorer/) |
| Choose graph, browser, database, or shared memory components | [Memory paths](#choose-a-memory-path) and [deployment surfaces](#deployment-surfaces) |

[Quick start](#remember-and-recall-in-30-seconds) · [Memory loop](#the-memory-loop) · [Limitations](#known-boundaries) · [Reproducible benchmarks](#reproduce-the-evidence)

The default retrieval path runs locally. Learning happens from recorded outcomes and feedback, not from reads alone. Hosted services remain optional and create a separate data boundary.

### How do local typed decisions work?

![Animated typed decision workflow: text input, local embeddings and decision heads, typed output or abstention](assets/ruvector/typed-decision-walkthrough.svg)

[`@ruvector/typesafe`](./npm/packages/typesafe) turns text into typed `choice`, `score`, and `noul` decisions using local embeddings and native decision heads, with a WASM fallback. It returns confidence and abstention information, supports labeled examples and evaluation, and can serve a Jev-compatible HTTP API. Use it for bounded tasks such as ticket routing, intent classification, and urgency assessment where a full language model call is unnecessary. The decision engine is a separate package; installing the root `ruvector` package does not enable it automatically.

On the [documented 150-ticket test split](./npm/packages/typesafe/README.md#measured), the local ONNX decision engine reports 4–10 ms p95 for its campaign configurations, with 77.3–84.0% department accuracy; the Jev replay reference reports 231 ms p95 and 85.3% accuracy. Those are workload-specific measurements, not a universal speed or quality guarantee. The default hash embedder is a test double and is not calibrated for production decisions. See the [typed decision quick start and benchmark details](./npm/packages/typesafe/README.md).

<a id="remember-and-recall-in-30-seconds"></a>

### Where is CLI memory stored?

The [npm quick start](#track-1-npm-ruvector) stores memory under the current project for later processes. The first semantic command downloads and caches `all-MiniLM-L6-v2`. Keep one embedding model and dimension per store; use `npx ruvector hooks reembed` before changing an existing store from hash to semantic embeddings. Inspect it with `npx ruvector hooks stats`.

## Embed persistent memory in Node.js

```bash
npm install ruvector
```

```javascript
const { OnnxEmbedder, VectorDB } = require('ruvector');

async function main() {
  const embedder = new OnnxEmbedder();
  await embedder.init();

  const db = new VectorDB({
    dimensions: 384,
    distanceMetric: 'cosine',
    storagePath: './agent-memory.db',
  });

  const memories = [
    {
      id: 'decision-1',
      text: 'The customer requires all inference to remain in Canada.',
      kind: 'decision',
    },
    {
      id: 'episode-1',
      text: 'The Toronto pilot passed its privacy review on Tuesday.',
      kind: 'episode',
    },
    {
      id: 'procedure-1',
      text: 'Escalate production access through the security owner.',
      kind: 'procedure',
    },
  ];

  for (const memory of memories) {
    const vector = await embedder.embedPassage(memory.text);
    await db.insert({
      id: memory.id,
      vector,
      metadata: {
        text: memory.text,
        kind: memory.kind,
        tenant: 'acme',
        createdAt: Date.now(),
      },
    });
  }

  const query = await embedder.embedQuery(
    'Where may the customer data be processed?',
  );

  const results = await db.search({
    vector: query,
    k: 3,
    filter: { tenant: 'acme' },
  });

  console.log(results.map(({ score, metadata }) => ({ score, ...metadata })));
}

main().catch(console.error);
```

Reopen the same `storagePath` in another process to recover the stored vectors, metadata, configuration, and searchability. Search `score` is a distance, so lower values are closer. See the [Node.js API](./docs/api/NODEJS_API.md) and [Rust API](./docs/api/RUST_API.md) for the complete interfaces.

## Language tutorials

Follow the complete [Node.js write and reopen tutorial](./examples/nodejs/README.md), [Python SDK and integration guide](./docs/python/README.md), or [Rust persistence tutorial](./examples/rust/README.md). The [examples hub](./examples/README.md) groups browser, graph, contrastive learning, runtime, and deployment examples by System 0, 1, and 2.

## The memory loop

![Animated RuVector memory workflow: encode, persist, recall, then record feedback for explicit adaptation](assets/ruvector/memory-feedback-walkthrough.svg)

<details>
<summary>View the detailed memory flow diagram</summary>

```mermaid
flowchart TD
    A[Capture an event, fact, or outcome] --> B[Create a local or external embedding]
    B --> C[Persist vectors, metadata, and relationships]
    C --> D[Recall by similarity, filters, time, or graph]
    D --> E[Use memory in an agent decision]
    E --> F[Record outcome and feedback]
    F --> G[Adapt ranking or learning state]
    G --> C
    C --> H[Compact, snapshot, branch, or replicate]
```

</details>

RuVector provides primitives for this loop. Your application remains responsible for deciding what is worth remembering, which evidence is trusted, when a memory expires, and which actions recalled context may influence.

## What memory means in RuVector

Memory classes are application semantics over vectors, metadata, and graphs. The core store is general purpose. RuVector currently exposes two typed layers:

1. [`ruvllm::context::AgenticMemory`](./crates/ruvllm/src/context/agentic_memory.rs) combines working, episodic, semantic, and procedural memory behind one runtime API. It is implemented, but its unified manager is currently in memory and its cross type consolidation method is not complete.

2. [`ruvector-core::AgenticDB`](./crates/ruvector-core/src/agenticdb.rs) persists Reflexion episodes, skills, causal edges, learning sessions, policy state, session turns, and a hash linked witness log. Its typed memory APIs support ONNX, Candle, and API embedding providers for semantic retrieval.

| Memory class | Representation | RuVector surface |
| --- | --- | --- |
| Working and session | Current task, scratchpad, tool cache, turns, namespace, TTL | [`WorkingMemory`](./crates/ruvllm/src/context/working_memory.rs), [`SessionStateIndex`](./crates/ruvector-core/src/agenticdb.rs) |
| Episodic and Reflexion | Trajectory, task, action, observation, critique, outcome | [`EpisodicMemory`](./crates/ruvllm/src/context/episodic_memory.rs), [`ReflexionEpisode`](./crates/ruvector-core/src/agenticdb.rs) |
| Semantic | Facts, confidence, source, tags, relations, collection | [`VectorDB`](./crates/ruvector-core), [`SemanticFact`](./crates/ruvllm/src/context/agentic_memory.rs) |
| Procedural | Skills, actions, triggers, examples, policies, Q values | [`ProceduralSkill`](./crates/ruvllm/src/context/agentic_memory.rs), [`PolicyMemoryStore`](./crates/ruvector-core/src/agenticdb.rs) |
| Causal and relational | Nodes, edges, hyperedges, Cypher paths | [`ruvector-graph`](./crates/ruvector-graph) |
| Learning | Trajectories, rewards, adapters, EWC state | [`SONA`](./crates/sona) |
| Shared | Contributions, provenance, voting, transfer | [`mcp-brain`](./crates/mcp-brain) |
| Auditable | Hash linked entries, snapshots, RVF witnesses | [`WitnessLog`](./crates/ruvector-core/src/agenticdb.rs), [`ruvector-snapshot`](./crates/ruvector-snapshot), [RVF](./crates/rvf) |

## Library and capability map

Choose libraries by responsibility. These groups are a navigation model: a library can serve more than one system. npm names below come from package manifests; crate links point to repository guides. Publication, platform support, and maturity vary by package. Installing `ruvector` does not install every library in this monorepo.

[System 0 libraries](#sensing-and-decision-libraries) · [System 1 libraries](#memory-and-learning-libraries) · [System 2 libraries](#runtime-orchestration-and-governance-libraries) · [All npm packages](./npm/packages) · [All Rust crates](./crates) · [All examples](./examples)

### System 0: Sense & Respond

![System 0: Sense & Respond library flow](./assets/ruvector/system-0-library-header.svg)

Encode incoming observations for retrieval and bounded decisions. [Typed decision tutorial](./npm/packages/typesafe/README.md) · [Semantic embeddings setup](./docs/adr/ADR-210-default-on-semantic-embeddings-minilm.md)

#### Sensing and decision libraries

| Library | Purpose | Guide or example |
| :--- | :--- | :--- |
| [`@ruvector/typesafe`](./npm/packages/typesafe/README.md) | Local choices, scores, and typed decisions | [Decision walkthrough](./npm/packages/typesafe/README.md) |
| [`@ruvector/cnn`](./npm/packages/ruvector-cnn/README.md) | Image feature extraction and embeddings | [CNN guide](./crates/ruvector-cnn/README.md) |
| [`@ruvector/router`](./npm/packages/router/README.md) | Match intent to a configured route | [Semantic routing](./npm/packages/router/README.md) |
| [`ruvector-embed-core`](./crates/ruvector-embed-core/README.md) | Embedding primitives | [Local ONNX example](./examples/onnx-embeddings/README.md) |
| [`ruvector-mmwave`](./crates/ruvector-mmwave/README.md) | Radar sensing components | [Crate guide](./crates/ruvector-mmwave/README.md) |
| [`ruvector-robotics`](./crates/ruvector-robotics/README.md) | Robotics integration | [Robotics core](./crates/agentic-robotics-core/README.md) |

**Try it:** [ONNX in WASM](./examples/onnx-embeddings-wasm/README.md) · [Browser with React](./examples/wasm-react/README.md) · [Browser without a framework](./examples/wasm-vanilla/README.md) · [Edge examples](./examples/edge/README.md).

#### Capture and encode

| Capability | What it enables | Surface |
| --- | --- | --- |
| Local semantic embeddings | Text memory without a per query API fee | [`OnnxEmbedder`](./docs/adr/ADR-210-default-on-semantic-embeddings-minilm.md) |
| External embeddings | Bring an existing embedding model or provider | [`EmbeddingProvider`](./crates/ruvector-core/src/embeddings.rs) |
| Embedding provenance | Track model, dimension, normalization, and query or passage role | [ADR 210](./docs/adr/ADR-210-default-on-semantic-embeddings-minilm.md) |
| Batch and parallel embedding | Higher throughput during memory ingestion | [ONNX implementation](./docs/adr/ADR-210-default-on-semantic-embeddings-minilm.md) |

### System 1: Learn & Remember

![System 1: Learn & Remember library flow](./assets/ruvector/system-1-library-header.svg)

Store context, reconstruct relevant evidence, and adapt from recorded feedback. [Node.js tutorial](#embed-persistent-memory-in-nodejs) · [Python tutorial](./docs/python/README.md) · [Contrastive training guide](./crates/ruvllm/src/training/README.md)

#### Memory and learning libraries

| Library | Purpose | Guide or example |
| :--- | :--- | :--- |
| [`@ruvector/kge`](./npm/packages/kge/README.md) | Knowledge graph embeddings and link prediction | [KGE tutorial](./npm/packages/kge/README.md) |
| [`@ruvector/graph-node`](./npm/packages/graph-node/README.md) | Native graph and hypergraph access | [Graph examples](./examples/graph/README.md) |
| [`@ruvector/sona`](./npm/packages/sona/README.md) | Adaptation from trajectories and rewards | [SONA architecture](./crates/sona/README.md) |
| [`@ruvector/diskann`](./npm/packages/diskann/README.md) | Disk oriented approximate nearest neighbors | [DiskANN guide](./npm/packages/diskann/README.md) |
| [`rvlite`](./npm/packages/rvlite/README.md) | Lightweight SQL, SPARQL, and Cypher memory | [rvlite tutorial](./npm/packages/rvlite/README.md) |
| [`ruvector-mincut`](./crates/ruvector-mincut/README.md) | Graph partition and coherence diagnostics | [MinCut examples](./examples/mincut/README.md) |
| [`ruvector-coherence`](./crates/ruvector-coherence/README.md) | Structural coherence components | [Coherence guide](./crates/ruvector-coherence/README.md) |
| [`ruvector-memory-admission`](./crates/ruvector-memory-admission/README.md) | Decide which observations enter memory | [Admission guide](./crates/ruvector-memory-admission/README.md) |
| [`ruvector-query-cache`](./crates/ruvector-query-cache/README.md) | Cache retrieval work | [Cache guide](./crates/ruvector-query-cache/README.md) |
| [`ruvector-recall-bounded`](./crates/ruvector-recall-bounded/README.md) | Bounded recall components | [Recall guide](./crates/ruvector-recall-bounded/README.md) |
| [`ruvector-retrieval-receipt`](./crates/ruvector-retrieval-receipt/README.md) | Retrieval evidence and receipts | [Receipt guide](./crates/ruvector-retrieval-receipt/README.md) |

**Try it:** [Node.js examples](./examples/nodejs/README.md) · [Rust examples](./examples/rust/README.md) · [MRAgent reconstruction](./examples/mragent/README.md) · [Contrastive training](./crates/ruvllm/src/training/README.md).

#### Persist and organize

| Capability | What it enables | Surface |
| --- | --- | --- |
| Durable vector storage | Vectors, metadata, deletes, and restart recovery | [`ruvector-core`](./crates/ruvector-core) |
| Unified four type runtime memory | Working, episodic, semantic, and procedural recall | [`AgenticMemory`](./crates/ruvllm/src/context/agentic_memory.rs) |
| Typed persistent agent records | Reflexion episodes, skills, causal edges, policy state, sessions, and witness logs | [`AgenticDB`](./crates/ruvector-core/src/agenticdb.rs) |
| HNSW and flat indexes | Approximate or exact local similarity search | [`ruvector-core`](./crates/ruvector-core/src/index) |
| Collections and aliases | Separate schemas and namespaces by workload | [`ruvector-collections`](./crates/ruvector-collections) |
| Graph and hypergraph storage | Explicit relationships and multi-hop memory | [`ruvector-graph`](./crates/ruvector-graph) |
| High write ingestion | Mutable L0 memory plus background L1 and L2 compaction | [`ruvector-lsm-ann`](./crates/ruvector-lsm-ann) |
| Edge and embedded persistence | Lightweight local vector storage through the RVF Core Profile | [`rvlite`](./crates/rvf/rvf-adapters/rvlite) |
| PostgreSQL extension | Keep vector memory beside relational data | [`ruvector-postgres`](./crates/ruvector-postgres) |

#### Recall and reconstruct

| Capability | Best use | Surface |
| --- | --- | --- |
| Dense similarity | General semantic recall | [`VectorDB::search`](./crates/ruvector-core/src/vector_db.rs) |
| Metadata filtering | Simple structured narrowing | [`SearchQuery`](./crates/ruvector-core/src/types.rs) |
| Sparse and dense fusion | Exact terms plus semantic meaning | [`ruvector-hybrid`](./crates/ruvector-hybrid), [ADR 256](./docs/adr/ADR-256-hybrid-sparse-dense-search.md) |
| Predicate aware ANN | Selective filters without post filter recall collapse | [`ruvector-acorn`](./crates/ruvector-acorn) |
| Temporal decay | Prefer recent memories when the domain changes | [`ruvector-temporal-coherence`](./crates/ruvector-temporal-coherence), [ADR 211](./docs/adr/ADR-211-temporal-coherence-agent-memory.md) |
| Coherence gating | Prefer memories supported by related observations | [`ruvector-temporal-coherence`](./crates/ruvector-temporal-coherence) |
| Graph reconstruction | Follow Cue, Tag, and Content associations instead of retrieving one flat chunk | [MRAgent example](./examples/mragent), [ADR 269](./docs/adr/ADR-269-mragent-graph-memory-darwin-optimization.md) |
| Multi-vector MaxSim | Late interaction over token or passage vectors | [`ruvector-maxsim`](./crates/ruvector-maxsim), [ADR 252](./docs/adr/ADR-252-multi-vector-maxsim.md) |
| GNN reranking | Rerank a noisy candidate graph | [`ruvector-gnn-rerank`](./crates/ruvector-gnn-rerank), [ADR 194](./docs/adr/ADR-194-gnn-rerank.md) |
| Matryoshka funnel | Coarse to fine search for truncatable embeddings | [`ruvector-matryoshka`](./crates/ruvector-matryoshka) |
| Disk backed ANN | Move read heavy indexes toward SSD scale | [`ruvector-diskann`](./crates/ruvector-diskann) |

#### Learn and adapt

| Capability | What changes | Trigger |
| --- | --- | --- |
| SONA MicroLoRA | Small adapter weights | Recorded trajectory and reward |
| EWC++ consolidation | Protects important learned weights from catastrophic forgetting | Explicit consolidation |
| Outcome aware routing | Policy and routing preferences | Success, failure, or quality signal |
| GNN reranking | Candidate ordering | Training data or configured reranker |
| Self reconstructing graph memory | Shortcut edges after successful reconstruction | Successful graph traversal |
| Darwin optimization | Retrieval and reconstruction configuration | External benchmark and promotion gate |

Reading or searching memory does not, by itself, mutate learned weights or guarantee better future results.

#### Consolidate, compress, and recover

| Capability | What it controls | Surface |
| --- | --- | --- |
| LRU, LFU, and coherence compaction | Which memories survive a capacity limit | [`ruvector-agent-memory`](./crates/ruvector-agent-memory), [ADR 252](./docs/adr/ADR-252-agent-memory-compaction.md) |
| Temporal tensor codecs | Low bit storage and temporal segment reuse | [`ruvector-temporal-tensor`](./crates/ruvector-temporal-tensor) |
| Product quantization | Compressed candidate search with exact query vectors | [`ruvector-pq-search`](./crates/ruvector-pq-search) |
| RaBitQ | Deterministic one bit candidate encoding and optional reranking | [`ruvector-rabitq`](./crates/ruvector-rabitq) |
| Graph condensation | Smaller graph memory while retaining original member provenance | [`ruvector-graph-condense`](./crates/ruvector-graph-condense) |
| Full snapshots | Serialized recovery data with compression and checksums | [`ruvector-snapshot`](./crates/ruvector-snapshot) |
| Copy on write branches | Isolated memory experiments without full copies | [RVF](./crates/rvf) |
| Cache consistency modes | Fresh, eventual, or frozen reads across data sources | [`ruvector-rulake`](./crates/ruvector-rulake) |

The `DbOptions.quantization` field in `ruvector-core` is persisted but is not currently applied to core storage or indexes. Use a specialized compression crate when physical compression is required. See the source note in [`types.rs`](./crates/ruvector-core/src/types.rs).

### System 2: Reason & Orchestrate

![System 2: Reason & Orchestrate library flow](./assets/ruvector/system-2-library-header.svg)

Combine memory with application reasoning, agent coordination, and explicit governance. Ruflo and MetaHarness are complementary external projects; RuVector supplies memory and supporting primitives. [MCP integration tutorial](#agent-integration) · [Graph reconstruction example](./examples/mragent) · [Ruflo guide](https://github.com/ruvnet/ruflo#readme)

#### Runtime, orchestration, and governance libraries

| Library | Purpose | Guide or example |
| :--- | :--- | :--- |
| [`@ruvector/ruvllm`](./npm/packages/ruvllm/README.md) | Local language model runtime | [RuVLLM examples](./examples/ruvLLM/README.md) |
| [`@ruvector/tiny-dancer`](./npm/packages/tiny-dancer/README.md) | Neural routing and circuit breakers | [Router tutorial](./npm/packages/tiny-dancer/README.md) |
| [`@ruvector/wasm-unified`](./npm/packages/ruvector-wasm-unified/README.md) | Unified browser and WASM API | [WASM guide](./npm/packages/ruvector-wasm-unified/README.md) |
| [`@ruvector/rvf`](./npm/packages/rvf/README.md) | Vector artifact SDK | [RVF examples](./examples/rvf/README.md) |
| [`@ruvector/rvf-mcp-server`](./npm/packages/rvf-mcp-server/README.md) | Expose RVF through MCP | [MCP server setup](./npm/packages/rvf-mcp-server/README.md) |
| [`@ruvector/rvforge`](./npm/packages/rvforge/README.md) | Turn RVF artifacts into signed installers | [RVForge guide](./npm/packages/rvforge/README.md) |
| [`rvAgent`](./crates/rvAgent/README.md) | Agent runtime components | [Runtime guide](./crates/rvAgent/README.md) |
| [`rvm`](./crates/rvm/README.md) | Execution substrate | [RVM guide](./crates/rvm/README.md) |
| [`ruvector-proof-gate`](./crates/ruvector-proof-gate/README.md) | Evidence and promotion gates | [Proof gate guide](./crates/ruvector-proof-gate/README.md) |
| [`ruvector-bounded-rag`](./crates/ruvector-bounded-rag/README.md) | Retrieval with bounded execution | [Bounded RAG guide](./crates/ruvector-bounded-rag/README.md) |
| [`ruvector-cluster-rag`](./crates/ruvector-cluster-rag/README.md) | Cluster based retrieval components | [Cluster RAG guide](./crates/ruvector-cluster-rag/README.md) |
| [`ruvector-server`](./crates/ruvector-server/README.md) | Service deployment | [Server guide](./crates/ruvector-server/README.md) |

**Try it:** [Agent to agent swarm](./examples/a2a-swarm/README.md) · [REFRAG pipeline](./examples/refrag-pipeline/README.md) · [Google Cloud examples](./examples/google-cloud/README.md) · [Python Agentforce example](./examples/python-salesforce-agentforce/README.md).

#### Govern and distribute

| Capability | What it provides | Surface |
| --- | --- | --- |
| Namespace isolation | Separate collections and schemas | [`ruvector-collections`](./crates/ruvector-collections) |
| Capability gated retrieval | Per vector 64 bit read masks inside search | [`ruvector-capgated`](./crates/ruvector-capgated), [ADR 268](./docs/adr/ADR-268-capability-gated-ann.md) |
| Tamper evident lineage | Hash linked records and witness verification | [RVF](./crates/rvf) |
| Replication primitives | Vector clocks, local change propagation, and conflict strategies | [`ruvector-replication`](./crates/ruvector-replication) |
| Raft primitives | Election, log, and metadata state machine components | [`ruvector-raft`](./crates/ruvector-raft) |
| Shared collective memory | Remote contributions, search, provenance, and voting | [`mcp-brain`](./crates/mcp-brain) |

## Choose a memory path

| Requirement | Start with | Add when needed |
| --- | --- | --- |
| Local agent or coding memory | `npx ruvector hooks` | ONNX semantic mode, MCP |
| Embedded Node.js service | `ruvector` and `VectorDB` | Graph, SONA, snapshots |
| Embedded Rust service | `ruvector-core` | Specialized retrieval crates |
| Typed in-process agent memory | `ruvllm::context::AgenticMemory` | External persistence and consolidation policy |
| High write event stream | `ruvector-lsm-ann` | Snapshot and compaction policy |
| Multi-hop enterprise knowledge | `ruvector-graph` | Hybrid cue search and reconstruction harness |
| Recency sensitive memory | `ruvector-temporal-coherence` | Learned half-life after domain evaluation |
| Memory constrained edge node | `ruvector-pq-search` or `ruvector-rabitq` | Exact reranking for critical recalls |
| Existing lake or warehouse | `ruvector-rulake` | RVF witness bundles |
| PostgreSQL estate | `ruvector-postgres` | Build and operate with `pgrx` separately |
| Cross-agent shared memory | `mcp-brain` | Explicit hosted data policy and trust controls |

## Agent integration

For automated agent integration, install the package locally and commit your lockfile:

```bash
npm install ruvector
RUVECTOR_MCP_PROFILE=readonly npx ruvector mcp start
```

List the currently available tools instead of relying on a hardcoded count:

```bash
npx ruvector mcp tools
```

Use `RUVECTOR_MCP_ALLOW` and `RUVECTOR_MCP_DENY` for an explicit tool policy. No policy preserves the broader compatibility surface, so production deployments should set one deliberately.

If you enable editor or coding hooks, inspect the generated configuration, keep the package local and pinned, and run `npx ruvector hooks verify`. Do not depend on a fresh `@latest` download inside each hook invocation.

## Deployment surfaces

![Animated RuVector deployment diagram: local native and browser applications, with optional services across a separate data boundary](assets/ruvector/deployment-boundaries.svg)

| Surface | Package or crate | Data boundary |
| --- | --- | --- |
| Node.js and TypeScript | [`ruvector`](https://www.npmjs.com/package/ruvector) | Local process and local files |
| Rust | [`ruvector-core`](https://crates.io/crates/ruvector-core) | Local process and local files |
| Browser | [`@ruvector/wasm`](https://www.npmjs.com/package/@ruvector/wasm) | Browser memory and browser storage |
| HTTP service | [`ruvector-server`](./crates/ruvector-server) | Your service boundary |
| PostgreSQL | [`ruvector-postgres`](./crates/ruvector-postgres) | Your database boundary |
| RVF cognitive container | [`crates/rvf`](./crates/rvf) | Portable signed artifact |
| Shared Brain | [`mcp-brain`](./crates/mcp-brain) | Optional hosted service |

Native npm binaries cover glibc Linux on x64 and arm64, macOS on x64 and arm64, and Windows on x64. Browser and other environments use separate packages. The root package's fallback mode is limited when neither the native core nor RVF can load; use `@ruvector/wasm` explicitly for browser vector operations. Validate the selected backend with:

```bash
npx ruvector info
```

## Security and governance

1. Treat embeddings as sensitive derivatives of source data. Apply the same classification, residency, access, and retention policy as the original content.

2. Collections and metadata filters organize memory; they are not a complete authorization boundary. Enforce identity and authorization in the application. Capability gated ANN is currently a research component with a 64 capability mask and documented side channel and recall limitations.

3. RVF witnesses and hash linked logs are tamper evident. They do not encrypt memory content or prevent an authorized process from reading it.

4. A delete from the live store does not automatically remove copies in snapshots, branches, replicas, exports, or hosted memory. Define retention and erasure across every copy.

5. Shared Brain is a hosted plane. Review its network, identity, provenance, poisoning, and data residency controls before sending enterprise memory.

6. Pin and prepopulate embedding models for offline or regulated deployments. The default npm semantic path downloads its model on first use.

7. Keep tool execution separate from memory retrieval. Retrieved context is untrusted input until policy checks and action authorization pass.

See [SECURITY.md](./SECURITY.md) for reporting and project security guidance.

## Known boundaries

1. The repository is a monorepo. Installing `ruvector` does not activate every crate in this capability map.

2. The unified four type `ruvllm::AgenticMemory` manager does not yet have native save and load support, and its episodic to semantic or procedural consolidation method currently returns no changes. Durable `VectorDB` storage and typed runtime memory are not yet one facade.

3. Core metadata filtering currently narrows the retrieved candidate set. Highly selective filters may return fewer than `k` relevant results. Evaluate ACORN or an application level prefilter for selective workloads.

4. Opening a persisted HNSW database currently enumerates stored vectors and rebuilds the index. Measure cold start time against the intended memory size.

5. Temporal coherence currently builds an exact pairwise coherence graph and is a proof of concept for moderate memory sets. The planned production path is an approximate neighbor graph.

6. Agent memory compaction is not yet wired into the default core, MCP, or RVF persistence path.

7. Full snapshot serialization exists, but incremental snapshots, scheduling, cloud backends, and direct `VectorDB` restoration are not complete on the current main branch.

8. Replication exposes local primitives and simulated transport behavior. Raft still has incomplete response transport and snapshot installation paths. These are not a complete production network replication plane.

9. GNN reranking, MRAgent reconstruction, and Darwin optimization are implemented research surfaces, not automatic behavior in `VectorDB::search`.

10. RVF and PostgreSQL are separate build surfaces and are excluded from the default workspace build because they require their own toolchains.

11. Performance depends on vector dimension, index parameters, filter selectivity, recall target, hardware, and backend. Run the included benchmark for the component and workload you intend to deploy.

## Reproduce the evidence

RuVector keeps benchmark code beside the implementation. These commands exercise memory relevant components without relying on unscoped cross product comparisons.

```bash
# Core vector search
cargo bench -p ruvector-core

# High write LSM memory
cargo run --release -p ruvector-lsm-ann --bin benchmark

# Temporal and coherence weighted recall
cargo run --release -p ruvector-temporal-coherence --bin tcd-benchmark

# Capability gated retrieval
cargo run --release -p ruvector-capgated --bin benchmark

# Matryoshka coarse to fine retrieval
cargo run --release -p ruvector-matryoshka --bin benchmark
```

Record dataset size, dimension, index configuration, hardware, latency percentiles, throughput, and recall together. A latency number without its recall target is not a useful retrieval benchmark. See the [benchmarking guide](./docs/benchmarks/BENCHMARKING_GUIDE.md).

## Build from source

```bash
git clone https://github.com/ruvnet/RuVector.git
cd RuVector
cargo test --workspace
```

The workspace requires Rust 1.77 or newer. RVF and PostgreSQL have separate build instructions in their component documentation.

## Frequently asked questions

### Can RuVector run offline?

Yes, the local retrieval path can operate without a database server or API key. Prepopulate the embedding model and required packages before disconnecting: the default npm semantic command downloads its model on first use. [Deployment requirements](#deployment-surfaces).

### Does RuVector learn whenever an agent reads memory?

No. Reads retrieve context. Learning requires recorded outcomes or feedback and the relevant configured learning component. Research reranking and optimization surfaces do not run automatically in `VectorDB::search`. [Capability map](#capability-map).

### Is the root package the entire RuVector platform?

No. `ruvector` is one entry point in a monorepo. Typed decisions, browser WASM, PostgreSQL, RVF, and specialized research crates have separate installation or build paths. [Choose a component](#choose-a-memory-path).

### Does vector similarity prove that a recalled fact is true?

No. Similarity identifies nearby representations. Your application must assess provenance, freshness, access, and task relevance before acting on recalled context. [Security and governance](#security-and-governance).

## Documentation

| Topic | Link |
| --- | --- |
| Documentation index | [docs/INDEX.md](./docs/INDEX.md) |
| Node.js API | [docs/api/NODEJS_API.md](./docs/api/NODEJS_API.md) |
| Rust API | [docs/api/RUST_API.md](./docs/api/RUST_API.md) |
| Cypher reference | [docs/api/CYPHER_REFERENCE.md](./docs/api/CYPHER_REFERENCE.md) |
| Architecture decisions | [docs/adr](./docs/adr) |
| Benchmarks | [docs/benchmarks](./docs/benchmarks) |
| Repository structure | [docs/REPO_STRUCTURE.md](./docs/REPO_STRUCTURE.md) |

## Contributing

Contributions are welcome. Start with the [contribution guide](./docs/development/CONTRIBUTING.md). New capability claims should include an implementation link and reproducible evidence.

## Related

[`ruvnet/LatentMesh`](https://github.com/ruvnet/LatentMesh) — a research prototype for causally-verified latent agent communication; ADR-005 names RuVector as the store for its raw→compressed→prototype→symbolic latent-memory continuum (design-stage, not yet wired to a live RuVector instance).

## License

RuVector is available under the [MIT License](./LICENSE).

Built by [rUv](https://ruv.io) and powering [Cognitum](https://cognitum.one).
