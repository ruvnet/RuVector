# RuVector Node.js tutorial

![Animated tutorial: install, write, reopen, verify](../../assets/ruvector/tutorial-steps.svg)

[All examples](../README.md) · [npm package](../../npm/packages/ruvector/README.md) · [API reference](../../docs/api/NODEJS_API.md)

Build semantic memory and retrieve it from a second process. Use a supported Node.js native backend; run `npx ruvector info` before continuing. The first embedding call downloads a local model.

## Install

Run in a new application directory:

```bash
npm init -y
npm install ruvector
npx ruvector info
```

## Write and reopen

Save this as `memory.cjs`. Both commands below must run from the same directory so they use the same database path.

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

  if (process.argv[2] === 'write') {
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

  }

  const query = await embedder.embedQuery(
    'Where may the customer data be processed?',
  );

  const results = await db.search({
    vector: query,
    k: 3,
    filter: { tenant: 'acme' },
  });

  if (!results.length) throw new Error('No stored records found. Run the write step first.');
  console.log(results.map(({ score, metadata }) => ({ score, ...metadata })));
}

main().catch(error => { console.error(error); process.exitCode = 1; });
```

```bash
node memory.cjs write
node memory.cjs read
```

**Expected result:** both processes return stored records with text, metadata, and distances. The second invocation skips insertion and proves that records survive the first process. Lower search distances are closer. Keep the embedding model and dimension consistent.

## Next examples

| Goal | Continue with |
| :--- | :--- |
| Give Claude Code access to project memory | [Claude Code MCP setup](../../README.md#claude-code-setup) |
| Return typed decisions | [typesafe tutorial](../../npm/packages/typesafe/README.md) |
| Follow graph relationships | [Graph examples](../graph/README.md) |
| Run search in a browser | [Vanilla WASM](../wasm-vanilla/README.md), [React WASM](../wasm-react/README.md) |
| Learn from explicit outcomes | [SONA guide](../../crates/sona/README.md) |

If the native backend cannot load, resolve the platform or installation problem before trusting an empty search result. The root package's limited fallback does not provide a working persistent vector store.
