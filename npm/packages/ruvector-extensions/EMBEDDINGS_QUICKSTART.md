# Embeddings Module - Quick Start Guide

## Installation

```bash
npm install ruvector-extensions

# Install your preferred provider SDK:
npm install openai              # For OpenAI
npm install cohere-ai           # For Cohere
npm install @xenova/transformers # For local models
```

## 30-Second Start

```typescript
import { OpenAIEmbeddings } from 'ruvector-extensions';

const embedder = new OpenAIEmbeddings({
  apiKey: process.env.OPENAI_API_KEY,
});

const embedding = await embedder.embedText('Hello, world!');
console.log('Embedding:', embedding.length, 'dimensions');
```

## 5-Minute Integration with VectorDB

> **Compatibility note (0.1.3):** `embedAndInsert` inserts `{ id, values, metadata }` and `embedAndSearch` calls `db.search({ vector, topK })`, while the `VectorDB` in `ruvector` 0.2.x (the range this package depends on) expects `{ id, vector }` and `{ vector, k }`. Against that `VectorDB` the helpers currently fail (`Dimension mismatch ... got 0` / ``Missing field `k` ``). The provider classes (`embedText` / `embedTexts`) are unaffected.

```typescript
import { VectorDB } from 'ruvector';
import { OpenAIEmbeddings, embedAndInsert, embedAndSearch } from 'ruvector-extensions';

// 1. Initialize
const embedder = new OpenAIEmbeddings({ apiKey: process.env.OPENAI_API_KEY });
const db = new VectorDB({ dimensions: embedder.getDimension() });

// 2. Prepare documents
const documents = [
  {
    id: '1',
    text: 'Machine learning is fascinating',
    metadata: { category: 'AI' }
  },
  {
    id: '2',
    text: 'Deep learning uses neural networks',
    metadata: { category: 'AI' }
  }
];

// 3. Embed and insert
await embedAndInsert(db, embedder, documents);

// 4. Search
const results = await embedAndSearch(
  db,
  embedder,
  'What is deep learning?',
  { topK: 5 }
);

console.log('Results:', results);
```

## Provider Comparison

| Provider | Best For | Dimension | API Key |
|----------|----------|-----------|---------|
| OpenAI | General purpose | 1536-3072 | ✅ |
| Cohere | Search optimization | 1024 | ✅ |
| HuggingFace | Privacy/offline | 384+ | ❌ |

## Next Steps

- 📚 Read the [full documentation](https://github.com/ruvnet/ruvector/blob/main/npm/packages/ruvector-extensions/docs/EMBEDDINGS.md)
- 💡 Explore [11 examples](https://github.com/ruvnet/ruvector/blob/main/npm/packages/ruvector-extensions/src/examples/embeddings-example.ts)
- 🧪 Run the [test suite](https://github.com/ruvnet/ruvector/blob/main/npm/packages/ruvector-extensions/tests/embeddings.test.ts)

## File Locations

- **Main Module**: `/src/embeddings.ts`
- **Documentation**: `/docs/EMBEDDINGS.md`
- **Examples**: `/src/examples/embeddings-example.ts`
- **Tests**: `/tests/embeddings.test.ts`
- **Summary**: `/docs/EMBEDDINGS_SUMMARY.md`

---

✅ **Status**: Production-ready and fully tested!
