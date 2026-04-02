# Claude Code v0.2.126 (0.2 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 0.2.126 |
| Series | 0.2 |
| Bundle size | 6.9MB |
| RVF size | 159.0KB |
| Vectors | 300 |
| RVF File ID | `679f4bde2ea78a28eaddf8bc08d25018` |
| Classes | 1049 |
| Functions | 13869 |
| Modules | 9 |
| Extracted | 2026-04-02T23:28:42+00:00 |

## Files

- `claude-code-v0.2.rvf` - Binary RVF container with HNSW index + witness chain
- `claude-code-v0.2.rvf.manifest.json` - Container manifest (vector ID map, metadata)
- `source/` - Extracted JavaScript module fragments

## RVF Container Details

The `.rvf` file is a real binary container created with the `@ruvector/rvf-node`
native backend. It contains:

- **128-dimensional fingerprint vectors** for each code fragment
- **HNSW index** (M=16, ef_construction=200) for fast similarity search
- **Cosine distance** metric
- **Witness chain** for provenance verification

To query this container:

```typescript
import { RvfDatabase } from '@ruvector/rvf';

const db = await RvfDatabase.openReadonly('./claude-code-v0.2.rvf');
const results = await db.query(queryVector, 10);
await db.close();
```
