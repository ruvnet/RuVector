# Claude Code v2.1.91 (2.1 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 2.1.91 |
| Series | 2.1 |
| Bundle size | 12.6MB |
| RVF size | 1056.8KB |
| Vectors | 2068 |
| RVF File ID | `6d9f5dde25a78a28ea8ba6def10250b8` |
| Classes | 1632 |
| Functions | 19906 |
| Modules | 9 |
| Extracted | 2026-04-02T23:29:04+00:00 |

## Files

- `claude-code-v2.1.rvf` - Binary RVF container with HNSW index + witness chain
- `claude-code-v2.1.rvf.manifest.json` - Container manifest (vector ID map, metadata)
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

const db = await RvfDatabase.openReadonly('./claude-code-v2.1.rvf');
const results = await db.query(queryVector, 10);
await db.close();
```
