# Claude Code v2.0.77 (2.0 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 2.0.77 |
| Series | 2.0 |
| Bundle size | 10.5MB |
| RVF size | 405.3KB |
| Vectors | 785 |
| RVF File ID | `6c9f5dde1ca78a28ea90df777cfa50b8` |
| Classes | 1612 |
| Functions | 20395 |
| Modules | 9 |
| Extracted | 2026-04-02T23:29:01+00:00 |

## Files

- `claude-code-v2.0.rvf` - Binary RVF container with HNSW index + witness chain
- `claude-code-v2.0.rvf.manifest.json` - Container manifest (vector ID map, metadata)
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

const db = await RvfDatabase.openReadonly('./claude-code-v2.0.rvf');
const results = await db.query(queryVector, 10);
await db.close();
```
