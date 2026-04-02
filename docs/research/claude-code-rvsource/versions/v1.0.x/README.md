# Claude Code v1.0.128 (1.0 series)

## Binary RVF Container

| Property | Value |
|----------|-------|
| Version | 1.0.128 |
| Series | 1.0 |
| Bundle size | 8.9MB |
| RVF size | 251.4KB |
| Vectors | 482 |
| RVF File ID | `689f54de1ca78a28ea7d0a38ceda5068` |
| Classes | 1390 |
| Functions | 16593 |
| Modules | 9 |
| Extracted | 2026-04-02T23:28:45+00:00 |

## Files

- `claude-code-v1.0.rvf` - Binary RVF container with HNSW index + witness chain
- `claude-code-v1.0.rvf.manifest.json` - Container manifest (vector ID map, metadata)
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

const db = await RvfDatabase.openReadonly('./claude-code-v1.0.rvf');
const results = await db.query(queryVector, 10);
await db.close();
```
