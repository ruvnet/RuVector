# Architecture Decision Records — Index Notes

## Duplicate ADR numbers (closeout 2026-07-03)

Three ADR numbers were each assigned to two distinct decisions. To keep existing
cross-references working, the files are **not** renumbered; instead each pair is
disambiguated with an `a`/`b` suffix (ordered by decision date). Cite the suffix
form when the bare number is ambiguous.

| Suffix | File | Decision |
|--------|------|----------|
| ADR-017a | `ADR-017-craftsman-ultra-30b-1bit-bitnet-integration.md` | Craftsman Ultra 30b 1bit — BitNet integration with RuvLLM (2026-02-03) |
| ADR-017b | `ADR-017-temporal-tensor-compression.md` | Temporal Tensor Compression with tiered quantization (2026-02-06) |
| ADR-029a | `ADR-029-rvf-canonical-format.md` | RVF as canonical binary format across all RuVector libraries (2026-02-13) |
| ADR-029b | `ADR-029-exo-ai-multiparadigm-integration.md` | EXO-AI multi-paradigm integration architecture (2026-02-27) |
| ADR-031a | `ADR-031-rvf-example-repository.md` | RVF example repository — 24 demonstrations (2026-02-14) |
| ADR-031b | `ADR-031-rvcow-branching-and-real-cognitive-containers.md` | Vector-native COW branching (RVCOW) and real cognitive containers |

## Status vocabulary

`Proposed` → decided but not built; `Accepted` → decided and being built or
built; `Implemented` → verified present in code at closeout; `Partially
Implemented` → some phases shipped, others outstanding; `Stale` → tracker no
longer reflects the codebase. Status fields updated during the 2026-07-03
closeout carry a dated note.
