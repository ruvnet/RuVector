# Retrieval benchmark

Measured 2026-10-11 on Linux x86_64, Python 3.12.14, LionAGI 0.35.3, RuVector 0.1.2, NumPy 2.5.4. Raw JSON includes environment and dataset SHA256.

10,000 normalized Gaussian vectors, 64 dimensions, 200 independent queries, k=10, seed 20261011. Ten queries warm the index before timing. Exact NumPy cosine ranking establishes recall ground truth. BLAS uses one thread. Each configuration runs once in its own process; these are exploratory observations, not confidence intervals or workload guarantees.

| ef_search | Recall@10 | Adapter p50 ms | Adapter p95 ms | Exact NumPy p50 ms |
| --- | --- | --- | --- | --- |
| 50 | 84.45% | 0.632 | 1.086 | 0.161 |
| 100 | 95.70% | 1.052 | 1.647 | 0.165 |
| 200 | 99.60% | 1.702 | 2.535 | 0.172 |
| 1000 | 100.00% | 3.679 | 6.047 | 0.163 |

The default ef_search=100 balances retrieval work and recall. Setting 200 increased measured recall to 99.6% at higher latency. The adapter honors ef_search directly instead of inflating it to max_k=1000. These measurements show the resulting cost/recall tradeoff; they do not isolate every source of variance. Filtered searches use exact pre-ranking filtering and bounded top-k selection, avoiding approximate post-filter recall loss. Deletion rebuilds the index, which reclaims tombstones but costs O(n) work and transient additional memory.

**NumPy exact search was faster on this dataset.** No claim of a RuVector speed advantage, embedding relevance, large-scale throughput, or production tail latency follows from this experiment. HNSW construction has randomness beyond the dataset seed. Concurrent load, filtered retrieval, deletion, network/LLM/embedding costs, Windows and macOS are not benchmarked here. Raw high-water RSS includes Python/native allocations and is not an exclusive index memory measurement.

Reproduce on Linux or macOS after installing the package:

```bash
python benchmarks/retrieval.py --n 10000 --dimensions 64 --queries 200 --ef 100 --output result.json
```

The benchmark uses the standard-library resource module, which is unavailable on Windows. CI runs a small Linux smoke benchmark, without unstable wall-clock pass thresholds. Application owners should benchmark representative embeddings and query distributions before choosing ef_search.
