# Report nearest-rank p99 latency for small Python benchmark runs

Status: proposed for review

## Problem and evidence

The CLI floors 0.99 times the sample count before selecting a latency. With two queries it reports the minimum as p99 while reporting the maximum as p50; counts3 and10 likewise understate the tail. Three controlled-clock actual CLI/native benchmark cases fail on upstream07933730, while the100-query control passes.

## Decision

Use the nearest-rank definition: ceil(0.99*n), converted to a zero-based index. Keep the measured sample values, existing p50 convention, recall evaluation and QPS calculation unchanged. This adds constant work and no vector-sized allocation.

## Validation and limits

Four tests invoke the real benchmark command and actual native index with a deterministic clock to isolate reporting behavior. These fixtures are not measured performance results. Full Python suite validation and independent exact-commit review are required. Nearest-rank percentiles on small samples remain coarse, and this does not establish a statistically stable production tail-latency estimate.

## Changelog

Fix understated p99 latency for query counts where 0.99*n is fractional.
