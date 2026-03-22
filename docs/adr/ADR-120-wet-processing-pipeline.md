# ADR-120: WET Processing Pipeline for Medical + CS Corpus Import

**Status:** Accepted
**Date:** 2026-03-22
**Author:** ruvector team

## Context

The CDX HTML extractor is broken -- it returns empty titles from Wayback Machine content due to inconsistent HTML structure across archived pages. Fixing the extractor would require handling thousands of edge cases across decades of web standards.

Common Crawl provides WET (Web Extracted Text) files that contain pre-extracted plain text. These files bypass all HTML parsing entirely.

## Decision

Process Common Crawl WET files instead of fixing the CDX HTML extractor for the medical + CS corpus import pipeline.

## Architecture

```
Download WET segment (~150MB gz)
  -> gunzip (streaming)
  -> Filter by 30 medical + CS domains
  -> Chunk content (300-8000 chars)
  -> Tag by domain + content keywords
  -> Batch inject into pi.ruv.io brain (10 items/batch)
```

### Components

| Script | Purpose |
|--------|---------|
| `scripts/wet-processor.sh` | Downloads and processes a single WET segment |
| `scripts/wet-filter-inject.js` | Parses WARC WET format, filters by domain, injects to brain |
| `scripts/wet-orchestrate.sh` | Orchestrates multi-segment processing |
| `scripts/wet-job.yaml` | Cloud Run Job config for parallel processing |

### Target Domains (30)

**Medical:** pubmed, ncbi, who.int, cancer.org, aad.org, skincancer.org, dermnetnz.org, melanoma.org, mayoclinic.org, clevelandclinic.org, medlineplus.gov, cdc.gov, nih.gov, nejm.org, thelancet.com, bmj.com

**CS/Research:** nature.com, sciencedirect.com, arxiv.org, acm.org, ieee.org, dl.acm.org, proceedings.mlr.press, openreview.net, paperswithcode.com, github.com, stackoverflow.com, medium.com, towardsdatascience.com, distill.pub

## Rationale

- WET files contain pre-extracted text -- no HTML parsing needed
- 100x faster than CDX+HTML extraction pipeline
- Same S3 cost model (public bucket, no auth)
- Each WET segment is ~150MB compressed, ~100K pages
- Streaming pipeline keeps memory usage under 1GB

## Cost Estimate

- ~90,000 WET segments across crawls 2020-2026
- Filter reduces to ~0.1% relevant pages (medical + CS domains)
- Estimated ~$200 total in compute (Cloud Run) for full corpus
- 6 weeks at 5 segments/day for complete import

## Consequences

- Bypasses HTML parsing entirely (positive)
- Text quality depends on Common Crawl's extraction (acceptable)
- No images or structured HTML elements (acceptable for text corpus)
- Requires streaming to handle 150MB+ files without memory issues (handled)
