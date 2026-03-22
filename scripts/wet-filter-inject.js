#!/usr/bin/env node
// WET Filter + Inject -- reads WARC WET from stdin, filters by domain, injects to brain
// Usage: gunzip -c segment.wet.gz | node wet-filter-inject.js --brain-url URL --domains dom1,dom2
'use strict';

const args = process.argv.slice(2);
function getArg(name, def) {
  const idx = args.indexOf(`--${name}`);
  return idx >= 0 && args[idx + 1] ? args[idx + 1] : def;
}

const BRAIN_URL = getArg('brain-url', 'https://pi.ruv.io');
const AUTH = getArg('auth', 'Authorization: Bearer ruvector-crawl-2026');
const BATCH_SIZE = parseInt(getArg('batch-size', '10'), 10);
const DOMAINS = getArg('domains', '').split(',').filter(Boolean);
const CRAWL_INDEX = getArg('crawl-index', 'CC-MAIN-2026-08');
const MIN_CONTENT_LENGTH = 300;
const MAX_CONTENT_LENGTH = 8000;

const stats = { total: 0, filtered: 0, injected: 0, errors: 0, batched: 0 };
let batch = [];

function matchesDomain(url) {
  return DOMAINS.some(d => url.includes(d));
}

function extractTitle(content) {
  const lines = content.trim().split('\n').filter(l => l.trim().length > 10);
  if (lines.length === 0) return '';
  let title = lines[0].trim();
  if (title.length > 150) title = title.slice(0, 147) + '...';
  return title;
}

function generateTags(url, content) {
  const tags = ['common-crawl', `crawl-${CRAWL_INDEX}`];

  if (url.includes('pubmed') || url.includes('ncbi')) tags.push('pubmed', 'medical');
  else if (url.includes('arxiv')) tags.push('arxiv', 'research');
  else if (url.includes('who.int')) tags.push('who', 'global-health');
  else if (url.includes('cancer.org')) tags.push('cancer', 'oncology');
  else if (url.includes('dermnetnz') || url.includes('aad.org')) tags.push('dermatology');
  else if (url.includes('melanoma')) tags.push('melanoma', 'skin-cancer');
  else if (url.includes('acm.org') || url.includes('ieee')) tags.push('computer-science');
  else if (url.includes('github') || url.includes('stackoverflow')) tags.push('programming');
  else if (url.includes('nature.com') || url.includes('nejm') || url.includes('lancet')) tags.push('journal', 'research');

  const lower = content.toLowerCase();
  if (lower.includes('melanoma')) tags.push('melanoma');
  if (lower.includes('machine learning') || lower.includes('deep learning')) tags.push('ml');
  if (lower.includes('cancer')) tags.push('cancer');

  return [...new Set(tags)].slice(0, 10);
}

async function flushBatch() {
  if (batch.length === 0) return;

  const items = batch.splice(0);
  try {
    const res = await fetch(`${BRAIN_URL}/v1/pipeline/inject/batch`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        [AUTH.split(': ')[0]]: AUTH.split(': ').slice(1).join(': '),
      },
      body: JSON.stringify({ source: 'common-crawl-wet', items }),
      signal: AbortSignal.timeout(30000),
    });

    if (res.ok) {
      const data = await res.json();
      stats.injected += data.accepted || 0;
      stats.errors += data.rejected || 0;
      stats.batched++;
      process.stderr.write(`  Batch ${stats.batched}: ${data.accepted}/${items.length} accepted\n`);
    } else {
      stats.errors += items.length;
      process.stderr.write(`  Batch failed: ${res.status}\n`);
    }
  } catch (err) {
    stats.errors += items.length;
    process.stderr.write(`  Batch error: ${err.message}\n`);
  }
}

async function processRecord(url, content) {
  stats.total++;

  if (!matchesDomain(url)) return;

  content = content.trim();
  if (content.length < MIN_CONTENT_LENGTH) return;
  if (content.length > MAX_CONTENT_LENGTH) content = content.slice(0, MAX_CONTENT_LENGTH);

  stats.filtered++;

  const title = extractTitle(content);
  if (!title) return;

  batch.push({
    source: 'common-crawl-wet',
    title,
    content,
    tags: generateTags(url, content),
    category: (url.includes('arxiv') || url.includes('acm') || url.includes('ieee'))
      ? 'architecture'
      : 'solution',
  });

  if (batch.length >= BATCH_SIZE) {
    await flushBatch();
  }
}

// Parse WARC WET format from stdin
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });

let recordUrl = '';
let recordContent = '';
let inRecord = false;
const pendingRecords = [];

rl.on('line', (line) => {
  if (line.startsWith('WARC/1.0')) {
    if (recordUrl && recordContent) {
      pendingRecords.push({ url: recordUrl, content: recordContent });
    }
    recordUrl = '';
    recordContent = '';
    inRecord = false;
  } else if (line.startsWith('WARC-Target-URI:')) {
    recordUrl = line.replace('WARC-Target-URI:', '').trim();
  } else if (line.startsWith('Content-Length:')) {
    inRecord = true;
  } else if (inRecord) {
    recordContent += line + '\n';
  }
});

rl.on('close', async () => {
  // Process last record
  if (recordUrl && recordContent) {
    pendingRecords.push({ url: recordUrl, content: recordContent });
  }

  // Process all records sequentially
  for (const rec of pendingRecords) {
    await processRecord(rec.url, rec.content);
  }

  // Flush remaining batch
  await flushBatch();

  console.log(JSON.stringify({
    total_records: stats.total,
    domain_matches: stats.filtered,
    injected: stats.injected,
    errors: stats.errors,
    batches_sent: stats.batched,
    crawl_index: CRAWL_INDEX,
  }, null, 2));
});
