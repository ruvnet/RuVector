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

// Default domain list: 60+ medical + CS domains
const DEFAULT_DOMAINS = [
  // Medical - Major Publishers & Journals
  'pubmed.ncbi.nlm.nih.gov', 'ncbi.nlm.nih.gov', 'who.int',
  'nature.com', 'nejm.org', 'bmj.com', 'thelancet.com',
  'jamanetwork.com', 'annals.org', 'sciencedirect.com',
  // Medical - Clinical Resources
  'mayoclinic.org', 'clevelandclinic.org', 'medlineplus.gov',
  'cdc.gov', 'nih.gov', 'webmd.com', 'healthline.com',
  'medscape.com', 'uptodate.com',
  // Medical - Oncology & Dermatology
  'cancer.org', 'aad.org', 'dermnetnz.org', 'melanoma.org',
  'asco.org', 'esmo.org', 'nccn.org', 'cancer.net',
  'mskcc.org', 'mdanderson.org', 'dana-farber.org',
  'dermcoll.edu.au', 'bad.org.uk', 'euroderm.org',
  'jaad.org', 'jidonline.org',
  // Medical - Publishers & Open Access
  'wiley.com', 'onlinelibrary.wiley.com', 'springer.com',
  'karger.com', 'thieme.com', 'mdpi.com', 'frontiersin.org',
  'plos.org', 'biomedcentral.com', 'cell.com', 'elsevier.com',
  // Medical - Regulatory & Evidence
  'clinicaltrials.gov', 'fda.gov', 'ema.europa.eu',
  'nice.org.uk', 'cochrane.org',
  'hopkinsmedicine.org', 'stanfordmedicine.org',
  // CS - Conferences & Journals
  'arxiv.org', 'acm.org', 'dl.acm.org', 'ieee.org',
  'ieeexplore.ieee.org', 'proceedings.neurips.cc',
  'aclanthology.org', 'jmlr.org', 'aaai.org', 'ijcai.org',
  'usenix.org', 'vldb.org', 'sigmod.org', 'icml.cc',
  'cvpr.thecvf.com', 'eccv.ecva.net', 'iccv.thecvf.com',
  'openreview.net', 'paperswithcode.com',
  // CS - Frameworks & Tools
  'huggingface.co', 'pytorch.org', 'tensorflow.org',
  'wandb.ai', 'mlflow.org', 'ray.io',
  'dmlc.cs.washington.edu',
  // CS - Research Labs & Universities
  'cs.stanford.edu', 'cs.berkeley.edu', 'cs.cmu.edu',
  'cs.mit.edu', 'deepmind.google', 'ai.meta.com',
  'research.google', 'microsoft.com/research',
  'blog.openai.com', 'anthropic.com',
];

function matchesDomain(url) {
  const allDomains = DOMAINS.length > 0 ? DOMAINS : DEFAULT_DOMAINS;
  return allDomains.some(d => url.includes(d));
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
  else if (url.includes('cancer.org') || url.includes('cancer.net') || url.includes('nccn.org')) tags.push('cancer', 'oncology');
  else if (url.includes('asco.org') || url.includes('esmo.org')) tags.push('oncology', 'clinical');
  else if (url.includes('mskcc.org') || url.includes('mdanderson.org') || url.includes('dana-farber.org')) tags.push('oncology', 'research');
  else if (url.includes('dermnetnz') || url.includes('aad.org') || url.includes('jaad.org')) tags.push('dermatology');
  else if (url.includes('dermcoll') || url.includes('bad.org.uk') || url.includes('euroderm')) tags.push('dermatology');
  else if (url.includes('jidonline')) tags.push('dermatology', 'research');
  else if (url.includes('melanoma')) tags.push('melanoma', 'skin-cancer');
  else if (url.includes('clinicaltrials.gov')) tags.push('clinical-trials', 'medical');
  else if (url.includes('fda.gov') || url.includes('ema.europa.eu')) tags.push('regulatory', 'medical');
  else if (url.includes('nice.org.uk') || url.includes('cochrane.org')) tags.push('evidence-based', 'medical');
  else if (url.includes('hopkinsmedicine') || url.includes('stanfordmedicine')) tags.push('medical', 'academic');
  else if (url.includes('webmd') || url.includes('healthline') || url.includes('medscape')) tags.push('medical', 'clinical');
  else if (url.includes('uptodate.com')) tags.push('medical', 'clinical-decision');
  else if (url.includes('acm.org') || url.includes('ieee') || url.includes('dl.acm.org')) tags.push('computer-science');
  else if (url.includes('neurips') || url.includes('icml') || url.includes('aaai.org')) tags.push('ml', 'conference');
  else if (url.includes('cvpr') || url.includes('eccv') || url.includes('iccv')) tags.push('computer-vision', 'conference');
  else if (url.includes('aclanthology')) tags.push('nlp', 'conference');
  else if (url.includes('usenix') || url.includes('vldb') || url.includes('sigmod')) tags.push('systems', 'conference');
  else if (url.includes('huggingface') || url.includes('pytorch') || url.includes('tensorflow')) tags.push('ml', 'framework');
  else if (url.includes('deepmind') || url.includes('ai.meta') || url.includes('research.google')) tags.push('ml', 'research-lab');
  else if (url.includes('openai') || url.includes('anthropic')) tags.push('ml', 'research-lab');
  else if (url.includes('cs.stanford') || url.includes('cs.berkeley') || url.includes('cs.cmu') || url.includes('cs.mit')) tags.push('computer-science', 'academic');
  else if (url.includes('openreview') || url.includes('paperswithcode')) tags.push('ml', 'research');
  else if (url.includes('github') || url.includes('stackoverflow')) tags.push('programming');
  else if (url.includes('nature.com') || url.includes('nejm') || url.includes('lancet')) tags.push('journal', 'research');
  else if (url.includes('jamanetwork') || url.includes('annals.org') || url.includes('bmj.com')) tags.push('journal', 'medical');
  else if (url.includes('frontiersin') || url.includes('plos.org') || url.includes('biomedcentral')) tags.push('open-access', 'research');
  else if (url.includes('cell.com') || url.includes('elsevier') || url.includes('springer') || url.includes('wiley')) tags.push('journal', 'publisher');
  else if (url.includes('mdpi.com') || url.includes('karger') || url.includes('thieme')) tags.push('journal', 'publisher');
  else if (url.includes('jmlr.org') || url.includes('ijcai.org')) tags.push('ml', 'journal');

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
