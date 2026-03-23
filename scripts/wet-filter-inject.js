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
  // Physics - High Energy & Particle
  'cern.ch', 'home.cern', 'inspirehep.net',
  'hep.ph', 'hep.th', 'physics.aps.org',
  'journals.aps.org', 'physicstoday.org',
  // Physics - Astronomy & Cosmology
  'adsabs.harvard.edu', 'nasa.gov', 'esa.int',
  'noirlab.edu', 'stsci.edu', 'caltech.edu',
  'ligo.org', 'jwst.nasa.gov',
  // Physics - Condensed Matter & Materials
  'materialsproject.org', 'nist.gov',
  'iop.org', 'iopscience.iop.org',
  // Physics - Quantum
  'quantum-journal.org', 'quantum.country',
  'qiskit.org', 'pennylane.ai',
  // Physics - General & Interdisciplinary
  'physicsworld.com', 'quantamagazine.org',
  'simonsfoundation.org', 'perimeterinstitute.ca',
  'kitp.ucsb.edu', 'slac.stanford.edu',
  'fermilab.gov', 'bnl.gov', 'ornl.gov',
  // GitHub & Code Intelligence
  'github.com', 'github.blog', 'docs.github.com',
  // Stack Overflow / Stack Exchange
  'stackoverflow.com', 'stackexchange.com',
  'stats.stackexchange.com', 'math.stackexchange.com',
  'physics.stackexchange.com', 'biology.stackexchange.com',
  'cs.stackexchange.com', 'datascience.stackexchange.com',
  // Patents & IP
  'patents.google.com', 'patft.uspto.gov',
  'worldwide.espacenet.com',
  // Preprint Servers (beyond arXiv)
  'biorxiv.org', 'medrxiv.org', 'chemrxiv.org',
  'ssrn.com', 'preprints.org', 'researchsquare.com',
  // Wikipedia & Reference
  'en.wikipedia.org', 'wikidata.org', 'wikimedia.org',
  // Regulatory & Government
  'regulations.gov', 'sec.gov', 'epa.gov',
  'energy.gov', 'nsf.gov', 'darpa.mil',
  // News & Analysis (science/tech)
  'techcrunch.com', 'arstechnica.com', 'wired.com',
  'technologyreview.com', 'newscientist.com',
  'sciencemag.org', 'scientificamerican.com',
  // Additional Academic
  'jstor.org', 'tandfonline.com', 'sagepub.com',
  'degruyter.com', 'oxfordjournals.org',
  'cambridge.org', 'royalsocietypublishing.org',
  // Data & Statistics
  'data.gov', 'kaggle.com', 'dataverse.harvard.edu',
  'zenodo.org', 'figshare.com', 'datadryad.org',
  // Additional Tech
  'medium.com', 'towardsdatascience.com', 'distill.pub',
  'lilianweng.github.io', 'colah.github.io',
  'karpathy.github.io', 'jalammar.github.io',
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
  // Physics
  else if (url.includes('cern.ch') || url.includes('home.cern')) tags.push('physics', 'cern', 'particle');
  else if (url.includes('inspirehep')) tags.push('physics', 'hep');
  else if (url.includes('physics.aps.org') || url.includes('journals.aps.org')) tags.push('physics', 'journal');
  else if (url.includes('adsabs') || url.includes('nasa.gov') || url.includes('stsci.edu')) tags.push('physics', 'astronomy');
  else if (url.includes('esa.int') || url.includes('jwst') || url.includes('ligo.org')) tags.push('physics', 'space');
  else if (url.includes('materialsproject') || url.includes('nist.gov')) tags.push('physics', 'materials');
  else if (url.includes('iop.org') || url.includes('iopscience')) tags.push('physics', 'journal');
  else if (url.includes('quantum-journal') || url.includes('qiskit') || url.includes('pennylane')) tags.push('physics', 'quantum');
  else if (url.includes('quantamagazine') || url.includes('physicsworld')) tags.push('physics', 'popular');
  else if (url.includes('fermilab') || url.includes('slac.stanford') || url.includes('bnl.gov')) tags.push('physics', 'national-lab');
  else if (url.includes('perimeterinstitute') || url.includes('kitp.ucsb') || url.includes('simonsfoundation')) tags.push('physics', 'institute');
  // GitHub & Code
  else if (url.includes('github.com')) tags.push('code', 'github');
  // Stack Exchange
  else if (url.includes('stackoverflow') || url.includes('stackexchange')) tags.push('qa', 'community');
  // Patents
  else if (url.includes('patents') || url.includes('patft.uspto') || url.includes('espacenet')) tags.push('patents', 'ip');
  // Preprints
  else if (url.includes('biorxiv')) tags.push('preprint', 'biology');
  else if (url.includes('medrxiv')) tags.push('preprint', 'medical');
  else if (url.includes('chemrxiv')) tags.push('preprint', 'chemistry');
  else if (url.includes('ssrn.com')) tags.push('preprint', 'social-science');
  // Wikipedia
  else if (url.includes('wikipedia.org')) tags.push('wikipedia', 'reference');
  // Government & Regulatory
  else if (url.includes('nsf.gov') || url.includes('darpa.mil')) tags.push('government', 'funding');
  else if (url.includes('energy.gov') || url.includes('epa.gov')) tags.push('government', 'policy');
  else if (url.includes('sec.gov')) tags.push('regulatory', 'finance');
  // Science News
  else if (url.includes('sciencemag') || url.includes('scientificamerican') || url.includes('newscientist')) tags.push('science-news');
  else if (url.includes('technologyreview') || url.includes('wired') || url.includes('arstechnica')) tags.push('tech-news');
  // Data Repositories
  else if (url.includes('kaggle') || url.includes('zenodo') || url.includes('figshare') || url.includes('datadryad')) tags.push('data', 'dataset');
  else if (url.includes('dataverse') || url.includes('data.gov')) tags.push('data', 'open-data');
  // Academic Publishers
  else if (url.includes('jstor') || url.includes('tandfonline') || url.includes('sagepub') || url.includes('cambridge.org')) tags.push('journal', 'academic');
  // Tech Blogs
  else if (url.includes('medium.com') || url.includes('towardsdatascience')) tags.push('blog', 'tech');
  else if (url.includes('lilianweng') || url.includes('colah') || url.includes('karpathy') || url.includes('jalammar')) tags.push('blog', 'ml-explainer');

  const lower = content.toLowerCase();
  if (lower.includes('melanoma')) tags.push('melanoma');
  if (lower.includes('machine learning') || lower.includes('deep learning')) tags.push('ml');
  if (lower.includes('cancer')) tags.push('cancer');
  // Physics keywords
  if (lower.includes('dark matter') || lower.includes('dark energy')) tags.push('cosmology');
  if (lower.includes('quantum') && !tags.includes('quantum')) tags.push('quantum');
  if (lower.includes('higgs') || lower.includes('boson')) tags.push('particle-physics');
  if (lower.includes('gravitational wave') || lower.includes('ligo')) tags.push('gravitational-waves');
  if (lower.includes('black hole')) tags.push('black-holes');
  if (lower.includes('superconductor') || lower.includes('condensed matter')) tags.push('condensed-matter');
  if (lower.includes('fusion') && lower.includes('energy')) tags.push('fusion-energy');
  if (lower.includes('neutrino')) tags.push('neutrino');
  if (lower.includes('string theory') || lower.includes('supersymmetry')) tags.push('theoretical');

  return [...new Set(tags)].slice(0, 10);
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

async function flushBatch() {
  if (batch.length === 0) return;

  // Rate limit: 1 batch per second to avoid saturating the brain
  await sleep(1000);

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
// Process records inline to avoid OOM — never buffer all records
let processQueue = Promise.resolve();

rl.on('line', (line) => {
  if (line.startsWith('WARC/1.0')) {
    if (recordUrl && recordContent) {
      // Process immediately, don't accumulate
      const url = recordUrl;
      const content = recordContent;
      processQueue = processQueue.then(() => processRecord(url, content));
    }
    recordUrl = '';
    recordContent = '';
    inRecord = false;
  } else if (line.startsWith('WARC-Target-URI:')) {
    recordUrl = line.replace('WARC-Target-URI:', '').trim();
  } else if (line.startsWith('Content-Length:')) {
    inRecord = true;
  } else if (inRecord) {
    // Limit content accumulation per record to prevent single-record bloat
    if (recordContent.length < MAX_CONTENT_LENGTH * 2) {
      recordContent += line + '\n';
    }
  }
});

rl.on('close', async () => {
  // Process last record
  if (recordUrl && recordContent) {
    await processQueue;
    await processRecord(recordUrl, recordContent);
  } else {
    await processQueue;
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
