#!/usr/bin/env node
// rvAgent Gemini Grounding Agents (ADR-122)
// Implements 4 phases that use Gemini with Google Search grounding
// to verify, relate, explore, and research brain knowledge.
//
// Usage:
//   node scripts/gemini-agents.js --phase fact-verify
//   node scripts/gemini-agents.js --phase relate
//   node scripts/gemini-agents.js --phase cross-domain
//   node scripts/gemini-agents.js --phase research
//   node scripts/gemini-agents.js --phase all

'use strict';

// ---------------------------------------------------------------------------
// Configuration (from env vars)
// ---------------------------------------------------------------------------
const BRAIN_URL = process.env.BRAIN_URL || 'https://pi.ruv.io';
const BRAIN_AUTH = process.env.BRAIN_AUTH || 'Bearer ruvector-crawl-2026';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_MODEL = process.env.GEMINI_MODEL || 'gemini-2.5-flash';
const MAX_MEMORIES = parseInt(process.env.MAX_MEMORIES || '20', 10);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Call Gemini with Google Search grounding enabled. */
async function callGemini(prompt) {
  const url =
    `https://generativelanguage.googleapis.com/v1beta/models/${GEMINI_MODEL}:generateContent?key=${GEMINI_API_KEY}`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      contents: [{ role: 'user', parts: [{ text: prompt }] }],
      tools: [{ google_search: {} }],
      generationConfig: { maxOutputTokens: 1024, temperature: 0.3 },
    }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`Gemini ${res.status}: ${body.slice(0, 300)}`);
  }

  const json = await res.json();
  const text = json?.candidates?.[0]?.content?.parts?.[0]?.text || '';
  const grounding = json?.candidates?.[0]?.groundingMetadata || null;
  return { text, grounding };
}

/** Search brain memories. */
async function brainSearch(query, limit = 5) {
  const res = await fetch(
    `${BRAIN_URL}/v1/memories/search?q=${encodeURIComponent(query)}&limit=${limit}`,
    { headers: { Authorization: BRAIN_AUTH } },
  );
  if (!res.ok) {
    console.warn(`  brainSearch error ${res.status}`);
    return [];
  }
  const data = await res.json();
  return Array.isArray(data) ? data : data.memories || [];
}

/** List brain memories by category. */
async function brainList(category, limit = 5) {
  const res = await fetch(
    `${BRAIN_URL}/v1/memories/list?category=${encodeURIComponent(category)}&limit=${limit}`,
    { headers: { Authorization: BRAIN_AUTH } },
  );
  if (!res.ok) {
    console.warn(`  brainList error ${res.status}`);
    return [];
  }
  const data = await res.json();
  return Array.isArray(data) ? data : data.memories || [];
}

/** Inject a memory into the brain pipeline. */
async function brainShare(item) {
  const res = await fetch(`${BRAIN_URL}/v1/pipeline/inject`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: BRAIN_AUTH },
    body: JSON.stringify({ source: 'gemini-agent', ...item }),
  });
  if (!res.ok) {
    console.warn(`  brainShare error ${res.status}`);
    return null;
  }
  return res.json();
}

/** Ground a symbolic proposition in the brain. */
async function groundProposition(subject, predicate, object, confidence) {
  try {
    await fetch(`${BRAIN_URL}/v1/ground`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Authorization: BRAIN_AUTH },
      body: JSON.stringify({ subject, predicate, object, confidence, source: 'gemini-grounding' }),
    });
  } catch (err) {
    console.warn(`  groundProposition error: ${err.message}`);
  }
}

/** Strip PHI (emails, phone numbers, SSNs, date patterns). */
function stripPHI(text) {
  return text
    .replace(/\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b/g, '[EMAIL]')
    .replace(/\b\d{3}[-.]?\d{3}[-.]?\d{4}\b/g, '[PHONE]')
    .replace(/\b\d{3}-\d{2}-\d{4}\b/g, '[SSN]')
    .replace(/\b\d{1,2}\/\d{1,2}\/\d{2,4}\b/g, '[DATE]');
}

/** Extract first JSON object from text. */
function parseJSON(text) {
  try {
    const match = text.match(/\{[\s\S]*\}/);
    if (match) return JSON.parse(match[0]);
  } catch { /* ignore parse errors */ }
  return null;
}

/** Sleep for ms milliseconds. */
function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

/** Retry-aware wrapper for callGemini with exponential backoff. */
async function callGeminiRetry(prompt, maxRetries = 3) {
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await callGemini(prompt);
    } catch (err) {
      const is429 = err.message.includes('429');
      const is500 = err.message.includes('500');
      if (attempt === maxRetries - 1) throw err;
      if (is429 || is500) {
        const delay = Math.pow(2, attempt) * 1000;
        console.warn(`  Gemini ${is429 ? '429' : '500'}, retrying in ${delay}ms...`);
        await sleep(delay);
      } else {
        throw err;
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Phase 1: Fact Verifier
// ---------------------------------------------------------------------------
async function factVerify() {
  console.log('\n--- Phase 1: Fact Verifier ---');
  const memories = await brainSearch('*', MAX_MEMORIES);
  console.log(`  Fetched ${memories.length} memories to verify`);

  let verified = 0;
  let contradicted = 0;
  let skipped = 0;

  for (const mem of memories) {
    const claim = (mem.title || '') + ': ' + (mem.content || '').slice(0, 500);
    const sanitized = stripPHI(claim);

    try {
      const result = await callGeminiRetry(
        `Verify this claim. Is it factually accurate based on current evidence? ` +
        `Respond with JSON: {"verified": true/false, "confidence": 0.0-1.0, "correction": "..." or null, "sources": ["url1", "url2"]}` +
        `\n\nClaim: ${sanitized}`,
      );

      const verification = parseJSON(result.text);
      if (verification) {
        const status = verification.verified ? 'verified' : 'contradicted';
        await brainShare({
          title: `Fact Check: ${(mem.title || '').slice(0, 80)}`,
          content:
            `Verification: ${verification.verified ? 'SUPPORTED' : 'CONTRADICTED'} ` +
            `(confidence: ${verification.confidence}). ` +
            `${verification.correction || 'No correction needed.'}`,
          tags: ['fact-check', status, 'gemini-grounding', 'phase-1'],
          category: 'pattern',
        });

        if (verification.verified) verified++;
        else contradicted++;

        if (result.grounding) {
          const srcCount = result.grounding.groundingChunks?.length ||
            result.grounding.sources || 0;
          const supCount = result.grounding.groundingSupports?.length ||
            result.grounding.supports || 0;
          console.log(`  [${status.toUpperCase()}] ${(mem.title || '').slice(0, 60)} — ${srcCount} sources, ${supCount} supports`);
        } else {
          console.log(`  [${status.toUpperCase()}] ${(mem.title || '').slice(0, 60)}`);
        }
      } else {
        skipped++;
        console.log(`  [SKIP] ${(mem.title || '').slice(0, 60)} — could not parse Gemini response`);
      }
    } catch (err) {
      skipped++;
      console.warn(`  [ERROR] ${(mem.title || '').slice(0, 60)}: ${err.message}`);
    }
  }

  console.log(`  Phase 1 complete: ${verified} verified, ${contradicted} contradicted, ${skipped} skipped`);
}

// ---------------------------------------------------------------------------
// Phase 2: Relation Generator
// ---------------------------------------------------------------------------
async function generateRelations() {
  console.log('\n--- Phase 2: Relation Generator ---');
  const categories = ['architecture', 'solution', 'pattern', 'security'];
  const allMems = [];

  for (const cat of categories) {
    const mems = await brainList(cat, 5);
    allMems.push(...mems);
  }
  console.log(`  Fetched ${allMems.length} memories across ${categories.length} categories`);

  let relationsFound = 0;
  let pairsEvaluated = 0;

  for (let i = 0; i < allMems.length; i++) {
    for (let j = i + 1; j < Math.min(allMems.length, i + 5); j++) {
      const a = allMems[i];
      const b = allMems[j];
      pairsEvaluated++;

      try {
        const result = await callGeminiRetry(
          `What is the relationship between these two knowledge items? ` +
          `Respond with JSON: {"predicate": "causes|implies|requires|contradicts|enables|similar_to|no_relationship", "confidence": 0.0-1.0, "explanation": "..."}` +
          `\n\nItem A: ${(a.title || '')}: ${(a.content || '').slice(0, 300)}` +
          `\n\nItem B: ${(b.title || '')}: ${(b.content || '').slice(0, 300)}`,
        );

        const rel = parseJSON(result.text);
        if (rel && rel.confidence > 0.6 && rel.predicate !== 'no_relationship') {
          await brainShare({
            title: `Relation: ${(a.title || '').slice(0, 30)} ${rel.predicate} ${(b.title || '').slice(0, 30)}`,
            content: `${rel.predicate}: ${rel.explanation}. Source A: ${a.id || 'unknown'}, Source B: ${b.id || 'unknown'}`,
            tags: ['relation', rel.predicate, 'gemini-grounding', 'horn-clause', 'phase-2'],
            category: 'pattern',
          });

          await groundProposition(a.title || '', rel.predicate, b.title || '', rel.confidence);
          relationsFound++;
          console.log(`  [REL] ${(a.title || '').slice(0, 25)} --${rel.predicate}--> ${(b.title || '').slice(0, 25)} (${rel.confidence})`);
        }
      } catch (err) {
        console.warn(`  [ERROR] pair ${i},${j}: ${err.message}`);
      }
    }
  }

  console.log(`  Phase 2 complete: ${relationsFound} relations from ${pairsEvaluated} pairs`);
}

// ---------------------------------------------------------------------------
// Phase 3: Cross-Domain Explorer
// ---------------------------------------------------------------------------
async function crossDomainExplore() {
  console.log('\n--- Phase 3: Cross-Domain Explorer ---');
  const domains = [
    { query: 'melanoma skin cancer treatment', domain: 'medical' },
    { query: 'neural network deep learning', domain: 'cs' },
    { query: 'quantum mechanics dark matter', domain: 'physics' },
  ];

  const domainMems = {};
  for (const d of domains) {
    domainMems[d.domain] = await brainSearch(d.query, 3);
    console.log(`  ${d.domain}: ${domainMems[d.domain].length} memories`);
  }

  let discoveries = 0;
  const domainKeys = Object.keys(domainMems);

  for (let i = 0; i < domainKeys.length; i++) {
    for (let j = i + 1; j < domainKeys.length; j++) {
      const dA = domainKeys[i];
      const dB = domainKeys[j];
      const memA = domainMems[dA][0];
      const memB = domainMems[dB][0];
      if (!memA || !memB) continue;

      try {
        const result = await callGeminiRetry(
          `These two items are from different fields (${dA} and ${dB}). ` +
          `Find a non-obvious connection between them. ` +
          `Respond with JSON: {"connection": "...", "strength": 0.0-1.0, "novelty": 0.0-1.0, "bridge_concept": "..."}` +
          `\n\nField ${dA}: ${(memA.title || '')}: ${(memA.content || '').slice(0, 300)}` +
          `\n\nField ${dB}: ${(memB.title || '')}: ${(memB.content || '').slice(0, 300)}`,
        );

        const conn = parseJSON(result.text);
        if (conn && conn.strength > 0.4) {
          await brainShare({
            title: `Cross-Domain: ${dA}<->${dB} via "${conn.bridge_concept}"`,
            content: conn.connection,
            tags: ['cross-domain', dA, dB, conn.bridge_concept || 'bridge', 'gemini-grounding', 'discovery', 'phase-3'],
            category: 'pattern',
          });

          discoveries++;
          console.log(`  [BRIDGE] ${dA}<->${dB}: "${conn.bridge_concept}" (strength: ${conn.strength}, novelty: ${conn.novelty})`);
        }
      } catch (err) {
        console.warn(`  [ERROR] ${dA}<->${dB}: ${err.message}`);
      }
    }
  }

  console.log(`  Phase 3 complete: ${discoveries} cross-domain discoveries`);
}

// ---------------------------------------------------------------------------
// Phase 4: Research Director
// ---------------------------------------------------------------------------
async function researchDirector() {
  console.log('\n--- Phase 4: Research Director ---');

  // 1. Check drift status
  let drift = null;
  try {
    const driftRes = await fetch(`${BRAIN_URL}/v1/drift`, {
      headers: { Authorization: BRAIN_AUTH },
    });
    if (driftRes.ok) drift = await driftRes.json();
  } catch (err) {
    console.warn(`  Drift endpoint unavailable: ${err.message}`);
  }

  if (drift) {
    console.log(`  Drift status: ${JSON.stringify(drift).slice(0, 200)}`);
  }

  // 2. Formulate research questions about recent topics
  const recentMems = await brainSearch('2025 2026 latest recent', 5);
  console.log(`  Found ${recentMems.length} recent memories for research`);

  let findings = 0;

  for (const mem of recentMems.slice(0, 3)) {
    try {
      const result = await callGeminiRetry(
        `Based on this knowledge, what are 2 important questions that need answering? ` +
        `Research these questions using current information. ` +
        `Respond with JSON: {"questions": ["q1", "q2"], "answers": [{"question": "q1", "answer": "...", "confidence": 0.0-1.0}]}` +
        `\n\nTopic: ${(mem.title || '')}: ${(mem.content || '').slice(0, 400)}`,
      );

      const research = parseJSON(result.text);
      if (research && research.answers) {
        for (const ans of research.answers) {
          if (ans.confidence > 0.5) {
            await brainShare({
              title: `Research: ${(ans.question || '').slice(0, 80)}`,
              content: ans.answer,
              tags: ['research', 'gemini-grounding', 'auto-research', 'discovery', 'phase-4'],
              category: 'solution',
            });
            findings++;
            console.log(`  [RESEARCH] ${(ans.question || '').slice(0, 70)} (conf: ${ans.confidence})`);
          }
        }
      }
    } catch (err) {
      console.warn(`  [ERROR] research on "${(mem.title || '').slice(0, 40)}": ${err.message}`);
    }
  }

  console.log(`  Phase 4 complete: ${findings} research findings injected`);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const phase =
  process.argv.find((a) => a.startsWith('--phase='))?.split('=')[1] ||
  process.argv[process.argv.indexOf('--phase') + 1] ||
  'all';

async function main() {
  console.log(`=== Gemini Grounding Agents — Phase: ${phase} ===`);
  console.log(`Brain: ${BRAIN_URL}, Model: ${GEMINI_MODEL}`);

  if (!GEMINI_API_KEY) {
    console.error('ERROR: GEMINI_API_KEY not set');
    process.exit(1);
  }

  // Get brain status before
  let beforeCount = 0;
  try {
    const before = await fetch(`${BRAIN_URL}/v1/status`).then((r) => r.json());
    beforeCount = before.total_memories || 0;
    console.log(`Brain before: ${beforeCount} memories`);
  } catch (err) {
    console.warn(`Could not fetch brain status: ${err.message}`);
  }

  // Execute requested phase(s)
  if (phase === 'fact-verify' || phase === 'all') await factVerify();
  if (phase === 'relate' || phase === 'all') await generateRelations();
  if (phase === 'cross-domain' || phase === 'all') await crossDomainExplore();
  if (phase === 'research' || phase === 'all') await researchDirector();

  // Get brain status after
  try {
    const after = await fetch(`${BRAIN_URL}/v1/status`).then((r) => r.json());
    const afterCount = after.total_memories || 0;
    console.log(`\nBrain after: ${afterCount} memories (+${afterCount - beforeCount})`);
  } catch (err) {
    console.warn(`Could not fetch brain status: ${err.message}`);
  }

  console.log('=== Done ===');
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
