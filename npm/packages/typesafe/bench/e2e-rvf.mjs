#!/usr/bin/env node
// Benchmark-only RVF adapter: stores ONNX/tokenizer, instructions, options and
// example bank. Restore re-embeds the bank and lazily refits heads; this is NOT
// a native frozen-head loader or a KGE benchmark. No remote API calls.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import { fork, execFileSync } from 'node:child_process';
import { once } from 'node:events';
import { isDeepStrictEqual } from 'node:util';
import { loadTickets, excludeHeldOutText, buildQuestions, majorityLabelFromTrain } from './lib/fixture.mjs';
import { trainTicketQuestions, runLocal, replayJev } from './lib/arms.mjs';
import { resolveLocalContext } from './lib/model-dir.mjs';
import { scoreRecords, toItemRecords, hostInfo } from './lib/receipt.mjs';
import { compareToJev, loadSlices } from './vs-jev.mjs';
import { latency } from './lib/metrics.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const writer = require('../../rvforge/dist/rvf-writer.js');
const reader = require('../../rvforge/dist/rvf.js');
const sha = (bytes) => createHash('sha256').update(bytes).digest('hex');
const OPTIONS = { probeIterations: 5000, choiceInstructions: true };
const SCHEMA = 'typesafe-bench/rvf-bundle@1';

// RVForge's real RVF codec, with an explicit benchmark payload schema.
function pack(entries) {
  let offset = 0;
  const segments = [];
  const files = [];
  for (const [name, bytes] of Object.entries(entries)) {
    files.push({ name, offset, bytes: bytes.length, sha256: sha(bytes) });
    const segment = writer.writeSegment(writer.SEGMENT_TYPE.META, bytes, files.length);
    segments.push(segment);
    offset += segment.length;
  }
  const manifest = Buffer.from(JSON.stringify({ schema: SCHEMA, files }));
  segments.push(writer.writeSegment(writer.SEGMENT_TYPE.MANIFEST, manifest, files.length + 1));
  segments.push(writer.rootManifestPage(Buffer.from(sha(manifest).slice(0, 32), 'hex'), offset, manifest.length));
  return Buffer.concat(segments);
}

function unpack(bytes, expectedHash) {
  assert.equal(sha(bytes), expectedHash, 'RVF digest mismatch');
  const root = reader.parseRootManifest(bytes.subarray(-reader.ROOT_MANIFEST_SIZE));
  assert(root.magicValid && root.checksumValid && root.version === 1, 'invalid RVF root');
  const readPayload = (offset) => {
    const header = reader.parseSegmentHeader(bytes, offset);
    assert(header.magicValid && header.version === 1 && header.flags === 0 && header.compression === 0);
    const end = offset + reader.SEGMENT_HEADER_SIZE + Number(header.payloadLength);
    assert(end <= bytes.length - reader.ROOT_MANIFEST_SIZE);
    const payload = bytes.subarray(offset + reader.SEGMENT_HEADER_SIZE, end);
    assert.equal(header.checksumAlgo, writer.CHECKSUM_ALGO_SHAKE256);
    assert.equal(writer.shake256_128(payload).toString('hex'), header.contentHash);
    return payload;
  };
  const manifest = JSON.parse(readPayload(Number(root.l1ManifestOffset)));
  assert.equal(manifest.schema, SCHEMA);
  const entries = {};
  for (const file of manifest.files) {
    assert(['model.onnx', 'tokenizer.json', 'config.json', 'bank.json'].includes(file.name));
    assert(!Object.hasOwn(entries, file.name), 'duplicate payload');
    const payload = readPayload(file.offset);
    assert.equal(payload.length, file.bytes);
    assert.equal(sha(payload), file.sha256);
    entries[file.name] = payload;
  }
  assert.equal(Object.keys(entries).length, 4);
  return entries;
}

async function serveBundle(path, digest) {
  const started = performance.now();
  const entries = unpack(readFileSync(path), digest);
  const config = JSON.parse(entries['config.json']);
  assert.equal(config.schema, SCHEMA);
  const dir = mkdtempSync(join(tmpdir(), 'typesafe-rvf-'));
  let handle;
  try {
    writeFileSync(join(dir, 'model.onnx'), entries['model.onnx']);
    writeFileSync(join(dir, 'tokenizer.json'), entries['tokenizer.json']);
    const { createTypesafe } = require('../dist/index.js');
    const { serve } = require('../dist/serve.js');
    const ts = createTypesafe({
      embedder: { kind: 'onnx', modelDir: dir, manifest: JSON.stringify(config.model) },
      engine: config.options,
    });
    ts.importBank(entries['bank.json'].toString());
    handle = await serve(ts, { port: 0, log: () => {} });
    process.send({ port: handle.port, questions: config.questions, restore_ms: performance.now() - started });
    await once(process, 'message');
  } finally {
    if (handle) await handle.close();
    rmSync(dir, { recursive: true, force: true });
  }
}

async function benchmark(outDir, configPath) {
  mkdirSync(outDir, { recursive: true });
  const selected = configPath ? JSON.parse(readFileSync(configPath)) : null;
  const modelName = selected?.model ?? 'bge-small-en-v1.5';
  const engineOptions = selected?.engine_options ?? OPTIONS;
  assert.equal(engineOptions.choiceInstructions, true, 'this benchmark requires choice instructions');
  const tickets = loadTickets();
  const pool = excludeHeldOutText(tickets.bySplit);
  const binding = require('../index.js');
  assert.equal(binding.backend, 'native');
  const context = resolveLocalContext({ embedder: 'onnx', model: modelName, modelDir: join(HERE, '../models') });
  assert(!context.onnx.unavailable, context.onnx.unavailable);
  const opts = { embedder: context.onnx.spec, engine: engineOptions };
  const started = performance.now();
  const engine = new binding.Engine(JSON.stringify(opts));
  const training = trainTicketQuestions(engine, pool.trainItems, tickets.questions, {
    shots: 10000, calibrationItems: tickets.bySplit.calibration,
  });
  assert(training.trained, JSON.stringify(training));
  const trainMs = performance.now() - started;
  const direct = runLocal(engine, tickets.bySplit.test, tickets.questions);
  assert(direct.available, direct.error);
  assert.equal(direct.records.length, 150);
  const spec = JSON.parse(context.onnx.spec.manifest);
  const model = spec.models ? spec.models[0] : spec;
  const config = {
    schema: SCHEMA, questions: buildQuestions(tickets.questions), options: engineOptions,
    model: { ...model, file: 'model.onnx', tokenizer_file: 'tokenizer.json' },
  };
  const bank = engine.exportBankJson();
  assert(!JSON.parse(bank).error);
  const rvf = pack({
    'model.onnx': readFileSync(join(context.onnx.spec.modelDir, model.file)),
    'tokenizer.json': readFileSync(join(context.onnx.spec.modelDir, model.tokenizer_file)),
    'config.json': Buffer.from(JSON.stringify(config)), 'bank.json': Buffer.from(bank),
  });
  const digest = sha(rvf);
  const rvfPath = join(outDir, 'decision.rvf');
  writeFileSync(rvfPath, rvf);
  // A corrupted payload must fail before model loading. Restore the byte after.
  rvf[reader.SEGMENT_HEADER_SIZE] ^= 1;
  assert.throws(() => unpack(rvf, digest), /RVF digest mismatch/);
  rvf[reader.SEGMENT_HEADER_SIZE] ^= 1;
  assert.equal(unpack(rvf, digest)['bank.json'].toString(), bank);

  const bootStart = performance.now();
  const child = fork(fileURLToPath(import.meta.url), ['--serve', rvfPath, digest], {
    stdio: ['ignore', 'inherit', 'inherit', 'ipc'],
  });
  const exited = once(child, 'exit');
  const ready = await Promise.race([
    once(child, 'message').then(([msg]) => msg),
    exited.then(([code]) => { throw new Error(`RVF server exited before ready (${code})`); }),
  ]);
  const bootMs = performance.now() - bootStart;
  const responses = [];
  const durations = [];
  const coldDurations = [];
  const mismatches = [];
  let predictionMatches = 0;
  let firstRequestMs;
  try {
    assert.deepEqual(ready.questions, config.questions);
    // One complete cold pass exercises lazy head fitting, tokenizer and server.
    // A second pass measures warm HTTP latency; both must reproduce direct output.
    for (let pass = 0; pass < 2; pass++) {
      for (const [index, item] of tickets.bySplit.test.entries()) {
        const t0 = performance.now();
        const resp = await fetch(`http://127.0.0.1:${ready.port}/v1/systemone`, {
          method: 'POST', headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ state: item.text, questions: ready.questions }),
          signal: AbortSignal.timeout(120000),
        });
        const result = await resp.json();
        const ms = performance.now() - t0;
        assert.equal(resp.status, 200, JSON.stringify(result));
        assert(!result.error);
        if (pass === 0 && index === 0) firstRequestMs = ms;
        const normalized = runLocal({ decideJson: () => JSON.stringify(result) }, [item], tickets.questions);
        assert(normalized.available, normalized.error);
        const actual = toItemRecords(normalized.records)[0];
        const expected = toItemRecords([direct.records[index]])[0];
        if (isDeepStrictEqual(actual.predicted, expected.predicted)) predictionMatches++;
        if (!isDeepStrictEqual(actual, expected)) mismatches.push({ pass, id: item.id, expected, actual });
        if (pass === 0) coldDurations.push(ms);
        if (pass === 1) { responses.push(result); durations.push(ms); }
      }
    }
  } finally {
    if (child.connected) child.send('stop');
    await exited;
  }
  let i = 0;
  const normalized = runLocal({ decideJson: () => JSON.stringify(responses[i++]) }, tickets.bySplit.test, tickets.questions);
  normalized.records.forEach((r, index) => { r.latencyMs = durations[index]; });
  const majorityLabels = Object.fromEntries(['urgent', 'frustration'].map(q => [q, majorityLabelFromTrain(pool.trainItems, q)]));
  const jev = JSON.parse(readFileSync(join(HERE, 'jev-baseline-2026-09-21.json')));
  const jevRecords = replayJev(jev, tickets.bySplit.test);
  assert(jevRecords.available);
  const records = toItemRecords(normalized.records);
  const slices = loadSlices(tickets);
  const receipt = {
    schema: 'typesafe-bench/e2e-rvf@1', generated_at: new Date().toISOString(),
    target: `TypeSafe + ${modelName} (not KGE or fine-tuned OpenJev)`,
    protocol: 'Train-only examples + calibration split; frozen gen-0 instructions; fresh-process RVF restore; 150 cold + 150 warm HTTP requests; frozen JEV replay, no live JEV calls.',
    limitations: ['Benchmark-only RVF adapter; bank is re-embedded and heads refitted, not frozen weights.', 'Warm pass repeats test texts and benefits from caches.', 'Local loopback vs historical remote JEV latency includes different network overhead.', 'No KGE text-to-graph decision adapter is exercised.'],
    source_sha: execFileSync('git', ['rev-parse', 'HEAD'], { cwd: HERE, encoding: 'utf8' }).trim(),
    source_dirty: execFileSync('git', ['status', '--porcelain', '--untracked-files=no'], { cwd: HERE, encoding: 'utf8' }).trim().length > 0,
    harness_sha256: sha(readFileSync(fileURLToPath(import.meta.url))),
    host: hostInfo(), binding: { version: binding.version(), backend: binding.backend,
      sha256: sha(readFileSync(join(HERE, '../native/typesafe.linux-x64-gnu.node'))) },
    model: config.model, engine_options: engineOptions, selection: selected,
    splits_hash: tickets.splitsHash,
    instructions_sha256: sha(Buffer.from(JSON.stringify(config.questions))),
    training: { ...training, excluded_heldout_texts: pool.excluded, elapsed_ms: trainMs },
    artifact: { file: 'decision.rvf', bytes: rvf.length, sha256: digest, corruption_rejected: true },
    timing: { fresh_process_to_ready_ms: bootMs, restore_ms: ready.restore_ms, first_http_request_ms: firstRequestMs,
      cold_start_to_first_decision_ms: bootMs + firstRequestMs,
      first_pass_http_ms: latency(coldDurations), warm_pass_http_ms: latency(durations) },
    parity: { cold_requests: 150, warm_requests: 150, exact_item_record_matches: 300 - mismatches.length,
      prediction_matches: predictionMatches, passed: mismatches.length === 0, mismatches },
    metrics: {
      local_rvf_http: scoreRecords(normalized.records, { departments: tickets.departments, majorityLabels, wallMs: durations.reduce((a,b) => a+b, 0) }),
      jev_frozen: scoreRecords(jevRecords.records, { departments: tickets.departments }),
    },
    paired: compareToJev({ records, jevBaseline: jev, testIds: slices.full, slices }),
    records,
  };
  writeFileSync(join(outDir, 'receipt.json'), JSON.stringify(receipt, null, 2) + '\n');
  console.log(JSON.stringify({ receipt: join(outDir, 'receipt.json'), artifact: receipt.artifact,
    timing: receipt.timing, parity: { ...receipt.parity, mismatches: mismatches.length },
    accuracy: Object.fromEntries(Object.entries(receipt.metrics).map(([arm, m]) => [arm, {
      department: m.choice_accuracy, urgent: m.urgent_accuracy, frustration: m.frustration_accuracy,
    }])),
  }, null, 2));
  // Write failure evidence for diagnosis, but NEVER turn a parity failure into
  // a passing run merely because label accuracy or latency looks attractive.
  if (mismatches.length) process.exitCode = 1;
}

if (process.argv[2] === '--serve') await serveBundle(process.argv[3], process.argv[4]);
else await benchmark(resolve(process.argv[2] ?? join(HERE, 'results', `e2e-rvf-${new Date().toISOString().replace(/[:.]/g, '-')}`)), process.argv[3]);
