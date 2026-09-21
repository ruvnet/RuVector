'use strict';

/**
 * Deterministic fake binding for the API and CLI tests. It is NOT the decision
 * engine — it returns structurally valid, Jev-shaped responses derived from the
 * request so the TypeScript layer, the CLI, and the server can be exercised
 * before the native/WASM binding exists. It records every request it saw on
 * `Engine.prototype.calls` via the instance, so a test can assert wire shape.
 *
 * Triggers: state "__ERR_LIMIT__" / "__ERR_INVALID__" / "__ERR_EMBEDDER__"
 * return the matching error envelope instead of a response.
 */

const META = () => ({
  confidence: 0.8,
  abstain: 0.1,
  calibrated: false,
  head: 'nearest-prototype',
  model: 'hash-bow-256@test-double',
  temperature: 1,
});

class Engine {
  constructor(optionsJson) {
    this.options = JSON.parse(optionsJson || '{}');
    this.calls = [];
  }

  decideJson(requestJson) {
    const req = JSON.parse(requestJson);
    this.calls.push(req);
    if (req.state === '__ERR_LIMIT__') {
      return JSON.stringify({ error: { kind: 'limit', message: 'state exceeds 16 KiB' } });
    }
    if (req.state === '__ERR_INVALID__') {
      return JSON.stringify({ error: { kind: 'invalid', message: 'no questions' } });
    }
    if (req.state === '__ERR_EMBEDDER__') {
      return JSON.stringify({ error: { kind: 'embedder', message: 'embedder failure' } });
    }
    const answers = {};
    for (const id of Object.keys(req.questions)) {
      const q = req.questions[id];
      const meta = META();
      if (q.type === 'choice') {
        const keys = Object.keys(q.criteria);
        const probabilities = {};
        keys.forEach((k, i) => {
          probabilities[k] = i === 0 ? 0.8 : 0.2 / Math.max(1, keys.length - 1);
        });
        answers[id] = Object.assign({ choice: keys[0], probabilities }, meta);
      } else if (q.type === 'score') {
        const legend = q.legend;
        const probabilities = legend.map((_, i) =>
          i === 0 ? 0.8 : 0.2 / Math.max(1, legend.length - 1),
        );
        answers[id] = Object.assign(
          { score: 0, legend: legend[0], probabilities },
          meta,
          { head: 'nearest-prototype' },
        );
      } else {
        answers[id] = Object.assign({ noul: 0.7 }, meta, {
          head: 'similarity-uncalibrated',
        });
      }
    }
    const usage = {
      embed_calls: 1,
      texts_embedded: Object.keys(req.questions).length + 1,
      state_bytes: Buffer.byteLength(req.state || '', 'utf8'),
    };
    return JSON.stringify({ answers, usage });
  }

  trainJson(trainJson) {
    const t = JSON.parse(trainJson);
    return JSON.stringify({
      question: t.question,
      accepted: (t.examples || []).length,
      rejected: 0,
      head: 'linear-probe',
      calibrated: false,
    });
  }

  statsJson() {
    return JSON.stringify({ questions: {}, bank_size: 0 });
  }
}

module.exports = {
  Engine,
  version: () => '0.1.0-fake',
  backend: 'wasm',
};
