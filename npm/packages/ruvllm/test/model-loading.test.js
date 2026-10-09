/**
 * Model loading through the native candle backend, without a real model:
 * missing files and unsupported GGUF architectures must fail loudly and leave
 * no model loaded. Skipped when the native binary cannot load models (none
 * installed, or a 2.0.x platform package); set RUVLLM_NATIVE_PATH to a
 * binary built with `npm run build:native` to run it.
 */
const { describe, test, before, after } = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');

const { RuvLLM } = require('../dist/cjs/index.js');

const supported = new RuvLLM().supportsModelLoading();

/** A GGUF v3 file holding only `general.architecture`. */
function ggufWithArchitecture(arch) {
  const u32 = (n) => {
    const b = Buffer.alloc(4);
    b.writeUInt32LE(n);
    return b;
  };
  const u64 = (n) => {
    const b = Buffer.alloc(8);
    b.writeBigUInt64LE(BigInt(n));
    return b;
  };
  const str = (s) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s)]);
  const GGUF_TYPE_STRING = 8;
  return Buffer.concat([
    Buffer.from('GGUF'),
    u32(3), // version
    u64(0), // tensor count
    u64(1), // metadata entries
    str('general.architecture'),
    u32(GGUF_TYPE_STRING),
    str(arch),
  ]);
}

describe('loadModel failures', { skip: !supported && 'native binary cannot load models' }, () => {
  let dir;
  before(() => {
    dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvllm-gguf-'));
  });
  after(() => fs.rmSync(dir, { recursive: true, force: true }));

  test('a missing file is RUVLLM_MODEL_NOT_FOUND, not a hub lookup', () => {
    const llm = new RuvLLM();
    assert.throws(() => llm.loadModel(path.join(dir, 'missing.gguf')), {
      code: 'RUVLLM_MODEL_NOT_FOUND',
      message: /no such file or directory/,
    });
    assert.strictEqual(llm.isModelLoaded(), false);
  });

  for (const arch of ['phi2', 'gemma2', 'qwen35']) {
    test(`GGUF architecture '${arch}' is rejected by name`, () => {
      const file = path.join(dir, `${arch}.gguf`);
      fs.writeFileSync(file, ggufWithArchitecture(arch));
      const llm = new RuvLLM();
      assert.throws(() => llm.loadModel(file), (e) => {
        assert.strictEqual(e.code, 'RUVLLM_MODEL_LOAD_FAILED');
        assert.match(e.message, new RegExp(`GGUF architecture '${arch}' is not supported`));
        assert.match(e.message, /supported: llama, mistral, qwen2, qwen3/);
        return true;
      });
      assert.strictEqual(llm.isModelLoaded(), false);
      assert.strictEqual(llm.modelInfo(), null);
      assert.throws(() => llm.generate('hi'), { code: 'RUVLLM_NO_LANGUAGE_MODEL' });
    });
  }

  test('a file that is not GGUF fails to load', () => {
    const file = path.join(dir, 'not-a-model.gguf');
    fs.writeFileSync(file, 'hello');
    assert.throws(() => new RuvLLM().loadModel(file), { code: 'RUVLLM_MODEL_LOAD_FAILED' });
  });
});
