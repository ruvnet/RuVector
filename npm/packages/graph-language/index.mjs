import { spawn } from 'node:child_process';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const LIMIT = 16 * 1024 * 1024;
export class RuGraphError extends Error {
  constructor(code, message, position) {
    super(message); this.name = 'RuGraphError'; this.code = code; this.position = position;
  }
}

/** The executable and model paths are trusted host configuration, never program input. */
export class RuGraph {
  constructor({ binary = process.env.RUVECTOR_LANGUAGE_BIN || 'rvg', timeoutMs = 30000 } = {}) {
    if (typeof binary !== 'string' || !binary || binary.includes('\0') || !Number.isInteger(timeoutMs) || timeoutMs <= 0 || timeoutMs > 300000) {
      throw new RuGraphError('INVALID_INPUT', 'Invalid executable or timeout');
    }
    this.binary = binary; this.timeoutMs = timeoutMs;
  }
  async invoke(args, input) {
    if (typeof input !== 'string' || !Array.isArray(args) || args.some(arg => typeof arg !== 'string' || arg.includes('\0'))) {
      throw new RuGraphError('INVALID_INPUT', 'Expected string input and arguments');
    }
    if (Buffer.byteLength(input) > LIMIT) throw new RuGraphError('INPUT_LIMIT', 'Input exceeds 16 MiB');
    return new Promise((resolve, reject) => {
      const child = spawn(this.binary, args, { shell: false, stdio: ['pipe', 'pipe', 'pipe'] });
      let settled = false; let size = 0; const stdout = []; const stderr = [];
      const finish = (error, value) => {
        if (settled) return; settled = true; clearTimeout(timer);
        if (error) { child.kill('SIGKILL'); reject(error); } else resolve(value);
      };
      const timer = setTimeout(() => finish(new RuGraphError('TIMEOUT', 'RuGraph exceeded the process timeout')), this.timeoutMs);
      const collect = (chunks) => (chunk) => {
        if (settled) return;
        size += chunk.length;
        if (size > LIMIT) finish(new RuGraphError('OUTPUT_LIMIT', 'Output exceeds 16 MiB'));
        else chunks.push(chunk);
      };
      child.stdout.on('data', collect(stdout)); child.stderr.on('data', collect(stderr));
      child.on('error', error => finish(new RuGraphError('EXECUTABLE_ERROR', error.message)));
      child.stdin.on('error', error => { if (error.code !== 'EPIPE') finish(new RuGraphError('IO_ERROR', error.message)); });
      child.on('close', code => {
        if (settled) return;
        let out; let err;
        try {
          const decoder = new TextDecoder('utf-8', { fatal: true });
          out = decoder.decode(Buffer.concat(stdout)); err = decoder.decode(Buffer.concat(stderr));
        } catch { finish(new RuGraphError('INVALID_OUTPUT', 'Runtime returned invalid UTF-8')); return; }
        if (code !== 0) {
          let data;
          try { data = decode(err); } catch { data = null; }
          if (!data || typeof data.code !== 'string' || !data.code || typeof data.message !== 'string' ||
              (data.position !== undefined && (!Number.isSafeInteger(data.position) || data.position < 0))) {
            finish(new RuGraphError('PROCESS_ERROR', err || `Exit ${code}`));
          } else finish(new RuGraphError(data.code, data.message, data.position));
        } else {
          try { finish(null, decode(out)); }
          catch (error) { finish(error instanceof RuGraphError ? error : new RuGraphError('INVALID_OUTPUT', 'Runtime returned invalid JSON')); }
        }
      });
      child.stdin.end(input);
    });
  }
  check(source) { return this.invoke(['check', '-'], source); }
  compile(source) { return this.invoke(['compile', '-'], source); }
  run(request, { modelsPath } = {}) {
    return this.invoke(['run', '--request', '-', ...modelArgs(modelsPath)], encode(request));
  }
  async audit(request, options) {
    const encodedRequest = encode(request); const encodedOptions = encode(options);
    let directory;
    try {
      directory = await mkdtemp(join(tmpdir(), 'rugraph-audit-'));
      const path = join(directory, 'options.json');
      await writeFile(path, encodedOptions, { mode: 0o600 });
      return await this.invoke(['audit', '--request', '-', '--audit-options', path], encodedRequest);
    } catch (error) {
      throw error instanceof RuGraphError ? error : new RuGraphError('IO_ERROR', error.message);
    } finally {
      if (directory) {
        try { await rm(directory, { recursive: true, force: true }); }
        catch (error) { throw new RuGraphError('IO_ERROR', error.message); }
      }
    }
  }
  async replay(request, receipt, { modelsPath } = {}) {
    const encodedRequest = encode(request); const encodedReceipt = encode(receipt);
    const models = modelArgs(modelsPath);
    let directory;
    try {
      directory = await mkdtemp(join(tmpdir(), 'ruvector-language-'));
      const path = join(directory, 'receipt.json');
      await writeFile(path, encodedReceipt, { mode: 0o600 });
      return await this.invoke(['replay', '--request', '-', '--receipt', path, ...models], encodedRequest);
    } catch (error) {
      throw error instanceof RuGraphError ? error : new RuGraphError('IO_ERROR', error.message);
    } finally {
      if (directory) {
        try { await rm(directory, { recursive: true, force: true }); }
        catch (error) { throw new RuGraphError('IO_ERROR', error.message); }
      }
    }
  }
}
function modelArgs(path) {
  if (path === undefined) return [];
  if (typeof path !== 'string' || !path || path.includes('\0') || path === '-' || path.startsWith('--')) throw new RuGraphError('INVALID_INPUT', 'Invalid modelsPath');
  return ['--models', path];
}
function encode(value) {
  let encoded;
  try {
    encoded = JSON.stringify(value, (_, item) => {
      if (typeof item === 'number') {
        if (!Number.isFinite(item)) throw new RuGraphError('INVALID_INPUT', 'Numbers must be finite');
        safeInteger(item);
      }
      return item;
    });
  } catch (error) {
    throw error instanceof RuGraphError ? error : new RuGraphError('INVALID_INPUT', error.message);
  }
  if (typeof encoded !== 'string') throw new RuGraphError('INVALID_INPUT', 'Expected JSON value');
  if (Buffer.byteLength(encoded) > LIMIT) throw new RuGraphError('INPUT_LIMIT', 'Input exceeds 16 MiB');
  return encoded;
}
function safeInteger(value) {
  if (Number.isInteger(value) && !Number.isSafeInteger(value)) {
    throw new RuGraphError('UNSAFE_INTEGER', 'Integer exceeds JavaScript exact range; use the Python SDK or Rust API');
  }
}
function decode(value) {
  const parsed = JSON.parse(value, (_, item) => {
    if (typeof item === 'number') {
      if (!Number.isFinite(item)) throw new RuGraphError('INVALID_OUTPUT', 'Runtime returned a nonfinite number');
      safeInteger(item);
    }
    return item;
  });
  if (parsed === null || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new RuGraphError('INVALID_OUTPUT', 'Runtime must return a JSON object');
  }
  return parsed;
}
