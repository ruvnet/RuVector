import { HTTPError } from './validation.mjs';

export function response(value, status = 200, headers = {}) {
  return new Response(JSON.stringify(value), { status, headers: {
    'Content-Type': 'application/json; charset=utf-8', 'Cache-Control': 'no-store',
    'X-Content-Type-Options': 'nosniff', ...headers,
  } });
}

export function tokenMatches(value, token) {
  if (typeof value !== 'string' || value.length !== token.length) return false;
  let different = 0;
  for (let index = 0; index < token.length; index++) different |= value.charCodeAt(index) ^ token.charCodeAt(index);
  return different === 0;
}

export async function readBytes(stream, limit, timeoutMs) {
  if (!stream || typeof stream.getReader !== 'function') throw new HTTPError(400, 'invalid_body', 'A body is required');
  const reader = stream.getReader();
  let timer;
  const cancel = () => { void reader.cancel('body rejected').catch(() => {}); };
  const read = async () => {
    let bytes = new Uint8Array(Math.min(limit, 8192)), size = 0;
    for (;;) {
      const { value, done } = await reader.read();
      if (done) return bytes.subarray(0, size);
      if (!(value instanceof Uint8Array)) throw new HTTPError(400, 'invalid_body', 'Body must contain bytes');
      if (value.byteLength > limit - size) {
        cancel();
        throw new HTTPError(413, 'body_too_large', 'Body exceeds byte limit');
      }
      const next = size + value.byteLength;
      if (next > bytes.length) {
        const grown = new Uint8Array(Math.min(limit, Math.max(next, bytes.length * 2)));
        grown.set(bytes); bytes = grown;
      }
      bytes.set(value, size); size = next;
    }
  };
  const deadline = new Promise((_, reject) => {
    timer = setTimeout(() => { cancel(); reject(new HTTPError(408, 'body_timeout', 'Body read timed out')); }, timeoutMs);
  });
  try { return await Promise.race([read(), deadline]); }
  finally { clearTimeout(timer); reader.releaseLock(); }
}

export async function readJSON(request, limit, timeoutMs) {
  const type = request.headers.get('content-type');
  if (!type || !/^application\/json(?:\s*;\s*charset\s*=\s*(?:utf-8|"utf-8"))?\s*$/i.test(type)) {
    throw new HTTPError(415, 'unsupported_media_type', 'Content-Type must be UTF-8 application/json');
  }
  const length = request.headers.get('content-length');
  if (length !== null && (!/^(0|[1-9][0-9]*)$/.test(length) || Number(length) > limit)) {
    throw new HTTPError(413, 'body_too_large', 'Body exceeds byte limit');
  }
  const bytes = await readBytes(request.body, limit, timeoutMs);
  try { return JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes)); }
  catch { throw new HTTPError(400, 'invalid_json', 'Body must be valid UTF-8 JSON'); }
}
