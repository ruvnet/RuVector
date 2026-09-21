/**
 * `kge serve` — a small HTTP surface over `node:http`. Exposes
 * `POST /v1/predict`, `POST /v1/compose` and `GET /healthz`. Binds 127.0.0.1 by
 * default. Request bodies are never logged (ADR-005): the access log carries
 * method, path, status and milliseconds only.
 *
 * This is a server, not a client: it uses `http.createServer` and never makes
 * an outbound request.
 */

import * as http from 'node:http';
import type { ComposeQuery, Kge, PredictQuery } from './client';
import { KgeError } from './errors';

/** Cap the request body at 1 MiB (ADR-005). */
const MAX_BODY_BYTES = 1024 * 1024;

export interface ServeOptions {
  port?: number;
  host?: string;
  /** Access-log sink (line already redacted of any body). Default: stdout. */
  log?: (line: string) => void;
}

export interface ServeHandle {
  server: http.Server;
  port: number;
  host: string;
  close(): Promise<void>;
}

function statusForError(err: KgeError): number {
  if (err.kind === 'unavailable') return 501;
  if (err.kind === 'scorer') return 500;
  return 400;
}

function readBody(req: http.IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];
    let size = 0;
    let settled = false;
    req.on('data', (chunk: Buffer) => {
      if (settled) return;
      size += chunk.length;
      if (size > MAX_BODY_BYTES) {
        settled = true;
        req.resume();
        reject(new KgeError('request body exceeds 1 MiB', 'limit'));
        return;
      }
      chunks.push(chunk);
    });
    req.on('end', () => {
      if (settled) return;
      settled = true;
      resolve(Buffer.concat(chunks).toString('utf8'));
    });
    req.on('error', () => {
      if (settled) return;
      settled = true;
      reject(new KgeError('request stream error', 'invalid'));
    });
  });
}

export function serve(kge: Kge, opts: ServeOptions = {}): Promise<ServeHandle> {
  const host = opts.host ?? '127.0.0.1';
  const port = opts.port ?? 8788;
  const writeLog = opts.log ?? ((line: string) => process.stdout.write(line + '\n'));

  const server = http.createServer((req, res) => {
    const started = Date.now();
    const method = req.method ?? 'GET';
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;

    const send = (status: number, payload: unknown): void => {
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(JSON.stringify(payload));
      writeLog(`${method} ${pathname} ${status} ${Date.now() - started}ms`);
    };

    if (method === 'GET' && pathname === '/healthz') {
      send(200, { status: 'ok', version: kge.version, backend: kge.backend });
      return;
    }

    const route =
      method === 'POST' && pathname === '/v1/predict'
        ? 'predict'
        : method === 'POST' && pathname === '/v1/compose'
          ? 'compose'
          : null;

    if (!route) {
      send(404, { error: { kind: 'invalid', message: 'not found' } });
      return;
    }

    readBody(req)
      .then((raw) => {
        let body: unknown;
        try {
          body = JSON.parse(raw);
        } catch {
          throw new KgeError('request body is not valid JSON', 'invalid');
        }
        return route === 'predict'
          ? kge.predict(body as PredictQuery<string>)
          : kge.compose(body as ComposeQuery<string>);
      })
      .then((result) => send(200, result))
      .catch((err: unknown) => {
        if (err instanceof KgeError) {
          send(statusForError(err), { error: { kind: err.kind, message: err.message } });
        } else {
          send(500, { error: { kind: 'scorer', message: 'internal error' } });
        }
      });
  });

  return new Promise<ServeHandle>((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => {
      server.removeListener('error', reject);
      const address = server.address();
      const boundPort = typeof address === 'object' && address ? address.port : port;
      resolve({
        server,
        port: boundPort,
        host,
        close: () =>
          new Promise<void>((res, rej) => server.close((e) => (e ? rej(e) : res()))),
      });
    });
  });
}
