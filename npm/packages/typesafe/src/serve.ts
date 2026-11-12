/**
 * `typesafe serve` — a Jev-compatible drop-in over `node:http`. Exposes
 * `POST /v1/systemone` (Jev's body/response shape) and `GET /healthz`. Binds
 * 127.0.0.1 by default. Request bodies are never logged (ADR-005): the access
 * log carries method, path, status and milliseconds only.
 *
 * This is a server, not a client: it uses `http.createServer` and never makes
 * an outbound request.
 */

import * as http from 'node:http';
import { TypesafeError } from './errors';
import type { SystemOneBody, Typesafe } from './client';

/** Cap the request body well above the 16 KiB `state` limit (ADR-005). */
const MAX_BODY_BYTES = 1024 * 1024;

export interface ServeOptions {
  port?: number;
  host?: string;
  /** Strip the five additive answer fields so responses match Jev exactly. */
  jevShapeOnly?: boolean;
  /** Access-log sink (line already redacted of any body). Default: stdout. */
  log?: (line: string) => void;
}

export interface ServeHandle {
  server: http.Server;
  port: number;
  host: string;
  close(): Promise<void>;
}

function statusForError(err: TypesafeError): number {
  return err.kind === 'embedder' ? 500 : 400;
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
        // Drain the rest into the void (do NOT destroy the socket — the caller
        // still writes a 400 on the response), then reject.
        req.resume();
        reject(new TypesafeError('request body exceeds 1 MiB', 'limit'));
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
      reject(new TypesafeError('request stream error', 'invalid'));
    });
  });
}

export function serve(ts: Typesafe, opts: ServeOptions = {}): Promise<ServeHandle> {
  const host = opts.host ?? '127.0.0.1';
  const port = opts.port ?? 8787;
  const writeLog = opts.log ?? ((line: string) => process.stdout.write(line + '\n'));

  const server = http.createServer((req, res) => {
    const started = Date.now();
    const method = req.method ?? 'GET';
    const pathname = new URL(req.url ?? '/', 'http://localhost').pathname;

    const send = (status: number, payload: unknown): void => {
      const text = JSON.stringify(payload);
      res.writeHead(status, { 'content-type': 'application/json' });
      res.end(text);
      writeLog(`${method} ${pathname} ${status} ${Date.now() - started}ms`);
    };

    if (method === 'GET' && pathname === '/healthz') {
      send(200, { status: 'ok', version: ts.version, backend: ts.backend });
      return;
    }

    if (method === 'POST' && pathname === '/v1/systemone') {
      readBody(req)
        .then((raw) => {
          let body: SystemOneBody;
          try {
            body = JSON.parse(raw) as SystemOneBody;
          } catch {
            throw new TypesafeError('request body is not valid JSON', 'invalid');
          }
          return ts.systemOne(body, { jevShapeOnly: opts.jevShapeOnly });
        })
        .then((response) => send(200, response))
        .catch((err: unknown) => {
          if (err instanceof TypesafeError) {
            send(statusForError(err), { error: { kind: err.kind, message: err.message } });
          } else {
            send(500, { error: { kind: 'embedder', message: 'internal error' } });
          }
        });
      return;
    }

    send(404, { error: { kind: 'invalid', message: 'not found' } });
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
