/**
 * Minimal ambient declarations for the Node APIs this package uses. The package
 * ships with zero dependencies (no `@types/node`), so these cover exactly the
 * surface referenced by `src/**` — nothing more. `skipLibCheck` keeps them from
 * being deep-checked; they exist only to type the calls we make.
 */

interface NodeWriteStream {
  write(chunk: string): boolean;
}

interface NodeProcess {
  argv: string[];
  stdout: NodeWriteStream;
  stderr: NodeWriteStream;
  env: Record<string, string | undefined>;
  exit(code?: number): never;
}

declare var process: NodeProcess;
declare var performance: { now(): number };
declare var __dirname: string;
declare var require: (id: string) => any;
declare var module: { exports: any };

interface Buffer {
  toString(encoding?: string): string;
  length: number;
}
declare var Buffer: {
  from(data: string, encoding?: string): Buffer;
  concat(list: Buffer[]): Buffer;
  byteLength(str: string, encoding?: string): number;
};

declare class URL {
  constructor(input: string, base?: string);
  pathname: string;
  href: string;
}

declare module 'node:fs' {
  export function readFileSync(path: string | number, encoding: string): string;
  export function writeFileSync(path: string, data: string): void;
  export function existsSync(path: string): boolean;
}

declare module 'node:url' {
  export function pathToFileURL(path: string): URL;
}

declare module 'node:module' {
  export function createRequire(path: string | URL): (id: string) => any;
}

declare module 'node:http' {
  export interface IncomingMessage {
    method?: string;
    url?: string;
    on(event: 'data', listener: (chunk: Buffer) => void): this;
    on(event: 'end', listener: () => void): this;
    on(event: 'error', listener: (err: Error) => void): this;
    resume(): void;
    destroy(): void;
  }
  export interface ServerResponse {
    writeHead(status: number, headers?: Record<string, string>): void;
    end(chunk?: string): void;
  }
  export interface Server {
    listen(port: number, host: string, callback: () => void): this;
    close(callback?: (err?: Error) => void): this;
    once(event: 'error', listener: (err: Error) => void): this;
    removeListener(event: 'error', listener: (err: Error) => void): this;
    address(): { port: number } | string | null;
  }
  export function createServer(
    handler: (req: IncomingMessage, res: ServerResponse) => void,
  ): Server;
}
