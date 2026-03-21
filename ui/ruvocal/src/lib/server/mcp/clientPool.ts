import { Client } from "@modelcontextprotocol/sdk/client";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import type { McpServerConfig } from "./httpClient";

const pool = new Map<string, Client>();

function keyOf(server: McpServerConfig) {
	const headers = Object.entries(server.headers ?? {})
		.sort(([a], [b]) => a.localeCompare(b))
		.map(([k, v]) => `${k}:${v}`)
		.join("|\u0000|");
	return `${server.url}|${headers}`;
}

export async function getClient(server: McpServerConfig, signal?: AbortSignal): Promise<Client> {
	const key = keyOf(server);
	const existing = pool.get(key);
	if (existing) {
		// Verify the cached client is still alive by checking transport state
		try {
			// If the transport is closed/errored, evict and reconnect
			if ((existing as unknown as { _transport?: { readyState?: number } })._transport?.readyState === 2) {
				pool.delete(key);
			} else {
				return existing;
			}
		} catch {
			return existing;
		}
	}

	let firstError: unknown;
	const client = new Client({ name: "chat-ui-mcp", version: "0.1.0" });
	const url = new URL(server.url);
	const requestInit: RequestInit = { headers: server.headers, signal };
	try {
		try {
			await client.connect(new StreamableHTTPClientTransport(url, { requestInit }));
		} catch (httpErr) {
			// Remember the original HTTP transport error so we can surface it if the fallback also fails.
			// Today we always show the SSE message, which is misleading when the real failure was HTTP (e.g. 500).
			firstError = httpErr;
			await client.connect(new SSEClientTransport(url, { requestInit }));
		}
	} catch (err) {
		try {
			await client.close?.();
		} catch {}
		// Prefer the HTTP error if both transports fail; otherwise fall back to the last error.
		if (firstError) {
			const message =
				"HTTP transport failed: " +
				String(firstError instanceof Error ? firstError.message : firstError) +
				"; SSE fallback failed: " +
				String(err instanceof Error ? err.message : err);
			throw new Error(message, { cause: err instanceof Error ? err : undefined });
		}
		throw err;
	}

	pool.set(key, client);
	return client;
}

export async function drainPool() {
	for (const [key, client] of pool) {
		try {
			await client.close?.();
		} catch {}
		pool.delete(key);
	}
}

export function evictFromPool(server: McpServerConfig): Client | undefined {
	const key = keyOf(server);
	const client = pool.get(key);
	if (client) {
		pool.delete(key);
	}
	return client;
}
