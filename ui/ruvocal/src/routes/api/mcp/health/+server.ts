import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StreamableHTTPClientTransport } from "@modelcontextprotocol/sdk/client/streamableHttp.js";
import { SSEClientTransport } from "@modelcontextprotocol/sdk/client/sse.js";
import type { KeyValuePair } from "$lib/types/Tool";
import { config } from "$lib/server/config";
import { logger } from "$lib/server/logger";
import type { RequestHandler } from "./$types";
import { isValidUrl } from "$lib/server/urlSafety";
import { isStrictHfMcpLogin, hasNonEmptyToken, isExaMcpServer } from "$lib/server/mcp/hf";

/**
 * Raw SSE MCP health check — keeps SSE connection alive while POSTing.
 * The MCP SDK's SSEClientTransport can drop the SSE stream between connect()
 * and send(), causing 404 on servers with aggressive session cleanup.
 */
async function rawSseHealthCheck(
	sseUrl: string,
	headersRecord: Record<string, string>,
	timeoutMs: number
): Promise<HealthCheckResponse | null> {
	const controller = new AbortController();
	const timeout = setTimeout(() => controller.abort(), timeoutMs);

	try {
		// 1. Open SSE stream and keep it alive
		const sseResp = await fetch(sseUrl, {
			headers: { ...headersRecord, Accept: "text/event-stream" },
			signal: controller.signal,
		});
		if (!sseResp.ok || !sseResp.body) return null;

		const reader = sseResp.body.getReader();
		const decoder = new TextDecoder();
		let buffer = "";

		// 2. Read until we get the endpoint event
		let messageEndpoint = "";
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });

			// Parse SSE: look for "event: endpoint\ndata: /messages?sessionId=..."
			const match = buffer.match(/data:\s*(\/messages\?sessionId=[^\n\r]+)/);
			if (match) {
				messageEndpoint = match[1];
				break;
			}
		}

		if (!messageEndpoint) {
			reader.cancel();
			return null;
		}

		const baseOrigin = new URL(sseUrl).origin;
		const postUrl = `${baseOrigin}${messageEndpoint}`;

		// 3. Send initialize (SSE still open via reader)
		const initResp = await fetch(postUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({
				jsonrpc: "2.0",
				method: "initialize",
				id: 1,
				params: {
					protocolVersion: "2024-11-05",
					capabilities: {},
					clientInfo: { name: "chat-ui-health-check", version: "1.0.0" },
				},
			}),
		});

		if (!initResp.ok) {
			reader.cancel();
			return null;
		}

		// 4. Read initialize response from SSE stream
		buffer = "";
		let initResult = "";
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			const msgMatch = buffer.match(/data:\s*(\{[^\n]+\})/);
			if (msgMatch) {
				initResult = msgMatch[1];
				break;
			}
		}

		if (!initResult) {
			reader.cancel();
			return null;
		}

		// 5. Send initialized notification
		await fetch(postUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ jsonrpc: "2.0", method: "notifications/initialized" }),
		});

		// 6. Request tools list
		const toolsResp = await fetch(postUrl, {
			method: "POST",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ jsonrpc: "2.0", method: "tools/list", id: 2 }),
		});

		if (!toolsResp.ok) {
			reader.cancel();
			return null;
		}

		// 7. Read tools response from SSE stream
		buffer = "";
		let toolsResult = "";
		while (true) {
			const { done, value } = await reader.read();
			if (done) break;
			buffer += decoder.decode(value, { stream: true });
			const msgMatch = buffer.match(/data:\s*(\{[^\n]+\})/);
			if (msgMatch) {
				toolsResult = msgMatch[1];
				break;
			}
		}

		// 8. Clean up
		reader.cancel();

		if (!toolsResult) return null;

		const parsed = JSON.parse(toolsResult);
		const tools = parsed?.result?.tools;
		if (!Array.isArray(tools)) return null;

		return {
			ready: true,
			tools: tools.map((t: { name: string; description?: string; inputSchema?: unknown }) => ({
				name: t.name,
				description: t.description,
				inputSchema: t.inputSchema,
			})),
			authRequired: false,
		};
	} catch (err) {
		logger.debug({ err }, "[MCP Health] Raw SSE health check failed");
		return null;
	} finally {
		clearTimeout(timeout);
		controller.abort();
	}
}

interface HealthCheckRequest {
	url: string;
	headers?: KeyValuePair[];
}

interface HealthCheckResponse {
	ready: boolean;
	tools?: Array<{
		name: string;
		description?: string;
		inputSchema?: unknown;
	}>;
	error?: string;
	authRequired?: boolean;
}

export const POST: RequestHandler = async ({ request, locals }) => {
	let client: Client | undefined;

	try {
		const body: HealthCheckRequest = await request.json();
		const { url, headers } = body;

		if (!url) {
			return new Response(JSON.stringify({ ready: false, error: "URL is required" }), {
				status: 400,
				headers: { "Content-Type": "application/json" },
			});
		}

		// URL validation handled above

		if (!isValidUrl(url)) {
			return new Response(
				JSON.stringify({
					ready: false,
					error: "Invalid or unsafe URL (only HTTPS is supported)",
				} as HealthCheckResponse),
				{ status: 400, headers: { "Content-Type": "application/json" } }
			);
		}

		// Inject Exa API key for mcp.exa.ai servers via URL param
		let finalUrl = url;
		try {
			const exaApiKey = config.EXA_API_KEY;
			if (isExaMcpServer(url) && hasNonEmptyToken(exaApiKey)) {
				const urlObj = new URL(url);
				if (!urlObj.searchParams.has("exaApiKey")) {
					urlObj.searchParams.set("exaApiKey", exaApiKey);
					finalUrl = urlObj.toString();
					logger.debug({}, "[MCP Health] injected Exa API key");
				}
			}
		} catch {
			// best-effort injection
		}

		const baseUrl = new URL(finalUrl);

		// Minimal header handling
		const headersRecord: Record<string, string> = headers?.length
			? Object.fromEntries(headers.map((h) => [h.key, h.value]))
			: {};
		if (!headersRecord["Accept"]) {
			headersRecord["Accept"] = "application/json, text/event-stream";
		}

		// If enabled, attach the logged-in user's HF token only for the official HF MCP endpoint
		try {
			const shouldForward = config.MCP_FORWARD_HF_USER_TOKEN === "true";
			const userToken =
				(locals as unknown as { hfAccessToken?: string } | undefined)?.hfAccessToken ??
				(locals as unknown as { token?: string } | undefined)?.token;
			const hasAuth = typeof headersRecord["Authorization"] === "string";
			const isHfMcpTarget = isStrictHfMcpLogin(url);
			if (shouldForward && !hasAuth && isHfMcpTarget && hasNonEmptyToken(userToken)) {
				headersRecord["Authorization"] = `Bearer ${userToken}`;
			}
		} catch {
			// best-effort overlay
		}

		// Add an abort timeout to outbound requests (align with fetch-url: 30s)
		const controller = new AbortController();
		const timeoutId = setTimeout(() => controller.abort(), 30000);
		const signal = controller.signal;
		const requestInit: RequestInit = {
			headers: headersRecord,
			signal,
		};

		let httpError: Error | undefined;
		let lastError: Error | undefined;

		// Try Streamable HTTP transport first
		try {
			logger.info({}, `[MCP Health] Trying HTTP transport for ${url}`);
			client = new Client({
				name: "chat-ui-health-check",
				version: "1.0.0",
			});

			const transport = new StreamableHTTPClientTransport(baseUrl, { requestInit });
			logger.info({}, `[MCP Health] Connecting to ${url}...`);
			await client.connect(transport);
			logger.info({}, `[MCP Health] Connected successfully via HTTP`);

			// Connection successful, get tools
			const toolsResponse = await client.listTools();

			// Disconnect after getting tools
			await client.close();

			if (toolsResponse && toolsResponse.tools) {
				const response: HealthCheckResponse = {
					ready: true,
					tools: toolsResponse.tools.map((tool) => ({
						name: tool.name,
						description: tool.description,
						inputSchema: tool.inputSchema,
					})),
					authRequired: false,
				};

				const res = new Response(JSON.stringify(response), {
					status: 200,
					headers: { "Content-Type": "application/json" },
				});
				clearTimeout(timeoutId);
				return res;
			} else {
				const res = new Response(
					JSON.stringify({
						ready: false,
						error: "Connected but no tools available",
						authRequired: false,
					} as HealthCheckResponse),
					{
						status: 503,
						headers: { "Content-Type": "application/json" },
					}
				);
				clearTimeout(timeoutId);
				return res;
			}
		} catch (error) {
			httpError = error instanceof Error ? error : new Error(String(error));
			lastError = httpError;
			logger.warn(lastError.message, "Streamable HTTP failed, trying SSE transport...");

			// Close failed client
			try {
				await client?.close();
			} catch {
				// Ignore
			}

			// Try raw SSE health check first — keeps connection alive during POST
			// (works around servers with aggressive session cleanup)
			const rawResult = await rawSseHealthCheck(url, headersRecord, 15000);
			if (rawResult) {
				const res = new Response(JSON.stringify(rawResult), {
					status: 200,
					headers: { "Content-Type": "application/json" },
				});
				clearTimeout(timeoutId);
				return res;
			}

			// Fall back to SDK SSE transport
			try {
				logger.info({}, `[MCP Health] Trying SDK SSE transport for ${url}`);
				client = new Client({
					name: "chat-ui-health-check",
					version: "1.0.0",
				});

				const sseTransport = new SSEClientTransport(baseUrl, { requestInit });
				await client.connect(sseTransport);
				const toolsResponse = await client.listTools();
				await client.close();

				if (toolsResponse && toolsResponse.tools) {
					const response: HealthCheckResponse = {
						ready: true,
						tools: toolsResponse.tools.map((tool) => ({
							name: tool.name,
							description: tool.description,
							inputSchema: tool.inputSchema,
						})),
						authRequired: false,
					};

					const res = new Response(JSON.stringify(response), {
						status: 200,
						headers: { "Content-Type": "application/json" },
					});
					clearTimeout(timeoutId);
					return res;
				}
			} catch (sdkSseError) {
				try { await client?.close(); } catch { /* ignore */ }
			}

			// Both SSE approaches failed
			{
				const sseError = new Error("SSE transport failed (raw + SDK)");
				lastError = sseError;
				// Prefer the HTTP error when both failed so UI shows the primary failure (e.g., HTTP 500) instead
				// of the fallback SSE message.
				if (httpError) {
					lastError = new Error(
						`HTTP transport failed: ${httpError.message}; SSE fallback failed: ${lastError.message}`,
						{ cause: sseError instanceof Error ? sseError : undefined }
					);
				}
				logger.error(lastError, "Both transports failed.");
			}
		}

		// Both transports failed
		let errorMessage = lastError?.message || "Failed to connect to MCP server";

		// Detect unauthorized to signal auth requirement
		const lower = (errorMessage || "").toLowerCase();
		const authRequired =
			lower.includes("unauthorized") ||
			lower.includes("forbidden") ||
			lower.includes("401") ||
			lower.includes("403");

		// Provide more helpful error messages
		if (authRequired) {
			errorMessage =
				"Authentication required. Provide appropriate Authorization headers in the server configuration.";
		} else if (errorMessage.includes("not valid JSON")) {
			errorMessage =
				"Server returned invalid response. This might not be a valid MCP endpoint. MCP servers should respond to POST requests at /mcp with JSON-RPC messages.";
		} else if (errorMessage.includes("fetch failed") || errorMessage.includes("ECONNREFUSED")) {
			errorMessage = `Cannot connect to ${url}. Please verify the server is running and accessible.`;
		} else if (errorMessage.includes("CORS")) {
			errorMessage = `CORS error. The MCP server needs to allow requests from this origin.`;
		}

		const res = new Response(
			JSON.stringify({
				ready: false,
				error: errorMessage,
				authRequired,
			} as HealthCheckResponse),
			{
				status: 503,
				headers: { "Content-Type": "application/json" },
			}
		);
		clearTimeout(timeoutId);
		return res;
	} catch (error) {
		logger.error(error, "MCP health check failed");

		// Clean up client if it exists
		try {
			await client?.close();
		} catch {
			// Ignore
		}

		const response: HealthCheckResponse = {
			ready: false,
			error: error instanceof Error ? error.message : "Unknown error",
		};

		const res = new Response(JSON.stringify(response), {
			status: 503,
			headers: { "Content-Type": "application/json" },
		});
		return res;
	}
};
