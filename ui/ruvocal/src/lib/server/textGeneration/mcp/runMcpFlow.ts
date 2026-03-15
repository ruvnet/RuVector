import { config } from "$lib/server/config";
import { MessageUpdateType, type MessageUpdate } from "$lib/types/MessageUpdate";
import { getMcpServers } from "$lib/server/mcp/registry";
import { isValidUrl } from "$lib/server/urlSafety";
import { resetMcpToolsCache, type McpToolMapping } from "$lib/server/mcp/tools";
import { getOpenAiToolsForMcp } from "$lib/server/mcp/tools";
import type {
	ChatCompletionChunk,
	ChatCompletionCreateParamsStreaming,
	ChatCompletionMessageParam,
	ChatCompletionMessageToolCall,
} from "openai/resources/chat/completions";
import type { Stream } from "openai/streaming";
import { buildToolPreprompt } from "../utils/toolPrompt";
import type { EndpointMessage } from "../../endpoints/endpoints";
import { resolveRouterTarget } from "./routerResolution";
import { executeToolCalls, type NormalizedToolCall } from "./toolInvocation";
import { drainPool } from "$lib/server/mcp/clientPool";
import type { TextGenerationContext } from "../types";
import {
	hasAuthHeader,
	isStrictHfMcpLogin,
	hasNonEmptyToken,
	isExaMcpServer,
} from "$lib/server/mcp/hf";
import { buildImageRefResolver } from "./fileRefs";
import { prepareMessagesWithFiles } from "$lib/server/textGeneration/utils/prepareFiles";
import { makeImageProcessor } from "$lib/server/endpoints/images";
import { logger } from "$lib/server/logger";
import { AbortedGenerations } from "$lib/server/abortedGenerations";

export type RunMcpFlowContext = Pick<
	TextGenerationContext,
	"model" | "conv" | "assistant" | "forceMultimodal" | "forceTools" | "provider" | "locals"
> & { messages: EndpointMessage[] };

// Return type: "completed" = MCP ran successfully, "not_applicable" = MCP didn't run, "aborted" = user aborted
export type McpFlowResult = "completed" | "not_applicable" | "aborted";

export async function* runMcpFlow({
	model,
	conv,
	messages,
	assistant,
	forceMultimodal,
	forceTools,
	provider,
	locals,
	preprompt,
	abortSignal,
	abortController,
	promptedAt,
	autopilot,
	autopilotMaxSteps,
}: RunMcpFlowContext & {
	preprompt?: string;
	abortSignal?: AbortSignal;
	abortController?: AbortController;
	promptedAt?: Date;
	autopilot?: boolean;
	autopilotMaxSteps?: number;
}): AsyncGenerator<MessageUpdate, McpFlowResult, undefined> {
	// Helper to check if generation should be aborted via DB polling
	// Also triggers the abort controller to cancel active streams/requests
	const checkAborted = (): boolean => {
		if (abortSignal?.aborted) return true;
		const abortTime = AbortedGenerations.getInstance().getAbortTime(conv._id.toString());
		if (abortTime && promptedAt && abortTime > promptedAt) {
			// Trigger the abort controller to cancel active streams
			if (abortController && !abortController.signal.aborted) {
				abortController.abort();
			}
			return true;
		}
		return false;
	};
	// Start from env-configured servers
	let servers = getMcpServers();
	try {
		logger.debug(
			{ baseServers: servers.map((s) => ({ name: s.name, url: s.url })), count: servers.length },
			"[mcp] base servers loaded"
		);
	} catch {}

	// Merge in request-provided custom servers (if any)
	try {
		const reqMcp = (
			locals as unknown as {
				mcp?: {
					selectedServers?: Array<{ name: string; url: string; headers?: Record<string, string> }>;
					selectedServerNames?: string[];
				};
			}
		)?.mcp;
		const custom = Array.isArray(reqMcp?.selectedServers) ? reqMcp?.selectedServers : [];
		if (custom.length > 0) {
			// Invalidate cached tool list when the set of servers changes at request-time
			resetMcpToolsCache();
			// Deduplicate by server name (request takes precedence)
			const byName = new Map<
				string,
				{ name: string; url: string; headers?: Record<string, string> }
			>();
			for (const s of servers) byName.set(s.name, s);
			for (const s of custom) byName.set(s.name, s);
			servers = [...byName.values()];
			try {
				logger.debug(
					{
						customProvidedCount: custom.length,
						mergedServers: servers.map((s) => ({
							name: s.name,
							url: s.url,
							hasAuth: !!s.headers?.Authorization,
						})),
					},
					"[mcp] merged request-provided servers"
				);
			} catch {}
		}

		// If the client specified a selection by name, filter to those
		const names = Array.isArray(reqMcp?.selectedServerNames)
			? reqMcp?.selectedServerNames
			: undefined;
		if (Array.isArray(names)) {
			const before = servers.map((s) => s.name);
			servers = servers.filter((s) => names.includes(s.name));
			try {
				logger.debug(
					{ selectedNames: names, before, after: servers.map((s) => s.name) },
					"[mcp] applied name selection"
				);
			} catch {}
		}
	} catch {
		// ignore selection merge errors and proceed with env servers
	}

	// Extract WASM tools early to check if we should continue even without HTTP servers
	const reqMcpForWasm = (
		locals as unknown as {
			mcp?: {
				wasmTools?: Array<{
					name: string;
					description?: string;
					inputSchema?: Record<string, unknown>;
					serverId: string;
				}>;
			};
		}
	)?.mcp;
	const wasmToolsFromClient = Array.isArray(reqMcpForWasm?.wasmTools) ? reqMcpForWasm.wasmTools : [];
	// Always have WASM tools available (default file tools are added server-side)
	const hasWasmTools = true;

	if (wasmToolsFromClient.length > 0) {
		logger.info(
			{ wasmToolCount: wasmToolsFromClient.length, wasmToolNames: wasmToolsFromClient.map((t) => t.name) },
			"[mcp] WASM tools detected from client"
		);
	}

	// If selection/merge yielded no servers, bail early UNLESS we have WASM tools
	if (servers.length === 0 && !hasWasmTools) {
		logger.warn({}, "[mcp] no MCP servers selected after merge/name filter and no WASM tools");
		return "not_applicable";
	}

	// Enforce server-side safety (public HTTPS only, no private ranges)
	{
		const before = servers.slice();
		servers = servers.filter((s) => {
			try {
				return isValidUrl(s.url);
			} catch {
				return false;
			}
		});
		try {
			const rejected = before.filter((b) => !servers.includes(b));
			if (rejected.length > 0) {
				logger.warn(
					{ rejected: rejected.map((r) => ({ name: r.name, url: r.url })) },
					"[mcp] rejected servers by URL safety"
				);
			}
		} catch {}
	}
	// Only return early if no HTTP servers AND no WASM tools
	if (servers.length === 0 && !hasWasmTools) {
		logger.warn({}, "[mcp] all selected MCP servers rejected by URL safety guard and no WASM tools");
		return "not_applicable";
	}

	// Optionally attach the logged-in user's HF token to the official HF MCP server only.
	// Never override an explicit Authorization header, and require token to look like an HF token.
	try {
		const shouldForward = config.MCP_FORWARD_HF_USER_TOKEN === "true";
		const userToken =
			(locals as unknown as { hfAccessToken?: string } | undefined)?.hfAccessToken ??
			(locals as unknown as { token?: string } | undefined)?.token;

		if (shouldForward && hasNonEmptyToken(userToken)) {
			const overlayApplied: string[] = [];
			servers = servers.map((s) => {
				try {
					if (isStrictHfMcpLogin(s.url) && !hasAuthHeader(s.headers)) {
						overlayApplied.push(s.name);
						return {
							...s,
							headers: { ...(s.headers ?? {}), Authorization: `Bearer ${userToken}` },
						};
					}
				} catch {
					// ignore URL parse errors and leave server unchanged
				}
				return s;
			});
			if (overlayApplied.length > 0) {
				try {
					logger.debug({ overlayApplied }, "[mcp] forwarded HF token to servers");
				} catch {}
			}
		}
	} catch {
		// best-effort overlay; continue if anything goes wrong
	}

	// Inject Exa API key for mcp.exa.ai servers via URL param (mcp.exa.ai doesn't support headers)
	try {
		const exaApiKey = config.EXA_API_KEY;
		if (hasNonEmptyToken(exaApiKey)) {
			const overlayApplied: string[] = [];
			servers = servers.map((s) => {
				try {
					if (isExaMcpServer(s.url)) {
						const url = new URL(s.url);
						if (!url.searchParams.has("exaApiKey")) {
							url.searchParams.set("exaApiKey", exaApiKey);
							overlayApplied.push(s.name);
							return { ...s, url: url.toString() };
						}
					}
				} catch {}
				return s;
			});
			if (overlayApplied.length > 0) {
				logger.debug({ overlayApplied }, "[mcp] injected Exa API key to servers");
			}
		}
	} catch {
		// best-effort injection; continue if anything goes wrong
	}

	logger.debug(
		{ count: servers.length, servers: servers.map((s) => s.name), hasWasmTools },
		"[mcp] servers configured"
	);
	// Only return if no HTTP servers AND no WASM tools
	if (servers.length === 0 && !hasWasmTools) {
		return "not_applicable";
	}

	// Gate MCP flow based on model tool support (aggregated) with user override
	// If WASM tools exist, force tools enabled
	try {
		const supportsTools = Boolean((model as unknown as { supportsTools?: boolean }).supportsTools);
		const toolsEnabled = Boolean(forceTools) || supportsTools || hasWasmTools;
		logger.debug(
			{
				model: model.id ?? model.name,
				supportsTools,
				forceTools: Boolean(forceTools),
				hasWasmTools,
				toolsEnabled,
			},
			"[mcp] tools gate evaluation"
		);
		if (!toolsEnabled) {
			logger.info(
				{ model: model.id ?? model.name },
				"[mcp] tools disabled for model; skipping MCP flow"
			);
			return "not_applicable";
		}
	} catch {
		// If anything goes wrong reading the flag, proceed (previous behavior)
	}

	const resolveFileRef = buildImageRefResolver(messages);
	const imageProcessor = makeImageProcessor({
		supportedMimeTypes: ["image/png", "image/jpeg"],
		preferredMimeType: "image/jpeg",
		maxSizeInMB: 1,
		maxWidth: 1024,
		maxHeight: 1024,
	});

	const hasImageInput = messages.some((msg) =>
		(msg.files ?? []).some(
			(file) => typeof file?.mime === "string" && file.mime.startsWith("image/")
		)
	);

	const { runMcp, targetModel, candidateModelId, resolvedRoute } = await resolveRouterTarget({
		model,
		messages,
		conversationId: conv._id.toString(),
		hasImageInput,
		locals,
	});

	// If WASM tools exist, force runMcp even if router says no
	if (!runMcp && !hasWasmTools) {
		logger.info(
			{ model: targetModel.id ?? targetModel.name, resolvedRoute },
			"[mcp] runMcp=false (routing chose non-tools candidate) and no WASM tools"
		);
		return "not_applicable";
	}
	if (!runMcp && hasWasmTools) {
		logger.info(
			{ model: targetModel.id ?? targetModel.name, hasWasmTools },
			"[mcp] runMcp=false but WASM tools present, forcing MCP flow"
		);
	}

	try {
		const { tools: oaTools, mapping } = await getOpenAiToolsForMcp(servers, {
			signal: abortSignal,
		});

		// ================================
		// rvAgent WASM Tools - Full Implementation
		// 17 tools with detailed descriptions for better LLM guidance
		// ================================
		const defaultWasmTools = [
			// ========== SYSTEM GUIDANCE (1 tool) ==========
			// ALWAYS call this first to understand ALL available tools
			{
				name: "system_guidance",
				description: `system_guidance() → CALL THIS FIRST. Get help on ALL available tools and how to use them.
Arguments: {} for full guide, or {"tool": "tool_name"} for specific tool help.
Examples: {"tool": "read_file"}, {"tool": "brain_search"}, {"tool": "gallery_load"}`,
				inputSchema: {
					type: "object",
					properties: {
						tool: { type: "string" },
						category: { type: "string", enum: ["all", "files", "memory", "tasks", "search", "brain", "gallery", "witness"] },
					},
				},
			},
			// ========== FILE OPERATIONS (5 tools) ==========
			// Use these to work with files in the virtual filesystem
			{
				name: "read_file",
				description: `read_file(path: string) → Read file contents.
Arguments: {"path": "filename.txt"}
Example: {"path": "src/index.ts"}`,
				inputSchema: {
					type: "object",
					properties: {
						path: { type: "string" },
					},
					required: ["path"],
				},
			},
			{
				name: "write_file",
				description: `write_file(path: string, content: string) → Create/overwrite file.
Arguments: {"path": "file.txt", "content": "text"}
Example: {"path": "hello.txt", "content": "Hello World"}`,
				inputSchema: {
					type: "object",
					properties: {
						path: { type: "string" },
						content: { type: "string" },
					},
					required: ["path", "content"],
				},
			},
			{
				name: "list_files",
				description: `list_files() → List all files in virtual filesystem.
Arguments: {}`,
				inputSchema: {
					type: "object",
					properties: {},
				},
			},
			{
				name: "delete_file",
				description: `delete_file(path: string) → Delete a file.
Arguments: {"path": "filename.txt"}
Example: {"path": "temp.txt"}`,
				inputSchema: {
					type: "object",
					properties: {
						path: { type: "string" },
					},
					required: ["path"],
				},
			},
			{
				name: "edit_file",
				description: `edit_file(path: string, old_content: string, new_content: string) → Replace text in file.
Arguments: {"path": "file.txt", "old_content": "old", "new_content": "new"}
Example: {"path": "config.json", "old_content": "v1", "new_content": "v2"}`,
				inputSchema: {
					type: "object",
					properties: {
						path: { type: "string" },
						old_content: { type: "string" },
						new_content: { type: "string" },
					},
					required: ["path", "old_content", "new_content"],
				},
			},

			// ========== SEARCH TOOLS (2 tools) ==========
			// Use these to find content or files
			{
				name: "grep",
				description: `grep(pattern: string, path?: string) → Search files for regex pattern.
Arguments: {"pattern": "search_term"}
Example: {"pattern": "TODO"} or {"pattern": "function", "path": "src/app.ts"}`,
				inputSchema: {
					type: "object",
					properties: {
						pattern: { type: "string" },
						path: { type: "string" },
					},
					required: ["pattern"],
				},
			},
			{
				name: "glob",
				description: `glob(pattern: string) → Find files matching pattern.
Arguments: {"pattern": "*.ts"}
Example: {"pattern": "src/*.js"}`,
				inputSchema: {
					type: "object",
					properties: {
						pattern: { type: "string" },
					},
					required: ["pattern"],
				},
			},
			{
				name: "todo_add",
				description: `todo_add(task: string) → Add a task to the list.
Arguments: {"task": "description"}
Example: {"task": "Write unit tests"}`,
				inputSchema: {
					type: "object",
					properties: {
						task: { type: "string" },
					},
					required: ["task"],
				},
			},
			{
				name: "todo_list",
				description: `todo_list() → List all tasks.
Arguments: {}`,
				inputSchema: {
					type: "object",
					properties: {},
				},
			},
			{
				name: "todo_complete",
				description: `todo_complete(id: string) → Mark task as done.
Arguments: {"id": "todo-1"}
Example: {"id": "todo-1"}`,
				inputSchema: {
					type: "object",
					properties: {
						id: { type: "string" },
					},
					required: ["id"],
				},
			},
			{
				name: "memory_store",
				description: `memory_store(key: string, value: string) → Store key-value pair.
Arguments: {"key": "name", "value": "data"}
Example: {"key": "auth-method", "value": "JWT tokens"}`,
				inputSchema: {
					type: "object",
					properties: {
						key: { type: "string" },
						value: { type: "string" },
						tags: { type: "array", items: { type: "string" } },
					},
					required: ["key", "value"],
				},
			},
			{
				name: "memory_search",
				description: `memory_search(query: string) → Search stored memories.
Arguments: {"query": "search_term"}
Example: {"query": "authentication"}`,
				inputSchema: {
					type: "object",
					properties: {
						query: { type: "string" },
						top_k: { type: "number" },
					},
					required: ["query"],
				},
			},
			{
				name: "witness_log",
				description: `witness_log(action: string) → Log to audit chain.
Arguments: {"action": "action_name"}
Example: {"action": "file_modified"}`,
				inputSchema: {
					type: "object",
					properties: {
						action: { type: "string" },
						data: { type: "object" },
					},
					required: ["action"],
				},
			},
			{
				name: "witness_verify",
				description: `witness_verify() → Verify audit chain integrity.
Arguments: {}`,
				inputSchema: {
					type: "object",
					properties: {},
				},
			},
			{
				name: "gallery_list",
				description: `gallery_list() → List all agent templates.
Arguments: {} or {"category": "security"}`,
				inputSchema: {
					type: "object",
					properties: {
						category: { type: "string" },
					},
				},
			},
			{
				name: "gallery_load",
				description: `gallery_load(id: string) → Load an agent template.
Arguments: {"id": "template-name"}
IDs: development-agent, research-agent, security-agent, minimal-agent
Example: {"id": "development-agent"}`,
				inputSchema: {
					type: "object",
					properties: {
						id: {
							type: "string",
							enum: ["development-agent", "research-agent", "security-agent", "multi-agent-orchestrator", "sona-learning-agent", "agi-container-builder", "witness-auditor", "minimal-agent"]
						},
					},
					required: ["id"],
				},
			},
			{
				name: "gallery_search",
				description: `gallery_search(query: string) → Search templates by keyword.
Arguments: {"query": "search_term"}
Example: {"query": "security"}`,
				inputSchema: {
					type: "object",
					properties: {
						query: { type: "string" },
					},
					required: ["query"],
				},
			},
		];

		// Combine client-provided WASM tools with default WASM tools
		const allWasmTools = [...wasmToolsFromClient];
		for (const dt of defaultWasmTools) {
			if (!allWasmTools.some((wt) => wt.name === dt.name)) {
				allWasmTools.push({
					name: dt.name,
					description: dt.description,
					inputSchema: dt.inputSchema,
					serverId: "__wasm__",
				});
			}
		}

		// Add WASM tools (default + client-provided)
		const wasmToolMapping: Record<string, McpToolMapping> = {};
		try {
			for (const wt of allWasmTools) {
				const fnName = wt.name.replace(/[^a-zA-Z0-9_-]/g, "_").slice(0, 64);
				// Avoid collision with server tools
				if (!(fnName in mapping)) {
					oaTools.push({
						type: "function",
						function: {
							name: fnName,
							description: wt.description ?? `File tool: ${wt.name}`,
							parameters: wt.inputSchema,
						},
					});
					wasmToolMapping[fnName] = {
						fnName,
						server: "__wasm__",
						tool: wt.name,
					};
					mapping[fnName] = wasmToolMapping[fnName];
				}
			}
			logger.info(
				{ wasmToolCount: allWasmTools.length, wasmTools: allWasmTools.map((t) => t.name) },
				"[mcp] added WASM file tools"
			);
		} catch (e) {
			logger.debug({ error: e }, "[mcp] failed to add WASM tools");
		}

		try {
			logger.info(
				{ toolCount: oaTools.length, toolNames: oaTools.map((t) => t.function.name) },
				"[mcp] openai tool defs built"
			);
		} catch {}
		if (oaTools.length === 0) {
			logger.warn({}, "[mcp] zero tools available after listing; skipping MCP flow");
			return "not_applicable";
		}

		const { OpenAI } = await import("openai");

		// Capture provider header (x-inference-provider) from the upstream OpenAI-compatible server.
		let providerHeader: string | undefined;
		const captureProviderFetch = async (
			input: RequestInfo | URL,
			init?: RequestInit
		): Promise<Response> => {
			const res = await fetch(input, init);
			const p = res.headers.get("x-inference-provider");
			if (p && !providerHeader) providerHeader = p;
			return res;
		};

		const openai = new OpenAI({
			apiKey: config.OPENAI_API_KEY || config.HF_TOKEN || "sk-",
			baseURL: config.OPENAI_BASE_URL,
			fetch: captureProviderFetch,
			defaultHeaders: {
				// Bill to organization if configured (HuggingChat only)
				...(config.isHuggingChat && locals?.billingOrganization
					? { "X-HF-Bill-To": locals.billingOrganization }
					: {}),
			},
		});

		const mmEnabled = (forceMultimodal ?? false) || targetModel.multimodal;
		logger.info(
			{
				targetModel: targetModel.id ?? targetModel.name,
				mmEnabled,
				route: resolvedRoute,
				candidateModelId,
				toolCount: oaTools.length,
				hasUserToken: Boolean((locals as unknown as { token?: string })?.token),
			},
			"[mcp] starting completion with tools"
		);
		let messagesOpenAI: ChatCompletionMessageParam[] = await prepareMessagesWithFiles(
			messages,
			imageProcessor,
			mmEnabled
		);
		const toolPreprompt = buildToolPreprompt(oaTools, autopilot);
		const prepromptPieces: string[] = [];
		if (toolPreprompt.trim().length > 0) {
			prepromptPieces.push(toolPreprompt);
		}
		if (typeof preprompt === "string" && preprompt.trim().length > 0) {
			prepromptPieces.push(preprompt);
		}
		const mergedPreprompt = prepromptPieces.join("\n\n");
		const hasSystemMessage = messagesOpenAI.length > 0 && messagesOpenAI[0]?.role === "system";
		if (hasSystemMessage) {
			if (mergedPreprompt.length > 0) {
				const existing = messagesOpenAI[0].content ?? "";
				const existingText = typeof existing === "string" ? existing : "";
				messagesOpenAI[0].content = mergedPreprompt + (existingText ? "\n\n" + existingText : "");
			}
		} else if (mergedPreprompt.length > 0) {
			messagesOpenAI = [{ role: "system", content: mergedPreprompt }, ...messagesOpenAI];
		}

		// Work around servers that reject `system` role
		if (
			typeof config.OPENAI_BASE_URL === "string" &&
			config.OPENAI_BASE_URL.length > 0 &&
			(config.OPENAI_BASE_URL.includes("hf.space") ||
				config.OPENAI_BASE_URL.includes("gradio.app")) &&
			messagesOpenAI[0]?.role === "system"
		) {
			messagesOpenAI[0] = { ...messagesOpenAI[0], role: "user" };
		}

		const parameters = { ...targetModel.parameters, ...assistant?.generateSettings } as Record<
			string,
			unknown
		>;
		const maxTokens =
			(parameters?.max_tokens as number | undefined) ??
			(parameters?.max_new_tokens as number | undefined) ??
			(parameters?.max_completion_tokens as number | undefined);

		const stopSequences =
			typeof parameters?.stop === "string"
				? parameters.stop
				: Array.isArray(parameters?.stop)
					? (parameters.stop as string[])
					: undefined;

		// Build model ID with optional provider suffix (e.g., "model:fastest" or "model:together")
		// Strip "models/" prefix for Gemini OpenAI-compatible API
		// Gemini's /models endpoint returns "models/gemini-2.5-flash" but
		// the chat completions API expects just "gemini-2.5-flash"
		let baseModelId = targetModel.id ?? targetModel.name;
		if (baseModelId.startsWith("models/")) {
			baseModelId = baseModelId.replace(/^models\//, "");
			logger.debug({ original: targetModel.id, stripped: baseModelId }, "[mcp] stripped models/ prefix from model ID");
		}
		const modelIdWithProvider =
			provider && provider !== "auto" ? `${baseModelId}:${provider}` : baseModelId;

		const completionBase: Omit<ChatCompletionCreateParamsStreaming, "messages"> = {
			model: modelIdWithProvider,
			stream: true,
			temperature: typeof parameters?.temperature === "number" ? parameters.temperature : undefined,
			top_p: typeof parameters?.top_p === "number" ? parameters.top_p : undefined,
			frequency_penalty:
				typeof parameters?.frequency_penalty === "number"
					? parameters.frequency_penalty
					: typeof parameters?.repetition_penalty === "number"
						? parameters.repetition_penalty
						: undefined,
			presence_penalty:
				typeof parameters?.presence_penalty === "number" ? parameters.presence_penalty : undefined,
			stop: stopSequences,
			max_tokens: typeof maxTokens === "number" ? maxTokens : undefined,
			tools: oaTools,
			tool_choice: "auto",
		};
		logger.info({ model: modelIdWithProvider, toolCount: oaTools.length, toolNames: oaTools.slice(0, 5).map(t => t.function?.name) }, "[mcp] completion base config");

		const toPrimitive = (value: unknown) => {
			if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
				return value;
			}
			return undefined;
		};

		const parseArgs = (raw: unknown): Record<string, unknown> => {
			if (typeof raw !== "string" || raw.trim().length === 0) return {};
			try {
				return JSON.parse(raw);
			} catch {
				return {};
			}
		};

		const processToolOutput = (
			text: string
		): {
			annotated: string;
			sources: { index: number; link: string }[];
		} => ({ annotated: text, sources: [] });

		let lastAssistantContent = "";
		let streamedContent = false;
		// Track whether we're inside a <think> block when the upstream streams
		// provider-specific reasoning tokens (e.g. `reasoning` or `reasoning_content`).
		let thinkOpen = false;

		if (resolvedRoute && candidateModelId) {
			yield {
				type: MessageUpdateType.RouterMetadata,
				route: resolvedRoute,
				model: candidateModelId,
			};
			logger.debug(
				{ route: resolvedRoute, model: candidateModelId },
				"[mcp] router metadata emitted"
			);
		}

		// Use configurable max steps (default: 10 for autopilot, 5 for non-autopilot)
		// Clamp to 1-50 range for safety
		const configuredMax = Math.min(50, Math.max(1, autopilotMaxSteps ?? 10));
		const maxLoops = autopilot ? configuredMax : Math.min(configuredMax, 5);
		logger.info({ autopilot, maxLoops }, "[mcp] starting loop with autopilot setting");
		for (let loop = 0; loop < maxLoops; loop += 1) {
			logger.info({ loop, autopilot, maxLoops }, "[mcp] === LOOP ITERATION START ===");
			// Check for abort at the start of each loop iteration
			if (checkAborted()) {
				logger.info({ loop }, "[mcp] aborting at start of loop iteration");
				return "aborted";
			}

			lastAssistantContent = "";
			streamedContent = false;

			// Gemini's OpenAI-compatible API doesn't properly support role: "tool" messages.
			// Transform tool result messages to role: "user" format for Gemini compatibility.
			// See: https://discuss.ai.google.dev/t/gemini-api-returns-an-error-when-trying-to-pass-tool-call-results-with-role-tool/64336
			const isGeminiModel = baseModelId.includes("gemini");
			let finalMessages = messagesOpenAI;

			if (isGeminiModel && loop > 0) {
				// Transform messages for Gemini: merge assistant tool_calls + tool results into user message
				finalMessages = [];
				let i = 0;
				while (i < messagesOpenAI.length) {
					const msg = messagesOpenAI[i];
					const msgAny = msg as unknown as Record<string, unknown>;

					// Check if this is an assistant message with tool_calls
					if (msg.role === "assistant" && msgAny.tool_calls) {
						const toolCalls = msgAny.tool_calls as Array<{ id: string; function: { name: string; arguments: string } }>;

						// Collect all following tool result messages
						const toolResults: Array<{ call_id: string; name: string; result: string }> = [];
						let j = i + 1;
						while (j < messagesOpenAI.length && messagesOpenAI[j].role === "tool") {
							const toolMsg = messagesOpenAI[j] as unknown as { tool_call_id: string; content: string };
							const matchingCall = toolCalls.find(tc => tc.id === toolMsg.tool_call_id);
							toolResults.push({
								call_id: toolMsg.tool_call_id,
								name: matchingCall?.function?.name ?? "unknown",
								result: String(toolMsg.content),
							});
							j++;
						}

						// Convert to Gemini-compatible format: user message with structured tool results
						if (toolResults.length > 0) {
							// Keep assistant message content but remove tool_calls
							const assistantContent = String(msgAny.content || "").trim();
							if (assistantContent) {
								finalMessages.push({ role: "assistant", content: assistantContent });
							}

							// Add tool results as a user message (Gemini workaround)
							const toolResultContent = toolResults.map(tr =>
								`[Tool Result: ${tr.name}]\n${tr.result}`
							).join("\n\n");
							finalMessages.push({ role: "user", content: toolResultContent });

							i = j; // Skip past the tool messages we processed
							continue;
						}
					}

					// Keep non-tool messages as-is (but skip role: "tool" if any remain)
					if (msg.role !== "tool") {
						finalMessages.push(msg);
					}
					i++;
				}

				logger.info({ originalCount: messagesOpenAI.length, transformedCount: finalMessages.length }, "[mcp] Gemini: transformed tool messages");
			}

			const completionRequest: ChatCompletionCreateParamsStreaming = {
				...completionBase,
				messages: finalMessages,
			};

			const completionStream: Stream<ChatCompletionChunk> = await openai.chat.completions.create(
				completionRequest,
				{
					signal: abortSignal,
					headers: {
						"ChatUI-Conversation-ID": conv._id.toString(),
						"X-use-cache": "false",
						...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
					},
				}
			);

			// If provider header was exposed, notify UI so it can render "via {provider}".
			if (providerHeader) {
				yield {
					type: MessageUpdateType.RouterMetadata,
					route: "",
					model: "",
					provider: providerHeader as unknown as import("@huggingface/inference").InferenceProvider,
				};
				logger.debug({ provider: providerHeader }, "[mcp] provider metadata emitted");
			}

			const toolCallState: Record<number, { id?: string; name?: string; arguments: string }> = {};
			let firstToolDeltaLogged = false;
			let sawToolCall = false;
			let tokenCount = 0;
			let chunkCount = 0;
			for await (const chunk of completionStream) {
				chunkCount++;
				const choice = chunk.choices?.[0];
				const delta = choice?.delta;
				if (!delta) continue;

				const chunkToolCalls = delta.tool_calls ?? [];
				// Log raw delta for debugging Gemini tool call format
				if (chunkToolCalls.length > 0 || (delta as Record<string, unknown>).functionCall) {
					logger.info({
						rawDelta: JSON.stringify(delta).slice(0, 500),
						toolCallsLength: chunkToolCalls.length,
						hasFunctionCall: !!(delta as Record<string, unknown>).functionCall
					}, "[mcp] raw streaming delta with tool info");
				}
				// Handle Gemini's native functionCall format (not OpenAI tool_calls)
				const geminiFC = (delta as Record<string, unknown>).functionCall as { name?: string; args?: Record<string, unknown> } | undefined;
				if (geminiFC?.name) {
					sawToolCall = true;
					const current = toolCallState[0] ?? { arguments: "" };
					current.name = geminiFC.name;
					if (geminiFC.args) {
						current.arguments = JSON.stringify(geminiFC.args);
					}
					current.id = current.id || `gemini_${Date.now()}`;
					toolCallState[0] = current;
					logger.info({ name: geminiFC.name, args: geminiFC.args }, "[mcp] Gemini native functionCall detected");
				}
				if (chunkToolCalls.length > 0) {
					sawToolCall = true;
					for (const call of chunkToolCalls) {
						const toolCall = call as unknown as {
							index?: number;
							id?: string;
							function?: { name?: string; arguments?: string | Record<string, unknown> };
						};
						const index = toolCall.index ?? 0;
						const current = toolCallState[index] ?? { arguments: "" };
						if (toolCall.id) current.id = toolCall.id;
						if (toolCall.function?.name) current.name = toolCall.function.name;
						// Handle arguments as either string or object (Gemini may send objects)
						const rawArgs = toolCall.function?.arguments;
						if (rawArgs) {
							if (typeof rawArgs === "string") {
								current.arguments += rawArgs;
							} else if (typeof rawArgs === "object") {
								// Gemini sends args as object - stringify it
								current.arguments = JSON.stringify(rawArgs);
								logger.info({ argsType: "object", args: rawArgs }, "[mcp] tool_call arguments received as object");
							}
						}
						toolCallState[index] = current;
						logger.debug({ index, id: toolCall.id, name: toolCall.function?.name, argsChunk: typeof rawArgs === "string" ? rawArgs?.slice(0, 100) : JSON.stringify(rawArgs)?.slice(0, 100) }, "[mcp] tool_call chunk processed");
					}
					if (!firstToolDeltaLogged) {
						try {
							const first =
								toolCallState[
									Object.keys(toolCallState)
										.map((k) => Number(k))
										.sort((a, b) => a - b)[0] ?? 0
								];
							logger.info(
								{ firstCallName: first?.name, hasId: Boolean(first?.id) },
								"[mcp] observed streamed tool_call delta"
							);
							firstToolDeltaLogged = true;
						} catch {}
					}
				}

				const deltaContent = (() => {
					if (typeof delta.content === "string") return delta.content;
					const maybeParts = delta.content as unknown;
					if (Array.isArray(maybeParts)) {
						return maybeParts
							.map((part) =>
								typeof part === "object" &&
								part !== null &&
								"text" in part &&
								typeof (part as Record<string, unknown>).text === "string"
									? String((part as Record<string, unknown>).text)
									: ""
							)
							.join("");
					}
					return "";
				})();

				// Provider-dependent reasoning fields (e.g., `reasoning` or `reasoning_content`).
				const deltaReasoning: string =
					typeof (delta as unknown as Record<string, unknown>)?.reasoning === "string"
						? ((delta as unknown as { reasoning?: string }).reasoning as string)
						: typeof (delta as unknown as Record<string, unknown>)?.reasoning_content === "string"
							? ((delta as unknown as { reasoning_content?: string }).reasoning_content as string)
							: "";

				// Merge reasoning + content into a single combined token stream, mirroring
				// the OpenAI adapter so the UI can auto-detect <think> blocks.
				let combined = "";
				if (deltaReasoning.trim().length > 0) {
					if (!thinkOpen) {
						combined += "<think>" + deltaReasoning;
						thinkOpen = true;
					} else {
						combined += deltaReasoning;
					}
				}

				if (deltaContent && deltaContent.length > 0) {
					if (thinkOpen) {
						combined += "</think>" + deltaContent;
						thinkOpen = false;
					} else {
						combined += deltaContent;
					}
				}

				if (combined.length > 0) {
					lastAssistantContent += combined;
					if (!sawToolCall) {
						streamedContent = true;
						yield { type: MessageUpdateType.Stream, token: combined };
						tokenCount += combined.length;
					}
				}

				// Periodic abort check during streaming
				if (checkAborted()) {
					logger.info({ loop, tokenCount }, "[mcp] aborting during stream");
					return "aborted";
				}
			}
			logger.info(
				{ sawToolCalls: Object.keys(toolCallState).length > 0, toolCallCount: Object.keys(toolCallState).length, tokens: tokenCount, loop, autopilot },
				"[mcp] completion stream closed"
			);

			// Check abort after stream completes
			if (checkAborted()) {
				logger.info({ loop }, "[mcp] aborting after stream completed");
				return "aborted";
			}

			// Auto-close any unclosed <think> block so reasoning from this loop
			// doesn't swallow content from subsequent iterations.  The client-side
			// regex matches `<think>` to end-of-string, so an unclosed block would
			// hide everything that follows.
			if (thinkOpen) {
				if (streamedContent) {
					yield { type: MessageUpdateType.Stream, token: "</think>" };
				}
				lastAssistantContent += "</think>";
				thinkOpen = false;
			}

			if (Object.keys(toolCallState).length > 0) {
				logger.info({
					toolCallState: Object.entries(toolCallState).map(([idx, c]) => ({
						index: idx,
						id: c?.id ?? "(missing)",
						name: c?.name ?? "(missing)",
						argsPreview: (c?.arguments ?? "").slice(0, 200)
					}))
				}, "[mcp] streaming tool calls accumulated");
				// If any streamed call is missing id, perform a quick non-stream retry to recover full tool_calls with ids
				const missingId = Object.values(toolCallState).some((c) => c?.name && !c?.id);
				let calls: NormalizedToolCall[];
				if (missingId) {
					logger.debug(
						{ loop },
						"[mcp] missing tool_call id in stream; retrying non-stream to recover ids"
					);
					const nonStream = await openai.chat.completions.create(
						{ ...completionBase, messages: messagesOpenAI, stream: false },
						{
							signal: abortSignal,
							headers: {
								"ChatUI-Conversation-ID": conv._id.toString(),
								"X-use-cache": "false",
								...(locals?.token ? { Authorization: `Bearer ${locals.token}` } : {}),
							},
						}
					);
					const rawMessage = nonStream.choices?.[0]?.message as unknown as Record<string, unknown>;
					// Log full raw message to see Gemini's actual format
					logger.info({
						rawMessageKeys: Object.keys(rawMessage || {}),
						rawMessageJson: JSON.stringify(rawMessage).slice(0, 1000),
						finishReason: nonStream.choices?.[0]?.finish_reason
					}, "[mcp] non-stream FULL raw message from API");

					// Check for Gemini's native functionCall format
					const geminiFunctionCall = rawMessage?.functionCall as { name?: string; args?: Record<string, unknown> } | undefined;
					let tc = nonStream.choices?.[0]?.message?.tool_calls ?? [];

					// If no tool_calls but has functionCall (Gemini native format)
					if (tc.length === 0 && geminiFunctionCall?.name) {
						logger.info({ geminiFunctionCall }, "[mcp] using Gemini native functionCall format");
						tc = [{
							id: `gemini_${Date.now()}`,
							type: "function" as const,
							function: {
								name: geminiFunctionCall.name,
								arguments: JSON.stringify(geminiFunctionCall.args ?? {})
							}
						}];
					}

					// Log parsed tool calls
					logger.info({
						rawToolCalls: tc.map(t => ({
							id: t.id,
							type: t.type,
							funcName: t.function?.name,
							funcArgs: t.function?.arguments?.slice(0, 200)
						})),
						toolCallCount: tc.length
					}, "[mcp] non-stream parsed tool_calls");

					calls = tc.map((t, idx) => {
						const rawArgs = t.function?.arguments;
						let argsStr = "";
						if (typeof rawArgs === "string") {
							argsStr = rawArgs;
						} else if (rawArgs && typeof rawArgs === "object") {
							argsStr = JSON.stringify(rawArgs);
							logger.info({ argsType: "object" }, "[mcp] non-stream arguments was object, stringified");
						}
						return {
							// Generate ID if Gemini API returns empty ID (known bug)
							id: t.id || `call_${Date.now()}_${idx}`,
							name: t.function?.name ?? "",
							arguments: argsStr,
						};
					});
					logger.debug({ calls: calls.map(c => ({ id: c.id, name: c.name, argsLen: c.arguments.length })) }, "[mcp] non-stream tool calls recovered");
				} else {
					// Allow calls without IDs (Gemini bug) - filter only by name
					calls = Object.values(toolCallState)
						.map((c) => (c?.name ? c : undefined))
						.filter(Boolean)
						.map((c, idx) => ({
							// Generate ID if missing (Gemini API known bug)
							id: c?.id || `call_${Date.now()}_${idx}`,
							name: c?.name ?? "",
							arguments: c?.arguments ?? "",
						})) as NormalizedToolCall[];
					logger.debug({ calls: calls.map(c => ({ id: c.id, name: c.name, argsLen: c.arguments.length })) }, "[mcp] stream tool calls with generated IDs");
				}

				// Include the assistant message with tool_calls so the next round
				// sees both the calls and their outputs, matching MCP branch behavior.
				const toolCalls: ChatCompletionMessageToolCall[] = calls.map((call) => ({
					id: call.id,
					type: "function",
					function: { name: call.name, arguments: call.arguments },
				}));

				// Avoid sending <think> content back to the model alongside tool_calls
				// to prevent confusing follow-up reasoning. Strip any think blocks.
				const assistantContentForToolMsg = lastAssistantContent.replace(
					/<think>[\s\S]*?(?:<\/think>|$)/g,
					""
				);
				const assistantToolMessage: ChatCompletionMessageParam = {
					role: "assistant",
					content: assistantContentForToolMsg,
					tool_calls: toolCalls,
				};

				const exec = executeToolCalls({
					calls,
					mapping,
					servers,
					parseArgs,
					resolveFileRef,
					toPrimitive,
					processToolOutput,
					abortSignal,
				});
				let toolMsgCount = 0;
				let toolRunCount = 0;
				for await (const event of exec) {
					if (event.type === "update") {
						yield event.update;
					} else {
						messagesOpenAI = [
							...messagesOpenAI,
							assistantToolMessage,
							...(event.summary.toolMessages ?? []),
						];
						toolMsgCount = event.summary.toolMessages?.length ?? 0;
						toolRunCount = event.summary.toolRuns?.length ?? 0;
						logger.info(
							{ toolMsgCount, toolRunCount },
							"[mcp] tools executed; continuing loop for follow-up completion"
						);
					}

					// Check abort during tool execution
					if (checkAborted()) {
						logger.info({ loop, toolMsgCount }, "[mcp] aborting during tool execution");
						return "aborted";
					}
				}

				// Check abort after all tools complete before continuing loop
				if (checkAborted()) {
					logger.info({ loop }, "[mcp] aborting after tool execution");
					return "aborted";
				}
				// Emit autopilot step event so the UI can show progress
				if (autopilot) {
					yield {
						type: MessageUpdateType.AutopilotStep,
						step: loop + 1,
						maxSteps: maxLoops,
						toolCount: toolRunCount,
					};
				}
				// Continue loop: next iteration will use tool messages to get the final content
				continue;
			}

			// No tool calls in this iteration
			// If a <think> block is still open, close it for the final output
			if (thinkOpen) {
				lastAssistantContent += "</think>";
				thinkOpen = false;
			}

			// Autopilot auto-continue: if the model stopped to ask a question or
			// explain what it plans to do instead of calling tools, re-prompt it
			// to continue executing autonomously.
			logger.info({ autopilot, loop, maxLoops, contentLength: lastAssistantContent.length }, "[mcp] checking autopilot continuation");
			if (autopilot && loop < maxLoops - 1) {
				const trimmed = lastAssistantContent.replace(/<think>[\s\S]*?(?:<\/think>|$)/g, "").trim();
				const looksLikeQuestion =
					trimmed.endsWith("?") ||
					/\b(shall I|should I|would you like|do you want|let me know|can I|please provide|provide a|tell me|specify|what would you|which one|what do you)\b/i.test(trimmed);
				const looksLikePartial =
					/\b(first|next|then|now I'll|I will|let me|I'm going to|here's my plan|for example|you could)\b/i.test(trimmed);
				// Also check if model is NOT using tools when it should (no definitive answer)
				const looksLikeWaiting =
					/\b(I can|I could|I am able to|available tools|here are|options)\b/i.test(trimmed) &&
					!trimmed.includes("I have") && !trimmed.includes("Here is the") && !trimmed.includes("The result");

				// Early completion detection - model gave a definitive answer
				const looksComplete =
					/\b(I have|Here is|Here's|The result|completed|done|finished|summary|in conclusion|to summarize)\b/i.test(trimmed) &&
					!looksLikeQuestion && !looksLikePartial;

				logger.info({ looksLikeQuestion, looksLikePartial, looksLikeWaiting, looksComplete, trimmedLength: trimmed.length, trimmedPreview: trimmed.slice(0, 200) }, "[mcp] autopilot pattern check");

				// Early stop if task looks complete
				if (looksComplete) {
					logger.info({ loop, maxLoops }, "[mcp] autopilot: early completion detected, stopping");
				}

				if ((looksLikeQuestion || looksLikePartial || looksLikeWaiting) && !looksComplete) {
					// Stream the partial content so user sees what the model said
					if (!streamedContent && trimmed.length > 0) {
						yield { type: MessageUpdateType.Stream, token: lastAssistantContent };
					}
					// Add the assistant's response and a continuation prompt with better guidance
					const autopilotGuidance = `Continue executing autonomously. Follow these guidelines:

1. USE TOOLS PROACTIVELY: Call the available tools immediately to accomplish the task. Do not describe what you could do - actually do it.
2. MAKE REASONABLE ASSUMPTIONS: If you need specific input (like a search query), infer it from the conversation context or use a sensible default.
3. CHAIN ACTIONS: After one tool returns results, process them and call the next tool if needed. Keep working until the task is complete.
4. NO QUESTIONS: Do not ask the user for clarification. Make your best judgment and proceed.
5. SUMMARIZE AT END: Once you have completed all necessary actions, provide a brief summary of what was accomplished.

Proceed now with tool calls.`;

					messagesOpenAI = [
						...messagesOpenAI,
						{ role: "assistant", content: lastAssistantContent },
						{
							role: "user",
							content: autopilotGuidance,
						},
					];
					logger.info(
						{ loop, looksLikeQuestion, looksLikePartial, looksLikeWaiting },
						"[mcp] autopilot auto-continue: re-prompting model to keep going"
					);
					// Emit autopilot step
					yield {
						type: MessageUpdateType.AutopilotStep,
						step: loop + 1,
						maxSteps: maxLoops,
						toolCount: 0,
					};
					continue;
				}
			}

			if (!streamedContent && lastAssistantContent.trim().length > 0) {
				yield { type: MessageUpdateType.Stream, token: lastAssistantContent };
			}
			yield {
				type: MessageUpdateType.FinalAnswer,
				text: lastAssistantContent,
				interrupted: false,
			};
			logger.info(
				{ length: lastAssistantContent.length, loop },
				"[mcp] final answer emitted (no tool_calls)"
			);
			return "completed";
		}
		logger.warn({}, "[mcp] exceeded tool-followup loops; falling back");
	} catch (err) {
		const msg = String(err ?? "");
		const isAbort =
			(abortSignal && abortSignal.aborted) ||
			msg.includes("AbortError") ||
			msg.includes("APIUserAbortError") ||
			msg.includes("Request was aborted");
		if (isAbort) {
			// Expected on user stop; keep logs quiet and do not treat as error
			logger.debug({}, "[mcp] aborted by user");
			return "aborted";
		}
		logger.warn({ err: msg }, "[mcp] flow failed, falling back to default endpoint");
	} finally {
		// ensure MCP clients are closed after the turn
		await drainPool();
	}

	return "not_applicable";
}
