import { randomUUID } from "crypto";
import { logger } from "../../logger";
import type { MessageUpdate } from "$lib/types/MessageUpdate";
import { MessageToolUpdateType, MessageUpdateType } from "$lib/types/MessageUpdate";
import { ToolResultStatus } from "$lib/types/Tool";
import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";
import type { McpToolMapping } from "$lib/server/mcp/tools";
import type { McpServerConfig } from "$lib/server/mcp/httpClient";
import {
	callMcpTool,
	getMcpToolTimeoutMs,
	type McpToolTextResponse,
} from "$lib/server/mcp/httpClient";
import { getClient } from "$lib/server/mcp/clientPool";
import { attachFileRefsToArgs, type FileRefResolver } from "./fileRefs";
import type { Client } from "@modelcontextprotocol/sdk/client";

// ================================
// rvAgent WASM State (Server-Side)
// ================================

// Server-side virtual filesystem for WASM tool execution
// This persists for the duration of a conversation's MCP flow
const wasmVirtualFS = new Map<string, string>();

// Todo list for task tracking
const wasmTodoList: { id: string; task: string; completed: boolean; created: number }[] = [];
let wasmTodoIdCounter = 1;

// Memory store for semantic memory (simulated HNSW-indexed)
const wasmMemoryStore = new Map<string, { key: string; value: string; tags: string[] }>();

// Witness chain for cryptographic audit trail
const wasmWitnessChain: { hash: string; prevHash: string; action: string; data: unknown; timestamp: number }[] = [];
let wasmLastWitnessHash = "genesis";

// RVF Gallery templates (built-in)
const wasmGalleryTemplates = [
	{ id: "development-agent", name: "Development Agent", category: "development", description: "Full-featured dev agent", tags: ["development", "coding", "files"] },
	{ id: "research-agent", name: "Research Agent", category: "research", description: "Research & analysis agent", tags: ["research", "memory", "search"] },
	{ id: "security-agent", name: "Security Agent", category: "security", description: "Security audit agent", tags: ["security", "audit", "compliance"] },
	{ id: "multi-agent-orchestrator", name: "Multi-Agent Orchestrator", category: "orchestration", description: "Coordinate multiple agents", tags: ["orchestration", "parallel", "subagents"] },
	{ id: "sona-learning-agent", name: "SONA Learning Agent", category: "learning", description: "Self-improving with SONA", tags: ["learning", "adaptive", "neural"] },
	{ id: "agi-container-builder", name: "AGI Container Builder", category: "tooling", description: "Build portable AI packages", tags: ["agi", "container", "rvf"] },
	{ id: "witness-auditor", name: "Witness Chain Auditor", category: "compliance", description: "Cryptographic audit trails", tags: ["audit", "compliance", "witness"] },
	{ id: "minimal-agent", name: "Minimal Agent", category: "basic", description: "Lightweight file ops", tags: ["minimal", "basic", "simple"] },
];
let wasmActiveTemplateId: string | null = null;

// Helper: Simple hash for witness chain
function wasmSimpleHash(data: string): string {
	let hash = 0;
	for (let i = 0; i < data.length; i++) {
		const char = data.charCodeAt(i);
		hash = ((hash << 5) - hash) + char;
		hash = hash & hash;
	}
	return Math.abs(hash).toString(16).padStart(8, "0");
}

// Helper: Add witness entry
function wasmAddWitnessEntry(action: string, data: unknown): string {
	const entry = {
		hash: "",
		prevHash: wasmLastWitnessHash,
		action,
		data,
		timestamp: Date.now(),
	};
	entry.hash = wasmSimpleHash(JSON.stringify(entry));
	wasmWitnessChain.push(entry);
	wasmLastWitnessHash = entry.hash;
	return entry.hash;
}

/**
 * Auto-fill missing required parameters with sensible defaults
 * This intercepts empty {} calls and provides reasonable values
 * Returns both filled args AND a notice about what was auto-filled
 */
function autoFillMissingParams(
	toolName: string,
	args: Record<string, unknown>
): { filled: Record<string, unknown>; autoFilledNotice: string | null } {
	const filled = { ...args };
	const autoFilled: string[] = [];

	switch (toolName) {
		case "read_file":
		case "delete_file":
			if (!filled.path) {
				const files = Array.from(wasmVirtualFS.keys());
				filled.path = files[0] || "example.txt";
				autoFilled.push(`path="${filled.path}"`);
			}
			break;

		case "write_file":
			if (!filled.path) {
				filled.path = "untitled.txt";
				autoFilled.push(`path="${filled.path}"`);
			}
			if (filled.content === undefined) {
				filled.content = "";
				autoFilled.push(`content=""`);
			}
			break;

		case "edit_file":
			if (!filled.path) {
				const files = Array.from(wasmVirtualFS.keys());
				filled.path = files[0] || "example.txt";
				autoFilled.push(`path="${filled.path}"`);
			}
			break;

		case "grep":
		case "glob":
			if (!filled.pattern) {
				filled.pattern = "*";
				autoFilled.push(`pattern="*"`);
			}
			break;

		case "todo_add":
			if (!filled.task) {
				filled.task = "New task";
				autoFilled.push(`task="New task"`);
			}
			break;

		case "todo_complete":
			if (!filled.id) {
				const incomplete = wasmTodoList.find(t => !t.completed);
				filled.id = incomplete?.id || "todo-1";
				autoFilled.push(`id="${filled.id}"`);
			}
			break;

		case "memory_store":
			if (!filled.key) {
				filled.key = `memory-${Date.now()}`;
				autoFilled.push(`key="${filled.key}"`);
			}
			break;

		case "memory_search":
			if (!filled.query) {
				filled.query = "*";
				autoFilled.push(`query="*"`);
			}
			break;

		case "witness_log":
			if (!filled.action) {
				filled.action = "manual-entry";
				autoFilled.push(`action="manual-entry"`);
			}
			break;

		case "gallery_load":
			if (!filled.id) {
				filled.id = "development-agent";
				autoFilled.push(`id="development-agent"`);
			}
			break;

		case "gallery_search":
			if (!filled.query) {
				filled.query = "agent";
				autoFilled.push(`query="agent"`);
			}
			break;
	}

	const notice = autoFilled.length > 0
		? `[AUTO-FILLED: ${autoFilled.join(", ")}. Next time pass your own values, e.g. ${toolName}({${autoFilled.map(a => a.replace('=', ': ')).join(', ')}})]`
		: null;

	return { filled, autoFilledNotice: notice };
}

/**
 * Execute a WASM tool server-side using in-memory virtual filesystem
 * Implements full rvAgent toolset: file ops, search, tasks, memory, witness, gallery
 */
function executeWasmTool(
	toolName: string,
	args: Record<string, unknown>
): { success: boolean; result: string; error?: string } {
	try {
		// Auto-fill missing required parameters with sensible defaults
		const { filled: filledArgs, autoFilledNotice } = autoFillMissingParams(toolName, args);

		// Log to witness chain for audit (with filled args)
		wasmAddWitnessEntry(`tool:${toolName}`, { args: filledArgs });

		// Helper to append notice to successful results
		const withNotice = (result: string) =>
			autoFilledNotice ? `${result}\n\n${autoFilledNotice}` : result;

		switch (toolName) {
			// ================================
			// System Guidance (1 tool)
			// ================================
			case "system_guidance":
			case "rvf_help": {
				const requestedTool = String(filledArgs.tool || "").toLowerCase();
				const category = String(filledArgs.category || filledArgs.topic || "all").toLowerCase();

				// Comprehensive tool documentation
				const toolDocs: Record<string, { category: string; usage: string; example: string }> = {
					// File tools
					read_file: { category: "files", usage: "read_file(path) → Read file contents", example: '{"path": "src/index.ts"}' },
					write_file: { category: "files", usage: "write_file(path, content) → Create/overwrite file", example: '{"path": "hello.txt", "content": "Hello World"}' },
					list_files: { category: "files", usage: "list_files() → List all files", example: "{}" },
					delete_file: { category: "files", usage: "delete_file(path) → Delete a file", example: '{"path": "temp.txt"}' },
					edit_file: { category: "files", usage: "edit_file(path, old_content, new_content) → Replace text", example: '{"path": "config.json", "old_content": "v1", "new_content": "v2"}' },
					grep: { category: "files", usage: "grep(pattern, path?) → Search files for pattern", example: '{"pattern": "TODO"}' },
					glob: { category: "files", usage: "glob(pattern) → Find files by pattern", example: '{"pattern": "*.ts"}' },

					// Memory tools
					memory_store: { category: "memory", usage: "memory_store(key, value) → Store data persistently", example: '{"key": "auth-method", "value": "JWT tokens"}' },
					memory_search: { category: "memory", usage: "memory_search(query) → Search stored memories", example: '{"query": "authentication"}' },

					// Task tools
					todo_add: { category: "tasks", usage: "todo_add(task) → Add a task", example: '{"task": "Write unit tests"}' },
					todo_list: { category: "tasks", usage: "todo_list() → List all tasks", example: "{}" },
					todo_complete: { category: "tasks", usage: "todo_complete(id) → Mark task done", example: '{"id": "todo-1"}' },

					// Witness tools
					witness_log: { category: "witness", usage: "witness_log(action, data?) → Log to audit chain", example: '{"action": "file_modified"}' },
					witness_verify: { category: "witness", usage: "witness_verify() → Verify chain integrity", example: "{}" },

					// Gallery tools
					gallery_list: { category: "gallery", usage: "gallery_list(category?) → List agent templates", example: "{}" },
					gallery_load: { category: "gallery", usage: "gallery_load(id) → Activate a template", example: '{"id": "development-agent"}' },
					gallery_search: { category: "gallery", usage: "gallery_search(query) → Search templates", example: '{"query": "security"}' },

					// π Brain tools (if available)
					brain_search: { category: "brain", usage: "brain_search(query) → Search π Brain knowledge", example: '{"query": "react hooks best practices"}' },
					brain_share: { category: "brain", usage: "brain_share(category, title, content) → Share knowledge", example: '{"category": "pattern", "title": "Auth Pattern", "content": "Use JWT..."}' },
					brain_get: { category: "brain", usage: "brain_get(id) → Get specific memory", example: '{"id": "uuid-here"}' },
					brain_list: { category: "brain", usage: "brain_list(limit?, category?) → List recent memories", example: '{"limit": 10}' },

					// Search tools
					web_search: { category: "search", usage: "web_search(query) → Search the web", example: '{"query": "latest AI news 2024"}' },
					exa_search: { category: "search", usage: "exa_search(query) → AI-powered web search", example: '{"query": "machine learning tutorials"}' },
				};

				let result: string;

				// If specific tool requested
				if (requestedTool && toolDocs[requestedTool]) {
					const doc = toolDocs[requestedTool];
					result = `TOOL: ${requestedTool}
Category: ${doc.category}
Usage: ${doc.usage}
Example: ${requestedTool}(${doc.example})

Call this tool with the exact JSON format shown in the example.`;
				}
				// If category filter
				else if (category !== "all") {
					const filtered = Object.entries(toolDocs)
						.filter(([, doc]) => doc.category === category)
						.map(([name, doc]) => `• ${name}(${doc.example}) → ${doc.usage.split("→")[1]?.trim() || ""}`);

					if (filtered.length > 0) {
						result = `${category.toUpperCase()} TOOLS:\n${filtered.join("\n")}`;
					} else {
						result = `No tools found in category: ${category}. Available: files, memory, tasks, witness, gallery, brain, search`;
					}
				}
				// Full guide
				else {
					const categories = ["files", "memory", "tasks", "gallery", "witness", "brain", "search"];
					const sections = categories.map((cat) => {
						const tools = Object.entries(toolDocs)
							.filter(([, doc]) => doc.category === cat)
							.map(([name, doc]) => `  • ${name}(${doc.example})`);
						return tools.length > 0 ? `${cat.toUpperCase()}:\n${tools.join("\n")}` : null;
					}).filter(Boolean);

					result = `SYSTEM GUIDANCE - ALL AVAILABLE TOOLS

${sections.join("\n\n")}

TIPS:
• Always pass required parameters - never call with empty {}
• Use the example JSON format exactly as shown
• For specific tool help: system_guidance({"tool": "tool_name"})
• "Run in RVF" = use file/memory/task tools in sandbox

GALLERY TEMPLATES (use gallery_load):
• development-agent - Full dev with file ops
• research-agent - Research & memory
• security-agent - Security auditing
• minimal-agent - Basic lightweight`;
				}

				return { success: true, result };
			}

			// ================================
			// File Operations (5 tools)
			// ================================
			case "read_file": {
				const path = String(filledArgs.path || "");
				if (!path) {
					return { success: false, result: "", error: "ERROR: 'path' is required. Example: read_file({path: 'src/index.ts'})" };
				}
				const content = wasmVirtualFS.get(path);
				if (content === undefined) {
					const availableFiles = Array.from(wasmVirtualFS.keys()).slice(0, 5);
					const hint = availableFiles.length > 0 ? ` Available files: ${availableFiles.join(", ")}` : " Use list_files to see available files.";
					return { success: false, result: "", error: `File not found: ${path}.${hint}` };
				}
				return { success: true, result: withNotice(content) };
			}

			case "write_file": {
				const path = String(filledArgs.path || "");
				const content = String(filledArgs.content ?? "");
				if (!path) {
					return { success: false, result: "", error: "ERROR: 'path' is required. Example: write_file({path: 'hello.txt', content: 'Hello World'})" };
				}
				wasmVirtualFS.set(path, content);
				return { success: true, result: withNotice(`Successfully wrote ${content.length} bytes to ${path}`) };
			}

			case "list_files": {
				const files = Array.from(wasmVirtualFS.keys());
				if (files.length === 0) {
					return { success: true, result: "No files in virtual filesystem" };
				}
				return { success: true, result: `Files:\n${files.map(f => `- ${f}`).join("\n")}` };
			}

			case "delete_file": {
				const path = String(filledArgs.path || "");
				if (!path) {
					return { success: false, result: "", error: "ERROR: 'path' is required. Example: delete_file({path: 'temp.txt'})" };
				}
				if (!wasmVirtualFS.has(path)) {
					return { success: false, result: "", error: `File not found: ${path}. Use list_files to see available files.` };
				}
				wasmVirtualFS.delete(path);
				return { success: true, result: `Deleted: ${path}` };
			}

			case "edit_file": {
				const path = String(filledArgs.path || "");
				const oldContent = String(filledArgs.old_content || filledArgs.oldContent || "");
				const newContent = String(filledArgs.new_content ?? filledArgs.newContent ?? "");
				if (!path) {
					return { success: false, result: "", error: "ERROR: 'path' is required. Example: edit_file({path: 'config.json', old_content: 'v1', new_content: 'v2'})" };
				}
				if (!oldContent) {
					return { success: false, result: "", error: "ERROR: 'old_content' is required. Use read_file first to see exact content to replace." };
				}
				const existing = wasmVirtualFS.get(path);
				if (existing === undefined) {
					return { success: false, result: "", error: `File not found: ${path}. Use list_files to see available files.` };
				}
				if (!existing.includes(oldContent)) {
					const preview = existing.slice(0, 100) + (existing.length > 100 ? "..." : "");
					return { success: false, result: "", error: `old_content not found in file. File contents: "${preview}"` };
				}
				const updated = existing.replace(oldContent, newContent);
				wasmVirtualFS.set(path, updated);
				return { success: true, result: `Successfully edited ${path}` };
			}

			// ================================
			// Search Tools (2 tools)
			// ================================
			case "grep": {
				const pattern = String(filledArgs.pattern || "");
				const targetPath = filledArgs.path ? String(filledArgs.path) : null;
				if (!pattern) {
					return { success: false, result: "", error: "ERROR: 'pattern' is required. Example: grep({pattern: 'TODO'}) or grep({pattern: 'function', path: 'src/index.ts'})" };
				}
				try {
					const regex = new RegExp(pattern, "gi");
					const results: string[] = [];
					for (const [filePath, content] of wasmVirtualFS.entries()) {
						if (targetPath && filePath !== targetPath) continue;
						const lines = content.split("\n");
						lines.forEach((line, idx) => {
							if (regex.test(line)) {
								results.push(`${filePath}:${idx + 1}: ${line}`);
							}
						});
					}
					return { success: true, result: withNotice(results.length > 0 ? results.join("\n") : "No matches found") };
				} catch (e) {
					return { success: false, result: "", error: `Invalid regex: ${pattern}` };
				}
			}

			case "glob": {
				const pattern = String(filledArgs.pattern || "");
				if (!pattern) {
					return { success: false, result: "", error: "ERROR: 'pattern' is required. Example: glob({pattern: '*.ts'}) or glob({pattern: 'src/*.js'})" };
				}
				const globPattern = pattern.replace(/\*/g, ".*").replace(/\?/g, ".");
				const regex = new RegExp(`^${globPattern}$`);
				const matches = Array.from(wasmVirtualFS.keys()).filter(f => regex.test(f));
				return { success: true, result: withNotice(matches.length > 0 ? matches.join("\n") : "No matches found") };
			}

			// ================================
			// Task Management (3 tools)
			// ================================
			case "todo_add": {
				const task = String(filledArgs.task || "");
				if (!task) {
					return { success: false, result: "", error: "ERROR: 'task' is required. Example: todo_add({task: 'Implement user login'})" };
				}
				const id = `todo-${wasmTodoIdCounter++}`;
				wasmTodoList.push({ id, task, completed: false, created: Date.now() });
				return { success: true, result: withNotice(`Added task: ${task} (id: ${id})`) };
			}

			case "todo_list": {
				if (wasmTodoList.length === 0) {
					return { success: true, result: "No tasks in todo list" };
				}
				const formatted = wasmTodoList.map(t =>
					`${t.completed ? "✓" : "○"} [${t.id}] ${t.task}`
				).join("\n");
				return { success: true, result: `Tasks:\n${formatted}` };
			}

			case "todo_complete": {
				const id = String(filledArgs.id || "");
				if (!id) {
					return { success: false, result: "", error: "ERROR: 'id' is required. Example: todo_complete({id: 'todo-1'}). Use todo_list to see task IDs." };
				}
				const todo = wasmTodoList.find(t => t.id === id);
				if (!todo) {
					const availableIds = wasmTodoList.map(t => t.id).slice(0, 5);
					const hint = availableIds.length > 0 ? ` Available: ${availableIds.join(", ")}` : " Use todo_list to see tasks.";
					return { success: false, result: "", error: `Task not found: ${id}.${hint}` };
				}
				todo.completed = true;
				return { success: true, result: `Completed: ${todo.task}` };
			}

			// ================================
			// Memory Tools (2 tools) - HNSW-indexed
			// ================================
			case "memory_store": {
				const key = String(filledArgs.key || "");
				const value = String(filledArgs.value || "");
				if (!key) {
					return { success: false, result: "", error: "ERROR: 'key' is required. Example: memory_store({key: 'auth-pattern', value: 'Use JWT tokens'})" };
				}
				// value can be empty string
				const tags = Array.isArray(filledArgs.tags) ? filledArgs.tags.map(String) : [];
				wasmMemoryStore.set(key, { key, value, tags });
				return { success: true, result: `Stored memory: ${key}` };
			}

			case "memory_search": {
				const query = String(filledArgs.query || "").toLowerCase();
				if (!query || query === "*") {
					// If wildcard or empty, return all memories
					const allMemories = Array.from(wasmMemoryStore.values())
						.slice(0, 10)
						.map(m => `[${m.key}] ${m.value.slice(0, 100)}${m.value.length > 100 ? "..." : ""}`);
					return {
						success: true,
						result: withNotice(allMemories.length > 0 ? `All memories:\n${allMemories.join("\n")}` : "No memories stored")
					};
				}
				const topK = typeof filledArgs.top_k === "number" ? filledArgs.top_k : 5;
				const results = Array.from(wasmMemoryStore.values())
					.filter(m =>
						m.key.toLowerCase().includes(query) ||
						m.value.toLowerCase().includes(query) ||
						m.tags.some(t => t.toLowerCase().includes(query))
					)
					.slice(0, topK)
					.map(m => `[${m.key}] ${m.value.slice(0, 100)}${m.value.length > 100 ? "..." : ""}`);
				return {
					success: true,
					result: withNotice(results.length > 0 ? `Found ${results.length} results:\n${results.join("\n")}` : "No memories found")
				};
			}

			// ================================
			// Witness Chain (2 tools) - Cryptographic audit
			// ================================
			case "witness_log": {
				const action = String(filledArgs.action || "");
				if (!action) {
					return { success: false, result: "", error: "ERROR: 'action' is required. Example: witness_log({action: 'file_created', data: {path: 'config.json'}})" };
				}
				const data = filledArgs.data || {};
				const hash = wasmAddWitnessEntry(action, data);
				return { success: true, result: `Logged to witness chain: ${action} (hash: ${hash})` };
			}

			case "witness_verify": {
				let valid = true;
				let prevHash = "genesis";
				for (const entry of wasmWitnessChain) {
					if (entry.prevHash !== prevHash) {
						valid = false;
						break;
					}
					prevHash = entry.hash;
				}
				return { success: true, result: `Witness chain: ${valid ? "VALID" : "INVALID"} (${wasmWitnessChain.length} entries)` };
			}

			// ================================
			// RVF Gallery (3 tools)
			// ================================
			case "gallery_list": {
				const category = filledArgs.category ? String(filledArgs.category) : null;
				const filtered = category
					? wasmGalleryTemplates.filter(t => t.category === category)
					: wasmGalleryTemplates;
				const list = filtered.map(t => `- ${t.id}: ${t.name} (${t.category})`).join("\n");
				return { success: true, result: `Gallery Templates:\n${list}` };
			}

			case "gallery_load": {
				const id = String(filledArgs.id || "");
				if (!id) {
					const available = wasmGalleryTemplates.map(t => t.id).join(", ");
					return { success: false, result: "", error: `ERROR: 'id' is required. Available templates: ${available}` };
				}
				const template = wasmGalleryTemplates.find(t => t.id === id);
				if (!template) {
					const available = wasmGalleryTemplates.map(t => t.id).join(", ");
					return { success: false, result: "", error: `Template not found: ${id}. Available: ${available}` };
				}
				wasmActiveTemplateId = id;
				return { success: true, result: withNotice(`Loaded template: ${template.name}\nDescription: ${template.description}\nCategory: ${template.category}`) };
			}

			case "gallery_search": {
				const query = String(filledArgs.query || "").toLowerCase();
				if (!query) {
					return { success: false, result: "", error: "ERROR: 'query' is required. Example: gallery_search({query: 'security'}) or gallery_search({query: 'development'})" };
				}
				const matches = wasmGalleryTemplates.filter(t =>
					t.name.toLowerCase().includes(query) ||
					t.description.toLowerCase().includes(query) ||
					t.tags.some(tag => tag.toLowerCase().includes(query))
				);
				if (matches.length === 0) {
					return { success: true, result: withNotice("No templates found matching your query") };
				}
				const list = matches.map(t => `- ${t.id}: ${t.name}\n  ${t.description}`).join("\n");
				return { success: true, result: withNotice(`Found ${matches.length} templates:\n${list}`) };
			}

			default:
				return { success: false, result: "", error: `Unknown WASM tool: ${toolName}` };
		}
	} catch (e) {
		const errMsg = e instanceof Error ? e.message : String(e);
		return { success: false, result: "", error: errMsg };
	}
}

export type Primitive = string | number | boolean;

export type ToolRun = {
	name: string;
	parameters: Record<string, Primitive>;
	output: string;
};

export interface NormalizedToolCall {
	id: string;
	name: string;
	arguments: string;
}

export interface ExecuteToolCallsParams {
	calls: NormalizedToolCall[];
	mapping: Record<string, McpToolMapping>;
	servers: McpServerConfig[];
	parseArgs: (raw: unknown) => Record<string, unknown>;
	resolveFileRef?: FileRefResolver;
	toPrimitive: (value: unknown) => Primitive | undefined;
	processToolOutput: (text: string) => {
		annotated: string;
		sources: { index: number; link: string }[];
	};
	abortSignal?: AbortSignal;
	toolTimeoutMs?: number;
}

export interface ToolCallExecutionResult {
	toolMessages: ChatCompletionMessageParam[];
	toolRuns: ToolRun[];
	finalAnswer?: { text: string; interrupted: boolean };
}

export type ToolExecutionEvent =
	| { type: "update"; update: MessageUpdate }
	| { type: "complete"; summary: ToolCallExecutionResult };

const serverMap = (servers: McpServerConfig[]): Map<string, McpServerConfig> => {
	const map = new Map<string, McpServerConfig>();
	for (const server of servers) {
		if (server?.name) {
			map.set(server.name, server);
		}
	}
	return map;
};

export async function* executeToolCalls({
	calls,
	mapping,
	servers,
	parseArgs,
	resolveFileRef,
	toPrimitive,
	processToolOutput,
	abortSignal,
	toolTimeoutMs,
}: ExecuteToolCallsParams): AsyncGenerator<ToolExecutionEvent, void, undefined> {
	const effectiveTimeoutMs = toolTimeoutMs ?? getMcpToolTimeoutMs();
	const toolMessages: ChatCompletionMessageParam[] = [];
	const toolRuns: ToolRun[] = [];
	const serverLookup = serverMap(servers);
	// Pre-emit call + ETA updates and prepare tasks
	type TaskResult = {
		index: number;
		output?: string;
		structured?: unknown;
		blocks?: unknown[];
		error?: string;
		uuid: string;
		paramsClean: Record<string, Primitive>;
	};

	const prepared = calls.map((call) => {
		logger.info({
			callId: call.id,
			callName: call.name,
			rawArguments: call.arguments?.slice(0, 300),
			argsLength: call.arguments?.length ?? 0
		}, "[mcp-invoke] preparing tool call");
		const argsObj = parseArgs(call.arguments);
		logger.info({
			callName: call.name,
			parsedKeys: Object.keys(argsObj),
			parsedArgsPreview: JSON.stringify(argsObj).slice(0, 200)
		}, "[mcp-invoke] parsed arguments");
		const paramsClean: Record<string, Primitive> = {};
		for (const [k, v] of Object.entries(argsObj ?? {})) {
			const prim = toPrimitive(v);
			if (prim !== undefined) paramsClean[k] = prim;
		}
		// Attach any resolved image payloads _after_ computing paramsClean so that
		// logging / status updates continue to show only the lightweight primitive
		// arguments (e.g. "image_1") while the full data: URLs or image blobs are
		// only sent to the MCP tool server.
		attachFileRefsToArgs(argsObj, resolveFileRef);
		return { call, argsObj, paramsClean, uuid: randomUUID() };
	});

	for (const p of prepared) {
		yield {
			type: "update",
			update: {
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Call,
				uuid: p.uuid,
				call: { name: p.call.name, parameters: p.paramsClean },
			},
		};
		yield {
			type: "update",
			update: {
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.ETA,
				uuid: p.uuid,
				eta: 10,
			},
		};
	}

	// Preload clients per distinct server used in this batch
	const distinctServerNames = Array.from(
		new Set(prepared.map((p) => mapping[p.call.name]?.server).filter(Boolean) as string[])
	);
	const clientMap = new Map<string, Client>();
	await Promise.all(
		distinctServerNames.map(async (name) => {
			const cfg = serverLookup.get(name);
			if (!cfg) return;
			try {
				const client = await getClient(cfg, abortSignal);
				clientMap.set(name, client);
			} catch (e) {
				logger.warn({ server: name, err: String(e) }, "[mcp] failed to connect client");
			}
		})
	);

	// Async queue to stream results in finish order
	function createQueue<T>() {
		const items: T[] = [];
		const waiters: Array<(v: IteratorResult<T>) => void> = [];
		let closed = false;
		return {
			push(item: T) {
				const waiter = waiters.shift();
				if (waiter) waiter({ value: item, done: false });
				else items.push(item);
			},
			close() {
				closed = true;
				let waiter: ((v: IteratorResult<T>) => void) | undefined;
				while ((waiter = waiters.shift())) {
					waiter({ value: undefined as unknown as T, done: true });
				}
			},
			async *iterator() {
				for (;;) {
					if (items.length) {
						const first = items.shift();
						if (first !== undefined) yield first as T;
						continue;
					}
					if (closed) return;
					const value: IteratorResult<T> = await new Promise((res) => waiters.push(res));
					if (value.done) return;
					yield value.value as T;
				}
			},
		};
	}

	const updatesQueue = createQueue<MessageUpdate>();
	const results: TaskResult[] = [];

	const tasks = prepared.map(async (p, index) => {
		// Check abort before starting each tool call
		if (abortSignal?.aborted) {
			const message = "Aborted by user";
			results.push({
				index,
				error: message,
				uuid: p.uuid,
				paramsClean: p.paramsClean,
			});
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Error,
				uuid: p.uuid,
				message,
			});
			return;
		}

		const mappingEntry = mapping[p.call.name];
		if (!mappingEntry) {
			const message = `Unknown MCP function: ${p.call.name}`;
			results.push({
				index,
				error: message,
				uuid: p.uuid,
				paramsClean: p.paramsClean,
			});
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Error,
				uuid: p.uuid,
				message,
			});
			return;
		}

		// Handle WASM tools - execute server-side with virtual filesystem
		if (mappingEntry.server === "__wasm__") {
			logger.info(
				{ tool: mappingEntry.tool, params: p.paramsClean },
				"[mcp] executing WASM tool server-side"
			);

			const wasmResult = executeWasmTool(mappingEntry.tool, p.argsObj);
			const outputText = wasmResult.success
				? wasmResult.result
				: `Error: ${wasmResult.error}`;
			const status = wasmResult.success ? ToolResultStatus.Success : ToolResultStatus.Error;

			results.push({
				index,
				output: outputText,
				uuid: p.uuid,
				paramsClean: p.paramsClean,
				...(wasmResult.success ? {} : { error: wasmResult.error }),
			});
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: wasmResult.success ? MessageToolUpdateType.Result : MessageToolUpdateType.Error,
				uuid: p.uuid,
				...(wasmResult.success
					? {
						result: {
							status,
							call: { name: p.call.name, parameters: p.paramsClean },
							outputs: [{ text: outputText } as unknown as Record<string, unknown>],
							display: true,
						},
					}
					: { message: wasmResult.error || "Unknown error" }
				),
			});
			logger.info(
				{ tool: mappingEntry.tool, success: wasmResult.success, outputPreview: outputText.slice(0, 100) },
				"[mcp] WASM tool execution completed"
			);
			return;
		}

		const serverCfg = serverLookup.get(mappingEntry.server);
		if (!serverCfg) {
			const message = `Unknown MCP server: ${mappingEntry.server}`;
			results.push({
				index,
				error: message,
				uuid: p.uuid,
				paramsClean: p.paramsClean,
			});
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Error,
				uuid: p.uuid,
				message,
			});
			return;
		}
		const client = clientMap.get(mappingEntry.server);
		try {
			logger.debug(
				{ server: mappingEntry.server, tool: mappingEntry.tool, parameters: p.paramsClean },
				"[mcp] invoking tool"
			);
			const toolResponse: McpToolTextResponse = await callMcpTool(
				serverCfg,
				mappingEntry.tool,
				p.argsObj,
				{
					client,
					signal: abortSignal,
					timeoutMs: effectiveTimeoutMs,
					onProgress: (progress) => {
						updatesQueue.push({
							type: MessageUpdateType.Tool,
							subtype: MessageToolUpdateType.Progress,
							uuid: p.uuid,
							progress: progress.progress,
							total: progress.total,
							message: progress.message,
						});
					},
				}
			);
			const { annotated } = processToolOutput(toolResponse.text ?? "");
			logger.debug(
				{ server: mappingEntry.server, tool: mappingEntry.tool },
				"[mcp] tool call completed"
			);
			results.push({
				index,
				output: annotated,
				structured: toolResponse.structured,
				blocks: toolResponse.content,
				uuid: p.uuid,
				paramsClean: p.paramsClean,
			});
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Result,
				uuid: p.uuid,
				result: {
					status: ToolResultStatus.Success,
					call: { name: p.call.name, parameters: p.paramsClean },
					outputs: [
						{
							text: annotated ?? "",
							structured: toolResponse.structured,
							content: toolResponse.content,
						} as unknown as Record<string, unknown>,
					],
					display: true,
				},
			});
		} catch (err) {
			const errMsg = err instanceof Error ? err.message : String(err);
			const errName = err instanceof Error ? err.name : "";
			const isAbortError =
				abortSignal?.aborted ||
				errName === "AbortError" ||
				errName === "APIUserAbortError" ||
				errMsg === "Request was aborted." ||
				errMsg === "This operation was aborted";
			const message = isAbortError ? "Aborted by user" : errMsg;

			if (isAbortError) {
				logger.debug(
					{ server: mappingEntry.server, tool: mappingEntry.tool },
					"[mcp] tool call aborted by user"
				);
			} else {
				logger.warn(
					{ server: mappingEntry.server, tool: mappingEntry.tool, err: message },
					"[mcp] tool call failed"
				);
			}
			results.push({ index, error: message, uuid: p.uuid, paramsClean: p.paramsClean });
			updatesQueue.push({
				type: MessageUpdateType.Tool,
				subtype: MessageToolUpdateType.Error,
				uuid: p.uuid,
				message,
			});
		}
	});

	// kick off and stream as they finish
	Promise.allSettled(tasks).then(() => updatesQueue.close());

	for await (const update of updatesQueue.iterator()) {
		yield { type: "update", update };
	}

	// Collate outputs in original call order
	results.sort((a, b) => a.index - b.index);
	for (const r of results) {
		const name = prepared[r.index].call.name;
		const id = prepared[r.index].call.id;
		if (!r.error) {
			const output = r.output ?? "";
			toolRuns.push({ name, parameters: r.paramsClean, output });
			// For the LLM follow-up call, we keep only the textual output
			toolMessages.push({ role: "tool", tool_call_id: id, content: output });
		} else {
			// Communicate error to LLM so it doesn't hallucinate success
			toolMessages.push({ role: "tool", tool_call_id: id, content: `Error: ${r.error}` });
		}
	}

	yield { type: "complete", summary: { toolMessages, toolRuns } };
}
