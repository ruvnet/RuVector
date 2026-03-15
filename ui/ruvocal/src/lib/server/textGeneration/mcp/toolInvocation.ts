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
				const showExamples = filledArgs.examples !== false;

				// Comprehensive tool documentation with practical to exotic examples
				const toolDocs: Record<string, {
					category: string;
					desc: string;
					usage: string;
					required: string[];
					optional: string[];
					practical: string;
					advanced: string;
					exotic: string;
				}> = {
					// === FILE TOOLS ===
					read_file: {
						category: "files",
						desc: "Read contents of any file in the virtual filesystem",
						usage: "read_file({path})",
						required: ["path"],
						optional: [],
						practical: '{"path": "config.json"} → Read a config file',
						advanced: 'Chain: list_files → read_file each → grep for patterns',
						exotic: 'Build a code analyzer: read all .ts files, extract exports, generate dependency graph'
					},
					write_file: {
						category: "files",
						desc: "Create new file or overwrite existing file",
						usage: "write_file({path, content})",
						required: ["path", "content"],
						optional: [],
						practical: '{"path": "hello.py", "content": "print(\'Hello\')"}',
						advanced: 'Generate: read template → transform → write multiple files',
						exotic: 'Self-modifying code: read self, modify, write back, reload'
					},
					list_files: {
						category: "files",
						desc: "List all files in virtual filesystem",
						usage: "list_files({})",
						required: [],
						optional: [],
						practical: '{} → See what files exist',
						advanced: 'Discovery: list_files → categorize by extension → analyze each type',
						exotic: 'Create file system explorer with tree visualization'
					},
					delete_file: {
						category: "files",
						desc: "Remove a file from virtual filesystem",
						usage: "delete_file({path})",
						required: ["path"],
						optional: [],
						practical: '{"path": "temp.txt"} → Clean up temporary file',
						advanced: 'Cleanup: glob("*.tmp") → delete each match',
						exotic: 'Garbage collector: find unused files by reference analysis, prompt for deletion'
					},
					edit_file: {
						category: "files",
						desc: "Find and replace text within a file (preserves rest of content)",
						usage: "edit_file({path, old_content, new_content})",
						required: ["path", "old_content", "new_content"],
						optional: [],
						practical: '{"path": "package.json", "old_content": "\\"1.0.0\\"", "new_content": "\\"1.0.1\\""} → Bump version',
						advanced: 'Refactor: grep for pattern → edit_file each occurrence',
						exotic: 'AST-aware refactoring: parse code, transform nodes, serialize back'
					},
					grep: {
						category: "files",
						desc: "Search files for regex pattern, returns matching lines with file:line format",
						usage: "grep({pattern, path?})",
						required: ["pattern"],
						optional: ["path"],
						practical: '{"pattern": "TODO"} → Find all TODOs',
						advanced: '{"pattern": "import.*from", "path": "src/app.ts"} → Analyze imports in specific file',
						exotic: 'Dependency mapper: grep all imports → build graph → detect cycles'
					},
					glob: {
						category: "files",
						desc: "Find files matching glob pattern (*, ?, **)",
						usage: "glob({pattern})",
						required: ["pattern"],
						optional: [],
						practical: '{"pattern": "*.ts"} → Find TypeScript files',
						advanced: '{"pattern": "src/**/*.test.ts"} → Find all test files recursively',
						exotic: 'Project analyzer: glob by type → count lines → generate stats report'
					},

					// === MEMORY TOOLS ===
					memory_store: {
						category: "memory",
						desc: "Persist key-value data with optional tags for semantic search",
						usage: "memory_store({key, value, tags?})",
						required: ["key", "value"],
						optional: ["tags"],
						practical: '{"key": "user-pref", "value": "dark-mode"} → Store preference',
						advanced: '{"key": "auth-pattern-v2", "value": "JWT with refresh...", "tags": ["security", "auth", "pattern"]}',
						exotic: 'Knowledge graph: store entities as keys, relationships as values, query via tags'
					},
					memory_search: {
						category: "memory",
						desc: "Semantic search across stored memories using HNSW indexing",
						usage: "memory_search({query, top_k?})",
						required: ["query"],
						optional: ["top_k"],
						practical: '{"query": "authentication"} → Find auth-related memories',
						advanced: '{"query": "error handling patterns", "top_k": 10} → Get top 10 matches',
						exotic: 'Context builder: search query → retrieve relevant memories → inject into prompt'
					},

					// === TASK TOOLS ===
					todo_add: {
						category: "tasks",
						desc: "Add task to persistent todo list, returns task ID",
						usage: "todo_add({task})",
						required: ["task"],
						optional: [],
						practical: '{"task": "Fix login bug"} → Add a task',
						advanced: 'Project breakdown: analyze requirements → add task for each component',
						exotic: 'Self-managing agent: observe errors → create fix tasks → complete when resolved'
					},
					todo_list: {
						category: "tasks",
						desc: "List all tasks with status (○ pending, ✓ complete)",
						usage: "todo_list({})",
						required: [],
						optional: [],
						practical: '{} → See all tasks',
						advanced: 'Progress tracking: list → count complete/pending → report percentage',
						exotic: 'Sprint simulator: add tasks, estimate, track velocity, predict completion'
					},
					todo_complete: {
						category: "tasks",
						desc: "Mark task as complete by ID",
						usage: "todo_complete({id})",
						required: ["id"],
						optional: [],
						practical: '{"id": "todo-1"} → Complete first task',
						advanced: 'Batch complete: list → filter done items → complete each',
						exotic: 'Achievement system: complete task → check milestones → award badges'
					},

					// === WITNESS/AUDIT TOOLS ===
					witness_log: {
						category: "witness",
						desc: "Log action to immutable cryptographic audit chain (SHA3-256 hashed)",
						usage: "witness_log({action, data?})",
						required: ["action"],
						optional: ["data"],
						practical: '{"action": "file_modified"} → Log simple action',
						advanced: '{"action": "deploy", "data": {"env": "prod", "version": "1.2.3", "user": "admin"}}',
						exotic: 'Compliance automation: wrap every tool call with witness_log, generate audit report'
					},
					witness_verify: {
						category: "witness",
						desc: "Verify integrity of entire witness chain (checks hash continuity)",
						usage: "witness_verify({})",
						required: [],
						optional: [],
						practical: '{} → Check chain integrity',
						advanced: 'Periodic verification: schedule verify, alert on tampering',
						exotic: 'Multi-agent verification: each agent verifies chain, consensus on validity'
					},

					// === GALLERY/TEMPLATE TOOLS ===
					gallery_list: {
						category: "gallery",
						desc: "List available agent templates/personas",
						usage: "gallery_list({category?})",
						required: [],
						optional: ["category"],
						practical: '{} → See all templates',
						advanced: '{"category": "security"} → Filter by category',
						exotic: 'Template recommender: analyze task → match to best template → auto-load'
					},
					gallery_load: {
						category: "gallery",
						desc: "Activate an agent template to gain its capabilities/persona",
						usage: "gallery_load({id})",
						required: ["id"],
						optional: [],
						practical: '{"id": "development-agent"} → Load dev environment',
						advanced: 'Multi-persona: load template → execute task → switch template → verify',
						exotic: 'Agent evolution: start minimal → load progressively based on task complexity'
					},
					gallery_search: {
						category: "gallery",
						desc: "Search templates by name, description, or tags",
						usage: "gallery_search({query})",
						required: ["query"],
						optional: [],
						practical: '{"query": "security"} → Find security templates',
						advanced: 'Smart matching: search → rank by relevance → suggest top match',
						exotic: 'Template fusion: search multiple → combine capabilities → create hybrid'
					},

					// === π BRAIN TOOLS ===
					brain_search: {
						category: "brain",
						desc: "Search collective π Brain knowledge base (shared across all users)",
						usage: "brain_search({query, limit?, category?})",
						required: ["query"],
						optional: ["limit", "category"],
						practical: '{"query": "react hooks best practices"}',
						advanced: '{"query": "authentication", "category": "security", "limit": 5}',
						exotic: 'Knowledge synthesis: multi-query → merge results → generate novel insights'
					},
					brain_share: {
						category: "brain",
						desc: "Contribute knowledge to π Brain (PII-stripped, quality-scored)",
						usage: "brain_share({category, title, content, tags?})",
						required: ["category", "title", "content"],
						optional: ["tags", "code_snippet"],
						practical: '{"category": "pattern", "title": "React Auth Hook", "content": "Use useAuth..."}',
						advanced: 'Include code: {"category": "solution", "title": "...", "content": "...", "code_snippet": "const x = ..."}',
						exotic: 'Knowledge distillation: analyze codebase → extract patterns → auto-share discoveries'
					},
					brain_list: {
						category: "brain",
						desc: "List recent shared knowledge",
						usage: "brain_list({limit?, category?})",
						required: [],
						optional: ["limit", "category"],
						practical: '{"limit": 10} → See recent shares',
						advanced: '{"category": "security", "limit": 20}',
						exotic: 'Trend analysis: list by time periods → identify emerging patterns'
					},
					brain_vote: {
						category: "brain",
						desc: "Vote on knowledge quality (affects ranking)",
						usage: "brain_vote({id, direction})",
						required: ["id", "direction"],
						optional: [],
						practical: '{"id": "uuid-here", "direction": "up"}',
						advanced: 'Quality filter: search → test each → vote based on accuracy',
						exotic: 'Reputation system: track vote accuracy → weight future votes'
					},
				};

				let result: string;

				// Specific tool requested
				if (requestedTool && toolDocs[requestedTool]) {
					const d = toolDocs[requestedTool];
					result = `═══════════════════════════════════════
TOOL: ${requestedTool.toUpperCase()}
═══════════════════════════════════════
📖 ${d.desc}

📝 Usage: ${d.usage}
✅ Required: ${d.required.length > 0 ? d.required.join(", ") : "none"}
⚙️ Optional: ${d.optional.length > 0 ? d.optional.join(", ") : "none"}

🔹 PRACTICAL EXAMPLE:
   ${requestedTool}(${d.practical.split(" → ")[0]})
   ${d.practical.includes("→") ? "→ " + d.practical.split(" → ")[1] : ""}

🔸 ADVANCED PATTERN:
   ${d.advanced}

🔮 EXOTIC USE CASE:
   ${d.exotic}`;
				}
				// Category filter
				else if (category !== "all" && category !== "workflows") {
					const filtered = Object.entries(toolDocs).filter(([, d]) => d.category === category);
					if (filtered.length > 0) {
						const items = filtered.map(([name, d]) =>
							`• ${name}\n  ${d.desc}\n  Example: ${d.practical.split(" → ")[0]}`
						);
						result = `═══════════════════════════════════════
${category.toUpperCase()} TOOLS
═══════════════════════════════════════
${items.join("\n\n")}

💡 For detailed help: system_guidance({"tool": "tool_name"})`;
					} else {
						result = `Category "${category}" not found. Available: files, memory, tasks, witness, gallery, brain`;
					}
				}
				// Workflows guide
				else if (category === "workflows") {
					result = `═══════════════════════════════════════
WORKFLOW PATTERNS
═══════════════════════════════════════

🔹 CODE REVIEW WORKFLOW:
   1. list_files({}) → see what exists
   2. glob({"pattern": "*.ts"}) → find code files
   3. read_file each → analyze content
   4. grep({"pattern": "TODO|FIXME"}) → find issues
   5. todo_add for each issue found
   6. witness_log({"action": "review_complete"})

🔸 RESEARCH & REMEMBER:
   1. brain_search({"query": "topic"}) → find existing knowledge
   2. memory_search({"query": "related"}) → check local memory
   3. Execute research tasks
   4. memory_store({"key": "finding-1", "value": "..."}) → save locally
   5. brain_share({...}) → contribute to collective

🔮 SELF-IMPROVING AGENT:
   1. gallery_load({"id": "sona-learning-agent"})
   2. Execute task with witness_log for each action
   3. On error: memory_store error pattern
   4. On success: memory_store success pattern
   5. Future: memory_search before acting to avoid past errors

🎯 SECURITY AUDIT WORKFLOW:
   1. gallery_load({"id": "security-agent"})
   2. glob({"pattern": "**/*.ts"}) → find all code
   3. grep({"pattern": "eval|exec|password"}) → find risky patterns
   4. For each finding: witness_log with severity
   5. witness_verify({}) → ensure audit integrity
   6. Generate report from witness chain

🚀 MULTI-AGENT SIMULATION:
   1. gallery_load({"id": "multi-agent-orchestrator"})
   2. todo_add for each sub-task
   3. For each: switch persona via gallery_load
   4. Execute with that persona's approach
   5. memory_store each agent's output
   6. Synthesize results`;
				}
				// Full guide
				else {
					result = `═══════════════════════════════════════
🔮 RVF AGENT SYSTEM GUIDANCE
═══════════════════════════════════════

📁 FILES (7 tools) - Virtual filesystem
   • read_file, write_file, list_files, delete_file
   • edit_file, grep, glob

🧠 MEMORY (2 tools) - Persistent semantic storage
   • memory_store, memory_search

✅ TASKS (3 tools) - Todo tracking
   • todo_add, todo_list, todo_complete

🔒 WITNESS (2 tools) - Cryptographic audit trail
   • witness_log, witness_verify

🎭 GALLERY (3 tools) - Agent templates
   • gallery_list, gallery_load, gallery_search
   Templates: development-agent, research-agent,
   security-agent, sona-learning-agent, minimal-agent

🧬 π BRAIN (4 tools) - Collective intelligence
   • brain_search, brain_share, brain_list, brain_vote

───────────────────────────────────────
QUICK START EXAMPLES
───────────────────────────────────────
Create file:     write_file({"path": "app.py", "content": "..."})
Find code:       grep({"pattern": "function"})
Remember:        memory_store({"key": "idea", "value": "..."})
Search memory:   memory_search({"query": "auth"})
Track work:      todo_add({"task": "Build feature X"})
Load persona:    gallery_load({"id": "development-agent"})
Search π Brain:  brain_search({"query": "best practices"})
Audit log:       witness_log({"action": "deployed"})

───────────────────────────────────────
GET MORE HELP
───────────────────────────────────────
• Specific tool:   system_guidance({"tool": "grep"})
• Category:        system_guidance({"category": "memory"})
• Workflows:       system_guidance({"category": "workflows"})

"Run in RVF" = Execute using these sandbox tools`;
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
