/**
 * WASM Integration Layer
 * Loads rvagent-wasm and provides TypeScript bindings
 */

import { browser } from "$app/environment";

// Types for WASM exports
export interface WasmMcpServer {
	handle_message(message: string): string;
	gallery(): WasmGallery;
}

export interface WasmGallery {
	list(): GalleryTemplate[];
	listByCategory(category: string): GalleryTemplate[];
	search(query: string): SearchResult[];
	get(id: string): GalleryTemplate;
	loadRvf(id: string): Uint8Array;
	setActive(id: string): void;
	getActive(): string | null;
	configure(configJson: string): void;
	getConfig(): unknown;
	addCustom(templateJson: string): void;
	removeCustom(id: string): void;
	getCategories(): Record<string, number>;
	count(): number;
	exportCustom(): GalleryTemplate[];
	importCustom(templatesJson: string): number;
}

export interface WasmRvfBuilder {
	addTool(toolJson: string): void;
	addTools(toolsJson: string): void;
	addPrompt(promptJson: string): void;
	addPrompts(promptsJson: string): void;
	addSkill(skillJson: string): void;
	addSkills(skillsJson: string): void;
	addMcpTools(mcpToolsJson: string): void;
	addCapabilities(capsJson: string): void;
	setOrchestrator(orchestratorJson: string): void;
	build(): Uint8Array;
}

export interface GalleryTemplate {
	id: string;
	name: string;
	description: string;
	category: string;
	version: string;
	author: string;
	tags: string[];
	builtin: boolean;
	tools?: ToolDefinition[];
	prompts?: AgentPrompt[];
	skills?: SkillDefinition[];
	mcp_tools?: McpToolEntry[];
	capabilities?: CapabilityDef[];
	orchestrator?: OrchestratorConfig;
}

export interface SearchResult {
	id: string;
	name: string;
	description: string;
	category: string;
	tags: string[];
	relevance: number;
}

export interface ToolDefinition {
	name: string;
	description: string;
	parameters: unknown;
	returns?: string;
}

export interface AgentPrompt {
	name: string;
	system_prompt: string;
	version: string;
}

export interface SkillDefinition {
	name: string;
	description: string;
	trigger: string;
	content: string;
}

export interface McpToolEntry {
	name: string;
	description: string;
	input_schema: unknown;
	group?: string;
}

export interface CapabilityDef {
	name: string;
	rights: string[];
	scope: string;
	delegation_depth: number;
}

export interface OrchestratorConfig {
	topology: string;
	agents: AgentNode[];
	connections: [string, string][];
}

export interface AgentNode {
	id: string;
	agent_type: string;
	prompt_ref: string;
}

// WASM module instance
let wasmModule: {
	WasmMcpServer: new () => WasmMcpServer;
	WasmGallery: new () => WasmGallery;
	WasmRvfBuilder: new () => WasmRvfBuilder;
} | null = null;

type WasmModuleType = {
	WasmMcpServer: new () => WasmMcpServer;
	WasmGallery: new () => WasmGallery;
	WasmRvfBuilder: new () => WasmRvfBuilder;
} | null;

let loadPromise: Promise<WasmModuleType> | null = null;

/**
 * Create a mock WASM module for development/testing when actual WASM isn't available
 */
function createMockWasmModule() {
	// Built-in templates for mock gallery
	const builtinTemplates: GalleryTemplate[] = [
		{
			id: "development-agent",
			name: "Development Agent",
			description: "Full-featured development agent with code editing, file management, and testing tools",
			category: "development",
			version: "1.0.0",
			author: "RuVector",
			tags: ["development", "coding", "testing", "files"],
			builtin: true,
			tools: [
				{ name: "read_file", description: "Read a file", parameters: {} },
				{ name: "write_file", description: "Write a file", parameters: {} },
				{ name: "edit_file", description: "Edit a file", parameters: {} },
				{ name: "list_files", description: "List files", parameters: {} },
				{ name: "delete_file", description: "Delete a file", parameters: {} },
			],
			prompts: [{ name: "developer", system_prompt: "You are a helpful developer assistant...", version: "1.0.0" }],
			skills: [],
			mcp_tools: [],
			capabilities: [],
		},
		{
			id: "research-agent",
			name: "Research Agent",
			description: "Research-focused agent with web search and document analysis capabilities",
			category: "research",
			version: "1.0.0",
			author: "RuVector",
			tags: ["research", "analysis", "documentation"],
			builtin: true,
			tools: [
				{ name: "web_search", description: "Search the web", parameters: {} },
				{ name: "analyze_document", description: "Analyze a document", parameters: {} },
			],
			prompts: [{ name: "researcher", system_prompt: "You are a research assistant...", version: "1.0.0" }],
			skills: [],
			mcp_tools: [],
			capabilities: [],
		},
		{
			id: "security-agent",
			name: "Security Agent",
			description: "Security-focused agent for code auditing and vulnerability scanning",
			category: "security",
			version: "1.0.0",
			author: "RuVector",
			tags: ["security", "audit", "vulnerabilities"],
			builtin: true,
			tools: [
				{ name: "scan_vulnerabilities", description: "Scan for vulnerabilities", parameters: {} },
				{ name: "audit_code", description: "Audit code for security issues", parameters: {} },
			],
			prompts: [{ name: "security", system_prompt: "You are a security expert...", version: "1.0.0" }],
			skills: [],
			mcp_tools: [],
			capabilities: [],
		},
	];

	// Virtual filesystem for mock MCP server
	const virtualFS = new Map<string, string>();
	let activeTemplateId: string | null = null;

	class MockWasmMcpServer implements WasmMcpServer {
		handle_message(message: string): string {
			const request = JSON.parse(message);
			const { method, params, id } = request;

			const response = {
				jsonrpc: "2.0",
				id,
				result: null as unknown,
				error: undefined as { code: number; message: string } | undefined,
			};

			switch (method) {
				case "initialize":
					response.result = {
						protocolVersion: "2024-11-05",
						serverInfo: { name: "rvagent-wasm-mock", version: "1.0.0" },
						capabilities: { tools: {}, prompts: {}, resources: {} },
					};
					break;

				case "tools/list":
					response.result = {
						tools: [
							{ name: "read_file", description: "Read a file", inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } },
							{ name: "write_file", description: "Write a file", inputSchema: { type: "object", properties: { path: { type: "string" }, content: { type: "string" } }, required: ["path", "content"] } },
							{ name: "list_files", description: "List files", inputSchema: { type: "object", properties: {} } },
							{ name: "delete_file", description: "Delete a file", inputSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] } },
							{ name: "edit_file", description: "Edit a file", inputSchema: { type: "object", properties: { path: { type: "string" }, old_content: { type: "string" }, new_content: { type: "string" } }, required: ["path", "old_content", "new_content"] } },
						],
					};
					break;

				case "tools/call":
					const { name, arguments: args } = params;
					switch (name) {
						case "read_file":
							const content = virtualFS.get(args.path);
							response.result = { content: [{ type: "text", text: content || "" }] };
							break;
						case "write_file":
							virtualFS.set(args.path, args.content);
							response.result = { content: [{ type: "text", text: "true" }] };
							break;
						case "list_files":
							response.result = { content: [{ type: "text", text: JSON.stringify([...virtualFS.keys()]) }] };
							break;
						case "delete_file":
							virtualFS.delete(args.path);
							response.result = { content: [{ type: "text", text: "true" }] };
							break;
						case "edit_file":
							const existing = virtualFS.get(args.path) || "";
							virtualFS.set(args.path, existing.replace(args.old_content, args.new_content));
							response.result = { content: [{ type: "text", text: "true" }] };
							break;
						default:
							response.error = { code: -32601, message: `Unknown tool: ${name}` };
					}
					break;

				case "gallery/list":
					response.result = { templates: builtinTemplates };
					break;

				case "gallery/load":
					const template = builtinTemplates.find((t) => t.id === params.id);
					if (template) {
						activeTemplateId = params.id;
						response.result = { template_id: template.id, name: template.name };
					} else {
						response.error = { code: -32602, message: `Template not found: ${params.id}` };
					}
					break;

				default:
					response.error = { code: -32601, message: `Method not found: ${method}` };
			}

			return JSON.stringify(response);
		}

		gallery(): WasmGallery {
			return new MockWasmGallery();
		}
	}

	class MockWasmGallery implements WasmGallery {
		list(): GalleryTemplate[] {
			return builtinTemplates;
		}

		listByCategory(category: string): GalleryTemplate[] {
			return builtinTemplates.filter((t) => t.category === category);
		}

		search(query: string): SearchResult[] {
			const q = query.toLowerCase();
			return builtinTemplates
				.filter((t) => t.name.toLowerCase().includes(q) || t.description.toLowerCase().includes(q) || t.tags.some((tag) => tag.toLowerCase().includes(q)))
				.map((t) => ({
					id: t.id,
					name: t.name,
					description: t.description,
					category: t.category,
					tags: t.tags,
					relevance: t.name.toLowerCase().includes(q) ? 1.0 : 0.5,
				}));
		}

		get(id: string): GalleryTemplate {
			const template = builtinTemplates.find((t) => t.id === id);
			if (!template) throw new Error(`Template not found: ${id}`);
			return template;
		}

		loadRvf(id: string): Uint8Array {
			const template = this.get(id);
			// Return mock RVF bytes (magic + version + minimal content)
			const encoder = new TextEncoder();
			const json = JSON.stringify(template);
			const jsonBytes = encoder.encode(json);
			const rvf = new Uint8Array(8 + jsonBytes.length);
			rvf.set([0x52, 0x56, 0x46, 0x00, 0x01, 0x00, 0x00, 0x00]); // RVF\0 + version
			rvf.set(jsonBytes, 8);
			return rvf;
		}

		setActive(id: string): void {
			activeTemplateId = id;
		}

		getActive(): string | null {
			return activeTemplateId;
		}

		configure(_configJson: string): void {}

		getConfig(): unknown {
			return {};
		}

		addCustom(_templateJson: string): void {}

		removeCustom(_id: string): void {}

		getCategories(): Record<string, number> {
			const categories: Record<string, number> = {};
			builtinTemplates.forEach((t) => {
				categories[t.category] = (categories[t.category] || 0) + 1;
			});
			return categories;
		}

		count(): number {
			return builtinTemplates.length;
		}

		exportCustom(): GalleryTemplate[] {
			return [];
		}

		importCustom(_templatesJson: string): number {
			return 0;
		}
	}

	class MockWasmRvfBuilder implements WasmRvfBuilder {
		private tools: unknown[] = [];
		private prompts: unknown[] = [];
		private skills: unknown[] = [];
		private mcpTools: unknown[] = [];
		private capabilities: unknown[] = [];
		private orchestrator: unknown = null;

		addTool(toolJson: string): void {
			this.tools.push(JSON.parse(toolJson));
		}

		addTools(toolsJson: string): void {
			this.tools.push(...JSON.parse(toolsJson));
		}

		addPrompt(promptJson: string): void {
			this.prompts.push(JSON.parse(promptJson));
		}

		addPrompts(promptsJson: string): void {
			this.prompts.push(...JSON.parse(promptsJson));
		}

		addSkill(skillJson: string): void {
			this.skills.push(JSON.parse(skillJson));
		}

		addSkills(skillsJson: string): void {
			this.skills.push(...JSON.parse(skillsJson));
		}

		addMcpTools(mcpToolsJson: string): void {
			this.mcpTools.push(...JSON.parse(mcpToolsJson));
		}

		addCapabilities(capsJson: string): void {
			this.capabilities.push(...JSON.parse(capsJson));
		}

		setOrchestrator(orchestratorJson: string): void {
			this.orchestrator = JSON.parse(orchestratorJson);
		}

		build(): Uint8Array {
			const content = {
				tools: this.tools,
				prompts: this.prompts,
				skills: this.skills,
				mcp_tools: this.mcpTools,
				capabilities: this.capabilities,
				orchestrator: this.orchestrator,
			};
			const encoder = new TextEncoder();
			const json = JSON.stringify(content);
			const jsonBytes = encoder.encode(json);
			const rvf = new Uint8Array(8 + jsonBytes.length);
			rvf.set([0x52, 0x56, 0x46, 0x00, 0x01, 0x00, 0x00, 0x00]); // RVF\0 + version
			rvf.set(jsonBytes, 8);
			return rvf;
		}
	}

	return {
		WasmMcpServer: MockWasmMcpServer as unknown as new () => WasmMcpServer,
		WasmGallery: MockWasmGallery as unknown as new () => WasmGallery,
		WasmRvfBuilder: MockWasmRvfBuilder as unknown as new () => WasmRvfBuilder,
	};
}

/**
 * Load the WASM module
 */
export async function loadWasm(): Promise<typeof wasmModule> {
	if (!browser) {
		return null;
	}

	if (wasmModule) {
		return wasmModule;
	}

	if (loadPromise) {
		return loadPromise;
	}

	loadPromise = (async () => {
		try {
			// Check if WASM is already loaded globally (e.g., via script tag in index.html)
			// To use real WASM, add this to your index.html:
			// <script type="module">
			//   import init, * as wasm from '/wasm/rvagent_wasm.js';
			//   await init();
			//   window.rvagent_wasm = wasm;
			// </script>
			if (typeof window !== "undefined" && (window as unknown as Record<string, unknown>).rvagent_wasm) {
				const wasm = (window as unknown as Record<string, unknown>).rvagent_wasm as {
					WasmMcpServer: new () => WasmMcpServer;
					WasmGallery: new () => WasmGallery;
					WasmRvfBuilder: new () => WasmRvfBuilder;
				};

				wasmModule = {
					WasmMcpServer: wasm.WasmMcpServer,
					WasmGallery: wasm.WasmGallery,
					WasmRvfBuilder: wasm.WasmRvfBuilder,
				};

				console.log("[WASM] rvagent-wasm loaded from global");
				return wasmModule;
			}

			// Use mock module for development/testing
			// The mock provides full MCP functionality with an in-memory virtual filesystem
			console.log("[WASM] Using mock rvagent-wasm implementation");
			wasmModule = createMockWasmModule();
			return wasmModule;
		} catch (error) {
			console.error("[WASM] Failed to initialize:", error);
			loadPromise = null;
			wasmModule = createMockWasmModule();
			return wasmModule;
		}
	})();

	return loadPromise;
}

/**
 * Check if WASM is loaded
 */
export function isWasmLoaded(): boolean {
	return wasmModule !== null;
}

/**
 * Get the WASM module (throws if not loaded)
 */
export function getWasm(): NonNullable<typeof wasmModule> {
	if (!wasmModule) {
		throw new Error("WASM module not loaded. Call loadWasm() first.");
	}
	return wasmModule;
}
