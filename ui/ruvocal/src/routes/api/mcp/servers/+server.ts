import type { MCPServer } from "$lib/types/Tool";
import { config } from "$lib/server/config";

// Built-in MCP servers always available (users can toggle them off)
const BUILTIN_SERVERS: Array<{ name: string; url: string }> = [
	{ name: "pi-brain", url: "https://mcp.pi.ruv.io" },
];

export async function GET() {
	// Parse MCP_SERVERS environment variable
	const mcpServersEnv = config.MCP_SERVERS || "[]";

	let envServers: Array<{ name: string; url: string; headers?: Record<string, string> }> = [];

	try {
		envServers = JSON.parse(mcpServersEnv);
		if (!Array.isArray(envServers)) {
			envServers = [];
		}
	} catch (error) {
		console.error("Failed to parse MCP_SERVERS env variable:", error);
		envServers = [];
	}

	// Merge built-in + env servers, env takes precedence by name
	const envNames = new Set(envServers.map((s) => s.name));
	const allServers = [
		...BUILTIN_SERVERS.filter((s) => !envNames.has(s.name)),
		...envServers,
	];

	// Convert internal server config to client MCPServer format
	const mcpServers: MCPServer[] = allServers.map((server) => ({
		id: `base-${server.name}`, // Stable ID based on name
		name: server.name,
		url: server.url,
		type: "base" as const,
		// headers intentionally omitted
		isLocked: false, // Base servers can be toggled by users
		status: undefined, // Status determined client-side via health check
	}));

	return Response.json(mcpServers);
}
