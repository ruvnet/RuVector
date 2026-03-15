import type { RouterExample } from "./routerExamples";

// Examples that showcase RuVector and π Brain capabilities
export const mcpExamples: RouterExample[] = [
	{
		title: "Search π collective",
		prompt: "Search the π Brain for patterns related to authentication best practices",
		followUps: [
			{
				title: "Security patterns",
				prompt: "Find security patterns for API key management",
			},
			{
				title: "Share a pattern",
				prompt: "Share a new pattern about JWT refresh token rotation",
			},
			{
				title: "View status",
				prompt: "Show the π Brain status and knowledge statistics",
			},
		],
	},
	{
		title: "Spawn agent swarm",
		prompt: "Initialize a swarm with 5 agents to research and implement a caching system",
		followUps: [
			{
				title: "Check status",
				prompt: "What's the current swarm status and agent health?",
			},
			{
				title: "Add specialist",
				prompt: "Spawn a security-architect agent to review the implementation",
			},
			{
				title: "View memory",
				prompt: "Search the swarm memory for cached decisions",
			},
		],
	},
	{
		title: "Knowledge transfer",
		prompt: "Transfer learning patterns from the 'rust' domain to 'typescript' domain",
		followUps: [
			{
				title: "Check drift",
				prompt: "Check knowledge drift status across domains",
			},
			{
				title: "View clusters",
				prompt: "Show me the knowledge partition clusters in the π Brain",
			},
			{
				title: "Quality stats",
				prompt: "What are the top quality patterns in the collective?",
			},
		],
	},
	{
		title: "Vector search",
		prompt: "Perform semantic search for error handling strategies in distributed systems",
		followUps: [
			{
				title: "Store pattern",
				prompt: "Store this circuit breaker pattern in memory for future reference",
			},
			{
				title: "Neural predict",
				prompt: "Use neural patterns to predict the best approach for this task",
			},
			{
				title: "Route task",
				prompt: "Route this task to the optimal agent type",
			},
		],
	},
	{
		title: "Create Brainpedia page",
		prompt: "Create a new Brainpedia page documenting the SPARC methodology for coding",
		followUps: [
			{
				title: "Add evidence",
				prompt: "Add test evidence to support the page content",
			},
			{
				title: "Submit delta",
				prompt: "Submit a correction delta with updated examples",
			},
			{
				title: "Promote page",
				prompt: "Check if the page meets promotion criteria to become canonical",
			},
		],
	},
	{
		title: "MCP tool discovery",
		prompt: "List all available MCP tools and their capabilities",
		followUps: [
			{
				title: "Brain tools",
				prompt: "Show me all π Brain tools for knowledge management",
			},
			{
				title: "Workflow tools",
				prompt: "What workflow automation tools are available?",
			},
			{
				title: "Memory tools",
				prompt: "How do I use the memory store and search tools?",
			},
		],
	},
	{
		title: "Agent coordination",
		prompt: "Orchestrate a code review with researcher, coder, and reviewer agents",
		followUps: [
			{
				title: "Hive consensus",
				prompt: "Propose a consensus vote on the implementation approach",
			},
			{
				title: "Broadcast",
				prompt: "Broadcast a message to all agents in the swarm",
			},
			{
				title: "Metrics",
				prompt: "Show agent performance metrics and task completion stats",
			},
		],
	},
	{
		title: "SONA learning",
		prompt: "Start a SONA trajectory to learn from this debugging session",
		followUps: [
			{
				title: "Record step",
				prompt: "Record this successful fix as a trajectory step",
			},
			{
				title: "Pattern search",
				prompt: "Search for similar patterns learned from past trajectories",
			},
			{
				title: "View stats",
				prompt: "Show SONA learning statistics and pattern confidence",
			},
		],
	},
];
