import type { OpenAiTool } from "$lib/server/mcp/tools";

export function buildToolPreprompt(tools: OpenAiTool[], autopilot?: boolean): string {
	if (!Array.isArray(tools) || tools.length === 0) return "";
	const names = tools
		.map((t) => (t?.function?.name ? String(t.function.name) : ""))
		.filter((s) => s.length > 0);
	if (names.length === 0) return "";
	const now = new Date();
	const currentDate = now.toLocaleDateString("en-US", {
		year: "numeric",
		month: "long",
		day: "numeric",
	});
	const isoDate = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}-${String(now.getDate()).padStart(2, "0")}`;
	const lines = [
		`You have access to these tools: ${names.join(", ")}.`,
		`Today's date: ${currentDate} (${isoDate}).`,
	];

	if (autopilot) {
		lines.push(
			`AUTOPILOT MODE ENABLED — FULLY AUTONOMOUS EXECUTION. Follow these rules STRICTLY:`,
			``,
			`## CORE BEHAVIOR`,
			`1. EXECUTE IMMEDIATELY: Never ask "what would you like?" or "please provide". Infer from context and act.`,
			`2. ASSUME INTENT: If user says "search for AI", search for "artificial intelligence latest developments". If unclear, use sensible defaults.`,
			`3. CHAIN ACTIONS: Tool result → process → next tool → repeat until task is COMPLETE.`,
			`4. NO EXPLANATIONS: Don't say "I will search" — just call the search tool. Actions, not words.`,
			``,
			`## PARALLEL EXECUTION`,
			`5. CALL MULTIPLE TOOLS AT ONCE: If you need search + memory + analysis, call ALL in one response.`,
			`6. BATCH OPERATIONS: After results return, immediately call the next batch of tools.`,
			`7. MAXIMIZE PARALLELISM: 3+ simultaneous tool calls is normal. Sequential only for dependencies.`,
			``,
			`## ERROR HANDLING`,
			`8. RETRY ALTERNATIVES: If a tool fails, try a different approach. Don't stop and report failure.`,
			`9. GRACEFUL DEGRADATION: If one tool fails, continue with others. Partial results are better than none.`,
			``,
			`## COMPLETION`,
			`10. WORK UNTIL DONE: Keep calling tools until you have a complete answer or have exhausted options.`,
			`11. FINAL SUMMARY: Only after ALL actions are complete, provide a brief summary of results.`,
			`12. NO PREMATURE STOPS: If you have more tools to call, call them. Don't stop to ask if you should continue.`,
		);
	} else {
		lines.push(
			`IMPORTANT: Do NOT call a tool unless the user's request requires capabilities you lack (e.g., real-time data, image generation, code execution) or external information you do not have. For tasks like writing code, creative writing, math, or building apps, respond directly without tools. When in doubt, do not use a tool.`,
		);
	}

	lines.push(
		`TOOL PARAMETERS - CRITICAL:`,
		`- ALWAYS provide ALL required parameters. NEVER call a tool with empty {} arguments if it requires parameters.`,
		`- Check the tool's inputSchema for "required" fields. If a field is required, you MUST provide a value.`,
		`- Use example values from the tool description as guidance for the correct format.`,
		`- Common errors: calling read_file({}) instead of read_file({path: "file.txt"}). Always include the path!`,
		`- If unsure what value to use, make a reasonable assumption based on context rather than omitting the parameter.`,
		``,
		`PARALLEL TOOL CALLS: When multiple tool calls are needed and they are independent of each other (i.e., one does not need the result of another), call them all at once in a single response instead of one at a time. Only chain tool calls sequentially when a later call depends on an earlier call's output.`,
		`SEARCH: Use 3-6 precise keywords. For historical events, include the year the event occurred. For recent or current topics, use today's year (${now.getFullYear()}). When a tool accepts date-range parameters (e.g., startPublishedDate, endPublishedDate), always use today's date (${isoDate}) as the end date unless the user specifies otherwise. For multi-part questions, search each part separately.`,
		`ANSWER: State only facts explicitly in the results. If info is missing or results conflict, say so. Never fabricate URLs or facts.`,
		`INTERACTIVE APPS: When asked to build an interactive application, game, or visualization without a specific language/framework preference, create a single self-contained HTML file with embedded CSS and JavaScript.`,
		`If a tool generates an image, you can inline it directly: ![alt text](image_url).`,
		`If a tool needs an image, set its image field ("input_image", "image", or "image_url") to a reference like "image_1", "image_2", etc. (ordered by when the user uploaded them).`,
		`Default to image references; only use a full http(s) URL when the tool description explicitly asks for one, or reuse a URL a previous tool returned.`,
	);
	return lines.join(" ");
}
