// Redirect console.log to stderr so it doesn't pollute MCP's stdout JSON-RPC channel
const origLog = console.log;
console.log = (...args) => console.error(...args);
require('../node_modules/agentic-qe/v3/dist/mcp/bundle.js');
