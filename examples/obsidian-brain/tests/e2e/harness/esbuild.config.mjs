import esbuild from "esbuild";
import builtins from "builtin-modules";

await esbuild.build({
	entryPoints: ["main.ts"],
	bundle: true,
	external: ["obsidian", "electron", ...builtins],
	format: "cjs",
	target: "es2020",
	outfile: "main.js",
	absWorkingDir: new URL(".", import.meta.url).pathname,
	platform: "node",
	logLevel: "info",
});
