import { describe, expect, it } from "vitest";
import { assertSafeIp, isValidPublicUrl, isValidUrl } from "./urlSafety";

describe("assertSafeIp", () => {
	it.each([
		"10.0.0.1",
		"10.255.255.255",
		"127.0.0.1",
		"169.254.169.254",
		"172.16.0.1",
		"192.168.1.1",
		"100.64.0.1",
		"0.0.0.0",
		"198.18.0.1",
		"224.0.0.1",
		"255.255.255.255",
		"::",
		"::1",
		"fe80::1",
		"fc00::1",
		"fd12:3456:789a::1",
		"::ffff:10.0.0.1",
		"::ffff:127.0.0.1",
		"::ffff:a00:1",
		"::ffff:7f00:1",
		"::ffff:a9fe:a9fe",
		"64:ff9b::a00:1",
		"ff02::1",
	])("blocks internal address %s", (ip) => {
		expect(() => assertSafeIp(ip, "internal.example")).toThrow(/internal/);
	});

	it.each(["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111", "::ffff:8.8.8.8", "::ffff:808:808"])(
		"allows public address %s",
		(ip) => {
			expect(() => assertSafeIp(ip, "public.example")).not.toThrow();
		}
	);
});

describe("isValidUrl (MCP server URLs)", () => {
	it("still accepts Docker-internal and localhost MCP endpoints", () => {
		expect(isValidUrl("http://mcp-server:8080/mcp")).toBe(true);
		expect(isValidUrl("http://localhost:3001/mcp")).toBe(true);
	});

	it("rejects private IP literals over https", () => {
		expect(isValidUrl("https://10.1.2.3/")).toBe(false);
		expect(isValidUrl("https://[fd00::1]/")).toBe(false);
	});
});

describe("isValidPublicUrl (server-side fetch of user-supplied URLs)", () => {
	it("accepts public https URLs", () => {
		expect(isValidPublicUrl("https://huggingface.co/file.txt")).toBe(true);
	});

	it.each([
		"http://secret/",
		"http://secret:8080/token",
		"https://secret/",
		"http://localhost/",
		"https://localhost/",
		"http://127.0.0.1/",
		"https://host.docker.internal/",
		"https://10.0.0.5/",
		"https://[::ffff:10.0.0.5]/",
		"https://[::ffff:a00:5]/",
		"https://[fd00::1]/",
		"https://169.254.169.254/latest/meta-data/",
		"http://example.com/",
		"ftp://example.com/",
		"not a url",
	])("rejects %s", (u) => {
		expect(isValidPublicUrl(u)).toBe(false);
	});
});
