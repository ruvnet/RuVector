import { Address4, Address6 } from "ip-address";
import { isIP } from "node:net";

const UNSAFE_IPV4_SUBNETS = [
	"0.0.0.0/8",
	"10.0.0.0/8",
	"100.64.0.0/10",
	"127.0.0.0/8",
	"169.254.0.0/16",
	"172.16.0.0/12",
	"192.0.0.0/24",
	"192.168.0.0/16",
	"198.18.0.0/15",
	"224.0.0.0/4",
	"240.0.0.0/4",
].map((s) => new Address4(s));

const UNSAFE_IPV6_SUBNETS = [
	"::/128", // unspecified
	"fc00::/7", // unique local
	"fec0::/10", // deprecated site-local
	"ff00::/8", // multicast
].map((s) => new Address6(s));

// IPv6 prefixes that carry an IPv4 address in their low 32 bits. The mapped
// form needs its own entry: URL parsing rewrites ::ffff:10.0.0.1 to
// ::ffff:a00:1, which ip-address does not report as is4().
const IPV4_EMBEDDING_SUBNETS = ["::ffff:0:0/96", "::/96", "64:ff9b::/96"].map((s) => new Address6(s));

function isUnsafeIpv4(addr: Address4): boolean {
	return UNSAFE_IPV4_SUBNETS.some((subnet) => addr.isInSubnet(subnet));
}

function isUnsafeIp(address: string): boolean {
	const family = isIP(address);

	if (family === 4) {
		return isUnsafeIpv4(new Address4(address));
	}

	if (family === 6) {
		const addr = new Address6(address);
		// Check IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
		if (addr.is4()) {
			return isUnsafeIpv4(addr.to4());
		}
		if (IPV4_EMBEDDING_SUBNETS.some((subnet) => addr.isInSubnet(subnet))) {
			if (addr.isInSubnet(UNSAFE_IPV6_SUBNETS[0])) return true;
			return isUnsafeIpv4(addr.to4());
		}
		return (
			addr.isLoopback() ||
			addr.isLinkLocal() ||
			UNSAFE_IPV6_SUBNETS.some((subnet) => addr.isInSubnet(subnet))
		);
	}

	return true; // Unknown format → block
}

/**
 * Synchronous URL validation: checks protocol and hostname string.
 */
export function isValidUrl(urlString: string): boolean {
	try {
		const url = new URL(urlString.trim());
		const hostname = url.hostname.toLowerCase();
		// Allow HTTP for localhost/loopback/Docker-internal (dev & local MCP bridge)
		if (
			hostname === "localhost" ||
			hostname === "127.0.0.1" ||
			hostname === "::1" ||
			hostname === "host.docker.internal"
		) {
			return url.protocol === "http:" || url.protocol === "https:";
		}
		// Allow HTTP for Docker-internal service names (no dots = private network)
		if (!hostname.includes(".") && url.protocol === "http:") {
			return true;
		}
		if (url.protocol !== "https:") {
			return false;
		}
		// If the hostname is a raw IP literal, validate it
		const cleanHostname = hostname.replace(/^\[|]$/g, "");
		if (isIP(cleanHostname)) {
			return !isUnsafeIp(cleanHostname);
		}
		return true;
	} catch {
		return false;
	}
}

/**
 * Strict validation for URLs the server fetches on behalf of a caller
 * (e.g. /api/fetch-url): public HTTPS only. Unlike isValidUrl, there is no
 * allowance for localhost or single-label (Docker/Kubernetes service) names,
 * which are reserved for operator-configured MCP servers.
 */
export function isValidPublicUrl(urlString: string): boolean {
	try {
		const url = new URL(urlString.trim());
		if (url.protocol !== "https:") return false;
		const hostname = url.hostname.toLowerCase().replace(/\.$/, "");
		const cleanHostname = hostname.replace(/^\[|]$/g, "");
		if (isIP(cleanHostname)) return !isUnsafeIp(cleanHostname);
		if (!hostname.includes(".")) return false;
		if (
			hostname === "localhost" ||
			hostname.endsWith(".localhost") ||
			hostname === "host.docker.internal" ||
			hostname.endsWith(".internal") ||
			hostname.endsWith(".local")
		) {
			return false;
		}
		return true;
	} catch {
		return false;
	}
}

/**
 * Assert that a resolved IP address is safe (not internal/private).
 * Throws if the IP is internal. Used in undici's custom DNS lookup
 * to validate IPs at connection time (prevents TOCTOU DNS rebinding).
 */
export function assertSafeIp(address: string, hostname: string): void {
	if (isUnsafeIp(address)) {
		throw new Error(`Resolved IP for ${hostname} is internal (${address})`);
	}
}
