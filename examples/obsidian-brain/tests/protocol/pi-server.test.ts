/*
 * Real pi.ruv.io integration — gated on `BRAIN_API_KEY` because the
 * endpoints require a bearer token that isn't appropriate to commit.
 * Asserts the exact response shapes PiClient parses.
 */

import { describe, expect, it } from "vitest";

const token = process.env.BRAIN_API_KEY ?? "";
const base = process.env.PI_URL ?? "https://pi.ruv.io";
const RUN = !!token;

async function bearerJSON<T>(path: string): Promise<T> {
	const resp = await fetch(base.replace(/\/$/, "") + path, {
		headers: { Authorization: `Bearer ${token}` },
	});
	if (!resp.ok) {
		throw new Error(`${path} → ${resp.status}: ${await resp.text()}`);
	}
	return (await resp.json()) as T;
}

describe.skipIf(!RUN)("protocol: live pi.ruv.io contract", () => {
	it("GET /v1/status — returns global stats (no auth)", async () => {
		const resp = await fetch(base.replace(/\/$/, "") + "/v1/status");
		expect(resp.ok).toBe(true);
		const s = (await resp.json()) as {
			total_memories: number;
			graph_edges: number;
			embedding_dim: number;
			drift_status: string;
		};
		expect(typeof s.total_memories).toBe("number");
		expect(s.total_memories).toBeGreaterThan(0);
		expect(typeof s.graph_edges).toBe("number");
		expect(s.embedding_dim).toBeGreaterThan(0);
		expect(typeof s.drift_status).toBe("string");
	});

	it("GET /v1/memories/list — bearer-gated, returns memory[] with required fields", async () => {
		const body = await bearerJSON<{
			memories: Array<{ id: string; category: string; content: string; title?: string; tags?: string[] }>;
			total_count: number;
			offset: number;
			limit: number;
		}>("/v1/memories/list?limit=3");
		expect(body.total_count).toBeGreaterThan(0);
		expect(Array.isArray(body.memories)).toBe(true);
		expect(body.memories.length).toBeGreaterThan(0);
		const m = body.memories[0];
		expect(typeof m.id).toBe("string");
		expect(m.id.length).toBeGreaterThan(8);
		expect(typeof m.category).toBe("string");
		expect(typeof m.content).toBe("string");
	});

	it("GET /v1/memories/search — bearer-gated, returns score-ranked array", async () => {
		const results = (await bearerJSON<
			| Array<{ id: string; category: string; content: string; score?: number }>
			| { memories: Array<{ id: string; category: string; content: string; score?: number }> }
		>("/v1/memories/search?q=hnsw&limit=3")) as unknown;
		const arr = Array.isArray(results)
			? results
			: (results as { memories: unknown[] }).memories;
		expect(Array.isArray(arr)).toBe(true);
		expect(arr.length).toBeGreaterThan(0);
		const first = arr[0] as { id: string; score?: number };
		expect(typeof first.id).toBe("string");
		// score may be absent on some memory types but if present should be a number
		if ("score" in first && first.score !== undefined) {
			expect(typeof first.score).toBe("number");
		}
	});

	it("bearer is required", async () => {
		const resp = await fetch(base.replace(/\/$/, "") + "/v1/memories/list?limit=1");
		expect(resp.status).toBe(401);
	});

	it("POST /v1/memories write-through (slow ~20s) returns created shape", async () => {
		const resp = await fetch(base.replace(/\/$/, "") + "/v1/memories", {
			method: "POST",
			headers: {
				"content-type": "application/json",
				Authorization: `Bearer ${token}`,
			},
			body: JSON.stringify({
				category: "pattern",
				content:
					"obsidian-brain plugin protocol test — delete-safe placeholder",
				title: "obsidian-brain protocol probe",
				tags: ["probe", "obsidian-brain"],
			}),
		});
		expect([200, 201]).toContain(resp.status);
		const body = (await resp.json()) as {
			id: string;
			quality_score: number;
			witness_hash: string;
			rvf_segments?: number;
		};
		expect(body.id).toMatch(
			/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/,
		);
		expect(typeof body.quality_score).toBe("number");
		expect(typeof body.witness_hash).toBe("string");
	}, 40_000);

	it("POST /v1/memories rejects unknown category with 422", async () => {
		const resp = await fetch(base.replace(/\/$/, "") + "/v1/memories", {
			method: "POST",
			headers: {
				"content-type": "application/json",
				Authorization: `Bearer ${token}`,
			},
			body: JSON.stringify({
				category: "totally-bogus-category-does-not-exist",
				content: "whatever",
				title: "unused",
				tags: [],
			}),
		});
		expect(resp.status).toBe(422);
	});
});
