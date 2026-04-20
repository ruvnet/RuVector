/*
 * Client for the shared pi.ruv.io brain.
 *
 * Endpoints (per https://pi.ruv.io):
 *   GET  /v1/status                           — unauth, returns global stats
 *   GET  /v1/memories/list?limit=&offset=&category=   — Bearer, paginated list
 *   GET  /v1/memories/search?q=&limit=        — Bearer, semantic search (+score)
 */

import { requestUrl, RequestUrlResponse } from "obsidian";

export interface PiStatus {
	total_memories: number;
	total_contributors: number;
	graph_nodes: number;
	graph_edges: number;
	cluster_count: number;
	embedding_dim: number;
	drift_status: string;
	lora_epoch?: number;
}

export interface PiMemory {
	id: string;
	title?: string;
	content: string;
	category?: string;
	tags?: string[];
	score?: number;
	created_at?: string;
	contributor_id?: string;
}

export interface PiCreateMemoryResult {
	id: string;
	partition_id?: string | null;
	quality_score: number;
	witness_hash: string;
	rvf_segments?: number;
}

/**
 * Fixed enum accepted by pi.ruv.io's POST /v1/memories endpoint. Anything
 * outside this set must be submitted via the `custom` newtype variant.
 */
export const PI_CATEGORIES = [
	"architecture",
	"pattern",
	"solution",
	"convention",
	"security",
	"performance",
	"tooling",
	"debug",
	"sota",
	"discovery",
	"hypothesis",
	"cross_domain",
	"neural_architecture",
	"compression",
	"self_learning",
	"reinforcement_learning",
	"graph_intelligence",
	"distributed_systems",
	"edge_computing",
	"hardware_acceleration",
	"quantum",
	"neuromorphic",
	"bio_computing",
	"cognitive_science",
	"formal_methods",
	"geopolitics",
	"climate",
	"biomedical",
	"space",
	"finance",
	"meta_cognition",
	"benchmark",
	"consciousness",
	"information_decomposition",
] as const;

export type PiCategory = (typeof PI_CATEGORIES)[number] | { custom: string };

export function piCategory(raw: string): PiCategory {
	const known = PI_CATEGORIES.find((c) => c === raw);
	if (known) return known;
	return { custom: raw };
}

export interface PiSearchResult {
	memories: PiMemory[];
	total_count: number;
}

export class PiError extends Error {
	constructor(
		public status: number,
		message: string,
		public body?: unknown,
	) {
		super(message);
		this.name = "PiError";
	}
}

export class PiClient {
	constructor(
		public url: string,
		public token: string,
	) {}

	get configured(): boolean {
		return !!this.url && !!this.token;
	}

	private async req<T>(
		method: "GET" | "POST",
		path: string,
		body?: unknown,
	): Promise<T> {
		const base = this.url.replace(/\/$/, "");
		let resp: RequestUrlResponse;
		try {
			resp = await requestUrl({
				url: base + path,
				method,
				headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
				contentType: body ? "application/json" : undefined,
				body: body ? JSON.stringify(body) : undefined,
				throw: false,
			});
		} catch (e) {
			throw new PiError(0, `pi.ruv.io network error: ${(e as Error).message}`);
		}
		if (resp.status >= 400) {
			throw new PiError(resp.status, `pi.ruv.io ${path} → ${resp.status}`, resp.json ?? resp.text);
		}
		return resp.json as T;
	}

	async status(): Promise<PiStatus> {
		return this.req<PiStatus>("GET", "/v1/status");
	}

	async list(limit: number, offset = 0, category?: string): Promise<PiMemory[]> {
		const q = new URLSearchParams({
			limit: String(limit),
			offset: String(offset),
		});
		if (category) q.set("category", category);
		const resp = await this.req<{ memories?: PiMemory[] } | PiMemory[]>(
			"GET",
			`/v1/memories/list?${q.toString()}`,
		);
		if (Array.isArray(resp)) return resp;
		return resp.memories ?? [];
	}

	async search(query: string, limit: number): Promise<PiMemory[]> {
		const q = new URLSearchParams({ q: query, limit: String(limit) });
		const resp = await this.req<PiMemory[] | { memories?: PiMemory[] }>(
			"GET",
			`/v1/memories/search?${q.toString()}`,
		);
		if (Array.isArray(resp)) return resp;
		return resp.memories ?? [];
	}

	/**
	 * Publish a memory to pi.ruv.io. The server runs AIDefence, DP-noising
	 * and RVF segmentation on ingest so requests can take ~20s to return.
	 */
	async createMemory(
		category: PiCategory,
		content: string,
		title: string,
		tags: string[],
	): Promise<PiCreateMemoryResult> {
		return this.req<PiCreateMemoryResult>("POST", "/v1/memories", {
			category,
			content,
			title,
			tags,
		});
	}
}
