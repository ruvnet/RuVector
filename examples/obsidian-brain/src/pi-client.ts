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

	private async req<T>(path: string): Promise<T> {
		const base = this.url.replace(/\/$/, "");
		let resp: RequestUrlResponse;
		try {
			resp = await requestUrl({
				url: base + path,
				method: "GET",
				headers: this.token ? { Authorization: `Bearer ${this.token}` } : {},
				throw: false,
			});
		} catch (e) {
			throw new PiError(0, `pi.ruv.io network error: ${(e as Error).message}`);
		}
		if (resp.status >= 400) {
			throw new PiError(resp.status, `pi.ruv.io ${path} → ${resp.status}`, resp.json);
		}
		return resp.json as T;
	}

	async status(): Promise<PiStatus> {
		return this.req<PiStatus>("/v1/status");
	}

	async list(limit: number, offset = 0, category?: string): Promise<PiMemory[]> {
		const q = new URLSearchParams({
			limit: String(limit),
			offset: String(offset),
		});
		if (category) q.set("category", category);
		const resp = await this.req<{ memories?: PiMemory[] } | PiMemory[]>(
			`/v1/memories/list?${q.toString()}`,
		);
		if (Array.isArray(resp)) return resp;
		return resp.memories ?? [];
	}

	async search(query: string, limit: number): Promise<PiMemory[]> {
		const q = new URLSearchParams({ q: query, limit: String(limit) });
		const resp = await this.req<PiMemory[] | { memories?: PiMemory[] }>(
			`/v1/memories/search?${q.toString()}`,
		);
		if (Array.isArray(resp)) return resp;
		return resp.memories ?? [];
	}
}
