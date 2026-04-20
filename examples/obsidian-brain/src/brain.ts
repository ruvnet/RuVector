/*
 * Brain HTTP client — talks to the local RuVector brain server
 * (default 127.0.0.1:9876) and embedder (default 127.0.0.1:9877).
 */

import { requestUrl, RequestUrlResponse } from "obsidian";

export interface Memory {
	id: string;
	category: string;
	content?: string;
	content_hash: string;
	created_at: number;
	quality?: number;
	score?: number;
}

export interface SearchResponse {
	results: Memory[];
	query_vector_dim: number;
}

export interface BrainInfo {
	version: string;
	memories_count: number;
	preference_pairs_count: number;
	db_path: string;
	blob_dir: string;
}

export interface ScanResult {
	safe: boolean;
	threat_level: string;
	threats: string[];
}

export interface CreateMemoryResult {
	id: string;
	content_hash: string;
	created_at: number;
}

export class BrainError extends Error {
	constructor(
		public status: number,
		message: string,
		public body?: unknown,
	) {
		super(message);
		this.name = "BrainError";
	}
}

export class BrainClient {
	constructor(
		public brainUrl: string,
		public embedderUrl: string,
		private timeoutMs = 5000,
	) {}

	private async req<T>(
		method: string,
		path: string,
		body?: unknown,
		base?: string,
	): Promise<T> {
		const url = (base ?? this.brainUrl).replace(/\/$/, "") + path;
		let resp: RequestUrlResponse;
		try {
			resp = await requestUrl({
				url,
				method,
				contentType: "application/json",
				body: body !== undefined ? JSON.stringify(body) : undefined,
				throw: false,
			});
		} catch (e) {
			throw new BrainError(0, `network error: ${(e as Error).message}`);
		}
		if (resp.status >= 400) {
			throw new BrainError(resp.status, `${method} ${path} → ${resp.status}`, resp.json);
		}
		return resp.json as T;
	}

	async health(): Promise<{ status: string; version: string; backend: string }> {
		return this.req("GET", "/health");
	}

	async info(): Promise<BrainInfo> {
		return this.req("GET", "/brain/info");
	}

	async indexStats(): Promise<Record<string, unknown>> {
		return this.req("GET", "/brain/index_stats");
	}

	async search(query: string, k = 8): Promise<SearchResponse> {
		return this.req("POST", "/brain/search", { query, k });
	}

	async searchByVector(vector: number[], k = 8): Promise<SearchResponse> {
		return this.req("POST", "/brain/search", { query_vector: vector, k });
	}

	async createMemory(
		category: string,
		content: string,
		embedding?: number[],
	): Promise<CreateMemoryResult> {
		const body: Record<string, unknown> = { category, content };
		if (embedding && embedding.length) body.embedding = embedding;
		return this.req("POST", "/memories", body);
	}

	async getMemory(id: string): Promise<Memory> {
		return this.req("GET", `/memories/${id}`);
	}

	async listMemories(
		offset = 0,
		limit = 50,
		category?: string,
	): Promise<{ memories: Memory[]; total: number; count: number; offset: number }> {
		const q = new URLSearchParams({ offset: String(offset), limit: String(limit) });
		if (category) q.set("category", category);
		return this.req("GET", `/memories?${q.toString()}`);
	}

	async scan(text: string): Promise<ScanResult> {
		return this.req("POST", "/security/scan", { text });
	}

	async createPreferencePair(
		chosenId: string,
		rejectedId: string,
		direction: string,
	): Promise<{ id: string; created_at: number }> {
		return this.req("POST", "/preference_pairs", {
			chosen_id: chosenId,
			rejected_id: rejectedId,
			direction,
		});
	}

	async listPreferencePairs(
		limit = 20,
		direction?: string,
	): Promise<{ count: number; pairs: Array<Record<string, unknown>> }> {
		const q = new URLSearchParams({ limit: String(limit) });
		if (direction) q.set("direction", direction);
		return this.req("GET", `/preference_pairs?${q.toString()}`);
	}

	async embed(text: string): Promise<number[]> {
		const url = this.embedderUrl.replace(/\/$/, "") + "/embed";
		let resp: RequestUrlResponse;
		try {
			resp = await requestUrl({
				url,
				method: "POST",
				contentType: "application/json",
				body: JSON.stringify({ texts: [text] }),
				throw: false,
			});
		} catch (e) {
			throw new BrainError(0, `embedder network error: ${(e as Error).message}`);
		}
		if (resp.status >= 400) {
			throw new BrainError(resp.status, `embedder ${resp.status}`, resp.json);
		}
		const data = resp.json as { vectors?: number[][]; embeddings?: number[][] };
		const first = (data.vectors ?? data.embeddings)?.[0];
		if (!first) throw new BrainError(resp.status, "embedder returned no vector");
		return first;
	}
}
