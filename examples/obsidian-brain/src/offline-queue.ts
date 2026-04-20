import { Notice, Plugin } from "obsidian";
import { BrainClient, BrainError } from "./brain";

/**
 * Persisted queue of pending POST /memories writes. Filled when the
 * brain is unreachable and drained on a periodic timer once health
 * comes back. Survives plugin reloads via data.json.
 *
 * Only applies to local-brain writes — pi.ruv.io writes are explicit
 * user actions and aren't queued.
 */
export interface QueuedMemory {
	category: string;
	content: string;
	queuedAt: number;
	/** Per-path deduping: one queued entry per vault path. */
	path?: string;
}

export interface OfflineQueueState {
	pending: QueuedMemory[];
}

export const EMPTY_QUEUE_STATE: OfflineQueueState = { pending: [] };

export class OfflineQueue {
	private timer: number | null = null;
	private draining = false;

	constructor(
		private plugin: Plugin,
		private brain: BrainClient,
		public state: OfflineQueueState,
		private onChange: () => void,
	) {}

	enqueue(entry: QueuedMemory): void {
		if (entry.path) {
			this.state.pending = this.state.pending.filter((q) => q.path !== entry.path);
		}
		this.state.pending.push(entry);
		this.onChange();
	}

	size(): number {
		return this.state.pending.length;
	}

	start(intervalMs = 30_000): void {
		if (this.timer) window.clearInterval(this.timer);
		this.timer = window.setInterval(() => void this.drain(), intervalMs);
	}

	stop(): void {
		if (this.timer) {
			window.clearInterval(this.timer);
			this.timer = null;
		}
	}

	/** Try to flush everything. Stops at the first network error. */
	async drain(): Promise<{ sent: number; failed: number }> {
		if (this.draining) return { sent: 0, failed: 0 };
		if (this.state.pending.length === 0) return { sent: 0, failed: 0 };
		this.draining = true;
		let sent = 0;
		let failed = 0;
		try {
			const remaining: QueuedMemory[] = [];
			for (const q of this.state.pending) {
				try {
					await this.brain.createMemory(q.category, q.content);
					sent++;
				} catch (e) {
					if (e instanceof BrainError && e.status === 0) {
						// Still offline — keep the rest queued.
						remaining.push(q);
						const idx = this.state.pending.indexOf(q);
						for (const later of this.state.pending.slice(idx + 1)) {
							remaining.push(later);
						}
						break;
					}
					// Hard rejection (422, 500, etc.) — drop it; we'd re-reject forever.
					failed++;
				}
			}
			this.state.pending = remaining;
			this.onChange();
			if (sent > 0) {
				new Notice(`Brain offline queue: replayed ${sent} pending write${sent === 1 ? "" : "s"}`);
			}
		} finally {
			this.draining = false;
		}
		return { sent, failed };
	}
}
