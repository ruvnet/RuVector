/**
 * DrAgnes Federated Learning Module
 *
 * SONA/LoRA-based federated learning with EWC++ regularization,
 * reputation-weighted aggregation, and Byzantine poisoning detection.
 */

/** LoRA configuration for low-rank adaptation */
export interface LoRAConfig {
	/** Rank of the low-rank decomposition (typically 2-8) */
	rank: number;
	/** Scaling factor alpha */
	alpha: number;
	/** Dropout rate for LoRA layers */
	dropout: number;
	/** Target modules for adaptation */
	targetModules: string[];
}

/** EWC++ configuration for continual learning */
export interface EWCConfig {
	/** Regularization strength */
	lambda: number;
	/** Online EWC decay factor (gamma) */
	gamma: number;
	/** Fisher information estimation samples */
	fisherSamples: number;
}

/** Federated aggregation strategy */
export type AggregationStrategy =
	| "fedavg"
	| "fedprox"
	| "reputation_weighted"
	| "trimmed_mean";

/** Federated learning configuration */
export interface FederatedConfig {
	/** LoRA adaptation settings */
	lora: LoRAConfig;
	/** EWC++ continual learning settings */
	ewc: EWCConfig;
	/** Aggregation strategy for combining updates */
	aggregation: AggregationStrategy;
	/** Minimum number of participants per round */
	minParticipants: number;
	/** Maximum rounds before forced aggregation */
	maxRoundsBeforeSync: number;
	/** Differential privacy noise multiplier */
	dpNoiseMultiplier: number;
	/** Gradient clipping norm */
	maxGradNorm: number;
}

/** A LoRA delta update from a local training round */
export interface LoRADelta {
	/** Node identifier (pseudonymous) */
	nodeId: string;
	/** Low-rank matrix A (down-projection) */
	matrixA: Float32Array;
	/** Low-rank matrix B (up-projection) */
	matrixB: Float32Array;
	/** Rank used */
	rank: number;
	/** Number of local training samples */
	localSamples: number;
	/** Local loss after training */
	localLoss: number;
	/** Round number */
	round: number;
	/** Timestamp */
	timestamp: string;
}

/** Population-level statistics for poisoning detection */
export interface PopulationStats {
	meanNorm: number;
	stdNorm: number;
	meanLoss: number;
	stdLoss: number;
	totalParticipants: number;
}

/** Poisoning detection result */
export interface PoisoningResult {
	isPoisoned: boolean;
	reason: string;
	normZScore: number;
	lossZScore: number;
}

/** Default federated learning configuration */
export const DEFAULT_FEDERATED_CONFIG: FederatedConfig = {
	lora: {
		rank: 2,
		alpha: 4,
		dropout: 0.05,
		targetModules: ["classifier.weight", "features.last_conv.weight"],
	},
	ewc: {
		lambda: 5000,
		gamma: 0.95,
		fisherSamples: 200,
	},
	aggregation: "reputation_weighted",
	minParticipants: 3,
	maxRoundsBeforeSync: 10,
	dpNoiseMultiplier: 1.1,
	maxGradNorm: 1.0,
};

/**
 * Compute a rank-r LoRA delta between local and global weights.
 *
 * Approximates (localWeights - globalWeights) as A * B^T where
 * A is (d x r) and B is (k x r).
 *
 * @param localWeights - Locally fine-tuned weight matrix (flattened)
 * @param globalWeights - Current global model weights (flattened)
 * @param rows - Number of rows in the weight matrix
 * @param cols - Number of columns in the weight matrix
 * @param rank - LoRA rank (default 2)
 * @returns Low-rank decomposition matrices A and B
 */
export function computeLoRADelta(
	localWeights: Float32Array,
	globalWeights: Float32Array,
	rows: number,
	cols: number,
	rank: number = 2
): { matrixA: Float32Array; matrixB: Float32Array } {
	if (localWeights.length !== globalWeights.length) {
		throw new Error("Weight dimensions must match");
	}
	if (localWeights.length !== rows * cols) {
		throw new Error(`Expected ${rows * cols} weights, got ${localWeights.length}`);
	}

	// Compute difference matrix
	const diff = new Float32Array(localWeights.length);
	for (let i = 0; i < diff.length; i++) {
		diff[i] = localWeights[i] - globalWeights[i];
	}

	// Truncated SVD via power iteration to get rank-r approximation
	const matrixA = new Float32Array(rows * rank);
	const matrixB = new Float32Array(cols * rank);

	for (let r = 0; r < rank; r++) {
		// Initialize random vector
		const v = new Float32Array(cols);
		for (let i = 0; i < cols; i++) {
			v[i] = Math.random() - 0.5;
		}
		normalizeVector(v);

		// Power iteration (10 iterations)
		const u = new Float32Array(rows);
		for (let iter = 0; iter < 10; iter++) {
			// u = diff * v
			for (let i = 0; i < rows; i++) {
				let sum = 0;
				for (let j = 0; j < cols; j++) {
					sum += diff[i * cols + j] * v[j];
				}
				u[i] = sum;
			}
			normalizeVector(u);

			// v = diff^T * u
			for (let j = 0; j < cols; j++) {
				let sum = 0;
				for (let i = 0; i < rows; i++) {
					sum += diff[i * cols + j] * u[i];
				}
				v[j] = sum;
			}
			normalizeVector(v);
		}

		// Compute singular value
		let sigma = 0;
		for (let i = 0; i < rows; i++) {
			let sum = 0;
			for (let j = 0; j < cols; j++) {
				sum += diff[i * cols + j] * v[j];
			}
			sigma += sum * u[i];
		}

		// Store rank component: A[:, r] = sqrt(sigma) * u, B[:, r] = sqrt(sigma) * v
		const sqrtSigma = Math.sqrt(Math.abs(sigma));
		const sign = sigma >= 0 ? 1 : -1;
		for (let i = 0; i < rows; i++) {
			matrixA[i * rank + r] = sqrtSigma * u[i] * sign;
		}
		for (let j = 0; j < cols; j++) {
			matrixB[j * rank + r] = sqrtSigma * v[j];
		}

		// Deflate: remove this component from diff
		for (let i = 0; i < rows; i++) {
			for (let j = 0; j < cols; j++) {
				diff[i * cols + j] -= sigma * u[i] * v[j];
			}
		}
	}

	return { matrixA, matrixB };
}

/**
 * Apply EWC++ regularization to a delta update.
 *
 * Penalizes changes to parameters that are important for previous tasks,
 * as measured by the Fisher information matrix diagonal.
 *
 * @param delta - Raw parameter update
 * @param fisherDiagonal - Diagonal of the Fisher information matrix
 * @param lambda - Regularization strength
 * @returns Regularized delta
 */
export function applyEWC(
	delta: Float32Array,
	fisherDiagonal: Float32Array,
	lambda: number
): Float32Array {
	if (delta.length !== fisherDiagonal.length) {
		throw new Error("Delta and Fisher diagonal must have same length");
	}

	const regularized = new Float32Array(delta.length);

	for (let i = 0; i < delta.length; i++) {
		// EWC penalty: lambda * F_i * delta_i^2
		// Effective update: delta_i / (1 + lambda * F_i)
		const penalty = 1 + lambda * fisherDiagonal[i];
		regularized[i] = delta[i] / penalty;
	}

	return regularized;
}

/**
 * Aggregate multiple LoRA deltas using reputation-weighted FedAvg.
 *
 * Each participant's contribution is weighted by their reputation score
 * (derived from historical accuracy, data quality, consistency).
 *
 * @param deltas - Array of LoRA delta updates
 * @param reputationWeights - Per-participant reputation scores [0, 1]
 * @returns Aggregated delta matrices
 */
export function aggregateDeltas(
	deltas: LoRADelta[],
	reputationWeights: number[]
): { matrixA: Float32Array; matrixB: Float32Array } {
	if (deltas.length === 0) {
		throw new Error("At least one delta required");
	}
	if (deltas.length !== reputationWeights.length) {
		throw new Error("Deltas and weights must have same length");
	}

	const rank = deltas[0].rank;
	const aSize = deltas[0].matrixA.length;
	const bSize = deltas[0].matrixB.length;

	// Normalize reputation weights to sum to 1
	const totalWeight = reputationWeights.reduce((a, b) => a + b, 0);
	const normalized = reputationWeights.map((w) => w / totalWeight);

	// Sample-weighted reputation: combine reputation with local sample count
	const sampleWeights = deltas.map((d, i) => normalized[i] * d.localSamples);
	const totalSampleWeight = sampleWeights.reduce((a, b) => a + b, 0);
	const finalWeights = sampleWeights.map((w) => w / totalSampleWeight);

	const aggA = new Float32Array(aSize);
	const aggB = new Float32Array(bSize);

	for (let di = 0; di < deltas.length; di++) {
		const w = finalWeights[di];
		const delta = deltas[di];

		if (delta.rank !== rank) {
			throw new Error(`Inconsistent ranks: expected ${rank}, got ${delta.rank}`);
		}

		for (let i = 0; i < aSize; i++) {
			aggA[i] += delta.matrixA[i] * w;
		}
		for (let i = 0; i < bSize; i++) {
			aggB[i] += delta.matrixB[i] * w;
		}
	}

	return { matrixA: aggA, matrixB: aggB };
}

/**
 * Detect potentially poisoned model updates using 2-sigma outlier detection.
 *
 * Flags updates whose weight norm or loss deviates more than 2 standard
 * deviations from the population mean.
 *
 * @param delta - The update to check
 * @param populationStats - Aggregate statistics from all participants
 * @returns Detection result with z-scores and reasoning
 */
export function detectPoisoning(
	delta: LoRADelta,
	populationStats: PopulationStats
): PoisoningResult {
	// Compute norm of the delta
	let normSq = 0;
	for (let i = 0; i < delta.matrixA.length; i++) {
		normSq += delta.matrixA[i] ** 2;
	}
	for (let i = 0; i < delta.matrixB.length; i++) {
		normSq += delta.matrixB[i] ** 2;
	}
	const norm = Math.sqrt(normSq);

	const normZScore = populationStats.stdNorm > 0
		? Math.abs(norm - populationStats.meanNorm) / populationStats.stdNorm
		: 0;

	const lossZScore = populationStats.stdLoss > 0
		? Math.abs(delta.localLoss - populationStats.meanLoss) / populationStats.stdLoss
		: 0;

	const reasons: string[] = [];
	if (normZScore > 2) {
		reasons.push(`weight norm z-score ${normZScore.toFixed(2)} exceeds 2-sigma threshold`);
	}
	if (lossZScore > 2) {
		reasons.push(`loss z-score ${lossZScore.toFixed(2)} exceeds 2-sigma threshold`);
	}

	return {
		isPoisoned: reasons.length > 0,
		reason: reasons.length > 0 ? reasons.join("; ") : "within normal range",
		normZScore,
		lossZScore,
	};
}

/**
 * Federated learning coordinator for DrAgnes nodes.
 *
 * Manages local model adaptation via LoRA, EWC++ regularization,
 * and secure aggregation with Byzantine fault detection.
 */
export class FederatedLearning {
	private config: FederatedConfig;
	private localDeltas: LoRADelta[] = [];
	private globalMatrixA: Float32Array | null = null;
	private globalMatrixB: Float32Array | null = null;
	private fisherDiagonal: Float32Array | null = null;
	private round = 0;
	private nodeId: string;

	constructor(nodeId: string, config: FederatedConfig = DEFAULT_FEDERATED_CONFIG) {
		this.nodeId = nodeId;
		this.config = config;
	}

	/**
	 * Contribute a local model update to the federated round.
	 *
	 * @param localWeights - Locally fine-tuned weights
	 * @param globalWeights - Current global weights
	 * @param rows - Weight matrix rows
	 * @param cols - Weight matrix cols
	 * @param localLoss - Loss on local validation set
	 * @param localSamples - Number of local training samples
	 * @returns The LoRA delta to send to the aggregator
	 */
	contributeUpdate(
		localWeights: Float32Array,
		globalWeights: Float32Array,
		rows: number,
		cols: number,
		localLoss: number,
		localSamples: number
	): LoRADelta {
		const { matrixA, matrixB } = computeLoRADelta(
			localWeights,
			globalWeights,
			rows,
			cols,
			this.config.lora.rank
		);

		// Apply EWC if Fisher information is available
		let finalA = matrixA;
		let finalB = matrixB;
		if (this.fisherDiagonal) {
			if (this.fisherDiagonal.length === matrixA.length) {
				finalA = applyEWC(matrixA, this.fisherDiagonal, this.config.ewc.lambda);
			}
			if (this.fisherDiagonal.length === matrixB.length) {
				finalB = applyEWC(matrixB, this.fisherDiagonal, this.config.ewc.lambda);
			}
		}

		// Apply gradient clipping
		clipByNorm(finalA, this.config.maxGradNorm);
		clipByNorm(finalB, this.config.maxGradNorm);

		// Add DP noise
		if (this.config.dpNoiseMultiplier > 0) {
			addGaussianNoise(finalA, this.config.dpNoiseMultiplier * this.config.maxGradNorm);
			addGaussianNoise(finalB, this.config.dpNoiseMultiplier * this.config.maxGradNorm);
		}

		const delta: LoRADelta = {
			nodeId: this.nodeId,
			matrixA: finalA,
			matrixB: finalB,
			rank: this.config.lora.rank,
			localSamples,
			localLoss,
			round: this.round,
			timestamp: new Date().toISOString(),
		};

		this.localDeltas.push(delta);
		return delta;
	}

	/**
	 * Receive and apply the aggregated global model update.
	 *
	 * @param matrixA - Aggregated A matrix
	 * @param matrixB - Aggregated B matrix
	 * @param newFisherDiagonal - Updated Fisher information (optional)
	 */
	receiveGlobalModel(
		matrixA: Float32Array,
		matrixB: Float32Array,
		newFisherDiagonal?: Float32Array
	): void {
		this.globalMatrixA = new Float32Array(matrixA);
		this.globalMatrixB = new Float32Array(matrixB);

		if (newFisherDiagonal) {
			if (this.fisherDiagonal) {
				// Online EWC++: exponential moving average of Fisher
				for (let i = 0; i < newFisherDiagonal.length; i++) {
					this.fisherDiagonal[i] =
						this.config.ewc.gamma * this.fisherDiagonal[i] +
						(1 - this.config.ewc.gamma) * newFisherDiagonal[i];
				}
			} else {
				this.fisherDiagonal = new Float32Array(newFisherDiagonal);
			}
		}

		this.round++;
	}

	/**
	 * Get the current local adaptation state.
	 *
	 * @returns Current global matrices, round, and delta history count
	 */
	getLocalAdaptation(): {
		globalMatrixA: Float32Array | null;
		globalMatrixB: Float32Array | null;
		round: number;
		totalContributions: number;
		hasFisherInfo: boolean;
		config: FederatedConfig;
	} {
		return {
			globalMatrixA: this.globalMatrixA,
			globalMatrixB: this.globalMatrixB,
			round: this.round,
			totalContributions: this.localDeltas.length,
			hasFisherInfo: this.fisherDiagonal !== null,
			config: this.config,
		};
	}
}

/** Normalize a vector in-place to unit length */
function normalizeVector(v: Float32Array): void {
	let norm = 0;
	for (let i = 0; i < v.length; i++) {
		norm += v[i] ** 2;
	}
	norm = Math.sqrt(norm);
	if (norm > 1e-10) {
		for (let i = 0; i < v.length; i++) {
			v[i] /= norm;
		}
	}
}

/** Clip vector by global norm in-place */
function clipByNorm(v: Float32Array, maxNorm: number): void {
	let normSq = 0;
	for (let i = 0; i < v.length; i++) {
		normSq += v[i] ** 2;
	}
	const norm = Math.sqrt(normSq);
	if (norm > maxNorm) {
		const scale = maxNorm / norm;
		for (let i = 0; i < v.length; i++) {
			v[i] *= scale;
		}
	}
}

/** Add Gaussian noise in-place for differential privacy */
function addGaussianNoise(v: Float32Array, sigma: number): void {
	for (let i = 0; i < v.length; i++) {
		// Box-Muller transform
		const u1 = Math.random();
		const u2 = Math.random();
		const z = Math.sqrt(-2 * Math.log(u1 + 1e-10)) * Math.cos(2 * Math.PI * u2);
		v[i] += z * sigma;
	}
}
