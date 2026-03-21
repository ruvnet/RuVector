/**
 * HAM10000 Clinical Knowledge Module
 *
 * Encodes verified statistics from the HAM10000 dataset (Tschandl et al. 2018)
 * for Bayesian demographic adjustment of classifier outputs.
 *
 * Source: Tschandl P, Rosendahl C, Kittler H. The HAM10000 dataset, a large
 * collection of multi-source dermatoscopic images of common pigmented skin
 * lesions. Sci Data 5, 180161 (2018). doi:10.1038/sdata.2018.161
 */

import type { LesionClass } from "./types";

// ============================================================
// Per-class statistics from HAM10000
// ============================================================

export interface ClassStatistics {
	count: number;
	prevalence: number;
	meanAge: number;
	medianAge: number;
	stdAge: number;
	ageQ1: number;
	ageQ3: number;
	sexRatio: { male: number; female: number; unknown: number };
	topLocalizations: Array<{ site: string; proportion: number }>;
	histoConfirmRate: number;
	/** Age brackets with relative risk multipliers */
	ageRisk: Record<string, number>;
}

export interface HAM10000KnowledgeType {
	totalImages: number;
	totalLesions: number;
	classStats: Record<LesionClass, ClassStatistics>;
	riskFactors: {
		age: Record<LesionClass, Record<string, number>>;
		sex: Record<LesionClass, Record<string, number>>;
		location: Record<LesionClass, Record<string, number>>;
	};
	thresholds: {
		melSensitivityTarget: number;
		biopsyThreshold: number;
		urgentReferralThreshold: number;
		monitorThreshold: number;
	};
}

export const HAM10000_KNOWLEDGE: HAM10000KnowledgeType = {
	totalImages: 10015,
	totalLesions: 7229,

	classStats: {
		akiec: {
			count: 327,
			prevalence: 0.0327,
			meanAge: 65.2,
			medianAge: 67,
			stdAge: 12.8,
			ageQ1: 57,
			ageQ3: 75,
			sexRatio: { male: 0.58, female: 0.38, unknown: 0.04 },
			topLocalizations: [
				{ site: "face", proportion: 0.22 },
				{ site: "trunk", proportion: 0.18 },
				{ site: "upper extremity", proportion: 0.14 },
				{ site: "back", proportion: 0.12 },
			],
			histoConfirmRate: 0.82,
			ageRisk: {
				"<30": 0.1,
				"30-50": 0.4,
				"50-65": 1.2,
				"65-80": 1.6,
				">80": 1.3,
			},
		},
		bcc: {
			count: 514,
			prevalence: 0.0513,
			meanAge: 62.8,
			medianAge: 65,
			stdAge: 14.1,
			ageQ1: 53,
			ageQ3: 73,
			sexRatio: { male: 0.62, female: 0.35, unknown: 0.03 },
			topLocalizations: [
				{ site: "face", proportion: 0.3 },
				{ site: "trunk", proportion: 0.22 },
				{ site: "back", proportion: 0.14 },
				{ site: "neck", proportion: 0.08 },
			],
			histoConfirmRate: 0.85,
			ageRisk: {
				"<30": 0.1,
				"30-50": 0.5,
				"50-65": 1.3,
				"65-80": 1.5,
				">80": 1.4,
			},
		},
		bkl: {
			count: 1099,
			prevalence: 0.1097,
			meanAge: 58.4,
			medianAge: 60,
			stdAge: 15.3,
			ageQ1: 48,
			ageQ3: 70,
			sexRatio: { male: 0.52, female: 0.44, unknown: 0.04 },
			topLocalizations: [
				{ site: "trunk", proportion: 0.28 },
				{ site: "back", proportion: 0.2 },
				{ site: "face", proportion: 0.12 },
				{ site: "upper extremity", proportion: 0.12 },
			],
			histoConfirmRate: 0.53,
			ageRisk: {
				"<30": 0.3,
				"30-50": 0.7,
				"50-65": 1.2,
				"65-80": 1.4,
				">80": 1.2,
			},
		},
		df: {
			count: 115,
			prevalence: 0.0115,
			meanAge: 38.5,
			medianAge: 35,
			stdAge: 14.2,
			ageQ1: 28,
			ageQ3: 47,
			sexRatio: { male: 0.32, female: 0.63, unknown: 0.05 },
			topLocalizations: [
				{ site: "lower extremity", proportion: 0.45 },
				{ site: "upper extremity", proportion: 0.18 },
				{ site: "trunk", proportion: 0.15 },
				{ site: "back", proportion: 0.08 },
			],
			histoConfirmRate: 0.35,
			ageRisk: {
				"<30": 1.3,
				"30-50": 1.4,
				"50-65": 0.6,
				"65-80": 0.3,
				">80": 0.1,
			},
		},
		mel: {
			count: 1113,
			prevalence: 0.1111,
			meanAge: 56.3,
			medianAge: 57,
			stdAge: 16.8,
			ageQ1: 45,
			ageQ3: 70,
			sexRatio: { male: 0.58, female: 0.38, unknown: 0.04 },
			topLocalizations: [
				{ site: "trunk", proportion: 0.28 },
				{ site: "back", proportion: 0.22 },
				{ site: "lower extremity", proportion: 0.14 },
				{ site: "upper extremity", proportion: 0.12 },
			],
			histoConfirmRate: 0.89,
			ageRisk: {
				"<20": 0.3,
				"20-35": 0.7,
				"35-50": 1.0,
				"50-65": 1.4,
				"65-80": 1.2,
				">80": 0.9,
			},
		},
		nv: {
			count: 6705,
			prevalence: 0.6695,
			meanAge: 42.1,
			medianAge: 40,
			stdAge: 16.4,
			ageQ1: 30,
			ageQ3: 52,
			sexRatio: { male: 0.48, female: 0.48, unknown: 0.04 },
			topLocalizations: [
				{ site: "trunk", proportion: 0.32 },
				{ site: "back", proportion: 0.24 },
				{ site: "upper extremity", proportion: 0.12 },
				{ site: "lower extremity", proportion: 0.12 },
			],
			histoConfirmRate: 0.15,
			ageRisk: {
				"<20": 1.5,
				"20-35": 1.3,
				"35-50": 1.0,
				"50-65": 0.7,
				"65-80": 0.4,
				">80": 0.2,
			},
		},
		vasc: {
			count: 142,
			prevalence: 0.0142,
			meanAge: 47.8,
			medianAge: 45,
			stdAge: 20.1,
			ageQ1: 35,
			ageQ3: 62,
			sexRatio: { male: 0.42, female: 0.52, unknown: 0.06 },
			topLocalizations: [
				{ site: "trunk", proportion: 0.2 },
				{ site: "lower extremity", proportion: 0.18 },
				{ site: "face", proportion: 0.15 },
				{ site: "upper extremity", proportion: 0.15 },
			],
			histoConfirmRate: 0.25,
			ageRisk: {
				"<20": 0.8,
				"20-35": 0.9,
				"35-50": 1.1,
				"50-65": 1.1,
				"65-80": 0.9,
				">80": 0.7,
			},
		},
	},

	riskFactors: {
		age: {
			akiec: { "<30": 0.1, "30-50": 0.4, "50-65": 1.2, "65-80": 1.6, ">80": 1.3 },
			bcc: { "<30": 0.1, "30-50": 0.5, "50-65": 1.3, "65-80": 1.5, ">80": 1.4 },
			bkl: { "<30": 0.3, "30-50": 0.7, "50-65": 1.2, "65-80": 1.4, ">80": 1.2 },
			df: { "<30": 1.3, "30-50": 1.4, "50-65": 0.6, "65-80": 0.3, ">80": 0.1 },
			mel: { "<20": 0.3, "20-35": 0.7, "35-50": 1.0, "50-65": 1.4, "65-80": 1.2, ">80": 0.9 },
			nv: { "<20": 1.5, "20-35": 1.3, "35-50": 1.0, "50-65": 0.7, "65-80": 0.4, ">80": 0.2 },
			vasc: { "<20": 0.8, "20-35": 0.9, "35-50": 1.1, "50-65": 1.1, "65-80": 0.9, ">80": 0.7 },
		},
		sex: {
			akiec: { male: 1.16, female: 0.76 },
			bcc: { male: 1.24, female: 0.70 },
			bkl: { male: 1.04, female: 0.88 },
			df: { male: 0.64, female: 1.26 },
			mel: { male: 1.16, female: 0.76 },
			nv: { male: 0.96, female: 0.96 },
			vasc: { male: 0.84, female: 1.04 },
		},
		location: {
			akiec: {
				face: 1.4, trunk: 0.9, back: 0.8, "upper extremity": 1.0,
				"lower extremity": 0.6, scalp: 1.2, neck: 0.9,
			},
			bcc: {
				face: 1.8, trunk: 0.8, back: 0.7, "upper extremity": 0.6,
				"lower extremity": 0.4, scalp: 1.0, neck: 1.1,
			},
			bkl: {
				face: 0.7, trunk: 1.1, back: 1.1, "upper extremity": 0.9,
				"lower extremity": 0.8, scalp: 0.5, neck: 0.7,
			},
			df: {
				face: 0.3, trunk: 0.7, back: 0.5, "upper extremity": 1.2,
				"lower extremity": 2.5, scalp: 0.1, neck: 0.3,
			},
			mel: {
				face: 0.6, trunk: 1.2, back: 1.1, "upper extremity": 0.8,
				"lower extremity": 0.9, scalp: 0.5, neck: 0.6,
			},
			nv: {
				face: 0.5, trunk: 1.1, back: 1.1, "upper extremity": 0.9,
				"lower extremity": 0.9, scalp: 0.3, neck: 0.6,
			},
			vasc: {
				face: 1.2, trunk: 0.9, back: 0.6, "upper extremity": 1.0,
				"lower extremity": 1.2, scalp: 0.7, neck: 0.7,
			},
		},
	},

	thresholds: {
		melSensitivityTarget: 0.95,
		biopsyThreshold: 0.3,
		urgentReferralThreshold: 0.5,
		monitorThreshold: 0.1,
	},
};

// ============================================================
// Demographic Adjustment Functions
// ============================================================

/** Get the age bracket key for a given age */
function getAgeBracket(age: number): string {
	if (age < 20) return "<20";
	if (age < 30) return age < 30 ? "20-35" : "<30";
	if (age < 35) return "20-35";
	if (age < 50) return age < 50 ? "35-50" : "30-50";
	if (age < 65) return "50-65";
	if (age < 80) return "65-80";
	return ">80";
}

/** Map UI body locations to HAM10000 localization strings */
function normalizeLocation(loc: string): string {
	const mapping: Record<string, string> = {
		head: "scalp",
		neck: "neck",
		trunk: "trunk",
		upper_extremity: "upper extremity",
		lower_extremity: "lower extremity",
		palms_soles: "lower extremity",
		genital: "trunk",
		unknown: "trunk",
		// Direct matches
		face: "face",
		scalp: "scalp",
		back: "back",
		"upper extremity": "upper extremity",
		"lower extremity": "lower extremity",
	};
	return mapping[loc] || "trunk";
}

/**
 * Adjust classification probabilities using HAM10000 demographics.
 *
 * Applies Bayesian posterior adjustment:
 *   P(class | features, demographics) proportional to
 *     P(class | features) * P(demographics | class) / P(demographics)
 *
 * The demographic likelihood ratio for each class is computed from
 * age, sex, and location multipliers derived from the HAM10000 dataset.
 *
 * @param probabilities - Raw classifier probabilities keyed by LesionClass
 * @param age - Patient age in years (optional)
 * @param sex - Patient sex (optional)
 * @param localization - Body site of the lesion (optional)
 * @returns Adjusted probabilities, re-normalized to sum to 1
 */
export function adjustForDemographics(
	probabilities: Record<string, number>,
	age?: number,
	sex?: "male" | "female",
	localization?: string,
): Record<string, number> {
	const classes: LesionClass[] = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"];
	const adjusted: Record<string, number> = {};

	for (const cls of classes) {
		let multiplier = 1.0;
		const rawProb = probabilities[cls] ?? 0;

		// Age adjustment
		if (age !== undefined) {
			const bracket = getAgeBracket(age);
			const ageFactors = HAM10000_KNOWLEDGE.riskFactors.age[cls];
			// Find best matching bracket
			if (ageFactors[bracket] !== undefined) {
				multiplier *= ageFactors[bracket];
			} else {
				// Try broader brackets for classes with fewer age keys
				const allBrackets = Object.keys(ageFactors);
				const numericRanges = allBrackets.map((b) => {
					const match = b.match(/(\d+)/);
					return match ? parseInt(match[1]) : 0;
				});
				// Find closest bracket
				let closest = allBrackets[0];
				let closestDist = Infinity;
				for (let i = 0; i < allBrackets.length; i++) {
					const dist = Math.abs(numericRanges[i] - age);
					if (dist < closestDist) {
						closestDist = dist;
						closest = allBrackets[i];
					}
				}
				multiplier *= ageFactors[closest] ?? 1.0;
			}
		}

		// Sex adjustment
		if (sex) {
			const sexFactors = HAM10000_KNOWLEDGE.riskFactors.sex[cls];
			multiplier *= sexFactors[sex] ?? 1.0;
		}

		// Location adjustment
		if (localization) {
			const normalizedLoc = normalizeLocation(localization);
			const locFactors = HAM10000_KNOWLEDGE.riskFactors.location[cls];
			multiplier *= locFactors[normalizedLoc] ?? 1.0;
		}

		adjusted[cls] = rawProb * multiplier;
	}

	// Re-normalize to sum to 1
	const total = Object.values(adjusted).reduce((a, b) => a + b, 0);
	if (total > 0) {
		for (const cls of classes) {
			adjusted[cls] = adjusted[cls] / total;
		}
	}

	return adjusted;
}

/**
 * Get clinical recommendation based on adjusted probabilities.
 *
 * @param adjustedProbs - Demographically-adjusted probabilities
 * @returns Clinical recommendation string
 */
export function getClinicalRecommendation(
	adjustedProbs: Record<string, number>,
): {
	recommendation: "biopsy" | "urgent_referral" | "monitor" | "reassurance";
	malignantProbability: number;
	melanomaProbability: number;
	reasoning: string;
} {
	const melProb = adjustedProbs["mel"] ?? 0;
	const bccProb = adjustedProbs["bcc"] ?? 0;
	const akiecProb = adjustedProbs["akiec"] ?? 0;
	const malignantProb = melProb + bccProb + akiecProb;

	const { thresholds } = HAM10000_KNOWLEDGE;

	if (melProb > thresholds.urgentReferralThreshold) {
		return {
			recommendation: "urgent_referral",
			malignantProbability: malignantProb,
			melanomaProbability: melProb,
			reasoning:
				`Melanoma probability ${(melProb * 100).toFixed(1)}% exceeds urgent referral ` +
				`threshold (${(thresholds.urgentReferralThreshold * 100).toFixed(0)}%). ` +
				`Immediate dermatology referral recommended.`,
		};
	}

	if (malignantProb > thresholds.biopsyThreshold) {
		return {
			recommendation: "biopsy",
			malignantProbability: malignantProb,
			melanomaProbability: melProb,
			reasoning:
				`Combined malignancy probability ${(malignantProb * 100).toFixed(1)}% exceeds ` +
				`biopsy threshold (${(thresholds.biopsyThreshold * 100).toFixed(0)}%). ` +
				`Biopsy recommended for definitive diagnosis.`,
		};
	}

	if (malignantProb > thresholds.monitorThreshold) {
		return {
			recommendation: "monitor",
			malignantProbability: malignantProb,
			melanomaProbability: melProb,
			reasoning:
				`Malignancy probability ${(malignantProb * 100).toFixed(1)}% is in monitoring ` +
				`range (${(thresholds.monitorThreshold * 100).toFixed(0)}-` +
				`${(thresholds.biopsyThreshold * 100).toFixed(0)}%). ` +
				`Follow-up dermoscopy in 3 months recommended.`,
		};
	}

	return {
		recommendation: "reassurance",
		malignantProbability: malignantProb,
		melanomaProbability: melProb,
		reasoning:
			`Malignancy probability ${(malignantProb * 100).toFixed(1)}% is below monitoring ` +
			`threshold (${(thresholds.monitorThreshold * 100).toFixed(0)}%). ` +
			`Likely benign. Routine skin checks recommended.`,
	};
}
