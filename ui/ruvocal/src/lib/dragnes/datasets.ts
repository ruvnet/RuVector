/**
 * DrAgnes Dataset Metadata and Device Specifications
 *
 * Reference data for training datasets, class distributions,
 * bias warnings, and DermLite dermoscope specifications.
 */

/** Dataset class distribution entry */
export interface ClassDistribution {
	count: number;
	percentage: number;
}

/** Fitzpatrick skin type distribution */
export interface FitzpatrickDistribution {
	I: number;
	II: number;
	III: number;
	IV: number;
	V: number;
	VI: number;
}

/** Dataset metadata */
export interface DatasetMetadata {
	name: string;
	fullName: string;
	source: string;
	license: string;
	totalImages: number;
	classes: Record<string, ClassDistribution>;
	fitzpatrickDistribution: Partial<FitzpatrickDistribution>;
	imagingModality: string;
	resolution: string;
	diagnosticMethod: string;
	biasWarning: string;
}

/** DermLite device specification */
export interface DermLiteSpec {
	name: string;
	magnification: string;
	fieldOfView: string;
	resolution: string;
	polarization: string[];
	contactMode: string[];
	connectivity: string;
	weight: string;
	ledSpectrum: string;
	price: string;
}

/**
 * Curated dermoscopy and clinical image datasets used for
 * training, validation, and fairness evaluation.
 */
export const DATASETS: Record<string, DatasetMetadata> = {
	HAM10000: {
		name: "HAM10000",
		fullName: "Human Against Machine with 10000 training images",
		source: "https://doi.org/10.1038/sdata.2018.161",
		license: "CC BY-NC-SA 4.0",
		totalImages: 10015,
		classes: {
			nv: { count: 6705, percentage: 66.95 },
			mel: { count: 1113, percentage: 11.11 },
			bkl: { count: 1099, percentage: 10.97 },
			bcc: { count: 514, percentage: 5.13 },
			akiec: { count: 327, percentage: 3.27 },
			vasc: { count: 142, percentage: 1.42 },
			df: { count: 115, percentage: 1.15 },
		},
		fitzpatrickDistribution: {
			I: 0.05,
			II: 0.35,
			III: 0.40,
			IV: 0.15,
			V: 0.04,
			VI: 0.01,
		},
		imagingModality: "dermoscopy",
		resolution: "600x450",
		diagnosticMethod: "histopathology (>50%), follow-up, expert consensus",
		biasWarning:
			"Underrepresents Fitzpatrick V-VI. Supplement with Fitzpatrick17k for fairness evaluation.",
	},

	ISIC_ARCHIVE: {
		name: "ISIC Archive",
		fullName: "International Skin Imaging Collaboration Archive",
		source: "https://www.isic-archive.com",
		license: "CC BY-NC 4.0",
		totalImages: 70000,
		classes: {
			nv: { count: 32542, percentage: 46.49 },
			mel: { count: 11720, percentage: 16.74 },
			bkl: { count: 6250, percentage: 8.93 },
			bcc: { count: 5210, percentage: 7.44 },
			akiec: { count: 3800, percentage: 5.43 },
			vasc: { count: 1100, percentage: 1.57 },
			df: { count: 890, percentage: 1.27 },
			scc: { count: 2480, percentage: 3.54 },
			other: { count: 6008, percentage: 8.58 },
		},
		fitzpatrickDistribution: {
			I: 0.08,
			II: 0.30,
			III: 0.35,
			IV: 0.18,
			V: 0.06,
			VI: 0.03,
		},
		imagingModality: "dermoscopy + clinical",
		resolution: "variable (up to 4000x3000)",
		diagnosticMethod: "histopathology, expert annotation",
		biasWarning:
			"Predominantly lighter skin tones. Use stratified sampling for fair evaluation.",
	},

	BCN20000: {
		name: "BCN20000",
		fullName: "Barcelona 20000 dermoscopic images dataset",
		source: "https://doi.org/10.1038/s41597-023-02405-z",
		license: "CC BY-NC-SA 4.0",
		totalImages: 19424,
		classes: {
			nv: { count: 12875, percentage: 66.28 },
			mel: { count: 2288, percentage: 11.78 },
			bkl: { count: 1636, percentage: 8.42 },
			bcc: { count: 1202, percentage: 6.19 },
			akiec: { count: 590, percentage: 3.04 },
			vasc: { count: 310, percentage: 1.60 },
			df: { count: 243, percentage: 1.25 },
			scc: { count: 280, percentage: 1.44 },
		},
		fitzpatrickDistribution: {
			I: 0.04,
			II: 0.38,
			III: 0.42,
			IV: 0.12,
			V: 0.03,
			VI: 0.01,
		},
		imagingModality: "dermoscopy",
		resolution: "1024x1024",
		diagnosticMethod: "histopathology",
		biasWarning:
			"Southern European population bias. Cross-validate with geographically diverse datasets.",
	},

	PH2: {
		name: "PH2",
		fullName: "PH2 dermoscopic image database",
		source: "https://doi.org/10.1109/EMBC.2013.6610779",
		license: "Research use only",
		totalImages: 200,
		classes: {
			nv: { count: 80, percentage: 40.0 },
			mel: { count: 40, percentage: 20.0 },
			bkl: { count: 80, percentage: 40.0 },
		},
		fitzpatrickDistribution: {
			II: 0.40,
			III: 0.45,
			IV: 0.15,
		},
		imagingModality: "dermoscopy",
		resolution: "768x560",
		diagnosticMethod: "expert consensus + histopathology",
		biasWarning:
			"Small dataset (200 images). Only 3 classes. Use for supplementary validation only.",
	},

	DERM7PT: {
		name: "Derm7pt",
		fullName: "Seven-point checklist dermoscopic dataset",
		source: "https://doi.org/10.1016/j.media.2018.11.010",
		license: "Research use only",
		totalImages: 1011,
		classes: {
			nv: { count: 575, percentage: 56.87 },
			mel: { count: 252, percentage: 24.93 },
			bkl: { count: 98, percentage: 9.69 },
			bcc: { count: 42, percentage: 4.15 },
			df: { count: 24, percentage: 2.37 },
			vasc: { count: 12, percentage: 1.19 },
			misc: { count: 8, percentage: 0.79 },
		},
		fitzpatrickDistribution: {
			I: 0.06,
			II: 0.32,
			III: 0.38,
			IV: 0.18,
			V: 0.04,
			VI: 0.02,
		},
		imagingModality: "clinical + dermoscopy paired",
		resolution: "variable",
		diagnosticMethod: "histopathology + 7-point checklist scoring",
		biasWarning:
			"Paired clinical/dermoscopic images. Melanoma-enriched relative to prevalence.",
	},

	FITZPATRICK17K: {
		name: "Fitzpatrick17k",
		fullName: "Fitzpatrick17k dermatology atlas across all skin tones",
		source: "https://doi.org/10.48550/arXiv.2104.09957",
		license: "CC BY-NC-SA 4.0",
		totalImages: 16577,
		classes: {
			inflammatory: { count: 5480, percentage: 33.06 },
			benign_neoplasm: { count: 4230, percentage: 25.52 },
			malignant_neoplasm: { count: 2890, percentage: 17.43 },
			infectious: { count: 2150, percentage: 12.97 },
			genodermatosis: { count: 920, percentage: 5.55 },
			other: { count: 907, percentage: 5.47 },
		},
		fitzpatrickDistribution: {
			I: 0.12,
			II: 0.18,
			III: 0.22,
			IV: 0.20,
			V: 0.16,
			VI: 0.12,
		},
		imagingModality: "clinical photography",
		resolution: "variable",
		diagnosticMethod: "clinical diagnosis, atlas annotation",
		biasWarning:
			"Essential for fairness evaluation. Use to audit model performance across all skin tones.",
	},

	PAD_UFES_20: {
		name: "PAD-UFES-20",
		fullName: "Smartphone skin lesion dataset from Brazil",
		source: "https://doi.org/10.1016/j.dib.2020.106221",
		license: "CC BY 4.0",
		totalImages: 2298,
		classes: {
			bcc: { count: 845, percentage: 36.77 },
			mel: { count: 52, percentage: 2.26 },
			scc: { count: 192, percentage: 8.35 },
			akiec: { count: 730, percentage: 31.77 },
			nv: { count: 244, percentage: 10.62 },
			sek: { count: 235, percentage: 10.23 },
		},
		fitzpatrickDistribution: {
			II: 0.15,
			III: 0.35,
			IV: 0.30,
			V: 0.15,
			VI: 0.05,
		},
		imagingModality: "smartphone camera",
		resolution: "variable (smartphone-captured)",
		diagnosticMethod: "histopathology",
		biasWarning:
			"Smartphone-captured (non-dermoscopic). Brazilian population. Useful for real-world phone-based screening validation.",
	},
};

/**
 * DermLite dermoscope device specifications.
 * Used for hardware compatibility and imaging parameter calibration.
 */
export const DERMLITE_SPECS: Record<string, DermLiteSpec> = {
	HUD: {
		name: "DermLite HUD",
		magnification: "10x",
		fieldOfView: "25mm",
		resolution: "1920x1080",
		polarization: ["polarized", "non_polarized"],
		contactMode: ["contact", "non_contact"],
		connectivity: "Bluetooth + USB-C",
		weight: "99g",
		ledSpectrum: "4500K",
		price: "$1,295",
	},
	DL5: {
		name: "DermLite DL5",
		magnification: "10x",
		fieldOfView: "25mm",
		resolution: "native (attaches to phone)",
		polarization: ["polarized", "non_polarized"],
		contactMode: ["contact", "non_contact"],
		connectivity: "magnetic phone mount",
		weight: "88g",
		ledSpectrum: "4100K",
		price: "$995",
	},
	DL4: {
		name: "DermLite DL4",
		magnification: "10x",
		fieldOfView: "24mm",
		resolution: "native (attaches to phone)",
		polarization: ["polarized", "non_polarized"],
		contactMode: ["contact"],
		connectivity: "phone adapter",
		weight: "95g",
		ledSpectrum: "4000K",
		price: "$849",
	},
	DL200: {
		name: "DermLite DL200 Hybrid",
		magnification: "10x",
		fieldOfView: "20mm",
		resolution: "native (standalone lens)",
		polarization: ["polarized"],
		contactMode: ["contact", "non_contact"],
		connectivity: "standalone (battery operated)",
		weight: "120g",
		ledSpectrum: "3800K",
		price: "$549",
	},
};
