/**
 * DrAgnes Configuration
 *
 * Central configuration for the DrAgnes dermatology intelligence module.
 * Controls CNN backbone, embedding dimensions, class taxonomy,
 * privacy parameters, brain sync, and performance budgets.
 */

export interface DragnesClassLabels {
  akiec: string;
  bcc: string;
  bkl: string;
  df: string;
  mel: string;
  nv: string;
  vasc: string;
}

export interface DragnesPrivacy {
  dpEpsilon: number;
  kAnonymity: number;
  witnessAlgorithm: string;
}

export interface DragnesBrain {
  url: string;
  namespace: string;
  syncIntervalMs: number;
}

export interface DragnesPerformance {
  maxInferenceMs: number;
  maxModelSizeMb: number;
}

export interface DragnesConfig {
  modelVersion: string;
  cnnBackbone: string;
  embeddingDim: number;
  projectedDim: number;
  classes: string[];
  classLabels: DragnesClassLabels;
  privacy: DragnesPrivacy;
  brain: DragnesBrain;
  performance: DragnesPerformance;
}

export const DRAGNES_CONFIG: DragnesConfig = {
  modelVersion: '0.1.0',
  cnnBackbone: 'mobilenet-v3-small',
  embeddingDim: 576,
  projectedDim: 128,
  classes: ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'],
  classLabels: {
    akiec: 'Actinic Keratosis',
    bcc: 'Basal Cell Carcinoma',
    bkl: 'Benign Keratosis',
    df: 'Dermatofibroma',
    mel: 'Melanoma',
    nv: 'Melanocytic Nevus',
    vasc: 'Vascular Lesion',
  },
  privacy: {
    dpEpsilon: 1.0,
    kAnonymity: 5,
    witnessAlgorithm: 'SHA-256',
  },
  brain: {
    url: 'https://pi.ruv.io',
    namespace: 'dragnes',
    syncIntervalMs: 300_000,
  },
  performance: {
    maxInferenceMs: 200,
    maxModelSizeMb: 5,
  },
};
