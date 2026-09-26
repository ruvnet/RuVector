// SPDX-License-Identifier: MIT
//
// End-to-end tests for the Earth 26-second pulse harness pipeline.
// Offline & deterministic — loads fixtures, exercises detect/extract/embed/
// score/validate.

import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { describe, it, expect } from 'vitest';

import { detectPulse, dominantPeriod } from '../src/detect-26s.js';
import { runPipeline } from '../src/pipeline.js';
import {
  cosineSimilarity,
  embedEvent,
  nearestNeighbors,
} from '../src/embed-events.js';
import { validate } from '../src/validate.js';
import type { EnvironmentContext, SeismicWindow } from '../src/types.js';

const here = dirname(fileURLToPath(import.meta.url));
const fixtures = join(here, '..', 'tests', 'fixtures');

function loadWindow(): SeismicWindow {
  return JSON.parse(
    readFileSync(join(fixtures, 'sample-window.json'), 'utf8'),
  ) as SeismicWindow;
}

function loadEnv(): EnvironmentContext {
  return JSON.parse(
    readFileSync(join(fixtures, 'sample-env.json'), 'utf8'),
  ) as EnvironmentContext;
}

describe('detectPulse', () => {
  it('finds a dominant period in the 24-28s band', () => {
    const window = loadWindow();
    const event = detectPulse(window);
    expect(event).not.toBeNull();
    expect(event!.dominantPeriodS).toBeGreaterThanOrEqual(24);
    expect(event!.dominantPeriodS).toBeLessThanOrEqual(28);
    expect(event!.frequencyHz).toBeGreaterThan(0);
    expect(event!.confidence).toBeGreaterThan(0);
    expect(event!.phaseCoherence).toBeGreaterThan(0);
  });

  it('dominantPeriod helper agrees with the detector', () => {
    const window = loadWindow();
    const p = dominantPeriod(window.samples, window.sampleRateHz, {
      lowPeriodS: 24,
      highPeriodS: 28,
    });
    expect(p).toBeGreaterThanOrEqual(24);
    expect(p).toBeLessThanOrEqual(28);
  });

  it('returns null on pure noise (no in-band peak)', () => {
    // Flat-ish window with no coherent oscillation.
    const flat: SeismicWindow = {
      stationId: 'NOISE',
      startIso: '2023-01-01T00:00:00Z',
      sampleRateHz: 1,
      samples: Array.from({ length: 64 }, () => 0),
    };
    expect(detectPulse(flat)).toBeNull();
  });
});

describe('runPipeline', () => {
  it('produces an event, features, embedding, and ranking', () => {
    const result = runPipeline(loadWindow(), loadEnv());
    expect(result.event).not.toBeNull();
    expect(result.features).toBeDefined();
    expect(result.embedding).toBeDefined();
    expect(result.ranking).toBeDefined();
    expect(result.ranking!.length).toBeGreaterThan(0);
  });

  it('ranks an ocean hypothesis on top for the Gulf of Guinea fixture', () => {
    const result = runPipeline(loadWindow(), loadEnv());
    const top = result.ranking![0].hypothesisId;
    expect(['ocean-shelf-resonance', 'coupled-ocean-geology']).toContain(top);
  });

  it('carries the environment source region onto the event', () => {
    const env = loadEnv();
    const result = runPipeline(loadWindow(), env);
    expect(result.event!.sourceRegion).toBe(env.sourceRegion);
  });
});

describe('embeddings', () => {
  it('cosineSimilarity of a vector with itself is ~1', () => {
    const result = runPipeline(loadWindow(), loadEnv());
    const v = result.embedding!.combined;
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 6);
  });

  it('builds separate waveform/environment/source sub-embeddings', () => {
    const result = runPipeline(loadWindow(), loadEnv());
    const e = result.embedding!;
    expect(e.waveform.length).toBeGreaterThan(0);
    expect(e.environment.length).toBeGreaterThan(0);
    expect(e.source.length).toBeGreaterThan(0);
    expect(e.combined.length).toBe(
      e.waveform.length + e.environment.length + e.source.length,
    );
  });

  it('nearestNeighbors ranks an identical event first', () => {
    const result = runPipeline(loadWindow(), loadEnv());
    const twin = embedEvent({ ...result.features!, eventId: 'twin' });
    const nn = nearestNeighbors(result.embedding!, [twin], 1);
    expect(nn[0].eventId).toBe('twin');
    expect(nn[0].score).toBeCloseTo(1, 6);
  });
});

describe('validate (promotion gate)', () => {
  it('REJECTS when leakage is present', () => {
    const report = validate({
      baselineError: 0.2,
      heldOutError: 0.1,
      pulseDetectionF1Delta: 0.1,
      falsePositiveDelta: -2,
      leakage: true,
      uncitedClaims: 0,
    });
    expect(report.passed).toBe(false);
    expect(report.leakageDetected).toBe(true);
  });

  it('ACCEPTS a clearly improving, clean case', () => {
    const report = validate({
      baselineError: 0.2,
      heldOutError: 0.1,
      pulseDetectionF1Delta: 0.1,
      falsePositiveDelta: -2,
      leakage: false,
      uncitedClaims: 0,
    });
    expect(report.passed).toBe(true);
    expect(report.improvementPct).toBeCloseTo(50, 5);
  });

  it('REJECTS when uncited claims exist', () => {
    const report = validate({
      baselineError: 0.2,
      heldOutError: 0.1,
      pulseDetectionF1Delta: 0.1,
      falsePositiveDelta: 0,
      leakage: false,
      uncitedClaims: 3,
    });
    expect(report.passed).toBe(false);
  });

  it('REJECTS when F1 improvement is below threshold', () => {
    const report = validate({
      baselineError: 0.2,
      heldOutError: 0.1,
      pulseDetectionF1Delta: 0.01,
      falsePositiveDelta: -1,
      leakage: false,
      uncitedClaims: 0,
    });
    expect(report.passed).toBe(false);
  });
});
