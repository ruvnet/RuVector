/**
 * Compile-time proof of the criteria-keyed generics. Checked by
 * `tsc --noEmit -p tsconfig.typetest.json`; never emitted or run. Each
 * `@ts-expect-error` must stay an error — if narrowing regresses, TypeScript
 * reports the now-unused directive and the type test fails.
 */

import type { Typesafe } from '../index';
import { choice, noul, score } from '../index';

declare const ts: Typesafe;

async function narrowing(): Promise<void> {
  const r = await ts.decide('my card was charged twice', {
    dept: choice({
      billing: 'charges and refunds',
      fraud: { what: 'unauthorised use', not_for: 'duplicate charges' },
    }),
    mood: score(['Calm', 'Irritated', 'Angry'] as const),
    urgent: noul('the sender needs a response soon'),
  });

  // choice narrows to the union of the criteria keys, top-level and under .answers.
  const winner: 'billing' | 'fraud' = r.dept.choice;
  const winner2: 'billing' | 'fraud' = r.answers.dept.choice;
  const billingProb: number = r.dept.probabilities.billing;
  // score legend narrows to the union of bucket labels.
  const mood: 'Calm' | 'Irritated' | 'Angry' = r.mood.legend;
  const bucket: number = r.mood.score;
  // noul is a number.
  const urgency: number = r.urgent.noul;

  // A key that is not in the criteria is a compile error.
  // @ts-expect-error 'shipping' is not one of the criteria keys
  const wrong: 'shipping' = r.dept.choice;
  // @ts-expect-error probabilities has no 'shipping' member
  const missingProb = r.dept.probabilities.shipping;
  // @ts-expect-error 'Furious' is not a legend bucket
  const wrongMood: 'Furious' = r.mood.legend;

  void [winner, winner2, billingProb, mood, bucket, urgency, wrong, missingProb, wrongMood];
}

// Without `as const`: the `const` type parameter still narrows the legend.
async function narrowingWithoutAsConst(): Promise<void> {
  const r = await ts.decide('hello', {
    mood: score(['low', 'high']),
  });
  const m: 'low' | 'high' = r.mood.legend;
  // @ts-expect-error 'medium' is not a legend bucket
  const bad: 'medium' = r.mood.legend;
  void [m, bad];
}

void narrowing;
void narrowingWithoutAsConst;
