/**
 * Compile-time proof of the schema-narrowed query verbs. Checked by
 * `tsc --noEmit -p tsconfig.typetest.json`; never emitted or run. Each
 * `@ts-expect-error` must stay an error — if narrowing regresses, TypeScript
 * reports the now-unused directive and the type test fails.
 */

import { createKge, defineSchema } from '../index';

const schema = defineSchema({ relations: ['bornIn', 'locatedIn'] as const });
const kge = createKge({ scorer: 'hole', schema });

function narrowing(): void {
  // predict accepts a schema relation, in either slot form.
  const tail = kge.predict({ s: 'Ada', r: 'bornIn', k: 5 });
  const head = kge.predict({ o: 'London', r: 'locatedIn' });
  const best: string = tail.candidates[0].entity;
  const alt: number = head.candidates[0].score;

  // compose and similarRelations narrow the same way.
  const composed = kge.compose({ r1: 'bornIn', r2: 'locatedIn', s: 'Ada' });
  const sim = kge.similarRelations({ r: 'bornIn' });

  // A relation not in the schema is a compile error.
  // @ts-expect-error 'marriedTo' is not one of the schema relations
  kge.predict({ s: 'Ada', r: 'marriedTo' });
  // @ts-expect-error 'unknown' is not a schema relation
  kge.similarRelations({ r: 'unknown' });
  // @ts-expect-error a query may not fill both s and o
  kge.predict({ s: 'Ada', o: 'London', r: 'bornIn' });

  void [best, alt, composed, sim];
}

// Without a schema, the relation argument is a plain string.
function unschemed(): void {
  const anyKge = createKge();
  const r = anyKge.predict({ s: 'x', r: 'any-relation-string' });
  const label: string = r.candidates[0].entity;
  void label;
}

void narrowing;
void unschemed;
