/**
 * Optional compile-time schema. `defineSchema({ relations: [...] as const })`
 * captures the relation labels as a string-literal union, so a model created
 * with that schema narrows the `r` argument of `predict` / `compose` /
 * `similarRelations` to exactly those labels. Runtime behaviour is unchanged —
 * the schema is a type-level convenience, not a validator.
 */

/** A schema carrying the relation-label union `R` (and optional entity union). */
export interface Schema<R extends string = string, E extends string = string> {
  readonly relations: readonly R[];
  readonly entities?: readonly E[];
}

/**
 * Build a {@link Schema}. The `const` type parameters infer the literal unions
 * from the arrays, so `defineSchema({ relations: ['a','b'] })` narrows without
 * an explicit `as const`.
 */
export function defineSchema<
  const R extends readonly string[],
  const E extends readonly string[] = readonly string[],
>(schema: { relations: R; entities?: E }): Schema<R[number], E[number]> {
  return { relations: schema.relations, entities: schema.entities };
}

/** The relation union carried by a schema. */
export type RelationOf<S> = S extends Schema<infer R, string> ? R : string;

/** The entity union carried by a schema (defaults to `string`). */
export type EntityOf<S> = S extends Schema<string, infer E> ? E : string;
