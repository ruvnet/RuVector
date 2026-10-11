# Preserve null predicates in Python Cypher queries

Status: proposed for review

## Problem and evidence

The Python graph evaluator lowers missing properties to Null but converts comparisons and boolean operands into false. Negation then admits missing-property nodes. Against fresh upstream07933730 native source, nine public GraphDB.query_cypher regression cases fail and eight controls pass. For example, NOT (n.x = 1 OR n.x = 2) returns only the missing-property records, although neither comparison is known for those records.

Cypher null predicates remain unknown, and WHERE admits only true. The reference truth tables are documented at https://neo4j.com/docs/cypher-manual/4.4/values-and-types/working-with-null/ .

## Decision

Keep EvalValue::Null through null comparisons and negation. Use three-valued logical operators: false dominates AND, true dominates OR, and an unknown operand makes XOR unknown. Keep the existing WHERE truthy boundary and supported query surface. IS NULL remains exempt from propagation. Preserve AND/OR decisive-left short-circuit behavior.

## Alternatives and limits

Rejecting every missing-property query would discard useful complete records. Treating unknown as false before NOT changes the result set incorrectly. This change does not implement projection, joins, regex, lexer operators, graph transactions or general expression typing. Nested-list equality is outside this targeted scalar predicate repair. The NAPI executor has separate code and is not changed here.

## Validation

Seventeen public Python queries cover absent/explicit-null values, known scalar values, comparison negation, AND/OR/XOR unknown propagation, and decisive false/true controls. Baseline: nine fail, eight pass, exact upstream source-built native artifact. Fresh corrected native, graph regression suite, scoped Rust checks and independent exact-commit review must pass before publication; results are preserved in the private campaign evidence.
