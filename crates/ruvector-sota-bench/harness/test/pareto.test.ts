import assert from "node:assert/strict";
import test from "node:test";
import {
  admitToFrontier,
  dominates,
  frontier,
  DEFAULT_OBJECTIVES,
  type ParetoPoint,
} from "../src/pareto.js";

const point = (id: string, primary: number, costPerWin: number): ParetoPoint => ({
  id, values: { primary, costPerWin },
});

test("dominance requires no worse on every objective and better on one", () => {
  const strong = point("strong", 0.9, 10);
  const weak = point("weak", 0.8, 20);
  assert.equal(dominates(strong, weak), true);
  assert.equal(dominates(weak, strong), false);
});

test("a scalar win bought with worse cost does not dominate", () => {
  // Higher primary, strictly worse cost. That is a trade, not a win, so
  // neither point dominates.
  const traded = point("traded", 0.95, 40);
  const incumbent = point("incumbent", 0.90, 10);
  assert.equal(dominates(traded, incumbent), false);
  assert.equal(dominates(incumbent, traded), false);
});

test("identical points do not dominate each other and both stay on the frontier", () => {
  const a = point("a", 0.9, 10);
  const b = point("b", 0.9, 10);
  assert.equal(dominates(a, b), false);
  assert.equal(dominates(b, a), false);
  assert.deepEqual(frontier([a, b]).map((entry) => entry.id), ["a", "b"]);
});

test("frontier keeps the non-dominated set in input order", () => {
  const points = [
    point("dominated", 0.5, 50),
    point("fast", 0.8, 30),
    point("accurate", 0.95, 60),
    point("cheap", 0.7, 5),
  ];
  assert.deepEqual(frontier(points).map((entry) => entry.id), ["fast", "accurate", "cheap"]);
});

test("non-finite objectives are refused on EVERY axis, not just the first", () => {
  // The comparison loop short-circuits on the first axis where a point is
  // worse, so validating inside it would only ever check axes the loop
  // reaches. A bad value on a LATER axis is the case that silently promotes:
  // NaN defeats both `<` and `>`, so it reads as "not worse" everywhere.
  // Every point here is deliberately worse on `primary` (axis 1) so the loop
  // would return before ever seeing axis 2.
  const sane = point("sane", 0.9, 10);
  for (const bad of [Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY]) {
    for (const broken of [
      { id: "axis1", values: { primary: bad, costPerWin: 10 } },
      { id: "axis2", values: { primary: 0.1, costPerWin: bad } },
    ] satisfies ParetoPoint[]) {
      assert.throws(() => dominates(broken, sane), /non-finite/, `${broken.id} a-vs-b ${bad}`);
      assert.throws(() => dominates(sane, broken), /non-finite/, `${broken.id} b-vs-a ${bad}`);
      assert.throws(() => frontier([sane, broken]), /non-finite/, `${broken.id} frontier ${bad}`);
      assert.throws(() => admitToFrontier(broken, [sane]), /non-finite/, `${broken.id} admit ${bad}`);
      assert.throws(() => admitToFrontier(sane, [broken]), /non-finite/, `${broken.id} member ${bad}`);
    }
  }
});

test("a missing objective is refused on any axis, including a later one", () => {
  // The shape an older sealed score carries: a field the current vector
  // expects simply is not there.
  const sane = point("sane", 0.9, 10);
  const missingFirst: ParetoPoint = { id: "missingFirst", values: { costPerWin: 10 } };
  const missingLater: ParetoPoint = { id: "missingLater", values: { primary: 0.1 } };
  for (const partial of [missingFirst, missingLater]) {
    assert.throws(() => dominates(partial, sane), /missing objective/, partial.id);
    assert.throws(() => dominates(sane, partial), /missing objective/, partial.id);
    assert.throws(() => admitToFrontier(partial, [sane]), /missing objective/, partial.id);
    assert.throws(() => admitToFrontier(sane, [partial]), /missing objective/, partial.id);
  }
});

test("adding an axis can only weaken the gate, which is why the vector is two", () => {
  // Records the MEDIUM-3 reasoning as an executable fact. Same two points:
  // dominated on the shipped vector, NOT dominated once a third axis on which
  // the candidate happens to be better is added.
  const incumbent: ParetoPoint = { id: "incumbent", values: { primary: 0.9, costPerWin: 10, p99Us: 100 } };
  const heavy: ParetoPoint = { id: "heavy", values: { primary: 0.9, costPerWin: 50, p99Us: 50 } };
  assert.equal(dominates(incumbent, heavy), true);
  assert.equal(
    dominates(incumbent, heavy, [
      { name: "primary", direction: "maximize" },
      { name: "costPerWin", direction: "minimize" },
      { name: "p99Us", direction: "minimize" },
    ]),
    false,
  );
  assert.deepEqual(DEFAULT_OBJECTIVES.map((o) => o.name), ["primary", "costPerWin"]);
});

test("admission blocks a dominated candidate and names the dominators", () => {
  const current = [point("incumbent", 0.9, 10), point("other", 0.85, 8)];
  const worse = point("worse", 0.5, 50);
  const admission = admitToFrontier(worse, current);
  assert.equal(admission.admitted, false);
  assert.deepEqual(admission.dominatedBy, ["incumbent", "other"]);
});

test("an empty frontier admits, because absence of evidence is not domination", () => {
  assert.deepEqual(admitToFrontier(point("first", 0.1, 999), []), {
    admitted: true, dominatedBy: [],
  });
});

test("objective specs are validated before any comparison", () => {
  const a = point("a", 0.9, 10);
  const b = point("b", 0.8, 20);
  assert.throws(() => dominates(a, b, []), /at least one objective/);
  assert.throws(
    () => dominates(a, b, [{ name: "primary", direction: "sideways" as never }]),
    /maximize or minimize/,
  );
  assert.throws(
    () => dominates(a, b, [...DEFAULT_OBJECTIVES, { name: "primary", direction: "maximize" }]),
    /duplicate/,
  );
});
