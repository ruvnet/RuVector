---
name: hypothesis-sweep
description: "Score and rank the candidate mechanisms for the 26-second pulse against current evidence, with contradictions."
---

# hypothesis-sweep

Rank the candidate mechanisms against the current evidence and emit an auditable report.

1. Gather the evidence vector per hypothesis: source stability, environmental correlation,
   out-of-sample prediction, contradiction survival, mechanistic plausibility, citation grounding.
2. Run `rankHypotheses(...)` from `src/score-hypotheses.ts`.
3. For the leading hypothesis, state its **best evidence** AND its **killer contradiction**
   (e.g. gliding tremors appearing during calm-sea windows).
4. Propose the **next test** that would most cheaply move the ranking.
5. Refuse to promote any hypothesis that fails the gate in `src/validate.ts`.

Output a ranked list with confidence, best evidence, contradiction, and next test — never a
single "we solved it" claim.
