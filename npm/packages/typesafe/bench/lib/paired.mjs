// JS port of the campaign's anytime-valid paired test,
// `PairedSequentialTest` in crates/ruvector-typesafe-core/src/loop_gate/sequential.rs
// (ADR-004 gate 2; ADR-007 §1b). The Rust type is not exposed through the
// binding, so this is a line-for-line port — same parameters, same clamps, same
// f32 rounding of alpha/lambda, same f64 wealth arithmetic — proven identical by
// bench/test/paired-golden.json (wealth paths emitted by the Rust type itself).
//
// For each DISCORDANT pair (one side right, the other wrong) wealth updates
// W ← W·(1 + λ·(X − ½)), X = 1 iff the challenger ("champion") won the pair.
// Concordant pairs are skipped. Rejection latches the first time the running
// maximum of W reaches 1/α and never un-latches. α = 0.05, λ = 0.5 are fixed by
// ADR-007 and never tuned.

const f32 = Math.fround;

export class PairedSequentialTest {
  /** `alpha` in [1e-6, 0.5], `lambda` in [0.01, 1.99] — clamped exactly like Rust. */
  constructor(alpha = 0.05, lambda = 0.5) {
    // Rust stores both as f32 and clamps in f32; fround reproduces that.
    this.alpha = Math.min(Math.max(f32(alpha), f32(1e-6)), f32(0.5));
    this.lambda = Math.min(Math.max(f32(lambda), f32(0.01)), f32(1.99));
    this.wealth = 1.0;
    this.maxWealth = 1.0;
    this.nChampionWins = 0;
    this.nBaselineWins = 0;
    this.rejected = false;
    this.nDiscordantAtRejection = null;
  }

  /** α = 0.05, λ = 0.5 (threshold 1/f32(0.05) = 19.99999970197678). */
  static standard() {
    return new PairedSequentialTest(0.05, 0.5);
  }

  get threshold() {
    return 1.0 / this.alpha;
  }

  /** Feed one pair. Returns `true` once the rejection has latched. */
  update(baselineCorrect, championCorrect) {
    const b = !!baselineCorrect;
    const c = !!championCorrect;
    if (b === c) return this.rejected; // concordant: no information
    const championWon = c && !b;
    const x = championWon ? 1.0 : 0.0;
    if (championWon) this.nChampionWins += 1;
    else this.nBaselineWins += 1;
    this.wealth *= 1.0 + this.lambda * (x - 0.5);
    if (this.wealth > this.maxWealth) this.maxWealth = this.wealth;
    if (!this.rejected && this.maxWealth >= this.threshold) {
      this.rejected = true;
      this.nDiscordantAtRejection = this.nChampionWins + this.nBaselineWins;
    }
    return this.rejected;
  }

  /** Feed `[[baselineCorrect, championCorrect], …]` in order. */
  updateAll(pairs) {
    for (const [b, c] of pairs) this.update(b, c);
    return this.rejected;
  }

  /** Same field names as Rust's `receipt::TestStatistic` (serde). */
  statistic() {
    const s = {
      alpha: this.alpha,
      lambda: this.lambda,
      wealth: this.wealth,
      max_wealth: this.maxWealth,
      threshold: this.threshold,
      n_champion_wins: this.nChampionWins,
      n_baseline_wins: this.nBaselineWins,
      rejected: this.rejected,
    };
    if (this.nDiscordantAtRejection !== null) s.n_discordant_at_rejection = this.nDiscordantAtRejection;
    return s;
  }
}
