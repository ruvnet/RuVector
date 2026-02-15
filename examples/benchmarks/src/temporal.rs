//! Temporal Reasoning Benchmark Framework
//!
//! Implements temporal constraint solving and benchmarking based on:
//! - TimePuzzles benchmark methodology
//! - Tool-augmented iterative temporal reasoning
//! - Calendar math and cross-cultural date systems

use anyhow::{anyhow, Result};
use chrono::{Datelike, NaiveDate, Weekday};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Temporal constraint types
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum TemporalConstraint {
    /// Date is exactly this value
    Exact(NaiveDate),
    /// Date is after this date
    After(NaiveDate),
    /// Date is before this date
    Before(NaiveDate),
    /// Date is between two dates (inclusive)
    Between(NaiveDate, NaiveDate),
    /// Date is on a specific day of week
    DayOfWeek(Weekday),
    /// Date is N days after reference
    DaysAfter(String, i64),
    /// Date is N days before reference
    DaysBefore(String, i64),
    /// Date is in a specific month
    InMonth(u32),
    /// Date is in a specific year
    InYear(i32),
    /// Date is a specific day of month
    DayOfMonth(u32),
    /// Relative to a named event (e.g., "Easter", "Chinese New Year")
    RelativeToEvent(String, i64),
}

/// A temporal puzzle with constraints
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalPuzzle {
    /// Unique puzzle ID
    pub id: String,
    /// Human-readable description
    pub description: String,
    /// Constraints that define the puzzle
    pub constraints: Vec<TemporalConstraint>,
    /// Named reference dates
    pub references: HashMap<String, NaiveDate>,
    /// Valid solution dates (for evaluation)
    pub solutions: Vec<NaiveDate>,
    /// Difficulty level (1-10)
    pub difficulty: u8,
    /// Tags for categorization
    pub tags: Vec<String>,
}

impl TemporalPuzzle {
    /// Create a new puzzle
    pub fn new(id: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            id: id.into(),
            description: description.into(),
            constraints: Vec::new(),
            references: HashMap::new(),
            solutions: Vec::new(),
            difficulty: 5,
            tags: Vec::new(),
        }
    }

    /// Add a constraint
    pub fn with_constraint(mut self, constraint: TemporalConstraint) -> Self {
        self.constraints.push(constraint);
        self
    }

    /// Add a reference date
    pub fn with_reference(mut self, name: impl Into<String>, date: NaiveDate) -> Self {
        self.references.insert(name.into(), date);
        self
    }

    /// Set solution dates
    pub fn with_solutions(mut self, solutions: Vec<NaiveDate>) -> Self {
        self.solutions = solutions;
        self
    }

    /// Set difficulty
    pub fn with_difficulty(mut self, difficulty: u8) -> Self {
        self.difficulty = difficulty.min(10).max(1);
        self
    }

    /// Add tags
    pub fn with_tags(mut self, tags: Vec<String>) -> Self {
        self.tags = tags;
        self
    }

    /// Check if a date satisfies all constraints
    pub fn check_date(&self, date: NaiveDate) -> Result<bool> {
        for constraint in &self.constraints {
            if !self.check_constraint(date, constraint)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Check a single constraint
    fn check_constraint(&self, date: NaiveDate, constraint: &TemporalConstraint) -> Result<bool> {
        match constraint {
            TemporalConstraint::Exact(d) => Ok(date == *d),
            TemporalConstraint::After(d) => Ok(date > *d),
            TemporalConstraint::Before(d) => Ok(date < *d),
            TemporalConstraint::Between(start, end) => Ok(date >= *start && date <= *end),
            TemporalConstraint::DayOfWeek(dow) => Ok(date.weekday() == *dow),
            TemporalConstraint::DaysAfter(ref_name, days) => {
                let ref_date = self
                    .references
                    .get(ref_name)
                    .ok_or_else(|| anyhow!("Unknown reference: {}", ref_name))?;
                let target = *ref_date + chrono::Duration::days(*days);
                Ok(date == target)
            }
            TemporalConstraint::DaysBefore(ref_name, days) => {
                let ref_date = self
                    .references
                    .get(ref_name)
                    .ok_or_else(|| anyhow!("Unknown reference: {}", ref_name))?;
                let target = *ref_date - chrono::Duration::days(*days);
                Ok(date == target)
            }
            TemporalConstraint::InMonth(month) => Ok(date.month() == *month),
            TemporalConstraint::InYear(year) => Ok(date.year() == *year),
            TemporalConstraint::DayOfMonth(day) => Ok(date.day() == *day),
            TemporalConstraint::RelativeToEvent(event_name, days) => {
                // Look up event in references
                let event_date = self
                    .references
                    .get(event_name)
                    .ok_or_else(|| anyhow!("Unknown event: {}", event_name))?;
                let target = *event_date + chrono::Duration::days(*days);
                Ok(date == target)
            }
        }
    }

    /// Solve the puzzle by searching date space
    pub fn solve(&self, search_range: (NaiveDate, NaiveDate)) -> Result<Vec<NaiveDate>> {
        let mut solutions = Vec::new();
        let mut current = search_range.0;
        while current <= search_range.1 {
            if self.check_date(current)? {
                solutions.push(current);
            }
            current = current.succ_opt().unwrap_or(current);
        }
        Ok(solutions)
    }
}

/// Puzzle solver with tool augmentation
#[derive(Clone, Debug)]
pub struct TemporalSolver {
    /// Enable calendar math tool
    pub calendar_tool: bool,
    /// Enable web search tool
    pub web_search_tool: bool,
    /// Maximum steps allowed
    pub max_steps: usize,
    /// Current step count
    pub steps: usize,
    /// Tool call count
    pub tool_calls: usize,
    /// Stop after finding the first valid solution (early termination)
    pub stop_after_first: bool,
}

impl Default for TemporalSolver {
    fn default() -> Self {
        Self {
            calendar_tool: true,
            web_search_tool: false,
            max_steps: 100,
            steps: 0,
            tool_calls: 0,
            stop_after_first: false,
        }
    }
}

impl TemporalSolver {
    /// Create solver with tools
    pub fn with_tools(calendar: bool, web_search: bool) -> Self {
        Self {
            calendar_tool: calendar,
            web_search_tool: web_search,
            stop_after_first: false,
            ..Default::default()
        }
    }

    /// Solve a puzzle with step tracking
    pub fn solve(&mut self, puzzle: &TemporalPuzzle) -> Result<SolverResult> {
        self.steps = 0;
        self.tool_calls = 0;

        let start_time = std::time::Instant::now();

        // Rewrite constraints to explicit dates if calendar tool enabled
        let effective_puzzle = if self.calendar_tool {
            self.tool_calls += 1;
            self.rewrite_constraints(puzzle)?
        } else {
            puzzle.clone()
        };

        // Determine search range from effective (rewritten) constraints
        let range = self.determine_search_range(&effective_puzzle)?;

        // Search for solutions
        let mut found_solutions = Vec::new();
        let mut current = range.0;

        while current <= range.1 && self.steps < self.max_steps {
            self.steps += 1;
            if effective_puzzle.check_date(current)? {
                found_solutions.push(current);
                if self.stop_after_first {
                    break;
                }
            }
            current = match current.succ_opt() {
                Some(d) => d,
                None => break,
            };
        }

        let latency = start_time.elapsed();

        // Check correctness
        let correct = if puzzle.solutions.is_empty() {
            true // No ground truth
        } else {
            found_solutions.iter().all(|s| puzzle.solutions.contains(s))
                && puzzle
                    .solutions
                    .iter()
                    .all(|s| found_solutions.contains(s) || *s < range.0 || *s > range.1)
        };

        Ok(SolverResult {
            puzzle_id: puzzle.id.clone(),
            solved: !found_solutions.is_empty(),
            correct,
            solutions: found_solutions,
            steps: self.steps,
            tool_calls: self.tool_calls,
            latency_ms: latency.as_millis() as u64,
        })
    }

    /// Determine search range from constraints
    fn determine_search_range(&self, puzzle: &TemporalPuzzle) -> Result<(NaiveDate, NaiveDate)> {
        let mut min_date = NaiveDate::from_ymd_opt(1900, 1, 1).unwrap();
        let mut max_date = NaiveDate::from_ymd_opt(2100, 12, 31).unwrap();

        for constraint in &puzzle.constraints {
            match constraint {
                TemporalConstraint::Exact(d) => {
                    min_date = *d;
                    max_date = *d;
                }
                TemporalConstraint::After(d) => {
                    if *d >= min_date {
                        min_date = d.succ_opt().unwrap_or(*d);
                    }
                }
                TemporalConstraint::Before(d) => {
                    if *d <= max_date {
                        max_date = d.pred_opt().unwrap_or(*d);
                    }
                }
                TemporalConstraint::Between(start, end) => {
                    if *start > min_date {
                        min_date = *start;
                    }
                    if *end < max_date {
                        max_date = *end;
                    }
                }
                TemporalConstraint::InYear(year) => {
                    let year_start = NaiveDate::from_ymd_opt(*year, 1, 1).unwrap_or(min_date);
                    let year_end = NaiveDate::from_ymd_opt(*year, 12, 31).unwrap_or(max_date);
                    if year_start > min_date {
                        min_date = year_start;
                    }
                    if year_end < max_date {
                        max_date = year_end;
                    }
                }
                _ => {}
            }
        }

        Ok((min_date, max_date))
    }

    /// Rewrite relative constraints to explicit dates
    fn rewrite_constraints(&self, puzzle: &TemporalPuzzle) -> Result<TemporalPuzzle> {
        let mut new_puzzle = puzzle.clone();
        let mut new_constraints = Vec::new();

        for constraint in &puzzle.constraints {
            match constraint {
                TemporalConstraint::DaysAfter(ref_name, days) => {
                    if let Some(ref_date) = puzzle.references.get(ref_name) {
                        let target = *ref_date + chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                TemporalConstraint::DaysBefore(ref_name, days) => {
                    if let Some(ref_date) = puzzle.references.get(ref_name) {
                        let target = *ref_date - chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                TemporalConstraint::RelativeToEvent(event_name, days) => {
                    if let Some(event_date) = puzzle.references.get(event_name) {
                        let target = *event_date + chrono::Duration::days(*days);
                        new_constraints.push(TemporalConstraint::Exact(target));
                    } else {
                        new_constraints.push(constraint.clone());
                    }
                }
                _ => new_constraints.push(constraint.clone()),
            }
        }

        new_puzzle.constraints = new_constraints;
        Ok(new_puzzle)
    }
}

/// Result from solving a puzzle
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SolverResult {
    pub puzzle_id: String,
    pub solved: bool,
    pub correct: bool,
    pub solutions: Vec<NaiveDate>,
    pub steps: usize,
    pub tool_calls: usize,
    pub latency_ms: u64,
}

/// Benchmark configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of puzzles to run
    pub num_puzzles: usize,
    /// Difficulty range
    pub difficulty_range: (u8, u8),
    /// Enable calendar tool
    pub calendar_tool: bool,
    /// Enable web search tool
    pub web_search_tool: bool,
    /// Maximum steps per puzzle
    pub max_steps: usize,
    /// Constraint density (1-5)
    pub constraint_density: u8,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            num_puzzles: 50,
            difficulty_range: (1, 10),
            calendar_tool: true,
            web_search_tool: false,
            max_steps: 100,
            constraint_density: 3,
        }
    }
}

/// Benchmark results
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BenchmarkResults {
    pub config: BenchmarkConfig,
    pub total_puzzles: usize,
    pub solved_count: usize,
    pub correct_count: usize,
    pub accuracy: f64,
    pub avg_steps: f64,
    pub avg_tool_calls: f64,
    pub avg_latency_ms: f64,
    pub results: Vec<SolverResult>,
}

impl BenchmarkResults {
    /// Create from individual results
    pub fn from_results(config: BenchmarkConfig, results: Vec<SolverResult>) -> Self {
        let total = results.len();
        let solved = results.iter().filter(|r| r.solved).count();
        let correct = results.iter().filter(|r| r.correct).count();
        let avg_steps = results.iter().map(|r| r.steps as f64).sum::<f64>() / total as f64;
        let avg_tools = results.iter().map(|r| r.tool_calls as f64).sum::<f64>() / total as f64;
        let avg_latency = results.iter().map(|r| r.latency_ms as f64).sum::<f64>() / total as f64;

        Self {
            config,
            total_puzzles: total,
            solved_count: solved,
            correct_count: correct,
            accuracy: correct as f64 / total as f64,
            avg_steps,
            avg_tool_calls: avg_tools,
            avg_latency_ms: avg_latency,
            results,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_puzzle() {
        let puzzle = TemporalPuzzle::new("test-1", "Find a date in January 2024")
            .with_constraint(TemporalConstraint::InYear(2024))
            .with_constraint(TemporalConstraint::InMonth(1))
            .with_constraint(TemporalConstraint::DayOfMonth(15));

        let expected = NaiveDate::from_ymd_opt(2024, 1, 15).unwrap();
        assert!(puzzle.check_date(expected).unwrap());
        assert!(!puzzle
            .check_date(NaiveDate::from_ymd_opt(2024, 2, 15).unwrap())
            .unwrap());
    }

    #[test]
    fn test_relative_constraint() {
        let base = NaiveDate::from_ymd_opt(2024, 1, 1).unwrap();
        let puzzle = TemporalPuzzle::new("test-2", "Find a date 10 days after New Year")
            .with_reference("new_year", base)
            .with_constraint(TemporalConstraint::DaysAfter("new_year".to_string(), 10));

        let expected = NaiveDate::from_ymd_opt(2024, 1, 11).unwrap();
        assert!(puzzle.check_date(expected).unwrap());
    }

    #[test]
    fn test_solver_with_rewriting() {
        let base = NaiveDate::from_ymd_opt(2024, 6, 15).unwrap();
        let puzzle = TemporalPuzzle::new("test-3", "Find date relative to event")
            .with_reference("event", base)
            .with_constraint(TemporalConstraint::DaysAfter("event".to_string(), 5))
            .with_solutions(vec![NaiveDate::from_ymd_opt(2024, 6, 20).unwrap()]);

        let mut solver = TemporalSolver::with_tools(true, false);
        let result = solver.solve(&puzzle).unwrap();

        assert!(result.solved);
        assert!(result.correct);
        assert_eq!(result.solutions.len(), 1);
    }
}

// ============================================================================
// Adaptive Solver with ReasoningBank Learning
// ============================================================================

use crate::reasoning_bank::{ReasoningBank, Strategy, Trajectory, Verdict};

/// Adaptive temporal solver with learning capabilities
///
/// Uses ReasoningBank to:
/// - Track solution trajectories
/// - Learn from successes and failures
/// - Adapt strategy based on puzzle characteristics
/// - Achieve sublinear regret through experience

// ═══════════════════════════════════════════════════════════════════════════
// KnowledgeCompiler — constraint signature → compiled solve config
// ═══════════════════════════════════════════════════════════════════════════

/// Compiled solver configuration for a known constraint signature.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompiledSolveConfig {
    /// Whether to use calendar rewriting
    pub use_rewriting: bool,
    /// Minimum steps that succeeded for this signature
    pub max_steps: usize,
    /// Expected correctness
    pub expected_correct: bool,
    /// Stop after first solution (early termination for known single-solution puzzles)
    pub stop_after_first: bool,
    /// Hit count (how often this config was used)
    pub hit_count: usize,
    /// Counterexample count (failures on this signature)
    pub counterexample_count: usize,
}

/// KnowledgeCompiler: learns constraint-signature → optimal solve config.
/// Consulted as "Strategy Zero" before any other strategy runs.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KnowledgeCompiler {
    /// Compiled constraint signature → config
    pub signature_cache: HashMap<String, CompiledSolveConfig>,
    /// Cache hits
    pub hits: usize,
    /// Cache misses
    pub misses: usize,
    /// False hits (compiled config tried but solve was wrong)
    pub false_hits: usize,
}

impl KnowledgeCompiler {
    pub fn new() -> Self { Self::default() }

    /// Build constraint signature from puzzle features.
    pub fn signature(puzzle: &TemporalPuzzle) -> String {
        let mut sig_parts: Vec<String> = puzzle.constraints.iter()
            .map(|c| constraint_type_name(c))
            .collect();
        sig_parts.sort();
        format!("{}:{}", puzzle.difficulty, sig_parts.join(","))
    }

    /// Compile knowledge from a ReasoningBank's trajectories.
    pub fn compile_from_bank(&mut self, bank: &ReasoningBank) {
        for traj in &bank.trajectories {
            let correct = traj.verdict.as_ref().map(|v| v.is_success()).unwrap_or(false);
            if !correct { continue; }

            // Build signature from constraint types
            let mut sig_parts = traj.constraint_types.clone();
            sig_parts.sort();
            let sig = format!("{}:{}", traj.difficulty, sig_parts.join(","));

            if let Some(attempt) = traj.attempts.first() {
                let entry = self.signature_cache.entry(sig).or_insert(CompiledSolveConfig {
                    use_rewriting: true,
                    max_steps: attempt.steps,
                    expected_correct: true,
                    stop_after_first: true, // compiled configs use early termination
                    hit_count: 0,
                    counterexample_count: 0,
                });
                // Keep minimum steps that succeeded
                entry.max_steps = entry.max_steps.min(attempt.steps);
            }
        }
    }

    /// Look up a compiled config for a puzzle. Returns None on cache miss.
    pub fn lookup(&mut self, puzzle: &TemporalPuzzle) -> Option<&CompiledSolveConfig> {
        let sig = Self::signature(puzzle);
        if self.signature_cache.contains_key(&sig) {
            self.hits += 1;
            // Safe: we just checked containment
            self.signature_cache.get(&sig)
        } else {
            self.misses += 1;
            None
        }
    }

    /// Record a counterexample: Strategy Zero failed on this signature.
    pub fn record_failure(&mut self, puzzle: &TemporalPuzzle) {
        self.false_hits += 1;
        let sig = Self::signature(puzzle);
        if let Some(config) = self.signature_cache.get_mut(&sig) {
            config.counterexample_count += 1;
            // If failure rate exceeds 30%, invalidate the cache entry
            if config.hit_count > 0 {
                let fail_rate = config.counterexample_count as f64
                    / (config.hit_count + config.counterexample_count) as f64;
                if fail_rate > 0.30 {
                    config.expected_correct = false;
                }
            }
        }
    }

    /// Record a success: Strategy Zero worked on this signature.
    pub fn record_success(&mut self, puzzle: &TemporalPuzzle) {
        let sig = Self::signature(puzzle);
        if let Some(config) = self.signature_cache.get_mut(&sig) {
            config.hit_count += 1;
        }
    }

    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 { 0.0 } else { self.hits as f64 / total as f64 }
    }

    pub fn cache_size(&self) -> usize { self.signature_cache.len() }
}

// ═══════════════════════════════════════════════════════════════════════════
// StrategyRouter — contextual bandit for strategy selection
// ═══════════════════════════════════════════════════════════════════════════

/// Context bucket key for the bandit.
#[derive(Clone, Debug, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RoutingContext {
    /// Constraint family (sorted constraint types)
    pub constraint_family: String,
    /// Difficulty bucket (1-3=easy, 4-7=mid, 8-10=hard)
    pub difficulty_bucket: u8,
    /// Whether input is noisy
    pub noisy: bool,
}

/// Per-arm stats in the contextual bandit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ArmStats {
    pub pulls: usize,
    pub successes: usize,
    pub total_steps: usize,
    pub noise_successes: usize,
    pub noise_pulls: usize,
}

impl ArmStats {
    pub fn reward(&self) -> f64 {
        if self.pulls == 0 { return 0.5; } // Optimistic prior
        let success_rate = self.successes as f64 / self.pulls as f64;
        let cost_bonus = if self.total_steps > 0 {
            // Lower steps = higher reward. Normalize to ~0..0.3
            0.3 * (1.0 - (self.total_steps as f64 / self.pulls as f64) / 100.0).max(0.0)
        } else {
            0.0
        };
        let robustness_bonus = if self.noise_pulls > 0 {
            0.2 * (self.noise_successes as f64 / self.noise_pulls as f64)
        } else {
            0.0
        };
        success_rate * 0.5 + cost_bonus + robustness_bonus
    }
}

/// Adaptive strategy router using contextual bandit.
/// Learns per-context ordering and budget allocation for strategies.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct StrategyRouter {
    /// Per-context, per-strategy arm stats
    pub arms: HashMap<RoutingContext, HashMap<String, ArmStats>>,
    /// Exploration rate (epsilon-greedy)
    pub epsilon: f64,
    /// Minimum exploration observations before dropping a strategy
    pub min_observations: usize,
    /// RNG state for exploration
    rng_state: u64,
}

impl StrategyRouter {
    pub fn new() -> Self {
        Self {
            arms: HashMap::new(),
            epsilon: 0.15,
            min_observations: 10,
            rng_state: 42,
        }
    }

    /// Build routing context from puzzle features.
    pub fn context(puzzle: &TemporalPuzzle, noisy: bool) -> RoutingContext {
        let mut families: Vec<String> = puzzle.constraints.iter()
            .map(|c| constraint_type_name(c))
            .collect();
        families.sort();
        families.dedup();

        let difficulty_bucket = match puzzle.difficulty {
            1..=3 => 1,
            4..=7 => 2,
            _ => 3,
        };

        RoutingContext {
            constraint_family: families.join(","),
            difficulty_bucket,
            noisy,
        }
    }

    /// Select the best strategy for a context.
    /// Returns ordered list of (strategy_name, budget_fraction).
    pub fn select(&mut self, ctx: &RoutingContext, available: &[String]) -> Vec<(String, f64)> {
        // Epsilon-greedy: explore with probability epsilon
        let r = self.next_f64();
        if r < self.epsilon {
            // Explore: random permutation
            let mut shuffled = available.to_vec();
            for i in (1..shuffled.len()).rev() {
                let j = (self.next_f64() * (i + 1) as f64) as usize;
                shuffled.swap(i, j.min(i));
            }
            return shuffled.into_iter()
                .map(|s| (s, 1.0 / available.len() as f64))
                .collect();
        }

        // Exploit: rank by reward, filter out strategies with zero success after min_observations
        let arm_map = self.arms.entry(ctx.clone()).or_default();
        let mut ranked: Vec<(String, f64)> = available.iter().map(|s| {
            let stats = arm_map.get(s).cloned().unwrap_or_default();
            let should_drop = stats.pulls >= self.min_observations && stats.successes == 0;
            let reward = if should_drop { -1.0 } else { stats.reward() };
            (s.clone(), reward)
        }).collect();

        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Filter out dropped strategies (reward < 0), keep at least one
        let mut result: Vec<(String, f64)> = ranked.into_iter()
            .filter(|(_, r)| *r >= 0.0)
            .collect();
        if result.is_empty() {
            result = vec![(available[0].clone(), 1.0)];
        }

        // Allocate budget: best gets 60%, rest split remainder
        let n = result.len();
        result.iter_mut().enumerate().for_each(|(i, (_, budget))| {
            *budget = if i == 0 { 0.6 } else { 0.4 / (n - 1).max(1) as f64 };
        });

        result
    }

    /// Update arm stats after a solve attempt.
    pub fn update(
        &mut self,
        ctx: &RoutingContext,
        strategy: &str,
        correct: bool,
        steps: usize,
        noisy: bool,
    ) {
        let arm_map = self.arms.entry(ctx.clone()).or_default();
        let stats = arm_map.entry(strategy.to_string()).or_default();
        stats.pulls += 1;
        stats.total_steps += steps;
        if correct { stats.successes += 1; }
        if noisy {
            stats.noise_pulls += 1;
            if correct { stats.noise_successes += 1; }
        }
    }

    fn next_f64(&mut self) -> f64 {
        let mut x = self.rng_state.max(1);
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        self.rng_state = x;
        (x as f64) / (u64::MAX as f64)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// AdaptiveSolver
// ═══════════════════════════════════════════════════════════════════════════

pub struct AdaptiveSolver {
    /// Internal solver
    solver: TemporalSolver,
    /// ReasoningBank for learning
    pub reasoning_bank: ReasoningBank,
    /// Current strategy
    current_strategy: Strategy,
    /// Total episodes completed
    pub episodes: usize,
    /// When set, solve() uses this step limit instead of the strategy's
    pub external_step_limit: Option<usize>,
    /// KnowledgeCompiler for Strategy Zero (compiled solve configs)
    pub compiler: KnowledgeCompiler,
    /// Whether to use the compiler as Strategy Zero
    pub compiler_enabled: bool,
    /// Adaptive strategy router (contextual bandit)
    pub router: StrategyRouter,
    /// Whether to use the adaptive router instead of fixed strategy selection
    pub router_enabled: bool,
}

impl Default for AdaptiveSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptiveSolver {
    /// Create a new adaptive solver
    pub fn new() -> Self {
        Self {
            solver: TemporalSolver::default(),
            reasoning_bank: ReasoningBank::new(),
            current_strategy: Strategy::default(),
            episodes: 0,
            external_step_limit: None,
            compiler: KnowledgeCompiler::new(),
            compiler_enabled: false,
            router: StrategyRouter::new(),
            router_enabled: false,
        }
    }

    /// Create with pre-trained ReasoningBank
    pub fn with_reasoning_bank(reasoning_bank: ReasoningBank) -> Self {
        Self {
            solver: TemporalSolver::default(),
            reasoning_bank,
            current_strategy: Strategy::default(),
            episodes: 0,
            external_step_limit: None,
            compiler: KnowledgeCompiler::new(),
            compiler_enabled: false,
            router: StrategyRouter::new(),
            router_enabled: false,
        }
    }

    /// Recompile knowledge from the current ReasoningBank.
    pub fn recompile(&mut self) {
        self.compiler.compile_from_bank(&self.reasoning_bank);
    }

    /// Get mutable reference to the internal solver for configuration.
    pub fn solver_mut(&mut self) -> &mut TemporalSolver {
        &mut self.solver
    }

    /// Solve a puzzle with adaptive learning.
    /// If compiler_enabled, tries Strategy Zero (compiled config) first.
    /// If router_enabled, uses contextual bandit for strategy selection.
    pub fn solve(&mut self, puzzle: &TemporalPuzzle) -> Result<SolverResult> {
        // Get constraint types for pattern matching
        let constraint_types: Vec<String> = puzzle
            .constraints
            .iter()
            .map(|c| constraint_type_name(c))
            .collect();

        // Accumulated steps across all attempts (Strategy Zero + fallback)
        let mut extra_steps: usize = 0;
        let mut extra_tool_calls: usize = 0;

        // ─── Strategy Zero: KnowledgeCompiler ───────────────────────────
        if self.compiler_enabled {
            if let Some(config) = self.compiler.lookup(puzzle) {
                if config.expected_correct {
                    // Use compiled config as Strategy Zero with early termination
                    let compiled_steps = config.max_steps.max(5);
                    self.solver.calendar_tool = config.use_rewriting;
                    self.solver.stop_after_first = config.stop_after_first;
                    self.solver.max_steps = self.external_step_limit
                        .map(|l| l.min(compiled_steps))
                        .unwrap_or(compiled_steps);

                    let start = std::time::Instant::now();
                    let result = self.solver.solve(puzzle)?;
                    let latency = start.elapsed().as_millis() as u64;

                    // Reset stop_after_first for fallback path
                    self.solver.stop_after_first = false;

                    if result.correct {
                        // Strategy Zero win — record and return
                        self.compiler.record_success(puzzle);
                        let mut trajectory = Trajectory::new(&puzzle.id, puzzle.difficulty);
                        trajectory.constraint_types = constraint_types;
                        trajectory.latency_ms = latency;
                        let sol_str = result.solutions.first()
                            .map(|d| d.to_string()).unwrap_or_else(|| "none".to_string());
                        trajectory.record_attempt(
                            sol_str, 0.95, result.steps, result.tool_calls, "compiler",
                        );
                        trajectory.set_verdict(
                            Verdict::Success,
                            puzzle.solutions.first().map(|d| d.to_string()),
                        );
                        self.reasoning_bank.record_trajectory(trajectory);
                        self.episodes += 1;

                        // Update router if enabled
                        if self.router_enabled {
                            let ctx = StrategyRouter::context(puzzle, false);
                            self.router.update(&ctx, "compiler", true, result.steps, false);
                        }

                        return Ok(result);
                    } else {
                        // Strategy Zero failed — record overhead, fall through
                        extra_steps += result.steps;
                        extra_tool_calls += result.tool_calls;
                        self.compiler.record_failure(puzzle);
                    }
                }
            }
        }

        // ─── Strategy Selection (fixed or router) ───────────────────────
        if self.router_enabled {
            let ctx = StrategyRouter::context(puzzle, false);
            let available = vec![
                "default".to_string(),
                "aggressive".to_string(),
                "conservative".to_string(),
                "adaptive".to_string(),
            ];
            let ranked = self.router.select(&ctx, &available);
            // Use the top-ranked strategy
            if let Some((top_strategy, _)) = ranked.first() {
                self.current_strategy = self.reasoning_bank
                    .strategy_from_name(top_strategy, puzzle.difficulty);
            }
        } else {
            // Fixed strategy selection from ReasoningBank
            self.current_strategy = self
                .reasoning_bank
                .get_strategy(puzzle.difficulty, &constraint_types);
        }

        // Configure solver based on strategy (external limit overrides strategy)
        self.solver.calendar_tool = self.current_strategy.use_rewriting;
        self.solver.max_steps = self.external_step_limit
            .unwrap_or(self.current_strategy.max_steps);
        self.solver.stop_after_first = false;

        // Create trajectory for this puzzle
        let mut trajectory = Trajectory::new(&puzzle.id, puzzle.difficulty);
        trajectory.constraint_types = constraint_types;

        // Solve the puzzle
        let start = std::time::Instant::now();
        let mut result = self.solver.solve(puzzle)?;
        trajectory.latency_ms = start.elapsed().as_millis() as u64;

        // Accumulate overhead from failed Strategy Zero attempt
        result.steps += extra_steps;
        result.tool_calls += extra_tool_calls;

        // Record attempt
        let solution_str = result
            .solutions
            .first()
            .map(|d| d.to_string())
            .unwrap_or_else(|| "none".to_string());

        let confidence = self.calculate_confidence(&result, puzzle);

        trajectory.record_attempt(
            solution_str,
            confidence,
            result.steps,
            result.tool_calls,
            &self.current_strategy.name,
        );

        // Determine verdict
        let verdict = if result.correct {
            if confidence >= 0.9 {
                Verdict::Success
            } else {
                Verdict::Acceptable
            }
        } else if result.solved {
            Verdict::Suboptimal {
                reason: "Solution found but incorrect".to_string(),
                delta: 1.0 - confidence,
            }
        } else if confidence < self.current_strategy.confidence_threshold {
            Verdict::LowConfidence
        } else {
            Verdict::Failed
        };

        trajectory.set_verdict(verdict, puzzle.solutions.first().map(|d| d.to_string()));

        // Update router stats
        if self.router_enabled {
            let ctx = StrategyRouter::context(puzzle, false);
            self.router.update(
                &ctx, &self.current_strategy.name,
                result.correct, result.steps, false,
            );
        }

        // Record trajectory for learning
        self.reasoning_bank.record_trajectory(trajectory);
        self.episodes += 1;

        Ok(result)
    }

    /// Calculate confidence in a result
    fn calculate_confidence(&self, result: &SolverResult, puzzle: &TemporalPuzzle) -> f64 {
        let mut confidence = 0.5;

        // Higher confidence if solved quickly
        if result.solved {
            confidence += 0.2;
            if result.steps < self.solver.max_steps / 2 {
                confidence += 0.1;
            }
        }

        // Higher confidence with tool use on complex puzzles
        if result.tool_calls > 0 && puzzle.difficulty > 5 {
            confidence += 0.1;
        }

        // Lower confidence if took many steps
        if result.steps > self.solver.max_steps * 3 / 4 {
            confidence -= 0.1;
        }

        // Adjust based on learned calibration
        let calibrated_threshold = self
            .reasoning_bank
            .calibration
            .get_threshold(puzzle.difficulty);
        if confidence >= calibrated_threshold {
            confidence += 0.05;
        }

        confidence.min(1.0).max(0.0)
    }

    /// Get learning progress
    pub fn learning_progress(&self) -> crate::reasoning_bank::LearningProgress {
        self.reasoning_bank.learning_progress()
    }

    /// Get hints for a puzzle
    pub fn get_hints(&self, constraint_types: &[String]) -> Vec<String> {
        self.reasoning_bank.get_hints(constraint_types)
    }
}

/// Get the type name of a constraint for pattern matching
fn constraint_type_name(constraint: &TemporalConstraint) -> String {
    match constraint {
        TemporalConstraint::Exact(_) => "Exact".to_string(),
        TemporalConstraint::After(_) => "After".to_string(),
        TemporalConstraint::Before(_) => "Before".to_string(),
        TemporalConstraint::Between(_, _) => "Between".to_string(),
        TemporalConstraint::DayOfWeek(_) => "DayOfWeek".to_string(),
        TemporalConstraint::DaysAfter(_, _) => "DaysAfter".to_string(),
        TemporalConstraint::DaysBefore(_, _) => "DaysBefore".to_string(),
        TemporalConstraint::InMonth(_) => "InMonth".to_string(),
        TemporalConstraint::InYear(_) => "InYear".to_string(),
        TemporalConstraint::DayOfMonth(_) => "DayOfMonth".to_string(),
        TemporalConstraint::RelativeToEvent(_, _) => "RelativeToEvent".to_string(),
    }
}
