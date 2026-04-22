//! Resource budget enforcement (ADR-103 B4).
//!
//! Provides [`BudgetEnforcer`] which tracks resource consumption against limits
//! defined in [`ResourceBudget`](crate::config::ResourceBudget).

use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::config::ResourceBudget;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Error indicating a resource budget has been exceeded.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BudgetError {
    /// Wall-clock time limit exceeded.
    TimeLimitExceeded { limit_secs: u32, elapsed_secs: u32 },
    /// Total tokens (input + output) exceeded.
    TokenLimitExceeded { limit: u64, consumed: u64 },
    /// Cost budget exceeded.
    CostLimitExceeded {
        limit_microdollars: u64,
        consumed_microdollars: u64,
    },
    /// Too many tool calls.
    ToolCallLimitExceeded { limit: u32, count: u32 },
    /// Too many external writes.
    ExternalWriteLimitExceeded { limit: u32, count: u32 },
}

impl std::fmt::Display for BudgetError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BudgetError::TimeLimitExceeded {
                limit_secs,
                elapsed_secs,
            } => {
                write!(f, "time limit exceeded: {elapsed_secs}s > {limit_secs}s")
            }
            BudgetError::TokenLimitExceeded { limit, consumed } => {
                write!(f, "token limit exceeded: {consumed} > {limit}")
            }
            BudgetError::CostLimitExceeded {
                limit_microdollars,
                consumed_microdollars,
            } => {
                write!(
                    f,
                    "cost limit exceeded: ${:.4} > ${:.4}",
                    *consumed_microdollars as f64 / 1_000_000.0,
                    *limit_microdollars as f64 / 1_000_000.0
                )
            }
            BudgetError::ToolCallLimitExceeded { limit, count } => {
                write!(f, "tool call limit exceeded: {count} > {limit}")
            }
            BudgetError::ExternalWriteLimitExceeded { limit, count } => {
                write!(f, "external write limit exceeded: {count} > {limit}")
            }
        }
    }
}

impl std::error::Error for BudgetError {}

// ---------------------------------------------------------------------------
// BudgetEnforcer
// ---------------------------------------------------------------------------

/// Tracks resource consumption and enforces limits from [`ResourceBudget`].
///
/// Thread-safe via atomics for concurrent tool execution (ADR-103 A2).
///
/// # Example
///
/// ```rust
/// use rvagent_core::config::ResourceBudget;
/// use rvagent_core::budget::BudgetEnforcer;
///
/// let budget = ResourceBudget {
///     max_tool_calls: 10,
///     ..Default::default()
/// };
/// let enforcer = BudgetEnforcer::new(budget);
///
/// // Before each tool call
/// enforcer.check_tool_call().expect("within budget");
/// enforcer.record_tool_call();
/// ```
#[derive(Debug)]
pub struct BudgetEnforcer {
    budget: ResourceBudget,
    start_time: Instant,
    consumed_tokens: AtomicU64,
    consumed_cost_microdollars: AtomicU64,
    tool_call_count: AtomicU32,
    external_write_count: AtomicU32,
}

impl BudgetEnforcer {
    /// Create a new enforcer with the given budget limits.
    pub fn new(budget: ResourceBudget) -> Self {
        Self {
            budget,
            start_time: Instant::now(),
            consumed_tokens: AtomicU64::new(0),
            consumed_cost_microdollars: AtomicU64::new(0),
            tool_call_count: AtomicU32::new(0),
            external_write_count: AtomicU32::new(0),
        }
    }

    /// Create an enforcer with no limits (all limits set to max).
    pub fn unlimited() -> Self {
        Self::new(ResourceBudget {
            max_time_secs: u32::MAX,
            max_tokens: u64::MAX,
            max_cost_microdollars: u64::MAX,
            max_tool_calls: u32::MAX,
            max_external_writes: u32::MAX,
        })
    }

    // -----------------------------------------------------------------------
    // Checks — return error if budget would be exceeded
    // -----------------------------------------------------------------------

    /// Check if the time limit has been exceeded.
    pub fn check_time(&self) -> Result<(), BudgetError> {
        if self.budget.max_time_secs == 0 {
            return Ok(()); // No limit
        }
        let elapsed = self.start_time.elapsed().as_secs() as u32;
        if elapsed > self.budget.max_time_secs {
            return Err(BudgetError::TimeLimitExceeded {
                limit_secs: self.budget.max_time_secs,
                elapsed_secs: elapsed,
            });
        }
        Ok(())
    }

    /// Check if another tool call would exceed the limit.
    pub fn check_tool_call(&self) -> Result<(), BudgetError> {
        if self.budget.max_tool_calls == 0 {
            return Ok(()); // No limit
        }
        let current = self.tool_call_count.load(Ordering::Relaxed);
        if current >= self.budget.max_tool_calls {
            return Err(BudgetError::ToolCallLimitExceeded {
                limit: self.budget.max_tool_calls,
                count: current,
            });
        }
        Ok(())
    }

    /// Check if adding `tokens` would exceed the limit.
    pub fn check_tokens(&self, tokens: u64) -> Result<(), BudgetError> {
        if self.budget.max_tokens == 0 {
            return Ok(()); // No limit
        }
        let current = self.consumed_tokens.load(Ordering::Relaxed);
        let after = current.saturating_add(tokens);
        if after > self.budget.max_tokens {
            return Err(BudgetError::TokenLimitExceeded {
                limit: self.budget.max_tokens,
                consumed: after,
            });
        }
        Ok(())
    }

    /// Check if adding `microdollars` would exceed the cost limit.
    pub fn check_cost(&self, microdollars: u64) -> Result<(), BudgetError> {
        if self.budget.max_cost_microdollars == 0 {
            return Ok(()); // No limit
        }
        let current = self.consumed_cost_microdollars.load(Ordering::Relaxed);
        let after = current.saturating_add(microdollars);
        if after > self.budget.max_cost_microdollars {
            return Err(BudgetError::CostLimitExceeded {
                limit_microdollars: self.budget.max_cost_microdollars,
                consumed_microdollars: after,
            });
        }
        Ok(())
    }

    /// Check if another external write would exceed the limit.
    pub fn check_external_write(&self) -> Result<(), BudgetError> {
        if self.budget.max_external_writes == 0 {
            return Ok(()); // No limit
        }
        let current = self.external_write_count.load(Ordering::Relaxed);
        if current >= self.budget.max_external_writes {
            return Err(BudgetError::ExternalWriteLimitExceeded {
                limit: self.budget.max_external_writes,
                count: current,
            });
        }
        Ok(())
    }

    /// Check all limits. Returns the first error encountered, or Ok if all pass.
    pub fn check_all(&self) -> Result<(), BudgetError> {
        self.check_time()?;
        // Token and cost checks require the delta, so we don't check them here.
        // Tool call and external write checks are pre-increment checks.
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Recording — increment counters after successful operations
    // -----------------------------------------------------------------------

    /// Record a tool call.
    pub fn record_tool_call(&self) {
        self.tool_call_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Record token usage (input + output).
    pub fn record_tokens(&self, tokens: u64) {
        self.consumed_tokens.fetch_add(tokens, Ordering::Relaxed);
    }

    /// Record cost in microdollars.
    pub fn record_cost(&self, microdollars: u64) {
        self.consumed_cost_microdollars
            .fetch_add(microdollars, Ordering::Relaxed);
    }

    /// Record an external write.
    pub fn record_external_write(&self) {
        self.external_write_count.fetch_add(1, Ordering::Relaxed);
    }

    // -----------------------------------------------------------------------
    // Getters
    // -----------------------------------------------------------------------

    /// Get the configured budget.
    pub fn budget(&self) -> &ResourceBudget {
        &self.budget
    }

    /// Elapsed time since the enforcer was created.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Current token consumption.
    pub fn tokens_consumed(&self) -> u64 {
        self.consumed_tokens.load(Ordering::Relaxed)
    }

    /// Current cost consumption in microdollars.
    pub fn cost_consumed(&self) -> u64 {
        self.consumed_cost_microdollars.load(Ordering::Relaxed)
    }

    /// Current tool call count.
    pub fn tool_calls(&self) -> u32 {
        self.tool_call_count.load(Ordering::Relaxed)
    }

    /// Current external write count.
    pub fn external_writes(&self) -> u32 {
        self.external_write_count.load(Ordering::Relaxed)
    }

    /// Percentage of budget consumed (0.0 - 1.0+) for each dimension.
    pub fn utilization(&self) -> BudgetUtilization {
        let elapsed_secs = self.start_time.elapsed().as_secs() as f64;
        BudgetUtilization {
            time: if self.budget.max_time_secs > 0 {
                elapsed_secs / self.budget.max_time_secs as f64
            } else {
                0.0
            },
            tokens: if self.budget.max_tokens > 0 {
                self.tokens_consumed() as f64 / self.budget.max_tokens as f64
            } else {
                0.0
            },
            cost: if self.budget.max_cost_microdollars > 0 {
                self.cost_consumed() as f64 / self.budget.max_cost_microdollars as f64
            } else {
                0.0
            },
            tool_calls: if self.budget.max_tool_calls > 0 {
                self.tool_calls() as f64 / self.budget.max_tool_calls as f64
            } else {
                0.0
            },
            external_writes: if self.budget.max_external_writes > 0 {
                self.external_writes() as f64 / self.budget.max_external_writes as f64
            } else {
                0.0
            },
        }
    }
}

/// Budget utilization as percentages (0.0 - 1.0+).
#[derive(Debug, Clone, Copy)]
pub struct BudgetUtilization {
    pub time: f64,
    pub tokens: f64,
    pub cost: f64,
    pub tool_calls: f64,
    pub external_writes: f64,
}

impl BudgetUtilization {
    /// The maximum utilization across all dimensions.
    pub fn max(&self) -> f64 {
        self.time
            .max(self.tokens)
            .max(self.cost)
            .max(self.tool_calls)
            .max(self.external_writes)
    }

    /// True if any dimension is at or over 100%.
    pub fn is_exceeded(&self) -> bool {
        self.max() >= 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unlimited() {
        let enforcer = BudgetEnforcer::unlimited();
        assert!(enforcer.check_time().is_ok());
        assert!(enforcer.check_tool_call().is_ok());
        assert!(enforcer.check_tokens(1_000_000_000).is_ok());
        assert!(enforcer.check_cost(1_000_000_000).is_ok());
        assert!(enforcer.check_external_write().is_ok());
    }

    #[test]
    fn test_tool_call_limit() {
        let budget = ResourceBudget {
            max_tool_calls: 3,
            ..Default::default()
        };
        let enforcer = BudgetEnforcer::new(budget);

        // First 3 should succeed
        for _ in 0..3 {
            assert!(enforcer.check_tool_call().is_ok());
            enforcer.record_tool_call();
        }

        // 4th should fail
        let result = enforcer.check_tool_call();
        assert!(matches!(
            result,
            Err(BudgetError::ToolCallLimitExceeded { limit: 3, count: 3 })
        ));
    }

    #[test]
    fn test_token_limit() {
        let budget = ResourceBudget {
            max_tokens: 1000,
            ..Default::default()
        };
        let enforcer = BudgetEnforcer::new(budget);

        // 500 tokens should pass
        assert!(enforcer.check_tokens(500).is_ok());
        enforcer.record_tokens(500);

        // Another 400 should pass (total 900)
        assert!(enforcer.check_tokens(400).is_ok());
        enforcer.record_tokens(400);

        // Another 200 would exceed (900 + 200 = 1100 > 1000)
        let result = enforcer.check_tokens(200);
        assert!(matches!(
            result,
            Err(BudgetError::TokenLimitExceeded { limit: 1000, .. })
        ));
    }

    #[test]
    fn test_cost_limit() {
        let budget = ResourceBudget {
            max_cost_microdollars: 1_000_000, // $1
            ..Default::default()
        };
        let enforcer = BudgetEnforcer::new(budget);

        // $0.50 should pass
        assert!(enforcer.check_cost(500_000).is_ok());
        enforcer.record_cost(500_000);

        // Another $0.60 would exceed
        let result = enforcer.check_cost(600_000);
        assert!(matches!(result, Err(BudgetError::CostLimitExceeded { .. })));
    }

    #[test]
    fn test_utilization() {
        let budget = ResourceBudget {
            max_tool_calls: 100,
            max_tokens: 10_000,
            ..Default::default()
        };
        let enforcer = BudgetEnforcer::new(budget);

        enforcer.record_tool_call();
        enforcer.record_tool_call();
        enforcer.record_tokens(1000);

        let util = enforcer.utilization();
        assert!((util.tool_calls - 0.02).abs() < 0.001);
        assert!((util.tokens - 0.1).abs() < 0.001);
        assert!(!util.is_exceeded());
    }

    #[test]
    fn test_error_display() {
        let err = BudgetError::CostLimitExceeded {
            limit_microdollars: 1_000_000,
            consumed_microdollars: 1_500_000,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("$1.5"));
        assert!(msg.contains("$1.0"));
    }
}
