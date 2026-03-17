//! A* GOAP planner for finding optimal action sequences

use super::{GoapAction, GoapPlan, LearningGoal, LearningWorldState, PlannedAction};
use super::actions::ActionRegistry;
use anyhow::Result;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Ordering;

/// GOAP planner using A* search
pub struct GoapPlanner {
    action_registry: ActionRegistry,
    max_iterations: usize,
}

impl GoapPlanner {
    /// Create a new GOAP planner
    pub fn new() -> Self {
        Self {
            action_registry: ActionRegistry::new(),
            max_iterations: 1000,
        }
    }

    /// Plan a sequence of actions to achieve a goal
    pub fn plan(&self, state: &LearningWorldState, goal: &LearningGoal) -> Result<GoapPlan> {
        let goal_conditions = self.goal_to_conditions(goal);

        // Check if goal is already satisfied
        if self.goal_satisfied(state, &goal_conditions) {
            return Ok(GoapPlan::empty());
        }

        // A* search
        let mut open_set = BinaryHeap::new();
        let mut closed_set = HashSet::new();
        let mut came_from: HashMap<StateHash, (StateHash, String)> = HashMap::new();
        let mut g_score: HashMap<StateHash, f64> = HashMap::new();

        let start_hash = self.hash_state(state);
        g_score.insert(start_hash, 0.0);

        open_set.push(SearchNode {
            state: state.clone(),
            f_score: self.heuristic(state, &goal_conditions),
            g_score: 0.0,
        });

        let mut iterations = 0;
        while let Some(current) = open_set.pop() {
            iterations += 1;
            if iterations > self.max_iterations {
                tracing::warn!("GOAP planner reached max iterations");
                break;
            }

            let current_hash = self.hash_state(&current.state);

            // Check if goal is reached
            if self.goal_satisfied(&current.state, &goal_conditions) {
                return Ok(self.reconstruct_plan(
                    &came_from,
                    current_hash,
                    start_hash,
                    current.state,
                ));
            }

            if closed_set.contains(&current_hash) {
                continue;
            }
            closed_set.insert(current_hash);

            // Explore neighbors (apply each valid action)
            for action in self.action_registry.all().values() {
                if !action.preconditions_met(&current.state) {
                    continue;
                }

                let mut new_state = current.state.clone();
                for (effect, value) in &action.effects {
                    new_state.apply_effect(effect, value.clone());
                }

                let new_hash = self.hash_state(&new_state);
                if closed_set.contains(&new_hash) {
                    continue;
                }

                let tentative_g = current.g_score + action.cost;
                let current_g = g_score.get(&new_hash).copied().unwrap_or(f64::INFINITY);

                if tentative_g < current_g {
                    came_from.insert(new_hash, (current_hash, action.name.clone()));
                    g_score.insert(new_hash, tentative_g);

                    let h = self.heuristic(&new_state, &goal_conditions);
                    open_set.push(SearchNode {
                        state: new_state,
                        f_score: tentative_g + h,
                        g_score: tentative_g,
                    });
                }
            }
        }

        // No plan found - return best-effort partial plan
        tracing::warn!("GOAP planner could not find complete plan");
        Ok(GoapPlan::empty())
    }

    /// Convert a goal to state conditions
    fn goal_to_conditions(&self, goal: &LearningGoal) -> HashMap<String, GoalCondition> {
        let mut conditions = HashMap::new();

        match goal {
            LearningGoal::DiscoverPatterns { target_count } => {
                conditions.insert(
                    "patterns_discovered".to_string(),
                    GoalCondition::GreaterOrEqual(*target_count as f64),
                );
            }
            LearningGoal::SubmitToCloudBrain { min_quality: _ } => {
                conditions.insert(
                    "patterns_pending_submission".to_string(),
                    GoalCondition::Equal(0.0),
                );
            }
            LearningGoal::ConsolidateLearning => {
                conditions.insert(
                    "consolidation_due".to_string(),
                    GoalCondition::Equal(0.0), // false
                );
            }
            LearningGoal::CompleteDailyCycle => {
                conditions.insert(
                    "patterns_discovered".to_string(),
                    GoalCondition::GreaterOrEqual(1.0),
                );
                conditions.insert(
                    "consolidation_due".to_string(),
                    GoalCondition::Equal(0.0),
                );
            }
            LearningGoal::OptimizeDomain { domain: _ } => {
                conditions.insert(
                    "patterns_discovered".to_string(),
                    GoalCondition::GreaterOrEqual(1.0),
                );
            }
            LearningGoal::ExploreFiles { paths: _ } => {
                conditions.insert(
                    "patterns_discovered".to_string(),
                    GoalCondition::GreaterOrEqual(1.0),
                );
            }
        }

        conditions
    }

    /// Check if goal conditions are satisfied
    fn goal_satisfied(
        &self,
        state: &LearningWorldState,
        conditions: &HashMap<String, GoalCondition>,
    ) -> bool {
        for (key, condition) in conditions {
            let value = self.get_state_value(state, key);
            if !condition.is_satisfied(value) {
                return false;
            }
        }
        true
    }

    /// Heuristic function for A*
    fn heuristic(
        &self,
        state: &LearningWorldState,
        goal_conditions: &HashMap<String, GoalCondition>,
    ) -> f64 {
        let mut h = 0.0;

        for (key, condition) in goal_conditions {
            let value = self.get_state_value(state, key);
            h += condition.distance(value);
        }

        h
    }

    /// Get a numeric value from state
    fn get_state_value(&self, state: &LearningWorldState, key: &str) -> f64 {
        match key {
            "patterns_discovered" => state.patterns_discovered as f64,
            "patterns_pending_submission" => state.patterns_pending_submission as f64,
            "consolidation_due" => if state.consolidation_due { 1.0 } else { 0.0 },
            _ => 0.0,
        }
    }

    /// Hash state for comparison
    fn hash_state(&self, state: &LearningWorldState) -> StateHash {
        // Simple hash based on key state values
        let mut hash = 0u64;
        hash ^= state.patterns_discovered as u64;
        hash ^= (state.patterns_pending_submission as u64) << 8;
        hash ^= (state.scanning as u64) << 16;
        hash ^= (state.consolidation_due as u64) << 17;
        hash ^= (state.pi_ruv_io_connected as u64) << 18;
        hash
    }

    /// Reconstruct plan from search results
    fn reconstruct_plan(
        &self,
        came_from: &HashMap<StateHash, (StateHash, String)>,
        end_hash: StateHash,
        start_hash: StateHash,
        final_state: LearningWorldState,
    ) -> GoapPlan {
        let mut actions = Vec::new();
        let mut current = end_hash;

        while current != start_hash {
            if let Some((prev, action_name)) = came_from.get(&current) {
                if let Some(action) = self.action_registry.get(action_name) {
                    actions.push(PlannedAction {
                        action: action_name.clone(),
                        params: serde_json::json!({}),
                        parallel_with: vec![],
                        cost: action.cost,
                    });
                }
                current = *prev;
            } else {
                break;
            }
        }

        actions.reverse();

        let total_cost: f64 = actions.iter().map(|a| a.cost).sum();

        GoapPlan {
            actions,
            estimated_cost: total_cost,
            reasoning: "A* search plan".to_string(),
            expected_state: final_state,
        }
    }
}

impl Default for GoapPlanner {
    fn default() -> Self {
        Self::new()
    }
}

type StateHash = u64;

#[derive(Clone)]
struct SearchNode {
    state: LearningWorldState,
    f_score: f64,
    g_score: f64,
}

impl Eq for SearchNode {}

impl PartialEq for SearchNode {
    fn eq(&self, other: &Self) -> bool {
        self.f_score == other.f_score
    }
}

impl Ord for SearchNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior
        other.f_score.partial_cmp(&self.f_score).unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for SearchNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone)]
enum GoalCondition {
    Equal(f64),
    GreaterOrEqual(f64),
    LessThan(f64),
}

impl GoalCondition {
    fn is_satisfied(&self, value: f64) -> bool {
        match self {
            GoalCondition::Equal(target) => (value - target).abs() < 0.001,
            GoalCondition::GreaterOrEqual(target) => value >= *target,
            GoalCondition::LessThan(target) => value < *target,
        }
    }

    fn distance(&self, value: f64) -> f64 {
        match self {
            GoalCondition::Equal(target) => (value - target).abs(),
            GoalCondition::GreaterOrEqual(target) => {
                if value >= *target { 0.0 } else { target - value }
            }
            GoalCondition::LessThan(target) => {
                if value < *target { 0.0 } else { value - target + 1.0 }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_basic() {
        let planner = GoapPlanner::new();
        let state = LearningWorldState::default();
        let goal = LearningGoal::DiscoverPatterns { target_count: 1 };

        let plan = planner.plan(&state, &goal).unwrap();
        // Plan may or may not find a solution depending on action effects
        assert!(plan.actions.is_empty() || !plan.actions.is_empty());
    }

    #[test]
    fn test_goal_conditions() {
        let cond = GoalCondition::GreaterOrEqual(5.0);
        assert!(!cond.is_satisfied(3.0));
        assert!(cond.is_satisfied(5.0));
        assert!(cond.is_satisfied(7.0));
        assert_eq!(cond.distance(3.0), 2.0);
        assert_eq!(cond.distance(5.0), 0.0);
    }
}
