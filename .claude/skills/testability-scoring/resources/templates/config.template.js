// Testability Scorer Configuration Template

module.exports = {
  // Application under test
  baseURL: 'https://www.saucedemo.com',

  // Scoring weights (must sum to 100)
  weights: {
    observability: 15,        // Console logs, state visibility, monitoring
    controllability: 15,       // API access, state manipulation, test data injection
    algorithmicSimplicity: 10, // Clear input-output relationships, low complexity
    algorithmicTransparency: 10, // Understandable logic, clear data flow
    explainability: 10,        // Documentation, comments, error messages
    similarity: 5,             // Standard patterns, familiar architecture
    algorithmicStability: 10,  // API versioning, backward compatibility
    unbugginess: 10,           // Low defect rate, test reliability
    smallness: 10,             // Manageable size, modular design
    decomposability: 5         // Component isolation, test unit granularity
  },

  // Grading scale
  grades: {
    A: 90,  // Excellent
    B: 80,  // Good
    C: 70,  // Acceptable
    D: 60,  // Below average
    F: 0    // Poor
  },

  // Report settings
  reports: {
    format: ['html', 'json', 'text'],  // Output formats
    directory: 'tests/reports',         // Report location
    autoOpen: true,                     // Open HTML report automatically
    includeAI: true,                    // AI-powered recommendations
    includeCharts: true,                // Visual charts
    includeHistory: true                // Historical comparison
  },

  // User types for comparative analysis (optional)
  userTypes: [
    { username: 'standard_user', password: 'secret_sauce', role: 'standard' },
    { username: 'locked_out_user', password: 'secret_sauce', role: 'locked' },
    { username: 'problem_user', password: 'secret_sauce', role: 'problem' },
    { username: 'performance_glitch_user', password: 'secret_sauce', role: 'performance' },
    { username: 'error_user', password: 'secret_sauce', role: 'error' },
    { username: 'visual_user', password: 'secret_sauce', role: 'visual' }
  ],

  // Browser configuration
  browsers: ['chromium', 'firefox', 'webkit'],

  // Timeouts
  timeouts: {
    test: 30000,      // Per test timeout
    navigation: 10000, // Page navigation
    action: 5000      // UI action timeout
  },

  // Thresholds for pass/fail
  thresholds: {
    overall: 70,           // Minimum overall score
    critical: {            // Critical principles must meet these minimums
      observability: 60,
      controllability: 60,
      unbugginess: 70
    }
  },

  // AI recommendations settings
  ai: {
    enabled: true,
    provider: 'local',  // 'local' or 'api'
    prioritize: 'impact', // 'impact', 'effort', or 'score'
    maxRecommendations: 10
  },

  // Historical tracking
  history: {
    enabled: true,
    directory: '.testability-history',
    retention: 90  // days
  }
};
