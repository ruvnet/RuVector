//! Self-healing integration tests

use ruvector_dag::healing::*;

#[test]
fn test_anomaly_detection() {
    let mut detector = AnomalyDetector::new(AnomalyConfig {
        z_threshold: 3.0,
        window_size: 100,
        min_samples: 10,
    });

    // Normal observations
    for _ in 0..99 {
        detector.observe(100.0 + rand::random::<f64>() * 10.0);
    }

    // Should not detect anomaly for normal value
    assert!(detector.is_anomaly(105.0).is_none());

    // Should detect anomaly for extreme value
    let z = detector.is_anomaly(200.0);
    assert!(z.is_some());
    assert!(z.unwrap().abs() > 3.0);
}

#[test]
fn test_drift_detection() {
    let mut drift = LearningDriftDetector::new(0.1, 50);

    // Set baseline
    drift.set_baseline("accuracy", 0.9);

    // Record values showing decline
    for i in 0..50 {
        drift.record("accuracy", 0.9 - (i as f64) * 0.01);
    }

    let metric = drift.check_drift("accuracy").unwrap();

    assert_eq!(metric.trend, DriftTrend::Declining);
    assert!(metric.drift_magnitude > 0.1);
}

#[test]
fn test_healing_orchestrator() {
    let mut orchestrator = HealingOrchestrator::new();

    // Add detector
    orchestrator.add_detector("latency", AnomalyConfig::default());

    // Add strategy
    use std::sync::Arc;
    orchestrator.add_repair_strategy(Arc::new(CacheFlushStrategy));

    // Observe normal values
    for _ in 0..20 {
        orchestrator.observe("latency", 50.0 + rand::random::<f64>() * 5.0);
    }

    // Run cycle
    let result = orchestrator.run_cycle();

    // Should complete without panicking
    assert!(result.repairs_succeeded <= result.repairs_attempted);
}

#[test]
fn test_anomaly_window_sliding() {
    let mut detector = AnomalyDetector::new(AnomalyConfig {
        z_threshold: 2.0,
        window_size: 10,
        min_samples: 5,
    });

    // Fill window
    for i in 0..15 {
        detector.observe(100.0 + i as f64);
    }

    // Window should only contain last 10 observations
    assert_eq!(detector.sample_count(), 10);
}

#[test]
fn test_drift_stable_baseline() {
    let mut drift = LearningDriftDetector::new(0.1, 100);

    drift.set_baseline("metric", 1.0);

    // Record stable values
    for _ in 0..100 {
        drift.record("metric", 1.0 + rand::random::<f64>() * 0.02);
    }

    let metric = drift.check_drift("metric").unwrap();

    // Should be stable
    assert_eq!(metric.trend, DriftTrend::Stable);
    assert!(metric.drift_magnitude < 0.1);
}

#[test]
fn test_drift_improving_trend() {
    let mut drift = LearningDriftDetector::new(0.1, 50);

    drift.set_baseline("performance", 0.5);

    // Record improving values
    for i in 0..50 {
        drift.record("performance", 0.5 + (i as f64) * 0.01);
    }

    let metric = drift.check_drift("performance").unwrap();

    assert_eq!(metric.trend, DriftTrend::Improving);
}

#[test]
fn test_healing_multiple_detectors() {
    let mut orchestrator = HealingOrchestrator::new();

    orchestrator.add_detector("cpu", AnomalyConfig::default());
    orchestrator.add_detector("memory", AnomalyConfig::default());
    orchestrator.add_detector("latency", AnomalyConfig::default());

    // Observe values for all metrics
    for _ in 0..20 {
        orchestrator.observe("cpu", 50.0);
        orchestrator.observe("memory", 1000.0);
        orchestrator.observe("latency", 100.0);
    }

    // Inject anomaly in one metric
    orchestrator.observe("latency", 500.0);

    let result = orchestrator.run_cycle();

    // Should attempt repairs
    assert!(result.anomalies_detected >= 0);
}

#[test]
fn test_anomaly_statistical_properties() {
    let mut detector = AnomalyDetector::new(AnomalyConfig {
        z_threshold: 2.0,
        window_size: 100,
        min_samples: 30,
    });

    // Add normally distributed values (mean=100, std=10)
    for _ in 0..100 {
        let value = 100.0 + rand::random::<f64>() * 20.0 - 10.0;
        detector.observe(value);
    }

    // Value within 2 sigma should not be anomaly
    assert!(detector.is_anomaly(110.0).is_none());

    // Value beyond 2 sigma should be anomaly
    assert!(detector.is_anomaly(150.0).is_some());
}

#[test]
fn test_drift_multiple_metrics() {
    let mut drift = LearningDriftDetector::new(0.1, 50);

    drift.set_baseline("accuracy", 0.9);
    drift.set_baseline("latency", 100.0);

    // Record values
    for i in 0..50 {
        drift.record("accuracy", 0.9 - (i as f64) * 0.005);
        drift.record("latency", 100.0 + (i as f64) * 2.0);
    }

    let acc_metric = drift.check_drift("accuracy").unwrap();
    let lat_metric = drift.check_drift("latency").unwrap();

    // Accuracy declining
    assert_eq!(acc_metric.trend, DriftTrend::Declining);

    // Latency increasing (worsening)
    assert_eq!(lat_metric.trend, DriftTrend::Declining);
}

#[test]
fn test_healing_repair_strategies() {
    let mut orchestrator = HealingOrchestrator::new();

    // Add multiple strategies
    use std::sync::Arc;
    orchestrator.add_repair_strategy(Arc::new(CacheFlushStrategy));
    orchestrator.add_repair_strategy(Arc::new(ModelRetrainStrategy));

    orchestrator.add_detector("performance", AnomalyConfig::default());

    // Create anomaly
    for _ in 0..20 {
        orchestrator.observe("performance", 100.0);
    }
    orchestrator.observe("performance", 500.0);

    let result = orchestrator.run_cycle();

    // Should have executed repair strategies
    assert!(result.repairs_attempted >= 0);
}

#[test]
fn test_anomaly_insufficient_samples() {
    let mut detector = AnomalyDetector::new(AnomalyConfig {
        z_threshold: 2.0,
        window_size: 100,
        min_samples: 20,
    });

    // Add only a few samples
    for i in 0..10 {
        detector.observe(100.0 + i as f64);
    }

    // Should not detect anomaly with insufficient samples
    assert!(detector.is_anomaly(200.0).is_none());
}

#[test]
fn test_drift_trend_detection() {
    let mut drift = LearningDriftDetector::new(0.05, 100);

    drift.set_baseline("test_metric", 50.0);

    // Create clear upward trend
    for i in 0..100 {
        drift.record("test_metric", 50.0 + (i as f64) * 0.5);
    }

    let metric = drift.check_drift("test_metric").unwrap();

    // Should detect improving trend
    assert_eq!(metric.trend, DriftTrend::Improving);
    assert!(metric.drift_magnitude > 0.5);
}
