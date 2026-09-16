#[path = "../src/routing/mod.rs"]
mod routing;
use routing::{Arc, MapIndex, RoadRouter, RoutingError};

fn arc(s: u32, t: u32, c: u32) -> Arc {
    Arc {
        source: s,
        target: t,
        cost: c,
    }
}
#[test]
fn directed_turns_parallel_closures_and_reuse() {
    let mut r = RoadRouter::new(
        4,
        vec![
            arc(0, 1, 1),
            arc(1, 2, 1),
            arc(1, 3, 2),
            arc(3, 2, 2),
            arc(0, 1, 5),
        ],
        vec![(0, 1)],
    )
    .unwrap();
    r.prepare(&[0, 2], 1000, || false).unwrap();
    let a = r.route(0, 2, true, 1000, || false).unwrap().unwrap();
    assert_eq!((a.cost, a.arcs), (5, vec![0, 2, 3]));
    assert!(r.route(2, 0, true, 1000, || false).unwrap().is_none());
    r.update(&[(2, None)]).unwrap();
    assert_eq!(r.landmark_count(), 2);
    assert_eq!(
        r.route(0, 2, true, 1000, || false).unwrap().unwrap().cost,
        6
    );
    r.update(&[(4, Some(0))]).unwrap();
    assert_eq!(r.landmark_count(), 0);
    assert_eq!(
        r.route(0, 2, true, 1000, || false).unwrap().unwrap().cost,
        1
    );
    r.prepare(&[0], 1000, || false).unwrap();
    r.update(&[(2, Some(2))]).unwrap();
    assert_eq!(r.landmark_count(), 0);
    assert_eq!(
        r.route(0, 0, true, 1000, || false).unwrap().unwrap().nodes,
        vec![0]
    );
}

#[test]
fn hostile_inputs_budgets_and_atomicity() {
    assert!(RoadRouter::new(0, vec![], vec![]).is_err());
    assert!(RoadRouter::new(routing::MAX_NODES + 1, vec![], vec![]).is_err());
    assert!(RoadRouter::new(2, vec![arc(0, 2, 1)], vec![]).is_err());
    assert!(RoadRouter::new(2, vec![arc(0, 1, u32::MAX)], vec![]).is_err());
    assert!(RoadRouter::from_arrays(2, &[0, 1, 0], &[1], &[]).is_err());
    assert!(RoadRouter::new(2, vec![arc(0, 1, 1)], vec![(0, 0)]).is_err());
    let mut r = RoadRouter::new(2, vec![arc(0, 1, 1)], vec![]).unwrap();
    assert_eq!(r.node_count(), 2);
    assert_eq!(r.arc_count(), 1);
    assert!(r.update(&[(0, None), (8, None)]).is_err());
    assert!(r.update(&[(0, None), (0, Some(2))]).is_err());
    assert_eq!(
        r.route(0, 1, true, 0, || false),
        Err(RoutingError::BudgetExceeded)
    );
    assert_eq!(
        r.route(0, 1, true, 100, || true),
        Err(RoutingError::Cancelled)
    );
    assert_eq!(r.route(0, 1, true, 100, || false).unwrap().unwrap().cost, 1);
    r.prepare(&[0], 100, || false).unwrap();
    assert!(r.prepare(&[1], 0, || false).is_err());
    assert_eq!(r.landmark_count(), 1);
    assert!(r.prepare(&[1], 100, || true).is_err());
    assert_eq!(r.landmark_count(), 1);
    assert!(r.prepare(&[0; 17], 100, || false).is_err());
    assert!(r.route(2, 0, true, 100, || false).is_err());
    r.set_coordinates(&[0., 0., 1., 1.]).unwrap();
    assert!(r.set_coordinates(&[f64::NAN, 0., 1., 1.]).is_err());
    assert_eq!(r.nearest(0., 0., 0.).unwrap().unwrap().node, 0);
}

// Independent Floyd-Warshall on the expanded arc-state graph, including source
// and sink. This does not share the production search/heuristic implementation.
fn oracle(
    n: usize,
    arcs: &[Arc],
    turns: &[(u32, u32)],
    source: usize,
    target: usize,
) -> Option<u64> {
    if source == target {
        return Some(0);
    }
    let m = arcs.len();
    let len = m + 2;
    let inf = u64::MAX / 4;
    let mut d = vec![vec![inf; len]; len];
    for (i, row) in d.iter_mut().enumerate() {
        row[i] = 0;
    }
    for (i, a) in arcs.iter().enumerate() {
        if a.source as usize == source {
            d[m][i] = a.cost as u64;
        }
        if a.target as usize == target {
            d[i][m + 1] = 0;
        }
        for (j, b) in arcs.iter().enumerate() {
            if a.target == b.source && !turns.contains(&(i as u32, j as u32)) {
                d[i][j] = d[i][j].min(b.cost as u64);
            }
        }
    }
    for k in 0..len {
        for i in 0..len {
            for j in 0..len {
                d[i][j] = d[i][j].min(d[i][k] + d[k][j]);
            }
        }
    }
    assert!(source < n && target < n);
    (d[m][m + 1] != inf).then_some(d[m][m + 1])
}
#[test]
fn directed_random_oracle_and_updates() {
    let mut seed = 20260915u64;
    let mut random = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        seed >> 32
    };
    for case in 0..160 {
        let n = 2 + random() as usize % 6;
        let mut arcs = Vec::new();
        let mut turns = Vec::new();
        for s in 0..n {
            for t in 0..n {
                if random() % 3 == 0 {
                    arcs.push(arc(s as u32, t as u32, (random() % 9) as u32));
                }
            }
        }
        if case % 2 == 0 {
            for (i, a) in arcs.iter().enumerate() {
                for (j, b) in arcs.iter().enumerate() {
                    if a.target == b.source && random() % 4 == 0 {
                        turns.push((i as u32, j as u32));
                    }
                }
            }
        }
        let mut r = RoadRouter::new(n, arcs.clone(), turns.clone()).unwrap();
        for phase in 0..3 {
            r.prepare(&[0, (n - 1) as u32], 100000, || false).unwrap();
            if phase > 0 && !arcs.is_empty() {
                let id = random() as usize % arcs.len();
                let cost = (random() % 12) as u32;
                r.update(&[(id as u32, Some(cost))]).unwrap();
                arcs[id].cost = cost;
            }
            for s in 0..n {
                for t in 0..n {
                    let expected = oracle(n, &arcs, &turns, s, t);
                    for alt in [false, true] {
                        let route = r.route(s as u32, t as u32, alt, 100000, || false).unwrap();
                        assert_eq!(
                            route.as_ref().map(|x| x.cost),
                            expected,
                            "case {case} {s}->{t} phase {phase}"
                        );
                        if let Some(path) = route {
                            assert_eq!(path.nodes.first(), Some(&(s as u32)));
                            assert_eq!(path.nodes.last(), Some(&(t as u32)));
                            let cost: u64 = path
                                .arcs
                                .iter()
                                .map(|&a| arcs[a as usize].cost as u64)
                                .sum();
                            assert_eq!(cost, path.cost);
                            for w in path.arcs.windows(2) {
                                assert!(!turns.contains(&(w[0], w[1])));
                            }
                            for (i, &id) in path.arcs.iter().enumerate() {
                                assert_eq!(arcs[id as usize].source, path.nodes[i]);
                                assert_eq!(arcs[id as usize].target, path.nodes[i + 1]);
                            }
                        }
                    }
                }
            }
        }
    }
}
#[test]
fn geographic_lookup_matches_linear_and_handles_date_line() {
    let coords = vec![0., 179.9, 0., -179.8, 90., 0., -90., 0., 0., 0.];
    let index = MapIndex::new(&coords).unwrap();
    assert_eq!(index.nearest(0., -180., 50000.).unwrap().unwrap().node, 0);
    assert!(index.nearest(50., 0., 1.).unwrap().is_none());
    assert!(index.nearest(f64::NAN, 0., 1.).is_err());
    assert!(index.nearest(0., 0., -1.).is_err());
    assert!(MapIndex::new(&[91., 0.]).is_err());
    assert!(MapIndex::new(&[0.]).is_err());
    let mut coords = Vec::new();
    for i in 0..1000 {
        coords.push((i * 73 % 18000) as f64 / 100. - 90.);
        coords.push((i * 197 % 36000) as f64 / 100. - 180.);
    }
    let index = MapIndex::new(&coords).unwrap();
    for i in 0..200 {
        let lat = (i * 127 % 18000) as f64 / 100. - 90.;
        let lon = (i * 337 % 36000) as f64 / 100. - 180.;
        let distance = |c: &[f64]| {
            let a = lat.to_radians();
            let b = c[0].to_radians();
            ((a - b) / 2.).sin().powi(2)
                + a.cos() * b.cos() * ((lon - c[1]).to_radians() / 2.).sin().powi(2)
        };
        let expected = coords
            .chunks_exact(2)
            .enumerate()
            .min_by(|a, b| distance(a.1).total_cmp(&distance(b.1)))
            .unwrap()
            .0;
        assert_eq!(
            index.nearest(lat, lon, 21_000_000.).unwrap().unwrap().node,
            expected as u32
        );
    }
}

fn field_event(
    id: &str,
    timestamp: u64,
    zone: &str,
    signals: (f32, f32, f32),
    privacy: &str,
    synthetic: bool,
) -> Vec<u8> {
    let (confidence, presence, motion) = signals;
    format!(r#"{{"event_id":"{id}","timestamp_ns":{timestamp},"observation":{{"zone_id":"{zone}","space_cell":null,"confidence":{confidence},"features":{{"presence":{presence},"motion_energy":{motion},"transient":0.0}},"privacy_class":"{privacy}"}},"provenance":{{"synthetic":{synthetic}}}}}"#).into_bytes()
}
fn field_event_sensor(
    id: &str,
    timestamp: u64,
    sensor: &str,
    zone: &str,
    presence: f32,
) -> Vec<u8> {
    format!(r#"{{"event_id":"{id}","timestamp_ns":{timestamp},"sensor":{{"device_id":"{sensor}"}},"observation":{{"zone_id":"{zone}","space_cell":null,"confidence":1.0,"features":{{"presence":{presence}}},"privacy_class":"P2"}},"provenance":{{"synthetic":true}}}}"#).into_bytes()
}
#[test]
fn rufield_verified_events_replan_and_expire_without_compounding() {
    use routing::rufield::{AwarenessPolicy, RuFieldRouter};
    let road = RoadRouter::new(
        4,
        vec![arc(0, 1, 10), arc(1, 3, 10), arc(0, 2, 30), arc(2, 3, 30)],
        vec![],
    )
    .unwrap();
    let mut aware = RuFieldRouter::new(
        road,
        AwarenessPolicy {
            max_penalty: 100,
            close_at_millionths: 900_000,
            ttl_ns: 100,
            max_lateness_ns: 10,
        },
    )
    .unwrap();
    aware.bind_zone("room-a".into(), 1).unwrap();
    assert_eq!(
        aware
            .route(0, 3, false, 1000, || false)
            .unwrap()
            .unwrap()
            .nodes,
        vec![0, 1, 3]
    );
    let update = aware
        .ingest_json(
            &field_event("e1", 100, "room-a", (1.0, 1.0, 1.0), "P2", false),
            true,
            100,
        )
        .unwrap();
    assert_eq!(
        (update.node, update.risk_millionths, update.changed_arcs),
        (1, 1_000_000, 2)
    );
    assert_eq!(
        aware
            .route(0, 3, false, 1000, || false)
            .unwrap()
            .unwrap()
            .nodes,
        vec![0, 2, 3]
    );
    let duplicate = aware
        .ingest_json(
            &field_event("e1", 100, "room-a", (1.0, 1.0, 1.0), "P2", false),
            true,
            100,
        )
        .unwrap();
    assert!(duplicate.duplicate);
    assert_eq!(duplicate.changed_arcs, 0);
    assert_eq!(aware.expire(200).unwrap(), 2);
    assert_eq!(
        aware
            .route(0, 3, false, 1000, || false)
            .unwrap()
            .unwrap()
            .cost,
        20
    );
    assert_eq!(aware.active_nodes(), 0);
}
#[test]
fn rufield_trust_privacy_time_and_mapping_fail_closed() {
    use routing::rufield::{AwarenessPolicy, RuFieldRouter};
    let make = || {
        RuFieldRouter::new(
            RoadRouter::new(2, vec![arc(0, 1, 5)], vec![]).unwrap(),
            AwarenessPolicy::default(),
        )
        .unwrap()
    };
    let mut aware = make();
    aware.bind_zone("z".into(), 0).unwrap();
    assert!(aware
        .ingest_json(
            &field_event("e", 10, "z", (1., 1., 0.), "P2", false),
            false,
            10
        )
        .is_err());
    assert!(aware
        .ingest_json(
            &field_event("e", 10, "z", (1., 1., 0.), "P4", true),
            true,
            10
        )
        .is_err());
    assert!(aware
        .ingest_json(
            &field_event("e", 10, "missing", (1., 1., 0.), "P2", true),
            false,
            10
        )
        .is_err());
    assert!(aware
        .ingest_json(
            &field_event("e", 10, "z", (f32::NAN, 1., 0.), "P2", true),
            false,
            10
        )
        .is_err());
    assert!(aware
        .ingest_json(&vec![b' '; 256 * 1024 + 1], true, 10)
        .is_err());
    let policy = AwarenessPolicy {
        max_penalty: 10,
        close_at_millionths: 1_000_001,
        ttl_ns: 100,
        max_lateness_ns: 10,
    };
    let mut aware = RuFieldRouter::new(
        RoadRouter::new(2, vec![arc(0, 1, 5)], vec![]).unwrap(),
        policy,
    )
    .unwrap();
    aware.bind_cell([1, 2, 3], 0).unwrap();
    let json=br#"{"event_id":"cell","timestamp_ns":100,"observation":{"zone_id":null,"space_cell":[1,2,3],"confidence":0.5,"features":{"presence":1.0},"privacy_class":"P1"},"provenance":{"synthetic":true}}"#;
    let update = aware.ingest_json(json, false, 100).unwrap();
    assert_eq!(update.risk_millionths, 500_000);
    assert_eq!(
        aware
            .route(0, 1, false, 100, || false)
            .unwrap()
            .unwrap()
            .cost,
        10
    );
    assert!(aware
        .ingest_json(
            &field_event("future", 1000, "z", (1., 1., 0.), "P2", true),
            false,
            100
        )
        .is_err());
}
#[test]
fn rufield_accepts_real_ruview_signed_wire_fixture() {
    use routing::rufield::{AwarenessPolicy, RuFieldRouter};
    let mut aware = RuFieldRouter::new(
        RoadRouter::new(2, vec![arc(0, 1, 5)], vec![]).unwrap(),
        AwarenessPolicy {
            ttl_ns: 200,
            max_lateness_ns: 100,
            ..AwarenessPolicy::default()
        },
    )
    .unwrap();
    aware.bind_zone("room-a".into(), 0).unwrap();
    let event = include_bytes!("fixtures/ruview-rufield-v0.1.json");
    let update = aware.ingest_json(event, true, 100).unwrap();
    assert_eq!(update.node, 0);
    assert!(update.risk_millionths > 800_000);
}

#[test]
fn rufield_aggregates_independent_sensors_per_node() {
    use routing::rufield::{AwarenessPolicy, RuFieldRouter};
    let mut aware = RuFieldRouter::new(
        RoadRouter::new(2, vec![arc(0, 1, 10)], vec![]).unwrap(),
        AwarenessPolicy {
            max_penalty: 100,
            close_at_millionths: 1_000_001,
            ttl_ns: 100,
            max_lateness_ns: 10,
        },
    )
    .unwrap();
    aware.bind_zone("shared".into(), 0).unwrap();
    aware
        .ingest_json(
            &field_event_sensor("a", 100, "sensor-a", "shared", 1.0),
            false,
            100,
        )
        .unwrap();
    aware
        .ingest_json(
            &field_event_sensor("b", 90, "sensor-b", "shared", 0.5),
            false,
            100,
        )
        .unwrap();
    assert_eq!(aware.active_nodes(), 1);
    assert_eq!(
        aware
            .route(0, 1, false, 100, || false)
            .unwrap()
            .unwrap()
            .cost,
        110
    );
    assert_eq!(aware.expire(190).unwrap(), 0);
    assert_eq!(aware.expire(200).unwrap(), 1);
    assert_eq!(aware.active_nodes(), 0);
    assert_eq!(
        aware
            .route(0, 1, false, 100, || false)
            .unwrap()
            .unwrap()
            .cost,
        10
    );
}
