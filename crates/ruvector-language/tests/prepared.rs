use ruvector_language::{
    runtime::{self, PreparedSnapshot},
    *,
};
#[path = "../examples/support/memory.rs"]
mod memory;

#[test]
fn prepared_queries_match_complete_receipts_for_varied_inputs_and_clocks() {
    for seed in 0..32 {
        let mut r = memory::request(32, seed);
        let prepared = PreparedSnapshot::new(&r.snapshot).unwrap();
        for now in [900, 1000, 1100] {
            r.context.now = now;
            let cold = runtime::execute(&r.program, &r.snapshot, &r.context, None).unwrap();
            let warm = prepared.execute(&r.program, &r.context, None).unwrap();
            assert_eq!(warm, cold);
            assert_eq!(
                warm,
                prepared.execute(&r.program, &r.context, None).unwrap()
            );
        }
    }
}
#[test]
fn preparation_cannot_cache_authority_or_bypass_logical_budgets() {
    let mut r = memory::request(32, 0);
    let prepared = PreparedSnapshot::new(&r.snapshot).unwrap();
    for budget in [1, 100, 1000, 10_000_000] {
        r.context.max_steps = budget;
        let a = runtime::execute(&r.program, &r.snapshot, &r.context, None);
        let b = prepared.execute(&r.program, &r.context, None);
        match (a, b) {
            (Ok(a), Ok(b)) => assert_eq!(a, b),
            (Err(a), Err(b)) => assert_eq!(a.code, b.code),
            _ => panic!("budget semantics differ"),
        }
    }
    r.context.capabilities.clear();
    assert_eq!(
        prepared
            .execute(&r.program, &r.context, None)
            .unwrap_err()
            .code,
        "UNAUTHORIZED"
    );
    r.context.max_results = 0;
    assert_eq!(
        prepared
            .execute(&r.program, &r.context, None)
            .unwrap_err()
            .code,
        "INVALID_INPUT"
    );
}
#[test]
fn invalid_snapshots_fail_before_they_can_be_prepared() {
    let mut r = memory::request(32, 0);
    r.snapshot.nodes.push(r.snapshot.nodes[0].clone());
    assert!(PreparedSnapshot::new(&r.snapshot).is_err());
    r.snapshot.nodes.pop();
    r.snapshot.nodes[0].vector = Some(vec![f32::NAN]);
    assert!(PreparedSnapshot::new(&r.snapshot).is_err());
}
#[test]
fn provider_is_called_on_each_prepared_execution() {
    struct Provider(std::cell::Cell<usize>);
    impl Predictor for Provider {
        fn predict(&self, _: &PredictionRequest) -> Result<PredictionBatch> {
            self.0.set(self.0.get() + 1);
            Err(Error::new("PROVIDER_UNAVAILABLE", "test provider"))
        }
    }
    let mut r = memory::request(32, 0);
    r.program=parser::parse(r#"program p { effect graph.read; effect model.predict; effect decision.emit; budget 10000000; let n = select ["m00000"]; let p = predict n "Related" model "m" top 1; emit p; }"#).unwrap();
    r.context.capabilities.insert(Effect::ModelPredict);
    let prepared = PreparedSnapshot::new(&r.snapshot).unwrap();
    let provider = Provider(std::cell::Cell::new(0));
    for _ in 0..2 {
        assert_eq!(
            prepared
                .execute(&r.program, &r.context, Some(&provider))
                .unwrap_err()
                .code,
            "PROVIDER_UNAVAILABLE"
        );
    }
    assert_eq!(provider.0.get(), 2);
    r.context.capabilities.remove(&Effect::ModelPredict);
    assert_eq!(
        prepared
            .execute(&r.program, &r.context, Some(&provider))
            .unwrap_err()
            .code,
        "UNAUTHORIZED"
    );
    assert_eq!(provider.0.get(), 2);
}
