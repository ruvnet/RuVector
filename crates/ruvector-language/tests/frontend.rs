use ruvector_language::{
    compiler::{self, MAX_BUDGET, MAX_DEPTH, MAX_LIST_ITEMS, MAX_STEPS, MAX_VECTOR_DIM},
    parser,
    types::{Effect, Op, Step},
};

const SIMPLE: &str = r#"program incident {
    effect graph.read;
    effect decision.emit;
    budget 100000;
    let nodeset = select ["api"];
    let evidence = observe nodeset "Failing" within 300;
    let decision = assess evidence;
    emit decision;
}"#;

const COMPLETE: &str = r#"// all executable operations
program diagnose {
    effect graph.read; effect model.predict; effect decision.emit;
    budget 100000;
    let s = nodes "Service";
    let selected = select ["api", "db"];
    let n = traverse s "DEPENDS_ON" depth 3;
    let v = similar n [0.1, -0.2, 3e-2] top 10;
    let e = observe v "Failing" within 300;
    let d = infer [e] using ["failure_rule"];
    let p = predict selected "Cause" model "incident_v1" top 3;
    let a = assess d;
    emit a;
}"#;

#[test]
fn text_and_json_are_canonically_identical() {
    let program = parser::parse(COMPLETE).unwrap();
    assert_eq!(program.steps.len(), 8);
    let json = compiler::canonical_json(&program).unwrap();
    let reconstructed = parser::parse_json(&json).unwrap();
    assert_eq!(program, reconstructed);
    assert_eq!(json, compiler::canonical_json(&reconstructed).unwrap());
    assert_eq!(program, parser::parse(&json).unwrap());
    let reordered = COMPLETE.replace(
        "effect graph.read; effect model.predict; effect decision.emit;",
        "effect decision.emit; effect graph.read; effect model.predict;",
    );
    assert_eq!(
        json,
        compiler::canonical_json(&parser::parse(&reordered).unwrap()).unwrap()
    );
}

#[test]
fn canonical_object_order_is_stable_across_json_field_order() {
    let a = r#"{"version":1,"name":"x","effects":["graph.read","decision.emit"],"budget":100,"steps":[{"bind":"n","op":"nodes","label":"Service"},{"bind":"e","op":"observe","input":"n","predicate":"p","within":1},{"bind":"d","op":"assess","input":"e"}],"emit":"d"}"#;
    let b = r#"{"emit":"d","steps":[{"label":"Service","op":"nodes","bind":"n"},{"within":1,"predicate":"p","input":"n","op":"observe","bind":"e"},{"input":"e","bind":"d","op":"assess"}],"budget":100,"effects":["decision.emit","graph.read"],"name":"x","version":1}"#;
    assert_eq!(
        compiler::canonical_json(&parser::parse_json(a).unwrap()).unwrap(),
        compiler::canonical_json(&parser::parse_json(b).unwrap()).unwrap()
    );
}

#[test]
fn rejects_unknown_fields_at_both_program_and_operation_boundaries() {
    let program = parser::parse(SIMPLE).unwrap();
    let mut json = serde_json::to_value(&program).unwrap();
    json["shell"] = serde_json::json!("exec");
    assert_eq!(
        parser::parse_json(&json.to_string()).unwrap_err().code,
        "json_schema"
    );
    json.as_object_mut().unwrap().remove("shell");
    json["steps"][0]["unchecked"] = serde_json::json!(true);
    assert_eq!(
        parser::parse_json(&json.to_string()).unwrap_err().code,
        "json_schema"
    );
}

#[test]
fn rejects_duplicate_keys_including_escaped_and_nested_keys() {
    for json in [
        r#"{"version":1,"version":2}"#,
        r#"{"nested":{"a":1,"\u0061":2}}"#,
        r#"[{"op":"nodes","op":"predict"}]"#,
    ] {
        let error = parser::strict_json(json).unwrap_err();
        assert_eq!(error.code, "json_parse");
        assert!(error.message.contains("duplicate JSON key"));
    }
}

#[test]
fn rejects_prediction_as_evidence_and_evidence_as_prediction_input() {
    let assessed_prediction = COMPLETE.replace("let a = assess d;", "let a = assess p;");
    assert_eq!(
        parser::parse(&assessed_prediction).unwrap_err().code,
        "type"
    );
    let inferred_prediction = COMPLETE.replace("infer [e]", "infer [p]").replace(
        "let p = predict selected \"Cause\" model \"incident_v1\" top 3;",
        "",
    );
    assert_eq!(
        parser::parse(&inferred_prediction).unwrap_err().code,
        "reference"
    );
    let inferred_prediction = COMPLETE
        .replace("let d = infer [e] using [\"failure_rule\"];", "")
        .replace(
            "let a = assess d;",
            "let d = infer [p] using [\"failure_rule\"]; let a = assess d;",
        );
    assert_eq!(
        parser::parse(&inferred_prediction).unwrap_err().code,
        "type"
    );
    assert_eq!(
        parser::parse(&COMPLETE.replace("predict selected", "predict e"))
            .unwrap_err()
            .code,
        "type"
    );
}

#[test]
fn validates_all_input_type_flows_and_emit() {
    for source in [
        SIMPLE.replace("assess evidence", "assess nodeset"),
        SIMPLE.replace("observe nodeset", "observe evidence"),
        SIMPLE.replace("emit decision", "emit evidence"),
        SIMPLE.replace("emit decision", "emit nodeset"),
    ] {
        assert!(parser::parse(&source).is_err());
    }
    let predictions = COMPLETE.replace("emit a", "emit p");
    assert!(parser::parse(&predictions).is_ok());
    let inferred = COMPLETE.replace(
        "let a = assess d;",
        "let d2 = infer [e, d] using [\"further_rule\"]; let a = assess d2;",
    );
    assert!(parser::parse(&inferred).is_ok());
}

#[test]
fn effects_are_required_and_unknown_or_duplicate_effects_fail() {
    for effect in ["graph.read", "decision.emit", "model.predict"] {
        let source = COMPLETE.replace(&format!("effect {effect};"), "");
        assert_eq!(parser::parse(&source).unwrap_err().code, "effect");
    }
    assert_eq!(
        parser::parse(&SIMPLE.replace("effect graph.read;", "effect graph.write;"))
            .unwrap_err()
            .code,
        "effect"
    );
    assert_eq!(
        parser::parse(&SIMPLE.replace(
            "effect graph.read;",
            "effect graph.read; effect graph.read;"
        ))
        .unwrap_err()
        .code,
        "duplicate_effect"
    );
    let json = compiler::canonical_json(&parser::parse(SIMPLE).unwrap()).unwrap();
    let duplicate = json.replace("\"graph.read\"", "\"graph.read\",\"graph.read\"");
    assert_eq!(
        parser::parse_json(&duplicate).unwrap_err().code,
        "duplicate_effect"
    );
}

#[test]
fn unknown_and_forward_references_and_duplicate_bindings_fail() {
    for source in [
        SIMPLE.replace("observe nodeset", "observe absent"),
        SIMPLE.replace("observe nodeset", "observe decision"),
    ] {
        assert_eq!(parser::parse(&source).unwrap_err().code, "reference");
    }
    let duplicate = SIMPLE.replace("let decision", "let evidence");
    assert_eq!(
        parser::parse(&duplicate).unwrap_err().code,
        "duplicate_binding"
    );
}

#[test]
fn integer_parameters_do_not_accept_float_notation_or_overflow() {
    for value in [
        "1.0",
        "1e3",
        "-1",
        "18446744073709551616",
        "01",
        "NaN",
        "1e999",
    ] {
        assert!(
            parser::parse(&SIMPLE.replace("budget 100000", &format!("budget {value}"))).is_err(),
            "accepted {value}"
        );
    }
    assert!(parser::parse(&COMPLETE.replace("depth 3", "depth 1.2")).is_err());
    assert!(parser::parse(&COMPLETE.replace("top 10", "top 2e0")).is_err());
}

#[test]
fn resource_and_schema_bounds_apply_to_rust_constructed_programs() {
    let original = parser::parse(COMPLETE).unwrap();
    let mut program = original.clone();
    program.version = 2;
    assert_eq!(compiler::check(&program).unwrap_err().code, "version");
    for budget in [0, MAX_BUDGET + 1] {
        program = original.clone();
        program.budget = budget;
        assert_eq!(compiler::check(&program).unwrap_err().code, "budget");
    }
    program = original.clone();
    program.steps = vec![program.steps[0].clone(); MAX_STEPS + 1];
    assert_eq!(compiler::check(&program).unwrap_err().code, "step_limit");
    for op in [
        Op::Traverse {
            input: "s".into(),
            relation: "r".into(),
            depth: MAX_DEPTH + 1,
        },
        Op::Similar {
            input: "s".into(),
            vector: vec![0.; MAX_VECTOR_DIM + 1],
            k: 1,
        },
        Op::Similar {
            input: "s".into(),
            vector: vec![f32::NAN],
            k: 1,
        },
        Op::Similar {
            input: "s".into(),
            vector: vec![f32::INFINITY],
            k: 1,
        },
        Op::Similar {
            input: "s".into(),
            vector: vec![0.],
            k: MAX_LIST_ITEMS + 1,
        },
        Op::Observe {
            input: "s".into(),
            predicate: "p".into(),
            within: 0,
        },
        Op::Select { ids: vec![] },
    ] {
        program = original.clone();
        program.steps[1].op = op;
        assert!(compiler::check(&program).is_err());
        assert!(compiler::canonical_json(&program).is_err());
    }
    program = original;
    program.effects.remove(&Effect::GraphRead);
    assert_eq!(compiler::check(&program).unwrap_err().code, "effect");
}

#[test]
fn malformed_text_never_silently_ignores_trailing_statements() {
    for source in [
        "".to_string(),
        SIMPLE.replace("emit decision;", ""),
        SIMPLE.replace("emit decision;", "emit decision; emit decision;"),
        format!("{SIMPLE} program ignored {{}}"),
        SIMPLE.replace("budget 100000;", "budget 100000; budget 1;"),
        SIMPLE.replace("[\"api\"]", "[\"api\",]"),
        SIMPLE.replace("assess evidence", "execute evidence"),
    ] {
        assert!(parser::parse(&source).is_err(), "accepted {source}");
    }
}

#[test]
fn json_strings_and_comments_preserve_unicode_and_escapes() {
    let source = SIMPLE
        .replace("\"api\"", r#""api//\"\u2603\n""#)
        .replace("let evidence", "// comment\n let evidence");
    let program = parser::parse(&source).unwrap();
    assert_eq!(
        program.steps[0].op,
        Op::Select {
            ids: vec!["api//\"☃\n".into()]
        }
    );
    for invalid in [
        "\"unterminated",
        "\"bad\\q\"",
        "\"raw\nnewline\"",
        "\"\\uD800\"",
    ] {
        assert!(parser::parse(&SIMPLE.replace("\"api\"", invalid)).is_err());
    }
}

#[test]
fn enforces_source_depth_and_collection_limits() {
    assert_eq!(
        parser::parse(&" ".repeat(parser::MAX_SOURCE_BYTES + 1))
            .unwrap_err()
            .code,
        "source_limit"
    );
    assert_eq!(
        parser::strict_json(&" ".repeat(parser::MAX_JSON_BYTES + 1))
            .unwrap_err()
            .code,
        "source_limit"
    );
    let deep = format!("{}0{}", "[".repeat(70), "]".repeat(70));
    assert!(parser::strict_json(&deep)
        .unwrap_err()
        .message
        .contains("nesting"));
    let wide = format!("[{}0]", "0,".repeat(100_000));
    assert!(parser::strict_json(&wide)
        .unwrap_err()
        .message
        .contains("array exceeds"));
    assert!(parser::strict_json("{} []").is_err());
}

#[test]
fn rejects_nonfinite_json_vectors_and_oversized_text_vectors() {
    let program = parser::parse(COMPLETE).unwrap();
    let json = compiler::canonical_json(&program).unwrap();
    assert!(parser::parse_json(&json.replace("0.1", "1e100")).is_err());
    assert!(parser::parse_json(&json.replace("0.1", "NaN")).is_err());
    let wide = format!("[{}0]", "0,".repeat(MAX_VECTOR_DIM));
    assert!(parser::parse(&COMPLETE.replace("[0.1, -0.2, 3e-2]", &wide)).is_err());
}

#[test]
fn invalid_names_and_zero_parameters_are_rejected() {
    let program = parser::parse(SIMPLE).unwrap();
    for name in ["", " ", "bad.name", "10invalid", "é"] {
        let mut invalid = program.clone();
        invalid.name = name.into();
        assert_eq!(compiler::check(&invalid).unwrap_err().code, "name");
    }
    let mut invalid = program.clone();
    invalid.steps[0] = Step {
        bind: "n".into(),
        op: Op::Nodes { label: " ".into() },
    };
    assert_eq!(compiler::check(&invalid).unwrap_err().code, "name");
    for replacement in [
        COMPLETE.replace("depth 3", "depth 0"),
        COMPLETE.replace("top 10", "top 0"),
        COMPLETE.replace("within 300", "within 0"),
        COMPLETE.replace("[e] using", "[] using"),
    ] {
        assert!(parser::parse(&replacement).is_err());
    }
}

#[test]
fn combined_request_enforces_program_boundary_before_effect_deduplication() {
    let program = parser::parse(SIMPLE).unwrap();
    let request = serde_json::json!({
        "program": program,
        "snapshot": {"version":1,"revision":"test","nodes":[],"edges":[],"observations":[],"rules":[]},
        "context": {"now":1,"capabilities":["graph.read","decision.emit"],"max_steps":100000,"max_results":100}
    });
    let source = request.to_string();
    assert!(parser::parse_request_json(&source).is_ok());
    let duplicate = source.replacen("\"effects\":[", "\"effects\":[\"graph.read\",", 1);
    assert_eq!(
        parser::parse_request_json(&duplicate).unwrap_err().code,
        "duplicate_effect"
    );
    let mut large_program = parser::parse(SIMPLE).unwrap();
    for index in large_program.steps.len()..MAX_STEPS {
        large_program.steps.push(Step {
            bind: format!("extra_{index}"),
            op: Op::Nodes {
                label: "x".repeat(1024),
            },
        });
    }
    assert_eq!(
        compiler::check(&large_program).unwrap_err().code,
        "source_limit"
    );
    let mut oversized = request;
    oversized["program"] = serde_json::to_value(large_program).unwrap();
    assert_eq!(
        parser::parse_request_json(&oversized.to_string())
            .unwrap_err()
            .code,
        "source_limit"
    );
}
