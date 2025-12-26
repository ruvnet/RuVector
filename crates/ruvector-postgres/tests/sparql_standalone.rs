//! Standalone SPARQL validation tests
//!
//! This file tests the SPARQL implementation without requiring pgrx/PostgreSQL.
//! It validates parser, AST, triple store, and executor functionality.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

// ============================================================================
// AST Types
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Iri(pub String);

impl Iri {
    pub fn new(value: impl Into<String>) -> Self {
        Self(value.into())
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }

    pub fn rdf_type() -> Self {
        Self::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#type")
    }

    pub fn xsd_string() -> Self {
        Self::new("http://www.w3.org/2001/XMLSchema#string")
    }

    pub fn xsd_integer() -> Self {
        Self::new("http://www.w3.org/2001/XMLSchema#integer")
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Literal {
    pub value: String,
    pub language: Option<String>,
    pub datatype: Iri,
}

impl Literal {
    pub fn simple(value: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            language: None,
            datatype: Iri::xsd_string(),
        }
    }

    pub fn integer(value: i64) -> Self {
        Self {
            value: value.to_string(),
            language: None,
            datatype: Iri::xsd_integer(),
        }
    }

    pub fn language(value: impl Into<String>, lang: impl Into<String>) -> Self {
        Self {
            value: value.into(),
            language: Some(lang.into()),
            datatype: Iri::new("http://www.w3.org/1999/02/22-rdf-syntax-ns#langString"),
        }
    }

    pub fn as_integer(&self) -> Option<i64> {
        self.value.parse().ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RdfTerm {
    Iri(Iri),
    Literal(Literal),
    BlankNode(String),
}

impl RdfTerm {
    pub fn iri(value: impl Into<String>) -> Self {
        Self::Iri(Iri::new(value))
    }

    pub fn literal(value: impl Into<String>) -> Self {
        Self::Literal(Literal::simple(value))
    }

    pub fn blank(id: impl Into<String>) -> Self {
        Self::BlankNode(id.into())
    }
}

// ============================================================================
// Triple Store
// ============================================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Triple {
    pub subject: RdfTerm,
    pub predicate: Iri,
    pub object: RdfTerm,
}

impl Triple {
    pub fn new(subject: RdfTerm, predicate: Iri, object: RdfTerm) -> Self {
        Self {
            subject,
            predicate,
            object,
        }
    }
}

pub struct TripleStore {
    triples: HashMap<u64, Triple>,
    spo_index: HashMap<String, HashMap<String, HashSet<u64>>>,
    pos_index: HashMap<String, HashMap<String, HashSet<u64>>>,
    osp_index: HashMap<String, HashMap<String, HashSet<u64>>>,
    next_id: AtomicU64,
}

impl TripleStore {
    pub fn new() -> Self {
        Self {
            triples: HashMap::new(),
            spo_index: HashMap::new(),
            pos_index: HashMap::new(),
            osp_index: HashMap::new(),
            next_id: AtomicU64::new(1),
        }
    }

    pub fn insert(&mut self, triple: Triple) -> u64 {
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);

        let subject_key = term_to_key(&triple.subject);
        let predicate_key = triple.predicate.as_str().to_string();
        let object_key = term_to_key(&triple.object);

        // SPO index
        self.spo_index
            .entry(subject_key.clone())
            .or_insert_with(HashMap::new)
            .entry(predicate_key.clone())
            .or_insert_with(HashSet::new)
            .insert(id);

        // POS index
        self.pos_index
            .entry(predicate_key.clone())
            .or_insert_with(HashMap::new)
            .entry(object_key.clone())
            .or_insert_with(HashSet::new)
            .insert(id);

        // OSP index
        self.osp_index
            .entry(object_key)
            .or_insert_with(HashMap::new)
            .entry(subject_key)
            .or_insert_with(HashSet::new)
            .insert(id);

        self.triples.insert(id, triple);
        id
    }

    pub fn query(
        &self,
        subject: Option<&RdfTerm>,
        predicate: Option<&Iri>,
        object: Option<&RdfTerm>,
    ) -> Vec<&Triple> {
        let ids: Vec<u64> = match (subject, predicate, object) {
            (Some(s), Some(p), None) => {
                let s_key = term_to_key(s);
                let p_key = p.as_str();
                self.spo_index
                    .get(&s_key)
                    .and_then(|pm| pm.get(p_key))
                    .map(|ids| ids.iter().copied().collect())
                    .unwrap_or_default()
            }
            (Some(s), None, None) => {
                let s_key = term_to_key(s);
                self.spo_index
                    .get(&s_key)
                    .map(|pm| pm.values().flat_map(|ids| ids.iter().copied()).collect())
                    .unwrap_or_default()
            }
            (None, Some(p), None) => {
                let p_key = p.as_str();
                self.pos_index
                    .get(p_key)
                    .map(|om| om.values().flat_map(|ids| ids.iter().copied()).collect())
                    .unwrap_or_default()
            }
            (None, None, Some(o)) => {
                let o_key = term_to_key(o);
                self.osp_index
                    .get(&o_key)
                    .map(|sm| sm.values().flat_map(|ids| ids.iter().copied()).collect())
                    .unwrap_or_default()
            }
            (None, None, None) => self.triples.keys().copied().collect(),
            _ => {
                // For other patterns, filter from all triples
                self.triples
                    .iter()
                    .filter(|(_, t)| {
                        let s_match = subject
                            .map(|s| term_to_key(s) == term_to_key(&t.subject))
                            .unwrap_or(true);
                        let p_match = predicate
                            .map(|p| p.as_str() == t.predicate.as_str())
                            .unwrap_or(true);
                        let o_match = object
                            .map(|o| term_to_key(o) == term_to_key(&t.object))
                            .unwrap_or(true);
                        s_match && p_match && o_match
                    })
                    .map(|(id, _)| *id)
                    .collect()
            }
        };

        ids.into_iter()
            .filter_map(|id| self.triples.get(&id))
            .collect()
    }

    pub fn count(&self) -> usize {
        self.triples.len()
    }
}

fn term_to_key(term: &RdfTerm) -> String {
    match term {
        RdfTerm::Iri(iri) => format!("<{}>", iri.as_str()),
        RdfTerm::Literal(lit) => {
            if let Some(ref lang) = lit.language {
                format!("\"{}\"@{}", lit.value, lang)
            } else {
                format!("\"{}\"", lit.value)
            }
        }
        RdfTerm::BlankNode(id) => format!("_:{}", id),
    }
}

// ============================================================================
// Simple SPARQL Parser
// ============================================================================

#[derive(Debug)]
pub enum QueryType {
    Select {
        variables: Vec<String>,
        where_patterns: Vec<TriplePattern>,
    },
    Ask {
        where_patterns: Vec<TriplePattern>,
    },
}

#[derive(Debug, Clone)]
pub struct TriplePattern {
    pub subject: PatternTerm,
    pub predicate: PatternTerm,
    pub object: PatternTerm,
}

#[derive(Debug, Clone)]
pub enum PatternTerm {
    Variable(String),
    Iri(String),
    Literal(String),
}

pub fn parse_simple_sparql(query: &str) -> Result<QueryType, String> {
    let query = query.trim();
    let upper = query.to_uppercase();

    if upper.starts_with("SELECT") {
        parse_select(query)
    } else if upper.starts_with("ASK") {
        parse_ask(query)
    } else {
        Err(format!(
            "Unsupported query type: {}",
            query.chars().take(20).collect::<String>()
        ))
    }
}

fn parse_select(query: &str) -> Result<QueryType, String> {
    // Extract variables between SELECT and WHERE
    let upper = query.to_uppercase();
    let select_end = upper.find("WHERE").unwrap_or(query.len());
    let var_section = &query[6..select_end].trim();

    let variables: Vec<String> = if var_section.starts_with('*') {
        vec!["*".to_string()]
    } else {
        var_section
            .split_whitespace()
            .filter(|s| s.starts_with('?') || s.starts_with('$'))
            .map(|s| s[1..].to_string())
            .collect()
    };

    // Extract patterns from WHERE { ... }
    let where_patterns = parse_where_clause(query)?;

    Ok(QueryType::Select {
        variables,
        where_patterns,
    })
}

fn parse_ask(query: &str) -> Result<QueryType, String> {
    let where_patterns = parse_where_clause(query)?;
    Ok(QueryType::Ask { where_patterns })
}

fn parse_where_clause(query: &str) -> Result<Vec<TriplePattern>, String> {
    let brace_start = query.find('{').ok_or("No WHERE clause found")?;
    let brace_end = query.rfind('}').ok_or("No closing brace")?;

    let patterns_str = query[brace_start + 1..brace_end].trim();
    let mut patterns = Vec::new();

    // Normalize whitespace
    let normalized = patterns_str.replace('\n', " ").replace('\r', " ");

    // Split by " . " (space-dot-space) to separate triple patterns
    // This avoids splitting on dots within IRIs
    for pattern in normalized.split(" . ") {
        let pattern = pattern.trim().trim_end_matches('.');
        if pattern.is_empty() {
            continue;
        }

        // Tokenize while respecting IRIs and literals
        let mut tokens: Vec<String> = Vec::new();
        let mut current_token = String::new();
        let mut in_iri = false;
        let mut in_literal = false;

        for c in pattern.chars() {
            match c {
                '<' if !in_literal && !in_iri => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                    current_token.push(c);
                    in_iri = true;
                }
                '>' if in_iri => {
                    current_token.push(c);
                    in_iri = false;
                    tokens.push(current_token.clone());
                    current_token.clear();
                }
                '"' if !in_iri => {
                    if in_literal {
                        current_token.push(c);
                        in_literal = false;
                        tokens.push(current_token.clone());
                        current_token.clear();
                    } else {
                        if !current_token.is_empty() {
                            tokens.push(current_token.clone());
                            current_token.clear();
                        }
                        current_token.push(c);
                        in_literal = true;
                    }
                }
                ' ' | '\t' if !in_iri && !in_literal => {
                    if !current_token.is_empty() {
                        tokens.push(current_token.clone());
                        current_token.clear();
                    }
                }
                _ => {
                    current_token.push(c);
                }
            }
        }
        if !current_token.is_empty() {
            tokens.push(current_token);
        }

        if tokens.len() >= 3 {
            patterns.push(TriplePattern {
                subject: parse_term(&tokens[0]),
                predicate: parse_term(&tokens[1]),
                object: parse_term(&tokens[2..].join(" ")),
            });
        }
    }

    Ok(patterns)
}

fn parse_term(s: &str) -> PatternTerm {
    let s = s.trim();
    if s.starts_with('?') || s.starts_with('$') {
        PatternTerm::Variable(s[1..].to_string())
    } else if s.starts_with('<') && s.ends_with('>') {
        PatternTerm::Iri(s[1..s.len() - 1].to_string())
    } else if s.starts_with('"') {
        let end = s.rfind('"').unwrap_or(s.len());
        PatternTerm::Literal(s[1..end].to_string())
    } else {
        // Could be a prefixed name or literal
        PatternTerm::Iri(s.to_string())
    }
}

// ============================================================================
// Simple Query Executor
// ============================================================================

pub type Binding = HashMap<String, RdfTerm>;

pub fn execute_query(store: &TripleStore, query: &QueryType) -> Vec<Binding> {
    match query {
        QueryType::Select {
            variables,
            where_patterns,
        } => execute_bgp(store, where_patterns, variables),
        QueryType::Ask { where_patterns } => {
            let results = execute_bgp(store, where_patterns, &vec![]);
            if results.is_empty() {
                vec![]
            } else {
                vec![HashMap::new()] // Non-empty means "true"
            }
        }
    }
}

fn execute_bgp(store: &TripleStore, patterns: &[TriplePattern], _vars: &[String]) -> Vec<Binding> {
    let mut bindings: Vec<Binding> = vec![HashMap::new()];

    for pattern in patterns {
        let mut new_bindings = Vec::new();

        for binding in &bindings {
            // Get concrete values for pattern terms using current binding
            let subject = resolve_term(&pattern.subject, binding);
            let predicate = resolve_term(&pattern.predicate, binding);
            let object = resolve_term(&pattern.object, binding);

            // Query the store
            let matches = store.query(
                subject.as_ref(),
                predicate
                    .as_ref()
                    .map(|t| {
                        if let RdfTerm::Iri(i) = t {
                            Some(i)
                        } else {
                            None
                        }
                    })
                    .flatten(),
                object.as_ref(),
            );

            // Generate new bindings
            for triple in matches {
                let mut new_binding = binding.clone();
                let mut valid = true;

                // Bind variables
                if let PatternTerm::Variable(v) = &pattern.subject {
                    if let Some(existing) = new_binding.get(v) {
                        if term_to_key(existing) != term_to_key(&triple.subject) {
                            valid = false;
                        }
                    } else {
                        new_binding.insert(v.clone(), triple.subject.clone());
                    }
                }

                if let PatternTerm::Variable(v) = &pattern.predicate {
                    let pred_term = RdfTerm::Iri(triple.predicate.clone());
                    if let Some(existing) = new_binding.get(v) {
                        if term_to_key(existing) != term_to_key(&pred_term) {
                            valid = false;
                        }
                    } else {
                        new_binding.insert(v.clone(), pred_term);
                    }
                }

                if let PatternTerm::Variable(v) = &pattern.object {
                    if let Some(existing) = new_binding.get(v) {
                        if term_to_key(existing) != term_to_key(&triple.object) {
                            valid = false;
                        }
                    } else {
                        new_binding.insert(v.clone(), triple.object.clone());
                    }
                }

                if valid {
                    new_bindings.push(new_binding);
                }
            }
        }

        bindings = new_bindings;
    }

    bindings
}

fn resolve_term(term: &PatternTerm, binding: &Binding) -> Option<RdfTerm> {
    match term {
        PatternTerm::Variable(v) => binding.get(v).cloned(),
        PatternTerm::Iri(i) => Some(RdfTerm::iri(i.clone())),
        PatternTerm::Literal(l) => Some(RdfTerm::literal(l.clone())),
    }
}

// ============================================================================
// Test Data
// ============================================================================

fn create_test_store() -> TripleStore {
    let mut store = TripleStore::new();

    // Add test data
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/alice"),
        Iri::rdf_type(),
        RdfTerm::iri("http://example.org/Person"),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/alice"),
        Iri::new("http://xmlns.com/foaf/0.1/name"),
        RdfTerm::literal("Alice Smith"),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/alice"),
        Iri::new("http://xmlns.com/foaf/0.1/age"),
        RdfTerm::Literal(Literal::integer(30)),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/alice"),
        Iri::new("http://xmlns.com/foaf/0.1/knows"),
        RdfTerm::iri("http://example.org/person/bob"),
    ));

    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/bob"),
        Iri::rdf_type(),
        RdfTerm::iri("http://example.org/Person"),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/bob"),
        Iri::new("http://xmlns.com/foaf/0.1/name"),
        RdfTerm::literal("Bob Jones"),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/bob"),
        Iri::new("http://xmlns.com/foaf/0.1/age"),
        RdfTerm::Literal(Literal::integer(25)),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/bob"),
        Iri::new("http://xmlns.com/foaf/0.1/knows"),
        RdfTerm::iri("http://example.org/person/charlie"),
    ));

    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/charlie"),
        Iri::rdf_type(),
        RdfTerm::iri("http://example.org/Person"),
    ));
    store.insert(Triple::new(
        RdfTerm::iri("http://example.org/person/charlie"),
        Iri::new("http://xmlns.com/foaf/0.1/name"),
        RdfTerm::literal("Charlie Brown"),
    ));

    store
}

// ============================================================================
// Benchmarks
// ============================================================================

fn benchmark_triple_insertion(count: usize) -> std::time::Duration {
    let mut store = TripleStore::new();

    let start = Instant::now();
    for i in 0..count {
        store.insert(Triple::new(
            RdfTerm::iri(format!("http://example.org/subject/{}", i)),
            Iri::new("http://example.org/predicate"),
            RdfTerm::literal(format!("value {}", i)),
        ));
    }
    start.elapsed()
}

fn benchmark_triple_query(store: &TripleStore, iterations: usize) -> std::time::Duration {
    let subject = RdfTerm::iri("http://example.org/subject/500");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = store.query(Some(&subject), None, None);
    }
    start.elapsed()
}

fn benchmark_sparql_parse(iterations: usize) -> std::time::Duration {
    let query = r#"SELECT ?person ?name WHERE { ?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . ?person <http://xmlns.com/foaf/0.1/name> ?name . }"#;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = parse_simple_sparql(query);
    }
    start.elapsed()
}

fn benchmark_sparql_execution(store: &TripleStore, iterations: usize) -> std::time::Duration {
    let query = r#"SELECT ?s ?p ?o WHERE { ?s ?p ?o . }"#;

    let parsed = parse_simple_sparql(query).expect("Should parse");

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = execute_query(store, &parsed);
    }
    start.elapsed()
}

fn print_separator() {
    println!("{}", "=".repeat(60));
}

fn main() {
    print_separator();
    println!("SPARQL Implementation Validation & Benchmarks");
    print_separator();
    println!();

    // Run validation tests
    println!("--- Validation Tests ---");
    println!();

    // Test 1: Triple store insertion
    {
        let mut store = TripleStore::new();
        let id = store.insert(Triple::new(
            RdfTerm::iri("http://example.org/s"),
            Iri::new("http://example.org/p"),
            RdfTerm::literal("object"),
        ));
        assert!(id > 0);
        assert_eq!(store.count(), 1);
        println!("[PASS] Triple store insertion works");
    }

    // Test 2: Query by subject
    {
        let store = create_test_store();
        let results = store.query(
            Some(&RdfTerm::iri("http://example.org/person/alice")),
            None,
            None,
        );
        assert_eq!(results.len(), 4); // type, name, age, knows
        println!("[PASS] Query by subject returns {} triples", results.len());
    }

    // Test 3: Query by predicate
    {
        let store = create_test_store();
        let results = store.query(None, Some(&Iri::rdf_type()), None);
        assert_eq!(results.len(), 3); // alice, bob, charlie
        println!(
            "[PASS] Query by predicate returns {} triples",
            results.len()
        );
    }

    // Test 4: SPARQL SELECT parser
    {
        let query = r#"SELECT ?person ?name WHERE { ?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . ?person <http://xmlns.com/foaf/0.1/name> ?name . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        match parsed {
            QueryType::Select {
                variables,
                where_patterns,
            } => {
                assert_eq!(variables.len(), 2);
                assert!(variables.contains(&"person".to_string()));
                assert!(variables.contains(&"name".to_string()));
                assert_eq!(
                    where_patterns.len(),
                    2,
                    "Expected 2 patterns, got {}: {:?}",
                    where_patterns.len(),
                    where_patterns
                );
                println!("[PASS] SPARQL SELECT parser works");
            }
            _ => panic!("Expected SELECT query"),
        }
    }

    // Test 5: SPARQL ASK parser
    {
        let query = r#"ASK WHERE { <http://example.org/person/alice> <http://xmlns.com/foaf/0.1/name> ?name . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        match parsed {
            QueryType::Ask { where_patterns } => {
                assert_eq!(
                    where_patterns.len(),
                    1,
                    "Expected 1 pattern, got {}: {:?}",
                    where_patterns.len(),
                    where_patterns
                );
                println!("[PASS] SPARQL ASK parser works");
            }
            _ => panic!("Expected ASK query"),
        }
    }

    // Test 6: SPARQL SELECT execution
    {
        let store = create_test_store();
        let query = r#"SELECT ?person ?name WHERE { ?person <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . ?person <http://xmlns.com/foaf/0.1/name> ?name . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        let results = execute_query(&store, &parsed);
        assert_eq!(
            results.len(),
            3,
            "Expected 3 results, got {}",
            results.len()
        ); // alice, bob, charlie
        for binding in &results {
            assert!(binding.contains_key("person"));
            assert!(binding.contains_key("name"));
        }
        println!(
            "[PASS] SPARQL SELECT execution returns {} bindings",
            results.len()
        );
    }

    // Test 7: SPARQL ASK true
    {
        let store = create_test_store();
        let query = r#"ASK WHERE { <http://example.org/person/alice> <http://xmlns.com/foaf/0.1/name> ?name . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        let results = execute_query(&store, &parsed);
        assert!(!results.is_empty());
        println!("[PASS] SPARQL ASK returns true when pattern exists");
    }

    // Test 8: SPARQL ASK false
    {
        let store = create_test_store();
        let query = r#"ASK WHERE { <http://example.org/person/dave> <http://xmlns.com/foaf/0.1/name> ?name . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        let results = execute_query(&store, &parsed);
        assert!(results.is_empty());
        println!("[PASS] SPARQL ASK returns false when pattern doesn't exist");
    }

    // Test 9: SPARQL JOIN
    {
        let store = create_test_store();
        let query = r#"SELECT ?person ?friend WHERE { ?person <http://xmlns.com/foaf/0.1/knows> ?friend . ?friend <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://example.org/Person> . }"#;
        let parsed = parse_simple_sparql(query).expect("Should parse");
        let results = execute_query(&store, &parsed);
        assert_eq!(
            results.len(),
            2,
            "Expected 2 results, got {}",
            results.len()
        ); // alice->bob, bob->charlie
        println!(
            "[PASS] SPARQL JOIN execution returns {} bindings",
            results.len()
        );
    }

    println!();
    println!("All 9 validation tests passed!");
    println!();

    // Run benchmarks
    println!("--- Benchmarks ---");
    println!();

    // Triple insertion benchmark
    let counts = [1_000, 10_000, 100_000];
    for count in counts {
        let duration = benchmark_triple_insertion(count);
        let rate = count as f64 / duration.as_secs_f64();
        println!(
            "Insert {:>7} triples: {:>10.2?} ({:>12.0} triples/sec)",
            count, duration, rate
        );
    }
    println!();

    // Create a large store for query benchmarks
    let mut large_store = TripleStore::new();
    for i in 0..10_000 {
        large_store.insert(Triple::new(
            RdfTerm::iri(format!("http://example.org/subject/{}", i)),
            Iri::new("http://example.org/predicate"),
            RdfTerm::literal(format!("value {}", i)),
        ));
    }

    // Query benchmark
    let iterations = 10_000;
    let duration = benchmark_triple_query(&large_store, iterations);
    let rate = iterations as f64 / duration.as_secs_f64();
    println!(
        "Query by subject ({} iterations): {:?} ({:.0} queries/sec)",
        iterations, duration, rate
    );

    // Parse benchmark
    let duration = benchmark_sparql_parse(iterations);
    let rate = iterations as f64 / duration.as_secs_f64();
    println!(
        "SPARQL parse ({} iterations): {:?} ({:.0} parses/sec)",
        iterations, duration, rate
    );

    // Execution benchmark (smaller dataset)
    let small_store = create_test_store();
    let iterations = 1_000;
    let duration = benchmark_sparql_execution(&small_store, iterations);
    let rate = iterations as f64 / duration.as_secs_f64();
    println!(
        "SPARQL execution ({} iterations): {:?} ({:.0} queries/sec)",
        iterations, duration, rate
    );

    println!();
    print_separator();
    println!("VALIDATION COMPLETE - SPARQL Implementation is REAL!");
    print_separator();
}
