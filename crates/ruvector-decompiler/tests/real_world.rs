//! Real-world-style test fixtures for the decompiler.
//!
//! Each fixture has a known original source and a hand-minified version.
//! We run the decompiler on the minified version, compare against ground
//! truth, and feed results into the self-learning feedback loop.

use ruvector_decompiler::inferrer::{InferenceFeedback, learn_from_ground_truth};
use ruvector_decompiler::{decompile, DecompileConfig, InferredName};

// ---------------------------------------------------------------------------
// Fixture A: Lodash-like utility library (~100 lines minified)
// ---------------------------------------------------------------------------

/// Original names for the lodash-like utility fixture.
const LODASH_ORIGINAL_NAMES: &[(&str, &str)] = &[
    ("a", "chunk"),
    ("b", "debounce"),
    ("c", "throttle"),
    ("d", "flatten"),
    ("e", "uniq"),
    ("f", "groupBy"),
    ("g", "memoize"),
    ("h", "deepClone"),
    ("i", "merge"),
    ("j", "pick"),
];

const LODASH_MINIFIED: &str = concat!(
    r#"var a=function(k,l){var m=[];for(var n=0;n<k.length;n+=l){m.push(k.slice(n,n+l))}return m};"#,
    r#"var b=function(o,p){var q=null;return function(){clearTimeout(q);q=setTimeout(function(){o.apply(null,arguments)},p)}};"#,
    r#"var c=function(r,s){var t=0;return function(){var u=Date.now();if(u-t>=s){t=u;return r.apply(null,arguments)}}};"#,
    r#"var d=function(v){var w=[];for(var x=0;x<v.length;x++){if(Array.isArray(v[x])){w=w.concat(d(v[x]))}else{w.push(v[x])}}return w};"#,
    r#"var e=function(y){var z={};var aa=[];for(var ab=0;ab<y.length;ab++){if(!z[y[ab]]){z[y[ab]]=true;aa.push(y[ab])}}return aa};"#,
    r#"var f=function(ac,ad){var ae={};for(var af=0;af<ac.length;af++){var ag=ad(ac[af]);if(!ae[ag]){ae[ag]=[]}ae[ag].push(ac[af])}return ae};"#,
    r#"var g=function(ah){var ai={};return function(aj){var ak=JSON.stringify(aj);if(ai[ak]!==undefined){return ai[ak]}var al=ah(aj);ai[ak]=al;return al}};"#,
    r#"var h=function(am){if(typeof am!=="object"||am===null){return am}var an=Array.isArray(am)?[]:{};for(var ao in am){an[ao]=h(am[ao])}return an};"#,
    r#"var i=function(ap,aq){var ar=h(ap);for(var as in aq){if(typeof aq[as]==="object"&&typeof ar[as]==="object"){ar[as]=i(ar[as],aq[as])}else{ar[as]=aq[as]}}return ar};"#,
    r#"var j=function(at,au){var av={};for(var aw=0;aw<au.length;aw++){if(at[au[aw]]!==undefined){av[au[aw]]=at[au[aw]]}}return av};"#,
);

// ---------------------------------------------------------------------------
// Fixture B: Express-like HTTP router (~150 lines minified)
// ---------------------------------------------------------------------------

const ROUTER_ORIGINAL_NAMES: &[(&str, &str)] = &[
    ("a", "createRouter"),
    ("b", "parseUrl"),
    ("c", "matchRoute"),
    ("d", "createResponse"),
    ("e", "bodyParser"),
    ("f", "corsMiddleware"),
    ("g", "loggerMiddleware"),
    ("h", "errorHandler"),
    ("i", "createApp"),
    ("j", "listen"),
    ("k", "addRoute"),
    ("l", "useMiddleware"),
];

const ROUTER_MINIFIED: &str = concat!(
    r#"var a=function(){var m=[];var n=[];return{routes:m,middleware:n,add:function(o,p,q){m.push({method:o,path:p,handler:q})},use:function(r){n.push(r)}}};"#,
    r#"var b=function(s){var t=s.indexOf("?");var u=t>=0?s.substring(0,t):s;var v=t>=0?s.substring(t+1):"";var w={};if(v){v.split("&").forEach(function(x){var y=x.split("=");w[y[0]]=decodeURIComponent(y[1]||"")})}return{path:u,query:w}};"#,
    r#"var c=function(z,aa,ab){for(var ac=0;ac<z.length;ac++){if(z[ac].method===aa&&z[ac].path===ab){return z[ac]}}return null};"#,
    r#"var d=function(){return{status:200,headers:{"Content-Type":"application/json"},body:"",json:function(ad){this.body=JSON.stringify(ad);return this},send:function(ae){this.body=ae;return this},setHeader:function(af,ag){this.headers[af]=ag;return this}}};"#,
    r#"var e=function(ah){try{return JSON.parse(ah)}catch(ai){return null}};"#,
    r#"var f=function(aj,ak,al){ak.setHeader("Access-Control-Allow-Origin","*");ak.setHeader("Access-Control-Allow-Methods","GET,POST,PUT,DELETE");al()};"#,
    r#"var g=function(am,an,ao){var ap=Date.now();ao();var aq=Date.now()-ap;console.log(am.method+" "+am.url+" "+aq+"ms")};"#,
    r#"var h=function(ar,as,at){try{at()}catch(au){console.error("Error:",au.message);as.status=500;as.json({error:au.message})}};"#,
    r#"var i=function(){var av=a();av.use(g);av.use(f);return{router:av,get:function(aw,ax){av.add("GET",aw,ax)},post:function(ay,az){av.add("POST",ay,az)},handle:function(ba){var bb=b(ba.url);var bc=c(av.routes,ba.method,bb.path);var bd=d();if(!bc){bd.status=404;bd.json({error:"Not found"});return bd}for(var be=0;be<av.middleware.length;be++){av.middleware[be](ba,bd,function(){})}bc.handler(ba,bd);return bd}}};"#,
    r#"var j=function(bf,bg){console.log("Server listening on port "+bg)};"#,
    r#"var k=function(bh,bi,bj,bk){bh.add(bi,bj,bk)};"#,
    r#"var l=function(bl,bm){bl.use(bm)};"#,
);

// ---------------------------------------------------------------------------
// Fixture C: Redux-like state management (~100 lines minified)
// ---------------------------------------------------------------------------

const REDUX_ORIGINAL_NAMES: &[(&str, &str)] = &[
    ("a", "createStore"),
    ("b", "combineReducers"),
    ("c", "applyMiddleware"),
    ("d", "bindActionCreators"),
    ("e", "createAction"),
    ("f", "thunkMiddleware"),
    ("g", "loggerMiddleware"),
    ("h", "subscribe"),
];

const REDUX_MINIFIED: &str = concat!(
    r#"var a=function(i,j){var k=j;var l=[];var m=function(){return k};var n=function(o){l.push(o);return function(){l=l.filter(function(p){return p!==o})}};var q=function(r){k=i(k,r);l.forEach(function(s){s(k)})};q({type:"@@INIT"});return{getState:m,subscribe:n,dispatch:q}};"#,
    r#"var b=function(t){return function(u,v){var w={};for(var x in t){w[x]=t[x](u[x],v)}return w}};"#,
    r#"var c=function(){var y=Array.prototype.slice.call(arguments);return function(z){return function(aa,ab){var ac=z(aa,ab);var ad=ac.dispatch;y.forEach(function(ae){ad=ae(ac)(ad)});return Object.assign({},ac,{dispatch:ad})}}};"#,
    r#"var d=function(af,ag){var ah={};for(var ai in af){ah[ai]=function(aj){return function(){ag(af[aj].apply(null,arguments))}}(ai)}return ah};"#,
    r#"var e=function(ak){return function(al){return{type:ak,payload:al}}};"#,
    r#"var f=function(am){return function(an){return function(ao){if(typeof ao==="function"){return ao(am.dispatch,am.getState)}return an(ao)}}};"#,
    r#"var g=function(ap){return function(aq){return function(ar){console.log("prev state:",ap.getState());console.log("action:",ar);var as=aq(ar);console.log("next state:",ap.getState());return as}}};"#,
    r#"var h=function(at,au){return at.subscribe(au)};"#,
);

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

struct FixtureResult {
    name: &'static str,
    decl_found: usize,
    decl_expected: usize,
    name_hits: usize,
    name_total: usize,
    module_count: usize,
    module_expected: usize,
    avg_confidence: f64,
    high_conf: usize,
    medium_conf: usize,
    low_conf: usize,
    feedback: Vec<InferenceFeedback>,
}

fn run_fixture(
    name: &'static str,
    minified: &str,
    original_names: &[(&str, &str)],
    target_modules: usize,
) -> FixtureResult {
    let config = DecompileConfig {
        target_modules: Some(target_modules),
        min_confidence: 0.0,
        generate_source_maps: true,
        generate_witness: true,
        ..DecompileConfig::default()
    };

    let result = decompile(minified, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();

    // Count name inference hits (semantic matching).
    let name_hits = count_semantic_hits(&result.inferred_names, original_names);

    // Confidence breakdown.
    let high_conf = result.inferred_names.iter().filter(|n| n.confidence > 0.9).count();
    let medium_conf = result
        .inferred_names
        .iter()
        .filter(|n| n.confidence >= 0.6 && n.confidence <= 0.9)
        .count();
    let low_conf = result.inferred_names.iter().filter(|n| n.confidence < 0.6).count();

    let avg_confidence = if !result.inferred_names.is_empty() {
        result.inferred_names.iter().map(|n| n.confidence).sum::<f64>()
            / result.inferred_names.len() as f64
    } else {
        0.0
    };

    // Build feedback for learning loop.
    let feedback = build_feedback(&result.inferred_names, original_names);

    FixtureResult {
        name,
        decl_found: total_decls,
        decl_expected: original_names.len(),
        name_hits,
        name_total: original_names.len(),
        module_count: result.modules.len(),
        module_expected: target_modules,
        avg_confidence,
        high_conf,
        medium_conf,
        low_conf,
        feedback,
    }
}

fn count_semantic_hits(
    inferred: &[InferredName],
    originals: &[(&str, &str)],
) -> usize {
    let mut hits = 0;
    for inf in inferred {
        let inf_lower = inf.inferred.to_lowercase();
        for &(minified_name, original_name) in originals {
            if inf.original != minified_name {
                continue;
            }
            let orig_lower = original_name.to_lowercase();
            // Fuzzy match: either contains the other, or shares a keyword >= 3 chars.
            if inf_lower.contains(&orig_lower)
                || orig_lower.contains(&inf_lower)
                || keyword_overlap(&inf_lower, &orig_lower)
            {
                hits += 1;
                break;
            }
        }
    }
    hits
}

fn keyword_overlap(a: &str, b: &str) -> bool {
    for part_a in a.split('_') {
        if part_a.len() >= 3 {
            for part_b in b.split('_') {
                if part_b.len() >= 3 && part_a == part_b {
                    return true;
                }
            }
        }
    }
    false
}

fn build_feedback(
    inferred: &[InferredName],
    originals: &[(&str, &str)],
) -> Vec<InferenceFeedback> {
    let mut feedback = Vec::new();
    for &(minified_name, correct_name) in originals {
        if let Some(inf) = inferred.iter().find(|n| n.original == minified_name) {
            let inf_lower = inf.inferred.to_lowercase();
            let correct_lower = correct_name.to_lowercase();
            let was_correct = inf_lower.contains(&correct_lower)
                || correct_lower.contains(&inf_lower)
                || keyword_overlap(&inf_lower, &correct_lower);

            feedback.push(InferenceFeedback {
                original: minified_name.to_string(),
                inferred: inf.inferred.clone(),
                correct: correct_name.to_string(),
                was_correct,
                evidence: inf.evidence.clone(),
            });
        }
    }
    feedback
}

fn print_result(r: &FixtureResult) {
    let decl_pct = if r.decl_expected > 0 {
        (r.decl_found as f64 / r.decl_expected as f64 * 100.0).min(100.0)
    } else {
        0.0
    };
    let name_pct = if r.name_total > 0 {
        r.name_hits as f64 / r.name_total as f64 * 100.0
    } else {
        0.0
    };

    println!("  {}: decls {}/{} ({:.0}%), names {}/{} ({:.0}%), modules {}/{}",
        r.name, r.decl_found, r.decl_expected, decl_pct,
        r.name_hits, r.name_total, name_pct,
        r.module_count, r.module_expected,
    );
    println!("    confidence: HIGH={}, MEDIUM={}, LOW={}, avg={:.2}",
        r.high_conf, r.medium_conf, r.low_conf, r.avg_confidence,
    );
}

#[test]
fn test_all_fixtures_with_learning() {
    println!("\n=== Before Learning ===");

    let lodash = run_fixture("lodash-utils", LODASH_MINIFIED, LODASH_ORIGINAL_NAMES, 2);
    let router = run_fixture("express-router", ROUTER_MINIFIED, ROUTER_ORIGINAL_NAMES, 3);
    let redux = run_fixture("redux-store", REDUX_MINIFIED, REDUX_ORIGINAL_NAMES, 2);

    print_result(&lodash);
    print_result(&router);
    print_result(&redux);

    // Feed all results into the learning loop.
    let all_feedback: Vec<InferenceFeedback> = lodash
        .feedback
        .iter()
        .chain(router.feedback.iter())
        .chain(redux.feedback.iter())
        .cloned()
        .collect();

    let (successes, failures) = learn_from_ground_truth(&all_feedback);

    println!("\n=== Learning Results ===");
    println!("  Successful patterns: {}", successes.len());
    println!("  Failed patterns: {}", failures.len());

    for s in &successes {
        println!(
            "    OK: {} -> {} (correct: {})",
            s.minified_name, s.inferred_name, s.correct_name
        );
    }
    for f in &failures {
        println!(
            "    MISS: {} -> {} (should be: {})",
            f.minified_name, f.inferred_name, f.correct_name
        );
    }

    // Basic assertions -- each fixture should find at least most declarations.
    assert!(
        lodash.decl_found >= 8,
        "lodash: expected at least 8 declarations, found {}",
        lodash.decl_found
    );
    assert!(
        router.decl_found >= 10,
        "router: expected at least 10 declarations, found {}",
        router.decl_found
    );
    assert!(
        redux.decl_found >= 6,
        "redux: expected at least 6 declarations, found {}",
        redux.decl_found
    );
}

#[test]
fn test_lodash_fixture_individual() {
    let r = run_fixture("lodash-utils", LODASH_MINIFIED, LODASH_ORIGINAL_NAMES, 2);
    print_result(&r);
    assert!(r.decl_found >= 8);
}

#[test]
fn test_router_fixture_individual() {
    let r = run_fixture("express-router", ROUTER_MINIFIED, ROUTER_ORIGINAL_NAMES, 3);
    print_result(&r);
    assert!(r.decl_found >= 10);
}

#[test]
fn test_redux_fixture_individual() {
    let r = run_fixture("redux-store", REDUX_MINIFIED, REDUX_ORIGINAL_NAMES, 2);
    print_result(&r);
    assert!(r.decl_found >= 6);
}
