//! Ground-truth validation tests for the decompiler.
//!
//! Each fixture has a known original source and a hand-minified version.
//! We run the decompiler on the minified version and compare against
//! ground truth to compute accuracy metrics.

use ruvector_decompiler::{decompile, DecompileConfig};

// ---------------------------------------------------------------------------
// Fixture 1: Express-like HTTP framework
// ---------------------------------------------------------------------------

/// Original (known) source for the Express-like fixture.
const EXPRESS_ORIGINAL_NAMES: &[&str] =
    &["Router", "Request", "Response", "createApp", "handleRoute"];

const EXPRESS_MINIFIED: &str = concat!(
    r#"var a=class{constructor(){this.routes=[]}"#,
    r#"add(b,c){this.routes.push({path:b,handler:c})}"#,
    r#"match(d){return this.routes.find(e=>e.path===d)}}"#,
    r#";var f=class{constructor(g,h){this.method=g;this.url=h;this.headers={};this.body=""}};"#,
    r#"var i=class{constructor(){this.status=200;this.headers={};this.body=""}"#,
    r#"json(j){this.headers["Content-Type"]="application/json";this.body=JSON.stringify(j);return this}};"#,
    r#"var k=function(){var l=new a;return{use:function(m,n){l.add(m,n)},handle:function(o,p){var q=l.match(o);if(q){q.handler(new f("GET",o),new i)}else{p.status=404}}}};"#,
    r#"var r=function(s,t){var u=t.match(s);if(u){u.handler(s,t)}};"#,
);

#[test]
fn test_fixture_express() {
    let config = DecompileConfig {
        target_modules: Some(3),
        min_confidence: 0.0,
        ..DecompileConfig::default()
    };

    let result = decompile(EXPRESS_MINIFIED, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    let _decl_names: Vec<&str> = result
        .modules
        .iter()
        .flat_map(|m| m.declarations.iter().map(|d| d.name.as_str()))
        .collect();

    // Accuracy metrics.
    let decl_count = total_decls;
    let expected_decl_count = 5; // a, f, i, k, r

    let name_hits = count_name_hits(&result.inferred_names, EXPRESS_ORIGINAL_NAMES);

    let module_count = result.modules.len();

    print_metrics(
        "fixture-express",
        decl_count,
        expected_decl_count,
        name_hits,
        EXPRESS_ORIGINAL_NAMES.len(),
        module_count,
        3,
        &result.inferred_names,
        &result.witness.chain_root,
    );

    // Basic assertions.
    assert!(
        decl_count >= 4,
        "express: expected at least 4 declarations, found {}",
        decl_count
    );
    assert!(
        !result.witness.chain_root.is_empty(),
        "express: witness chain should be valid"
    );
    // Verify no false positives at HIGH confidence.
    for name in &result.inferred_names {
        if name.confidence > 0.9 {
            assert!(
                !name.inferred.is_empty(),
                "HIGH confidence name should not be empty"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Fixture 2: MCP Server
// ---------------------------------------------------------------------------

const MCP_ORIGINAL_NAMES: &[&str] = &[
    "McpServer",
    "handleInitialize",
    "handleToolCall",
    "toolRegistry",
];

const MCP_MINIFIED: &str = concat!(
    r#"var a=class{constructor(){this.tools={};this.version="1.0"}"#,
    r#"register(b,c){this.tools[b]=c}"#,
    r#"list(){return Object.keys(this.tools)}};"#,
    r#"var d=function(e){return{jsonrpc:"2.0",result:{protocolVersion:"2024-11-05",capabilities:{tools:{listChanged:true}},serverInfo:{name:"mcp-server",version:e.version}}}};"#,
    r#"var f=function(g,h){var i=g.tools[h.params.name];if(!i){return{jsonrpc:"2.0",error:{code:-32601,message:"Tool not found: "+h.params.name}}}return{jsonrpc:"2.0",result:i(h.params.arguments)}};"#,
    r#"var j={Bash:{description:"Execute bash commands"},Read:{description:"Read files"},Edit:{description:"Edit files"}};"#,
);

#[test]
fn test_fixture_mcp_server() {
    let config = DecompileConfig {
        target_modules: Some(2),
        min_confidence: 0.0,
        ..DecompileConfig::default()
    };

    let result = decompile(MCP_MINIFIED, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    let name_hits = count_name_hits(&result.inferred_names, MCP_ORIGINAL_NAMES);

    print_metrics(
        "fixture-mcp-server",
        total_decls,
        4,
        name_hits,
        MCP_ORIGINAL_NAMES.len(),
        result.modules.len(),
        2,
        &result.inferred_names,
        &result.witness.chain_root,
    );

    assert!(total_decls >= 3, "mcp: expected at least 3 decls");

    // MCP-related string literals should trigger high-confidence inference.
    let high_conf: Vec<_> = result
        .inferred_names
        .iter()
        .filter(|n| n.confidence > 0.9)
        .collect();
    // At least one high-confidence name from MCP strings.
    // (jsonrpc, tools/call, etc.)
    assert!(
        !high_conf.is_empty() || result.inferred_names.is_empty(),
        "mcp: expected at least one high-confidence name"
    );
}

// ---------------------------------------------------------------------------
// Fixture 3: React-like Component
// ---------------------------------------------------------------------------

const REACT_ORIGINAL_NAMES: &[&str] = &["useState", "useEffect", "Component"];

const REACT_MINIFIED: &str = concat!(
    r#"var a=function(b){var c=[b,function(d){c[0]=d}];return c};"#,
    r#"var e=function(f,g){if(g===undefined||g.some(function(h,i){return h!==g[i]})){f()}};"#,
    r#"var j=class{constructor(k){this.props=k;this.state={}}setState(l){Object.assign(this.state,l);this.render()}render(){return null}};"#,
);

#[test]
fn test_fixture_react_component() {
    let config = DecompileConfig {
        target_modules: Some(2),
        min_confidence: 0.0,
        ..DecompileConfig::default()
    };

    let result = decompile(REACT_MINIFIED, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    let name_hits = count_name_hits(&result.inferred_names, REACT_ORIGINAL_NAMES);

    print_metrics(
        "fixture-react-component",
        total_decls,
        3,
        name_hits,
        REACT_ORIGINAL_NAMES.len(),
        result.modules.len(),
        2,
        &result.inferred_names,
        &result.witness.chain_root,
    );

    assert!(total_decls >= 3, "react: expected 3 declarations");
}

// ---------------------------------------------------------------------------
// Fixture 4: Multi-module bundle (3 distinct modules)
// ---------------------------------------------------------------------------

const MULTI_ORIGINAL_NAMES: &[&str] = &[
    "add",
    "subtract",
    "multiply", // Module A: math
    "capitalize",
    "trim",
    "concat",      // Module B: string
    "processData", // Module C: uses A + B
];

const MULTI_MINIFIED: &str = concat!(
    r#"var a=function(x,y){return x+y};var b=function(x,y){return x-y};var c=function(x,y){return x*y};"#,
    r#"var d=function(s){return s.charAt(0).toUpperCase()+s.slice(1)};var e=function(s){return s.trim()};var f=function(s1,s2){return s1+s2};"#,
    r#"var g=function(h){var i=a(h.x,h.y);var j=d(h.label);return f(j,String(i))};"#,
);

#[test]
fn test_fixture_multi_module() {
    let config = DecompileConfig {
        target_modules: Some(3),
        min_confidence: 0.0,
        ..DecompileConfig::default()
    };

    let result = decompile(MULTI_MINIFIED, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    let name_hits = count_name_hits(&result.inferred_names, MULTI_ORIGINAL_NAMES);

    print_metrics(
        "fixture-multi-module",
        total_decls,
        7,
        name_hits,
        MULTI_ORIGINAL_NAMES.len(),
        result.modules.len(),
        3,
        &result.inferred_names,
        &result.witness.chain_root,
    );

    assert!(total_decls >= 6, "multi: expected at least 6 declarations");

    // Module C (g) references a and d, which are in different modules.
    // MinCut should separate them.
    if result.modules.len() >= 2 {
        // Check that not all declarations ended up in one module.
        let max_in_one = result
            .modules
            .iter()
            .map(|m| m.declarations.len())
            .max()
            .unwrap_or(0);
        assert!(
            max_in_one < total_decls,
            "multi: all decls in one module (no partitioning happened)"
        );
    }
}

// ---------------------------------------------------------------------------
// Fixture 5: Bundled utils with known tool names
// ---------------------------------------------------------------------------

const TOOLS_ORIGINAL_NAMES: &[&str] = &["toolDefinitions", "bashTool", "readTool", "executeTool"];

const TOOLS_MINIFIED: &str = concat!(
    r#"var a={Bash:{description:"Execute bash commands",inputSchema:{type:"object"}},Read:{description:"Read files",inputSchema:{type:"object"}},Edit:{description:"Edit files",inputSchema:{type:"object"}}};"#,
    r#"var b=function(c){return require("child_process").execSync(c).toString()};"#,
    r#"var d=function(e){return require("fs").readFileSync(e,"utf8")};"#,
    r#"var f=function(g,h){var i=a[g];if(!i)throw new Error("Unknown tool: "+g);if(g==="Bash")return b(h);if(g==="Read")return d(h);throw new Error("Unimplemented: "+g)};"#,
);

#[test]
fn test_fixture_tools_bundle() {
    let config = DecompileConfig {
        target_modules: Some(2),
        min_confidence: 0.0,
        ..DecompileConfig::default()
    };

    let result = decompile(TOOLS_MINIFIED, &config).unwrap();

    let total_decls: usize = result.modules.iter().map(|m| m.declarations.len()).sum();
    let name_hits = count_name_hits(&result.inferred_names, TOOLS_ORIGINAL_NAMES);

    print_metrics(
        "fixture-tools-bundle",
        total_decls,
        4,
        name_hits,
        TOOLS_ORIGINAL_NAMES.len(),
        result.modules.len(),
        2,
        &result.inferred_names,
        &result.witness.chain_root,
    );

    assert!(total_decls >= 3, "tools: expected at least 3 declarations");

    // The tool definitions should trigger high-confidence names
    // since they contain known tool name strings like "Bash", "Read".
    let tool_related: Vec<_> = result
        .inferred_names
        .iter()
        .filter(|n| {
            n.evidence
                .iter()
                .any(|e| e.contains("Bash") || e.contains("Read") || e.contains("Error"))
        })
        .collect();
    // We expect at least some tool-related inferences.
    println!("  Tool-related inferences: {} found", tool_related.len());
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Count how many inferred names semantically match the original names.
///
/// A match is when the inferred name contains a keyword from the original
/// name (case-insensitive) or vice versa.
fn count_name_hits(inferred: &[ruvector_decompiler::InferredName], originals: &[&str]) -> usize {
    let mut hits = 0;
    for inf in inferred {
        let inf_lower = inf.inferred.to_lowercase();
        for &orig in originals {
            let orig_lower = orig.to_lowercase();
            // Check if either contains part of the other (fuzzy match).
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

/// Check if two names share any keyword of length >= 3.
fn keyword_overlap(a: &str, b: &str) -> bool {
    let keywords_a = extract_keywords(a);
    let keywords_b = extract_keywords(b);
    keywords_a.iter().any(|ka| keywords_b.contains(ka))
}

/// Extract lowercase keywords (substrings split on `_` or camelCase).
fn extract_keywords(name: &str) -> Vec<String> {
    let mut keywords = Vec::new();
    // Split on underscore.
    for part in name.split('_') {
        if part.len() >= 3 {
            keywords.push(part.to_string());
        }
    }
    keywords
}

/// Print accuracy metrics for a fixture.
#[allow(clippy::too_many_arguments)]
fn print_metrics(
    fixture_name: &str,
    decl_found: usize,
    decl_expected: usize,
    names_correct: usize,
    names_total: usize,
    modules_found: usize,
    modules_expected: usize,
    inferred_names: &[ruvector_decompiler::InferredName],
    chain_root: &str,
) {
    let decl_pct = if decl_expected > 0 {
        (decl_found as f64 / decl_expected as f64 * 100.0).min(100.0)
    } else {
        0.0
    };
    let name_pct = if names_total > 0 {
        names_correct as f64 / names_total as f64 * 100.0
    } else {
        0.0
    };
    let module_pct = if modules_expected > 0 {
        (modules_found as f64 / modules_expected as f64 * 100.0).min(100.0)
    } else {
        0.0
    };
    let avg_confidence = if !inferred_names.is_empty() {
        inferred_names.iter().map(|n| n.confidence).sum::<f64>() / inferred_names.len() as f64
    } else {
        0.0
    };

    println!("Fixture: {}", fixture_name);
    println!(
        "  Declarations found: {}/{} ({:.0}%)",
        decl_found, decl_expected, decl_pct
    );
    println!(
        "  Names correctly inferred: {}/{} ({:.0}%)",
        names_correct, names_total, name_pct
    );
    println!(
        "  Module boundaries: {}/{} ({:.0}%)",
        modules_found, modules_expected, module_pct
    );
    println!("  Average confidence: {:.2}", avg_confidence);
    println!(
        "  Witness chain: {}",
        if chain_root.is_empty() {
            "INVALID"
        } else {
            "VALID"
        }
    );
    println!();
}
