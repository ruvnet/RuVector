//! Animated README banner, following the repo convention set by
//! `assets/ruvector/packages/*.svg`: 1000x240, `#0c1324` panel, `#5cf2c3`
//! accent, `<title>`/`<desc>` + `aria-labelledby`, CSS `flow`/`lit`
//! keyframes and a `prefers-reduced-motion` opt-out. The right-hand panel
//! draws the package as a hub woven to its related packages.

pub fn esc(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// Greedy word wrap.
pub fn wrap(s: &str, width: usize, max_lines: usize) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    let mut cur = String::new();
    for w in s.split_whitespace() {
        if !cur.is_empty() && cur.len() + 1 + w.len() > width {
            lines.push(std::mem::take(&mut cur));
        }
        if !cur.is_empty() {
            cur.push(' ');
        }
        cur.push_str(w);
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    if lines.len() > max_lines {
        lines.truncate(max_lines);
        if let Some(l) = lines.last_mut() {
            l.push('…');
        }
    }
    lines
}

pub fn banner(name: &str, tagline: &str, category: &str, related: &[String]) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="240" viewBox="0 0 1000 240" role="img" aria-labelledby="title desc"><title id="title">{} — {}</title><desc id="desc">Animated RuVector package banner. The package sits at the hub of a woven web of related ruvnet packages; links light up in turn.</desc>"#,
        esc(name),
        esc(tagline)
    ));
    s.push_str(r#"<style>text{font-family:Arial,sans-serif}.wire{fill:none;stroke:#5cf2c3;stroke-width:2;stroke-dasharray:8 7;animation:flow 3s linear infinite}.lit{animation:lit 6s ease-in-out infinite}@keyframes flow{to{stroke-dashoffset:-60}}@keyframes lit{0%,100%{opacity:.3}12%,40%{opacity:1}}@media(prefers-reduced-motion:reduce){.wire,.lit{animation:none}}</style>"#);
    s.push_str(r##"<rect width="1000" height="240" rx="18" fill="#0c1324"/><path d="M0 42H1000M0 226H1000" stroke="#223047"/>"##);
    s.push_str(&format!(
        r##"<text x="30" y="29" fill="#5cf2c3" font-size="13" letter-spacing="3">RUVECTOR / FABRIC / {}</text><text x="30" y="92" fill="#f2f6ff" font-size="29" font-weight="bold">{}</text>"##,
        esc(&category.to_ascii_uppercase()),
        esc(name)
    ));
    for (i, l) in wrap(tagline, 62, 2).iter().enumerate() {
        s.push_str(&format!(
            r##"<text x="30" y="{}" fill="#abbad0" font-size="16">{}</text>"##,
            126 + 22 * i,
            esc(l)
        ));
    }
    s.push_str(&format!(
        r##"<text x="30" y="200" fill="#8297b1" font-size="14">npm install {}</text>"##,
        esc(name)
    ));
    s.push_str(r##"<rect x="600" y="58" width="370" height="152" rx="14" fill="#111e32" stroke="#223047"/>"##);
    let (cx, cy) = (785.0f64, 134.0f64);
    let nodes: Vec<&String> = related.iter().take(6).collect();
    let n = nodes.len().max(1) as f64;
    for (i, r) in nodes.iter().enumerate() {
        let a = std::f64::consts::PI * (0.15 + 1.7 * i as f64 / n.max(2.0));
        let (x, y) = (
            cx + 150.0 * a.cos(),
            cy - 52.0 * a.sin().abs() * if i % 2 == 0 { 1.0 } else { -1.0 },
        );
        let delay = 0.8 * i as f64;
        s.push_str(&format!(r##"<path d="M{cx:.0} {cy:.0}L{x:.0} {y:.0}" stroke="#374354"/><path class="wire" d="M{cx:.0} {cy:.0}L{x:.0} {y:.0}" style="animation-delay:{delay}s"/>"##));
        s.push_str(&format!(r##"<circle cx="{x:.0}" cy="{y:.0}" r="6" fill="#5cf2c3" class="lit" style="animation-delay:{delay}s"/>"##));
        let short = r.rsplit('/').next().unwrap_or(r);
        let anchor = if x < cx { "end" } else { "start" };
        let tx = if x < cx { x - 10.0 } else { x + 10.0 };
        s.push_str(&format!(r##"<text x="{tx:.0}" y="{:.0}" fill="#8297b1" font-size="11" text-anchor="{anchor}">{}</text>"##, y + 4.0, esc(short)));
    }
    s.push_str(&format!(r##"<circle cx="{cx:.0}" cy="{cy:.0}" r="11" fill="#bd9bff" class="lit"/><text x="612" y="76" fill="#8297b1" font-size="11">woven into the ruvnet fabric</text>"##));
    s.push_str("</svg>\n");
    s
}
