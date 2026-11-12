//! Small animated flow SVG for a flagship README (ADR-356): the package's
//! real pipeline as boxes joined by a path, a token travelling along it
//! (SMIL `animateMotion`) and each stage lighting in turn. Self-contained:
//! no external fonts, scripts or `@import`, well under the 20 KB cap, and
//! no numbers (numeric claims live only in the ledger-rendered block).

use crate::svg::esc;
use anyhow::{bail, Result};

pub const MAX_BYTES: usize = 20 * 1024;

pub fn file_name(pkg: &str) -> String {
    format!("{}-flow.svg", pkg.trim_start_matches('@').replace('/', "-"))
}

pub fn render(pkg: &str, steps: &[String]) -> Result<String> {
    if !(2..=6).contains(&steps.len()) {
        bail!("{pkg}: a flow needs 2-6 steps");
    }
    for s in steps {
        if s.chars().any(|c| c.is_ascii_digit()) {
            bail!("{pkg}: flow step {s:?} carries a number (ledger-only)");
        }
        if s.chars().count() > 22 {
            bail!("{pkg}: flow step {s:?} longer than 22 chars");
        }
    }
    let n = steps.len();
    let (bw, gap, h) = (150usize, 46usize, 64usize);
    let w = n * bw + (n - 1) * gap + 40;
    let total_h = 150;
    let y = 52;
    let dur = 1.2 * n as f64;
    let mut s = format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}" height="{total_h}" viewBox="0 0 {w} {total_h}" role="img" aria-label="{}">
<title>{}</title>
<style>
.bg{{fill:#0d1117}}.box{{fill:#161b22;stroke:#30363d;stroke-width:1.5}}
.lbl{{fill:#e6edf3;font:600 13px system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif}}
.ttl{{fill:#7d8590;font:500 12px system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif}}
.wire{{fill:none;stroke:#30363d;stroke-width:2;stroke-dasharray:6 6}}
.tok{{fill:#58a6ff}}
@media (prefers-reduced-motion: reduce){{animate,animateMotion{{display:none}}}}
</style>
<rect class="bg" width="{w}" height="{total_h}" rx="10"/>
<text class="ttl" x="20" y="28">{}</text>
"##,
        esc(&format!("{pkg}: {}", steps.join(" to "))),
        esc(pkg),
        esc(pkg)
    );
    let cy = y + h / 2;
    let x0 = 20 + bw / 2;
    let x1 = 20 + (n - 1) * (bw + gap) + bw / 2;
    s.push_str(&format!(
        "<path id=\"wire\" class=\"wire\" d=\"M{x0} {cy} H{x1}\"><animate attributeName=\"stroke-dashoffset\" from=\"24\" to=\"0\" dur=\"1s\" repeatCount=\"indefinite\"/></path>\n"
    ));
    for (i, step) in steps.iter().enumerate() {
        let x = 20 + i * (bw + gap);
        let begin = 1.2 * i as f64;
        s.push_str(&format!(
            "<g><rect class=\"box\" x=\"{x}\" y=\"{y}\" width=\"{bw}\" height=\"{h}\" rx=\"8\"><animate attributeName=\"stroke\" values=\"#30363d;#58a6ff;#30363d\" keyTimes=\"0;0.15;1\" dur=\"{dur:.1}s\" begin=\"{begin:.1}s\" repeatCount=\"indefinite\"/></rect><text class=\"lbl\" x=\"{}\" y=\"{}\" text-anchor=\"middle\">{}</text></g>\n",
            x + bw / 2,
            cy + 5,
            esc(step)
        ));
    }
    s.push_str(&format!(
        "<circle class=\"tok\" r=\"6\"><animateMotion dur=\"{dur:.1}s\" repeatCount=\"indefinite\" keyPoints=\"0;1\" keyTimes=\"0;1\" calcMode=\"linear\"><mpath xlink:href=\"#wire\"/></animateMotion></circle>\n</svg>\n"
    ));
    if s.len() > MAX_BYTES {
        bail!("{pkg}: flow svg {} bytes > {MAX_BYTES}", s.len());
    }
    Ok(s)
}
