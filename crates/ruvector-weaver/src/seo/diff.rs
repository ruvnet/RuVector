//! Minimal unified diff (prefix/suffix trim + Myers O(ND) on the changed
//! middle), enough to show the exact change a proposal makes to one file.

/// Myers shortest edit script as (op, line) with op in ' ', '-', '+'.
fn myers<'a>(a: &[&'a str], b: &[&'a str]) -> Vec<(char, &'a str)> {
    let (n, m) = (a.len() as isize, b.len() as isize);
    let max = (n + m) as usize;
    let off = max as isize;
    let mut v = vec![0isize; 2 * max + 2];
    let mut trace: Vec<Vec<isize>> = Vec::new();
    'outer: for d in 0..=max as isize {
        trace.push(v.clone());
        let mut k = -d;
        while k <= d {
            let i = (k + off) as usize;
            let mut x = if k == -d || (k != d && v[i - 1] < v[i + 1]) {
                v[i + 1]
            } else {
                v[i - 1] + 1
            };
            let mut y = x - k;
            while x < n && y < m && a[x as usize] == b[y as usize] {
                x += 1;
                y += 1;
            }
            v[i] = x;
            if x >= n && y >= m {
                break 'outer;
            }
            k += 2;
        }
    }
    let (mut x, mut y) = (n, m);
    let mut out = Vec::new();
    for d in (0..trace.len() as isize).rev() {
        let v = &trace[d as usize];
        let k = x - y;
        let i = (k + off) as usize;
        let pk = if k == -d || (k != d && v[i - 1] < v[i + 1]) {
            k + 1
        } else {
            k - 1
        };
        let px = v[(pk + off) as usize];
        let py = px - pk;
        while x > px && y > py {
            out.push((' ', a[(x - 1) as usize]));
            x -= 1;
            y -= 1;
        }
        if d > 0 {
            if x == px {
                out.push(('+', b[(y - 1) as usize]));
            } else {
                out.push(('-', a[(x - 1) as usize]));
            }
        }
        x = px;
        y = py;
    }
    out.reverse();
    out
}

/// Unified diff of `before` -> `after` with 3 lines of context; empty when equal.
pub fn unified(path: &str, before: &str, after: &str) -> String {
    if before == after {
        return String::new();
    }
    let a: Vec<&str> = before.lines().collect();
    let b: Vec<&str> = after.lines().collect();
    let pre = a.iter().zip(&b).take_while(|(x, y)| x == y).count();
    let suf = a[pre..]
        .iter()
        .rev()
        .zip(b[pre..].iter().rev())
        .take_while(|(x, y)| x == y)
        .count();
    let mut ops: Vec<(char, &str)> = a[..pre].iter().map(|l| (' ', *l)).collect();
    ops.extend(myers(&a[pre..a.len() - suf], &b[pre..b.len() - suf]));
    ops.extend(a[a.len() - suf..].iter().map(|l| (' ', *l)));
    let mut out = format!("--- a/{path}\n+++ b/{path}\n");
    let changed: Vec<usize> = (0..ops.len()).filter(|i| ops[*i].0 != ' ').collect();
    let mut k = 0;
    while k < changed.len() {
        let start = changed[k].saturating_sub(3);
        let mut end = changed[k];
        while k < changed.len() && changed[k] <= end + 6 {
            end = changed[k];
            k += 1;
        }
        let end = (end + 4).min(ops.len());
        let (mut la, mut lb) = (1, 1);
        for (c, _) in &ops[..start] {
            if *c != '+' {
                la += 1;
            }
            if *c != '-' {
                lb += 1;
            }
        }
        let ca = ops[start..end].iter().filter(|(c, _)| *c != '+').count();
        let cb = ops[start..end].iter().filter(|(c, _)| *c != '-').count();
        out.push_str(&format!("@@ -{la},{ca} +{lb},{cb} @@\n"));
        for (c, l) in &ops[start..end] {
            out.push_str(&format!("{c}{l}\n"));
        }
    }
    out
}
