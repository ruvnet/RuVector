//! Minimal insertion-ordered JSON writer (byte-stable output, no serde).

pub enum J {
    Obj(Vec<(&'static str, J)>),
    Arr(Vec<J>),
    Str(String),
    Num(u64),
    Bool(bool),
    Null,
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}
pub fn s(x: impl Into<String>) -> J {
    J::Str(x.into())
}
pub fn h(b: &[u8]) -> J {
    J::Str(hex(b))
}
pub fn n(x: impl Into<u64>) -> J {
    J::Num(x.into())
}

impl J {
    pub fn render(&self) -> String {
        let mut o = String::new();
        self.write(&mut o, 0);
        o.push('\n');
        o
    }
    fn write(&self, o: &mut String, ind: usize) {
        let pad = |o: &mut String, k: usize| o.push_str(&"  ".repeat(k));
        match self {
            J::Null => o.push_str("null"),
            J::Bool(b) => o.push_str(if *b { "true" } else { "false" }),
            J::Num(x) => o.push_str(&x.to_string()),
            J::Str(x) => {
                o.push('"');
                for c in x.chars() {
                    match c {
                        '"' => o.push_str("\\\""),
                        '\\' => o.push_str("\\\\"),
                        '\0' => o.push_str("\\u0000"),
                        c => o.push(c),
                    }
                }
                o.push('"');
            }
            J::Arr(v)
                if v.iter()
                    .all(|e| matches!(e, J::Num(_) | J::Str(_) | J::Null | J::Bool(_))) =>
            {
                o.push('[');
                for (i, e) in v.iter().enumerate() {
                    if i > 0 {
                        o.push_str(", ");
                    }
                    e.write(o, ind);
                }
                o.push(']');
            }
            J::Arr(v) => {
                o.push_str("[\n");
                for (i, e) in v.iter().enumerate() {
                    pad(o, ind + 1);
                    e.write(o, ind + 1);
                    o.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                pad(o, ind);
                o.push(']');
            }
            J::Obj(v) => {
                o.push_str("{\n");
                for (i, (k, e)) in v.iter().enumerate() {
                    pad(o, ind + 1);
                    o.push_str(&format!("\"{k}\": "));
                    e.write(o, ind + 1);
                    o.push_str(if i + 1 < v.len() { ",\n" } else { "\n" });
                }
                pad(o, ind);
                o.push('}');
            }
        }
    }
}
