use crate::{
    compiler::{MAX_LIST_ITEMS, MAX_STEPS, MAX_VECTOR_DIM},
    error::{Error, Result},
    types::{Effect, Op, Program, Step},
};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq)]
enum Kind {
    Word(String),
    String(String),
    Number(String),
    Symbol(char),
    End,
}
#[derive(Debug, Clone)]
struct Token {
    kind: Kind,
    at: usize,
}

pub(super) fn parse(source: &str) -> Result<Program> {
    let mut parser = Parser {
        source,
        cursor: 0,
        token: Token {
            kind: Kind::End,
            at: 0,
        },
    };
    parser.advance()?;
    parser.word("program")?;
    let name = parser.identifier()?;
    parser.symbol('{')?;
    let mut effects = BTreeSet::new();
    let mut budget = None;
    let mut steps = Vec::new();
    let emit;
    loop {
        let statement_at = parser.token.at;
        match parser.identifier()?.as_str() {
            "effect" => {
                let effect = match parser.identifier()?.as_str() {
                    "graph.read" => Effect::GraphRead,
                    "model.predict" => Effect::ModelPredict,
                    "decision.emit" => Effect::DecisionEmit,
                    other => {
                        return Err(Error::new("effect", format!("unsupported effect {other}"))
                            .at(statement_at))
                    }
                };
                if !effects.insert(effect) {
                    return Err(
                        Error::new("duplicate_effect", "duplicate effect declaration")
                            .at(statement_at),
                    );
                }
            }
            "budget" => {
                if budget.is_some() {
                    return Err(
                        Error::new("syntax", "duplicate budget declaration").at(statement_at)
                    );
                }
                budget = Some(parser.integer()?);
            }
            "let" => {
                if steps.len() == MAX_STEPS {
                    return Err(
                        Error::new("step_limit", "program exceeds 1024 steps").at(statement_at)
                    );
                }
                let bind = parser.identifier()?;
                parser.symbol('=')?;
                let op = parser.operation()?;
                steps.push(Step { bind, op });
            }
            "emit" => {
                emit = parser.identifier()?;
                parser.symbol(';')?;
                parser.symbol('}')?;
                if parser.token.kind != Kind::End {
                    return Err(parser.error("unexpected input after program"));
                }
                break;
            }
            other => {
                return Err(Error::new(
                    "syntax",
                    format!("expected effect, budget, let, or final emit; found {other}"),
                )
                .at(statement_at))
            }
        }
        parser.symbol(';')?;
    }
    Ok(Program {
        version: 1,
        name,
        effects,
        budget: budget.ok_or_else(|| Error::new("syntax", "missing budget declaration"))?,
        steps,
        emit,
    })
}

struct Parser<'a> {
    source: &'a str,
    cursor: usize,
    token: Token,
}
impl Parser<'_> {
    fn operation(&mut self) -> Result<Op> {
        let at = self.token.at;
        match self.identifier()?.as_str() {
            "nodes" => Ok(Op::Nodes {
                label: self.string()?,
            }),
            "select" => Ok(Op::Select {
                ids: self.strings()?,
            }),
            "traverse" => {
                let input = self.identifier()?;
                let relation = self.string()?;
                self.word("depth")?;
                Ok(Op::Traverse {
                    input,
                    relation,
                    depth: self.size()?,
                })
            }
            "similar" => {
                let input = self.identifier()?;
                let vector = self.vector()?;
                self.word("top")?;
                Ok(Op::Similar {
                    input,
                    vector,
                    k: self.size()?,
                })
            }
            "observe" => {
                let input = self.identifier()?;
                let predicate = self.string()?;
                self.word("within")?;
                Ok(Op::Observe {
                    input,
                    predicate,
                    within: self.integer()?,
                })
            }
            "infer" => {
                let inputs = self.identifiers()?;
                self.word("using")?;
                Ok(Op::Infer {
                    inputs,
                    rules: self.strings()?,
                })
            }
            "predict" => {
                let input = self.identifier()?;
                let relation = self.string()?;
                self.word("model")?;
                let model = self.string()?;
                self.word("top")?;
                Ok(Op::Predict {
                    input,
                    relation,
                    model,
                    k: self.size()?,
                })
            }
            "assess" => Ok(Op::Assess {
                input: self.identifier()?,
            }),
            other => Err(Error::new("operation", format!("unsupported operation {other}")).at(at)),
        }
    }
    fn strings(&mut self) -> Result<Vec<String>> {
        self.list(MAX_LIST_ITEMS, Self::string)
    }
    fn identifiers(&mut self) -> Result<Vec<String>> {
        self.list(MAX_LIST_ITEMS, Self::identifier)
    }
    fn vector(&mut self) -> Result<Vec<f32>> {
        self.list(MAX_VECTOR_DIM, Self::float)
    }
    fn list<T>(&mut self, max: usize, read: fn(&mut Self) -> Result<T>) -> Result<Vec<T>> {
        self.symbol('[')?;
        let mut items = Vec::new();
        if self.token.kind == Kind::Symbol(']') {
            self.advance()?;
            return Ok(items);
        }
        loop {
            if items.len() == max {
                return Err(self.error("list exceeds its item limit"));
            }
            items.push(read(self)?);
            if self.token.kind == Kind::Symbol(']') {
                self.advance()?;
                break;
            }
            self.symbol(',')?;
        }
        Ok(items)
    }
    fn identifier(&mut self) -> Result<String> {
        match self.token.kind.clone() {
            Kind::Word(value) => {
                self.advance()?;
                Ok(value)
            }
            _ => Err(self.error("expected identifier")),
        }
    }
    fn string(&mut self) -> Result<String> {
        match self.token.kind.clone() {
            Kind::String(value) => {
                self.advance()?;
                Ok(value)
            }
            _ => Err(self.error("expected JSON quoted string")),
        }
    }
    fn integer(&mut self) -> Result<u64> {
        match self.token.kind.clone() {
            Kind::Number(value) if value.bytes().all(|b| b.is_ascii_digit()) => {
                let number = value
                    .parse()
                    .map_err(|_| self.error("integer exceeds u64 range"))?;
                self.advance()?;
                Ok(number)
            }
            _ => Err(self.error("expected unsigned integer")),
        }
    }
    fn size(&mut self) -> Result<usize> {
        usize::try_from(self.integer()?).map_err(|_| self.error("integer exceeds platform size"))
    }
    fn float(&mut self) -> Result<f32> {
        match self.token.kind.clone() {
            Kind::Number(value) => {
                let number: f32 = serde_json::from_str(&value)
                    .map_err(|_| self.error("expected finite f32 number"))?;
                if !number.is_finite() {
                    return Err(self.error("expected finite f32 number"));
                }
                self.advance()?;
                Ok(number)
            }
            _ => Err(self.error("expected numeric vector component")),
        }
    }
    fn word(&mut self, expected: &str) -> Result<()> {
        if self.token.kind != Kind::Word(expected.into()) {
            return Err(self.error(&format!("expected {expected}")));
        }
        self.advance()
    }
    fn symbol(&mut self, expected: char) -> Result<()> {
        if self.token.kind != Kind::Symbol(expected) {
            return Err(self.error(&format!("expected {expected}")));
        }
        self.advance()
    }
    fn error(&self, message: &str) -> Error {
        Error::new("syntax", message).at(self.token.at)
    }
    fn advance(&mut self) -> Result<()> {
        self.skip_space();
        let at = self.cursor;
        let bytes = self.source.as_bytes();
        if at == bytes.len() {
            self.token = Token {
                kind: Kind::End,
                at,
            };
            return Ok(());
        }
        let current = bytes[at];
        let kind = match current {
            b'{' | b'}' | b'[' | b']' | b'=' | b';' | b',' => {
                self.cursor += 1;
                Kind::Symbol(current as char)
            }
            b'"' => {
                self.cursor += 1;
                let mut escaped = false;
                let mut closed = false;
                while self.cursor < bytes.len() {
                    let b = bytes[self.cursor];
                    self.cursor += 1;
                    if escaped {
                        escaped = false;
                        continue;
                    }
                    if b == b'\\' {
                        escaped = true;
                    } else if b == b'"' {
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    return Err(Error::new("syntax", "unterminated quoted string").at(at));
                }
                let value = serde_json::from_str(&self.source[at..self.cursor]).map_err(|e| {
                    Error::new("syntax", format!("invalid quoted string: {e}")).at(at)
                })?;
                Kind::String(value)
            }
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => {
                self.cursor += 1;
                while self.cursor < bytes.len()
                    && (bytes[self.cursor].is_ascii_alphanumeric()
                        || matches!(bytes[self.cursor], b'_' | b'.'))
                {
                    self.cursor += 1;
                }
                Kind::Word(self.source[at..self.cursor].into())
            }
            b'0'..=b'9' | b'-' => {
                self.cursor += 1;
                while self.cursor < bytes.len()
                    && (bytes[self.cursor].is_ascii_digit()
                        || matches!(bytes[self.cursor], b'.' | b'e' | b'E' | b'+' | b'-'))
                {
                    self.cursor += 1;
                }
                let raw = &self.source[at..self.cursor];
                let _: serde_json::Number = serde_json::from_str(raw)
                    .map_err(|_| Error::new("syntax", "invalid JSON number").at(at))?;
                Kind::Number(raw.into())
            }
            _ => return Err(Error::new("syntax", "unexpected character").at(at)),
        };
        self.token = Token { kind, at };
        Ok(())
    }
    fn skip_space(&mut self) {
        let bytes = self.source.as_bytes();
        loop {
            while self.cursor < bytes.len() && bytes[self.cursor].is_ascii_whitespace() {
                self.cursor += 1;
            }
            if bytes.get(self.cursor..self.cursor + 2) == Some(b"//") {
                while self.cursor < bytes.len() && bytes[self.cursor] != b'\n' {
                    self.cursor += 1;
                }
            } else {
                break;
            }
        }
    }
}
