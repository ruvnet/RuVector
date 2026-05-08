// Sparse-Mario — a Super Mario Bros level autoregressive generator
// built on `ruvllm_sparse_attention`.
//
// Iteration 1 scaffold: corpus + tokenizer + stats. No model yet.
// Run with: cargo run --release --example sparse_mario --features parallel

use std::collections::HashMap;

// VGLC-style tile alphabet (Super Mario Bros).
// https://github.com/TheVGLC/TheVGLC (MIT). The three slices below are
// hand-authored, public-domain compositions in the same alphabet.
//
//   - = sky / empty
//   X = solid ground
//   S = breakable brick
//   ? = active question block
//   Q = used question block
//   o = coin
//   < > = pipe top
//   [ ] = pipe body
//   E   = enemy (goomba)
//   B   = cannon ball
//   b   = cannon top
//   M   = mario start

pub const LEVELS: &[&str] = &[
    // Slice A — opening, single pipe, one ? block, two goombas
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
-------------------------------oo-----------------\n\
-----------?--------SSS?S-------------------------\n\
--------------------------------------<>----------\n\
--------------------E---------------E-[]----------\n\
M-------------------------------------[]----------\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",

    // Slice B — staircase, double-pipe, brick ceiling
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------SSSSSSSSSSSSSSSS--------------------------\n\
--------------------------------------------------\n\
-----------------oo-------------------------------\n\
--?------SSS-----?S-S-------------oo--------------\n\
--------------------------------------------------\n\
-----------E-------------<>--------------<>-------\n\
-----------------E--E----[]------E-------[]-------\n\
M-------------XX---------[]----XXXX------[]-------\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX\n\
XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX",

    // Slice C — cannons, gap, coin shower
    "\
--------------------------------------------------\n\
--------------------------------------------------\n\
--------------------------------------------------\n\
-------------------oooooooo-----------------------\n\
--------------------------------------------------\n\
------?-------SSSSS--------SSSSS------------------\n\
--------------------------------------------------\n\
-----------------b---------------b----------------\n\
-----E-----E-----B--E-----E------B------E---------\n\
M-----------------B--------------B----------------\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX\n\
XXXXXXXXXXXXXXXXXXBXXXXXXXXXXXXXXBXXXXX-----XXXXXX",
];

/// Tile vocabulary in deterministic order. Index = token id.
pub const VOCAB: &[char] = &[
    '-', // 0  sky
    'X', // 1  ground
    'S', // 2  breakable brick
    '?', // 3  active ? block
    'Q', // 4  used ? block
    'o', // 5  coin
    '<', // 6  pipe top-left
    '>', // 7  pipe top-right
    '[', // 8  pipe body-left
    ']', // 9  pipe body-right
    'E', // 10 enemy (goomba)
    'B', // 11 cannon ball
    'b', // 12 cannon top
    'M', // 13 mario start
    '\n',// 14 row separator
];

/// Char → token id. Returns None for unknown characters so the corpus stays clean.
pub fn encode_char(c: char) -> Option<u8> {
    VOCAB.iter().position(|&v| v == c).map(|i| i as u8)
}

/// Token id → char.
pub fn decode_token(t: u8) -> char {
    VOCAB.get(t as usize).copied().unwrap_or('?')
}

/// Encode a level slice into a flat token stream, including row separators.
pub fn encode_level(level: &str) -> Vec<u8> {
    level.chars().filter_map(encode_char).collect()
}

/// Encode the entire embedded corpus into one concatenated token stream,
/// with the row-separator token between successive levels too (so the model
/// learns slice boundaries).
pub fn encode_corpus() -> Vec<u8> {
    let nl = encode_char('\n').unwrap();
    let mut out = Vec::new();
    for (i, lvl) in LEVELS.iter().enumerate() {
        if i > 0 {
            out.push(nl);
        }
        out.extend(encode_level(lvl));
    }
    out
}

/// Width (column count) of a level slice. All embedded slices share width=50.
pub fn level_width(level: &str) -> usize {
    level.lines().next().map(|r| r.chars().count()).unwrap_or(0)
}

/// Tile distribution over the full corpus. Returns map char → count.
pub fn tile_distribution(tokens: &[u8]) -> HashMap<char, usize> {
    let mut m = HashMap::new();
    for &t in tokens {
        *m.entry(decode_token(t)).or_insert(0) += 1;
    }
    m
}

fn main() {
    let tokens = encode_corpus();
    let dist = tile_distribution(&tokens);

    println!("== Sparse-Mario corpus ==");
    println!("levels        : {}", LEVELS.len());
    println!("total tokens  : {}", tokens.len());
    println!("vocab size    : {}", VOCAB.len());
    println!("level widths  : {:?}", LEVELS.iter().map(|l| level_width(l)).collect::<Vec<_>>());
    println!();
    println!("Tile distribution:");
    let mut entries: Vec<_> = dist.iter().collect();
    entries.sort_by(|a, b| b.1.cmp(a.1));
    let total = tokens.len() as f64;
    for (c, n) in entries {
        let pct = (*n as f64 / total) * 100.0;
        let label = if *c == '\n' { "\\n".to_string() } else { c.to_string() };
        println!("  {:>3}  {:>5}  {:>5.1}%", label, n, pct);
    }
    println!();
    println!("(iter 1 scaffold — model + generation land in iter 2-3)");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vocab_roundtrip() {
        for (i, &c) in VOCAB.iter().enumerate() {
            assert_eq!(encode_char(c), Some(i as u8));
            assert_eq!(decode_token(i as u8), c);
        }
    }

    #[test]
    fn corpus_nonempty_and_known_tiles() {
        let toks = encode_corpus();
        assert!(toks.len() > 1000, "corpus should have at least 1k tokens, got {}", toks.len());
        for &t in &toks {
            assert!((t as usize) < VOCAB.len(), "out-of-range token {}", t);
        }
    }

    #[test]
    fn each_level_has_mario_start() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            assert!(lvl.contains('M'), "level {} missing mario start tile", i);
        }
    }

    #[test]
    fn each_level_has_ground_floor() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            let last = lvl.lines().last().unwrap_or("");
            let solid = last.chars().filter(|&c| c == 'X').count();
            assert!(
                solid > last.chars().count() / 2,
                "level {} bottom row should be mostly ground",
                i
            );
        }
    }

    #[test]
    fn levels_are_rectangular() {
        for (i, lvl) in LEVELS.iter().enumerate() {
            let w = level_width(lvl);
            for (r, row) in lvl.lines().enumerate() {
                assert_eq!(
                    row.chars().count(), w,
                    "level {} row {} width mismatch (expected {}, got {})",
                    i, r, w, row.chars().count()
                );
            }
        }
    }
}
