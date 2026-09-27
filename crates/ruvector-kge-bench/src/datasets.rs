//! Canonical dataset loaders (plan M3), byte-compatible with
//! `npm/packages/kge/bench/datasets/*.mjs`: the same commit-pinned URLs, the
//! same sha256 pins, the same canonical counts, the same `\r?\n` / tab parse,
//! the same sha256-bucket `transfer` carve (~30% of valid) and the same
//! `splits_hash`. Cache file names match the JS loaders, so a cache warmed by
//! either tool serves both. A file that fails its pin is refused (never
//! cached, never parsed); a dataset whose counts differ from the canonical
//! ones is refused.
//!
//! Ids: entities and relations are numbered by **sorted label** (UTF-16 code
//! unit order, as JS sorts), never by first appearance, so the id order — and
//! the vocabulary-order hash recorded in receipts and exports — depends only
//! on the dataset bytes.

use crate::canon::{atomic_write, sha256_hex};
use anyhow::{bail, Context, Result};
use ruvector_kge::{Triple, TripleStore};
use serde::Serialize;
use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

/// Canonical sizes, asserted on the full (unsliced) parsed splits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Expected {
    pub train: usize,
    pub valid: usize,
    pub test: usize,
    pub entities: usize,
    pub relations: usize,
}

/// One dataset's immutable source description (mirrors the `.mjs` constants).
#[derive(Debug)]
pub struct DatasetSpec {
    pub name: &'static str,
    pub commit: &'static str,
    /// Cache file suffix: `.txt` for the RotatE-repo datasets, none for CoDEx.
    pub cache_ext: &'static str,
    /// `[train, valid, test]` URLs.
    pub urls: [&'static str; 3],
    /// `[train, valid, test]` sha256 pins.
    pub pins: [&'static str; 3],
    pub expected: Expected,
    pub licence: &'static str,
}

const KGE: &str = "2e440e0f9c687314d5ff67ead68ce985dc446e3a";
const CODEX: &str = "3132e426c2a6b643b70bad679905a3a6270be440";
macro_rules! kge_url {
    ($dir:literal, $f:literal) => {
        concat!(
            "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/",
            "2e440e0f9c687314d5ff67ead68ce985dc446e3a/data/",
            $dir,
            "/",
            $f
        )
    };
}
macro_rules! codex_url {
    ($f:literal) => {
        concat!(
            "https://raw.githubusercontent.com/tsafavi/codex/",
            "3132e426c2a6b643b70bad679905a3a6270be440/data/triples/codex-m/",
            $f
        )
    };
}

/// Every supported dataset. Constants copied from the `.mjs` loaders; the
/// `datasets_match_js_loaders` test re-reads those files and fails on drift.
pub static SPECS: &[DatasetSpec] = &[
    DatasetSpec {
        name: "fb15k237",
        commit: KGE,
        cache_ext: ".txt",
        urls: [
            kge_url!("FB15k-237", "train.txt"),
            kge_url!("FB15k-237", "valid.txt"),
            kge_url!("FB15k-237", "test.txt"),
        ],
        pins: [
            "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae",
            "cf6309010852f6a8d47a45df830a426415d1ee6f7a3970a8376ff1fb81db4a5c",
            "5711cf41623ceb4eacc50eb6108a3ca6565c7492e3caaf82a3e355cc660d1574",
        ],
        expected: Expected {
            train: 272115,
            valid: 17535,
            test: 20466,
            entities: 14541,
            relations: 237,
        },
        licence: "Freebase-derived; follows source, not redistributed",
    },
    DatasetSpec {
        name: "wn18rr",
        commit: KGE,
        cache_ext: ".txt",
        urls: [
            kge_url!("wn18rr", "train.txt"),
            kge_url!("wn18rr", "valid.txt"),
            kge_url!("wn18rr", "test.txt"),
        ],
        pins: [
            "038612e783c215ee5f3ca9fbfca27b8d0739be1028fe4ee7c174aecf0b83d5df",
            "453ce7202afa58094a04d2b1560ee2b02660f1c260b32ce6651c8ccedd1028ab",
            "0383bceaaa1096cf3c03ec021ed0048068e2355dbfc0239b292cefdac821cec5",
        ],
        expected: Expected {
            train: 86835,
            valid: 3034,
            test: 3134,
            entities: 40943,
            relations: 11,
        },
        licence: "WordNet-derived; follows source, not redistributed",
    },
    DatasetSpec {
        name: "codexm",
        commit: CODEX,
        cache_ext: "",
        urls: [
            codex_url!("train.txt"),
            codex_url!("valid.txt"),
            codex_url!("test.txt"),
        ],
        pins: [
            "d99c3437ab51690391a26d96976adf6e5494dba7ef6902e77000551bfa566556",
            "11c323096367354940846b9ed940c5dabddc1d20a70474222babe7e7fd28b28d",
            "0575ce05e4ce915e395bb4f708e8df9547989dd4c8c71407cbfd78c19daffcf7",
        ],
        expected: Expected {
            train: 185584,
            valid: 10310,
            test: 10311,
            entities: 17050,
            relations: 51,
        },
        licence: "CC-BY-4.0 (per CoDEx README/paper arXiv:2009.07810; Wikidata-derived)",
    },
    DatasetSpec {
        name: "yago310",
        commit: KGE,
        cache_ext: ".txt",
        urls: [
            kge_url!("YAGO3-10", "train.txt"),
            kge_url!("YAGO3-10", "valid.txt"),
            kge_url!("YAGO3-10", "test.txt"),
        ],
        pins: [
            "afb9b51c68d1c997e85655477045b4c5146cd2a6b50b6eea5373383f35dcb12a",
            "c9018b77ec77e99f8d48bc3258d404f10b2049a6c7b4ac6eb663697e799dc6f5",
            "003887ca8a34c90fcaf9b0250b1c30a4b5f617f80f8cf0aeab9723333061598a",
        ],
        expected: Expected {
            train: 1079040,
            valid: 5000,
            test: 5000,
            entities: 123182,
            relations: 37,
        },
        licence: "YAGO-derived; follows source, not redistributed",
    },
];

const SPLITS: [&str; 3] = ["train", "valid", "test"];

/// Look up a dataset by its harness name.
pub fn spec(name: &str) -> Result<&'static DatasetSpec> {
    SPECS.iter().find(|s| s.name == name).with_context(|| {
        format!("unknown dataset '{name}' (known: fb15k237, wn18rr, codexm, yago310)")
    })
}

impl DatasetSpec {
    /// JS `cacheFileName` / codexm naming: `<name>@<commit[..8]>-<split><ext>`.
    pub fn cache_file(&self, split: &str) -> String {
        format!(
            "{}@{}-{}{}",
            self.name,
            &self.commit[..8],
            split,
            self.cache_ext
        )
    }
}

/// A labelled triple as parsed from a file.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RawTriple {
    pub s: String,
    pub r: String,
    pub o: String,
}

impl RawTriple {
    fn key(&self) -> String {
        format!("{} {} {}", self.s, self.r, self.o)
    }
}

/// JS `parseTriplesTsv`: split on `\r?\n`, skip blank lines, take the first
/// three tab fields (or, failing that, whitespace fields).
pub fn parse_triples(text: &str) -> Vec<RawTriple> {
    let mut out = Vec::new();
    for line in text.split('\n') {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if line.trim().is_empty() {
            continue;
        }
        let tabs: Vec<&str> = line.split('\t').collect();
        let parts: Vec<&str> = if tabs.len() >= 3 {
            tabs
        } else {
            line.split_whitespace().collect()
        };
        if parts.len() >= 3 {
            out.push(RawTriple {
                s: parts[0].into(),
                r: parts[1].into(),
                o: parts[2].into(),
            });
        }
    }
    out
}

/// JS `carveTransfer`: bucket = first 8 hex digits of sha256("s r o") mod 100;
/// bucket < 30 goes to `transfer`. Returns `(valid, transfer)`.
pub fn carve_transfer(valid: Vec<RawTriple>) -> (Vec<RawTriple>, Vec<RawTriple>) {
    let (mut rest, mut transfer) = (Vec::new(), Vec::new());
    for t in valid {
        let h = sha256_hex(t.key());
        let bucket = u32::from_str_radix(&h[..8], 16).expect("hex") % 100;
        if bucket < 30 {
            transfer.push(t)
        } else {
            rest.push(t)
        }
    }
    (rest, transfer)
}

/// JS `stableSplitHash`: sha256 of the sorted `"s r o"` keys joined by `|`.
pub fn stable_split_hash<'a>(triples: impl IntoIterator<Item = &'a RawTriple>) -> String {
    let mut keys: Vec<String> = triples.into_iter().map(RawTriple::key).collect();
    keys.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
    sha256_hex(keys.join("|"))
}

/// A loaded, verified, id-mapped dataset.
#[derive(Debug, Clone)]
pub struct Dataset {
    pub name: String,
    pub licence: String,
    pub sources: BTreeMap<String, String>,
    pub file_hashes: BTreeMap<String, String>,
    pub splits_hash: String,
    /// Per-split `stable_split_hash` (train, valid, transfer, test).
    pub split_hashes: BTreeMap<String, String>,
    /// sha256 of the entity labels in id order, `\n`-joined (labels never
    /// leave this struct; only the hash is recorded).
    pub entity_vocab_hash: String,
    pub relation_vocab_hash: String,
    pub num_entities: usize,
    pub num_relations: usize,
    pub train: Vec<Triple>,
    pub valid: Vec<Triple>,
    pub transfer: Vec<Triple>,
    pub test: Vec<Triple>,
}

/// Counts in the harness's `graphCounts` shape.
#[derive(Debug, Clone, Serialize)]
pub struct Counts {
    pub train: usize,
    pub valid: usize,
    pub transfer: usize,
    pub test: usize,
    pub entities: usize,
    pub relations: usize,
}

impl Dataset {
    pub fn counts(&self) -> Counts {
        Counts {
            train: self.train.len(),
            valid: self.valid.len(),
            transfer: self.transfer.len(),
            test: self.test.len(),
            entities: self.num_entities,
            relations: self.num_relations,
        }
    }

    /// The filter store: train ∪ valid ∪ transfer, plus test when
    /// `include_test` (only `--final` / `--verify-final`). Asserts the store's
    /// triple set equals the union of those splits (plan M3 leakage assert).
    pub fn filter_store(&self, include_test: bool) -> Result<TripleStore> {
        let mut all: Vec<Triple> = Vec::new();
        for split in [&self.train, &self.valid, &self.transfer] {
            all.extend_from_slice(split);
        }
        if include_test {
            all.extend_from_slice(&self.test);
        }
        let union: HashSet<Triple> = all.iter().copied().collect();
        let store =
            TripleStore::with_counts(all, Some(self.num_entities), Some(self.num_relations))?;
        let got: HashSet<Triple> = store.triples().iter().copied().collect();
        if got != union {
            bail!("{}: filter store differs from the split union", self.name);
        }
        Ok(store)
    }

    /// The training store (train split only).
    pub fn train_store(&self) -> Result<TripleStore> {
        Ok(TripleStore::with_counts(
            self.train.clone(),
            Some(self.num_entities),
            Some(self.num_relations),
        )?)
    }

    /// Leakage assert: train ∩ (valid ∪ transfer ∪ test) = ∅ by triple.
    pub fn assert_no_leakage(&self) -> Result<()> {
        let train: HashSet<Triple> = self.train.iter().copied().collect();
        for (name, split) in [
            ("valid", &self.valid),
            ("transfer", &self.transfer),
            ("test", &self.test),
        ] {
            let n = split.iter().filter(|t| train.contains(t)).count();
            if n > 0 {
                bail!(
                    "{}: leakage — {n} {name} triple(s) also in train; refusing",
                    self.name
                );
            }
        }
        Ok(())
    }
}

/// Build a dataset from the three verified files' bytes (`[train, valid,
/// test]`): parse, assert canonical counts, carve, map ids, assert leakage.
pub fn from_bytes(spec: &DatasetSpec, files: [&[u8]; 3]) -> Result<Dataset> {
    let text = |i: usize| {
        std::str::from_utf8(files[i])
            .with_context(|| format!("{}: {} is not UTF-8", spec.name, SPLITS[i]))
    };
    let train = parse_triples(text(0)?);
    let valid_full = parse_triples(text(1)?);
    let test = parse_triples(text(2)?);

    let mut ents: Vec<&str> = Vec::new();
    let mut rels: Vec<&str> = Vec::new();
    for t in train.iter().chain(&valid_full).chain(&test) {
        ents.push(&t.s);
        ents.push(&t.o);
        rels.push(&t.r);
    }
    let sort_dedup = |v: &mut Vec<&str>| {
        v.sort_by(|a, b| a.encode_utf16().cmp(b.encode_utf16()));
        v.dedup();
    };
    sort_dedup(&mut ents);
    sort_dedup(&mut rels);
    let got = Expected {
        train: train.len(),
        valid: valid_full.len(),
        test: test.len(),
        entities: ents.len(),
        relations: rels.len(),
    };
    if got != spec.expected {
        bail!(
            "{}: not the canonical dataset (got {got:?}, expected {:?}) — refusing",
            spec.name,
            spec.expected
        );
    }
    let (valid, transfer) = carve_transfer(valid_full.clone());
    let splits_hash = stable_split_hash(train.iter().chain(&valid).chain(&transfer).chain(&test));
    let mut split_hashes = BTreeMap::new();
    for (k, v) in [
        ("train", &train),
        ("valid", &valid),
        ("transfer", &transfer),
        ("test", &test),
    ] {
        split_hashes.insert(k.to_string(), stable_split_hash(v.iter()));
    }
    let eid: BTreeMap<&str, u32> = ents
        .iter()
        .enumerate()
        .map(|(i, &e)| (e, i as u32))
        .collect();
    let rid: BTreeMap<&str, u32> = rels
        .iter()
        .enumerate()
        .map(|(i, &r)| (r, i as u32))
        .collect();
    let map = |v: &[RawTriple]| -> Vec<Triple> {
        v.iter()
            .map(|t| Triple::new(eid[t.s.as_str()], rid[t.r.as_str()], eid[t.o.as_str()]))
            .collect()
    };
    let ds = Dataset {
        name: spec.name.into(),
        licence: spec.licence.into(),
        sources: SPLITS
            .iter()
            .zip(spec.urls)
            .map(|(k, u)| (k.to_string(), u.to_string()))
            .collect(),
        file_hashes: SPLITS
            .iter()
            .zip(files)
            .map(|(k, b)| (k.to_string(), sha256_hex(b)))
            .collect(),
        splits_hash,
        split_hashes,
        entity_vocab_hash: sha256_hex(ents.join("\n")),
        relation_vocab_hash: sha256_hex(rels.join("\n")),
        num_entities: ents.len(),
        num_relations: rels.len(),
        train: map(&train),
        valid: map(&valid),
        transfer: map(&transfer),
        test: map(&test),
    };
    ds.assert_no_leakage()?;
    Ok(ds)
}

/// Fetch (or read from cache) and verify one file against its pin. Bytes
/// that fail the pin are refused and never written to the cache.
pub fn fetch_verified(url: &str, pin: &str, cache: &Path) -> Result<Vec<u8>> {
    if cache.exists() {
        let bytes = std::fs::read(cache)?;
        let got = sha256_hex(&bytes);
        if got != pin {
            bail!("sha256 mismatch for cached file {} (source {url}): expected {pin}, got {got}; delete it to re-fetch", cache.display());
        }
        return Ok(bytes);
    }
    let resp = ureq::get(url)
        .call()
        .with_context(|| format!("network error fetching {url}"))?;
    let mut bytes = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_reader(), &mut bytes)?;
    let got = sha256_hex(&bytes);
    if got != pin {
        bail!("sha256 mismatch fetched from {url}: expected {pin}, got {got}; refusing");
    }
    atomic_write(cache, &bytes)?;
    Ok(bytes)
}

/// Load `name` through `cache_dir` (fetching missing files), verified.
pub fn load(name: &str, cache_dir: &Path) -> Result<Dataset> {
    let spec = spec(name)?;
    let mut files: Vec<Vec<u8>> = Vec::with_capacity(3);
    for (i, split) in SPLITS.iter().enumerate() {
        files.push(fetch_verified(
            spec.urls[i],
            spec.pins[i],
            &cache_dir.join(spec.cache_file(split)),
        )?);
    }
    from_bytes(spec, [&files[0], &files[1], &files[2]])
}

/// Default cache: `$RVKGE_CACHE_DIR`, else the JS harness cache
/// (`npm/packages/kge/bench/.cache`, gitignored) under the enclosing repo.
pub fn default_cache_dir() -> PathBuf {
    if let Ok(d) = std::env::var("RVKGE_CACHE_DIR") {
        return PathBuf::from(d);
    }
    let mut d = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
    loop {
        if d.join(".git").exists() {
            return d.join("npm/packages/kge/bench/.cache");
        }
        if !d.pop() {
            return PathBuf::from(".kge-cache");
        }
    }
}
