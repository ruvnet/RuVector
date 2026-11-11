//! `config/weaver/topics.toml`: the canonical topic taxonomy (ADR-356).
//! Every non-brand topic carries the demand it was admitted on (npm keyword
//! total, GitHub topic repo count), so "trending" is a recorded measurement,
//! not a guess. Packages and repos map to 3-6 primary topics; packages
//! without an explicit entry are inferred from their own keywords, name and
//! description, never from popularity alone.

use crate::model::{Graph, Package};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Meta {
    pub homepage: String,
    pub measured_at: String,
    #[serde(default = "d_kw_min")]
    pub keywords_min: usize,
    #[serde(default = "d_kw_max")]
    pub keywords_max: usize,
    /// Proposals aim here (inside the 8-20 band, away from stuffing).
    #[serde(default = "d_kw_target")]
    pub keywords_target: usize,
    #[serde(default = "d_topics_max")]
    pub repo_topics_max: usize,
    #[serde(default = "d_desc_max")]
    pub description_max: usize,
    #[serde(default = "d_stale")]
    pub stale_days: i64,
    /// Demand floor: a topic is admitted when EITHER count clears its floor.
    pub min_npm_kw_total: u64,
    pub min_gh_topic_repos: u64,
    /// Packages whose manifests another lane owns: proposed, never applied.
    #[serde(default)]
    pub hands_off_packages: Vec<String>,
    /// Repos another lane owns: proposed, never applied (repo or package level).
    #[serde(default)]
    pub hands_off_repos: Vec<String>,
}

fn d_kw_min() -> usize {
    8
}
fn d_kw_max() -> usize {
    20
}
fn d_kw_target() -> usize {
    16
}
fn d_topics_max() -> usize {
    20
}
fn d_desc_max() -> usize {
    200
}
fn d_stale() -> i64 {
    180
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum TopicKind {
    #[default]
    Topic,
    /// Family names (`ruvector`, `agentdb`): legal keywords, never primary,
    /// exempt from the demand floor.
    Brand,
    /// Exact-feature long-tail term (a pilot search theme such as
    /// `link-prediction`): measured and recorded, legal as a keyword, never
    /// primary, exempt from the floor because it names a verified feature.
    Niche,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Topic {
    pub slug: String,
    /// Phrase the description check looks for ("vector database").
    pub phrase: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default)]
    pub kind: TopicKind,
    #[serde(default)]
    pub npm_kw_total: u64,
    #[serde(default)]
    pub npm_top5_monthly: u64,
    #[serde(default)]
    pub gh_topic_repos: u64,
}

impl Topic {
    /// Demand score used for ordering (log-free: the two counts are on
    /// different scales, so each is normalised by its floor).
    pub fn demand(&self, meta: &Meta) -> f64 {
        self.npm_kw_total as f64 / meta.min_npm_kw_total.max(1) as f64
            + self.gh_topic_repos as f64 / meta.min_gh_topic_repos.max(1) as f64
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PackageTopics {
    pub name: String,
    pub primary: Vec<String>,
    /// Hand-written description (<= description_max, contains a primary phrase).
    #[serde(default)]
    pub description: Option<String>,
    /// Steps of the package's real flow, rendered as the animated README SVG.
    #[serde(default)]
    pub flow: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RepoTopics {
    pub slug: String,
    pub topics: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct File {
    meta: Meta,
    #[serde(default)]
    topic: Vec<Topic>,
    #[serde(default)]
    package: Vec<PackageTopics>,
    #[serde(default)]
    repo: Vec<RepoTopics>,
}

#[derive(Debug, Clone)]
pub struct Taxonomy {
    pub meta: Meta,
    pub topics: Vec<Topic>,
    pub packages: Vec<PackageTopics>,
    pub repos: Vec<RepoTopics>,
    index: BTreeMap<String, usize>,
}

/// Lower-case, trimmed, spaces/underscores to hyphens.
pub fn norm(kw: &str) -> String {
    kw.trim().to_ascii_lowercase().replace([' ', '_'], "-")
}

fn bare(name: &str) -> &str {
    name.rsplit('/').next().unwrap_or(name)
}

impl Taxonomy {
    pub fn parse(text: &str) -> Result<Self> {
        let f: File = toml::from_str(text).context("parsing topics.toml")?;
        let mut index = BTreeMap::new();
        for (i, t) in f.topic.iter().enumerate() {
            if t.slug != norm(&t.slug) {
                bail!("topic {:?} is not normalised", t.slug);
            }
            if t.kind == TopicKind::Topic
                && t.npm_kw_total < f.meta.min_npm_kw_total
                && t.gh_topic_repos < f.meta.min_gh_topic_repos
            {
                bail!(
                    "topic {} below the demand floor (npm {} < {}, gh {} < {})",
                    t.slug,
                    t.npm_kw_total,
                    f.meta.min_npm_kw_total,
                    t.gh_topic_repos,
                    f.meta.min_gh_topic_repos
                );
            }
            for k in std::iter::once(&t.slug).chain(&t.aliases) {
                if index.insert(norm(k), i).is_some() {
                    bail!("keyword {k:?} maps to two topics");
                }
            }
        }
        let tax = Self {
            meta: f.meta,
            topics: f.topic,
            packages: f.package,
            repos: f.repo,
            index,
        };
        for p in &tax.packages {
            if !(3..=6).contains(&p.primary.len()) {
                bail!("package {} needs 3-6 primary topics", p.name);
            }
            for t in &p.primary {
                match tax.topic(t) {
                    Some(x) if x.kind == TopicKind::Topic => {}
                    _ => bail!("package {}: primary {t:?} is not a demand topic", p.name),
                }
            }
            if let Some(d) = &p.description {
                if d.chars().count() > tax.meta.description_max {
                    bail!("package {}: description over the limit", p.name);
                }
                if !tax.has_phrase(d, &p.primary) {
                    bail!("package {}: description lacks a primary phrase", p.name);
                }
            }
        }
        for r in &tax.repos {
            if r.topics.len() > tax.meta.repo_topics_max {
                bail!(
                    "repo {} has more than {} topics",
                    r.slug,
                    tax.meta.repo_topics_max
                );
            }
            for t in &r.topics {
                if tax.topic(t).is_none() {
                    bail!("repo {}: topic {t:?} not in the taxonomy", r.slug);
                }
            }
        }
        Ok(tax)
    }

    /// Topic for a keyword (slug or alias, normalised).
    pub fn topic(&self, kw: &str) -> Option<&Topic> {
        self.index.get(&norm(kw)).map(|i| &self.topics[*i])
    }

    /// Is `kw` legal for package `pkg`: a taxonomy slug/alias, or one of the
    /// package's own name tokens (`agentdb`, `rvf`).
    pub fn keyword_ok(&self, pkg: &str, kw: &str) -> bool {
        let k = norm(kw);
        self.topic(&k).is_some()
            || bare(pkg) == k
            || bare(pkg).split(['-', '.']).any(|t| t.len() > 2 && t == k)
    }

    pub fn package(&self, name: &str) -> Option<&PackageTopics> {
        self.packages.iter().find(|p| p.name == name)
    }

    pub fn repo(&self, slug: &str) -> Option<&RepoTopics> {
        self.repos
            .iter()
            .find(|r| r.slug.eq_ignore_ascii_case(slug))
    }

    pub fn hands_off_package(&self, name: &str) -> bool {
        self.meta.hands_off_packages.iter().any(|p| p == name)
    }

    pub fn hands_off_repo(&self, slug: &str) -> bool {
        self.meta
            .hands_off_repos
            .iter()
            .any(|r| r.eq_ignore_ascii_case(slug))
    }

    /// Does `text` contain the phrase (or slug / an alias) of any of `topics`?
    pub fn has_phrase(&self, text: &str, topics: &[String]) -> bool {
        let hay = format!(
            " {} ",
            norm(text).replace(['.', ',', ';', ':', '(', ')'], " ")
        );
        topics.iter().filter_map(|t| self.topic(t)).any(|t| {
            std::iter::once(&t.phrase)
                .chain(std::iter::once(&t.slug))
                .chain(&t.aliases)
                .any(|p| {
                    let p = norm(p);
                    hay.contains(&format!("-{p}-"))
                        || hay.contains(&format!(" {p} "))
                        || hay.contains(&format!(" {p}-"))
                        || hay.contains(&format!("-{p} "))
                })
        })
    }

    /// Relevance score of each demand topic for a package: its own
    /// keywords (3), name tokens (3), description phrase (2).
    fn scores(&self, p: &Package) -> Vec<(usize, f64)> {
        let mut score: BTreeMap<usize, f64> = BTreeMap::new();
        for k in &p.keywords {
            if let Some(i) = self.index.get(&norm(k)) {
                *score.entry(*i).or_default() += 3.0;
            }
        }
        for t in bare(&p.name).split(['-', '.']) {
            if let Some(i) = self.index.get(t) {
                *score.entry(*i).or_default() += 3.0;
            }
        }
        for (i, t) in self.topics.iter().enumerate() {
            if self.has_phrase(&p.description, std::slice::from_ref(&t.slug)) {
                *score.entry(i).or_default() += 2.0;
            }
        }
        let mut v: Vec<(usize, f64)> = score
            .into_iter()
            .filter(|(i, _)| self.topics[*i].kind == TopicKind::Topic)
            .collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    self.topics[b.0]
                        .demand(&self.meta)
                        .partial_cmp(&self.topics[a.0].demand(&self.meta))
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });
        v
    }

    /// 3-6 primary topics: explicit entry, else inferred from the package's
    /// own signals, topped up from its repo's topics. May return fewer than
    /// 3 when nothing relevant is known (the audit reports it).
    pub fn primary_for(&self, graph: &Graph, p: &Package) -> Vec<String> {
        if let Some(e) = self.package(&p.name) {
            return e.primary.clone();
        }
        let mut out: Vec<String> = self
            .scores(p)
            .into_iter()
            .take(6)
            .map(|(i, _)| self.topics[i].slug.clone())
            .collect();
        if out.len() < 3 {
            for t in self.repo_topics(graph, p.repo_slug.as_deref().unwrap_or("")) {
                if out.len() >= 3 {
                    break;
                }
                if !out.contains(&t) && self.topic(&t).map(|x| x.kind) == Some(TopicKind::Topic) {
                    out.push(t);
                }
            }
        }
        out
    }

    /// Repo topics: explicit entry, else the union of its packages' primary
    /// topics ordered by how many packages share them, then demand (<= max).
    pub fn repo_topics(&self, graph: &Graph, slug: &str) -> Vec<String> {
        if let Some(r) = self.repo(slug) {
            return r.topics.clone();
        }
        let Some(repo) = graph.repo(slug) else {
            return vec![];
        };
        let mut count: BTreeMap<String, usize> = BTreeMap::new();
        for name in &repo.packages {
            let Some(p) = graph.get(name) else { continue };
            let prim = match self.package(name) {
                Some(e) => e.primary.clone(),
                None => self
                    .scores(p)
                    .into_iter()
                    .take(6)
                    .map(|(i, _)| self.topics[i].slug.clone())
                    .collect(),
            };
            for t in prim {
                *count.entry(t).or_default() += 1;
            }
        }
        let mut v: Vec<(String, usize)> = count.into_iter().collect();
        v.sort_by(|a, b| {
            b.1.cmp(&a.1).then(
                self.demand_of(&b.0)
                    .partial_cmp(&self.demand_of(&a.0))
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
        });
        v.into_iter()
            .take(self.meta.repo_topics_max)
            .map(|(t, _)| t)
            .collect()
    }

    fn demand_of(&self, slug: &str) -> f64 {
        self.topic(slug)
            .map(|t| t.demand(&self.meta))
            .unwrap_or(0.0)
    }

    /// Other demand topics relevant to the package beyond its primaries,
    /// most relevant first (keyword padding source).
    pub fn secondary_for(&self, p: &Package, primary: &[String]) -> Vec<String> {
        self.scores(p)
            .into_iter()
            .map(|(i, _)| self.topics[i].slug.clone())
            .filter(|s| !primary.contains(s))
            .collect()
    }

    /// Brand keywords that apply to a package (its family name tokens).
    pub fn brands_for(&self, p: &Package) -> Vec<String> {
        let name = norm(&p.name);
        self.topics
            .iter()
            .filter(|t| t.kind == TopicKind::Brand)
            .filter(|t| {
                std::iter::once(&t.slug)
                    .chain(&t.aliases)
                    .any(|a| name.contains(&norm(a)))
            })
            .map(|t| t.slug.clone())
            .collect()
    }
}
