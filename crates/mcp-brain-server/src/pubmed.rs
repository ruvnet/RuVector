//! PubMed discovery pipeline for π.ruv.io (ADR-094).
//!
//! Fetches biomedical abstracts from NCBI E-utilities, processes them through
//! the web memory ingestion pipeline, and detects emerging research clusters,
//! contradictions, and knowledge evolution patterns.
//!
//! Data source: NCBI PubMed E-utilities (public, no API key required for <3 req/sec)
//! - esearch: find PMIDs by query
//! - efetch: retrieve abstracts in XML
//!
//! Uses the live π.ruv.io brain server API at https://pi.ruv.io

use crate::embeddings::{EmbeddingEngine, EMBED_DIM};
use crate::graph::{cosine_similarity, KnowledgeGraph};
use crate::web_ingest;
use crate::web_memory::*;
use chrono::Utc;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// NCBI E-utilities base URL
const ESEARCH_URL: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi";
const EFETCH_URL: &str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi";

/// π.ruv.io brain server API
const BRAIN_API: &str = "https://pi.ruv.io";

/// Maximum abstracts per fetch batch (NCBI limit: 10K per request)
const FETCH_BATCH_SIZE: usize = 100;

/// Rate limit: NCBI allows 3 requests/second without API key
const RATE_LIMIT_MS: u64 = 350;

// ── PubMed Article Types ────────────────────────────────────────────────

/// A PubMed article parsed from E-utilities XML.
#[derive(Debug, Clone)]
pub struct PubMedArticle {
    pub pmid: String,
    pub title: String,
    pub abstract_text: String,
    pub authors: Vec<String>,
    pub journal: String,
    pub pub_date: String,
    pub mesh_terms: Vec<String>,
    pub references: Vec<String>,
}

impl PubMedArticle {
    /// Convert to a CleanedPage for the web memory ingestion pipeline.
    pub fn to_cleaned_page(&self) -> CleanedPage {
        let text = if self.abstract_text.is_empty() {
            self.title.clone()
        } else {
            format!("{}\n\n{}", self.title, self.abstract_text)
        };

        let mut tags = self.mesh_terms.clone();
        if !self.journal.is_empty() {
            tags.push(format!("journal:{}", self.journal));
        }

        CleanedPage {
            url: format!("https://pubmed.ncbi.nlm.nih.gov/{}/", self.pmid),
            text,
            title: self.title.clone(),
            meta_description: self.abstract_text.chars().take(300).collect(),
            links: self
                .references
                .iter()
                .map(|pmid| format!("https://pubmed.ncbi.nlm.nih.gov/{pmid}/"))
                .collect(),
            language: "en".to_string(),
            embedding: vec![],
            tags,
        }
    }
}

// ── E-utilities Fetch ───────────────────────────────────────────────────

/// Search PubMed and return PMIDs matching the query.
pub async fn esearch(
    client: &reqwest::Client,
    query: &str,
    max_results: usize,
) -> Result<Vec<String>, String> {
    let url = format!(
        "{ESEARCH_URL}?db=pubmed&term={}&retmax={max_results}&retmode=json&sort=date",
        urlencoding::encode(query)
    );

    let resp = client
        .get(&url)
        .header("User-Agent", "RuVector-PiRuvIo/0.1 (https://pi.ruv.io)")
        .send()
        .await
        .map_err(|e| format!("esearch request failed: {e}"))?;

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("esearch JSON parse failed: {e}"))?;

    let ids = body["esearchresult"]["idlist"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    Ok(ids)
}

/// Fetch article details for a batch of PMIDs.
pub async fn efetch(
    client: &reqwest::Client,
    pmids: &[String],
) -> Result<Vec<PubMedArticle>, String> {
    if pmids.is_empty() {
        return Ok(vec![]);
    }

    let id_str = pmids.join(",");
    let url = format!("{EFETCH_URL}?db=pubmed&id={id_str}&rettype=xml&retmode=xml");

    let resp = client
        .get(&url)
        .header("User-Agent", "RuVector-PiRuvIo/0.1 (https://pi.ruv.io)")
        .send()
        .await
        .map_err(|e| format!("efetch request failed: {e}"))?;

    let xml = resp
        .text()
        .await
        .map_err(|e| format!("efetch text read failed: {e}"))?;

    Ok(parse_pubmed_xml(&xml))
}

/// Parse PubMed XML response into article structs.
///
/// Uses simple string-based parsing (no XML crate dependency) to extract
/// key fields from PubMedArticleSet XML. Handles the common MEDLINE format.
fn parse_pubmed_xml(xml: &str) -> Vec<PubMedArticle> {
    let mut articles = Vec::new();

    // Split by <PubmedArticle> tags
    for article_xml in xml.split("<PubmedArticle>").skip(1) {
        let pmid = extract_tag(article_xml, "PMID").unwrap_or_default();
        if pmid.is_empty() {
            continue;
        }

        let title = extract_tag(article_xml, "ArticleTitle").unwrap_or_default();
        let abstract_text = extract_abstract(article_xml);
        let journal = extract_tag(article_xml, "Title").unwrap_or_default();
        let pub_date = extract_pub_date(article_xml);
        let authors = extract_authors(article_xml);
        let mesh_terms = extract_mesh_terms(article_xml);
        let references = extract_references(article_xml);

        articles.push(PubMedArticle {
            pmid,
            title,
            abstract_text,
            authors,
            journal,
            pub_date,
            mesh_terms,
            references,
        });
    }

    articles
}

/// Extract text content of the first occurrence of an XML tag.
fn extract_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");

    let start_pos = xml.find(&open)?;
    // Find the end of the opening tag (handle attributes)
    let content_start = xml[start_pos..].find('>')? + start_pos + 1;
    let end_pos = xml[content_start..].find(&close)? + content_start;

    let content = &xml[content_start..end_pos];
    // Strip inner XML tags for clean text
    Some(strip_xml_tags(content))
}

/// Extract the full abstract text (may have multiple AbstractText segments).
fn extract_abstract(xml: &str) -> String {
    let mut parts = Vec::new();

    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find("<AbstractText") {
        let abs_start = search_from + start;
        let content_start = match xml[abs_start..].find('>') {
            Some(p) => abs_start + p + 1,
            None => break,
        };
        let end = match xml[content_start..].find("</AbstractText>") {
            Some(p) => content_start + p,
            None => break,
        };

        // Check for Label attribute
        let tag_str = &xml[abs_start..content_start];
        let label = if let Some(lpos) = tag_str.find("Label=\"") {
            let lstart = lpos + 7;
            tag_str[lstart..]
                .find('"')
                .map(|end| &tag_str[lstart..lstart + end])
        } else {
            None
        };

        let text = strip_xml_tags(&xml[content_start..end]);
        if let Some(label) = label {
            parts.push(format!("{label}: {text}"));
        } else {
            parts.push(text);
        }

        search_from = end;
    }

    parts.join(" ")
}

/// Extract publication date from MedlineDate or Year/Month.
fn extract_pub_date(xml: &str) -> String {
    if let Some(date) = extract_tag(xml, "MedlineDate") {
        return date;
    }
    let year = extract_tag(xml, "Year").unwrap_or_default();
    let month = extract_tag(xml, "Month").unwrap_or_default();
    if !year.is_empty() {
        if !month.is_empty() {
            format!("{year} {month}")
        } else {
            year
        }
    } else {
        String::new()
    }
}

/// Extract author last names.
fn extract_authors(xml: &str) -> Vec<String> {
    let mut authors = Vec::new();
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find("<Author") {
        let author_start = search_from + start;
        let author_end = match xml[author_start..].find("</Author>") {
            Some(p) => author_start + p,
            None => break,
        };
        let author_xml = &xml[author_start..author_end];
        let last = extract_tag(author_xml, "LastName").unwrap_or_default();
        let first = extract_tag(author_xml, "ForeName").unwrap_or_default();
        if !last.is_empty() {
            if !first.is_empty() {
                authors.push(format!("{last} {first}"));
            } else {
                authors.push(last);
            }
        }
        search_from = author_end;
    }
    authors
}

/// Extract MeSH terms.
fn extract_mesh_terms(xml: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find("<DescriptorName") {
        let desc_start = search_from + start;
        let content_start = match xml[desc_start..].find('>') {
            Some(p) => desc_start + p + 1,
            None => break,
        };
        let end = match xml[content_start..].find("</DescriptorName>") {
            Some(p) => content_start + p,
            None => break,
        };
        let term = strip_xml_tags(&xml[content_start..end]);
        if !term.is_empty() {
            terms.push(term);
        }
        search_from = end;
    }
    terms
}

/// Extract reference PMIDs from CommentsCorrections or ReferenceList.
fn extract_references(xml: &str) -> Vec<String> {
    let mut refs = Vec::new();
    let mut search_from = 0;
    while let Some(start) = xml[search_from..].find("<ArticleId IdType=\"pubmed\">") {
        let content_start = search_from + start + 27; // length of the opening tag
        let end = match xml[content_start..].find("</ArticleId>") {
            Some(p) => content_start + p,
            None => break,
        };
        let pmid = xml[content_start..end].trim().to_string();
        if !pmid.is_empty() && pmid.chars().all(|c| c.is_ascii_digit()) {
            refs.push(pmid);
        }
        search_from = end;
    }
    refs
}

/// Strip XML tags from a string, leaving only text content.
fn strip_xml_tags(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut in_tag = false;
    for ch in s.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => result.push(ch),
            _ => {}
        }
    }
    // Collapse whitespace
    result.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Discovery Engine ────────────────────────────────────────────────────

/// Results from analyzing a batch of PubMed articles.
#[derive(Debug)]
pub struct DiscoveryReport {
    /// Total articles ingested
    pub articles_ingested: usize,
    /// Articles rejected (duplicates, too short)
    pub articles_rejected: usize,
    /// Compression stats
    pub compression: CompressionStats,
    /// Emerging topics (high-novelty clusters)
    pub emerging_topics: Vec<EmergingTopic>,
    /// Potential contradictions detected
    pub contradictions: Vec<ContradictionSignal>,
    /// Most connected articles (citation hubs)
    pub citation_hubs: Vec<CitationHub>,
    /// Domain-level statistics
    pub domain_stats: HashMap<String, usize>,
}

/// An emerging topic detected from high-novelty articles.
#[derive(Debug)]
pub struct EmergingTopic {
    pub representative_title: String,
    pub representative_pmid: String,
    pub novelty_score: f32,
    pub mesh_terms: Vec<String>,
    pub article_count: usize,
}

/// A potential contradiction signal between two articles.
#[derive(Debug)]
pub struct ContradictionSignal {
    pub article_a_pmid: String,
    pub article_a_title: String,
    pub article_b_pmid: String,
    pub article_b_title: String,
    /// Cosine similarity (low = potentially contradictory if same topic)
    pub similarity: f64,
    /// Shared MeSH terms (high overlap + low similarity = contradiction signal)
    pub shared_mesh_terms: Vec<String>,
}

/// A citation hub article.
#[derive(Debug)]
pub struct CitationHub {
    pub pmid: String,
    pub title: String,
    pub citation_count: usize,
}

/// Run the discovery pipeline on a set of PubMed articles.
///
/// Processes articles through the web memory ingestion pipeline, then
/// analyzes the results for emerging topics, contradictions, and citation hubs.
pub fn analyze_discoveries(
    articles: &[PubMedArticle],
    embedding_engine: &EmbeddingEngine,
) -> DiscoveryReport {
    let mut graph = KnowledgeGraph::new();
    let existing_hashes = HashSet::new();
    let existing_embeddings: Vec<(Uuid, Vec<f32>)> = Vec::new();

    // Convert articles to CleanedPages
    let pages: Vec<CleanedPage> = articles.iter().map(|a| a.to_cleaned_page()).collect();

    // Run through ingestion pipeline
    let (accepted, response) = web_ingest::ingest_batch(
        &pages,
        "pubmed",
        embedding_engine,
        &mut graph,
        &existing_hashes,
        &existing_embeddings,
    );

    // Map PMIDs to accepted WebMemories for cross-referencing
    let pmid_to_mem: HashMap<String, &WebMemory> = accepted
        .iter()
        .filter_map(|m| {
            let pmid = m.source_url.trim_end_matches('/').rsplit('/').next()?;
            Some((pmid.to_string(), m))
        })
        .collect();

    // ── Emerging Topics: high-novelty articles with MeSH context ────
    let mut emerging_topics = Vec::new();
    let mut high_novelty: Vec<&WebMemory> =
        accepted.iter().filter(|m| m.novelty_score > 0.5).collect();
    high_novelty.sort_by(|a, b| b.novelty_score.partial_cmp(&a.novelty_score).unwrap());

    for mem in high_novelty.iter().take(10) {
        let pmid = mem
            .source_url
            .trim_end_matches('/')
            .rsplit('/')
            .next()
            .unwrap_or("?");
        let article = articles.iter().find(|a| a.pmid == pmid);
        emerging_topics.push(EmergingTopic {
            representative_title: mem.base.title.clone(),
            representative_pmid: pmid.to_string(),
            novelty_score: mem.novelty_score,
            mesh_terms: article.map(|a| a.mesh_terms.clone()).unwrap_or_default(),
            article_count: 1,
        });
    }

    // ── Contradiction Detection: same MeSH terms + low embedding similarity ──
    let mut contradictions = Vec::new();
    let accepted_refs: Vec<(&WebMemory, &PubMedArticle)> = accepted
        .iter()
        .filter_map(|m| {
            let pmid = m.source_url.trim_end_matches('/').rsplit('/').next()?;
            let article = articles.iter().find(|a| a.pmid == pmid)?;
            Some((m, article))
        })
        .collect();

    // Compare pairs that share ≥2 MeSH terms but have low embedding similarity
    for i in 0..accepted_refs.len().min(200) {
        for j in (i + 1)..accepted_refs.len().min(200) {
            let (mem_a, art_a) = &accepted_refs[i];
            let (mem_b, art_b) = &accepted_refs[j];

            let shared_mesh: Vec<String> = art_a
                .mesh_terms
                .iter()
                .filter(|t| art_b.mesh_terms.contains(t))
                .cloned()
                .collect();

            if shared_mesh.len() >= 2 {
                let sim = cosine_similarity(&mem_a.base.embedding, &mem_b.base.embedding);
                // Same topic (shared MeSH) but dissimilar embeddings → contradiction signal
                if sim < 0.4 {
                    contradictions.push(ContradictionSignal {
                        article_a_pmid: art_a.pmid.clone(),
                        article_a_title: art_a.title.clone(),
                        article_b_pmid: art_b.pmid.clone(),
                        article_b_title: art_b.title.clone(),
                        similarity: sim,
                        shared_mesh_terms: shared_mesh,
                    });
                }
            }
        }
    }
    contradictions.sort_by(|a, b| a.similarity.partial_cmp(&b.similarity).unwrap());
    contradictions.truncate(10);

    // ── Citation Hubs: most-referenced articles ─────────────────────
    let mut citation_counts: HashMap<String, usize> = HashMap::new();
    for article in articles {
        for ref_pmid in &article.references {
            *citation_counts.entry(ref_pmid.clone()).or_insert(0) += 1;
        }
    }
    let mut citation_hubs: Vec<CitationHub> = citation_counts
        .into_iter()
        .filter_map(|(pmid, count)| {
            let article = articles.iter().find(|a| a.pmid == pmid)?;
            Some(CitationHub {
                pmid,
                title: article.title.clone(),
                citation_count: count,
            })
        })
        .collect();
    citation_hubs.sort_by(|a, b| b.citation_count.cmp(&a.citation_count));
    citation_hubs.truncate(10);

    // ── Domain stats (journal distribution) ─────────────────────────
    let mut domain_stats: HashMap<String, usize> = HashMap::new();
    for mem in &accepted {
        *domain_stats.entry(mem.domain.clone()).or_insert(0) += 1;
    }

    DiscoveryReport {
        articles_ingested: response.accepted,
        articles_rejected: response.rejected,
        compression: response.compression,
        emerging_topics,
        contradictions,
        citation_hubs,
        domain_stats,
    }
}

// ── Brain Server Integration ────────────────────────────────────────────

/// Push discovered memories to the live π.ruv.io brain server.
pub async fn push_to_brain(
    client: &reqwest::Client,
    articles: &[PubMedArticle],
    embeddings: &HashMap<String, Vec<f32>>,
) -> Result<usize, String> {
    let mut pushed = 0;

    for article in articles {
        if article.abstract_text.len() < 50 {
            continue;
        }

        let embedding = embeddings.get(&article.pmid).cloned().unwrap_or_default();

        let mut tags = article.mesh_terms.clone();
        tags.push(format!("pmid:{}", article.pmid));
        tags.push(format!("journal:{}", article.journal));
        tags.push("source:pubmed".to_string());
        if !article.pub_date.is_empty() {
            tags.push(format!("date:{}", article.pub_date));
        }

        let body = serde_json::json!({
            "category": "pattern",
            "title": article.title,
            "content": article.abstract_text,
            "tags": tags,
            "embedding": embedding,
        });

        let resp = client
            .post(format!("{BRAIN_API}/v1/memories"))
            .json(&body)
            .send()
            .await;

        match resp {
            Ok(r) if r.status().is_success() => pushed += 1,
            Ok(r) => {
                tracing::warn!(
                    "Brain push failed for PMID {}: HTTP {}",
                    article.pmid,
                    r.status()
                );
            }
            Err(e) => {
                tracing::warn!("Brain push failed for PMID {}: {e}", article.pmid);
            }
        }

        // Rate limit: don't hammer the brain server
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }

    Ok(pushed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_pubmed_xml_basic() {
        let xml = r#"
        <PubmedArticleSet>
        <PubmedArticle>
            <MedlineCitation>
                <PMID>12345678</PMID>
                <Article>
                    <ArticleTitle>Effect of drug X on disease Y</ArticleTitle>
                    <Abstract>
                        <AbstractText Label="BACKGROUND">Background info here.</AbstractText>
                        <AbstractText Label="RESULTS">Results found that drug X works.</AbstractText>
                        <AbstractText Label="CONCLUSIONS">Drug X is effective.</AbstractText>
                    </Abstract>
                    <AuthorList>
                        <Author>
                            <LastName>Smith</LastName>
                            <ForeName>John</ForeName>
                        </Author>
                        <Author>
                            <LastName>Jones</LastName>
                            <ForeName>Alice</ForeName>
                        </Author>
                    </AuthorList>
                    <Journal>
                        <Title>Nature Medicine</Title>
                    </Journal>
                </Article>
                <MeshHeadingList>
                    <MeshHeading>
                        <DescriptorName>Drug Therapy</DescriptorName>
                    </MeshHeading>
                    <MeshHeading>
                        <DescriptorName>Clinical Trials</DescriptorName>
                    </MeshHeading>
                </MeshHeadingList>
            </MedlineCitation>
            <PubmedData>
                <ArticleIdList>
                    <ArticleId IdType="pubmed">12345678</ArticleId>
                </ArticleIdList>
                <ReferenceList>
                    <Reference>
                        <ArticleIdList>
                            <ArticleId IdType="pubmed">11111111</ArticleId>
                        </ArticleIdList>
                    </Reference>
                </ReferenceList>
            </PubmedData>
        </PubmedArticle>
        </PubmedArticleSet>
        "#;

        let articles = parse_pubmed_xml(xml);
        assert_eq!(articles.len(), 1);

        let a = &articles[0];
        assert_eq!(a.pmid, "12345678");
        assert_eq!(a.title, "Effect of drug X on disease Y");
        assert!(a.abstract_text.contains("Background info here"));
        assert!(a.abstract_text.contains("RESULTS: Results found"));
        assert!(a.abstract_text.contains("Drug X is effective"));
        assert_eq!(a.authors, vec!["Smith John", "Jones Alice"]);
        assert_eq!(a.journal, "Nature Medicine");
        assert_eq!(a.mesh_terms, vec!["Drug Therapy", "Clinical Trials"]);
        assert!(a.references.contains(&"11111111".to_string()));
    }

    #[test]
    fn article_to_cleaned_page() {
        let article = PubMedArticle {
            pmid: "99999999".into(),
            title: "Test Article Title".into(),
            abstract_text: "This is the abstract text.".into(),
            authors: vec!["Author A".into()],
            journal: "Test Journal".into(),
            pub_date: "2026".into(),
            mesh_terms: vec!["Term1".into(), "Term2".into()],
            references: vec!["11111111".into()],
        };

        let page = article.to_cleaned_page();
        assert_eq!(page.url, "https://pubmed.ncbi.nlm.nih.gov/99999999/");
        assert!(page.text.contains("Test Article Title"));
        assert!(page.text.contains("This is the abstract"));
        assert_eq!(page.links.len(), 1);
        assert!(page.tags.contains(&"Term1".to_string()));
        assert!(page.tags.contains(&"journal:Test Journal".to_string()));
    }

    #[test]
    fn strip_xml_tags_basic() {
        assert_eq!(strip_xml_tags("hello <b>world</b>"), "hello world");
        assert_eq!(strip_xml_tags("<p>text</p>"), "text");
        assert_eq!(strip_xml_tags("no tags"), "no tags");
        assert_eq!(strip_xml_tags("  multiple   spaces  "), "multiple spaces");
    }

    #[test]
    fn extract_tag_with_attributes() {
        let xml = r#"<PMID Version="1">12345</PMID>"#;
        assert_eq!(extract_tag(xml, "PMID"), Some("12345".to_string()));
    }

    #[test]
    fn extract_abstract_multi_segment() {
        let xml = r#"
            <AbstractText Label="BACKGROUND">Background.</AbstractText>
            <AbstractText Label="METHODS">Methods.</AbstractText>
            <AbstractText>Unlabeled.</AbstractText>
        "#;
        let result = extract_abstract(xml);
        assert!(result.contains("BACKGROUND: Background."));
        assert!(result.contains("METHODS: Methods."));
        assert!(result.contains("Unlabeled."));
    }
}
