//! Wikipedia corpus reader.
//!
//! Assumes the corpus has already been extracted to a directory of plain-text
//! shards by `scripts/fetch-simple-wiki.sh`. We do NOT do XML parsing or
//! bzip2 decoding in v1 — that is the fetch script's job.
//!
//! Shard format: one paragraph per line, blank lines separate articles.
//! Files match the glob `shard-*.txt` inside `corpus_dir`.

use super::DataError;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

const MIN_ARTICLE_LEN: usize = 50;

/// Wiki corpus rooted at a directory of `shard-*.txt` files.
pub struct WikiCorpus {
    corpus_dir: PathBuf,
    shards: Vec<PathBuf>,
}

impl WikiCorpus {
    /// Open a corpus by scanning `corpus_dir` for `shard-*.txt` files.
    pub fn new(corpus_dir: PathBuf) -> Result<Self, DataError> {
        if !corpus_dir.is_dir() {
            return Err(DataError::Corpus(format!(
                "corpus dir does not exist: {}",
                corpus_dir.display()
            )));
        }

        let mut shards: Vec<PathBuf> = fs::read_dir(&corpus_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.is_file()
                    && p.file_name()
                        .and_then(|n| n.to_str())
                        .map(|n| n.starts_with("shard-") && n.ends_with(".txt"))
                        .unwrap_or(false)
            })
            .collect();
        shards.sort();

        if shards.is_empty() {
            return Err(DataError::Corpus(format!(
                "no shard-*.txt files found in {}",
                corpus_dir.display()
            )));
        }

        Ok(Self { corpus_dir, shards })
    }

    /// Path the corpus was opened from.
    pub fn path(&self) -> &Path {
        &self.corpus_dir
    }

    /// Number of shards discovered.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Streaming iterator over articles across all shards.
    ///
    /// An "article" is the run of non-empty lines between blank-line separators.
    /// Stub articles (< 50 chars) are filtered out.
    pub fn iter_articles(&self) -> WikiArticleIter {
        WikiArticleIter::new(self.shards.clone())
    }

    /// Count articles by scanning all shards. O(n) over total bytes.
    pub fn article_count(&self) -> Result<usize, DataError> {
        Ok(self.iter_articles().count())
    }
}

/// Streaming article iterator. Yields cleaned article text strings.
pub struct WikiArticleIter {
    shards: std::vec::IntoIter<PathBuf>,
    current: Option<BufReader<fs::File>>,
    buf: String,
}

impl WikiArticleIter {
    fn new(shards: Vec<PathBuf>) -> Self {
        Self {
            shards: shards.into_iter(),
            current: None,
            buf: String::new(),
        }
    }

    fn open_next_shard(&mut self) -> Result<bool, DataError> {
        match self.shards.next() {
            Some(path) => {
                let f = fs::File::open(&path)?;
                self.current = Some(BufReader::new(f));
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn read_one_article(&mut self) -> Result<Option<String>, DataError> {
        loop {
            // Open a shard if we don't have one.
            if self.current.is_none() && !self.open_next_shard()? {
                return Ok(None);
            }

            self.buf.clear();
            let reader = self.current.as_mut().unwrap();
            let mut line = String::new();
            let mut saw_content = false;

            loop {
                line.clear();
                let n = reader.read_line(&mut line)?;
                if n == 0 {
                    // EOF on this shard.
                    self.current = None;
                    break;
                }
                let trimmed = line.trim();
                if trimmed.is_empty() {
                    if saw_content {
                        // End of article.
                        break;
                    }
                    // Otherwise: still consuming leading blank lines.
                    continue;
                }
                if saw_content {
                    self.buf.push(' ');
                }
                self.buf.push_str(trimmed);
                saw_content = true;
            }

            if saw_content {
                let cleaned = clean_article(&self.buf);
                if cleaned.len() >= MIN_ARTICLE_LEN {
                    return Ok(Some(cleaned));
                }
                // Else: drop stub, loop to try the next article.
            }
            // If !saw_content here, we need to advance to next shard (current=None set above).
        }
    }
}

impl Iterator for WikiArticleIter {
    type Item = String;

    fn next(&mut self) -> Option<String> {
        match self.read_one_article() {
            Ok(opt) => opt,
            Err(_) => None,
        }
    }
}

/// Collapse whitespace runs into single spaces, trim ends.
fn clean_article(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut prev_space = false;
    for c in raw.chars() {
        if c.is_whitespace() {
            if !prev_space && !out.is_empty() {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    fn write_shard(dir: &Path, name: &str, content: &str) {
        let mut f = fs::File::create(dir.join(name)).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn test_open_corpus() {
        let tmp = TempDir::new().unwrap();
        write_shard(
            tmp.path(),
            "shard-0001.txt",
            "Article one is sufficiently long to pass the stub filter easily.\n\nArticle two also has enough characters to be retained as content.\n",
        );

        let corpus = WikiCorpus::new(tmp.path().to_path_buf()).unwrap();
        assert_eq!(corpus.shard_count(), 1);
        let articles: Vec<_> = corpus.iter_articles().collect();
        assert_eq!(articles.len(), 2);
    }

    #[test]
    fn test_stub_filtering() {
        let tmp = TempDir::new().unwrap();
        write_shard(
            tmp.path(),
            "shard-0001.txt",
            "tiny\n\nThis article is long enough to survive the stub filter easily.\n",
        );
        let corpus = WikiCorpus::new(tmp.path().to_path_buf()).unwrap();
        let articles: Vec<_> = corpus.iter_articles().collect();
        assert_eq!(articles.len(), 1);
    }
}
