//! Streaming query results using AsyncIterator pattern

use crate::types::*;
use futures::stream::Stream;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Streaming query result iterator
#[napi]
pub struct QueryResultStream {
    inner: Pin<Box<dyn Stream<Item = JsQueryResult> + Send>>,
}

impl QueryResultStream {
    /// Create a new query result stream
    pub fn new(stream: Pin<Box<dyn Stream<Item = JsQueryResult> + Send>>) -> Self {
        Self { inner: stream }
    }
}

#[napi]
impl QueryResultStream {
    /// Get the next result from the stream.
    ///
    /// **Not implemented.** This always returns `null`, and no method on
    /// `GraphDatabase` returns a `QueryResultStream` in the first place — there
    /// is no `db.queryStream()`. The previous doc comment here showed exactly
    /// that call, which does not exist. Use `db.query()` until streaming is
    /// built; this type is exported only to keep the shape reserved.
    #[napi]
    pub fn next(&mut self) -> Result<Option<JsQueryResult>> {
        // This would poll the stream in a real implementation
        Ok(None)
    }
}

/// Streaming hyperedge result iterator
#[napi]
pub struct HyperedgeStream {
    results: Vec<JsHyperedgeResult>,
    index: usize,
}

impl HyperedgeStream {
    /// Create a new hyperedge stream
    pub fn new(results: Vec<JsHyperedgeResult>) -> Self {
        Self { results, index: 0 }
    }
}

#[napi]
impl HyperedgeStream {
    /// Get the next hyperedge result.
    ///
    /// Note: no method on `GraphDatabase` returns a `HyperedgeStream` — there is
    /// no `db.searchHyperedgesStream()`, which the previous doc example here
    /// claimed. Use `db.searchHyperedges()`, which returns the full array.
    #[napi]
    pub fn next(&mut self) -> Result<Option<JsHyperedgeResult>> {
        if self.index < self.results.len() {
            let result = self.results[self.index].clone();
            self.index += 1;
            Ok(Some(result))
        } else {
            Ok(None)
        }
    }

    /// Collect all remaining results
    #[napi]
    pub fn collect(&mut self) -> Vec<JsHyperedgeResult> {
        let remaining = self.results[self.index..].to_vec();
        self.index = self.results.len();
        remaining
    }
}

/// Node stream iterator
#[napi]
pub struct NodeStream {
    nodes: Vec<JsNode>,
    index: usize,
}

impl NodeStream {
    /// Create a new node stream
    pub fn new(nodes: Vec<JsNode>) -> Self {
        Self { nodes, index: 0 }
    }
}

#[napi]
impl NodeStream {
    /// Get the next node
    #[napi]
    pub fn next(&mut self) -> Result<Option<JsNode>> {
        if self.index < self.nodes.len() {
            let node = self.nodes[self.index].clone();
            self.index += 1;
            Ok(Some(node))
        } else {
            Ok(None)
        }
    }

    /// Collect all remaining nodes
    #[napi]
    pub fn collect(&mut self) -> Vec<JsNode> {
        let remaining = self.nodes[self.index..].to_vec();
        self.index = self.nodes.len();
        remaining
    }
}
