//! Variant 3 — CommunityRAG: community centroid routing + member rerank.
//!
//! At query time:
//! 1. Find the nearest community centroid (O(C) where C = community count).
//! 2. Retrieve all vectors assigned to that community.
//! 3. Re-score by exact L2 distance to query.
//! 4. Return top-k.
//!
//! This variant excels when queries are coherent with a task cluster —
//! e.g., an agent asking about "Python debugging" retrieves all Python-related
//! memories rather than a mixed bag of generically similar embeddings.
//!
//! **Tradeoff**: cross-community recall may be lower than FlatScan or GraphHop
//! when the query genuinely spans multiple communities. A future extension is
//! to query the top-C communities and merge results — but that is left as a
//! production refinement (see ADR-272).

use crate::{community::Communities, CommunitySearch, Hit, VectorMeta, l2_sq};

pub struct CommunityRAG {
    vectors: Vec<Vec<f32>>,
    metas: Vec<VectorMeta>,
    /// Detected community partition (None until `build()` is called).
    communities: Option<Communities>,
    /// Cosine similarity threshold for the community graph.
    edge_threshold: f32,
    /// Members of each community, indexed by community label.
    members: Vec<Vec<usize>>,
}

impl CommunityRAG {
    pub fn new(edge_threshold: f32) -> Self {
        Self {
            vectors: Vec::new(),
            metas: Vec::new(),
            communities: None,
            edge_threshold,
            members: Vec::new(),
        }
    }

    /// Read-only access to the detected community labels (after build).
    pub fn detected_labels(&self) -> Option<&[usize]> {
        self.communities.as_ref().map(|c| c.labels.as_slice())
    }

    pub fn num_communities(&self) -> usize {
        self.communities.as_ref().map(|c| c.num_communities).unwrap_or(0)
    }
}

impl CommunitySearch for CommunityRAG {
    fn insert(&mut self, vector: &[f32], community: usize) {
        let id = self.vectors.len();
        self.metas.push(VectorMeta { id, community });
        self.vectors.push(vector.to_vec());
    }

    fn build(&mut self) {
        let comm = Communities::detect(&self.vectors, self.edge_threshold);
        let nc = comm.num_communities;

        // Index members per community.
        let mut members = vec![Vec::new(); nc];
        for (i, &lab) in comm.labels.iter().enumerate() {
            members[lab].push(i);
        }

        self.members = members;
        self.communities = Some(comm);
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let comm = match &self.communities {
            Some(c) => c,
            None => return vec![],
        };

        // Route to the nearest community.
        let (nearest_comm, _dist) = comm.nearest_community(query);
        let candidates = &self.members[nearest_comm];

        // Re-score all community members by exact L2.
        let mut hits: Vec<Hit> = candidates
            .iter()
            .map(|&id| Hit { id, distance: l2_sq(query, &self.vectors[id]) })
            .collect();
        hits.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        hits.truncate(k);
        hits
    }

    fn memory_bytes(&self) -> usize {
        let vec_bytes: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let meta_bytes = self.metas.len() * std::mem::size_of::<VectorMeta>();
        let member_bytes: usize = self.members.iter().map(|m| m.len() * 8).sum();
        let centroid_bytes = self
            .communities
            .as_ref()
            .map(|c| c.centroids.iter().map(|cv| cv.len() * 4).sum::<usize>())
            .unwrap_or(0);
        vec_bytes + meta_bytes + member_bytes + centroid_bytes
    }

    fn name(&self) -> &'static str {
        "CommunityRAG"
    }
}
