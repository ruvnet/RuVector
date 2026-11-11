//! Lane assignment: one active lane per repo, independent items batched.

use crate::model::{Lane, LaneStatus, WorkItem};
use std::collections::{BTreeMap, BTreeSet};

/// One lane per repo; the top item is `active`, independent items (distinct
/// candidate, no shared evidence file, public disclosure) ride along as
/// `batched`. Repos with an in-flight ledger lane stay deferred.
pub fn assign_lanes(
    items: &mut [WorkItem],
    busy: &BTreeMap<String, Vec<String>>,
    max_lanes: usize,
    max_batch: usize,
) -> Vec<Lane> {
    let mut lanes: Vec<Lane> = Vec::new();
    let mut lane_of: BTreeMap<String, usize> = BTreeMap::new();
    let mut files: Vec<BTreeSet<String>> = Vec::new();
    let mut cands: Vec<BTreeSet<String>> = Vec::new();
    for it in items.iter_mut() {
        let repo = it.repo.to_ascii_lowercase();
        it.status = LaneStatus::Deferred;
        if busy.contains_key(&repo) {
            continue;
        }
        let my_files: BTreeSet<String> = it.evidence.iter().map(|s| s.file.clone()).collect();
        let my_cand = it
            .candidate
            .clone()
            .or(it.subject.clone())
            .unwrap_or_default();
        match lane_of.get(&repo).copied() {
            None if lanes.len() < max_lanes => {
                lane_of.insert(repo.clone(), lanes.len());
                lanes.push(Lane {
                    repo,
                    items: vec![it.id.clone()],
                });
                files.push(my_files);
                cands.push(BTreeSet::from([my_cand]));
                it.status = LaneStatus::Active;
            }
            Some(l) => {
                let lead_private = items_private(&lanes[l]);
                let independent = it.disclosure == "public_pr"
                    && !lead_private
                    && !cands[l].contains(&my_cand)
                    && files[l].is_disjoint(&my_files);
                if independent && lanes[l].items.len() < max_batch {
                    lanes[l].items.push(it.id.clone());
                    files[l].extend(my_files);
                    cands[l].insert(my_cand);
                    it.status = LaneStatus::Batched;
                }
            }
            None => {}
        }
    }
    lanes
}

fn items_private(l: &Lane) -> bool {
    l.items
        .first()
        .map(|id| id.starts_with("security::"))
        .unwrap_or(false)
}
