use rand::prelude::*;
use rustc_hash::FxHashSet;
use crate::core::leiden::CsrGraph;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn partition_dataset(clusters: Vec<usize>, graph: &CsrGraph, test_ratio: f32, seed: Option<usize>, post_filtering: bool) -> Result<(Vec<usize>, Vec<usize>), &'static str>{
    let mut rng = if let Some(s) = seed {
        StdRng::seed_from_u64(s as u64)
    } else {
        let ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;
        StdRng::seed_from_u64(ms)
    };
    let &num_clusters = clusters.iter().max().ok_or("Found no clusters because the vector is empty")?;
    let mut all_clusters: Vec<usize> = (0..num_clusters).collect();
    all_clusters.shuffle(&mut rng);

    let num_test = (test_ratio * num_clusters as f32) as usize;
    let test_clusters = FxHashSet::from_iter(all_clusters[0..num_test].iter().copied());

    let mut test_ids = FxHashSet::default();
    let mut train_ids = Vec::with_capacity(clusters.len() - num_test);
    for (v, cluster) in clusters.iter().enumerate() {
        if test_clusters.contains(cluster) {
            test_ids.insert(v);
        } else {
            let is_test_connected = graph.neighbors(v)
                .iter()
                .any(|&(u, _)| test_clusters.contains(&clusters[u]));
            if post_filtering && is_test_connected {
                // filtered out — skip
            } else {
                train_ids.push(v);
            }
        }
    }

    Ok((train_ids, test_ids.iter().copied().collect()))
}