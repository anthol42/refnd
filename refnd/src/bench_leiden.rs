use std::time::Instant;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use refnd::core::EdgeStore;
use refnd::core::leiden::{find_communities, INWeightType, LeidenObjective};

fn main() {
    let path = std::env::args().nth(1).expect("usage: bench_leiden <file.edgestr>");

    eprint!("Loading {path} ... ");
    let t = Instant::now();
    let edges = EdgeStore::load(&path).expect("failed to load edgestr");
    eprintln!("done in {:.2}s  ({} nodes, {} edges)", t.elapsed().as_secs_f64(), edges.node_count, edges.len());

    eprint!("Building CsrGraph ... ");
    let t = Instant::now();
    let graph = edges.graph(INWeightType::SimilarityComplement);
    eprintln!("done in {:.2}s", t.elapsed().as_secs_f64());

    eprintln!("Running Leiden (Modularity, γ=1.0) ...");
    let t = Instant::now();
    let membership = find_communities(graph, 1.0, 0.01, 2, LeidenObjective::Modularity);
    let elapsed = t.elapsed().as_secs_f64();

    let n_communities = {
        let mut ids = membership.clone();
        ids.sort_unstable();
        ids.dedup();
        ids.len()
    };
    eprintln!("Leiden finished in {elapsed:.3}s  →  {n_communities} communities");
}
