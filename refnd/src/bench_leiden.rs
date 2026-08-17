use std::time::Instant;
use mimalloc::MiMalloc;
use rustc_hash::FxHashMap;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

use refnd::core::EdgeStore;
use refnd::core::leiden::{find_communities, fast_find_communities, gve_find_communities, CsrGraph, INWeightType, LeidenObjective};

/// Usage: bench_leiden <file.edgestr> [modularity|cpm] [gamma] [beta] [iterations] [seq|par|gve]
///
/// `gamma` is the resolution parameter (modularity resolution, or CPM resolution
/// directly). `iterations` is the max number of top-level Leiden restarts passed
/// to `find_communities` (0 = run until the partition stops changing). The final
/// arg picks the implementation: sequential (`leiden::find_communities`), parallel
/// (`leidenp::fast_find_communities`), or the GVE-Leiden port
/// (`gveleiden::gve_find_communities`, ignores `beta` -- its refinement is
/// deterministic); defaults to `seq`.
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: bench_leiden <file.edgestr> [modularity|cpm] [gamma] [beta] [iterations] [seq|par|gve]");
    let objective = match args.next().as_deref() {
        None | Some("modularity") => LeidenObjective::Modularity,
        Some("cpm") => LeidenObjective::CPM,
        Some(other) => panic!("unknown objective {other:?}: expected 'modularity' or 'cpm'"),
    };
    let gamma: f32 = args.next().map(|s| s.parse().expect("gamma must be a f32")).unwrap_or(1.0);
    let beta: f64 = args.next().map(|s| s.parse().expect("beta must be a f64")).unwrap_or(0.01);
    let n_iterations: usize = args.next().map(|s| s.parse().expect("iterations must be a usize")).unwrap_or(2);
    let impl_name = match args.next().as_deref() {
        None | Some("seq") => "seq",
        Some("par") => "par",
        Some("gve") => "gve",
        Some(other) => panic!("unknown implementation {other:?}: expected 'seq', 'par', or 'gve'"),
    };

    eprint!("Loading {path} ... ");
    let t = Instant::now();
    let edges = EdgeStore::load(&path).expect("failed to load edgestr");
    eprintln!("done in {:.2}s  ({} nodes, {} edges)", t.elapsed().as_secs_f64(), edges.node_count, edges.len());

    eprint!("Building CsrGraph ... ");
    let t = Instant::now();
    let graph = edges.graph(INWeightType::SimilarityComplement);
    eprintln!("done in {:.2}s", t.elapsed().as_secs_f64());

    eprintln!("Running Leiden ({impl_name}, {objective:?}, γ={gamma}, β={beta}, iterations={n_iterations}) ...");
    let t = Instant::now();
    let stats_graph = graph.clone();
    let membership = match impl_name {
        "par" => fast_find_communities(graph, gamma, beta, n_iterations, objective.clone()),
        "gve" => gve_find_communities(graph, gamma, n_iterations, objective.clone()),
        _ => find_communities(graph, gamma, beta, n_iterations, objective.clone()),
    };
    let elapsed = t.elapsed().as_secs_f64();
    eprintln!("Leiden finished in {elapsed:.3}s");

    eprint!("Computing quality stats ... ");
    let t = Instant::now();
    report_stats(&stats_graph, &membership, gamma, &objective);
    eprintln!("  (computed in {:.2}s)", t.elapsed().as_secs_f64());
}

/// Reports community count/size distribution and the exact quality function Leiden
/// optimizes -- H = Σ_c [e_c - (resolution/2) * K_c²], the same gain formula used
/// in `fastmove_nodes`/`merge_nodes` -- plus the fraction of edge weight that
/// crosses community boundaries. Meant to catch quality regressions when comparing
/// against a parallel implementation, not just wall-clock time.
fn report_stats(graph: &CsrGraph, membership: &[usize], gamma: f32, objective: &LeidenObjective) {
    let n = graph.n;

    // Community ids returned by find_communities aren't guaranteed dense/contiguous.
    let mut sorted_ids = membership.to_vec();
    sorted_ids.sort_unstable();
    sorted_ids.dedup();
    let n_communities = sorted_ids.len();
    let id_to_idx: FxHashMap<usize, usize> = sorted_ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let compact: Vec<usize> = membership.iter().map(|&id| id_to_idx[&id]).collect();

    // node_weight / resolution mirror find_communities exactly, since that's what
    // the algorithm actually optimized.
    let node_weight: Vec<f32> = match objective {
        LeidenObjective::Modularity => (0..n).map(|v| graph.strength(v)).collect(),
        LeidenObjective::CPM => vec![1.0f32; n],
    };
    let resolution: f32 = match objective {
        LeidenObjective::Modularity => gamma / node_weight.iter().sum::<f32>(),
        LeidenObjective::CPM => gamma,
    };

    let mut size: Vec<u64> = vec![0; n_communities];
    let mut k_c: Vec<f64> = vec![0.0; n_communities];
    for v in 0..n {
        let c = compact[v];
        size[c] += 1;
        k_c[c] += node_weight[v] as f64;
    }

    // e_c: internal edge weight per community, and total edge weight. Each
    // undirected edge sits on both endpoints' adjacency lists (self-loops on one),
    // so `u >= v` counts it once. NOTE: computed from `adj`, not `graph.m` --
    // `graph.m` is summed from the *raw* pre-transform edge weights in
    // `CsrGraph::new`, not the transformed weights actually stored in `adj` /
    // used by `strength()`/the algorithm, so it disagrees with everything else
    // (harmless today since nothing in `leiden.rs` reads `graph.m`, but worth
    // knowing if that ever changes).
    let mut e_c: Vec<f64> = vec![0.0; n_communities];
    let mut total_weight: f64 = 0.0;
    for v in 0..n {
        let c = compact[v];
        for &(u, w) in graph.neighbors(v) {
            let u = u as usize;
            if u >= v {
                let w = w as f64;
                total_weight += w;
                if compact[u] == c { e_c[c] += w; }
            }
        }
    }

    let internal_weight: f64 = e_c.iter().sum();
    let quality: f64 = e_c.iter().zip(&k_c)
        .map(|(&e, &k)| e - 0.5 * resolution as f64 * k * k)
        .sum();

    let mut sorted_sizes = size;
    sorted_sizes.sort_unstable();
    let pct = |p: f64| sorted_sizes[((sorted_sizes.len() - 1) as f64 * p).round() as usize];
    let singletons = sorted_sizes.iter().filter(|&&s| s == 1).count();

    eprintln!("Communities: {n_communities}");
    eprintln!(
        "  sizes: min={} p50={} p90={} p99={} max={} mean={:.1}",
        sorted_sizes[0], pct(0.5), pct(0.9), pct(0.99), sorted_sizes[n_communities - 1],
        n as f64 / n_communities as f64,
    );
    eprintln!("  singletons: {singletons} ({:.2}% of communities)", 100.0 * singletons as f64 / n_communities as f64);
    eprintln!(
        "  edge weight: {:.2}% internal, {:.2}% cross-community ({:.1} / {:.1} total)",
        100.0 * internal_weight / total_weight,
        100.0 * (1.0 - internal_weight / total_weight),
        internal_weight, total_weight,
    );
    eprintln!("  objective ({objective:?}, γ={gamma}): H = {quality:.4}");
}
