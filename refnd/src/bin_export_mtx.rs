use std::io::{BufWriter, Write};
use refnd::core::EdgeStore;
use refnd::core::leiden::INWeightType;

/// One-off diagnostic exporter: dumps a .edgestr graph (as CsrGraph sees it,
/// i.e. post SimilarityComplement transform, matching bench_leiden.rs
/// exactly) to Matrix Market format for feeding into the reference
/// GVE-Leiden C++ implementation. Not meant to be a permanent tool.
fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: bin_export_mtx <file.edgestr> <out.mtx>");
    let out_path = args.next().expect("usage: bin_export_mtx <file.edgestr> <out.mtx>");

    eprint!("Loading {path} ... ");
    let edges = EdgeStore::load(&path).expect("failed to load edgestr");
    eprintln!("done ({} nodes, {} edges)", edges.node_count, edges.len());

    eprint!("Building CsrGraph ... ");
    let graph = edges.graph(INWeightType::SimilarityComplement);
    eprintln!("done");

    let total_directed: usize = (0..graph.n).map(|v| graph.neighbors(v).len()).sum();
    eprintln!("Writing {out_path} ({total_directed} directed entries) ...");

    let f = std::fs::File::create(&out_path).expect("failed to create output file");
    let mut w = BufWriter::with_capacity(64 * 1024 * 1024, f);
    writeln!(w, "%%MatrixMarket matrix coordinate real general").unwrap();
    writeln!(w, "{} {} {}", graph.n, graph.n, total_directed).unwrap();
    for v in 0..graph.n {
        for &(u, wt) in graph.neighbors(v) {
            writeln!(w, "{} {} {}", v + 1, u + 1, wt).unwrap();
        }
    }
    w.flush().unwrap();
    eprintln!("done");
}
