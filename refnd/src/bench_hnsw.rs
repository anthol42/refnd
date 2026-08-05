//! Full BELKA-parameter HNSW pipeline benchmark, entirely in Rust: reads a packed
//! fingerprint cache written by `bench_fingerprints` (256 bytes/fingerprint, no
//! header), builds the `HNSWState`, and runs `build()` -- timing and RSS-tracking
//! each phase separately, the same way `belka.py` does on the Python side. This is
//! the "before" baseline for the ongoing dev-branch reimplementation of the
//! HNSW-scaling-investigation fixes: run this at a given N before and after each
//! fix lands to measure its actual impact on this codebase, rather than trusting
//! the numbers from the other branch's implementation.
//!
//! Uses BELKA's dataset config: proximity_threshold=0.4, keep_all_edges=false,
//! cache_capacity=0, strict_ef=true (see refnd-paper's src/datasets.py DATASETS["belka"]
//! and belka.py) -- everything else is HNSWConfig::default().
//!
//! Usage: bench_hnsw <fp_cache.bin> [n_limit]

use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::time::Instant;

use mimalloc::MiMalloc;

use fixedbitset::FixedBitSet;
use refnd::core::hnsw::{HNSWConfig, HNSWState};
use refnd::kernels::molecules::tanimoto::Tanimoto;
use refnd::utils::BitFingerprint;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FP_BITS: usize = 2048;
const FP_BYTES: usize = FP_BITS / 8; // 256
const WORDS_PER_FP: usize = FP_BITS / (usize::BITS as usize); // 32 on 64-bit

fn rss_gb() -> f64 {
    proc_status_field("VmRSS:")
}

fn peak_rss_gb() -> f64 {
    proc_status_field("VmHWM:")
}

fn proc_status_field(prefix: &str) -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(prefix) {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / (1024.0 * 1024.0);
        }
    }
    0.0
}

/// Reads packed fingerprints (as written by `bench_fingerprints`: 256 bytes each,
/// `Morgan::fingerprint(..).words()` little-endian, no header) straight into
/// `FixedBitSet`'s block storage -- one `read_exact` + one `with_capacity_and_blocks`
/// per record, no bit-by-bit unpacking.
fn load_fingerprints(path: &str, n_limit: Option<usize>) -> Vec<BitFingerprint> {
    let file_len = std::fs::metadata(path).expect("failed to stat fp cache").len() as usize;
    let n_in_file = file_len / FP_BYTES;
    let n = n_limit.map(|l| l.min(n_in_file)).unwrap_or(n_in_file);

    let mut reader = BufReader::new(File::open(path).expect("failed to open fp cache"));
    let mut out = Vec::with_capacity(n);
    let mut buf = [0u8; FP_BYTES];
    for _ in 0..n {
        reader.read_exact(&mut buf).expect("failed to read fp record");
        let mut words = [0usize; WORDS_PER_FP];
        for (i, w) in words.iter_mut().enumerate() {
            *w = usize::from_le_bytes(buf[i * 8..i * 8 + 8].try_into().unwrap());
        }
        let bits = FixedBitSet::with_capacity_and_blocks(FP_BITS, words);
        out.push(BitFingerprint::new(bits));
    }
    out
}

fn main() {
    let mut args = env::args().skip(1);
    let fp_path = args.next().expect("usage: bench_hnsw <fp_cache.bin> [n_limit]");
    let n_limit: Option<usize> = args.next().and_then(|s| s.parse().ok());

    eprintln!("=== bench_hnsw (BELKA params) ===");
    eprintln!("fp cache: {fp_path}");
    if let Some(n) = n_limit {
        eprintln!("limit:    first {n}");
    }
    eprintln!("RSS at start: {:.3} GB", rss_gb());

    let t0 = Instant::now();
    let data = load_fingerprints(&fp_path, n_limit);
    eprintln!(
        "Loaded {} fingerprints in {:.1}s",
        data.len(),
        t0.elapsed().as_secs_f64()
    );
    eprintln!("RSS after load: {:.3} GB", rss_gb());

    let mut config = HNSWConfig::default();
    config.proximity_threshold = 0.4;
    config.keep_all_edges = false;
    config.cache_capacity = 0;
    config.strict_ef = true;
    eprintln!("config: {config:?}");

    let t0 = Instant::now();
    let mut state = HNSWState::new(data, Tanimoto, config);
    eprintln!("Construction took {:.3}s", t0.elapsed().as_secs_f64());
    eprintln!("RSS after construction: {:.3} GB", rss_gb());

    let t0 = Instant::now();
    state.build(None).expect("build() failed");
    eprintln!("build() took {:.1}s", t0.elapsed().as_secs_f64());
    eprintln!("RSS after build(): {:.3} GB", rss_gb());

    eprintln!("Peak RSS (VmHWM): {:.3} GB", peak_rss_gb());
}
