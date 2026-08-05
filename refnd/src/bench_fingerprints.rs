//! Rust-native reimplementation of refnd-paper's `src/fingerprints.py
//! compute_and_cache_fingerprints_to_disk`: read SMILES from a text file (one per
//! line), compute Morgan(radius=2, 2048 bits) fingerprints in parallel, and write
//! them packed (256 bytes each, no header) straight to disk.
//!
//! Molecules are processed in fixed-size chunks (parallelized within a chunk via
//! rayon, chunks handled sequentially) instead of collecting every fingerprint into
//! one in-memory Vec/list first — the same rationale as the Python pipeline's
//! streaming design, but achieved here with plain OS threads instead of a
//! ProcessPoolExecutor: fingerprinting doesn't need Python's GIL workaround, so
//! there's no per-item (de)serialization cost crossing a process boundary.
//!
//! Usage: bench_fingerprints <smiles.txt> <out.bin> [n_limit]

use std::env;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::time::Instant;

use mimalloc::MiMalloc;
use rayon::prelude::*;

use molprint_core::smiles::parse_smiles;
use molprint_fp::{morgan::Morgan, traits::Fingerprinter};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

const FP_BITS: usize = 2048;
const FP_BYTES: usize = FP_BITS / 8; // 256
const CHUNK_SIZE: usize = 65_536;

/// Peak resident set size ever reached by this process, read from
/// `/proc/self/status`'s `VmHWM` line. Cheaper than polling RSS ourselves and
/// immune to sampling gaps between polls.
fn peak_rss_gb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: f64 = rest.trim().trim_end_matches("kB").trim().parse().unwrap_or(0.0);
            return kb / (1024.0 * 1024.0);
        }
    }
    0.0
}

/// Fingerprint one SMILES and pack it into 256 bytes. `None` if RDKit's
/// molprint-core equivalent can't parse it.
/// BELKA SMILES mark the DNA-barcode attachment point with a dummy `[Dy]` atom
/// (dysprosium has no real chemical role here -- it's just an inert tag, chosen
/// because it never occurs in the actual screened molecules). RDKit's full
/// periodic table parses it as a real atom; molprint-core's element table is
/// scoped to common organic/druglike elements and doesn't include the
/// lanthanide series, so `[Dy]` is stripped before parsing. This means the
/// resulting fingerprint's radius-2 environment near the tag differs slightly
/// from RDKit's (which keeps Dy as a real graph node) -- acceptable for a
/// throughput/memory benchmark, but this pipeline is not yet a bit-for-bit
/// drop-in replacement for the RDKit-based one.
fn pack_fp(smiles: &str) -> Option<[u8; FP_BYTES]> {
    let cleaned = smiles.replace("[Dy]", "");
    let mol = parse_smiles(&cleaned).ok()?;
    let fp = Morgan::new(2, FP_BITS).fingerprint(&mol);
    let mut bytes = [0u8; FP_BYTES];
    for (i, word) in fp.words().iter().enumerate() {
        bytes[i * 8..i * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    Some(bytes)
}

/// Fingerprints `chunk` in parallel, writes successful ones to `writer` in
/// input order, and returns (n_written, n_missing). `chunk` is cleared on return.
fn flush_chunk(chunk: &mut Vec<String>, writer: &mut BufWriter<File>) -> (u64, u64) {
    let packed: Vec<Option<[u8; FP_BYTES]>> = chunk.par_iter().map(|s| pack_fp(s)).collect();
    let mut n_written = 0u64;
    let mut n_missing = 0u64;
    for p in packed {
        match p {
            Some(bytes) => {
                writer.write_all(&bytes).expect("write to out file failed");
                n_written += 1;
            }
            None => n_missing += 1,
        }
    }
    chunk.clear();
    (n_written, n_missing)
}

fn main() {
    let mut args = env::args().skip(1);
    let smi_path = args
        .next()
        .expect("usage: bench_fingerprints <smiles.txt> <out.bin> [n_limit]");
    let out_path = args
        .next()
        .expect("usage: bench_fingerprints <smiles.txt> <out.bin> [n_limit]");
    let n_limit: Option<usize> = args.next().and_then(|s| s.parse().ok());

    eprintln!("=== bench_fingerprints ===");
    eprintln!("input:  {smi_path}");
    eprintln!("output: {out_path}");
    if let Some(n) = n_limit {
        eprintln!("limit:  first {n} lines");
    }
    eprintln!("peak RSS at start: {:.3} GB", peak_rss_gb());

    let reader = BufReader::new(File::open(&smi_path).expect("failed to open smiles file"));
    let mut writer = BufWriter::new(File::create(&out_path).expect("failed to create out file"));

    let mut chunk: Vec<String> = Vec::with_capacity(CHUNK_SIZE);
    let mut n_written: u64 = 0;
    let mut n_missing: u64 = 0;
    let mut n_read: usize = 0;
    let mut n_chunks_done: u64 = 0;

    let t0 = Instant::now();
    for line in reader.lines() {
        let line = line.expect("failed to read line");
        if line.is_empty() {
            continue;
        }
        chunk.push(line);
        n_read += 1;

        if chunk.len() == CHUNK_SIZE {
            let (w, m) = flush_chunk(&mut chunk, &mut writer);
            n_written += w;
            n_missing += m;
            n_chunks_done += 1;
            if n_chunks_done % 8 == 0 {
                eprintln!(
                    "  {n_read} read, {n_written} written, {:.1}s elapsed",
                    t0.elapsed().as_secs_f64()
                );
            }
        }

        if let Some(limit) = n_limit {
            if n_read >= limit {
                break;
            }
        }
    }
    if !chunk.is_empty() {
        let (w, m) = flush_chunk(&mut chunk, &mut writer);
        n_written += w;
        n_missing += m;
    }
    writer.flush().expect("final flush failed");
    let elapsed = t0.elapsed().as_secs_f64();

    eprintln!("Fingerprinted {n_written} molecules ({n_missing} unparseable) in {elapsed:.1}s");
    eprintln!("Throughput: {:.0} molecules/s", n_written as f64 / elapsed);
    eprintln!("peak RSS (VmHWM): {:.3} GB", peak_rss_gb());
}
