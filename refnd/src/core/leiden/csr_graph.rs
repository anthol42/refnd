use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use rayon::prelude::*;
use crate::core::hnsw::measure;
#[cfg(feature = "monitor")]
use crate::core::hnsw::LockStat;

#[derive(Clone)]
pub enum  INWeightType{
    Similarity,
    Distance,
    SimilarityComplement,
    Unweighted,
}

/// Tracks `new_par`'s one deliberately-sequential step (the exclusive scan
/// over `offsets`) -- see `gveleiden.rs`'s `STAT_SEQ_*` stats, reported
/// together as part of the sequential-time breakdown in `LEIDEN.md`.
#[cfg(feature = "monitor")]
pub static STAT_NEWPAR_SCAN: LockStat = LockStat::new();

#[inline]
fn map_weight(inweight_type: &INWeightType, w: f32) -> f32{
    match inweight_type {
        INWeightType::Similarity => {w}
        INWeightType::Distance => {1.0 / (1.0 + w)}
        INWeightType::SimilarityComplement => {1.0 - w}
        INWeightType::Unweighted => {1.0}
    }
}

/// Raw-pointer wrapper letting multiple threads write disjoint elements of
/// one buffer concurrently. Used by `new_par`'s scatter pass, where each
/// write's target slot is proven unique by construction (an atomic
/// per-source cursor hands out a distinct position within that source's
/// exclusive `[offset, offset+degree)` range). `.get()` is a method (not
/// direct field access) so a disjoint-closure-capturing closure captures
/// the whole `Sync` wrapper rather than reaching through to the raw
/// pointer field directly, which would bypass the `Sync` impl.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    unsafe fn get(&self, offset: usize) -> *mut T { unsafe { self.0.add(offset) } }
}

#[derive(Clone)]
pub struct CsrGraph {
    pub n: usize,
    pub m: f32,           // total weight (each edge counted once)
    offsets: Vec<usize>,
    adj: Vec<(u32, f32)>, // (neighbor, weight) — u32 to halve cache pressure
}

impl CsrGraph {
    /// Builds an undirected CSR graph from a raw edge list.
    ///
    /// - `n`: number of nodes.
    /// - `edges`: `(src, dst, w)` triples; each is stored on both endpoints' adjacency
    ///   lists (self-loops occupy a single slot).
    /// - `inweight_type`: how to interpret the raw `w` value in `edges` and convert it
    ///   to a similarity-like edge weight used by Leiden:
    ///   - `Similarity`: `w` is already a similarity — used as-is.
    ///   - `Distance`: `w` is a distance, mapped to `1 / (1 + w)` so closer nodes get
    ///     higher weight.
    ///   - `SimilarityComplement`: `w` is `1 - similarity`, mapped back via `1 - w`.
    ///   - `Unweighted`: `w` is ignored and every edge weight is set to `1.0`.
    pub fn new(n: usize, edges: &[(u32, u32, f32)], inweight_type: INWeightType) -> Self {
        let m = edges.iter().map(|&(_, _, w)| map_weight(&inweight_type, w)).sum();

        // Degree count — self-loops occupy one slot, not two
        let mut offsets = vec![0usize; n + 1];
        for &(src, dst, _) in edges {
            let (src, dst) = (src as usize, dst as usize);
            offsets[src + 1] += 1;
            if src != dst { offsets[dst + 1] += 1; }
        }
        for i in 1..=n { offsets[i] += offsets[i - 1]; }

        let mut adj = vec![(0u32, 0.0f32); offsets[n]];
        let mut cursor = offsets[..n].to_vec();

        for &(src, dst, mut w) in edges {
            let (src, dst) = (src as usize, dst as usize);
            w = map_weight(&inweight_type, w);
            adj[cursor[src]] = (dst as u32, w);
            cursor[src] += 1;
            if src != dst {
                adj[cursor[dst]] = (src as u32, w);
                cursor[dst] += 1;
            }
        }

        Self { n, m, offsets, adj }
    }

    /// Parallel equivalent of `new`, for building large graphs (e.g.
    /// per-level aggregated graphs in `gveleiden.rs`/`leidenp.rs`) without
    /// `new`'s two O(edges) sequential passes (degree count, edge
    /// placement) becoming the dominant cost. `m` and the degree count are
    /// computed via a parallel reduction/atomic-increment pass; the edge
    /// placement pass becomes a parallel scatter using an atomic
    /// per-source cursor (initialized to that source's offset) so every
    /// edge claims a unique, disjoint slot -- same counting-sort shape as
    /// `gveleiden.rs::community_vertices`. Only the exclusive scan over
    /// `offsets` (O(n), not O(edges)) stays sequential -- a single
    /// cache-friendly linear pass, measured to not be worth parallelizing
    /// at the node/community counts this is called with (see
    /// `LEIDEN.md`'s sequential-cost breakdown).
    ///
    /// Also fixes a precision issue `new` still has: `new`'s `m` sums
    /// directly into an `f32` accumulator, which silently saturates well
    /// short of the true value once the running sum's magnitude exceeds
    /// what `f32` can represent an increment of (confirmed on the
    /// production dataset: `new`'s `m` read back as exactly a power of two,
    /// nowhere near the true total). This accumulates in `f64` and casts
    /// down once at the end.
    pub fn new_par(n: usize, edges: &[(u32, u32, f32)], inweight_type: INWeightType, pool: &rayon::ThreadPool) -> Self {
        let m: f64 = pool.install(|| {
            edges.par_iter().map(|&(_, _, w)| map_weight(&inweight_type, w) as f64).sum()
        });

        // Parallel degree count -- self-loops occupy one slot, not two.
        let degree: Vec<AtomicUsize> = pool.install(|| {
            (0..=n).into_par_iter().map(|_| AtomicUsize::new(0)).collect()
        });
        pool.install(|| {
            edges.par_iter().for_each(|&(src, dst, _)| {
                let (src, dst) = (src as usize, dst as usize);
                degree[src + 1].fetch_add(1, Ordering::Relaxed);
                if src != dst { degree[dst + 1].fetch_add(1, Ordering::Relaxed); }
            });
        });

        let mut offsets = vec![0usize; n + 1];
        measure!({
            for i in 0..=n { offsets[i] = degree[i].load(Ordering::Relaxed); }
            for i in 1..=n { offsets[i] += offsets[i - 1]; }
        }, STAT_NEWPAR_SCAN);

        // `vec![(0u32, 0.0f32); offsets[n]]` would fill the whole buffer
        // sequentially only for the parallel scatter below to immediately
        // overwrite every element -- doubling the memory traffic for no
        // benefit. SAFETY: every index in `[0, offsets[n])` is written
        // exactly once by the scatter (each edge's `cursor` claim lands in
        // its source's exclusive `[offsets[src], offsets[src+1])` range,
        // covering the whole buffer), before `adj` is ever read; `(u32,
        // f32)` has no `Drop` impl, so nothing can go wrong with the
        // momentarily-uninitialized capacity in between.
        let mut adj: Vec<(u32, f32)> = Vec::with_capacity(offsets[n]);
        unsafe { adj.set_len(offsets[n]); }
        let cursor: Vec<AtomicUsize> = pool.install(|| {
            offsets[..n].par_iter().map(|&o| AtomicUsize::new(o)).collect()
        });
        let adj_ptr = SendPtr(adj.as_mut_ptr());
        pool.install(|| {
            edges.par_iter().for_each(|&(src, dst, mut w)| {
                let (src, dst) = (src as usize, dst as usize);
                w = map_weight(&inweight_type, w);
                let pos = cursor[src].fetch_add(1, Ordering::Relaxed);
                unsafe { adj_ptr.get(pos).write((dst as u32, w)); }
                if src != dst {
                    let pos = cursor[dst].fetch_add(1, Ordering::Relaxed);
                    unsafe { adj_ptr.get(pos).write((src as u32, w)); }
                }
            });
        });

        Self { n, m: m as f32, offsets, adj }
    }

    /// Adjacency list of `v` as (neighbor, weight) pairs.
    #[inline]
    pub fn neighbors(&self, v: usize) -> &[(u32, f32)] {
        &self.adj[self.offsets[v]..self.offsets[v + 1]]
    }

    /// Sum of edge weights incident to `v` (self-loops counted once).
    #[inline]
    pub fn strength(&self, v: usize) -> f32 {
        self.neighbors(v).iter().map(|&(_, w)| w).sum()
    }

    /// Induced subgraph on `nodes`. New ids are assigned in the order `nodes` is given.
    /// Returns the subgraph plus a map from old node id to new node id.
    pub fn subgraph(&self, nodes: &[usize]) -> (Self, BTreeMap<usize, usize>) {
        let old_to_new: BTreeMap<usize, usize> = nodes
            .iter()
            .enumerate()
            .map(|(new_id, &old_id)| (old_id, new_id))
            .collect();

        let mut edges = Vec::new();
        for (&old_src, &new_src) in &old_to_new {
            for &(old_dst, w) in self.neighbors(old_src) {
                let old_dst = old_dst as usize;
                if old_src > old_dst {
                    continue; // already added from the other endpoint
                }
                if let Some(&new_dst) = old_to_new.get(&old_dst) {
                    edges.push((new_src as u32, new_dst as u32, w));
                }
            }
        }

        (Self::new(old_to_new.len(), &edges, INWeightType::Similarity), old_to_new)
    }
}
