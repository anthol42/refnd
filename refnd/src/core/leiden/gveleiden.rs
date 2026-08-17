//! Direct port of the GVE-Leiden algorithm (arxiv.org/abs/2312.13936,
//! github.com/puzzlef/leiden-communities-openmp, `inc/leiden.hxx`'s
//! `leidenInvokeOmp`/`leidenMoveOmpW`/`leidenAggregateOmpW`), not an
//! incremental parallelization of `leiden.rs` like `leidenp.rs` is. Three
//! algorithmic differences from `leidenp.rs` (not just implementation
//! details) are the point of this port, all straight from the reference:
//!
//! - **Local-moving stops on a tolerance, not a fixed point.** Each round's
//!   total gain (`el`, the sum of every move's diff) is compared against a
//!   shrinking threshold; the round loop breaks once `el <= tolerance`,
//!   capped at `max_iterations` rounds regardless. `leidenp.rs` instead runs
//!   until literally no node wants to move.
//! - **Refinement is a single deterministic sweep, not a per-cluster
//!   probabilistic search.** Every node starts as its own singleton
//!   "refined" community; one parallel pass lets each node try to merge into
//!   its best same-local-move-community neighbour, using an atomic
//!   claim (`fetch_update`) so a community can be joined by at most one
//!   winner even under concurrent attempts. No shuffle, no RNG, no
//!   `exp(diff/beta)` likelihood weighting -- `beta` doesn't exist in this
//!   port.
//! - **Aggregation stops early once compression stalls.** If refined
//!   cluster count is already `>= aggregation_tolerance` fraction of the
//!   current level's node count, the level loop breaks instead of
//!   aggregating -- avoids grinding through levels that barely shrink the
//!   graph (confirmed wasteful on the production benchmark: level 4 through
//!   level 8 shrank node count by well under 1%, each still costing full
//!   fastmove/merge/aggregate wall time in `leidenp.rs`).
//!
//! `retrieve_clusters`'s sequential per-node `push` bucketing (an unsolved,
//! explicitly-flagged bottleneck in `leidenp.rs`) is also replaced here,
//! for the one place this port still needs it (aggregation): a parallel
//! counting sort (`community_vertices`) using a first pass of atomic
//! per-community counts, a sequential exclusive scan over community count
//! (cheap -- O(communities), not O(nodes)), then a parallel scatter using
//! atomic per-community cursors into disjoint slots of one shared buffer.
//! Refinement itself needs no per-cluster enumeration at all (unlike
//! `merge_nodes`), since it's one flat pass over all nodes filtered by the
//! local-move community bound inline -- so this counting sort runs once per
//! level, not twice.

use super::{CsrGraph, INWeightType, reindex_membership};
use crate::core::hnsw::measure;
#[cfg(feature = "monitor")]
use crate::core::hnsw::LockStat;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rayon::prelude::*;
use super::leiden::LeidenObjective;

/// Parallel sections never spawn more than this many worker threads.
const MAX_THREADS: usize = 8;

/// Upper bound on how many communities rayon batches into one dispatch task
/// (`par_iter().with_min_len(..)`), same reasoning as `leidenp.rs`'s
/// `MERGE_TASK_MAX_BATCH`: most communities are tiny, so per-task dispatch
/// overhead needs amortizing across several. Scaled down from this cap by
/// the actual community count at the call site (see its own comment).
const TASK_MAX_BATCH: usize = 1024;

#[cfg(feature = "monitor")]
pub static STAT_LOCAL_MOVE: LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_REFINE:     LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_AGGREGATE:  LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_FLATTEN:    LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_REINDEX:    LockStat = LockStat::new();
/// Sequential-cost tracking, requested to quantify how much of the
/// remaining wall time is actually unparallelized work (vs. parallel work
/// that just isn't scaling to 8 threads as well as hoped) -- see
/// `LEIDEN.md`'s measurement.
#[cfg(feature = "monitor")]
pub static STAT_SEQ_SETUP:   LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_SEQ_REINDEX: LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_SEQ_SCAN:    LockStat = LockStat::new();

// Per-thread hashtable scratch shared by all three phases (never called
// re-entrantly, so one buffer suffices): `vcout[c]` accumulates weight to
// candidate community `c`, using GVE's own zero-as-sentinel trick
// (`if !vcout[c] { vcs.push(c) }`) instead of a separate bitset -- one
// fewer container, one fewer memory access per neighbour touched. Grown,
// never shrunk, to the largest community-id space seen so far by this
// thread, matching `leidenp.rs`'s scratch-reuse convention.
thread_local! {
    static SCAN_SCRATCH: RefCell<(Vec<f32>, Vec<u32>)> = RefCell::new((Vec::new(), Vec::new()));
}

/// Cheap, well-scrambling integer hash (murmur3's `fmix64` finalizer) used
/// to decorrelate processing order from node id without an actual
/// (inherently sequential) Fisher-Yates shuffle -- see `parallel_order`.
#[inline]
fn scramble(i: u32) -> u64 {
    let x = i as u64;
    let x = (x ^ (x >> 33)).wrapping_mul(0xff51afd7ed558ccd);
    let x = (x ^ (x >> 33)).wrapping_mul(0xc4ceb9fe1a85ec53);
    x ^ (x >> 33)
}

/// A `[0, n)` permutation, decorrelated from node id, computed fully in
/// parallel. `local_move`/`refine` process nodes in this order (not raw
/// `0..n`) because this project's RGP-generated graphs frequently have
/// node ids correlated with similarity/generation order (confirmed
/// elsewhere in this codebase -- see `leidenp.rs`'s note on `merge_nodes`'s
/// cluster-id-correlated stragglers); ascending-id dynamic-chunk scheduling
/// then gives each thread an internally over-similar chunk, causing
/// systematic within-chunk over-merging before any cross-chunk information
/// exchange -- confirmed empirically on the toy dataset (community count/
/// quality only matched the sequential reference once order was
/// decorrelated). A true random shuffle (`rand`'s Fisher-Yates) fixed this
/// but is inherently sequential and measured as a real cost at production
/// scale (~570ms at n=50M, called at least twice per level) -- sorting by a
/// cheap hash of each index instead needs no such property beyond
/// "decorrelated from id" and is embarrassingly parallel, while still
/// provably visiting every index exactly once (it's a sort of the same
/// `[0, n)` set, not a formula-based permutation that could silently drop
/// or duplicate an index on some input).
fn parallel_order(n: usize, pool: &rayon::ThreadPool) -> Vec<u32> {
    let mut order: Vec<u32> = (0..n as u32).collect();
    pool.install(|| {
        order.par_sort_unstable_by_key(|&i| scramble(i));
    });
    order
}

/// Atomically adds `val` to the f32 stored (as bits) in `cell` -- std has no `AtomicF32`.
fn atomic_f32_add(cell: &AtomicU32, val: f32) {
    cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f32::from_bits(bits) + val).to_bits())
    }).unwrap();
}

/// Raw-pointer wrapper letting multiple threads write disjoint elements of
/// one buffer concurrently (used by `community_vertices`'s scatter pass,
/// where each write's target slot is proven unique by construction: an
/// atomic per-community cursor hands out a distinct position within that
/// community's exclusive `[offset, offset+count)` range). `.get()` is a
/// method (not direct field access) so a disjoint-closure-capturing
/// closure captures the whole `Sync` wrapper rather than silently reaching
/// through to the raw pointer field, which would bypass the `Sync` impl.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline]
    unsafe fn get(&self, offset: usize) -> *mut T { unsafe { self.0.add(offset) } }
}

struct LeidenConfig {
    pub resolution: f32,
    /// Local-moving round stops once a round's total gain drops to or below
    /// this (absolute, same units as `weight_to_cluster`/diff scores --
    /// i.e. graph-edge-weight scale, not a normalized [-0.5, 1] modularity
    /// scale like the reference's literal `1e-2` default assumes). Shrinks
    /// by `tolerance_drop` after every aggregation level, same as GVE.
    pub tolerance: f64,
    pub tolerance_drop: f64,
    /// Stop aggregating once `refined_clusters / current_level_nodes` is at
    /// or above this fraction -- further levels would barely shrink the
    /// graph. GVE's own default (0.8) is used unchanged.
    pub aggregation_tolerance: f64,
    pub max_iterations: usize,
    pub max_passes: usize,
}

struct LeidenState {
    graph: CsrGraph,
    node_weight: Vec<f32>,
    membership: Vec<u32>,
    pool: rayon::ThreadPool,
}

impl LeidenState {
    pub fn find_partition(&mut self, config: &LeidenConfig) -> bool {
        let (mut aggregated_membership, mut aggregated_graph, mut aggregated_node_weights, mut super_node_map) = measure!({
            let mut aggregated_membership: Vec<u32> = self.membership.clone();
            let aggregated_graph = self.graph.clone();
            let aggregated_node_weights: Vec<f32> = self.node_weight.clone();
            let super_node_map: Vec<u32> = (0..self.graph.n as u32).collect();
            measure!(reindex_membership(&mut aggregated_membership, aggregated_graph.n), STAT_SEQ_REINDEX);
            (aggregated_membership, aggregated_graph, aggregated_node_weights, super_node_map)
        }, STAT_SEQ_SETUP);

        let mut changed = false;
        let mut tolerance = config.tolerance;
        let mut level = 0usize;
        loop {
            let (did_change, rounds) = measure!(
                self.local_move(&aggregated_graph, &aggregated_node_weights, config, tolerance, &mut aggregated_membership),
                STAT_LOCAL_MOVE
            );
            changed = changed || did_change;
            let nb_clusters = measure!(reindex_membership(&mut aggregated_membership, aggregated_graph.n), STAT_SEQ_REINDEX);

            measure!({
                if level > 0 {
                    self.pool.install(|| {
                        self.membership.par_iter_mut().enumerate().for_each(|(node_id, m)| {
                            *m = aggregated_membership[super_node_map[node_id] as usize];
                        });
                    });
                } else {
                    self.membership.copy_from_slice(&aggregated_membership);
                }
            }, STAT_FLATTEN);

            // GVE's own stop conditions: local-moving barely did anything
            // this level, we've hit the pass cap, or aggregation is no
            // longer worth its cost (compression has stalled).
            if rounds <= 1 || level + 1 >= config.max_passes { break; }
            let compression = nb_clusters as f64 / aggregated_graph.n as f64;
            if compression >= config.aggregation_tolerance { break; }

            let mut refined_membership = measure!(
                self.refine(&aggregated_graph, &aggregated_node_weights, config, &aggregated_membership),
                STAT_REFINE
            );
            let nb_refined_clusters = measure!(reindex_membership(&mut refined_membership, aggregated_graph.n), STAT_SEQ_REINDEX);

            measure!({
                self.pool.install(|| {
                    super_node_map.par_iter_mut().for_each(|s| { *s = refined_membership[*s as usize]; });
                });
            }, STAT_REINDEX);

            (aggregated_graph, aggregated_membership, aggregated_node_weights) = measure!(
                self.aggregate(&aggregated_graph, &aggregated_node_weights, &aggregated_membership, &refined_membership, nb_refined_clusters),
                STAT_AGGREGATE
            );

            tolerance /= config.tolerance_drop;
            level += 1;
        }

        changed
    }

    /// Local-moving phase (`leidenMoveOmpW`, non-refine). Every node is
    /// swept in parallel each round (`with_min_len`-chunked, mirroring the
    /// reference's `schedule(dynamic, 2048)`); `vaff` gates which nodes
    /// actually do work, cleared on entry and re-set on a moved node's
    /// neighbours, so later rounds visit a shrinking fraction of the array
    /// even though the scan itself is still full-width. An explicit
    /// shrinking active-list (ported from `leidenp.rs::fastmove_nodes`,
    /// which measurably needed one) was tried here and measured *slower*
    /// end-to-end despite doing genuinely less total work: this phase's
    /// tolerance-based stopping already caps it to ~12-13 rounds on the
    /// production dataset (vs. ~40 for `fastmove_nodes`'s run-to-fixed-point
    /// design), so the wasted per-round `vaff` scan was never the dominant
    /// cost here the way it was there -- round 1 alone (unavoidable in
    /// either design, everyone starts active) is already >50% of a level's
    /// total local-moving time. The active-list version's extra bookkeeping
    /// (a 4x-larger `queued_for` allocation than `vaff`'s bitmask,
    /// `ThreadPool::broadcast` every round, scattered- instead of
    /// fixed-shuffled-order access) cost more than the scan it saved.
    /// Reverted; see `LEIDEN_IMPROVEMENTS.md` for the measurements.
    /// Community ids are vertex ids throughout (`[0, n)`) -- unlike
    /// `leidenp.rs`'s reserved `graph.n + v` singleton-id trick, no id
    /// space doubling is needed: a node's own community always exists
    /// (initialized to itself) and simply becomes empty, not deallocated,
    /// when it moves away, so nothing needs recycling.
    fn local_move(&self, graph: &CsrGraph, node_weights: &[f32], config: &LeidenConfig,
                  tolerance: f64, membership: &mut Vec<u32>) -> (bool, usize) {
        let n = graph.n;
        // All three setup buffers are per-index, no cross-element
        // dependency, so unlike `reindex_membership` (which has a genuine
        // serial dependency through its `next_id` counter) these are
        // trivially parallel -- measured as a real cost at production
        // scale (level 0: 50M elements each) once `aggregate`'s bigger
        // sequential costs were fixed.
        let cluster_weight: Vec<AtomicU32> = self.pool.install(|| {
            (0..n).into_par_iter().map(|_| AtomicU32::new(0)).collect()
        });
        self.pool.install(|| {
            (0..n).into_par_iter().for_each(|v| {
                atomic_f32_add(&cluster_weight[membership[v] as usize], node_weights[v]);
            });
        });
        let membership_atomic: Vec<AtomicU32> = self.pool.install(|| {
            membership.par_iter().map(|&m| AtomicU32::new(m)).collect()
        });
        let vaff: Vec<AtomicBool> = self.pool.install(|| {
            (0..n).into_par_iter().map(|_| AtomicBool::new(true)).collect()
        });
        let changed = AtomicBool::new(false);

        let order = parallel_order(n, &self.pool);

        let min_len = (n / (MAX_THREADS * 4)).clamp(1, TASK_MAX_BATCH);
        let mut round = 0usize;
        loop {
            round += 1;
            let el: f64 = self.pool.install(|| {
                order.par_iter().with_min_len(min_len).map(|&v| {
                    let v = v as usize;
                    // A plain `load` first, before the atomic `swap` that
                    // actually claims the flag: on x86 a relaxed atomic
                    // load compiles to an ordinary `mov` (no `LOCK`
                    // prefix), while `swap` always needs a `LOCK XCHG`
                    // (cache-line-owning, expensive under any contention)
                    // regardless of whether the value even changes. Later
                    // rounds have a shrinking active fraction, so most of
                    // these `n` checks are misses -- paying `LOCK XCHG` for
                    // every one of them, not just the genuinely active
                    // nodes, was a real, measured cost the reference
                    // (a plain, non-atomic `vector<char>` read for the skip
                    // check) doesn't pay at all.
                    if !vaff[v].load(Ordering::Relaxed) { return 0.0; }
                    if !vaff[v].swap(false, Ordering::Relaxed) { return 0.0; }
                    let current = membership_atomic[v].load(Ordering::Relaxed) as usize;

                    SCAN_SCRATCH.with(|scratch| {
                        let (vcout, vcs) = &mut *scratch.borrow_mut();
                        if vcout.len() < n { vcout.resize(n, 0.0); }

                        for &(u, w) in graph.neighbors(v) {
                            let u = u as usize;
                            if u == v { continue; }
                            let c = membership_atomic[u].load(Ordering::Relaxed) as usize;
                            if vcout[c] == 0.0 { vcs.push(c as u32); }
                            vcout[c] += w;
                        }

                        // The baseline ("stay in `current`") must be scored
                        // with the exact same formula as every candidate,
                        // including excluding v's own weight from
                        // `current`'s cluster weight -- an implicit 0.0
                        // baseline (as if `current` were always as cheap as
                        // an empty community) systematically favours moving
                        // away even when staying is objectively better,
                        // since `vcout[current]` is usually positive. See
                        // `LEIDEN.md`'s note on `leidenp.rs::fastmove_nodes`
                        // for the same bug caught there previously.
                        let current_weight_excl_v = f32::from_bits(cluster_weight[current].load(Ordering::Relaxed)) - node_weights[v];
                        let baseline = vcout[current] - config.resolution * node_weights[v] * current_weight_excl_v;
                        let mut best = current;
                        let mut best_score = baseline;
                        for &c in vcs.iter() {
                            let c = c as usize;
                            if c != current {
                                let cw = f32::from_bits(cluster_weight[c].load(Ordering::Relaxed));
                                let score = vcout[c] - config.resolution * node_weights[v] * cw;
                                if score > best_score { best = c; best_score = score; }
                            }
                            vcout[c] = 0.0;
                        }
                        vcs.clear();

                        if best != current {
                            atomic_f32_add(&cluster_weight[current], -node_weights[v]);
                            atomic_f32_add(&cluster_weight[best], node_weights[v]);
                            membership_atomic[v].store(best as u32, Ordering::Relaxed);
                            changed.store(true, Ordering::Relaxed);
                            for &(u, _) in graph.neighbors(v) {
                                let u = u as usize;
                                if u != v { vaff[u].store(true, Ordering::Relaxed); }
                            }
                            (best_score - baseline) as f64
                        } else {
                            0.0
                        }
                    })
                }).sum()
            });
            if el <= tolerance || round >= config.max_iterations { break; }
        }

        self.pool.install(|| {
            membership.par_iter_mut().zip(membership_atomic.par_iter()).for_each(|(m, a)| {
                *m = a.load(Ordering::Relaxed);
            });
        });
        (changed.load(Ordering::Relaxed), round)
    }

    /// Refinement phase (`leidenMoveW<REFINE=true>`). A single parallel
    /// sweep, not a round loop: the reference always breaks after one pass
    /// for the refine case (its `fc` convergence check is short-circuited
    /// away by the `REFINE` compile-time flag), so refinement never
    /// iterates to a fixed point the way local-moving does -- it's meant to
    /// be one cheap deterministic tidy-up pass, not its own optimization
    /// loop. Every node starts as its own singleton community (`vcom[u] =
    /// u`); a node may only merge into a same-`vcob`-bound neighbour's
    /// community, and only wins the merge if an atomic `fetch_update`
    /// confirms its own community was still an untouched singleton at that
    /// instant (`ctot[v] == vtot[v]`) -- this is the reference's "capture,
    /// subtract, revert-if-stolen" pattern expressed as one CAS retry loop
    /// instead of two separate atomic ops, which is equivalent but avoids
    /// ever leaving `ctot[v]` in a transiently-wrong state between the two.
    /// Concurrent joiners of the *same* target community never conflict
    /// with each other, only with that target's own attempt to leave --
    /// `atomic_f32_add` on the target's weight always just succeeds.
    fn refine(&self, graph: &CsrGraph, node_weights: &[f32], config: &LeidenConfig,
              vcob: &[u32]) -> Vec<u32> {
        let n = graph.n;
        let refined: Vec<AtomicU32> = self.pool.install(|| {
            (0..n as u32).into_par_iter().map(AtomicU32::new).collect()
        });
        let ctot: Vec<AtomicU32> = self.pool.install(|| {
            node_weights.par_iter().map(|&w| AtomicU32::new(w.to_bits())).collect()
        });

        // Decorrelated for the same reason as `local_move`'s `order`:
        // ascending id order plus dynamic-chunk scheduling gives each
        // thread an internally-correlated slice of nodes to process, which
        // biases *which* singleton wins each contested CAS claim in a way
        // that measurably hurt quality on this project's RGP-generated
        // graphs (confirmed empirically -- see `local_move`'s doc comment).
        let order = parallel_order(n, &self.pool);

        let min_len = (n / (MAX_THREADS * 4)).clamp(1, TASK_MAX_BATCH);
        self.pool.install(|| {
            order.par_iter().with_min_len(min_len).for_each(|&v| {
                let v = v as usize;
                let vtot_v = node_weights[v];
                if f32::from_bits(ctot[v].load(Ordering::Relaxed)) > vtot_v { return; }

                SCAN_SCRATCH.with(|scratch| {
                    let (vcout, vcs) = &mut *scratch.borrow_mut();
                    if vcout.len() < n { vcout.resize(n, 0.0); }

                    for &(u, w) in graph.neighbors(v) {
                        let u = u as usize;
                        if u == v || vcob[u] != vcob[v] { continue; }
                        let c = refined[u].load(Ordering::Relaxed) as usize;
                        if vcout[c] == 0.0 { vcs.push(c as u32); }
                        vcout[c] += w;
                    }

                    let mut best = usize::MAX;
                    let mut best_gain = 0.0f32;
                    for &c in vcs.iter() {
                        let c = c as usize;
                        if c != v {
                            let cw = f32::from_bits(ctot[c].load(Ordering::Relaxed));
                            let diff = vcout[c] - config.resolution * vtot_v * cw;
                            if diff > best_gain { best = c; best_gain = diff; }
                        }
                        vcout[c] = 0.0;
                    }
                    vcs.clear();

                    if best != usize::MAX {
                        let claimed = ctot[v].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
                            let cur = f32::from_bits(bits);
                            if cur > vtot_v { None } else { Some((cur - vtot_v).to_bits()) }
                        }).is_ok();
                        if claimed {
                            atomic_f32_add(&ctot[best], vtot_v);
                            refined[v].store(best as u32, Ordering::Relaxed);
                        }
                    }
                });
            });
        });

        self.pool.install(|| {
            refined.into_par_iter().map(AtomicU32::into_inner).collect()
        })
    }

    /// Parallel counting sort: buckets `[0, n)` by `membership[v] as usize`
    /// (values in `[0, n_communities)`) into one flat `Vec<u32>`, returning
    /// `(offsets, members)` where community `c`'s members are
    /// `members[offsets[c]..offsets[c+1]]`. Two passes -- parallel count via
    /// atomics, sequential O(communities) exclusive scan, parallel scatter
    /// via per-community atomic cursors into disjoint slots (safe per
    /// `SendPtr`'s doc comment) -- replacing `leidenp.rs`'s
    /// `retrieve_clusters`, which is a genuinely serial multi-writer
    /// `push` loop (explicitly flagged there as an unaddressed bottleneck).
    fn community_vertices(&self, membership: &[u32], n: usize, n_communities: usize) -> (Vec<usize>, Vec<u32>) {
        let degree: Vec<AtomicU32> = self.pool.install(|| {
            (0..n_communities).into_par_iter().map(|_| AtomicU32::new(0)).collect()
        });
        self.pool.install(|| {
            membership.par_iter().for_each(|&c| { degree[c as usize].fetch_add(1, Ordering::Relaxed); });
        });

        let mut offsets = vec![0usize; n_communities + 1];
        measure!({
            let mut acc = 0usize;
            for c in 0..n_communities {
                offsets[c] = acc;
                acc += degree[c].load(Ordering::Relaxed) as usize;
            }
            offsets[n_communities] = acc;
        }, STAT_SEQ_SCAN);

        let cursor: Vec<AtomicU32> = self.pool.install(|| {
            (0..n_communities).into_par_iter().map(|_| AtomicU32::new(0)).collect()
        });
        // SAFETY: every index in `[0, n)` is written exactly once by the
        // scatter below (`offsets` partitions `[0, n)` across communities
        // exactly, one atomic-cursor-claimed slot per node), before
        // `members` is ever read; `u32` has no `Drop` impl. Same reasoning
        // as `aggregate`'s `aggregated_edges` buffer.
        let mut members: Vec<u32> = Vec::with_capacity(n);
        unsafe { members.set_len(n); }
        let members_ptr = SendPtr(members.as_mut_ptr());
        self.pool.install(|| {
            (0..n).into_par_iter().for_each(|v| {
                let c = membership[v] as usize;
                let pos = cursor[c].fetch_add(1, Ordering::Relaxed) as usize;
                unsafe { members_ptr.get(offsets[c] + pos).write(v as u32); }
            });
        });

        (offsets, members)
    }

    /// Aggregation phase (`leidenAggregateOmpW`). Each refined community's
    /// work -- scan members' edges, sum weight to neighbouring
    /// communities, pick a representative -- is independent and runs
    /// concurrently over the CSR built by `community_vertices`, same shape
    /// as `leidenp.rs`'s `aggregate` otherwise (per-thread hashtable
    /// scratch, each task returns its own local edge list concatenated by
    /// a cheap sequential pass afterward).
    fn aggregate(&self, graph: &CsrGraph, node_weights: &[f32], membership: &[u32],
                 refined_membership: &[u32], nb_refined_clusters: usize) -> (CsrGraph, Vec<u32>, Vec<f32>) {
        let (offsets, members) = self.community_vertices(refined_membership, graph.n, nb_refined_clusters);

        let min_len = (nb_refined_clusters / (MAX_THREADS * 4)).clamp(1, TASK_MAX_BATCH);
        let results: Vec<(Vec<(u32, u32, f32)>, f32, u32)> = self.pool.install(|| {
            (0..nb_refined_clusters).into_par_iter().with_min_len(min_len).map(|c| {
                SCAN_SCRATCH.with(|scratch| {
                    let (vcout, vcs) = &mut *scratch.borrow_mut();
                    if vcout.len() < nb_refined_clusters { vcout.resize(nb_refined_clusters, 0.0); }

                    let mut local_edges = Vec::new();
                    let mut weight_sum = 0.0f32;
                    for &v in &members[offsets[c]..offsets[c + 1]] {
                        let v = v as usize;
                        for &(u, w) in graph.neighbors(v) {
                            let c2 = refined_membership[u as usize] as usize;
                            if c2 > c { // count each edge once
                                if vcout[c2] == 0.0 { vcs.push(c2 as u32); }
                                vcout[c2] += w;
                            }
                        }
                        weight_sum += node_weights[v];
                    }

                    for &c2 in vcs.iter() {
                        let c2 = c2 as usize;
                        local_edges.push((c as u32, c2 as u32, vcout[c2]));
                        vcout[c2] = 0.0;
                    }
                    vcs.clear();

                    let representative = membership[members[offsets[c]] as usize];
                    (local_edges, weight_sum, representative)
                })
            }).collect()
        });

        // Concatenating each community's local edge list with a plain
        // `.extend()` loop is itself O(edges) and fully sequential --
        // measured as a real cost at production scale (~270ms at level 0
        // alone), not just `CsrGraph::new`'s own internals. Replaced with
        // the same counting-sort shape as `community_vertices`: an
        // O(communities) sequential exclusive scan over each community's
        // edge count, then a parallel scatter into disjoint slices of one
        // pre-sized buffer.
        let mut edge_offsets = vec![0usize; nb_refined_clusters + 1];
        let acc = measure!({
            let mut acc = 0usize;
            for (c, (local_edges, _, _)) in results.iter().enumerate() {
                edge_offsets[c] = acc;
                acc += local_edges.len();
            }
            edge_offsets[nb_refined_clusters] = acc;
            acc
        }, STAT_SEQ_SCAN);

        // SAFETY: every index in `[0, acc)` is written exactly once by the
        // scatter below (`edge_offsets` partitions `[0, acc)` across
        // communities exactly, and each community's task copies its full
        // `local_edges` slice into its assigned range), before
        // `aggregated_edges` is ever read; `(u32, u32, f32)` has no `Drop`
        // impl. A `vec![(0,0,0.0); acc]` pre-fill here would double the
        // memory traffic for no benefit, same reasoning as `CsrGraph::
        // new_par`'s `adj` buffer.
        let mut aggregated_edges: Vec<(u32, u32, f32)> = Vec::with_capacity(acc);
        unsafe { aggregated_edges.set_len(acc); }
        let mut aggregated_node_weights: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut aggregated_membership: Vec<u32> = vec![0; nb_refined_clusters];
        let edges_ptr = SendPtr(aggregated_edges.as_mut_ptr());
        self.pool.install(|| {
            results.par_iter().enumerate().zip(aggregated_node_weights.par_iter_mut()).zip(aggregated_membership.par_iter_mut())
                .for_each(|(((c, (local_edges, weight_sum, representative)), out_weight), out_membership)| {
                    let start = edge_offsets[c];
                    for (i, &e) in local_edges.iter().enumerate() {
                        unsafe { edges_ptr.get(start + i).write(e); }
                    }
                    *out_weight = *weight_sum;
                    *out_membership = *representative;
                });
        });

        (CsrGraph::new_par(nb_refined_clusters, &aggregated_edges, INWeightType::Similarity, &self.pool),
         aggregated_membership,
         aggregated_node_weights)
    }
}

pub fn gve_find_communities(graph: CsrGraph, gamma: f32, n_iterations: usize,
                             objective: LeidenObjective) -> Vec<usize> {
    let (resolution, node_weights) = match objective {
        LeidenObjective::Modularity => {
            let node_strengths: Vec<_> = (0..graph.n).map(|v| graph.strength(v)).collect();
            (gamma / node_strengths.iter().sum::<f32>(), node_strengths)
        }
        LeidenObjective::CPM => (gamma, vec![1.0f32; graph.n]),
    };
    // `tolerance` is in raw diff units (graph-edge-weight scale), not the
    // reference's normalized [-0.5, 1] modularity scale, so it's tied to
    // `graph.m` (the natural upper bound on a single `weight_to_cluster`
    // sum) rather than reused as the reference's literal 1e-2 default --
    // see `LeidenConfig::tolerance`'s doc comment.
    // `graph.m` is computed from raw pre-transform edge weights (see
    // `bench_leiden.rs`'s own note on it), not the actual post-transform
    // weights `neighbors()`/`strength()` return and that `weight_to_cluster`
    // accumulates -- using it here would scale `tolerance` against the
    // wrong quantity entirely. Sum of vertex strengths (halved, since each
    // edge is counted from both endpoints) is the real total edge weight.
    let total_weight: f64 = (0..graph.n).map(|v| graph.strength(v) as f64).sum::<f64>() / 2.0;
    let config = LeidenConfig {
        resolution,
        tolerance: 1e-4 * total_weight,
        tolerance_drop: 10.0,
        aggregation_tolerance: 0.995,
        max_iterations: 20,
        max_passes: 10,
    };
    let membership: Vec<u32> = (0..graph.n as u32).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_THREADS)
        .build()
        .expect("failed to build gveleiden thread pool");
    let mut leiden_state = LeidenState { graph, node_weight: node_weights, membership, pool };

    #[cfg(feature = "monitor")]
    let t_total = std::time::Instant::now();
    for _ in 0..(if n_iterations > 0 { n_iterations } else { usize::MAX }) {
        let changed = leiden_state.find_partition(&config);
        if !changed { break; }
    }

    #[cfg(feature = "monitor")]
    {
        eprintln!("── GVE-Leiden timings ──────────────────────────");
        STAT_LOCAL_MOVE.report("  local_move     ");
        STAT_REFINE.report("  refine         ");
        STAT_AGGREGATE.report("  aggregate      ");
        STAT_FLATTEN.report("  flatten/setup  ");
        STAT_REINDEX.report("  supernode_remap");
        eprintln!("── sequential-only parts (see LEIDEN.md) ────────");
        STAT_SEQ_SETUP.report("  find_partition setup clone");
        STAT_SEQ_REINDEX.report("  reindex_membership        ");
        STAT_SEQ_SCAN.report("  small exclusive scans     ");
        super::csr_graph::STAT_NEWPAR_SCAN.report("  new_par's own exclusive scan");
        let seq_ns: u64 = [&STAT_SEQ_SETUP, &STAT_SEQ_REINDEX, &STAT_SEQ_SCAN].iter()
            .map(|s| s.wait_ns.load(Ordering::Relaxed)).sum::<u64>()
            + super::csr_graph::STAT_NEWPAR_SCAN.wait_ns.load(Ordering::Relaxed);
        let total_ns = t_total.elapsed().as_nanos() as u64;
        eprintln!(
            "  total sequential: {:.3}s of {:.3}s wall ({:.2}%)",
            seq_ns as f64 / 1e9, total_ns as f64 / 1e9, 100.0 * seq_ns as f64 / total_ns as f64
        );
        eprintln!("────────────────────────────────────────────────");
    }

    leiden_state.membership.into_iter().map(|x| x as usize).collect()
}
