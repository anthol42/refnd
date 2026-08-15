use super::{CsrGraph, INWeightType, reindex_membership};
use crate::core::hnsw::measure;
#[cfg(feature = "monitor")]
use crate::core::hnsw::LockStat;
use fixedbitset::FixedBitSet;
use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use rand::prelude::*;
use rand::rng;
use rayon::prelude::*;
use super::leiden::LeidenObjective;

/// Parallel sections never spawn more than this many worker threads.
const MAX_THREADS: usize = 8;

/// Round cap for `fastmove_nodes`'s parallel local-moving loop. Even with
/// near-real-time atomic visibility between threads (see `fastmove_nodes`),
/// synchronous parallel local-moving can still settle into a small persistent
/// back-and-forth that never reaches a literal fixed point (confirmed
/// empirically on a toy graph: raising this cap 20x didn't shrink the
/// residual further, so it's a genuine steady-state cycle, not slow
/// convergence). This bounds the wasted work on that residual rather than
/// looping forever. Not a workaround specific to this port: both NetworKit's
/// PLM and the published GVE-Leiden design (github.com/puzzlef/leiden-
/// communities-openmp, arxiv.org/abs/2312.13936) cap their own move phase the
/// same way, for the same reason -- losing the sequential algorithm's
/// monotonic-improvement guarantee under concurrent updates is a known,
/// accepted property of the technique, not a bug to eliminate outright.
const MAX_FASTMOVE_ROUNDS: usize = 100;

/// Upper bound on how many original clusters rayon batches into one
/// `merge_nodes` dispatch task (`par_iter().with_min_len(..)`). At production
/// scale most clusters are tiny/singleton, so one task per cluster makes
/// rayon's own per-task dispatch overhead dominate the real work -- batching
/// consecutive clusters into a task amortizes that overhead without changing
/// what each cluster's `merge_nodes` call actually does.
///
/// The actual `min_len` passed at the call site is this value clamped down to
/// `nb_clusters / (MAX_THREADS * 4)`, not used directly -- rayon's splitter
/// only splits a range while `len / 2 >= min_len`, so a fixed `min_len` this
/// large would fully serialize (onto a single thread) any level with fewer
/// than `2 * MERGE_TASK_MAX_BATCH` clusters. That's not a rare edge case:
/// `nb_clusters` shrinks every level as aggregation proceeds, so later,
/// more-aggregated levels -- exactly the ones with fewer but individually
/// costlier clusters -- would be hit hardest. Scaling `min_len` down with
/// `nb_clusters` keeps enough splits available for full 8-way distribution at
/// every level while still capping batch size at this value once there's
/// plenty of clusters to batch.
const MERGE_TASK_MAX_BATCH: usize = 1024;

#[cfg(feature = "monitor")]
pub static STAT_FASTMOVE:   LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_MERGE:      LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_AGGREGATE:  LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_REINDEX:    LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_FLATTEN:    LockStat = LockStat::new();
#[cfg(feature = "monitor")]
pub static STAT_SETUP:      LockStat = LockStat::new();

/// Scratch reused across every `merge_nodes` call handled by one worker
/// thread. `merge_nodes` used to allocate ~8 fresh heap buffers on *every*
/// call -- called 270M+ times at production scale, mostly for tiny/singleton
/// clusters (median cluster size 1) -- so allocation itself, not dispatch or
/// algorithmic work, turned out to be the dominant cost (batching dispatch
/// via `with_min_len` barely moved `merge_nodes`'s true wall-clock share, see
/// `LEIDEN_IMPROVEMENTS.md`, which is the tell that allocation was the real
/// bottleneck all along). `cluster_weights`/`cluster_out_weight`/
/// `weight_to_cluster` are indexed by *local* cluster id `[0, n)` (`n` =
/// this call's cluster size) and grown, never shrunk, to the largest `n` seen
/// so far by this thread; every entry a call actually uses gets freshly
/// written before being read (no per-call reset pass needed), *except*
/// `non_singleton_cluster` and `new_cluster`, which use a read-before-first-
/// write sentinel pattern -- those track their own touched indices
/// (`*_touched`) and undo just those at the end of the call, same
/// self-cleanup idea `weight_to_cluster`/`is_neighbor_cluster` already used
/// (reset immediately after use, inside the algorithm's own loop).
struct MergeScratch {
    /// `v`'s local refined-cluster label, indexed by global node id (like the
    /// single shared buffer in the sequential version) -- kept one-per-thread
    /// so concurrent clusters never alias the same slots.
    refined_membership: Vec<u32>,
    cluster_weights: Vec<f32>,
    cluster_out_weight: Vec<f32>,
    non_singleton_cluster: FixedBitSet,
    non_singleton_touched: Vec<u32>,
    weight_to_cluster: Vec<f32>,
    is_neighbor_cluster: FixedBitSet,
    neighbor_clusters: Vec<u32>,
    cum_likelihood: Vec<f64>,
    /// Scratch for compacting `refined_membership` into a dense local id
    /// range at the end of the call (formerly `clean_refined_membership`'s
    /// own fresh `Vec` each time).
    new_cluster: Vec<u32>,
    new_cluster_touched: Vec<u32>,
}

impl MergeScratch {
    const fn new() -> Self {
        Self {
            refined_membership: Vec::new(),
            cluster_weights: Vec::new(),
            cluster_out_weight: Vec::new(),
            non_singleton_cluster: FixedBitSet::new(),
            non_singleton_touched: Vec::new(),
            weight_to_cluster: Vec::new(),
            is_neighbor_cluster: FixedBitSet::new(),
            neighbor_clusters: Vec::new(),
            cum_likelihood: Vec::new(),
            new_cluster: Vec::new(),
            new_cluster_touched: Vec::new(),
        }
    }
}

thread_local! {
    static MERGE_SCRATCH: RefCell<MergeScratch> = RefCell::new(MergeScratch::new());

    /// Per-thread scratch reused across all `aggregate` per-cluster calls handled
    /// by this worker: `weight_to_cluster[c2]` accumulates edge weight to
    /// candidate neighbour super-node `c2`, `is_neighbor_cluster` dedupes it.
    /// Indexed by refined-cluster id (the *new*, coarser graph's node count),
    /// resized (never shrunk) lazily on first use -- same reasoning as
    /// `MERGE_SCRATCH`. `neighbor_clusters` is cleared (not resized) after
    /// every call, same as `merge_nodes`'s copy of the same idea.
    static AGGREGATE_SCRATCH: RefCell<(Vec<f32>, FixedBitSet, Vec<u32>)> = RefCell::new((Vec::new(), FixedBitSet::new(), Vec::new()));

    /// Per-thread scratch reused across all `decide_move` calls handled by this
    /// worker, indexed by cluster id. Sized to `2*graph.n` (not `graph.n`) --
    /// see `fastmove_nodes` for why the id space is doubled.
    static FASTMOVE_SCRATCH: RefCell<(Vec<f32>, FixedBitSet)> = RefCell::new((Vec::new(), FixedBitSet::new()));

    /// Per-thread accumulator for `fastmove_nodes`'s next-round active list.
    /// `.map(|v| Vec::new()).collect::<Vec<Vec<u32>>>()` was tried first and
    /// materializes one `Vec` header per *processed* node, not per node that
    /// actually gets queued -- at 50M nodes/round that's ~1.2GB of mostly-
    /// empty headers just to immediately flatten and discard. Pushing into a
    /// persistent per-thread buffer instead, drained via `ThreadPool::
    /// broadcast` after each round, means only `MAX_THREADS` buffers ever
    /// need gathering, and their backing allocations are reused round to
    /// round like every other per-thread scratch in this file.
    static FASTMOVE_NEXT_ACTIVE: RefCell<Vec<u32>> = RefCell::new(Vec::new());
}

/// Atomically adds `val` to the f32 stored (as bits) in `cell` -- std has no
/// `AtomicF32`. Used for `fastmove_nodes`'s cluster-weight bookkeeping, which
/// multiple threads update concurrently.
fn atomic_f32_add(cell: &AtomicU32, val: f32) {
    cell.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |bits| {
        Some((f32::from_bits(bits) + val).to_bits())
    }).unwrap();
}

struct LeidenConfig {
    pub resolution: f32,
    pub beta: f64
}
struct LeidenState {
    graph: CsrGraph,
    node_weight: Vec<f32>,
    membership: Vec<u32>,
    pool: rayon::ThreadPool,
}

impl LeidenState {
    pub fn find_partition(&mut self, config: &LeidenConfig) -> bool{
        // Initialize temporary buffers
        #[allow(unused_assignments)]
        let mut nb_clusters = 0;
        let (mut refined_membership, mut cluster_scratch, mut super_node_map,
             mut aggregated_node_weights, mut aggregated_membership, mut aggregated_graph) = measure!({
            let refined_membership: Vec<u32> = vec![0; self.graph.n];
            let cluster_scratch: Vec<Vec<u32>> = vec![vec![]; self.graph.n]; // clusters
            let super_node_map: Vec<u32> = (0..self.graph.n as u32).collect(); // aggregate_vertex
            let aggregated_node_weights: Vec<f32> = self.node_weight.clone(); // i_vertex_out_weight
            let mut aggregated_membership: Vec<u32> = self.membership.clone(); // i_membership
            let aggregated_graph = self.graph.clone(); // i_graph and i_edge_weights

            // Ensure the cluster_ids are from [0 to k)
            #[allow(unused_assignments)]
            { nb_clusters = reindex_membership(&mut aggregated_membership, aggregated_graph.n); }
            (refined_membership, cluster_scratch, super_node_map, aggregated_node_weights, aggregated_membership, aggregated_graph)
        }, STAT_SETUP);

        let mut changed = false;
        let mut did_changed;
        let mut continue_clustering;
        let mut level = 0;
        loop {
            // Move nodes in order to increase the quality
            (did_changed, nb_clusters) = measure!(self.fastmove_nodes(
                &aggregated_graph,
                &aggregated_node_weights,
                &config,
                &mut aggregated_membership,
            ), STAT_FASTMOVE);
            changed = changed || did_changed;

            continue_clustering = nb_clusters < aggregated_graph.n;
            if continue_clustering {
                // Flatten membership. At level 0, `super_node_map` is still the identity
                // mapping ((0..n).collect(), set before this loop and only ever mutated
                // further down in this same iteration), so `aggregated_membership[super_node_map[v]]`
                // degenerates to `aggregated_membership[v]` for every node -- a straight copy.
                // Doing it as a copy instead of the general indexed gather lets the compiler
                // emit a single memcpy instead of a per-node indirect load.
                measure!({
                    if level > 0 {
                        // Independent per-index gather (`self.membership[i]`
                        // only ever depends on `super_node_map[i]` and reads
                        // of the read-only `aggregated_membership`), so it's
                        // safe to run across the same capped pool as
                        // everything else here.
                        self.pool.install(|| {
                            self.membership.par_iter_mut().enumerate().for_each(|(node_id, m)| {
                                *m = aggregated_membership[super_node_map[node_id] as usize];
                            });
                        });
                    } else {
                        self.membership.copy_from_slice(&aggregated_membership);
                    }
                    self.retrieve_clusters(&mut cluster_scratch, &aggregated_membership);
                }, STAT_FLATTEN);
                // ensure refined membership is correct size
                refined_membership.truncate(aggregated_graph.n);
                // Refine each cluster. Clusters are disjoint node sets, so each one's
                // local refinement (`merge_nodes`) is independent -- run them in
                // parallel, each returning its own locally-compacted refined ids
                // ([0, k_i)) instead of writing straight into a shared/globally-offset
                // buffer. A cheap sequential pass then turns those per-cluster id
                // spaces into one globally-contiguous range (`nb_refined_clusters`),
                // which is the only genuinely serial part of this phase.
                let state = &*self;
                // See `MERGE_TASK_MAX_BATCH` for why this is scaled down from
                // that cap rather than used directly.
                let merge_min_len = (nb_clusters / (MAX_THREADS * 4)).clamp(1, MERGE_TASK_MAX_BATCH);
                let cluster_results: Vec<(Vec<u32>, usize)> = state.pool.install(|| {
                    cluster_scratch[..nb_clusters]
                        .par_iter_mut()
                        .enumerate()
                        .with_min_len(merge_min_len)
                        .map(|(cluster_idx, members)| measure!(
                            state.merge_nodes(
                                &aggregated_graph,
                                &aggregated_node_weights,
                                members,
                                &aggregated_membership,
                                cluster_idx as u32,
                                &config,
                            ),
                            STAT_MERGE
                        ))
                        .collect()
                });
                let mut nb_refined_clusters = 0usize;
                for (cluster_idx, (local_ids, k_i)) in cluster_results.into_iter().enumerate() {
                    for (&v, local_id) in cluster_scratch[cluster_idx].iter().zip(local_ids) {
                        refined_membership[v as usize] = nb_refined_clusters as u32 + local_id;
                    }
                    nb_refined_clusters += k_i;
                    cluster_scratch[cluster_idx].clear();
                }

                // If the refinement didn't aggregate any cluster, we aggregate based on the
                // original clustering obtained by fastmove_nodes
                if nb_refined_clusters >= aggregated_graph.n {
                    refined_membership = aggregated_membership.clone();
                    nb_refined_clusters = nb_clusters;
                }

                // Compute super node mapping. Independent per-index
                // read-modify-write (each `super_node_map[i]` only ever
                // depends on its own prior value plus a read of the
                // read-only `refined_membership`), safe to parallelize the
                // same way as the flatten step above.
                measure!({
                    self.pool.install(|| {
                        super_node_map.par_iter_mut().for_each(|super_node_id| {
                            *super_node_id = refined_membership[*super_node_id as usize];
                        });
                    });
                }, STAT_REINDEX);
                (aggregated_graph, aggregated_membership, aggregated_node_weights) = measure!(self.aggregate(
                    &aggregated_graph,
                    &aggregated_node_weights,
                    &aggregated_membership,
                    &refined_membership,
                    nb_refined_clusters
                ), STAT_AGGREGATE);

                level += 1
            }
            // Optimization ended for this pass, we quit the loop
            if !continue_clustering { break; }
        }

        changed
    }

    fn retrieve_clusters(&self, cluster_scratch: &mut Vec<Vec<u32>>, aggregated_membership: &Vec<u32>) {
        for (node_id, &membership) in aggregated_membership.iter().enumerate() {
            cluster_scratch[membership as usize].push(node_id as u32);
        }
    }

    /// Local-moving phase, redesigned around a fused parallel decide+apply loop
    /// instead of the sequential version's async FIFO queue (every move
    /// immediately visible to the next) -- following the design used by real
    /// parallel Louvain/Leiden implementations (NetworKit's PLM; the published
    /// GVE-Leiden, github.com/puzzlef/leiden-communities-openmp): every node is
    /// swept in parallel each round; each thread scans its own node's
    /// neighbours, decides the best move, and commits it immediately via
    /// atomics, all within the same pass -- no separate frozen-snapshot decide
    /// phase followed by a serial apply/reconciliation phase. This matters
    /// operationally, not just architecturally: an earlier version of this
    /// function *did* freeze a snapshot and reconcile in a serial pass, and its
    /// serial apply cost scaled with the size of the active set each round --
    /// which is the *entire graph* on round 1. At production scale (50M
    /// nodes) that serial pass alone dominated the whole call, leaving 7 of 8
    /// worker threads idle and making the "parallel" version slower than the
    /// sequential one. The fused design keeps every round's work -- including
    /// the neighbour scan needed to find requeue candidates -- inside the
    /// parallel sweep.
    ///
    /// Which nodes need reconsidering each round is tracked via an explicit,
    /// shrinking `active: Vec<u32>` work list (round 1: everyone; later
    /// rounds: whoever a mover's neighbour-scan queued) rather than a full
    /// `(0..graph.n)` scan checking a per-node flag every round -- on the
    /// production dataset's first level this loop runs ~40 rounds, so a full
    /// scan every round meant ~40x more index checks than nodes actually
    /// needing reconsideration past the first couple of rounds, a real,
    /// measured cost. `queued_for: Vec<AtomicU32>` (the round number a node
    /// is queued for, not a plain boolean) dedupes queueing within a round
    /// without needing any reset pass between rounds -- see its own doc
    /// comment for why a boolean-plus-reset design raced. Each task collects
    /// newly-affected neighbours into `FASTMOVE_NEXT_ACTIVE`, a per-thread
    /// accumulator drained via `ThreadPool::broadcast` after the round --
    /// not a `.map(..).collect::<Vec<Vec<u32>>>()`, which materializes one
    /// (mostly-empty) `Vec` header per *processed* node rather than per node
    /// actually queued (~1.2GB of headers for round 1 alone at 50M nodes),
    /// see `FASTMOVE_NEXT_ACTIVE`'s doc comment. `membership`/`cluster_weights` become atomic for
    /// the duration of this call so concurrent threads can read/update them
    /// safely; updates are `Relaxed` since nothing here needs stronger
    /// ordering than "eventually visible to other threads" -- there's no
    /// synchronization-dependent invariant beyond the atomics' own
    /// read-modify-write correctness. (Unlike the sequential version, there's
    /// no `cluster_degree`/empty-cluster-recycling bookkeeping to maintain
    /// here -- see the singleton-id note below for why.)
    ///
    /// "Become a new singleton" needs a cluster id nobody else could possibly
    /// claim at the same time, without a shared/serial allocator -- solved by
    /// reserving `graph.n + v` as node `v`'s own private singleton id: unique
    /// by construction, so claiming it needs no synchronization at all. Hence
    /// cluster ids in this function range over `[0, 2*graph.n)`, not
    /// `[0, graph.n)`.
    ///
    /// Trade-off: two nodes can still race to move into the same cluster (or
    /// swap into each other's) based on a momentarily-stale read, since
    /// there's no per-decision re-validation here -- accepted rather than
    /// guarded against, same as both reference implementations, because with
    /// atomics the staleness window is tiny (a handful of concurrent
    /// instructions) rather than a full round, so it self-corrects almost
    /// immediately via the next round's active-list requeue instead of
    /// settling into a sustained oscillation. `MAX_FASTMOVE_ROUNDS` bounds the
    /// residual either way. `pytests/test_leiden_accuracy.py` is there to
    /// catch it if any of this ever drifts quality beyond normal stochastic
    /// noise.
    fn fastmove_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      config: &LeidenConfig,
                      membership: &mut Vec<u32>) -> (bool, usize){
        // Cluster ids [0, graph.n) are real clusters; [graph.n, 2*graph.n) are
        // each node's reserved private singleton id (node v -> graph.n + v).
        let id_space = 2 * graph.n;

        let cluster_weights: Vec<AtomicU32> = (0..id_space).map(|_| AtomicU32::new(0.0f32.to_bits())).collect();
        for v in 0..graph.n {
            let c = membership[v] as usize;
            atomic_f32_add(&cluster_weights[c], node_weights[v]);
        }
        let membership_atomic: Vec<AtomicU32> = membership.iter().map(|&m| AtomicU32::new(m)).collect();
        // `queued_for[v]` is the round number `v` is (or was last) queued
        // for -- whichever neighbour-scan first wants `v` reconsidered in
        // round R swaps this to R and, if it wasn't R already, pushes `v`
        // into that round's work list (so `v` is queued at most once per
        // round). Using the round number itself (rather than a plain
        // boolean reset to false at the start of each round) avoids needing
        // any reset pass at all: a stale stamp from round R-1 or earlier
        // never collides with the current target round R, since every round
        // uses a distinct, ever-increasing stamp. An earlier version used a
        // boolean cleared at the *start* of a node's own processing, which
        // raced a concurrent neighbour's mark of that same node (both
        // active this same round) -- fixable with a separate reset pass
        // before each round's real work, but that pass's own O(active)
        // cost roughly cancelled this whole optimization's benefit. This
        // sidesteps the race without paying for a reset pass at all.
        let queued_for: Vec<AtomicU32> = (0..graph.n).map(|_| AtomicU32::new(0)).collect();
        // Round 1 has no better candidate set than "everyone" (matches the
        // previous full-array-scan design's implicit round-1 behaviour).
        let mut active: Vec<u32> = (0..graph.n as u32).collect();

        let changed = AtomicBool::new(false);
        let mut round: u32 = 0;
        loop {
            round += 1;
            let any_moved = AtomicBool::new(false);
            // Scanning every one of `graph.n` nodes each round to find the
            // (rapidly shrinking, in practice) handful still affected was
            // real, measured waste: on the production dataset's first level,
            // this loop runs ~40 rounds, so a full-array scan every round
            // meant ~40x more index checks than nodes actually needing
            // reconsideration past the first couple of rounds. `active` is
            // the explicit, shrinking work list this round actually needs to
            // visit. Newly-affected neighbours are pushed into
            // `FASTMOVE_NEXT_ACTIVE`, a per-thread accumulator drained via
            // `broadcast` after the round -- not `.map(|v| Vec::new())
            // .collect::<Vec<Vec<u32>>>()`, which was tried first and
            // materializes one (mostly-empty) `Vec` header *per processed
            // node*, not per node actually queued; at 50M nodes/round that's
            // ~1.2GB of headers just to immediately flatten and discard,
            // which entirely cancelled this optimization's benefit.
            let next_round = round + 1;
            self.pool.install(|| {
                active.par_iter().for_each(|&v| {
                    let v = v as usize;
                    let Some(target) = self.decide_move(graph, node_weights, config, id_space, &membership_atomic, &cluster_weights, v) else { return; };

                    // decide_move only ever returns a target different from v's
                    // current cluster, and v is owned by exactly one task per
                    // round, so membership_atomic[v] can't have changed since.
                    let current = membership_atomic[v].load(Ordering::Relaxed) as usize;
                    atomic_f32_add(&cluster_weights[current], -node_weights[v]);
                    atomic_f32_add(&cluster_weights[target], node_weights[v]);
                    membership_atomic[v].store(target as u32, Ordering::Relaxed);
                    changed.store(true, Ordering::Relaxed);
                    any_moved.store(true, Ordering::Relaxed);

                    FASTMOVE_NEXT_ACTIVE.with(|scratch| {
                        let next = &mut *scratch.borrow_mut();
                        for &(u, _) in graph.neighbors(v) {
                            let u = u as usize;
                            if queued_for[u].swap(next_round, Ordering::Relaxed) != next_round {
                                next.push(u as u32);
                            }
                        }
                    });
                });
            });
            active = self.pool.broadcast(|_| {
                FASTMOVE_NEXT_ACTIVE.with(|scratch| std::mem::take(&mut *scratch.borrow_mut()))
            }).into_iter().flatten().collect();
            if !any_moved.load(Ordering::Relaxed) || round as usize >= MAX_FASTMOVE_ROUNDS { break; }
        }

        for (m, a) in membership.iter_mut().zip(&membership_atomic) {
            *m = a.load(Ordering::Relaxed);
        }
        let nb_clusters = reindex_membership(membership, id_space);
        (changed.load(Ordering::Relaxed), nb_clusters)
    }

    /// Scores node `v`'s candidate moves against the current (concurrently
    /// updated, not frozen) cluster state, returning the best target cluster
    /// id, or `None` to stay. Safe to call concurrently for different `v`:
    /// every shared read is atomic, and the only mutable state is the
    /// per-thread `FASTMOVE_SCRATCH` buffer, which different nodes on the same
    /// thread reuse sequentially and different threads never share.
    fn decide_move(&self, graph: &CsrGraph,
                   node_weights: &Vec<f32>,
                   config: &LeidenConfig,
                   id_space: usize,
                   membership: &Vec<AtomicU32>,
                   cluster_weights: &Vec<AtomicU32>,
                   v: usize) -> Option<usize> {
        let current_cluster = membership[v].load(Ordering::Relaxed) as usize;
        let singleton_cluster = graph.n + v;
        let weight = |c: usize| f32::from_bits(cluster_weights[c].load(Ordering::Relaxed));

        FASTMOVE_SCRATCH.with(|scratch| {
            let (weight_to_cluster, is_neighbor_cluster) = &mut *scratch.borrow_mut();
            if weight_to_cluster.len() < id_space {
                weight_to_cluster.resize(id_space, 0.0);
                is_neighbor_cluster.grow(id_space);
            }
            let mut neighbor_clusters: Vec<u32> = Vec::new();

            for &(u, w) in graph.neighbors(v) {
                let u = u as usize;
                if u != v {
                    let c = membership[u].load(Ordering::Relaxed) as usize;
                    if !is_neighbor_cluster.put(c) {
                        neighbor_clusters.push(c as u32);
                    }
                    weight_to_cluster[c] += w;
                }
            }

            // Calculate the score for each cluster to find the best one. Every
            // candidate is scored against a cluster weight that EXCLUDES v's
            // own contribution -- v isn't a member of any other candidate
            // cluster, so `current_cluster` must be treated the same way for a
            // fair comparison. "Become a new singleton" is evaluated first
            // (matching the sequential version's ordering, for the same
            // tie-breaking behaviour) -- its diff is always exactly 0, since an
            // empty cluster has no weight and no edges point to it yet.
            let current_cluster_weight = weight(current_cluster) - node_weights[v];
            let mut best: Option<usize> = None;
            let mut max_diff = weight_to_cluster[current_cluster] -
                config.resolution * (node_weights[v] * current_cluster_weight);
            if 0.0 > max_diff && current_cluster != singleton_cluster {
                best = Some(singleton_cluster);
                max_diff = 0.0;
            }
            for &c in &neighbor_clusters {
                let c = c as usize;
                let cw = if c == current_cluster { current_cluster_weight } else { weight(c) };
                let diff = weight_to_cluster[c] -
                    config.resolution * (node_weights[v] * cw);
                // Only consider positive improvements
                if diff > max_diff {
                    best = if c == current_cluster { None } else { Some(c) };
                    max_diff = diff;
                }
                weight_to_cluster[c] = 0.0;
                is_neighbor_cluster.set(c, false);
            }

            best
        })
    }

    /// Refines one (disjoint) cluster's members in isolation and returns the
    /// resulting locally-compacted refined-cluster id for each member (aligned
    /// 1:1 with `cluster_members`'s order on return, i.e. post-shuffle) plus
    /// the number of distinct ids used ([0, count)). Safe to call concurrently
    /// for different clusters: all scratch is the per-thread `MERGE_SCRATCH`
    /// buffer, which different clusters on the same thread reuse sequentially
    /// and different threads never share.
    ///
    /// Every local-cluster-indexed field of `MergeScratch` is grown, never
    /// shrunk, to this call's `n` -- see `MergeScratch`'s doc comment for why
    /// (and why most of them need no per-call reset even though they're
    /// reused). The only heap allocation left is the returned `Vec<u32>`
    /// itself, which is genuinely owned output data handed back to the
    /// caller, not internal scratch.
    fn merge_nodes(&self, graph: &CsrGraph,
                      node_weights: &Vec<f32>,
                      cluster_members: &mut Vec<u32>,
                      membership: &Vec<u32>,
                      cluster_idx: u32,
                      config: &LeidenConfig) -> (Vec<u32>, usize) {
        let n = cluster_members.len();

        MERGE_SCRATCH.with(|scratch| {
            let s = &mut *scratch.borrow_mut();
            if s.refined_membership.len() < graph.n {
                s.refined_membership.resize(graph.n, 0);
            }
            if s.cluster_weights.len() < n {
                s.cluster_weights.resize(n, 0.0);
                s.cluster_out_weight.resize(n, 0.0);
                s.weight_to_cluster.resize(n, 0.0);
                s.non_singleton_cluster.grow(n);
                s.is_neighbor_cluster.grow(n);
                s.new_cluster.resize(n, 0);
            }
            let MergeScratch { refined_membership, cluster_weights, cluster_out_weight,
                non_singleton_cluster, non_singleton_touched, weight_to_cluster,
                is_neighbor_cluster, neighbor_clusters, cum_likelihood,
                new_cluster, new_cluster_touched } = s;

            let mut total_node_weight: f32 = 0.0;
            for (c, &v) in cluster_members.iter().enumerate() {
                let v = v as usize;
                refined_membership[v] = c as u32;
                // Plain assignment, not `+=` -- `c` is touched exactly once
                // in this loop (bijective with `cluster_members`), so this is
                // correct regardless of what a previous, larger call left in
                // this slot.
                cluster_weights[c] = node_weights[v];
                total_node_weight += node_weights[v];

                // Accumulates over possibly-multiple neighbours in this
                // cluster, so (unlike `cluster_weights[c]` above) needs an
                // explicit reset before summing into it.
                cluster_out_weight[c] = 0.0;
                for &(u, w) in graph.neighbors(v) {
                    let u = u as usize;
                    if u != v && membership[u] == cluster_idx {
                        cluster_out_weight[c] += w;
                    }
                }
            }

            // Main loop in a random order
            cluster_members.shuffle(&mut rng());

            for &v in cluster_members.iter() {
                let v = v as usize;
                let current_cluster = refined_membership[v] as usize;
                let node_weight_prod = cluster_weights[current_cluster] * (total_node_weight - cluster_weights[current_cluster]);

                if !non_singleton_cluster.contains(current_cluster) &&
                    cluster_out_weight[current_cluster] >= node_weight_prod * config.resolution {
                    // Remove the node from the cluster.
                    // Since cluster is a singleton, the cluster weight becomes 0
                    cluster_weights[current_cluster] = 0.0;

                    // Find neighbouring clusters, and also add the current cluster to ensure the node
                    // can stay in its current cluster
                    neighbor_clusters.push(current_cluster as u32);
                    is_neighbor_cluster.set(current_cluster, true);
                    for &(u, w) in graph.neighbors(v) {
                        let u = u as usize;
                        if u != v && membership[u] == cluster_idx {
                            let c = refined_membership[u] as usize;
                            if !is_neighbor_cluster.put(c) {
                                neighbor_clusters.push(c as u32);
                            }
                            weight_to_cluster[c] += w;
                        }
                    }

                    // Calculate diffs and likelihoods
                    let mut best_cluster = current_cluster;
                    let mut max_diff = 0.0f32;
                    let mut total_cum_likelihood = 0.0f64;
                    for &c in neighbor_clusters.iter() {
                        let c = c as usize;
                        let node_weight_prod = cluster_weights[c] * (total_node_weight - cluster_weights[c]);

                        if cluster_out_weight[c] >= config.resolution * node_weight_prod {
                            let diff = weight_to_cluster[c] - config.resolution * (node_weights[v] * cluster_weights[c]);
                            if diff > max_diff {
                                best_cluster = c;
                                max_diff = diff;
                            }
                            if diff >= 0.0 {
                                total_cum_likelihood += ((diff as f64) / config.beta).exp();
                            }
                        }
                        cum_likelihood.push(total_cum_likelihood);
                        // Reset scratch buffers
                        weight_to_cluster[c] = 0.0;
                        is_neighbor_cluster.set(c, false);
                    }
                    let chosen_cluster = if total_cum_likelihood.is_finite() {
                        let r = rng().random_range(0.0..total_cum_likelihood);
                        let chosen_idx = cum_likelihood.partition_point(|&x| x < r);
                        neighbor_clusters[chosen_idx] as usize
                    } else {
                        best_cluster
                    };
                    // Reset the remaining scratch buffers
                    neighbor_clusters.clear();
                    cum_likelihood.clear();

                    // Move node to randomly chosen cluster
                    cluster_weights[chosen_cluster] += node_weights[v];
                    // Update the cluster_out_weight state as the sum of edge weight going out of
                    // clusters may have changed
                    if chosen_cluster != current_cluster {
                        for &(u, w) in graph.neighbors(v) {
                            let u = u as usize;
                            if membership[u] == cluster_idx {
                                if refined_membership[u] as usize == chosen_cluster {
                                    cluster_out_weight[chosen_cluster] -= w;
                                }else {
                                    cluster_out_weight[chosen_cluster] += w;
                                }
                            }
                        }
                        refined_membership[v] = chosen_cluster as u32;
                        if !non_singleton_cluster.put(chosen_cluster) {
                            non_singleton_touched.push(chosen_cluster as u32);
                        }
                    }
                }
            }

            // Compact `cluster_members`' refined-cluster labels into a dense
            // local range [0, count) -- `new_cluster` uses the same
            // read-before-first-write sentinel pattern as
            // `non_singleton_cluster`, so self-cleans via `new_cluster_touched`
            // below instead of needing a fresh zeroed buffer.
            let mut nb_local_clusters = 0u32;
            for &v in cluster_members.iter() {
                let c = refined_membership[v as usize] as usize;
                if new_cluster[c] == 0 {
                    nb_local_clusters += 1;
                    new_cluster[c] = nb_local_clusters;
                    new_cluster_touched.push(c as u32);
                }
            }
            let compacted = cluster_members.iter()
                .map(|&v| new_cluster[refined_membership[v as usize] as usize] - 1)
                .collect();

            for &c in new_cluster_touched.iter() {
                new_cluster[c as usize] = 0;
            }
            new_cluster_touched.clear();
            for &c in non_singleton_touched.iter() {
                non_singleton_cluster.set(c as usize, false);
            }
            non_singleton_touched.clear();

            (compacted, nb_local_clusters as usize)
        })
    }

    fn aggregate(&self,
                 graph: &CsrGraph,
                 node_weights: &Vec<f32>,
                 membership: &Vec<u32>,
                 refined_membership: &Vec<u32>,
                 nb_refined_clusters: usize) -> (CsrGraph, Vec<u32>, Vec<f32>) {
        let mut refined_clusters: Vec<Vec<u32>> = vec![Vec::new(); nb_refined_clusters];
        self.retrieve_clusters(&mut refined_clusters, refined_membership);

        // Each super-node `c`'s work (scan its members' edges, sum their weight,
        // pick a representative) only reads shared state and writes to its own
        // `c`-indexed output slot -- independent across clusters, so runs in
        // parallel. Each task returns its own local edge list instead of pushing
        // into one shared `aggregated_edges`; concatenated below.
        let results: Vec<(Vec<(u32, u32, f32)>, f32, u32)> = self.pool.install(|| {
            refined_clusters.par_iter().enumerate().map(|(c, refined_cluster)| {
                AGGREGATE_SCRATCH.with(|scratch| {
                    let (weight_to_cluster, is_neighbor_cluster, neighbor_clusters) = &mut *scratch.borrow_mut();
                    if weight_to_cluster.len() < nb_refined_clusters {
                        weight_to_cluster.resize(nb_refined_clusters, 0.0);
                        is_neighbor_cluster.grow(nb_refined_clusters);
                    }

                    let mut local_edges = Vec::new();
                    let mut node_weight_sum = 0.0f32;
                    // Iterate on all nodes in refined cluster to get neighbour cluster and weights
                    for &v in refined_cluster {
                        let v = v as usize;
                        // Then iterate on edges to find neighbour clusters
                        for &(u, w) in graph.neighbors(v) {
                            let c2 = refined_membership[u as usize] as usize;
                            // To consider each edge once
                            if c2 > c {
                                if !is_neighbor_cluster.put(c2) {
                                    neighbor_clusters.push(c2 as u32);
                                }
                                weight_to_cluster[c2] += w;
                            }
                        }
                        node_weight_sum += node_weights[v];
                    }

                    // Actually add edges
                    for &c2 in neighbor_clusters.iter() {
                        let c2 = c2 as usize;
                        local_edges.push((c as u32, c2 as u32, weight_to_cluster[c2]));

                        // Reset scratch buffer
                        weight_to_cluster[c2] = 0.0;
                        is_neighbor_cluster.set(c2, false);
                    }
                    neighbor_clusters.clear();

                    // Representative membership of super node
                    (local_edges, node_weight_sum, membership[refined_cluster[0] as usize])
                })
            }).collect()
        });

        let mut aggregated_edges: Vec<(u32, u32, f32)> = Vec::new();
        let mut aggregated_node_weights: Vec<f32> = vec![0.0; nb_refined_clusters];
        let mut aggregated_membership: Vec<u32> = vec![0; nb_refined_clusters];
        for (c, (local_edges, node_weight_sum, representative)) in results.into_iter().enumerate() {
            aggregated_edges.extend(local_edges);
            aggregated_node_weights[c] = node_weight_sum;
            aggregated_membership[c] = representative;
        }

        (CsrGraph::new(nb_refined_clusters, &aggregated_edges, INWeightType::Similarity),
        aggregated_membership,
        aggregated_node_weights)
    }
}

pub fn fast_find_communities(graph: CsrGraph, gamma: f32, beta: f64, n_iterations: usize,
                        objective: LeidenObjective) -> Vec<usize> {
    let (resolution, node_weights) = match objective {
        LeidenObjective::Modularity => {
            let node_strengths: Vec<_> = (0..graph.n).into_iter().map(|v| graph.strength(v)).collect();
            (gamma / node_strengths.iter().sum::<f32>(), node_strengths)
        }
        LeidenObjective::CPM => {
            (gamma, vec![1.0f32; graph.n])
        }
    };
    let membership: Vec<u32> = (0..graph.n as u32).collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(MAX_THREADS)
        .build()
        .expect("failed to build leidenp thread pool");
    let mut leiden_state = LeidenState{graph, node_weight: node_weights, membership, pool};
    let config = LeidenConfig{resolution, beta: beta};

    for _ in 0..(if n_iterations > 0 {n_iterations} else {usize::MAX}) {
        let changed = leiden_state.find_partition(&config);
        if !changed { break; }
    }

    #[cfg(feature = "monitor")]
    {
        eprintln!("── Leiden timings ──────────────────────────────");
        STAT_FASTMOVE.report("  fastmove_nodes");
        STAT_MERGE.report("  merge_nodes    ");
        STAT_AGGREGATE.report("  aggregate      ");
        STAT_FLATTEN.report("  flatten/retrieve");
        STAT_REINDEX.report("  supernode_remap");
        STAT_SETUP.report("  setup (clone)  ");
        eprintln!("────────────────────────────────────────────────");
    }

    // Convert u32 membership back to usize for the public API
    leiden_state.membership.into_iter().map(|x| x as usize).collect()
}
