# Leiden: correctness work done, parallelization plan next

## Correctness (done, this session)

Two real bugs found and fixed via line-by-line diff against igraph's C reference
implementation (`src/community/leiden.c`), not just from noticing bad output:

1. **Flatten-skip bug** (`find_partition`): `self.membership` was only written
   inside `if level > 0`, ported verbatim from igraph's C code -- but that guard
   is only safe in C because its `membership` output pointer is *aliased* onto
   the level-0 working buffer (so level-0 writes are already visible with no
   copy needed). This port clones instead of aliasing, so the guard silently
   discarded the entire clustering whenever the hierarchy converged in exactly
   two levels. Fixed with an explicit `copy_from_slice` for the level-0 case.
   Also explains why `n_iterations=0` (run to convergence) never terminated:
   every retry rediscovered and re-discarded the same result.

2. **Unpenalized stay-baseline** (`fastmove_nodes`): the "stay in current
   cluster" baseline used the raw, un-penalized edge weight, while every
   candidate move was correctly penalized by
   `resolution * node_weight(v) * cluster_weight(c)`. That gave staying put a
   discount no actual move got, biasing the local search toward worse local
   optima. Invisible on trivially-separated graphs (any real improvement
   cleared the inflated bar anyway); a measurable, consistent quality gap on
   anything harder. Fixed by applying the identical penalized formula to the
   baseline.

`merge_nodes` and `aggregate` were also diffed line-by-line against
`leiden_merge_vertices`/`leiden_aggregate` and found to already match --
no changes needed there.

**Verification**: `pytests/test_leiden_accuracy.py` -- runs both refnd and
igraph's Leiden N=20 times each on Karate Club (real + the genuine Zachary 1977
weighted edges), and Stochastic Block Model graphs at three difficulty tiers
(unweighted + weighted, within-block edges weighted higher than between-block),
under both Modularity and CPM objectives. Compares achieved objective quality
(not partition identity -- harder graphs legitimately have multiple
equally-good optima) via a one-sided Mann-Whitney U test, so a single unlucky
run can't produce a false pass or fail. Currently green.

## Baseline runtime (re-measured, post-fix, with per-phase breakdown)

CPM (γ=0.000008, β=0.01, iterations=2), on the production dataset
(`combined_train_test_layer0_0.2.edgestr`: 50,085,827 nodes, 104,357,200
edges), built with `--features monitor`: **175.971s** total, 15,599,528
communities, 82.17% singletons, H = 83,920,786.7. Essentially unchanged from
the pre-fix number (179.497s) -- confirms the two correctness fixes didn't
mask a hidden performance cost, so this is a clean starting point.

Per-phase breakdown (`STAT_*` counters, `cargo build --release --features
monitor`), summed across all 19 aggregation levels:

| phase | calls | total | % of wall time |
|---|---|---|---|
| `fastmove_nodes` | 21 | 92.3s | 52.5% |
| `merge_nodes` | 302,002,309 | 47.5s | 27.0% |
| `aggregate` | 19 | 21.9s | 12.5% |
| flatten/retrieve | 19 | 5.2s | 3.0% |
| supernode_remap | 19 | 1.6s | 0.9% |

`fastmove_nodes` alone is over half the runtime -- despite being only 21 calls
(one per level, internally looping over all active nodes), each call does
repeated O(degree) neighbor scans per node until the level stabilizes. This
means the decide/apply restructuring (plan section 3 below) has the largest
ceiling on wall-clock impact of the three phases, even though it requires the
most restructuring. `merge_nodes` is the easiest win (already embarrassingly
parallel, no restructuring) and is a solid second-place target at 27%.

`bench_leiden` (`refnd/src/bench_leiden.rs`) takes `<file> [modularity|cpm]
[gamma] [beta] [iterations]` and reports community count, size distribution
(min/p50/p90/p99/max/mean), singleton %, cross-community edge weight %, and the
achieved objective value -- use it before/after each parallelization step to
catch a quality regression, not just to time it. Build with `--features
monitor` for the per-phase breakdown above (adds negligible overhead: relaxed
atomic increments only).

## Production-scale comparison against igraph

Same dataset, same params (CPM, γ=0.000008, β=0.01, iterations=2). igraph's
own Leiden was run once via `pytests/save_igraph_partition.py`, which saves
the resulting membership + a metadata JSON next to the source dataset
(`combined_train_test_layer0_0.2.igraph_cpm_g8e-06_b0.01_i2.{membership.npy,meta.json}`)
so this expensive run never needs repeating -- reuse those files for any
future refnd-vs-igraph comparison on this dataset instead of re-running
igraph.

| | refnd (Rust) | igraph (Python) |
|---|---|---|
| algorithm time | 175.971s | 422.727s |
| communities | 15,599,528 | 16,057,818 |
| quality (normalized, `H/m`) | **0.9584** | 0.9218 |

refnd is ~2.4x faster *and* reaches higher quality (+0.037) than igraph on
the real production graph -- consistent with, not contradicted by, the
smaller-scale equivalence testing in `pytests/test_leiden_accuracy.py`, which
only claims parity, not superiority (its Mann-Whitney test asserts refnd
isn't *worse*, not that it can't be fractionally better). refnd's own
objective value is unnormalized (`H = Σ[e_c - (resolution/2)*K_c²]`); igraph's
`.quality` is normalized by total edge weight `m` (`H/m` -- see
`pytests/test_leiden_accuracy.py`'s `_cpm_quality` for the derivation), so the
175.971s run's `total_weight` (87,561,968.8, from `bench_leiden`'s own stats
output) was used to convert refnd's H (83,920,786.7) onto igraph's scale for
this comparison.

Note: `pytests/save_igraph_partition.py`'s own igraph-call timer (422.727s) is
the only trustworthy number from that run -- an earlier version of the script
that also re-ran refnd's Leiden in Python and left an untimed Python-level
list-comprehension step (splitting `EdgeStore.edges()`'s `(u,v,w)` triples
into separate `edge_pairs`/`weights` lists) turned a ~8-minute measured
workload into a ~2h42m wall-clock run. That gap is a benchmarking-script
inefficiency (CPython per-object overhead materializing two full 104M-element
lists), not a cost inherent to igraph or refnd -- harmless here since it's a
one-off prep script, but worth avoiding if this script is ever extended.

## Parallelization plan

`find_partition` runs three phases per level: `fastmove_nodes` (local moving),
`merge_nodes` (refinement), `aggregate` (coarsening). `rayon` and `dashmap`
(rayon feature) are already crate dependencies, already used elsewhere
(`core/hnsw/build.rs`, `search.rs`) -- no new dependencies needed.

### 1. `merge_nodes` -- parallelize first (easy, safe)

`for cluster_idx in 0..nb_clusters { merge_nodes(...) }` already operates on
disjoint node sets per cluster (each `cluster_scratch[cluster_idx]` is a
distinct partition of the nodes) -- the cheapest, safest win.

Approach: have each cluster's call return `(Vec<(node_id, local_cluster_id)>,
local_cluster_count)` instead of writing through the shared `&mut
refined_membership` / `nb_refined_clusters`. Run clusters via rayon
`.into_par_iter().map(...)`. Then a cheap sequential prefix-sum turns each
cluster's local cluster ids into a global id range, and a final pass applies
the writes. No `unsafe` needed.

### 2. `aggregate` -- parallelize second (easy-ish, safe)

The per-supernode loop (`for (c, refined_cluster) in
refined_clusters.iter().enumerate()`) is independent for reading -- each `c`
only scans its own members' edges -- but appends to a single shared
`aggregated_edges: Vec<(u32,u32,f32)>`.

Approach: rayon `.map()` per supernode producing a local edge `Vec`, then
flatten/concat (or `.fold()` + `.reduce()`). `weight_to_cluster`
/`is_neighbor_cluster`/`neighbor_clusters` become per-task scratch instead of
one shared buffer.

### 3. `fastmove_nodes` -- needed a fundamentally different design, not just restructuring

The originally-planned decide/apply-rounds design (frozen-snapshot parallel
decide + serial apply/reconciliation, analogous to `merge_nodes`/`aggregate`)
turned out not to work for this phase, for two compounding reasons discovered
during implementation:

- **Quality**: batching decisions against a snapshot frozen at each round's
  start let neighbors swap into each other's clusters based on stale data,
  undoing each other every round -- genuine, persistent oscillation, not slow
  convergence (confirmed by raising the round cap 20x with no improvement).
  Re-validating each decision against *current* weights during the (still
  serial) apply step helped a lot but didn't fully eliminate it.
- **Performance**: even before quality was fully sorted out, this design had a
  separate, more severe problem at production scale -- the serial apply
  phase's cost scales with the active-set size each round, which is the
  *entire graph* on round 1. At 50M nodes, that alone made the "parallel"
  version slower than sequential, with only 1 of 8 threads ever busy (confirmed
  via `top -H`, plainly reported by the user watching it run).

Both problems trace back to the same root cause: a frozen-snapshot-then-
serial-reconciliation architecture. `merge_nodes`/`aggregate` could get away
with this because their serial reconciliation pass is O(clusters), always far
smaller than the graph; `fastmove_nodes`'s natural granularity is per-node,
so its reconciliation pass is O(n) -- no way to shrink that by restructuring
alone.

**What actually shipped**, following two real, published, benchmarked
reference implementations (NetworKit's PLM, `networkit/cpp/community/PLM.cpp`;
the GVE-Leiden paper/implementation, arxiv.org/abs/2312.13936,
github.com/puzzlef/leiden-communities-openmp, `inc/leiden.hxx`'s
`leidenMoveOmpW`): a **fused** parallel decide+apply, no separate phases at
all. Every node is swept in parallel each round (`(0..graph.n).into_par_iter()`
inside the shared 8-thread pool); each thread scans its own node's neighbors,
decides the best move, and commits it *immediately* via atomics
(`membership`/`cluster_weights`/`cluster_degree` become `Vec<AtomicU32>` for
the call's duration, `Ordering::Relaxed` throughout -- nothing here needs
stronger ordering than eventual visibility). An `affected: Vec<AtomicBool>`
flag per node (cleared when processed, set on a mover's neighbors) replaces
the explicit shrinking active-list both the old design and the references'
predecessors use. No per-decision re-validation is needed: the staleness
window between a read and another thread's concurrent write is now a handful
of instructions, not a full round, so residual conflicts self-correct almost
immediately via the next round's wake-ups instead of settling into sustained
oscillation. `MAX_FASTMOVE_ROUNDS` (100) still caps the loop regardless --
both reference implementations cap their own move phase the same way
(`maxIter`/`L`), confirming this is standard practice for the technique, not
a hack specific to this port.

One detail neither reference needed: "become a new singleton cluster" needs a
cluster id nobody else could claim at the same instant, without a shared/
serial allocator. Solved by reserving `graph.n + v` as node `v`'s own private
singleton id -- unique by construction, so claiming it needs no
synchronization at all. Cluster ids in this function therefore range over
`[0, 2*graph.n)`, not `[0, graph.n)`; `reindex_membership` is called with the
doubled bound at the end to compact back down to a dense range for the next
phase (`aggregate`).

A soundness-focused code review (concurrency changes get extra scrutiny) came
back clean -- no `unsafe`, no data races, `Relaxed` ordering genuinely
sufficient everywhere it's used -- but caught one real, worth-fixing issue:
`cluster_degree` (atomically incremented/decremented on every committed move)
was fully maintained but never actually read anywhere, a leftover from the
abandoned empty-cluster-recycling design now replaced by the singleton-id
trick above. Removed, along with two minor nits (a provably-redundant guard,
and hand-rolled CAS-loop replaced with `AtomicU32::fetch_update`) -- this
wasn't just cleanup: at 50M-node scale it was 2 fewer atomic RMW ops per
committed move, dropping the time further, from 74.651s to **69.939s**, with
no quality change (H=83,917,375.3, still in the same noise band).

Result: **69.939s** for `fastmove_nodes`'s share of a full run (down from
92.3s sequential, on top of the other two phases already being parallel) --
see the final results table below. Verified via the toy-dataset variance
check (tight, consistent quality matching the sequential baseline band) and
the production benchmark (thread activity confirmed healthy via repeated
`top -H` snapshots, quality within normal stochastic noise of the sequential
baseline).

## Final results (all three phases parallelized, 8 threads)

Same dataset and params throughout (CPM, γ=0.000008, β=0.01, iterations=2).

| | sequential | +`merge_nodes` | +`aggregate` | +`fastmove_nodes` (final) |
|---|---|---|---|---|
| total time | 175.971s | 155.241s | 142.853s | **69.939s** |
| speedup vs. sequential | 1.00x | 1.13x | 1.23x | **2.52x** |
| communities | 15,599,528 | 15,599,669 | 15,600,059 | 15,599,796 |
| H | 83,920,786.7 | 83,922,677.6 | 83,920,135.8 | 83,917,375.3 |

Quality is stable across all four rows (all within normal stochastic
run-to-run noise -- the toy-dataset variance checks done at each step
established that band). Per-phase breakdown from the pre-cleanup run
(`--features monitor`, 76.722s total -- close enough to the final 69.939s to
still be representative of the relative split):

| phase | total | % of wall time |
|---|---|---|
| `merge_nodes` | 57.2s | 74.6% |
| `fastmove_nodes` | 21.8s | 28.4% |
| `aggregate` | 17.4s | 22.7% |
| flatten/retrieve | 5.3s | 6.9% |
| supernode_remap | 1.5s | 1.9% |

(Percentages don't sum to 100% -- phases overlap across levels in this
accounting.) Note the shift: `merge_nodes` is now the *largest* single cost,
having gone from 47.5s (27.0% of a 175.971s total) to 57.2s in absolute terms
even though its share of a much-shorter total grew further -- its own review
flagged a plausible reason (`refnd/src/core/leiden/leidenp.rs`'s
`merge_nodes`): the vast majority of its 286M+ calls operate on tiny clusters
(median size 1), so per-task dispatch overhead likely dominates actual work
for most of them. `fastmove_nodes` improved by >4x in absolute terms (92.3s
sequential contribution -> 21.8s here) despite the accounting change. Further
gains from here would most likely come from reducing `merge_nodes`'s per-task
overhead (e.g. batching multiple tiny clusters per rayon task) rather than
from further restructuring any of the three phases -- not attempted this
session.

## `merge_nodes` further optimization attempt (this session)

The "74.6% of wall time" figure in the table above turned out to be
misleading, discovered while chasing it. `measure!`'s `LockStat.record()` is a
plain `fetch_add` -- for phases called once per level (`fastmove_nodes`,
`aggregate`), that's a true wall-clock number, but `merge_nodes` is called
once *per original cluster* (270M+ times), each call's duration summed into
the same shared atomic from up to 8 concurrent threads. That sum isn't
comparable to a wall-clock measurement: it's inflated both by counting
parallel work multiple times over (up to 8x) and by real contention on the
shared `LockStat` atomic itself under that call volume. Subtracting the other
four (genuinely wall-clock) phases from total wall time gives `merge_nodes`'s
*true* contribution as roughly 30s of the 73s total (~41%), not 57.2s/74.6% --
still the largest single phase, but not as dominant as it looked.

Two architectural rewrites were attempted and both **measurably regressed**
wall-clock time, so neither shipped:

- **Vertex-centric fused-atomic, single pass** (mirroring `fastmove_nodes`'s
  own redesign and GVE-Leiden's `REFINE`-mode reuse of its move kernel): one
  parallel sweep over all vertices instead of one rayon task per original
  cluster, candidates restricted to same-parent-cluster via a per-edge filter
  instead of a per-cluster task boundary. In isolation this did cut
  `merge_nodes`'s own (now directly wall-clock-comparable) cost from ~30s to
  ~27s -- but total wall time still came out slightly *worse* (~73-76s vs
  ~70.5s baseline, 3 non-monitor runs), because `aggregate`'s cost grew by a
  similar amount (~17.1s -> ~20.1s). Root cause: the sequential design's
  strict per-cluster shuffle allows merges to cascade within a single pass
  (A joins B, then C -- processed later in the same shuffle -- sees A+B
  already merged and joins too); a single parallel pass can't reproduce that,
  since each vertex only gets one atomic decide+commit. Less cascading means
  more, smaller refined clusters survive per level, pushing extra supernodes
  and edges onto the next `aggregate` call -- a real cost that happened to
  roughly cancel the dispatch-overhead savings.
- **Same design plus a bounded round loop** (`affected`-flag driven, capped at
  20 rounds, intended to recover the lost cascading by letting missed merges
  catch up in later rounds): made things *worse*, not better -- ~79.4s avg (3
  non-monitor runs). `aggregate`'s cost didn't shrink at all (~20.4s, no
  different from the single-pass version), so the extra rounds were pure
  overhead with no offsetting benefit. Reverted along with the single-pass
  version.

**What shipped instead**: the per-cluster task-dispatch-overhead diagnosis was
directionally correct, just smaller in magnitude than the misleading metric
suggested -- so the fix is a single-line, zero-semantic-risk change rather
than a rewrite. `cluster_scratch[..nb_clusters].par_iter_mut().enumerate()`
gets `.with_min_len(merge_min_len)` added before `.map(...)`, batching several
original clusters into one rayon task; `merge_nodes`'s body, its per-cluster
sequential refinement logic (including the cascading-merge behavior above),
and every other call site are untouched.

`min_len` is *not* a fixed constant -- a first attempt at
`with_min_len(1024)` passed a soundness/correctness review that caught a real
bug before it shipped: rayon's splitter only splits a range while
`len/2 >= min_len`, so a fixed `min_len` of 1024 fully serializes the whole
dispatch (onto a single thread) whenever a level has fewer than 2048 clusters
-- and `nb_clusters` shrinks every level as aggregation proceeds, so this hit
hardest on later, more-aggregated levels where per-cluster work is largest,
exactly backwards from the fix's intent. Fixed by computing
`min_len = (nb_clusters / (MAX_THREADS * 4)).clamp(1, MERGE_TASK_MAX_BATCH)`
at the call site instead: this keeps enough splits available for full 8-way
distribution at every level (verified against rayon's actual `try_split`
condition, not just by inspection) while still capping batch size at 1024
once there's plenty of clusters to batch. A second review round came back
clean.

Result: **~69.7s** average (5 non-monitor production runs across two builds:
70.604s, 68.903s, 69.469s, 69.814s, 69.867s) vs **~70.5s** average on the
unmodified baseline (70.799s, 70.243s) -- a modest (~1%) but real,
reproducible improvement, with zero risk to correctness or quality (same
algorithm, same code path, only rayon's internal task-splitting granularity
changes). Quality unaffected (toy-dataset variance checks stayed within the
established band throughout). This is a much smaller win than initially
targeted -- see the corrected `merge_nodes` cost estimate above for why
"close to 8x" was never really on the table for this specific phase, and the
two failed rewrite attempts for why the bigger architectural win didn't
materialize either. `aggregate` (the next-largest genuinely-wall-clock phase
at ~17s) and the remaining sequential bookkeeping (`flatten/retrieve`,
`supernode_remap`, each level's full graph clone in `find_partition`) are the
more promising remaining targets, not attempted this session.

## Pushing further toward the 7x target (this session, continued)

The user asked to keep pushing until at least 7x vs. sequential (175.971s / 7
≈ 25s). Starting point for this round: ~69.7s (2.52x, the `with_min_len`
batching fix above).

**`merge_nodes` allocation elimination.** It allocated ~8 fresh heap buffers
(`cluster_weights`, `cluster_out_weight`, two `FixedBitSet`s,
`weight_to_cluster`, `neighbor_clusters`, `cum_likelihood`, plus
`clean_refined_membership`'s own `new_cluster`) on *every* call -- 270M+ calls
at production scale, mostly for tiny/singleton clusters, so this was billions
of allocations. Moved everything into a per-thread `MergeScratch` struct
(same `thread_local!` idiom as `MERGE_SCRATCH`/`AGGREGATE_SCRATCH` already
used), grown-never-shrunk to the largest cluster seen per thread.
`cluster_weights`/`cluster_out_weight` use plain assignment on first touch
each call (every index in `[0,n)` gets freshly written before being read, so
no reset needed); `non_singleton_cluster` and the compaction buffer
(`new_cluster`) use a read-before-first-write sentinel pattern, so they track
their own touched indices and self-clean just those, rather than a full
`[0,n)` reset. `cluster_degree` (present in the old scratch) was dropped
entirely -- re-confirmed dead/write-only in both `leiden.rs` and the prior
`leidenp.rs` via `grep`. Same treatment applied to `aggregate`'s
`neighbor_clusters` scratch. Result: merge_nodes's own per-call avg dropped
~207ns -> ~148-160ns, and combined with the next item below, wall time
dropped modestly (~69.7s -> ~68.7s across the two changes).

**Two failed load-imbalance fixes for `merge_nodes` (both reverted).**
Instrumentation (temporary, added and removed this session) revealed the
*real* remaining cost driver: cluster sizes are extremely skewed (p50=1,
p99=7, but a max in the tens of thousands to ~100K+) *and* correlated with
cluster id -- hundreds of large clusters, cumulatively a few seconds of real
work, landed within a span of a few thousand consecutive ids (a handful of
`with_min_len` batches). Since one cluster's refinement is an inherently
serial shuffle/cascade, those batches became multi-second stragglers running
alone on 1-2 threads while the other 6-7 sat idle -- confirmed via `top -H`
and per-call timing (one dispatch summing to ~4.6s of real work took 11.3s
wall clock). Two fixes were tried and **both regressed wall time** despite
directly fixing the confirmed straggler (level-0 merge dispatch dropped from
~11.3s to ~3s in isolation, in both cases):
- **LPT-style reordering** (physically move the largest ~64 clusters to the
  front, dispatch them as individual unbatched tasks via `rayon::join`/
  `.chain()` alongside the small-cluster bulk): ~72.35s avg, worse than the
  ~69.7s starting point.
- **Random shuffle of dispatch order** (breaks the id/size correlation so
  every batch gets a representative mix), tried two ways: physically
  reordering `cluster_scratch` via `mem::take` (~89.9s -- the O(n) move of
  ~15-20M `Vec` headers per level cancelled the fix), then avoiding that
  reorder cost via `unsafe` permutation-based pointer access (~102.15s --
  *worse*, not better, suggesting the raw-pointer access pattern also cost
  the compiler some optimization headroom the safe `&mut` reference had).

Root cause of *why* fixing the confirmed straggler still lost overall:
shuffling scrambles which supernode ids end up adjacent to which, which
turned out to matter for cache locality in *every downstream phase*
(`aggregate`, and the next level's `fastmove_nodes` operating on the
resulting coarser graph), not just the dispatch being fixed. The original
(correlated) ordering was apparently providing real, load-bearing locality
that a straggler-avoidance reordering broke. **Both attempts were fully
reverted**; the straggler cost is a known, accepted limitation, not solved
this session. A future attempt should probably preserve the original
dispatch order for everything *except* the offending stragglers, verified
against downstream phase costs specifically -- not just the dispatch phase
being optimized -- before considering it a win.

**`fastmove_nodes`: active-list redesign for the round loop.** Instrumentation
showed the round loop runs far more rounds than expected -- ~38-43 on the
production dataset's first level (50M nodes) -- and each round was a full
`(0..graph.n).into_par_iter()` scan just to check a per-node `affected` flag,
even though the truly-affected set shrinks fast (measured: round 1 ~33M,
round 5 ~1.2M, round 20 ~5.7K, round 40 ~1K). Replaced with an explicit
`active: Vec<u32>` work list, threaded round-to-round. Two designs were tried
and discarded before the one that worked, both benchmarked back to the
pre-change baseline despite being theoretically sound:
- A `affected: Vec<AtomicBool>` reset to `false` at the start of a node's own
  processing raced a concurrent neighbour's mark of that same node (both
  active the same round) -- caught via code review before shipping, not by a
  test. Fixed with a separate reset-before-work pass, but that pass's own
  O(active) cost per round cancelled the benefit.
- Collecting each task's newly-affected neighbours via
  `.map(|v| Vec::new()).collect::<Vec<Vec<u32>>>()` materializes one `Vec`
  header per *processed* node, not per node actually queued -- ~1.2GB of
  mostly-empty headers for round 1 alone at 50M nodes, just to immediately
  flatten and discard.

What shipped: `queued_for: Vec<AtomicU32>` stores the round number a node is
queued for (not a plain boolean) -- a neighbour-scan's
`queued_for[u].swap(next_round, Relaxed) != next_round` dedupes queueing
within a round with no reset pass needed at all, since every round uses a
distinct, ever-increasing stamp that a stale value from an earlier round can
never collide with. Newly-affected neighbours are pushed into
`FASTMOVE_NEXT_ACTIVE`, a per-thread accumulator (same `thread_local!` idiom
as everywhere else in this file) drained via `ThreadPool::broadcast` after
each round -- gathering `MAX_THREADS` buffers per round instead of one
(mostly-empty) `Vec` per node. Reviewed and verified race-free (the epoch
stamp's dedup is a single atomic RMW, globally exclusive across threads;
`broadcast` runs strictly after the preceding `install()` returns, so there's
no window where a thread's own `RefCell` write and its drain can race).
Result: `fastmove_nodes`'s own share dropped from ~19-20s to ~18.1-18.3s;
combined with the next item below, wall time dropped from ~68.7s to ~67.2s.

**Parallelized two remaining sequential O(graph.n) bookkeeping loops.** The
flatten step's membership gather (`self.membership[i] =
aggregated_membership[super_node_map[i]]`) and the supernode-remap step
(`super_node_map[i] = refined_membership[super_node_map[i]]`) are both simple
independent per-index reads of one array plus a write to another -- no shared
mutable state, no cross-iteration dependency, verified structurally (the
three arrays involved are separately-owned `Vec`s that Rust's ownership rules
already guarantee can't alias). Wrapped both in
`self.pool.install(|| ....par_iter_mut()...)`. `supernode_remap`'s own share
dropped from ~1.5-1.6s to ~0.39s; `flatten/retrieve`'s from ~5.2s to ~3.78s
(only the gather half was parallelizable this way -- `retrieve_clusters`'s
per-node `cluster_scratch[c].push(...)` bucketing is a genuine multi-writer
scatter and wasn't attempted this session). Wall time dropped from ~67.2s to
**~64.25s** (5 non-monitor runs: 63.539s, 63.387s, 64.800s, 66.490s, 63.045s).

**Current state: ~64.25s, 2.74x vs. the 175.971s sequential baseline.** Still
well short of 7x (~25s). Every change this session was verified via the toy
dataset's variance-band check (unchanged: communities in the high-80s to
low-100s, H roughly 18470-18520 for `cpm γ=0.01 β=0.01`) and multiple
production-scale runs before and after, with a persistent review-subagent
(kept alive across rounds via direct `SendMessage`, rather than re-spawned
fresh each time, specifically to avoid re-deriving codebase/rayon-internals
context every round) reviewing every change, iterated until clean, before
committing. Remaining known targets, none attempted to completion this
session: `aggregate` itself (~16.5s true cost, already parallel per-supernode
but not otherwise optimized this session -- e.g. its own `local_edges`
per-call `Vec::new()` is analogous to the allocation pattern just fixed in
`merge_nodes`, not yet addressed); `retrieve_clusters`'s sequential
multi-writer bucketing; and the `merge_nodes` load-imbalance straggler
problem, confirmed real but unsolved (see above). Given the diminishing,
partly-negative returns from the load-imbalance attempts specifically, 7x may
not be reachable through this style of incremental fix alone -- it likely
needs either a genuinely different algorithmic approach to the straggler
problem (one that doesn't disturb downstream locality) or accepting a lower
ceiling.

**Standing gap, not addressed this session**: `pytests/test_leiden_accuracy.py`
still only exercises the sequential `leiden::find_communities` path, not
`leidenp::fast_find_communities` -- flagged repeatedly by the review subagent
given how much has now landed on the parallel path (full `merge_nodes` and
`fastmove_nodes` rewrites, `aggregate` scratch reuse, two newly-parallelized
loops) with zero automated regression coverage specific to it.

## Recommended order of work (historical -- all three steps below are done)

1. ~~Fix the `monitor` feature's pre-existing, unrelated compile error~~ --
   **done**. It was a temporary-value-dropped-while-borrowed bug in
   `add_edge`'s sparse branch (`core/hnsw/mod.rs`): `measure!` binds `$expr`'s
   *result* to a `let`, but `m.entry(a).or_insert_with(...).write()`'s
   `RwLockWriteGuard` borrows from the unbound `entry(...)` temporary, which
   drops at the end of that `let` statement -- before the guard could be
   returned from the macro's block. Fixed by binding the entry to a local
   first (`set_neighbourhood`, right below it, already did this correctly --
   same fix, applied to `add_edge`).
2. ~~Parallelize `merge_nodes`~~ -- **done** (commit `0fd59cd`).
3. ~~Parallelize `aggregate`~~ -- **done** (commit `6257f7b`).
4. ~~Parallelize `fastmove_nodes`~~ -- **done**, via the fused-atomic redesign
   above, not the originally-planned decide/apply rounds.
5. Re-benchmark after each step (timing + quality via `bench_leiden`,
   correctness via `pytests/test_leiden_accuracy.py`) -- done throughout.

Re-run `pytests/test_leiden_accuracy.py` after any future change to
`leidenp.rs` -- it's cheap (well under a second) and it's what would have
caught both correctness bugs fixed earlier this session. It currently only
exercises `leiden::find_communities` (the sequential reference); extending it
(or adding a parallel-specific variant) to also cover
`leidenp::fast_find_communities` is a natural next step, not done this
session.
