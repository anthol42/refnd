"""Cross-checks refnd's Leiden implementation against igraph's reference Leiden.

Two ground truths are involved:
  - the *real* ground truth (Karate Club's documented faction split, or an SBM's
    generative block assignment) -- printed for context, not asserted on.
  - igraph's own Leiden output -- the actual target. The goal is for refnd to
    optimize as well as igraph, not to independently beat it or the real ground
    truth.

The metric that matters is *quality* (the objective value each partition
achieves under the shared, implementation-independent scorer below), not
whether the two implementations land on the identical partition: harder graphs
legitimately have several distinct, equally-good local optima, so requiring
refnd to reproduce igraph's specific partition (e.g. via pairwise NMI) rejects
valid alternative optima as "disagreement" and produces false failures.

Every dataset carries edge weights -- either genuine (Karate Club's published
Zachary 1977 interaction counts) or synthetic (SBM edges weighted higher within
a block than between blocks, mimicking a real similarity graph), or a flat 1.0
for the unweighted variants. This matters because the core gain formula
(`weight_to_cluster[c] - resolution * node_weight(v) * cluster_weight(c)`) uses
edge weights directly -- a weight-scaling bug would be invisible on an
unweighted graph, where every weight collapses to 1.0 anyway.

Leiden's refinement step is stochastic, so a single run proves nothing. Each
config runs both implementations N_REPEATS times and checks with a one-sided
Mann-Whitney U test that refnd's quality distribution isn't significantly
below igraph's.
"""
import random
from collections import defaultdict

import igraph as ig
import networkx as nx
import pytest
from scipy.stats import mannwhitneyu

from refnd.core import CsrGraph, EdgeStore, INWeightType, LeidenObjective, find_communities

N_REPEATS = 20
ALPHA = 0.05


def _to_edgestore(g: ig.Graph, weights) -> EdgeStore:
    return EdgeStore(g.vcount(), [(u, v, w) for (u, v), w in zip(g.get_edgelist(), weights)])


def _cpm_resolution(g: ig.Graph, weights) -> float:
    """Average edge weight over all possible pairs -- a standard default CPM
    resolution, generalizing graph density (the unweighted special case) to
    weighted graphs."""
    n = g.vcount()
    return 2 * sum(weights) / (n * (n - 1))


def _karate_club():
    g_nx = nx.karate_club_graph()
    truth = [0 if g_nx.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(g_nx.number_of_nodes())]
    g = ig.Graph(n=g_nx.number_of_nodes(), edges=list(g_nx.edges()))
    return g, truth, [1.0] * g.ecount()


def _karate_club_weighted():
    """Uses the genuine Zachary 1977 edge weights (social-interaction counts)
    networkx ships with `karate_club_graph()`, not synthetic ones."""
    g_nx = nx.karate_club_graph()
    truth = [0 if g_nx.nodes[i]["club"] == "Mr. Hi" else 1 for i in range(g_nx.number_of_nodes())]
    edges = list(g_nx.edges(data="weight"))
    g = ig.Graph(n=g_nx.number_of_nodes(), edges=[(u, v) for u, v, _ in edges])
    return g, truth, [float(w) for _, _, w in edges]


def _sbm(block_sizes, p_in, p_out, weighted=False):
    k = len(block_sizes)
    pref = [[p_in if i == j else p_out for j in range(k)] for i in range(k)]
    g = ig.Graph.SBM(pref, block_sizes)
    truth = [b for b, size in enumerate(block_sizes) for _ in range(size)]
    if weighted:
        # Within-block edges get a higher weight than between-block edges, like a
        # real similarity graph where same-cluster points are more alike.
        weights = [
            random.uniform(0.5, 1.0) if truth[u] == truth[v] else random.uniform(0.05, 0.3)
            for u, v in g.get_edgelist()
        ]
    else:
        weights = [1.0] * g.ecount()
    return g, truth, weights


DATASETS = {
    "karate_club": _karate_club,
    "karate_club_weighted": _karate_club_weighted,
    "sbm_easy": lambda: _sbm([40] * 4, 0.30, 0.02),
    "sbm_easy_weighted": lambda: _sbm([40] * 4, 0.30, 0.02, weighted=True),
    "sbm_medium": lambda: _sbm([40] * 4, 0.20, 0.05),
    "sbm_medium_weighted": lambda: _sbm([40] * 4, 0.20, 0.05, weighted=True),
    "sbm_hard": lambda: _sbm([40] * 4, 0.15, 0.08),
    "sbm_hard_weighted": lambda: _sbm([40] * 4, 0.15, 0.08, weighted=True),
}


def _modularity_quality(g, weights, gamma):
    return lambda membership: g.modularity(membership, weights=weights, resolution=gamma)


def _cpm_quality(g, weights, gamma):
    """H = [sum_internal_edge_weight*2 - gamma * sum_c(K_c^2)] / (2m), matching
    igraph's `leiden_quality` (src/community/leiden.c) exactly -- verified to
    reproduce igraph's own `VertexClustering.quality` bit-for-bit on CPM
    partitions. K_c is cluster *size* (CPM's vertex weight is always 1,
    regardless of edge weights), only the internal-edge term is weighted."""
    m = sum(weights)
    edges = g.get_edgelist()

    def quality(membership):
        internal = sum(2.0 * w for (u, v), w in zip(edges, weights) if membership[u] == membership[v])
        sizes = defaultdict(float)
        for c in membership:
            sizes[c] += 1.0
        return (internal - gamma * sum(s * s for s in sizes.values())) / (2 * m)

    return quality


# name -> (igraph objective_function string, our LeidenObjective, resolution fn, quality fn)
OBJECTIVES = {
    "modularity": ("modularity", LeidenObjective.Modularity, lambda g, w: 1.0, _modularity_quality),
    "cpm": ("CPM", LeidenObjective.CPM, _cpm_resolution, _cpm_quality),
}


def _mean(xs):
    return sum(xs) / len(xs)


def _run_comparison(dataset_name, objective_name):
    g, truth, weights = DATASETS[dataset_name]()
    ig_objective, our_objective, resolution_fn, quality_fn = OBJECTIVES[objective_name]
    gamma = resolution_fn(g, weights)
    quality = quality_fn(g, weights, gamma)

    csr = CsrGraph(_to_edgestore(g, weights), inweight_type=INWeightType.Similarity)

    our_runs = [
        find_communities(csr, gamma=gamma, beta=0.01, n_iterations=0, objective=our_objective)
        for _ in range(N_REPEATS)
    ]
    igraph_runs = [
        list(g.community_leiden(objective_function=ig_objective, weights=weights,
                                 resolution=gamma, beta=0.01, n_iterations=-1).membership)
        for _ in range(N_REPEATS)
    ]

    return {
        "our_quality": [quality(r) for r in our_runs],
        "igraph_quality": [quality(r) for r in igraph_runs],
        "our_vs_truth": [ig.compare_communities(truth, r, method="nmi") for r in our_runs],
        "igraph_vs_truth": [ig.compare_communities(truth, r, method="nmi") for r in igraph_runs],
    }


def _check_matches_igraph(dataset_name, objective_name):
    r = _run_comparison(dataset_name, objective_name)
    _, p_value = mannwhitneyu(r["our_quality"], r["igraph_quality"], alternative="less")

    print(
        f"{dataset_name:20s} {objective_name:11s}  "
        f"quality: ours={_mean(r['our_quality']):.4f} igraph={_mean(r['igraph_quality']):.4f}  p={p_value:.3f}  |  "
        f"vs-truth (NMI): ours={_mean(r['our_vs_truth']):.3f} igraph={_mean(r['igraph_vs_truth']):.3f}"
    )

    assert p_value >= ALPHA, (
        f"{dataset_name}/{objective_name}: refnd's quality is significantly lower than igraph's "
        f"(ours mean={_mean(r['our_quality']):.4f} vs igraph mean={_mean(r['igraph_quality']):.4f}, "
        f"p={p_value:.4f})"
    )


@pytest.mark.parametrize("objective_name", OBJECTIVES.keys())
@pytest.mark.parametrize("dataset_name", DATASETS.keys())
def test_matches_igraph(dataset_name, objective_name):
    _check_matches_igraph(dataset_name, objective_name)


if __name__ == "__main__":
    for dataset_name in DATASETS:
        for objective_name in OBJECTIVES:
            _check_matches_igraph(dataset_name, objective_name)
