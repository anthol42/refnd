"""Metric computation helpers for the HNSW ablation study."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from rich import print
from refnd.utils import BitFingerprint
from refnd.core import (
    CsrGraph,
    EdgeStore,
    HNSWState,
    LeidenObjective,
    find_communities,
    find_components,
    partition,
)
from refnd.kernels import KernelVariant
from tqdm import tqdm
if TYPE_CHECKING:
    from datasets import DatasetConfig


def _shuffle_sample(sample: Any, rng: np.random.Generator) -> Any:
    """Return a copy of *sample* with its elements randomly permuted."""
    if isinstance(sample, str):
        chars = list(sample)
        rng.shuffle(chars)
        return "".join(chars)
    return BitFingerprint.random(len(sample), sample.count())



def null_model(
    data: list[Any],
    cfg: "DatasetConfig",
    n_samples: int = 10_000_000,
    seed: int = 42,
) -> float:
    """Estimate P(distance <= threshold) under a permutation null model.

    Generates n_samples random pairs of element-shuffled samples, computes
    all distances in one parallelized zip_kernel call (Rust/rayon), and
    returns the fraction within the proximity threshold.
    Used as gamma for CPM Leiden.
    """
    from refnd.kernels import zip_kernel

    rng   = np.random.default_rng(seed)
    idx_a = rng.integers(0, len(data), size=n_samples)
    idx_b = rng.integers(0, len(data), size=n_samples)
    list_a = [_shuffle_sample(data[i], rng) for i in idx_a]
    list_b = [_shuffle_sample(data[i], rng) for i in idx_b]
    print("  [dim]Running kernel...[/]")
    scores = zip_kernel(
        cfg.modality, list_a, list_b,
        n_threads=0, progress=True,
        **cfg.kernel_params,
    )
    return sum(1 for s in scores if s <= cfg.proximity_threshold) / n_samples


def layer0_to_edge_store(adj: list[list[int]], n: int) -> EdgeStore:
    """Convert get_layer(0) adjacency lists to an unweighted EdgeStore."""
    seen: set[tuple[int, int]] = set()
    edges = []
    for src, neighbors in enumerate(adj):
        for dst in neighbors:
            key = (min(src, dst), max(src, dst))
            if key not in seen:
                seen.add(key)
                edges.append((key[0], key[1], 1.0))
    return EdgeStore(n, edges)


def edge_set(es: EdgeStore) -> set[tuple[int, int]]:
    return {(min(s, d), max(s, d)) for s, d, _ in es.edges()}


def pct_edges_recovered(hnsw_es: EdgeStore, exact_es: EdgeStore) -> float:
    exact = edge_set(exact_es)
    hnsw = edge_set(hnsw_es)
    return len(hnsw & exact) / len(exact)


def missed_edge_weight_dist(hnsw_es: EdgeStore, exact_es: EdgeStore) -> dict:
    exact_edges_list = exact_es.edges()
    hnsw_pairs = edge_set(hnsw_es)
    missed_w = [w for s, d, w in exact_edges_list
                if (min(s, d), max(s, d)) not in hnsw_pairs]
    if not missed_w:
        return {k: 0.0 for k in ("mean", "std", "median", "p10", "p25", "p75", "p90")}
    a = np.array(missed_w, dtype=np.float32)
    return {
        "mean":   float(a.mean()),
        "std":    float(a.std()),
        "median": float(np.median(a)),
        "p10":    float(np.percentile(a, 10)),
        "p25":    float(np.percentile(a, 25)),
        "p75":    float(np.percentile(a, 75)),
        "p90":    float(np.percentile(a, 90)),
    }


def pct_missed_inter_community(
    hnsw_es: EdgeStore,
    exact_es: EdgeStore,
    exact_communities: list[int],
) -> float:
    """% of missed HNSW edges that are inter-community in the exact graph."""
    hnsw_pairs = edge_set(hnsw_es)
    missed = [(s, d) for s, d, _ in exact_es.edges()
              if (min(s, d), max(s, d)) not in hnsw_pairs]
    if not missed:
        return 0.0
    inter = sum(1 for s, d in missed if exact_communities[s] != exact_communities[d])
    return inter / len(missed)


def top3_component_community_counts(hnsw_graph: CsrGraph, objective: LeidenObjective, gamma: float) -> list[dict]:
    """Sizes of the 3 largest components and how many communities fall in each."""
    components = find_components(hnsw_graph)
    communities = find_communities(hnsw_graph, objective=objective, gamma=gamma)

    comp_sizes = Counter(components)
    top3 = [comp_id for comp_id, _ in comp_sizes.most_common(3)]

    results = []
    for comp_id in top3:
        node_mask = [i for i, c in enumerate(components) if c == comp_id]
        unique_communities = len({communities[i] for i in node_mask})
        results.append({"size": comp_sizes[comp_id], "n_communities": unique_communities})
    return results


def _build_subgraph(
    train_idx: list[int],
    hnsw_es: EdgeStore,
    graph_weighted: bool,
    graph_is_distance: bool,
) -> CsrGraph:
    """Build a sub-graph and community labels restricted to train_idx nodes.

    Returns (sub_graph, sub_communities) where node IDs are re-indexed 0..n_train-1.
    """
    train_set = set(train_idx)
    node_map  = {old: new for new, old in enumerate(train_idx)}
    sub_edges = [
        (node_map[s], node_map[d], w)
        for s, d, w in hnsw_es.edges()
        if s in train_set and d in train_set
    ]
    sub_es = EdgeStore(len(train_idx), sub_edges)
    return sub_es.graph(weighted=graph_weighted, is_weight_distance=graph_is_distance)


def split_and_violations(
    hnsw_communities: list[int],
    hnsw_es: EdgeStore,
    hnsw_graph: CsrGraph,
    graph_weighted: bool,
    graph_is_distance: bool,
    data: list[Any],
    labels: np.ndarray,
    embs: torch.Tensor | None,
    variant: KernelVariant,
    proximity_threshold: float,
    metric: str | None,
    kernel_params: dict,
    n_repeats: int = 10,
) -> dict:
    """Run both splits (with/without post-filtering) over n_repeats seeds.

    The validation split inside each MLP run uses community-based partitioning
    on the train sub-graph, mirroring the outer test split strategy.
    Reports mean and std of MLP score and violation count across repeats.
    """
    n_communities = len(set(hnsw_communities))

    if metric is None or embs is None:
        # No supervised task — report split sizes and violations only, no MLP score.
        results = {}
        for post_filter in (False, True):
            key = "postfilter" if post_filter else "no_postfilter"
            if n_communities <= 1:
                results[key] = {"n_test_mean": float("nan"), "n_train_mean": float("nan"),
                                "p_violations_mean[%]": float("nan"), "p_violations_std[%]": float("nan")}
                continue
            p_violations_list, n_tests, n_trains = [], [], []
            for seed in range(n_repeats):
                train_idx, test_idx = partition(
                    hnsw_communities, hnsw_graph,
                    test_ratio=0.2, seed=seed, post_filtering=post_filter,
                )
                test_data  = [data[i] for i in test_idx]
                train_data = [data[i] for i in train_idx]
                train_hnsw = HNSWState(variant, train_data, proximity_threshold=proximity_threshold, **kernel_params)
                train_hnsw.build(progress=False)
                nn_results = train_hnsw.search(test_data, k=1, ef=50, threads=0, progress=False)
                p_viol = 100 * sum(1 for hits in nn_results if hits and hits[0][1] < proximity_threshold) / len(test_data)
                p_violations_list.append(p_viol)
                n_tests.append(len(list(test_idx)))
                n_trains.append(len(list(train_idx)))
            viol_arr = np.array(p_violations_list, dtype=float)
            results[key] = {
                "n_test_mean":          float(np.mean(n_tests)),
                "n_train_mean":         float(np.mean(n_trains)),
                "p_violations_mean[%]": float(viol_arr.mean()),
                "p_violations_std[%]":  float(viol_arr.std()),
            }
        return results

    from mlp import train_eval_mlp

    results = {}
    for post_filter in (False, True):
        key = "postfilter" if post_filter else "no_postfilter"
        if n_communities <= 1:
            results[key] = {
                "n_test_mean": float('nan'),
                "n_train_mean": float('nan'),
                "p_violations_mean[%]": float('nan'),
                "p_violations_std[%]": float('nan'),
                "mlp_score_mean": float('nan'),
                "mlp_score_std": float('nan'),
            }
            continue

        scores, p_violations_list, n_tests, n_trains = [], [], [], []
        for seed in range(n_repeats):
            print(f"  [dim]Evaluating {key} with seed {seed}...[/]")
            train_idx, test_idx = partition(
                hnsw_communities, hnsw_graph,
                test_ratio=0.2, seed=seed, post_filtering=post_filter,
            )
            train_idx = list(train_idx)
            test_idx  = list(test_idx)

            test_data  = [data[i] for i in test_idx]
            train_data = [data[i] for i in train_idx]
            train_hnsw = HNSWState(
                variant, train_data,
                proximity_threshold=proximity_threshold,
                **kernel_params,
            )
            train_hnsw.build(progress=True)
            nn_results = train_hnsw.search(test_data, k=1, ef=50, threads=0, progress=True)
            p_viol = 100 * sum(
                1 for hits in nn_results if hits and hits[0][1] < proximity_threshold
            ) / len(test_data)

            train_sub_graph = _build_subgraph(
                train_idx, hnsw_es, graph_weighted, graph_is_distance,
            )
            train_communities_local = [hnsw_communities[i] for i in train_idx]
            inner_train_local, val_local = partition(
                train_communities_local, train_sub_graph,
                test_ratio=0.15, seed=seed, post_filtering=post_filter,
            )
            inner_train_idx = [train_idx[i] for i in inner_train_local]
            val_idx         = [train_idx[i] for i in val_local]

            mlp_result = train_eval_mlp(
                embs, labels,
                train_idx=inner_train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                metric=metric,
                seed=seed,
            )

            scores.append(mlp_result["score"])
            p_violations_list.append(p_viol)
            n_tests.append(len(test_idx))
            n_trains.append(len(train_idx))

        scores_arr = np.array(scores)
        viol_arr   = np.array(p_violations_list, dtype=float)
        results[key] = {
            "n_test_mean":           float(np.mean(n_tests)),
            "n_train_mean":          float(np.mean(n_trains)),
            "p_violations_mean[%]":  float(viol_arr.mean()),
            "p_violations_std[%]":   float(viol_arr.std()),
            "mlp_score_mean":        float(scores_arr.mean()),
            "mlp_score_std":         float(scores_arr.std()),
        }
    return results


def compute_graph_metrics(
    hnsw_es: EdgeStore,
    exact_es: EdgeStore,
    hnsw_communities: list[int],
    hnsw_graph: CsrGraph,
    exact_graph: CsrGraph,
    graph_weighted: bool,
    graph_is_distance: bool,
    hnsw_cd_time: float,
    hnsw_build_time: float,
    data: list[Any],
    labels: np.ndarray,
    embs: torch.Tensor | None,
    variant: KernelVariant,
    proximity_threshold: float,
    metric: str | None,
    kernel_params: dict,
    com_objective: LeidenObjective,
    gamma: float,
) -> dict:
    exact_communities = find_communities(exact_graph)
    exact_edges_list  = exact_es.edges()
    pct_exact_inter   = (
        sum(1 for s, d, _ in exact_edges_list if exact_communities[s] != exact_communities[d])
        / len(exact_edges_list)
        if exact_edges_list else 0.0
    )

    return {
        "hnsw_build_time_s":          hnsw_build_time,
        "community_detection_time_s": hnsw_cd_time,
        "pct_edges_recovered":        pct_edges_recovered(hnsw_es, exact_es),
        "missed_edge_weight_dist":    missed_edge_weight_dist(hnsw_es, exact_es),
        "pct_inter_community_exact":  pct_exact_inter,
        "pct_missed_inter_community": pct_missed_inter_community(hnsw_es, exact_es, exact_communities),
        "top3_components":            top3_component_community_counts(hnsw_graph, com_objective, gamma),
        "split":                      split_and_violations(
            hnsw_communities=hnsw_communities,
            hnsw_es=hnsw_es,
            hnsw_graph=hnsw_graph,
            graph_weighted=graph_weighted,
            graph_is_distance=graph_is_distance,
            data=data,
            labels=labels,
            embs=embs,
            variant=variant,
            proximity_threshold=proximity_threshold,
            metric=metric,
            kernel_params=kernel_params,
        ),
    }
